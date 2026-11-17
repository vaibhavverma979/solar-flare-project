"""
embedder.py — Encode X-ray light-curve windows into dense vector embeddings.

Two embedding strategies are provided:
1.  StatisticalEmbedder  — handcrafted features (always works, no deps)
2.  LightCurveEmbedder   — 1-D ConvNet encoder (requires PyTorch)

Both produce fixed-length 128-d vectors suitable for cosine-similarity
search in ChromaDB.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


# ── Base class ────────────────────────────────────────────────────────────────

class BaseEmbedder(ABC):
    """Interface for light-curve embedding strategies."""

    @abstractmethod
    def embed(self, flux_window: np.ndarray) -> np.ndarray:
        """
        Encode a 1-D flux array into a fixed-length embedding.

        Parameters
        ----------
        flux_window : np.ndarray
            Shape (T,) — the raw or normalised flux values over a time window.

        Returns
        -------
        np.ndarray
            Shape (embedding_dim,).
        """
        ...

    @property
    @abstractmethod
    def dim(self) -> int:
        """Embedding dimensionality."""
        ...


# ── Statistical (handcrafted) embedder ────────────────────────────────────────

class StatisticalEmbedder(BaseEmbedder):
    """
    Produce a 128-d embedding from handcrafted statistical features of a
    flux window.  Zero-padded to full dimension.

    Feature groups (total ≤128 values):
        - basic stats (8): mean, std, min, max, median, skew, kurtosis, range
        - temporal (8): rise_rate, decay_rate, peak_position, first_quarter_mean,
                        second_quarter_mean, third_quarter_mean, fourth_quarter_mean, trend_slope
        - spectral (16): top-16 FFT magnitudes (captures periodicity)
        - shape (8): histogram bins (normalised to 1)
        - derivatives (8): mean/std/max of first & second derivatives
        - zero-padding to 128
    """

    def __init__(self, embedding_dim: int = 128):
        self._dim = embedding_dim

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, flux_window: np.ndarray) -> np.ndarray:
        flux = np.asarray(flux_window, dtype=np.float64).ravel()

        if len(flux) == 0:
            return np.zeros(self._dim, dtype=np.float32)

        # Normalise (z-score)
        mu, sigma = float(np.mean(flux)), float(np.std(flux))
        z = (flux - mu) / max(sigma, 1e-12)

        features = []

        # ── Basic stats (8) ───────────────────────────────────────────────
        features.extend([
            mu,
            sigma,
            float(np.min(flux)),
            float(np.max(flux)),
            float(np.median(flux)),
            float(_safe_skew(z)),
            float(_safe_kurtosis(z)),
            float(np.ptp(flux)),
        ])

        # ── Temporal features (8) ────────────────────────────────────────
        n = len(z)
        peak_idx = int(np.argmax(z))
        peak_pos = peak_idx / max(n - 1, 1)
        # Rise rate = max positive diff, Decay rate = max negative diff
        diffs = np.diff(z)
        rise_rate = float(np.max(diffs)) if len(diffs) > 0 else 0.0
        decay_rate = float(np.min(diffs)) if len(diffs) > 0 else 0.0
        q = max(n // 4, 1)
        features.extend([
            rise_rate,
            decay_rate,
            peak_pos,
            float(np.mean(z[:q])),
            float(np.mean(z[q:2*q])),
            float(np.mean(z[2*q:3*q])),
            float(np.mean(z[3*q:])),
            float(np.polyfit(np.arange(n), z, 1)[0]) if n > 1 else 0.0,
        ])

        # ── Spectral features (16) — top-16 FFT magnitudes ──────────────
        fft_mag = np.abs(np.fft.rfft(z))
        fft_top = np.sort(fft_mag)[::-1][:16]
        if len(fft_top) < 16:
            fft_top = np.pad(fft_top, (0, 16 - len(fft_top)))
        features.extend(fft_top.tolist())

        # ── Shape histogram (8 bins) ─────────────────────────────────────
        hist, _ = np.histogram(z, bins=8, density=True)
        features.extend(hist.tolist())

        # ── Derivative features (8) ──────────────────────────────────────
        d1 = np.diff(z) if len(z) > 1 else np.array([0.0])
        d2 = np.diff(d1) if len(d1) > 1 else np.array([0.0])
        features.extend([
            float(np.mean(d1)),
            float(np.std(d1)),
            float(np.max(np.abs(d1))),
            float(np.mean(np.abs(d1))),
            float(np.mean(d2)),
            float(np.std(d2)),
            float(np.max(np.abs(d2))),
            float(np.mean(np.abs(d2))),
        ])

        # ── Pad / truncate to embedding_dim ──────────────────────────────
        vec = np.array(features, dtype=np.float32)
        if len(vec) < self._dim:
            vec = np.pad(vec, (0, self._dim - len(vec)))
        else:
            vec = vec[:self._dim]

        # L2-normalise for cosine similarity
        norm = float(np.linalg.norm(vec))
        if norm > 1e-12:
            vec /= norm

        return vec


# ── 1-D ConvNet embedder (PyTorch) ───────────────────────────────────────────

class LightCurveEmbedder(BaseEmbedder):
    """
    1-D CNN encoder that maps a variable-length flux window to a 128-d
    embedding.  Uses adaptive average pooling to handle variable input
    lengths.

    Falls back to StatisticalEmbedder if PyTorch is unavailable.
    """

    def __init__(self, embedding_dim: int = 128, model_path: str | None = None):
        self._dim = embedding_dim
        self._model_path = model_path
        self._pytorch_available = False
        self._model = None
        self._fallback = StatisticalEmbedder(embedding_dim)

        try:
            import torch
            self._pytorch_available = True
            self._build_model()
        except ImportError:
            pass

    @property
    def dim(self) -> int:
        return self._dim

    def _build_model(self):
        import torch
        import torch.nn as nn

        class ConvEncoder(nn.Module):
            def __init__(self, embed_dim):
                super().__init__()
                self.conv_stack = nn.Sequential(
                    nn.Conv1d(1, 32, kernel_size=7, padding=3),
                    nn.BatchNorm1d(32),
                    nn.ReLU(),
                    nn.Conv1d(32, 64, kernel_size=5, padding=2),
                    nn.BatchNorm1d(64),
                    nn.ReLU(),
                    nn.Conv1d(64, 128, kernel_size=3, padding=1),
                    nn.BatchNorm1d(128),
                    nn.ReLU(),
                )
                self.pool = nn.AdaptiveAvgPool1d(1)
                self.fc = nn.Linear(128, embed_dim)

            def forward(self, x):
                # x: (batch, 1, T)
                h = self.conv_stack(x)
                h = self.pool(h).squeeze(-1)
                return self.fc(h)

        self._model = ConvEncoder(self._dim)

        if self._model_path:
            state = torch.load(self._model_path, map_location="cpu", weights_only=True)
            self._model.load_state_dict(state)

        self._model.eval()

    def embed(self, flux_window: np.ndarray) -> np.ndarray:
        flux = np.asarray(flux_window, dtype=np.float64).ravel()

        if len(flux) == 0:
            return np.zeros(self._dim, dtype=np.float32)

        if not self._pytorch_available or self._model is None:
            return self._fallback.embed(flux)

        import torch

        # z-score normalise
        mu, sigma = float(np.mean(flux)), float(np.std(flux))
        z = ((flux - mu) / max(sigma, 1e-12)).astype(np.float32)

        tensor = torch.tensor(z).unsqueeze(0).unsqueeze(0)  # (1, 1, T)
        with torch.no_grad():
            embedding = self._model(tensor).cpu().numpy()[0]

        # L2-normalise
        norm = float(np.linalg.norm(embedding))
        if norm > 1e-12:
            embedding /= norm

        return embedding.astype(np.float32)


# ── Utility functions ─────────────────────────────────────────────────────────

def _safe_skew(x: np.ndarray) -> float:
    """Compute skewness without scipy."""
    n = len(x)
    if n < 3:
        return 0.0
    mu = np.mean(x)
    sigma = np.std(x, ddof=1)
    if sigma < 1e-12:
        return 0.0
    return float((n / ((n - 1) * (n - 2))) * np.sum(((x - mu) / sigma) ** 3))


def _safe_kurtosis(x: np.ndarray) -> float:
    """Compute excess kurtosis without scipy."""
    n = len(x)
    if n < 4:
        return 0.0
    mu = np.mean(x)
    sigma = np.std(x, ddof=1)
    if sigma < 1e-12:
        return 0.0
    m4 = np.mean((x - mu) ** 4)
    return float(m4 / (sigma ** 4) - 3.0)
