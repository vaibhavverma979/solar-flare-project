"""
patch_tst.py — Patch Time-Series Transformer for long-horizon flare forecasting.

Implements a simplified version of PatchTST (Nie et al., 2023 — "A Time Series
is Worth 64 Words: Long-term Forecasting with Transformers").

Key ideas:
    1.  **Patching**: divide the input time series into non-overlapping patches
        of length P.  This is analogous to ViT's image patches.
    2.  **Channel independence**: each feature channel is processed independently
        through the same Transformer encoder (parameter sharing).
    3.  **Longer context**: patching reduces the token count from T to T/P,
        enabling attention over much longer windows (6 hrs / 3 days) within
        feasible memory budgets.

Default configuration:
    - context_len = 360  (6-hour window at 1-min cadence)
    - patch_len   = 16   → 22 patches per channel
    - d_model     = 64
    - n_heads     = 4
    - n_layers    = 3
    - output      = sigmoid probability (binary flare yes/no)

Falls back to LSTMForecaster when PyTorch is not available.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from src.forecast.baseline_model import BaseForecaster


class PatchTSTForecaster(BaseForecaster):
    """
    PatchTST-based binary forecaster implementing the BaseForecaster protocol.

    Parameters
    ----------
    context_len : int
        Number of time steps in the input window.
    patch_len : int
        Length of each non-overlapping patch.
    d_model : int
        Transformer hidden dimension.
    n_heads : int
        Number of attention heads.
    n_layers : int
        Number of Transformer encoder layers.
    epochs : int
        Training epochs.
    lr : float
        Learning rate.
    dropout : float
        Dropout probability.
    """

    def __init__(
        self,
        context_len: int = 360,
        patch_len: int = 16,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 3,
        epochs: int = 10,
        lr: float = 1e-3,
        dropout: float = 0.1,
    ):
        self.context_len = context_len
        self.patch_len = patch_len
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.epochs = epochs
        self.lr = lr
        self.dropout = dropout
        self.is_fitted = False

        self._pytorch_available = False
        self._net = None
        self._fallback = None

        try:
            import torch
            self._pytorch_available = True
        except ImportError:
            from src.forecast.sequence_model import LSTMForecaster
            self._fallback = LSTMForecaster(epochs=epochs, lr=lr)

    def fit(self, X: np.ndarray, y: np.ndarray):
        if not self._pytorch_available:
            self._fallback.fit(X, y)
            self.is_fitted = True
            return self

        import torch
        import torch.nn as nn
        from torch.utils.data import TensorDataset, DataLoader

        if X.ndim == 2:
            X = np.expand_dims(X, axis=-1)  # (N, T) → (N, T, 1)

        n_samples, seq_len, n_features = X.shape

        # Pad/truncate to context_len
        X_proc = self._pad_or_truncate(X, self.context_len)

        # Build model
        self._net = self._build_patchtst(n_features)
        optimizer = torch.optim.AdamW(self._net.parameters(), lr=self.lr, weight_decay=1e-4)
        criterion = nn.BCELoss()
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)

        x_tensor = torch.tensor(X_proc, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)

        dataset = TensorDataset(x_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=min(64, max(1, n_samples)), shuffle=True)

        self._net.train()
        for epoch in range(self.epochs):
            for bx, by in loader:
                optimizer.zero_grad()
                pred = self._net(bx)
                loss = criterion(pred, by)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._net.parameters(), 1.0)
                optimizer.step()
            scheduler.step()

        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted or len(X) == 0:
            return np.zeros(len(X), dtype=float)

        if not self._pytorch_available:
            return self._fallback.predict_proba(X)

        import torch

        if X.ndim == 2:
            X = np.expand_dims(X, axis=-1)

        X_proc = self._pad_or_truncate(X, self.context_len)

        self._net.eval()
        with torch.no_grad():
            x_tensor = torch.tensor(X_proc, dtype=torch.float32)
            preds = self._net(x_tensor).cpu().numpy()

        return np.atleast_1d(preds)

    def _pad_or_truncate(self, X: np.ndarray, target_len: int) -> np.ndarray:
        """Pad (left-pad with zeros) or truncate sequences to target_len."""
        n, seq_len, n_feat = X.shape
        if seq_len == target_len:
            return X
        elif seq_len > target_len:
            return X[:, -target_len:, :]
        else:
            pad = np.zeros((n, target_len - seq_len, n_feat), dtype=X.dtype)
            return np.concatenate([pad, X], axis=1)

    def _build_patchtst(self, n_features: int):
        """Construct the PatchTST network."""
        import torch
        import torch.nn as nn

        context_len = self.context_len
        patch_len = self.patch_len
        d_model = self.d_model
        n_heads = self.n_heads
        n_layers = self.n_layers
        dropout = self.dropout

        n_patches = context_len // patch_len

        class PatchEmbedding(nn.Module):
            """Convert (B, T, C) → (B * C, n_patches, d_model) via patching."""

            def __init__(self):
                super().__init__()
                self.proj = nn.Linear(patch_len, d_model)
                self.pos_enc = nn.Parameter(torch.randn(1, n_patches, d_model) * 0.02)

            def forward(self, x):
                # x: (B, T, C)
                B, T, C = x.shape
                # Truncate T to be divisible by patch_len
                T_eff = (T // patch_len) * patch_len
                x = x[:, -T_eff:, :]  # right-align

                # Reshape to patches: (B, n_patches, patch_len, C)
                x = x.reshape(B, T_eff // patch_len, patch_len, C)

                # Channel independence: (B, n_patches, patch_len, C) → (B*C, n_patches, patch_len)
                x = x.permute(0, 3, 1, 2).reshape(B * C, T_eff // patch_len, patch_len)

                # Project patches to d_model
                x = self.proj(x)  # (B*C, n_patches, d_model)

                # Add positional encoding
                actual_patches = x.shape[1]
                x = x + self.pos_enc[:, :actual_patches, :]

                return x, B, C

        class PatchTSTNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.patch_embed = PatchEmbedding()

                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=n_heads,
                    dim_feedforward=d_model * 4,
                    dropout=dropout,
                    batch_first=True,
                    norm_first=True,
                )
                self.transformer = nn.TransformerEncoder(
                    encoder_layer, num_layers=n_layers
                )
                self.norm = nn.LayerNorm(d_model)

                # Classification head: aggregate across patches and channels
                self.head = nn.Sequential(
                    nn.Linear(d_model, d_model // 2),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_model // 2, 1),
                    nn.Sigmoid(),
                )

            def forward(self, x):
                # x: (B, T, C)
                z, B, C = self.patch_embed(x)  # (B*C, n_patches, d_model)

                # Transformer encoder
                z = self.transformer(z)  # (B*C, n_patches, d_model)
                z = self.norm(z)

                # Global average pooling over patches
                z = z.mean(dim=1)  # (B*C, d_model)

                # Reshape back: (B, C, d_model)
                z = z.reshape(B, C, d_model)

                # Average over channels
                z = z.mean(dim=1)  # (B, d_model)

                # Classification
                out = self.head(z).squeeze(-1)  # (B,)
                return out

        return PatchTSTNet()
