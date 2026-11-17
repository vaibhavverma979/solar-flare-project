"""
visual_features.py — Physics-motivated feature extraction from SUIT UV images.

Extracts spatial and intensity features that correlate with flare
productivity without requiring the full CNN.  These features are
appended to the time-series feature DataFrame for multi-modal fusion.

Feature set
-----------
1.  uv_intensity_mean       — Mean UV intensity (chromospheric heating proxy)
2.  uv_intensity_std        — Intensity std (activity level proxy)
3.  uv_gradient_magnitude   — Spatial gradient norm (edge complexity)
4.  active_region_count      — # bright regions above threshold (connected components)
5.  active_region_area_frac  — Fraction of disc occupied by active regions
6.  polarity_inversion_proxy — Variance of Laplacian (magnetic neutral line complexity)
7.  max_gradient_asymmetry   — Max directional gradient imbalance (shear proxy)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.vision.suit_reader import SuitImage


@dataclass
class VisualFeatures:
    """Container for extracted SUIT visual features."""

    uv_intensity_mean: float
    uv_intensity_std: float
    uv_gradient_magnitude: float
    active_region_count: int
    active_region_area_frac: float
    polarity_inversion_proxy: float
    max_gradient_asymmetry: float
    timestamp: str  # ISO-8601 UTC

    def to_dict(self) -> dict:
        return {
            "uv_intensity_mean": self.uv_intensity_mean,
            "uv_intensity_std": self.uv_intensity_std,
            "uv_gradient_magnitude": self.uv_gradient_magnitude,
            "active_region_count": self.active_region_count,
            "active_region_area_frac": self.active_region_area_frac,
            "polarity_inversion_proxy": self.polarity_inversion_proxy,
            "max_gradient_asymmetry": self.max_gradient_asymmetry,
        }


def extract_visual_features(suit_image: SuitImage) -> VisualFeatures:
    """
    Extract physics-motivated visual features from a SUIT image.

    Parameters
    ----------
    suit_image : SuitImage
        Normalised SUIT observation.

    Returns
    -------
    VisualFeatures
        Feature vector for multi-modal fusion.
    """
    img = suit_image.image.astype(np.float64)

    # 1–2. Basic intensity statistics
    uv_mean = float(np.mean(img))
    uv_std = float(np.std(img))

    # 3. Gradient magnitude (Sobel-like finite differences)
    gy = np.diff(img, axis=0)
    gx = np.diff(img, axis=1)
    min_h = min(gy.shape[0], gx.shape[0])
    min_w = min(gy.shape[1], gx.shape[1])
    grad_map = np.sqrt(gy[:min_h, :min_w] ** 2 + gx[:min_h, :min_w] ** 2)
    uv_grad_mag = float(np.mean(grad_map))

    # 4–5. Active region detection (connected component analysis via flood fill)
    threshold = uv_mean + 2.0 * uv_std
    binary = (img > threshold).astype(np.uint8)
    ar_count, ar_area_frac = _count_connected_components(binary)

    # 6. Polarity inversion proxy (variance of Laplacian)
    lap = _laplacian(img)
    polarity_proxy = float(np.var(lap))

    # 7. Gradient asymmetry (max directional imbalance → shear proxy)
    gy_abs_mean = float(np.mean(np.abs(gy)))
    gx_abs_mean = float(np.mean(np.abs(gx)))
    denom = max(gy_abs_mean + gx_abs_mean, 1e-12)
    asymmetry = float(abs(gy_abs_mean - gx_abs_mean) / denom)

    return VisualFeatures(
        uv_intensity_mean=uv_mean,
        uv_intensity_std=uv_std,
        uv_gradient_magnitude=uv_grad_mag,
        active_region_count=ar_count,
        active_region_area_frac=ar_area_frac,
        polarity_inversion_proxy=polarity_proxy,
        max_gradient_asymmetry=asymmetry,
        timestamp=suit_image.timestamp.isoformat(),
    )


# ── Internal helpers ──────────────────────────────────────────────────────────

def _count_connected_components(binary: np.ndarray) -> tuple[int, float]:
    """
    Count connected components in a binary mask using a simple
    label-propagation (no scipy/opencv dependency).

    Returns (count, area_fraction).
    """
    try:
        from scipy.ndimage import label as scipy_label
        labelled, n = scipy_label(binary)
        total_pixels = binary.size
        ar_pixels = int(np.sum(binary))
        return int(n), float(ar_pixels / total_pixels) if total_pixels > 0 else 0.0
    except ImportError:
        pass

    # Fallback: naive row-scan estimate
    h, w = binary.shape
    visited = np.zeros_like(binary, dtype=bool)
    count = 0
    total_active = 0

    for r in range(h):
        for c in range(w):
            if binary[r, c] == 1 and not visited[r, c]:
                # BFS flood fill
                count += 1
                queue = [(r, c)]
                visited[r, c] = True
                while queue:
                    cr, cc = queue.pop(0)
                    total_active += 1
                    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and binary[nr, nc] == 1:
                            visited[nr, nc] = True
                            queue.append((nr, nc))

    total_pixels = h * w
    area_frac = float(total_active / total_pixels) if total_pixels > 0 else 0.0
    return count, area_frac


def _laplacian(img: np.ndarray) -> np.ndarray:
    """Discrete Laplacian (5-point stencil)."""
    lap = np.zeros_like(img)
    lap[1:-1, 1:-1] = (
        img[:-2, 1:-1] + img[2:, 1:-1] +
        img[1:-1, :-2] + img[1:-1, 2:] -
        4 * img[1:-1, 1:-1]
    )
    return lap
