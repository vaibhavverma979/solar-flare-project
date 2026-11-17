"""
suit_reader.py — FITS reader for SUIT (Solar Ultraviolet Imaging Telescope) images.

SUIT is one of the seven payloads on Aditya-L1, operating in 11 narrowband
channels spanning 200–400 nm (near-UV).  Images are 2832 × 2832 pixels in
the standard PRADAN FITS format.

This reader handles both real SUIT FITS files from PRADAN and synthetic
test images produced by SyntheticSuitGenerator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class SuitImage:
    """Container for a single SUIT observation frame."""

    image: np.ndarray                        # (H, W) float32, normalised [0, 1]
    timestamp: datetime                      # UTC observation time
    wavelength_nm: float                     # central filter wavelength (nm)
    exposure_s: float                        # exposure duration (seconds)
    image_size: tuple[int, int] = (0, 0)     # (height, width)
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        self.image_size = (self.image.shape[0], self.image.shape[1])


def read_suit(path: Path | str) -> SuitImage:
    """
    Read a SUIT FITS file and return a normalised SuitImage.

    Parameters
    ----------
    path : Path or str
        Path to the SUIT FITS file.

    Returns
    -------
    SuitImage
        Normalised image with metadata.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"SUIT file not found: {path}")

    try:
        from astropy.io import fits as afits
    except ImportError:
        raise ImportError("astropy is required to read SUIT FITS files")

    with afits.open(str(path)) as hdul:
        # Primary HDU typically contains the image data
        hdu = hdul[0] if hdul[0].data is not None else hdul[1]
        raw = hdu.data.astype(np.float32)
        header = hdu.header

        # Normalise to [0, 1]
        vmin, vmax = np.nanmin(raw), np.nanmax(raw)
        if vmax > vmin:
            image = (raw - vmin) / (vmax - vmin)
        else:
            image = np.zeros_like(raw)

        # Parse metadata
        obs_time = _parse_timestamp(header)
        wavelength = float(header.get("WAVELNTH", header.get("WAVELENG", 214.0)))
        exposure = float(header.get("EXPTIME", header.get("EXPOSURE", 1.0)))

        meta = {
            k: str(header[k])
            for k in ("OBJECT", "TELESCOP", "INSTRUME", "FILTER", "NAXIS1", "NAXIS2")
            if k in header
        }

    return SuitImage(
        image=image,
        timestamp=obs_time,
        wavelength_nm=wavelength,
        exposure_s=exposure,
        metadata=meta,
    )


def _parse_timestamp(header) -> datetime:
    """Extract UTC datetime from a FITS header."""
    for key in ("DATE-OBS", "DATE_OBS", "DATE-BEG"):
        val = header.get(key)
        if val:
            try:
                from astropy.time import Time
                return Time(val, format="isot", scale="utc").to_datetime(timezone=timezone.utc)
            except Exception:
                pass
    return datetime.now(tz=timezone.utc)


# ──────────────────────────────────────────────────────────────────────────────
# Synthetic image generator (for testing without real SUIT data)
# ──────────────────────────────────────────────────────────────────────────────

class SyntheticSuitGenerator:
    """
    Generate synthetic solar UV images that mimic SUIT observations for
    testing.  Creates a limb-darkened solar disc with configurable sunspot
    regions of varying magnetic complexity.

    Parameters
    ----------
    size : int
        Image side length in pixels (default 512 — faster than full 2832).
    seed : int or None
        Random seed for reproducibility.
    """

    MAGNETIC_CLASSES = ("alpha", "beta", "beta-gamma", "beta-gamma-delta")

    def __init__(self, size: int = 512, seed: Optional[int] = None):
        self.size = size
        self.rng = np.random.default_rng(seed)

    def generate(
        self,
        n_sunspots: int = 3,
        complexity: str = "beta-gamma",
        timestamp: Optional[datetime] = None,
    ) -> SuitImage:
        """
        Generate a single synthetic SUIT-like image.

        Parameters
        ----------
        n_sunspots : int
            Number of sunspot groups to place on the disc.
        complexity : str
            One of 'alpha', 'beta', 'beta-gamma', 'beta-gamma-delta'.
        timestamp : datetime or None
            If None, uses current UTC time.

        Returns
        -------
        SuitImage
        """
        img = self._make_solar_disc()
        img = self._add_sunspots(img, n_sunspots, complexity)
        img = self._add_noise(img)

        # Normalise
        vmin, vmax = img.min(), img.max()
        if vmax > vmin:
            img = (img - vmin) / (vmax - vmin)
        else:
            img = np.zeros_like(img)

        ts = timestamp or datetime.now(tz=timezone.utc)

        return SuitImage(
            image=img.astype(np.float32),
            timestamp=ts,
            wavelength_nm=214.0,
            exposure_s=1.0,
            metadata={
                "synthetic": "true",
                "n_sunspots": str(n_sunspots),
                "complexity": complexity,
            },
        )

    # ── internal helpers ─────────────────────────────────────────────────

    def _make_solar_disc(self) -> np.ndarray:
        """Create limb-darkened solar disc on a dark background."""
        s = self.size
        y, x = np.ogrid[-s // 2 : s // 2, -s // 2 : s // 2]
        r = np.sqrt(x ** 2 + y ** 2).astype(np.float32)
        radius = s * 0.42

        # Limb darkening (Eddington approximation: I/I0 ≈ 0.4 + 0.6·μ)
        mu = np.sqrt(np.clip(1 - (r / radius) ** 2, 0, 1))
        disc = (0.4 + 0.6 * mu) * (r < radius)
        return disc

    def _add_sunspots(self, img: np.ndarray, n: int, complexity: str) -> np.ndarray:
        """Place Gaussian 'sunspot' blobs with polarity structure."""
        s = self.size
        radius = s * 0.35
        complexity_idx = self.MAGNETIC_CLASSES.index(complexity) if complexity in self.MAGNETIC_CLASSES else 1

        for _ in range(n):
            # Random position on disc
            angle = self.rng.uniform(0, 2 * np.pi)
            dist = self.rng.uniform(0, radius * 0.7)
            cx = int(s // 2 + dist * np.cos(angle))
            cy = int(s // 2 + dist * np.sin(angle))

            # Spot size scales with complexity
            base_sigma = s * 0.02 * (1 + complexity_idx * 0.5)
            n_components = 1 + complexity_idx  # alpha=1, beta=2, bg=3, bgd=4

            for comp in range(n_components):
                offset_x = int(self.rng.normal(0, base_sigma * 0.8))
                offset_y = int(self.rng.normal(0, base_sigma * 0.8))
                sigma = base_sigma * self.rng.uniform(0.5, 1.5)
                amplitude = self.rng.uniform(0.3, 0.8)

                # Alternating polarity (dark/bright) for complex groups
                sign = -1 if comp % 2 == 0 else 0.5
                y, x = np.ogrid[0:s, 0:s]
                blob = amplitude * np.exp(
                    -(((x - cx - offset_x) ** 2 + (y - cy - offset_y) ** 2)
                      / (2 * sigma ** 2))
                )
                img = img + sign * blob.astype(np.float32)

        return img

    def _add_noise(self, img: np.ndarray) -> np.ndarray:
        """Add photon-counting Poisson-like noise."""
        noise = self.rng.normal(0, 0.02, size=img.shape).astype(np.float32)
        return img + noise
