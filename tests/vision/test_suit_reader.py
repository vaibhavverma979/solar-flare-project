"""Tests for the SUIT FITS reader and synthetic image generator."""

import numpy as np
import pytest
from datetime import datetime, timezone

from src.vision.suit_reader import SyntheticSuitGenerator, SuitImage


class TestSyntheticSuitGenerator:
    """Test the synthetic SUIT image generator."""

    def test_generate_default(self):
        gen = SyntheticSuitGenerator(size=128, seed=42)
        img = gen.generate()
        assert isinstance(img, SuitImage)
        assert img.image.shape == (128, 128)
        assert img.image.dtype == np.float32
        assert img.image.min() >= 0.0
        assert img.image.max() <= 1.0

    def test_generate_all_complexity_levels(self):
        gen = SyntheticSuitGenerator(size=64, seed=123)
        for complexity in ("alpha", "beta", "beta-gamma", "beta-gamma-delta"):
            img = gen.generate(n_sunspots=2, complexity=complexity)
            assert img.image.shape == (64, 64)
            assert img.metadata.get("complexity") == complexity

    def test_generate_with_timestamp(self):
        gen = SyntheticSuitGenerator(size=64, seed=0)
        ts = datetime(2026, 7, 31, 12, 0, 0, tzinfo=timezone.utc)
        img = gen.generate(timestamp=ts)
        assert img.timestamp == ts

    def test_generate_reproducible(self):
        gen1 = SyntheticSuitGenerator(size=64, seed=99)
        gen2 = SyntheticSuitGenerator(size=64, seed=99)
        img1 = gen1.generate(n_sunspots=3, complexity="beta")
        img2 = gen2.generate(n_sunspots=3, complexity="beta")
        np.testing.assert_array_equal(img1.image, img2.image)

    def test_image_size_stored(self):
        gen = SyntheticSuitGenerator(size=256, seed=0)
        img = gen.generate()
        assert img.image_size == (256, 256)

    def test_metadata_fields(self):
        gen = SyntheticSuitGenerator(size=64, seed=0)
        img = gen.generate(n_sunspots=5, complexity="beta-gamma-delta")
        assert img.wavelength_nm == 214.0
        assert img.exposure_s == 1.0
        assert img.metadata["synthetic"] == "true"
        assert img.metadata["n_sunspots"] == "5"
