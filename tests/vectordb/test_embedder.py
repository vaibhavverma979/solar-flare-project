"""Tests for the light-curve embedder module."""

import numpy as np
import pytest

from src.vectordb.embedder import StatisticalEmbedder, LightCurveEmbedder


class TestStatisticalEmbedder:
    """Test the handcrafted statistical embedder."""

    def test_embed_shape(self):
        embedder = StatisticalEmbedder(embedding_dim=128)
        flux = np.random.rand(60)
        emb = embedder.embed(flux)
        assert emb.shape == (128,)
        assert emb.dtype == np.float32

    def test_embed_l2_normalised(self):
        embedder = StatisticalEmbedder(embedding_dim=128)
        flux = np.random.rand(100)
        emb = embedder.embed(flux)
        norm = np.linalg.norm(emb)
        np.testing.assert_almost_equal(norm, 1.0, decimal=5)

    def test_embed_empty_array(self):
        embedder = StatisticalEmbedder(embedding_dim=128)
        emb = embedder.embed(np.array([]))
        assert emb.shape == (128,)
        assert np.all(emb == 0)

    def test_embed_single_value(self):
        embedder = StatisticalEmbedder(embedding_dim=128)
        emb = embedder.embed(np.array([42.0]))
        assert emb.shape == (128,)

    def test_embed_reproducible(self):
        embedder = StatisticalEmbedder(embedding_dim=128)
        flux = np.random.RandomState(42).rand(60)
        emb1 = embedder.embed(flux)
        emb2 = embedder.embed(flux)
        np.testing.assert_array_equal(emb1, emb2)

    def test_similar_curves_have_high_similarity(self):
        embedder = StatisticalEmbedder(embedding_dim=128)
        flux1 = np.linspace(0, 1, 60) ** 2
        flux2 = np.linspace(0, 1, 60) ** 2.1  # Very similar
        emb1 = embedder.embed(flux1)
        emb2 = embedder.embed(flux2)
        sim = float(np.dot(emb1, emb2))
        assert sim > 0.9  # Should be very similar

    def test_different_curves_have_lower_similarity(self):
        embedder = StatisticalEmbedder(embedding_dim=128)
        flux1 = np.ones(60)  # Flat
        flux2 = np.exp(np.linspace(0, 5, 60))  # Exponential rise
        emb1 = embedder.embed(flux1)
        emb2 = embedder.embed(flux2)
        sim = float(np.dot(emb1, emb2))
        assert sim < 0.95  # Should be less similar

    def test_dim_property(self):
        embedder = StatisticalEmbedder(embedding_dim=64)
        assert embedder.dim == 64
        emb = embedder.embed(np.random.rand(30))
        assert emb.shape == (64,)


class TestLightCurveEmbedder:
    """Test the CNN-based embedder (may fall back to statistical)."""

    def test_embed_shape(self):
        embedder = LightCurveEmbedder(embedding_dim=128)
        flux = np.random.rand(60)
        emb = embedder.embed(flux)
        assert emb.shape == (128,)
        assert emb.dtype == np.float32

    def test_embed_empty(self):
        embedder = LightCurveEmbedder(embedding_dim=128)
        emb = embedder.embed(np.array([]))
        assert emb.shape == (128,)

    def test_dim_property(self):
        embedder = LightCurveEmbedder(embedding_dim=128)
        assert embedder.dim == 128
