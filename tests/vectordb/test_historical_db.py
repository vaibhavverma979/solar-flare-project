"""Tests for the historical flare vector database."""

import numpy as np
import pytest

from src.vectordb.historical_db import (
    HistoricalFlareDB,
    HistoricalFlareEntry,
    HistoricalMatch,
)
from src.vectordb.embedder import StatisticalEmbedder


class TestHistoricalFlareDB:
    """Test the ChromaDB-backed (or fallback) historical flare database."""

    def test_auto_seeds_canonical_events(self):
        db = HistoricalFlareDB(persist_dir=None)
        assert db.count() >= 40  # Should have ~50 canonical events

    def test_search_returns_matches(self):
        db = HistoricalFlareDB(persist_dir=None)
        embedder = StatisticalEmbedder(embedding_dim=128)
        # Create a query similar to an X-class flare (high peak, fast rise)
        flux = np.exp(np.linspace(0, 3, 60)) * 1e-4
        emb = embedder.embed(flux)
        matches = db.search_similar(emb, top_k=5)
        assert len(matches) <= 5
        assert len(matches) > 0
        assert all(isinstance(m, HistoricalMatch) for m in matches)

    def test_match_fields(self):
        db = HistoricalFlareDB(persist_dir=None)
        embedder = StatisticalEmbedder(embedding_dim=128)
        flux = np.random.rand(60)
        emb = embedder.embed(flux)
        matches = db.search_similar(emb, top_k=1)
        if matches:
            m = matches[0]
            assert m.event_name != ""
            assert m.date != ""
            assert m.flare_class != ""
            assert 0 <= m.similarity <= 1
            assert m.narrative != ""

    def test_add_custom_event(self):
        db = HistoricalFlareDB(persist_dir=None)
        initial_count = db.count()

        custom_emb = np.random.rand(128).astype(np.float32)
        custom_emb /= np.linalg.norm(custom_emb)

        entry = HistoricalFlareEntry(
            event_id="TEST_001",
            event_name="Test Custom Event",
            date="2026-01-01",
            flare_class="M5.0",
            peak_flux=5.0e-5,
            duration_min=45,
            description="Test event for unit testing",
            embedding=custom_emb,
        )
        db.add_event(entry)
        assert db.count() == initial_count + 1

    def test_similarity_ordering(self):
        """Closer embeddings should have higher similarity scores."""
        db = HistoricalFlareDB(persist_dir=None)
        embedder = StatisticalEmbedder(embedding_dim=128)
        flux = np.random.RandomState(42).rand(60)
        emb = embedder.embed(flux)
        matches = db.search_similar(emb, top_k=5)
        if len(matches) >= 2:
            # First match should have highest similarity
            assert matches[0].similarity >= matches[-1].similarity
