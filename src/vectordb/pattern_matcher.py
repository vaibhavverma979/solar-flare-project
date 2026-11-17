"""
pattern_matcher.py — Orchestrator tying the embedder and historical DB together.

Thread-safe, lazy-initialising wrapper that provides a single call:
    match_current_pattern(flux_window) → PatternMatchResult
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from src.vectordb.embedder import LightCurveEmbedder, StatisticalEmbedder
from src.vectordb.historical_db import HistoricalFlareDB, HistoricalMatch


@dataclass
class PatternMatchResult:
    """Complete result from the pattern-matching pipeline."""

    query_embedding: np.ndarray = field(repr=False)  # 128-d
    top_matches: list[HistoricalMatch] = field(default_factory=list)
    best_match_name: str = ""
    best_match_similarity: float = 0.0
    best_match_narrative: str = ""
    database_size: int = 0
    embedder_type: str = ""  # "cnn" or "statistical"

    def to_dict(self) -> dict:
        """Serialise for API response (excludes raw embedding)."""
        return {
            "best_match": {
                "event_name": self.best_match_name,
                "similarity": self.best_match_similarity,
                "narrative": self.best_match_narrative,
            },
            "top_matches": [
                {
                    "event_name": m.event_name,
                    "date": m.date,
                    "flare_class": m.flare_class,
                    "similarity": m.similarity,
                    "description": m.description,
                    "peak_flux": m.peak_flux,
                    "narrative": m.narrative,
                }
                for m in self.top_matches
            ],
            "database_size": self.database_size,
            "embedder_type": self.embedder_type,
        }


class PatternMatcher:
    """
    Thread-safe pattern matcher that encodes real-time flux windows and
    queries the historical flare vector database.

    Parameters
    ----------
    db_persist_dir : str or None
        Path to persist the ChromaDB data.  None = in-memory.
    embedding_dim : int
        Embedding dimensionality (must match DB and embedder).
    use_cnn_embedder : bool
        If True, use the 1-D ConvNet embedder (requires PyTorch).
        If False, use the statistical embedder (no dependencies).
    """

    def __init__(
        self,
        db_persist_dir: Optional[str] = None,
        embedding_dim: int = 128,
        use_cnn_embedder: bool = True,
    ):
        self._lock = threading.Lock()
        self._embedding_dim = embedding_dim
        self._db_persist_dir = db_persist_dir

        # Lazy init
        self._db: Optional[HistoricalFlareDB] = None
        self._embedder = None

        if use_cnn_embedder:
            try:
                self._embedder = LightCurveEmbedder(embedding_dim)
                self._embedder_type = "cnn"
            except Exception:
                self._embedder = StatisticalEmbedder(embedding_dim)
                self._embedder_type = "statistical"
        else:
            self._embedder = StatisticalEmbedder(embedding_dim)
            self._embedder_type = "statistical"

    def _ensure_db(self) -> HistoricalFlareDB:
        """Lazy-initialise the database (thread-safe)."""
        if self._db is None:
            with self._lock:
                if self._db is None:
                    self._db = HistoricalFlareDB(
                        persist_dir=self._db_persist_dir,
                        embedding_dim=self._embedding_dim,
                    )
        return self._db

    def match_current_pattern(
        self,
        flux_window: np.ndarray,
        top_k: int = 5,
    ) -> PatternMatchResult:
        """
        Encode the current flux window and find the most similar historical
        flare profiles.

        Parameters
        ----------
        flux_window : np.ndarray
            1-D array of flux values from the current observation window.
        top_k : int
            Number of best matches to return.

        Returns
        -------
        PatternMatchResult
        """
        flux = np.asarray(flux_window, dtype=np.float64).ravel()

        if len(flux) == 0:
            return PatternMatchResult(
                query_embedding=np.zeros(self._embedding_dim, dtype=np.float32),
                embedder_type=self._embedder_type,
            )

        # Encode
        embedding = self._embedder.embed(flux)

        # Search
        db = self._ensure_db()
        matches = db.search_similar(embedding, top_k=top_k)

        best = matches[0] if matches else None

        return PatternMatchResult(
            query_embedding=embedding,
            top_matches=matches,
            best_match_name=best.event_name if best else "",
            best_match_similarity=best.similarity if best else 0.0,
            best_match_narrative=best.narrative if best else "No historical match found.",
            database_size=db.count(),
            embedder_type=self._embedder_type,
        )
