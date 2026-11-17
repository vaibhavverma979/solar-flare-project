"""Tests for the pattern matcher orchestrator."""

import numpy as np
import pytest

from src.vectordb.pattern_matcher import PatternMatcher, PatternMatchResult


class TestPatternMatcher:
    """Test the end-to-end pattern matching pipeline."""

    def test_match_returns_correct_type(self):
        matcher = PatternMatcher(use_cnn_embedder=False)
        flux = np.random.rand(60)
        result = matcher.match_current_pattern(flux, top_k=5)
        assert isinstance(result, PatternMatchResult)

    def test_match_returns_top_k_results(self):
        matcher = PatternMatcher(use_cnn_embedder=False)
        flux = np.exp(np.linspace(0, 3, 60)) * 1e-4
        result = matcher.match_current_pattern(flux, top_k=3)
        assert len(result.top_matches) <= 3

    def test_match_empty_flux(self):
        matcher = PatternMatcher(use_cnn_embedder=False)
        result = matcher.match_current_pattern(np.array([]), top_k=5)
        assert isinstance(result, PatternMatchResult)
        assert result.best_match_name == ""

    def test_match_has_narrative(self):
        matcher = PatternMatcher(use_cnn_embedder=False)
        flux = np.random.rand(60) * 1e-4
        result = matcher.match_current_pattern(flux, top_k=1)
        if result.top_matches:
            assert result.best_match_narrative != ""
            assert "match" in result.best_match_narrative.lower() or "similar" in result.best_match_narrative.lower()

    def test_to_dict_serialisation(self):
        matcher = PatternMatcher(use_cnn_embedder=False)
        flux = np.random.rand(60)
        result = matcher.match_current_pattern(flux, top_k=3)
        d = result.to_dict()
        assert "best_match" in d
        assert "top_matches" in d
        assert "database_size" in d
        assert "embedder_type" in d

    def test_database_size_positive(self):
        matcher = PatternMatcher(use_cnn_embedder=False)
        flux = np.random.rand(60)
        result = matcher.match_current_pattern(flux, top_k=1)
        assert result.database_size >= 40

    def test_embedder_type_is_statistical(self):
        matcher = PatternMatcher(use_cnn_embedder=False)
        flux = np.random.rand(60)
        result = matcher.match_current_pattern(flux, top_k=1)
        assert result.embedder_type == "statistical"
