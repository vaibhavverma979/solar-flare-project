"""Tests for the PatchTST forecaster model."""

import numpy as np
import pytest

from src.forecast.baseline_model import BaseForecaster
from src.forecast.sequence_model import build_model


class TestPatchTSTFactory:
    """Test that the build_model factory creates PatchTST correctly."""

    def test_build_patchtst(self):
        model = build_model("patchtst", context_len=32, patch_len=8, epochs=1)
        assert isinstance(model, BaseForecaster)

    def test_build_patch_tst_underscore(self):
        model = build_model("patch_tst", context_len=32, patch_len=8, epochs=1)
        assert isinstance(model, BaseForecaster)


class TestPatchTSTForecaster:
    """Test the PatchTST model (may fall back to LSTM)."""

    def test_fit_and_predict_2d(self):
        model = build_model("patchtst", context_len=32, patch_len=8, epochs=1)
        X = np.random.rand(10, 32).astype(np.float32)
        y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        model.fit(X, y)
        proba = model.predict_proba(X)
        assert proba.shape[0] == 10
        assert np.all((proba >= 0) & (proba <= 1))

    def test_fit_and_predict_3d(self):
        model = build_model("patchtst", context_len=32, patch_len=8, epochs=1)
        X = np.random.rand(8, 32, 3).astype(np.float32)  # 3 features
        y = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        model.fit(X, y)
        proba = model.predict_proba(X)
        assert proba.shape[0] == 8
        assert np.all((proba >= 0) & (proba <= 1))

    def test_predict_empty_returns_zeros(self):
        model = build_model("patchtst", context_len=32, patch_len=8, epochs=1)
        proba = model.predict_proba(np.array([]))
        assert len(proba) == 0

    def test_predict_before_fit_returns_zeros(self):
        model = build_model("patchtst", context_len=32, patch_len=8)
        X = np.random.rand(5, 32).astype(np.float32)
        proba = model.predict_proba(X)
        assert np.all(proba == 0)

    def test_pad_short_sequences(self):
        """Sequences shorter than context_len should be zero-padded."""
        model = build_model("patchtst", context_len=64, patch_len=8, epochs=1)
        X = np.random.rand(5, 20, 2).astype(np.float32)  # Only 20 steps, context=64
        y = np.array([0, 1, 0, 1, 0])
        model.fit(X, y)
        proba = model.predict_proba(X)
        assert proba.shape[0] == 5

    def test_truncate_long_sequences(self):
        """Sequences longer than context_len should be truncated."""
        model = build_model("patchtst", context_len=32, patch_len=8, epochs=1)
        X = np.random.rand(5, 100, 2).astype(np.float32)  # 100 steps > context=32
        y = np.array([0, 1, 0, 1, 0])
        model.fit(X, y)
        proba = model.predict_proba(X)
        assert proba.shape[0] == 5
