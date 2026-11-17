"""
ml_detector.py — ML-based false-positive suppression for nowcasting (Phase 3).

Physics context
---------------
Sigma-threshold detectors (threshold_detector.py) are fast but generate false
positives from:
- Cosmic ray showers (short spike, not instrument-coincident)
- SAA particle precipitation (usually flagged by quality_flag but not always)
- Statistical noise bursts in quiet-sun data

A lightweight 1D-CNN or XGBoost classifier trained on the time-series window
around each candidate event can distinguish these from genuine flares by
learning the characteristic FRED shape of a real flare.

This module defines a common Detector interface so that the threshold
detector and ML detector are interchangeable behind a single API.

TODO (Phase 3): Implement XGBoostDetector and CNNDetector.
"""


import numpy as np
from typing import Any


class BaseDetector:
    """Abstract base class for all nowcast detectors."""

    def fit(self, X, y):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError

    def predict_proba(self, X):
        raise NotImplementedError


class RandomForestDetector(BaseDetector):
    """RandomForest classifier for false-positive flare suppression."""

    def __init__(self, n_estimators: int = 100, random_state: int = 42):
        from sklearn.ensemble import RandomForestClassifier
        self.model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
        self.is_fitted = False

    def fit(self, X, y):
        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, X):
        if not self.is_fitted:
            return np.ones(len(X), dtype=int)
        return self.model.predict(X)

    def predict_proba(self, X):
        if not self.is_fitted:
            return np.ones(len(X), dtype=float)
        probs = self.model.predict_proba(X)
        return probs[:, 1] if probs.ndim == 2 and probs.shape[1] > 1 else probs.ravel()


class XGBoostDetector(BaseDetector):
    """XGBoost or GradientBoosting classifier for nowcast detection."""

    def __init__(self, n_estimators: int = 100, random_state: int = 42):
        try:
            from xgboost import XGBClassifier
            self.model: Any = XGBClassifier(n_estimators=n_estimators, random_state=random_state)
        except ImportError:
            from sklearn.ensemble import GradientBoostingClassifier
            self.model = GradientBoostingClassifier(n_estimators=n_estimators, random_state=random_state)
        self.is_fitted = False

    def fit(self, X, y):
        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, X):
        if not self.is_fitted:
            return np.ones(len(X), dtype=int)
        return self.model.predict(X)

    def predict_proba(self, X):
        if not self.is_fitted:
            return np.ones(len(X), dtype=float)
        probs = self.model.predict_proba(X)
        return probs[:, 1] if probs.ndim == 2 and probs.shape[1] > 1 else probs.ravel()

