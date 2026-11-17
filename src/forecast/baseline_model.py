"""
baseline_model.py — Logistic regression / random forest baseline forecaster (Phase 4).

Implement this *before* the LSTM so we have a working end-to-end pipeline
and a meaningful baseline to compare the sequence model against.

Common interface with sequence_model.py
---------------------------------------
Both models must implement: fit(X, y), predict_proba(X), and save/load.
This lets the API serve either model behind the same endpoint.

TODO (Phase 4): Implement LogisticBaseline and RandomForestBaseline.
"""


import pickle
from pathlib import Path
import numpy as np


class BaseForecaster:
    def fit(self, X, y):
        raise NotImplementedError

    def predict_proba(self, X):
        raise NotImplementedError

    def save(self, path: str | Path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str | Path):
        with open(path, "rb") as f:
            return pickle.load(f)


def _flatten_x(X: np.ndarray) -> np.ndarray:
    if X.ndim == 3:
        n, seq_len, n_feat = X.shape
        return X.reshape(n, seq_len * n_feat)
    return X


class LogisticBaseline(BaseForecaster):
    def __init__(self, C: float = 1.0, max_iter: int = 1000, random_state: int = 42):
        from sklearn.linear_model import LogisticRegression
        self.model = LogisticRegression(C=C, max_iter=max_iter, random_state=random_state)
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray):
        X_flat = _flatten_x(X)
        self.model.fit(X_flat, y)
        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted or len(X) == 0:
            return np.zeros(len(X), dtype=float)
        X_flat = _flatten_x(X)
        probs = self.model.predict_proba(X_flat)
        return probs[:, 1] if probs.ndim == 2 and probs.shape[1] > 1 else probs.ravel()


class RandomForestBaseline(BaseForecaster):
    def __init__(self, n_estimators: int = 100, random_state: int = 42):
        from sklearn.ensemble import RandomForestClassifier
        self.model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray):
        X_flat = _flatten_x(X)
        self.model.fit(X_flat, y)
        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted or len(X) == 0:
            return np.zeros(len(X), dtype=float)
        X_flat = _flatten_x(X)
        probs = self.model.predict_proba(X_flat)
        return probs[:, 1] if probs.ndim == 2 and probs.shape[1] > 1 else probs.ravel()

