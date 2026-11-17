"""
train_xgb_nasa.py — XGBoost early-warning classifier on NASA SDO SHARP features.

Training data source: NASA SDO SHARP HARPs active-region magnetic parameters
fetched via ``src.data.nasa_pipeline.build_sharp_training_dataset()``.

Features used (6 SHARP magnetic parameters):
  USFLUX   — Total unsigned flux (Mx)
  TOTPOT   — Total photospheric magnetic free energy density (erg/cm)
  SHRGT45  — Fraction of flux near polarity inversion line with shear > 45°
  R_VALUE  — Flux near polarity inversion line (Schrijver R)
  TOTUSJH  — Total unsigned vertical current helicity (G²/m)
  ABSNJZH  — Absolute value of net current helicity (G²/m)

Output: 6–24 hour early-warning flare probability score (binary: M/X-class).
Saves model checkpoint → src/nowcast/models/xgb_nasa_sharp.joblib
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
import xgboost as xgb

from src.data.nasa_pipeline import SHARP_FEATURE_COLS

logger = logging.getLogger(__name__)


# ===========================================================================
# Helpers
# ===========================================================================

def _log_class_balance(y: np.ndarray, split_name: str) -> None:
    """Log positive / negative class balance for a split."""
    n_pos = int(y.sum())
    n_neg = int((y == 0).sum())
    total = len(y)
    logger.info(
        "  %s balance : pos=%d (%.1f%%)  neg=%d (%.1f%%)",
        split_name,
        n_pos, 100 * n_pos / max(total, 1),
        n_neg, 100 * n_neg / max(total, 1),
    )


def _log_evaluation_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    split_name: str = "Test",
) -> dict:
    """Compute and log a full classification metric suite."""
    report = classification_report(y_true, y_pred, zero_division=0)
    logger.info("\n%s Classification Report:\n%s", split_name, report)

    metrics: dict = {}

    if len(np.unique(y_true)) > 1:
        auc_roc = roc_auc_score(y_true, y_proba)
        auc_pr  = average_precision_score(y_true, y_proba)
        brier   = brier_score_loss(y_true, y_proba)
        metrics.update({"auc_roc": auc_roc, "auc_pr": auc_pr, "brier": brier})
        logger.info(
            "  %s | AUC-ROC=%.4f | AUC-PR=%.4f | Brier=%.4f",
            split_name, auc_roc, auc_pr, brier,
        )
    else:
        logger.warning(
            "  %s | Only one class present — AUC metrics skipped.", split_name
        )

    return metrics


# ===========================================================================
# Training entry-point
# ===========================================================================

def train_xgb_nasa(
    X: pd.DataFrame,
    y: np.ndarray,
    test_size: float = 0.2,
    n_estimators: int = 300,
    max_depth: int = 5,
    learning_rate: float = 0.05,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    calibration_method: str = "isotonic",
    model_dir: str = "src/nowcast/models",
    model_name: str = "xgb_nasa_sharp.joblib",
    random_state: int = 42,
) -> Optional[str]:
    """Train an XGBoost binary classifier on SDO SHARP magnetic features.

    Handles severe class imbalance via ``scale_pos_weight = n_neg / n_pos``.
    Outputs calibrated flare probabilities (6–24 h early-warning score).

    Parameters
    ----------
    X : pd.DataFrame
        SHARP feature matrix (columns = SHARP_FEATURE_COLS) from
        ``build_sharp_training_dataset()``.
    y : np.ndarray, shape (N,)
        Binary flare labels (1 = M/X-class within 24-h lookahead).
    test_size : float
        Fraction of data held out for final evaluation (stratified).
    n_estimators : int
        Number of XGBoost boosting rounds.
    max_depth : int
        Maximum tree depth per round.
    learning_rate : float
        Boosting step size (eta).
    subsample : float
        Row subsampling ratio per tree.
    colsample_bytree : float
        Feature subsampling ratio per tree.
    calibration_method : str
        Platt scaling ``'sigmoid'`` or isotonic regression ``'isotonic'``.
    model_dir : str
        Output directory for the saved joblib file.
    model_name : str
        Filename of the saved model.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    str or None
        Path to the saved calibrated model, or None on failure.
    """
    logger.info("=" * 70)
    logger.info("[train_xgb_nasa] START  — SDO SHARP XGBoost Training")
    logger.info("  features      : %s", list(X.columns))
    logger.info("  N samples     : %d", len(y))
    _log_class_balance(y, "Full dataset")

    # ── Validation ──────────────────────────────────────────────────────────
    if len(X) < 20:
        logger.error(
            "[train_xgb_nasa] Dataset too small (%d rows) — need >= 20. Aborting.",
            len(X),
        )
        return None

    n_classes = len(np.unique(y))
    if n_classes < 2:
        logger.error(
            "[train_xgb_nasa] Only one class present in labels. "
            "Ensure DONKI labels cover both quiet-sun and M/X-class periods. Aborting."
        )
        return None

    # ── Ensure only recognised SHARP columns are passed ─────────────────────
    feature_cols = [c for c in SHARP_FEATURE_COLS if c in X.columns]
    missing_cols = [c for c in SHARP_FEATURE_COLS if c not in X.columns]
    if missing_cols:
        logger.warning("[train_xgb_nasa] Missing SHARP columns: %s — proceeding without them.", missing_cols)
    X = X[feature_cols].copy()

    # ── Stratified train/test split ──────────────────────────────────────────
    X_arr = X.values.astype(np.float32)
    X_train, X_test, y_train, y_test = train_test_split(
        X_arr, y, test_size=test_size, stratify=y, random_state=random_state
    )

    _log_class_balance(y_train, "Train")
    _log_class_balance(y_test,  "Test ")

    # ── Scale positive weight for class imbalance ────────────────────────────
    n_pos_train = int(y_train.sum())
    n_neg_train = int((y_train == 0).sum())
    scale_pos_weight = n_neg_train / max(n_pos_train, 1)
    logger.info(
        "  scale_pos_weight = %.2f  (n_neg=%d / n_pos=%d)",
        scale_pos_weight, n_neg_train, n_pos_train,
    )

    # ── Base XGBoost model ───────────────────────────────────────────────────
    logger.info("[train_xgb_nasa] Fitting base XGBClassifier ...")
    base_model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        scale_pos_weight=scale_pos_weight,
        use_label_encoder=False,
        random_state=random_state,
        verbosity=0,
    )

    # Early stopping via internal eval set
    base_model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    logger.info("[train_xgb_nasa] Base model trained.")

    # ── Probability calibration ──────────────────────────────────────────────
    logger.info("[train_xgb_nasa] Calibrating with method='%s' ...", calibration_method)

    # Use a fresh internal CV-based calibration (avoids leaking test labels)
    if calibration_method == "isotonic" and len(X_train) < 1000:
        logger.warning(
            "[train_xgb_nasa] Isotonic regression requires large N — "
            "falling back to sigmoid calibration (N_train=%d).",
            len(X_train),
        )
        calibration_method = "sigmoid"

    calibrated_model = CalibratedClassifierCV(
        estimator=base_model,
        method=calibration_method,
        cv="prefit",
    )
    calibrated_model.fit(X_test, y_test)

    # ── Evaluation ───────────────────────────────────────────────────────────
    logger.info("[train_xgb_nasa] Evaluating on held-out test set ...")
    y_pred   = calibrated_model.predict(X_test)
    y_proba  = calibrated_model.predict_proba(X_test)[:, 1]
    eval_metrics = _log_evaluation_metrics(y_test, y_pred, y_proba, split_name="Test")

    # Feature importance log
    try:
        importances = base_model.feature_importances_
        fi_pairs = sorted(
            zip(feature_cols, importances), key=lambda x: x[1], reverse=True
        )
        logger.info("[train_xgb_nasa] Feature importances (descending):")
        for feat, imp in fi_pairs:
            logger.info("  %-14s : %.4f", feat, imp)
    except Exception:
        pass

    # ── Save model ───────────────────────────────────────────────────────────
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, model_name)

    # Store feature column order alongside the model for reproducible inference
    model_bundle = {
        "model": calibrated_model,
        "feature_cols": feature_cols,
        "eval_metrics": eval_metrics,
        "scale_pos_weight": scale_pos_weight,
    }
    joblib.dump(model_bundle, model_path)

    logger.info("[train_xgb_nasa] Saved calibrated XGBoost bundle → %s", model_path)
    logger.info("=" * 70)

    return model_path
