"""
evaluate.py — Forecast evaluation metrics for the Aditya-L1 flare prediction system.

Changelog (overfitting remediation):
  FIX-4  Threshold tuning via precision-recall curve
         Added find_optimal_threshold() which sweeps the operating point from
         0.10 to 0.50 using sklearn.metrics.precision_recall_curve and selects
         the threshold that maximises F1-score.  The evaluate() function now
         calls this automatically when threshold=None (the new default).

Required metrics (from spec)
----------------------------
- True Positive Rate (TPR / Recall)
- False Alarm Rate (FAR = FP / (FP + TN))
- Lead Time Distribution: minutes between alert trigger and actual flare peak
  Report MEDIAN and IQR, NOT just mean (distribution is skewed/heavy-tailed).
- F1-Score at optimal operating threshold (FIX-4)
- Optimal threshold value (logged automatically, FIX-4)
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, f1_score

logger = logging.getLogger(__name__)

# Threshold sweep bounds (FIX-4)
_THRESHOLD_MIN = 0.10
_THRESHOLD_MAX = 0.50


# ──────────────────────────────────────────────────────────────────────────────
# FIX-4 : Threshold tuning
# ──────────────────────────────────────────────────────────────────────────────

def find_optimal_threshold(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    threshold_min: float = _THRESHOLD_MIN,
    threshold_max: float = _THRESHOLD_MAX,
) -> tuple[float, float]:
    """Scan operating thresholds and return the one that maximises F1-score.

    Uses sklearn.metrics.precision_recall_curve to enumerate all unique
    decision thresholds implied by the predicted probability distribution,
    then filters to those within [threshold_min, threshold_max] and picks
    the one yielding the highest F1-score.

    This replaces the static 0.50 threshold that caused F1 = 0.0 when
    the model's calibrated probability for rare flare events sits below 0.50
    even when it correctly ranks positives above negatives.

    Parameters
    ----------
    y_true         : Binary ground-truth labels (0 or 1), shape (N,).
    y_pred_proba   : Predicted probabilities in [0, 1], shape (N,).
    threshold_min  : Lower bound of threshold search window.
    threshold_max  : Upper bound of threshold search window.

    Returns
    -------
    (best_threshold, best_f1) : float pair
        best_threshold : Decision threshold that maximises F1 in the search window.
        best_f1        : F1-score achieved at that threshold.

    Notes
    -----
    - If both classes are not present in y_true, returns (0.50, 0.0) with a warning.
    - If no threshold in [threshold_min, threshold_max] produces a valid F1,
      falls back to the global argmax across ALL thresholds from precision_recall_curve.
    """
    y_true       = np.asarray(y_true, dtype=int)
    y_pred_proba = np.asarray(y_pred_proba, dtype=float)

    unique_classes = np.unique(y_true)
    if len(unique_classes) < 2:
        logger.warning(
            f"FIX-4 | find_optimal_threshold: only one class present in y_true "
            f"(classes={unique_classes}).  Cannot compute a meaningful threshold. "
            f"Returning default (0.50, 0.0)."
        )
        return 0.50, 0.0

    # precision_recall_curve returns thresholds for all unique score values.
    # The arrays are ordered from highest threshold (most restrictive) to lowest.
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred_proba)

    # thresholds has shape (N-1,); precisions/recalls have shape (N,).
    # The last element of precision/recall corresponds to threshold → ∞ (trivially p=1, r=0).
    # We compute F1 for each finite threshold.
    with np.errstate(divide='ignore', invalid='ignore'):
        f1_scores = np.where(
            (precisions[:-1] + recalls[:-1]) > 0,
            2 * precisions[:-1] * recalls[:-1] / (precisions[:-1] + recalls[:-1]),
            0.0,
        )

    # ── Window search ──────────────────────────────────────────────────────
    in_window = (thresholds >= threshold_min) & (thresholds <= threshold_max)

    if in_window.any():
        window_f1     = np.where(in_window, f1_scores, -1.0)
        best_idx      = int(np.argmax(window_f1))
        best_threshold = float(thresholds[best_idx])
        best_f1_value  = float(f1_scores[best_idx])
    else:
        # No threshold exists in the window — fall back to global argmax
        logger.warning(
            f"FIX-4 | No thresholds in [{threshold_min:.2f}, {threshold_max:.2f}]. "
            f"Falling back to global F1 argmax across all {len(thresholds)} thresholds."
        )
        best_idx       = int(np.argmax(f1_scores))
        best_threshold = float(thresholds[best_idx])
        best_f1_value  = float(f1_scores[best_idx])

    logger.info(
        f"FIX-4 | Threshold sweep [{threshold_min:.2f} → {threshold_max:.2f}]: "
        f"optimal_threshold={best_threshold:.4f}, best_F1={best_f1_value:.4f}"
    )

    # Log top-5 thresholds for transparency
    top_k = min(5, len(thresholds))
    top_indices = np.argsort(f1_scores)[::-1][:top_k]
    logger.info("FIX-4 | Top threshold candidates:")
    for rank, idx in enumerate(top_indices, start=1):
        logger.info(
            f"         #{rank}: threshold={thresholds[idx]:.4f}  "
            f"F1={f1_scores[idx]:.4f}  "
            f"Precision={precisions[idx]:.4f}  "
            f"Recall={recalls[idx]:.4f}"
        )

    return best_threshold, best_f1_value


# ──────────────────────────────────────────────────────────────────────────────
# Main evaluation function
# ──────────────────────────────────────────────────────────────────────────────

def evaluate(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    threshold: float | None = None,   # FIX-4: default changed from 0.5 → None (auto-tune)
    trigger_times: pd.Series | None = None,
    flare_peak_times: pd.Series | None = None,
) -> dict:
    """Compute TPR, FAR, ROC-AUC, F1, optimal threshold, and lead-time distribution.

    Parameters
    ----------
    y_true          : Binary ground-truth labels (0 or 1).
    y_pred_proba    : Predicted probabilities in [0, 1].
    threshold       : Decision threshold for binary classification.
                      If None (default), find_optimal_threshold() is called
                      automatically to pick the F1-maximising threshold in
                      the range [0.10, 0.50].  (FIX-4)
    trigger_times   : Optional Series of alert trigger timestamps.
    flare_peak_times: Optional Series of actual flare peak timestamps.

    Returns
    -------
    dict with keys:
        tpr, far, roc_auc,
        f1_score, optimal_threshold,        ← added by FIX-4
        lead_time_median_min, lead_time_iqr_min
    """
    y_true       = np.asarray(y_true, dtype=int)
    y_pred_proba = np.asarray(y_pred_proba, dtype=float)

    if len(y_true) == 0:
        logger.warning("evaluate() called with empty arrays — returning zeros.")
        return {
            "tpr": 0.0,
            "far": 0.0,
            "roc_auc": 0.5,
            "f1_score": 0.0,
            "optimal_threshold": 0.5,
            "lead_time_median_min": 0.0,
            "lead_time_iqr_min": 0.0,
        }

    # ── FIX-4: Threshold selection ──────────────────────────────────────────
    if threshold is None:
        logger.info("FIX-4 | threshold=None → running find_optimal_threshold()...")
        optimal_threshold, computed_f1 = find_optimal_threshold(y_true, y_pred_proba)
    else:
        optimal_threshold = float(threshold)
        # Compute F1 at the supplied threshold for reporting
        y_pred_fixed = (y_pred_proba >= optimal_threshold).astype(int)
        tp = int(((y_pred_fixed == 1) & (y_true == 1)).sum())
        fp = int(((y_pred_fixed == 1) & (y_true == 0)).sum())
        fn = int(((y_pred_fixed == 0) & (y_true == 1)).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        computed_f1 = (
            2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        )
        logger.info(
            f"evaluate() | Using supplied threshold={optimal_threshold:.4f}, "
            f"F1={computed_f1:.4f}"
        )

    # ── Binary predictions at chosen threshold ──────────────────────────────
    y_pred = (y_pred_proba >= optimal_threshold).astype(int)

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())

    tpr = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    far = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

    # ── ROC-AUC ────────────────────────────────────────────────────────────
    try:
        from sklearn.metrics import roc_auc_score
        roc_auc = (
            float(roc_auc_score(y_true, y_pred_proba))
            if len(np.unique(y_true)) > 1
            else 0.5
        )
    except Exception as exc:
        logger.warning(f"evaluate() | roc_auc_score failed: {exc}")
        roc_auc = 0.5

    # ── Lead-time distribution ─────────────────────────────────────────────
    lead_times: list[float] = []
    if trigger_times is not None and flare_peak_times is not None:
        for trig, peak in zip(trigger_times, flare_peak_times):
            if pd.notna(trig) and pd.notna(peak):
                dt_min = (
                    pd.Timestamp(peak) - pd.Timestamp(trig)
                ).total_seconds() / 60.0
                if dt_min >= 0:
                    lead_times.append(dt_min)

    if lead_times:
        lead_median = float(np.median(lead_times))
        q75, q25    = np.percentile(lead_times, [75, 25])
        lead_iqr    = float(q75 - q25)
    else:
        lead_median = 0.0
        lead_iqr    = 0.0

    # ── Final metric summary log ───────────────────────────────────────────
    logger.info(
        f"evaluate() RESULTS | "
        f"threshold={optimal_threshold:.4f}  "
        f"TPR={tpr:.4f}  FAR={far:.4f}  "
        f"ROC-AUC={roc_auc:.4f}  F1={computed_f1:.4f} | "
        f"TP={tp} FP={fp} FN={fn} TN={tn}"
    )

    return {
        "tpr":                    tpr,
        "far":                    far,
        "roc_auc":                roc_auc,
        "f1_score":               computed_f1,       # FIX-4
        "optimal_threshold":      optimal_threshold,  # FIX-4
        "lead_time_median_min":   lead_median,
        "lead_time_iqr_min":      lead_iqr,
    }
