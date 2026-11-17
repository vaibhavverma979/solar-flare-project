"""
test_pipeline.py — Out-of-Sample Evaluation Pipeline for Aditya-L1 Solar Flare Models
========================================================================================
Senior Space ML Engineer evaluation harness.

Evaluates two trained models on completely unseen FITS data from data/test_fits/:
  1. PatchTST Transformer  — 30-minute solar flare probability forecast + SXR regression
  2. XGBoost Classifier    — Real-time nowcast confidence across 4 flare classes

Usage:
    python test_pipeline.py [--data_dir data/test_fits] [--threshold 0.5] [--device cpu]
    python test_pipeline.py --inject_flare          # overlay synthetic M-class spike on quiet data

Outputs:
    - Classification reports (precision / recall / F1 per class)
    - MSE for peak SXR regression
    - ROC-AUC score
    - Confusion matrix
    - Threshold sweep for optimal F1
"""

import os
import sys
import argparse
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("test_pipeline")


# ─────────────────────────────────────────────────────────────────────────────
# BANNER
# ─────────────────────────────────────────────────────────────────────────────
BANNER = """
================================================================================
  ISRO Aditya-L1 Flare Prediction System -- Out-of-Sample Evaluation Pipeline
================================================================================
  Models:  PatchTST Transformer (30-min forecast)
           XGBoost Classifier   (nowcast confidence)
  Data:    data/test_fits/  [unseen, held-out FITS files]
================================================================================
"""


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1  — DATA INGESTION
# ─────────────────────────────────────────────────────────────────────────────
def load_and_engineer_test_data(data_dir: str) -> pd.DataFrame:
    """
    Parse test FITS files and run the full feature engineering pipeline
    WITHOUT fitting the scaler (prevents data leakage).

    Returns:
        df_features (pd.DataFrame): Fully engineered, scaler-transformed DataFrame.
    """
    _section("1. DATA INGESTION & FEATURE ENGINEERING")

    try:
        from src.data.isro_fits_parser import ISROFITSParser
    except ImportError as e:
        logger.error("Cannot import ISROFITSParser: %s", e)
        sys.exit(1)

    try:
        from src.data.flare_features import AstrophysicalFeatureEngineer
    except ImportError as e:
        logger.error("Cannot import AstrophysicalFeatureEngineer: %s", e)
        sys.exit(1)

    # ── Validate test data directory ──────────────────────────────────────────
    test_dir = Path(data_dir)
    if not test_dir.exists():
        logger.warning(
            "Test data directory '%s' does not exist. Creating it with a synthetic "
            "fallback dataset for demonstration purposes.", data_dir
        )
        test_dir.mkdir(parents=True, exist_ok=True)
        df_raw = _generate_synthetic_fallback()
    else:
        fits_files = (
            list(test_dir.glob("*.fits"))
            + list(test_dir.glob("*.lc"))
            + list(test_dir.glob("*.lc.gz"))
            + list(test_dir.glob("*.fits.gz"))
        )
        if not fits_files:
            logger.warning(
                "No FITS/lightcurve files found in '%s'. "
                "Using a synthetic fallback dataset for demonstration.", data_dir
            )
            df_raw = _generate_synthetic_fallback()
        else:
            logger.info("Found %d file(s) in '%s'.", len(fits_files), data_dir)
            # ── Parse with ISROFITSParser (same ERFA fallback as training) ────
            # Resampling at '1min' to match the training cadence
            parser = ISROFITSParser(data_dir=str(test_dir), resample_freq="1min")
            df_raw = parser.process()

            if df_raw.empty:
                logger.warning(
                    "Parser returned an empty DataFrame (possibly no valid time columns). "
                    "Switching to synthetic fallback."
                )
                df_raw = _generate_synthetic_fallback()

    logger.info("Raw test rows parsed: %d", len(df_raw))
    _log_dataframe_summary(df_raw, "Raw Test Data")

    # ── Feature Engineering (fit_scaler=False -> uses saved scaler) ──────────
    # CRITICAL: We must call engineer.process(fit_scaler=False) to call
    # scaler.transform() only, never fit_transform(), preventing data leakage.
    logger.info("Running AstrophysicalFeatureEngineer with fit_scaler=False ...")
    engineer = AstrophysicalFeatureEngineer(df_raw, model_dir="src/forecast/models")
    df_features = engineer.process(fit_scaler=False)

    logger.info("Engineered test rows: %d  |  Columns: %s", len(df_features), list(df_features.columns))
    return df_features


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2  — SCALER & MODEL LOADING
# ─────────────────────────────────────────────────────────────────────────────
def load_models(device: str):
    """
    Load all trained artefacts:
      - fits_scaler.joblib       (already applied in section 1, loaded here for audit)
      - patchtst_fits_trained.pt (PyTorch state dict)
      - xgb_nowcast_fits.joblib  (XGBoost / calibrated classifier)

    Returns:
        patchtst_model, xgb_model, device_obj
    """
    _section("2. LOADING TRAINED MODEL ARTEFACTS")

    import joblib
    import torch
    from src.forecast.train_fits_model import PatchTST

    SCALER_PATH   = "src/forecast/models/fits_scaler.joblib"
    PATCHTST_PATH = "src/forecast/models/patchtst_fits_trained.pt"
    XGB_PATH      = "src/nowcast/models/xgb_nowcast_fits.joblib"

    # ── Scaler (audit load only — transform already applied via FeatureEngineer)
    _check_file(SCALER_PATH)
    scaler = joblib.load(SCALER_PATH)
    logger.info("[Scaler]   Loaded RobustScaler from '%s'", SCALER_PATH)
    logger.info("           Center_: %s", np.round(scaler.center_[:4], 4))
    logger.info("           Scale_:  %s", np.round(scaler.scale_[:4], 4))
    logger.info("           NOTE:  transform() was applied in Section 1 — no leakage.")

    # ── PatchTST Model ────────────────────────────────────────────────────────
    _check_file(PATCHTST_PATH)
    device_obj = torch.device(device)

    # Infer architecture from the saved state dict to avoid hardcoding mismatch
    state_dict = torch.load(PATCHTST_PATH, map_location=device_obj, weights_only=True)

    # ── Backwards-compatibility remap ─────────────────────────────────────────
    # Older checkpoints saved heads as bare nn.Linear (keys: "flare_head.weight").
    # The current PatchTST wraps them in nn.Sequential([Dropout, Linear]), so
    # PyTorch expects  "flare_head.1.weight".  We remap on-the-fly so we can
    # still load old checkpoints without retraining.
    HEAD_NAMES = ("flare_head", "sxr_head", "hxr_head")
    remapped = {}
    needs_remap = any(
        f"{h}.weight" in state_dict for h in HEAD_NAMES
    )
    if needs_remap:
        logger.info(
            "[PatchTST] Detected old-style flat head keys — remapping to "
            "Sequential (.1.*) format for backward compatibility."
        )
        for key, val in state_dict.items():
            new_key = key
            for head in HEAD_NAMES:
                if key == f"{head}.weight":
                    new_key = f"{head}.1.weight"
                elif key == f"{head}.bias":
                    new_key = f"{head}.1.bias"
            remapped[new_key] = val
        state_dict = remapped

    # value_embedding.weight shape: [d_model, patch_size]
    # flare_head.1.weight shape:    [1, head_dim]  where head_dim = patch_num * d_model * c_in
    head_weight_key = "flare_head.1.weight" if needs_remap else "flare_head.weight"
    head_dim   = state_dict[head_weight_key].shape[1]
    d_model    = state_dict["value_embedding.weight"].shape[0]
    patch_size = state_dict["value_embedding.weight"].shape[1]
    seq_len    = 360
    stride     = 8
    patch_num  = int((seq_len - patch_size) / stride + 1)
    c_in       = head_dim // (patch_num * d_model)

    logger.info("[PatchTST] Architecture inferred from state dict:")
    logger.info("           c_in=%d | seq_len=%d | patch_size=%d | stride=%d",
                c_in, seq_len, patch_size, stride)
    logger.info("           d_model=%d | patch_num=%d | head_dim=%d",
                d_model, patch_num, head_dim)

    patchtst_model = PatchTST(
        c_in=c_in, seq_len=seq_len,
        patch_size=patch_size, stride=stride,
        d_model=d_model, n_heads=4, e_layers=3,
    ).to(device_obj)

    patchtst_model.load_state_dict(state_dict)
    patchtst_model.eval()   # Critical: inference mode (disables dropout / batch-norm)
    logger.info("[PatchTST] Loaded weights from '%s' — model is in EVAL mode.", PATCHTST_PATH)

    # ── XGBoost Model ─────────────────────────────────────────────────────────
    _check_file(XGB_PATH)
    xgb_model = joblib.load(XGB_PATH)
    logger.info("[XGBoost]  Loaded classifier from '%s'", XGB_PATH)
    logger.info("           Type: %s", type(xgb_model).__name__)

    return patchtst_model, xgb_model, device_obj


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3  — PATCHTST EVALUATION  (30-Min Forecast)
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_patchtst(
    df: pd.DataFrame,
    model,
    device,
    threshold: float = 0.5,
    batch_size: int = 64,
) -> dict:
    """
    Evaluate the PatchTST multi-task Transformer on the test set.

    Steps:
      - Build FlareSlidingWindowDataset (context=360, horizon=30)
      - Run DataLoader (batch_size=64, shuffle=False) for consistent ordering
      - Collect sigmoid(flare_logits) -> probabilities
      - Binary threshold at `threshold` -> predictions
      - Print classification_report vs ground truth
      - Calculate MSE for peak SXR regression head

    Returns:
        dict of evaluation metrics
    """
    _section("3. PATCHTST TRANSFORMER EVALUATION  (30-Min Flare Forecast)")

    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from sklearn.metrics import (
        classification_report, roc_auc_score,
        confusion_matrix, f1_score,
        precision_score, recall_score,
    )
    from src.forecast.dataset import FlareSlidingWindowDataset

    # ── Build Dataset ─────────────────────────────────────────────────────────
    context_length   = 360    # 6-hour window at 1-min cadence
    forecast_horizon = 30     # 30-minute forecast horizon
    m_class_threshold = 5.0   # Scaled threshold used during training

    dataset = FlareSlidingWindowDataset(
        df,
        context_length=context_length,
        forecast_horizon=forecast_horizon,
        m_class_threshold=m_class_threshold,
    )

    if len(dataset) == 0:
        logger.error(
            "Dataset is empty — need at least %d rows for context+horizon. "
            "Test data is too short to evaluate PatchTST.",
            context_length + forecast_horizon,
        )
        return {}

    logger.info("Test dataset windows: %d  (context=%d, horizon=%d)",
                len(dataset), context_length, forecast_horizon)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # ── Inference Loop (inside torch.no_grad()) ───────────────────────────────
    all_proba    = []
    all_preds    = []
    all_labels   = []
    all_sxr_pred = []
    all_sxr_true = []
    sigmoid      = nn.Sigmoid()

    logger.info("Running inference on %d batches (batch_size=%d) ...", len(loader), batch_size)

    model.eval()
    with torch.no_grad():
        for batch_idx, (batch_x, batch_y) in enumerate(loader):
            batch_x       = batch_x.to(device)
            flare_flag    = batch_y["flare_flag"].cpu()   # [B, 1]
            peak_sxr_true = batch_y["peak_sxr"].cpu()    # [B, 1]

            outputs = model(batch_x)

            # Sigmoid activation converts raw logits to probabilities [0, 1]
            proba    = sigmoid(outputs["flare_logits"]).cpu().numpy().flatten()
            preds    = (proba >= threshold).astype(int)
            sxr_pred = outputs["sxr_pred"].cpu().numpy().flatten()

            all_proba.extend(proba.tolist())
            all_preds.extend(preds.tolist())
            all_labels.extend(flare_flag.numpy().flatten().astype(int).tolist())
            all_sxr_pred.extend(sxr_pred.tolist())
            all_sxr_true.extend(peak_sxr_true.numpy().flatten().tolist())

            if (batch_idx + 1) % 10 == 0:
                logger.info("  Batch %d/%d processed.", batch_idx + 1, len(loader))

    y_true   = np.array(all_labels)
    y_proba  = np.array(all_proba)
    y_pred   = np.array(all_preds)
    sxr_true = np.array(all_sxr_true)   # already on the normalised/log-scaled axis
    sxr_pred = np.array(all_sxr_pred)   # sxr_head output — same normalised scale

    # ── Binary Classification Report ─────────────────────────────────────────
    print()
    print("=" * 70)
    print("  PatchTST  --  Flare / No-Flare Binary Classification Report")
    print("  Decision Threshold : %s  |  Test Windows : %d" % (threshold, len(y_true)))
    print("=" * 70)
    print()

    unique_labels = np.unique(y_true)

    # ── Zero-flare guard ─────────────────────────────────────────────────────
    # When the held-out set covers only quiet-sun periods, y_true.sum() == 0.
    # Recall and F1 for the positive class are undefined (0/0). We log a clear
    # explanation so the operator is not misled by zeroed metrics.
    if int(y_true.sum()) == 0:
        logger.info(
            "Test dataset contains only background quiet sun data "
            "(no M/X flare events present to evaluate Recall/F1)."
        )
        logger.info(
            "Tip: use --inject_flare to overlay a synthetic M-class spike and "
            "verify the model's eruption-detection capability."
        )

    if len(unique_labels) < 2:
        logger.warning(
            "Only class '%s' present in ground-truth labels. "
            "Classification report will be degenerate. "
            "Increase test set size or lower m_class_threshold.",
            unique_labels,
        )

    # Pass labels=[0, 1] explicitly so the report always shows both rows,
    # even when only one class appears in y_true or y_pred (avoids ValueError).
    target_names = ["No Flare (< M-Class)", "Flare (>= M-Class)"]
    print(classification_report(
        y_true, y_pred,
        labels=[0, 1],
        target_names=target_names,
        zero_division=0,
        digits=4,
    ))

    # ── ROC-AUC ───────────────────────────────────────────────────────────────
    if len(unique_labels) > 1:
        roc_auc = roc_auc_score(y_true, y_proba)
        print("  ROC-AUC Score         : %.4f" % roc_auc)
    else:
        roc_auc = float("nan")
        print("  ROC-AUC Score         : N/A  (only one class present)")

    # ── Confusion Matrix ──────────────────────────────────────────────────────
    cm = confusion_matrix(y_true, y_pred)
    print()
    print("  Confusion Matrix (rows=Actual, cols=Predicted):")
    print("                No-Flare    Flare")
    row_labels = ["No-Flare  ", "Flare     "]
    for i, row_label in enumerate(row_labels):
        if i < len(cm):
            print("  %s  %8d  %8d" % (row_label, cm[i][0], cm[i][1] if len(cm[i]) > 1 else 0))

    # ── Threshold Sweep ───────────────────────────────────────────────────────
    best_thresh = threshold
    best_f1     = float("nan")
    if len(unique_labels) > 1:
        print()
        print("  Threshold Sweep (step=0.05)  --  Flare-class F1:")
        print("  %10s  %10s  %10s  %10s" % ("Threshold", "F1-Score", "Precision", "Recall"))
        best_f1 = 0.0
        for t in np.arange(0.10, 0.95, 0.05):
            preds_t = (y_proba >= t).astype(int)
            f1_t    = f1_score(y_true, preds_t, pos_label=1, zero_division=0)
            prec_t  = precision_score(y_true, preds_t, pos_label=1, zero_division=0)
            rec_t   = recall_score(y_true, preds_t, pos_label=1, zero_division=0)
            marker  = "  <-- default" if abs(t - threshold) < 0.01 else ""
            print("  %10.2f  %10.4f  %10.4f  %10.4f%s" % (t, f1_t, prec_t, rec_t, marker))
            if f1_t > best_f1:
                best_f1     = f1_t
                best_thresh = round(float(t), 2)
        print("\n  Best threshold for Flare-class F1: %.2f  (F1 = %.4f)" % (best_thresh, best_f1))

    # ── SXR Regression Report  (normalised / log-scaled axis) ─────────────────
    # sxr_pred and sxr_true both live on the *normalised* scale produced by
    # AstrophysicalFeatureEngineer (RobustScaler / log1p transform).  Computing
    # MSE/MAE here avoids the extreme dynamic range of raw W m⁻² flux values
    # (e.g., 1e-7 – 1e-4) which would make the error figures unreadably small.
    mse_sxr = float(np.mean((sxr_pred - sxr_true) ** 2))
    mae_sxr = float(np.mean(np.abs(sxr_pred - sxr_true)))

    print()
    print("=" * 70)
    print("  PatchTST  --  Peak SXR Regression  [normalised / log-scaled units]")
    print("=" * 70)
    print("  Mean Squared Error  (MSE) [normalised] : %.6f" % mse_sxr)
    print("  Mean Absolute Error (MAE) [normalised] : %.6f" % mae_sxr)
    print("  NOTE: metrics are on the RobustScaler/log1p axis, NOT raw W m⁻².")
    print("        Lower values indicate better regression fit on the trained scale.")
    print("  True  SXR range [norm]                 : [%.4f,  %.4f]" % (sxr_true.min(), sxr_true.max()))
    print("  Pred  SXR range [norm]                 : [%.4f,  %.4f]" % (sxr_pred.min(), sxr_pred.max()))
    print()

    return {
        "patchtst_roc_auc":      roc_auc,
        "patchtst_mse_sxr":      mse_sxr,
        "patchtst_mae_sxr":      mae_sxr,
        "patchtst_best_f1":      best_f1,
        "patchtst_best_thresh":  best_thresh,
        "patchtst_n_windows":    int(len(y_true)),
        "patchtst_flare_rate":   float(y_true.mean()),
    }


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4  — XGBOOST EVALUATION  (Nowcast Confidence)
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_xgboost(df: pd.DataFrame, xgb_model) -> dict:
    """
    Evaluate the XGBoost nowcast classifier on the test set.

    Extracts the same 5-feature rolling windows (window_size=60) used in training:
        peak_sxr, peak_hxr, rise_time, decay_slope, neupert_score

    Class mapping (must match train_xgb.py exactly):
        0 = Quiet / B-Class   (peak_sxr < 1.0)
        1 = C-Class           (1.0 <= peak_sxr < 3.0)
        2 = M-Class           (3.0 <= peak_sxr < 5.0)
        3 = X-Class           (peak_sxr >= 5.0)
    """
    _section("4. XGBOOST NOWCAST CLASSIFIER EVALUATION  (Confidence Scores)")

    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    from src.nowcast.train_xgb import extract_nowcast_features

    window_size = 60  # Must match training window (60 steps at 1-min cadence)

    # ── Extract Features ──────────────────────────────────────────────────────
    logger.info("Extracting nowcast rolling-window features (window_size=%d) ...", window_size)

    try:
        X_test, y_test = extract_nowcast_features(df, window_size=window_size)
    except Exception as e:
        logger.error("Feature extraction failed: %s", e)
        return {}

    if len(X_test) == 0:
        logger.error("No feature windows extracted. Test set too small (need > %d rows).", window_size)
        return {}

    logger.info("Extracted %d feature windows  |  Feature shape: %s", len(X_test), X_test.shape)
    class_map = {0: "Quiet/B", 1: "C-Class", 2: "M-Class", 3: "X-Class"}
    _log_class_distribution(y_test, class_map)

    # ── Inference ─────────────────────────────────────────────────────────────
    logger.info("Running XGBoost inference ...")
    y_pred = xgb_model.predict(X_test)

    try:
        y_proba = xgb_model.predict_proba(X_test)  # [N, 4]
        max_confidence = y_proba.max(axis=1)
        logger.info("Mean prediction confidence : %.3f  |  Min: %.3f  |  Max: %.3f",
                    max_confidence.mean(), max_confidence.min(), max_confidence.max())
    except AttributeError:
        y_proba = None
        logger.info("predict_proba not available on this classifier variant.")

    # ── Classification Report ─────────────────────────────────────────────────
    class_names     = ["Quiet/B-Class", "C-Class", "M-Class", "X-Class"]
    present_classes = sorted(np.unique(np.concatenate([y_test, y_pred])).astype(int).tolist())
    target_names_present = [class_names[i] for i in present_classes if i < len(class_names)]

    print()
    print("=" * 70)
    print("  XGBoost  --  Nowcast Flare Class Confidence Report")
    print("  Windows evaluated : %d  |  Window size : %d steps" % (len(y_test), window_size))
    print("=" * 70)
    print()
    print(classification_report(
        y_test, y_pred,
        labels=present_classes,
        target_names=target_names_present,
        zero_division=0,
        digits=4,
    ))

    # ── Confusion Matrix ──────────────────────────────────────────────────────
    cm = confusion_matrix(y_test, y_pred, labels=present_classes)
    print("  Confusion Matrix (rows=Actual, cols=Predicted):")
    header_cols = "".join("  %11s" % class_names[c] for c in present_classes)
    print("  " + " " * 13 + header_cols)
    for i, true_class in enumerate(present_classes):
        row_str = "  ".join("%10d" % v for v in cm[i])
        print("  %-13s  %s" % (class_names[true_class], row_str))

    # ── Per-class Confidence Analysis ─────────────────────────────────────────
    if y_proba is not None:
        print()
        print("  Per-Class Confidence Statistics (from predict_proba):")
        print("  %-14s  %10s  %10s  %10s" % ("Class", "Mean Conf", "Std Conf", "Max Conf"))
        for i, cname in enumerate(class_names):
            mask = (y_test == i)
            if mask.sum() > 0:
                conf = y_proba[mask, i]
                print("  %-14s  %10.4f  %10.4f  %10.4f" % (cname, conf.mean(), conf.std(), conf.max()))
            else:
                print("  %-14s  %10s  %10s  %10s" % (cname, "N/A", "N/A", "N/A"))

    print()

    # ── Feature Importances ───────────────────────────────────────────────────
    _print_xgb_feature_importance(xgb_model, list(X_test.columns))

    acc = accuracy_score(y_test, y_pred)
    return {
        "xgb_accuracy":     float(acc),
        "xgb_n_windows":    int(len(y_test)),
        "xgb_classes_seen": present_classes,
    }


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _section(title: str):
    width = 78
    print()
    print("=" * width)
    print("  " + title)
    print("=" * width)
    logger.info(title)


def _check_file(path: str):
    if not os.path.exists(path):
        logger.error(
            "Required file not found: '%s'\n"
            "  Run 'python train.py' first to generate model artefacts.", path
        )
        sys.exit(1)
    size_mb = os.path.getsize(path) / 1_048_576
    logger.info("  Found: '%s'  (%.2f MB)", path, size_mb)


def _log_dataframe_summary(df: pd.DataFrame, label: str):
    logger.info("[%s] Shape: %s", label, df.shape)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        logger.info("[%s] Numeric columns: %s", label, numeric_cols)


def _log_class_distribution(y: np.ndarray, class_map: dict):
    unique, counts = np.unique(y, return_counts=True)
    logger.info("Class distribution in test set:")
    for cls, cnt in zip(unique, counts):
        label = class_map.get(int(cls), str(cls))
        pct   = 100.0 * cnt / len(y)
        logger.info("  Class %d (%s): %d samples  (%.1f%%)", cls, label, cnt, pct)


def _print_xgb_feature_importance(model, feature_names: list):
    """Print feature importances; handles CalibratedClassifierCV wrapper."""
    try:
        base_model  = getattr(model, "estimator", model)
        importances = base_model.feature_importances_
        print("  XGBoost Feature Importances (gain):")
        sorted_idx = np.argsort(importances)[::-1]
        for rank, idx in enumerate(sorted_idx):
            fname = feature_names[idx] if idx < len(feature_names) else ("feature_%d" % idx)
            print("    %d. %-20s  %.4f" % (rank + 1, fname, importances[idx]))
        print()
    except AttributeError:
        logger.debug("Feature importances not accessible from the calibrated wrapper.")


def _generate_synthetic_fallback(n_minutes: int = 1440) -> pd.DataFrame:
    """
    Generates a synthetic 24-hour Aditya-L1 telemetry DataFrame when no real
    FITS files are available. Mimics ISROFITSParser output schema exactly.

    Includes:
      - Quiet solar background (hours 0-7 and 10-17)
      - Simulated C-class flare event  at t=480 min (duration ~40 min)
      - Simulated M-class flare event  at t=1080 min (duration ~30 min)
    """
    logger.info("Generating synthetic fallback dataset (%d minutes) ...", n_minutes)
    rng  = np.random.default_rng(seed=42)
    time = pd.date_range("2026-01-01", periods=n_minutes, freq="1min")

    sxr_bg = 0.5 + rng.normal(0, 0.05, n_minutes)
    hxr_bg = 0.3 + rng.normal(0, 0.03, n_minutes)

    # C-class flare
    flare_c = np.zeros(n_minutes)
    c_start, c_peak, c_end = 480, 500, 520
    flare_c[c_start:c_peak] = np.linspace(0, 2.5, c_peak - c_start)
    flare_c[c_peak:c_end]   = np.linspace(2.5, 0, c_end - c_peak)

    # M-class flare
    flare_m = np.zeros(n_minutes)
    m_start, m_peak, m_end = 1080, 1095, 1110
    flare_m[m_start:m_peak] = np.linspace(0, 6.0, m_peak - m_start)
    flare_m[m_peak:m_end]   = np.linspace(6.0, 0, m_end - m_peak)

    sxr_total = np.clip(sxr_bg + flare_c + flare_m, 0, None)
    hxr_total = np.clip(hxr_bg + (flare_c + flare_m) * 0.5, 0, None)
    sdd1 = sxr_total * 0.4 + rng.normal(0, 0.01, n_minutes)
    sdd2 = sxr_total * 0.6 + rng.normal(0, 0.01, n_minutes)
    cdte = hxr_total * 0.55 + rng.normal(0, 0.01, n_minutes)
    czt  = hxr_total * 0.45 + rng.normal(0, 0.01, n_minutes)

    df = pd.DataFrame({
        "timestamp":        time,
        "solexs_sdd1":      sdd1,
        "solexs_sdd2":      sdd2,
        "solexs_total_sxr": sxr_total,
        "hel1os_cdte":      cdte,
        "hel1os_czt":       czt,
        "hel1os_total_hxr": hxr_total,
    })
    logger.info(
        "Synthetic data: %d rows | SXR [%.2f, %.2f] | HXR [%.2f, %.2f]",
        len(df), df["solexs_total_sxr"].min(), df["solexs_total_sxr"].max(),
        df["hel1os_total_hxr"].min(), df["hel1os_total_hxr"].max(),
    )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# SYNTHETIC FLARE INJECTION  (--inject_flare mode)
# ─────────────────────────────────────────────────────────────────────────────
def inject_flare_into_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Overlay a simulated M-class flare spike onto an existing (quiet) engineered
    DataFrame so that the pipeline can verify the model flags the eruption even
    when the real test set contains no flare events.

    Strategy
    --------
    The injection targets the *scaled* feature columns produced by
    AstrophysicalFeatureEngineer so that values live on the same normalised
    scale the model was trained on.

    Columns mutated (if present):
        solexs_total_sxr  — primary SXR channel
        solexs_sdd1       — SDD-1 sub-detector
        solexs_sdd2       — SDD-2 sub-detector
        hel1os_total_hxr  — HXR channel (correlated rise)
        peak_sxr          — pre-computed feature (if engineered)
        peak_hxr          — pre-computed feature (if engineered)
        neupert_score     — derivative-based activity index (if engineered)

    The spike is placed at 60 % of the time series length (well inside the
    sliding-window range) and lasts 30 minutes:
        rise  : 15 min linear ramp  0 -> M_AMPLITUDE
        decay : 15 min linear decay M_AMPLITUDE -> 0

    The amplitude is chosen so that the *normalised* SXR value exceeds the
    m_class_threshold used in FlareSlidingWindowDataset (default=5.0).

    Returns
    -------
    pd.DataFrame  — copy of ``df`` with the injected spike applied.
    """
    _section("SYNTHETIC FLARE INJECTION  (--inject_flare mode)")

    df_inj = df.copy()
    n = len(df_inj)

    # Spike parameters (normalised / scaled units)
    M_AMPLITUDE   = 8.0    # clearly above m_class_threshold=5.0 on the scaled axis
    HXR_AMPLITUDE = 4.0    # correlated HXR rise (~50 % of SXR amplitude)
    RISE_MINUTES  = 15
    DECAY_MINUTES = 15
    SPIKE_DURATION = RISE_MINUTES + DECAY_MINUTES

    # Place spike at 60 % into the series (guarantees a full context window before it)
    spike_start = int(0.60 * n)
    spike_peak  = spike_start + RISE_MINUTES
    spike_end   = spike_start + SPIKE_DURATION

    if spike_end >= n:
        spike_start = max(0, n - SPIKE_DURATION - 1)
        spike_peak  = spike_start + RISE_MINUTES
        spike_end   = spike_start + SPIKE_DURATION

    sxr_bump = np.zeros(n)
    sxr_bump[spike_start:spike_peak] = np.linspace(0, M_AMPLITUDE,   RISE_MINUTES)
    sxr_bump[spike_peak:spike_end]   = np.linspace(M_AMPLITUDE, 0,   DECAY_MINUTES)

    hxr_bump = np.zeros(n)
    hxr_bump[spike_start:spike_peak] = np.linspace(0, HXR_AMPLITUDE, RISE_MINUTES)
    hxr_bump[spike_peak:spike_end]   = np.linspace(HXR_AMPLITUDE, 0, DECAY_MINUTES)

    # Apply the bump to whichever raw flux columns are present
    for sxr_col in ("solexs_total_sxr", "solexs_sdd1", "solexs_sdd2"):
        if sxr_col in df_inj.columns:
            df_inj[sxr_col] = df_inj[sxr_col].values + sxr_bump
            logger.info("  Injected SXR bump into column '%s'", sxr_col)

    for hxr_col in ("hel1os_total_hxr", "hel1os_cdte", "hel1os_czt"):
        if hxr_col in df_inj.columns:
            df_inj[hxr_col] = df_inj[hxr_col].values + hxr_bump
            logger.info("  Injected HXR bump into column '%s'", hxr_col)

    # Also elevate pre-computed engineered features if already present
    for feat_col in ("peak_sxr", "peak_hxr"):
        if feat_col in df_inj.columns:
            df_inj.loc[df_inj.index[spike_start:spike_end], feat_col] += (
                sxr_bump[spike_start:spike_end]
            )
            logger.info("  Elevated engineered feature '%s' during spike window.", feat_col)

    if "neupert_score" in df_inj.columns:
        # Neupert score = integral of HXR derivative; approximate by adding a
        # positive delta proportional to the HXR ramp-up phase
        neupert_bump = np.zeros(n)
        neupert_bump[spike_start:spike_peak] = np.linspace(0, HXR_AMPLITUDE * 0.6, RISE_MINUTES)
        df_inj["neupert_score"] = df_inj["neupert_score"].values + neupert_bump
        logger.info("  Elevated engineered feature 'neupert_score' during spike window.")

    logger.info(
        "Synthetic M-class flare injected: rows %d–%d  "
        "| SXR amplitude=%.1f  | HXR amplitude=%.1f",
        spike_start, spike_end - 1, M_AMPLITUDE, HXR_AMPLITUDE,
    )
    logger.info(
        "Injection summary — SXR max after injection: %.4f  (threshold=5.0)",
        df_inj[[c for c in ("solexs_total_sxr", "solexs_sdd1") if c in df_inj.columns]]
        .max().max(),
    )
    return df_inj


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5  — FINAL SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
def print_summary(patchtst_metrics: dict, xgb_metrics: dict):
    _section("5. EVALUATION SUMMARY  (Aditya-L1 Dashboard Metrics)")
    print()
    print("  +----------------------------------------------+-----------+")
    print("  | Metric                                       |     Value |")
    print("  +----------------------------------------------+-----------+")

    def _row(label, value):
        if isinstance(value, float):
            val_str = ("%.4f" % value) if not np.isnan(value) else "       N/A"
        elif value is None:
            val_str = "       N/A"
        else:
            val_str = str(value)
        print("  | %-44s | %9s |" % (label, val_str))

    _row("PatchTST  Test Windows",            patchtst_metrics.get("patchtst_n_windows", "N/A"))
    _row("PatchTST  Flare Rate in Test Set",  patchtst_metrics.get("patchtst_flare_rate", float("nan")))
    _row("PatchTST  ROC-AUC Score",           patchtst_metrics.get("patchtst_roc_auc",   float("nan")))
    _row("PatchTST  Best Flare-Class F1",     patchtst_metrics.get("patchtst_best_f1",   float("nan")))
    _row("PatchTST  Best Decision Threshold", patchtst_metrics.get("patchtst_best_thresh", float("nan")))
    _row("PatchTST  Peak SXR MSE",            patchtst_metrics.get("patchtst_mse_sxr",   float("nan")))
    _row("PatchTST  Peak SXR MAE",            patchtst_metrics.get("patchtst_mae_sxr",   float("nan")))
    print("  +----------------------------------------------+-----------+")
    _row("XGBoost   Test Windows",            xgb_metrics.get("xgb_n_windows", "N/A"))
    _row("XGBoost   Overall Accuracy",        xgb_metrics.get("xgb_accuracy",  float("nan")))
    print("  +----------------------------------------------+-----------+")

    # Actionable recommendations
    print()
    print("  Recommendations:")
    auc = patchtst_metrics.get("patchtst_roc_auc", float("nan"))
    acc = xgb_metrics.get("xgb_accuracy", float("nan"))
    if not np.isnan(auc):
        if auc >= 0.90:
            print("    [PatchTST] EXCELLENT AUC (>= 0.90) -- model generalises well to unseen data.")
        elif auc >= 0.75:
            print("    [PatchTST] GOOD AUC (0.75 - 0.90) -- consider more training data or augmentation.")
        else:
            print("    [PatchTST] LOW AUC (< 0.75) -- review feature engineering or extend training.")
    if not np.isnan(acc):
        if acc >= 0.85:
            print("    [XGBoost]  HIGH accuracy (>= 0.85) -- nowcast model is production-ready.")
        elif acc >= 0.70:
            print("    [XGBoost]  MODERATE accuracy -- check class imbalance / window_size tuning.")
        else:
            print("    [XGBoost]  LOW accuracy -- review data quality, thresholds, or feature set.")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Out-of-sample evaluation for Aditya-L1 solar flare prediction models."
    )
    p.add_argument(
        "--data_dir", type=str, default="data/test_fits",
        help="Directory containing held-out test FITS files (default: data/test_fits).",
    )
    p.add_argument(
        "--threshold", type=float, default=0.5,
        help="Binary decision threshold for PatchTST flare classification (default: 0.5).",
    )
    p.add_argument(
        "--batch_size", type=int, default=64,
        help="DataLoader batch size for PatchTST inference (default: 64).",
    )
    p.add_argument(
        "--device", type=str, default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="Torch inference device (default: cpu).",
    )
    p.add_argument(
        "--inject_flare", action="store_true", default=False,
        help=(
            "Overlay a synthetic M-class flare spike onto the (possibly all-quiet) "
            "test data before running inference.  Use this to verify the model flags "
            "a simulated eruption when no real flare events are present in the test set."
        ),
    )
    return p.parse_args()


def main():
    args = parse_args()

    print(BANNER)
    logger.info("Configuration:")
    logger.info("  data_dir     : %s", args.data_dir)
    logger.info("  threshold    : %.2f", args.threshold)
    logger.info("  batch_size   : %d", args.batch_size)
    logger.info("  device       : %s", args.device)
    logger.info("  inject_flare : %s", args.inject_flare)

    # ── Auto-detect GPU / MPS availability ───────────────────────────────────
    import torch
    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available — falling back to CPU.")
        args.device = "cpu"
    if args.device == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        logger.warning("MPS not available — falling back to CPU.")
        args.device = "cpu"

    # ── Stage 1: Parse + feature-engineer test FITS (no scaler leakage) ──────
    df_features = load_and_engineer_test_data(args.data_dir)

    # ── Synthetic flare injection (--inject_flare) ────────────────────────────
    # Overlays a simulated M-class spike onto the engineered DataFrame so we can
    # verify the model detects the eruption even when the test set is all-quiet.
    # The injection is applied AFTER feature engineering intentionally: the bump
    # directly elevates the normalised feature values that the model consumes,
    # which is the most faithful test of the classifier's discrimination boundary.
    if args.inject_flare:
        logger.warning(
            "--inject_flare is ACTIVE.  Results reflect synthetic-flare conditions, "
            "not the real test set distribution."
        )
        df_features = inject_flare_into_dataframe(df_features)

    # ── Stage 2: Load all model artefacts ─────────────────────────────────────
    patchtst_model, xgb_model, device_obj = load_models(args.device)

    # ── Stage 3: PatchTST evaluation ──────────────────────────────────────────
    patchtst_metrics = evaluate_patchtst(
        df_features, patchtst_model, device_obj,
        threshold=args.threshold,
        batch_size=args.batch_size,
    )

    # ── Stage 4: XGBoost evaluation ───────────────────────────────────────────
    xgb_metrics = evaluate_xgboost(df_features, xgb_model)

    # ── Stage 5: Summary ──────────────────────────────────────────────────────
    print_summary(patchtst_metrics, xgb_metrics)

    logger.info("Evaluation pipeline complete.")


if __name__ == "__main__":
    main()
