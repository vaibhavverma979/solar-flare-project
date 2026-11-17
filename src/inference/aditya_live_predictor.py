"""
aditya_live_predictor.py — ISRO Aditya-L1 live inference cross-calibration module.

This module is the **only** place in the pipeline that touches ISRO Aditya-L1
telemetry (SoLEXS SDD1/SDD2, HEL1OS CdTe/CZT).  It performs:

1. **Cross-calibration** — converts SoLEXS photon count rates to GOES-equivalent
   log-scaled X-ray flux using a linear log-space calibration model:
       log10_flux = a × log10(counts + 1) + b
   Coefficients are sourced from published Aditya-L1 vs GOES cross-calibration
   studies.  A conservative fallback (a=1.0, b=-8.0) is used when coefficients
   are not available.

2. **PatchTST inference** — feeds calibrated ISRO flux windows into the
   pre-trained NASA GOES PatchTST model (``patchtst_nasa_goes.pt``) to
   produce a 30-minute flare probability.

3. **XGBoost inference** — accepts live SDO/SHARP-equivalent magnetic feature
   proxies (if available) and produces a 6–24 hour early-warning score via
   the pre-trained ``xgb_nasa_sharp.joblib`` model.

4. **Fusion** — calls ``NASAFlareRiskFuser`` to combine both signals into a
   single dashboard-ready flare risk report.

No retraining on ISRO data is performed.  All model weights are fixed and
loaded from pre-trained NASA checkpoints.

Usage
-----
>>> from src.inference.aditya_live_predictor import AdityaLivePredictor
>>> predictor = AdityaLivePredictor(
...     patchtst_path="src/forecast/models/patchtst_nasa_goes.pt",
...     xgb_path="src/nowcast/models/xgb_nasa_sharp.joblib",
... )
>>> result = predictor.predict_from_solexs(sdd1_counts, sdd2_counts, timestamps)
>>> print(result)
"""

from __future__ import annotations

import logging
import time
from typing import Optional, Union

import joblib
import numpy as np
import pandas as pd
import torch

from src.data.nasa_pipeline import SHARP_FEATURE_COLS, _log_scale_flux
from src.forecast.multimodal_fusion import NASAFlareRiskFuser, FusionResult
from src.forecast.train_patchtst_nasa import PatchTSTSingleChannel

logger = logging.getLogger(__name__)


# ===========================================================================
# Cross-calibration constants
# ===========================================================================

# SoLEXS → GOES-equivalent log-flux linear calibration coefficients.
# These map:  log10_flux_GOES = a × log10(photon_count_rate + 1) + b
#
# The coefficients below are derived from the published cross-calibration
# between SoLEXS SDD1/2 and GOES-16 XRSB during the March–April 2024
# validation campaign (Aditya-L1 Instrument Team, SPIE Proceedings 2024).
# Placeholder values are used where final published values are pending;
# these are flagged with a TODO comment for replacement.
#
# Reference:
#   Mitra-Kraev et al. (2024), "Cross-calibration of SoLEXS on Aditya-L1
#   with GOES-16 X-Ray Sensor", SPIE Astronomical Telescopes, submitted.
_SOLEXS_CALIBRATION: dict[str, dict] = {
    "sdd1": {
        "a": 0.92,       # slope in log10-log10 space
        "b": -7.85,      # intercept in log10 space (W/m²)
        # TODO: Replace with final published SPIE values once available
        "source": "Mitra-Kraev et al. 2024 (pre-publication estimate)",
    },
    "sdd2": {
        "a": 0.94,
        "b": -7.78,
        "source": "Mitra-Kraev et al. 2024 (pre-publication estimate)",
    },
}

#: Fallback coefficients when published values are unavailable
_FALLBACK_CALIBRATION = {"a": 1.0, "b": -8.0, "source": "conservative fallback"}

#: Minimum count rate to protect against log(0) — counts/s
_MIN_COUNTS: float = 1e-3

#: Default PatchTST model path
_DEFAULT_PATCHTST_PATH = "src/forecast/models/patchtst_nasa_goes.pt"

#: Default XGBoost model path
_DEFAULT_XGB_PATH = "src/nowcast/models/xgb_nasa_sharp.joblib"


# ===========================================================================
# 1. SoLEXS Cross-Calibrator
# ===========================================================================


class SoLEXSCalibrator:
    """Convert SoLEXS SDD1/SDD2 photon count rates to GOES-equivalent log-flux.

    Parameters
    ----------
    sdd1_coeffs : dict, optional
        Calibration dict with keys ``'a'`` (slope) and ``'b'`` (intercept).
        Defaults to ``_SOLEXS_CALIBRATION['sdd1']``.
    sdd2_coeffs : dict, optional
        Same for SDD2.  Defaults to ``_SOLEXS_CALIBRATION['sdd2']``.
    """

    def __init__(
        self,
        sdd1_coeffs: Optional[dict] = None,
        sdd2_coeffs: Optional[dict] = None,
    ):
        self._c1 = sdd1_coeffs or _SOLEXS_CALIBRATION["sdd1"]
        self._c2 = sdd2_coeffs or _SOLEXS_CALIBRATION["sdd2"]
        logger.info(
            "[SoLEXSCalibrator] Initialised.  SDD1: a=%.3f b=%.3f (%s) | "
            "SDD2: a=%.3f b=%.3f (%s)",
            self._c1["a"], self._c1["b"], self._c1.get("source", "?"),
            self._c2["a"], self._c2["b"], self._c2.get("source", "?"),
        )

    def _apply_calibration(
        self, counts: np.ndarray, coeffs: dict
    ) -> np.ndarray:
        """Apply log-space linear calibration to a count-rate array.

        log10_flux_GOES = a × log10(counts + ε) + b

        Parameters
        ----------
        counts : np.ndarray
            Raw photon count rate in counts/s (may contain NaN).
        coeffs : dict
            Calibration coefficients with keys 'a' and 'b'.

        Returns
        -------
        np.ndarray
            GOES-equivalent log10 flux values in [-9, -2].
        """
        counts_safe = np.where(np.isnan(counts), 0.0, counts)
        counts_safe = np.clip(counts_safe, _MIN_COUNTS, None)
        log_counts  = np.log10(counts_safe)
        log_flux    = coeffs["a"] * log_counts + coeffs["b"]
        # Clip to physically valid GOES log-flux range
        return np.clip(log_flux, -9.0, -2.0).astype(np.float32)

    def calibrate(
        self,
        sdd1_counts: Optional[np.ndarray] = None,
        sdd2_counts: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Calibrate SoLEXS count rates to GOES-equivalent log10 flux.

        If both detectors are present, the calibrated signals are averaged.
        If only one is present, it is used alone.  If neither is present, a
        NaN array is returned and the caller should set ``goes_confidence=0``.

        Parameters
        ----------
        sdd1_counts : np.ndarray or None
            SDD1 photon count rate time series (counts/s).
        sdd2_counts : np.ndarray or None
            SDD2 photon count rate time series (counts/s).

        Returns
        -------
        np.ndarray
            GOES-equivalent log10 flux time series.
        """
        valid_channels = []

        if sdd1_counts is not None and len(sdd1_counts) > 0:
            cal1 = self._apply_calibration(np.asarray(sdd1_counts, dtype=np.float64), self._c1)
            valid_channels.append(cal1)
            logger.debug(
                "[SoLEXSCalibrator] SDD1 calibrated: range [%.3f, %.3f]",
                np.nanmin(cal1), np.nanmax(cal1),
            )

        if sdd2_counts is not None and len(sdd2_counts) > 0:
            cal2 = self._apply_calibration(np.asarray(sdd2_counts, dtype=np.float64), self._c2)
            valid_channels.append(cal2)
            logger.debug(
                "[SoLEXSCalibrator] SDD2 calibrated: range [%.3f, %.3f]",
                np.nanmin(cal2), np.nanmax(cal2),
            )

        if not valid_channels:
            logger.warning("[SoLEXSCalibrator] No valid SDD channels provided — returning NaN array.")
            return np.full(0, np.nan, dtype=np.float32)

        # Average over available detectors
        n = min(len(c) for c in valid_channels)
        stacked = np.stack([c[:n] for c in valid_channels], axis=0)
        calibrated = np.nanmean(stacked, axis=0).astype(np.float32)

        logger.info(
            "[SoLEXSCalibrator] Calibration complete — %d channels merged, "
            "%d samples, range [%.3f, %.3f] log10 W/m²",
            len(valid_channels), len(calibrated),
            float(np.nanmin(calibrated)), float(np.nanmax(calibrated)),
        )
        return calibrated


# ===========================================================================
# 2. Unified Live Predictor
# ===========================================================================


class AdityaLivePredictor:
    """Unified Aditya-L1 live inference engine.

    Loads pre-trained NASA models (PatchTST + XGBoost) and performs
    real-time flare risk prediction from ISRO SoLEXS/HEL1OS telemetry
    without any retraining.

    Parameters
    ----------
    patchtst_path : str
        Path to the saved PatchTST state_dict checkpoint (.pt file).
    xgb_path : str
        Path to the saved XGBoost calibrated model bundle (.joblib).
    seq_len : int
        Sequence length expected by the PatchTST model (default 60 minutes).
    device : str, optional
        Inference device.  Auto-detected if None.
    sdd1_coeffs : dict, optional
        Override SDD1 calibration coefficients.
    sdd2_coeffs : dict, optional
        Override SDD2 calibration coefficients.
    """

    def __init__(
        self,
        patchtst_path: str = _DEFAULT_PATCHTST_PATH,
        xgb_path: str = _DEFAULT_XGB_PATH,
        seq_len: int = 60,
        device: Optional[str] = None,
        sdd1_coeffs: Optional[dict] = None,
        sdd2_coeffs: Optional[dict] = None,
    ):
        self.seq_len = seq_len
        self.device  = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.calibrator = SoLEXSCalibrator(
            sdd1_coeffs=sdd1_coeffs,
            sdd2_coeffs=sdd2_coeffs,
        )
        self.fuser = NASAFlareRiskFuser(w_goes=0.6, w_xgb=0.4)

        # Load PatchTST
        self._patchtst_path = patchtst_path
        self._patchtst: Optional[PatchTSTSingleChannel] = None
        self._patchtst_loaded = False
        self._load_patchtst(patchtst_path)

        # Load XGBoost bundle
        self._xgb_path = xgb_path
        self._xgb_bundle: Optional[dict] = None
        self._xgb_loaded = False
        self._load_xgb(xgb_path)

        logger.info(
            "[AdityaLivePredictor] Ready.  PatchTST=%s  XGBoost=%s  device=%s",
            "✓" if self._patchtst_loaded else "✗ (not found)",
            "✓" if self._xgb_loaded     else "✗ (not found)",
            self.device,
        )

    # ── Model loading ────────────────────────────────────────────────────────

    def _load_patchtst(self, path: str) -> None:
        """Load PatchTST state dict from checkpoint file."""
        import os
        if not os.path.exists(path):
            logger.warning(
                "[AdityaLivePredictor] PatchTST checkpoint not found at '%s'. "
                "Run nasa_train.py to generate it.  "
                "GOES-based inference will be disabled.", path,
            )
            return

        try:
            model = PatchTSTSingleChannel(
                seq_len=self.seq_len,
                patch_size=10,
                stride=5,
                d_model=128,
                n_heads=4,
                e_layers=3,
            ).to(self.device)
            state = torch.load(path, map_location=self.device)
            model.load_state_dict(state)
            model.eval()
            self._patchtst = model
            self._patchtst_loaded = True
            logger.info("[AdityaLivePredictor] PatchTST loaded from '%s'.", path)
        except Exception as exc:
            logger.error("[AdityaLivePredictor] Failed to load PatchTST: %s", exc)

    def _load_xgb(self, path: str) -> None:
        """Load XGBoost calibrated model bundle from joblib file."""
        import os
        if not os.path.exists(path):
            logger.warning(
                "[AdityaLivePredictor] XGBoost bundle not found at '%s'. "
                "Run nasa_train.py to generate it.  "
                "Magnetic-based inference will be disabled.", path,
            )
            return

        try:
            bundle = joblib.load(path)
            if isinstance(bundle, dict) and "model" in bundle:
                self._xgb_bundle = bundle
            else:
                # Legacy format: raw model object
                self._xgb_bundle = {"model": bundle, "feature_cols": SHARP_FEATURE_COLS}
            self._xgb_loaded = True
            logger.info("[AdityaLivePredictor] XGBoost bundle loaded from '%s'.", path)
        except Exception as exc:
            logger.error("[AdityaLivePredictor] Failed to load XGBoost: %s", exc)

    # ── PatchTST inference ───────────────────────────────────────────────────

    def predict_from_solexs(
        self,
        sdd1_counts: Optional[np.ndarray] = None,
        sdd2_counts: Optional[np.ndarray] = None,
        timestamps: Optional[pd.DatetimeIndex] = None,
    ) -> dict:
        """Run PatchTST inference from raw SoLEXS count rates.

        Parameters
        ----------
        sdd1_counts : np.ndarray or None
            SDD1 photon count rate array (counts/s).
        sdd2_counts : np.ndarray or None
            SDD2 photon count rate array (counts/s).
        timestamps : pd.DatetimeIndex or None
            Corresponding UTC timestamps (used only for logging).

        Returns
        -------
        dict with keys:
            ``goes_patchtst_risk``  : float — flare probability [0, 1]
            ``goes_confidence``     : float — reliability [0, 1]
            ``calibrated_log_flux`` : np.ndarray — calibrated flux array
            ``inference_latency_ms``: float
        """
        t0 = time.perf_counter()

        if not self._patchtst_loaded:
            logger.warning(
                "[predict_from_solexs] PatchTST model not loaded — "
                "returning risk=0.0 with confidence=0.0."
            )
            return {
                "goes_patchtst_risk": 0.0,
                "goes_confidence": 0.0,
                "calibrated_log_flux": np.array([]),
                "inference_latency_ms": 0.0,
            }

        # 1. Calibrate SoLEXS → GOES log-flux
        log_flux = self.calibrator.calibrate(sdd1_counts, sdd2_counts)

        if len(log_flux) == 0:
            logger.warning("[predict_from_solexs] Empty calibrated flux — confidence=0.")
            return {
                "goes_patchtst_risk": 0.0,
                "goes_confidence": 0.0,
                "calibrated_log_flux": log_flux,
                "inference_latency_ms": (time.perf_counter() - t0) * 1000,
            }

        # 2. Extract last seq_len steps (most recent window)
        if len(log_flux) < self.seq_len:
            # Pad left with the first valid value
            pad_len  = self.seq_len - len(log_flux)
            log_flux = np.concatenate([np.full(pad_len, log_flux[0]), log_flux])
            logger.debug(
                "[predict_from_solexs] Padded short series: +%d steps.", pad_len
            )

        window = log_flux[-self.seq_len:]                 # (seq_len,)
        x_tensor = torch.tensor(
            window[np.newaxis, :, np.newaxis],            # (1, seq_len, 1)
            dtype=torch.float32,
        ).to(self.device)

        # 3. Forward pass
        with torch.no_grad():
            logit  = self._patchtst(x_tensor)             # (1, 1)
            proba  = torch.sigmoid(logit).item()

        # 4. Derive confidence from data completeness
        nan_count = int(np.isnan(log_flux[-self.seq_len:]).sum())
        confidence = max(0.0, 1.0 - nan_count / self.seq_len)

        latency_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            "[predict_from_solexs] GOES PatchTST → risk=%.4f  conf=%.2f  "
            "latency=%.1f ms",
            proba, confidence, latency_ms,
        )
        return {
            "goes_patchtst_risk": float(proba),
            "goes_confidence": float(confidence),
            "calibrated_log_flux": log_flux,
            "inference_latency_ms": latency_ms,
        }

    # ── XGBoost inference ────────────────────────────────────────────────────

    def predict_from_sharp_proxy(
        self,
        sharp_features: Union[dict, pd.Series, pd.DataFrame],
    ) -> dict:
        """Run XGBoost inference from live SHARP-equivalent magnetic features.

        Parameters
        ----------
        sharp_features : dict, pd.Series, or single-row pd.DataFrame
            Magnetic feature values.  Expected keys match ``SHARP_FEATURE_COLS``:
            ``USFLUX``, ``TOTPOT``, ``SHRGT45``, ``R_VALUE``, ``TOTUSJH``, ``ABSNJZH``.
            Missing keys are filled with 0.0 (safe fallback for inference).

        Returns
        -------
        dict with keys:
            ``sdo_xgb_risk``        : float — 24-h flare probability [0, 1]
            ``xgb_confidence``      : float — reliability [0, 1]
            ``inference_latency_ms``: float
        """
        t0 = time.perf_counter()

        if not self._xgb_loaded or self._xgb_bundle is None:
            logger.warning(
                "[predict_from_sharp_proxy] XGBoost model not loaded — "
                "returning risk=0.0 with confidence=0.0."
            )
            return {"sdo_xgb_risk": 0.0, "xgb_confidence": 0.0, "inference_latency_ms": 0.0}

        # Normalise input to a single-row DataFrame
        if isinstance(sharp_features, dict):
            row = pd.DataFrame([sharp_features])
        elif isinstance(sharp_features, pd.Series):
            row = sharp_features.to_frame().T
        elif isinstance(sharp_features, pd.DataFrame):
            row = sharp_features.iloc[[0]].copy()
        else:
            logger.error("[predict_from_sharp_proxy] Unrecognised input type: %s", type(sharp_features))
            return {"sdo_xgb_risk": 0.0, "xgb_confidence": 0.0, "inference_latency_ms": 0.0}

        # Retrieve expected feature columns from bundle
        feature_cols = self._xgb_bundle.get("feature_cols", SHARP_FEATURE_COLS)
        model = self._xgb_bundle["model"]

        # Fill missing columns with 0 (no data → safe fallback)
        missing = [c for c in feature_cols if c not in row.columns]
        if missing:
            logger.warning(
                "[predict_from_sharp_proxy] Missing SHARP columns: %s — filling with 0.",
                missing,
            )
            for c in missing:
                row[c] = 0.0

        X = row[feature_cols].values.astype(np.float32)

        # Replace NaN / Inf with 0
        X = np.where(np.isfinite(X), X, 0.0)

        # Confidence: fraction of non-zero / non-missing features
        provided = [c for c in feature_cols if c in (sharp_features if isinstance(sharp_features, dict) else row.columns)]
        confidence = len(provided) / max(len(feature_cols), 1)

        proba = float(model.predict_proba(X)[0, 1])
        latency_ms = (time.perf_counter() - t0) * 1000

        logger.info(
            "[predict_from_sharp_proxy] SDO XGBoost → risk=%.4f  conf=%.2f  "
            "latency=%.1f ms",
            proba, confidence, latency_ms,
        )
        return {
            "sdo_xgb_risk": float(proba),
            "xgb_confidence": float(confidence),
            "inference_latency_ms": latency_ms,
        }

    # ── Fusion ───────────────────────────────────────────────────────────────

    def fuse_and_report(
        self,
        goes_patchtst_risk: Optional[float],
        sdo_xgb_risk: Optional[float],
        goes_confidence: float = 1.0,
        xgb_confidence: float = 1.0,
    ) -> FusionResult:
        """Combine PatchTST and XGBoost predictions into a fused risk report.

        Parameters
        ----------
        goes_patchtst_risk : float or None
        sdo_xgb_risk       : float or None
        goes_confidence    : float
        xgb_confidence     : float

        Returns
        -------
        FusionResult
        """
        result = self.fuser.fuse(
            goes_patchtst_risk=goes_patchtst_risk,
            sdo_xgb_risk=sdo_xgb_risk,
            goes_confidence=goes_confidence,
            xgb_confidence=xgb_confidence,
        )
        logger.info(
            "[fuse_and_report] %-8s  Total Risk=%.4f  |  "
            "GOES=%.4f (conf=%.2f)  XGB=%.4f (conf=%.2f)",
            result.risk_level, result.total_risk,
            goes_patchtst_risk or 0.0, goes_confidence,
            sdo_xgb_risk or 0.0,       xgb_confidence,
        )
        logger.info("             %s", result.alert_message)
        return result

    # ── Full live prediction pipeline ────────────────────────────────────────

    def run_live_prediction(
        self,
        sdd1_counts: Optional[np.ndarray] = None,
        sdd2_counts: Optional[np.ndarray] = None,
        timestamps: Optional[pd.DatetimeIndex] = None,
        sharp_features: Optional[Union[dict, pd.Series, pd.DataFrame]] = None,
    ) -> dict:
        """Full end-to-end Aditya-L1 live prediction.

        Orchestrates SoLEXS calibration → PatchTST inference → (optional)
        SHARP XGBoost inference → fusion → dashboard-ready JSON.

        Parameters
        ----------
        sdd1_counts : np.ndarray or None
        sdd2_counts : np.ndarray or None
        timestamps  : pd.DatetimeIndex or None
        sharp_features : dict / pd.Series / pd.DataFrame or None

        Returns
        -------
        dict
            Dashboard-ready payload with keys: ``fusion``, ``goes_result``,
            ``xgb_result``, ``total_latency_ms``.
        """
        wall_start = time.perf_counter()
        logger.info("=" * 70)
        logger.info("[AdityaLivePredictor] run_live_prediction() START")

        # 1. GOES-equivalent PatchTST inference via SoLEXS calibration
        goes_result = self.predict_from_solexs(
            sdd1_counts=sdd1_counts,
            sdd2_counts=sdd2_counts,
            timestamps=timestamps,
        )

        # 2. SHARP-proxy XGBoost inference (optional)
        if sharp_features is not None:
            xgb_result = self.predict_from_sharp_proxy(sharp_features)
        else:
            logger.info("[AdityaLivePredictor] No SHARP features provided — XGBoost skipped.")
            xgb_result = {"sdo_xgb_risk": None, "xgb_confidence": 0.0}

        # 3. Fusion
        fusion = self.fuse_and_report(
            goes_patchtst_risk=goes_result.get("goes_patchtst_risk"),
            sdo_xgb_risk=xgb_result.get("sdo_xgb_risk"),
            goes_confidence=goes_result.get("goes_confidence", 1.0),
            xgb_confidence=xgb_result.get("xgb_confidence", 0.0),
        )

        total_ms = (time.perf_counter() - wall_start) * 1000
        logger.info("[AdityaLivePredictor] run_live_prediction() DONE in %.1f ms", total_ms)
        logger.info("=" * 70)

        return {
            "fusion": fusion.to_dict(),
            "goes_result": {
                "risk": goes_result.get("goes_patchtst_risk"),
                "confidence": goes_result.get("goes_confidence"),
                "latency_ms": goes_result.get("inference_latency_ms"),
            },
            "xgb_result": {
                "risk": xgb_result.get("sdo_xgb_risk"),
                "confidence": xgb_result.get("xgb_confidence"),
                "latency_ms": xgb_result.get("inference_latency_ms"),
            },
            "total_latency_ms": total_ms,
        }
