"""
multimodal_fusion.py — NASA dual-model flare risk fusion engine.

Combines predictions from two independently trained NASA/NOAA models into
a single weighted flare risk score using the formula:

    Total_Risk = (0.4 × SDO_XGBoost_Magnetic_Risk) + (0.6 × GOES_PatchTST_Xray_Risk)

The 60/40 weighting reflects that GOES X-ray time-series has demonstrated
higher short-term (30-min) flare predictability than magnetic proxy features
alone (consistent with literature: Bobra & Couvidat 2015, Barnes et al. 2016).

Both model confidences can be down-weighted independently — e.g., if SHARP
data is unavailable for a given observation window, ``xgb_confidence=0``
automatically redistributes the full weight to the PatchTST signal.

Note
----
This module is separate from ``src/nowcast/multimodal_fusion.py``, which
implements the legacy 4-modality fusion (SUIT CNN, VectorDB, Neupert, LSTM).
Both can co-exist in the system; the API / dashboard should prefer this module
when operating in NASA-data-trained inference mode.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ===========================================================================
# Data classes
# ===========================================================================


@dataclass
class FusionResult:
    """Result from the NASA dual-model fusion engine.

    Attributes
    ----------
    total_risk : float
        Weighted flare risk score in [0, 1].
    risk_level : str
        Human-readable alert tier: LOW / MODERATE / HIGH / SEVERE / EXTREME.
    goes_patchtst_risk : float
        Raw PatchTST probability (GOES X-ray channel).
    sdo_xgb_risk : float
        Raw XGBoost probability (SDO SHARP magnetic channel).
    goes_contribution : float
        Effective weighted contribution of the PatchTST signal to total_risk.
    xgb_contribution : float
        Effective weighted contribution of the XGBoost signal to total_risk.
    dominant_source : str
        Which model contributed more to the final score.
    goes_confidence : float
        Confidence weight applied to the PatchTST signal (0–1).
    xgb_confidence : float
        Confidence weight applied to the XGBoost signal (0–1).
    alert_message : str
        Pre-formatted dashboard alert string.
    """

    total_risk: float
    risk_level: str
    goes_patchtst_risk: float
    sdo_xgb_risk: float
    goes_contribution: float
    xgb_contribution: float
    dominant_source: str
    goes_confidence: float
    xgb_confidence: float
    alert_message: str = field(default="")

    def to_dict(self) -> dict:
        """Serialise to a dashboard-ready JSON-compatible dict."""
        return {
            "total_risk": round(float(self.total_risk), 4),
            "risk_level": self.risk_level,
            "alert_message": self.alert_message,
            "models": {
                "goes_patchtst": {
                    "risk": round(float(self.goes_patchtst_risk), 4),
                    "weight_nominal": 0.6,
                    "confidence": round(float(self.goes_confidence), 4),
                    "effective_contribution": round(float(self.goes_contribution), 4),
                },
                "sdo_xgboost": {
                    "risk": round(float(self.sdo_xgb_risk), 4),
                    "weight_nominal": 0.4,
                    "confidence": round(float(self.xgb_confidence), 4),
                    "effective_contribution": round(float(self.xgb_contribution), 4),
                },
            },
            "dominant_source": self.dominant_source,
            "fusion_formula": "0.6 × GOES_PatchTST + 0.4 × SDO_XGBoost (confidence-adjusted)",
        }


# ===========================================================================
# Risk tier mapping
# ===========================================================================

_RISK_TIERS = [
    (0.90, "EXTREME"),
    (0.75, "SEVERE"),
    (0.55, "HIGH"),
    (0.30, "MODERATE"),
    (0.0,  "LOW"),
]


def _risk_level(prob: float) -> str:
    """Map a fused probability to a 5-tier risk level string."""
    for threshold, label in _RISK_TIERS:
        if prob >= threshold:
            return label
    return "LOW"


def _alert_message(risk_level: str, total_risk: float) -> str:
    """Generate a concise dashboard alert message."""
    messages = {
        "EXTREME": f"🔴 EXTREME flare risk ({total_risk:.1%}). Immediate space-weather action advised.",
        "SEVERE":  f"🟠 SEVERE flare risk ({total_risk:.1%}). Monitor closely — M/X event likely.",
        "HIGH":    f"🟡 HIGH flare risk ({total_risk:.1%}). Elevated M-class probability.",
        "MODERATE":f"🔵 MODERATE flare risk ({total_risk:.1%}). Increased solar activity.",
        "LOW":     f"🟢 LOW flare risk ({total_risk:.1%}). Quiet sun conditions.",
    }
    return messages.get(risk_level, f"Risk: {total_risk:.1%}")


# ===========================================================================
# Fusion engine
# ===========================================================================


class NASAFlareRiskFuser:
    """Weighted fusion of GOES PatchTST and SDO XGBoost flare risk scores.

    Nominal weights
    ---------------
    * GOES PatchTST (X-ray time-series) : 0.60
    * SDO XGBoost (magnetic features)   : 0.40

    If either model's confidence is < 1.0, its effective weight is reduced
    proportionally and the remaining weight is redistributed to the other
    model.  This ensures graceful degradation when one data source is
    unavailable:

    .. code-block::

        eff_w_ts  = 0.6 × goes_confidence
        eff_w_xgb = 0.4 × xgb_confidence
        total_eff = eff_w_ts + eff_w_xgb  (normalised to avoid zero division)
        total_risk = (eff_w_ts × goes_risk + eff_w_xgb × xgb_risk) / total_eff

    Parameters
    ----------
    w_goes : float
        Nominal weight for the PatchTST / GOES X-ray channel (default 0.6).
    w_xgb : float
        Nominal weight for the XGBoost / SDO SHARP channel (default 0.4).
    """

    def __init__(self, w_goes: float = 0.6, w_xgb: float = 0.4):
        total = w_goes + w_xgb
        self.w_goes = w_goes / total
        self.w_xgb  = w_xgb  / total

    def fuse(
        self,
        goes_patchtst_risk: Optional[float],
        sdo_xgb_risk: Optional[float],
        goes_confidence: float = 1.0,
        xgb_confidence: float = 1.0,
    ) -> FusionResult:
        """Compute the weighted flare risk score.

        Parameters
        ----------
        goes_patchtst_risk : float or None
            Probability from the GOES PatchTST model (0–1).
            Pass ``None`` or set ``goes_confidence=0`` when unavailable.
        sdo_xgb_risk : float or None
            Probability from the SDO XGBoost model (0–1).
            Pass ``None`` or set ``xgb_confidence=0`` when unavailable.
        goes_confidence : float
            Reliability of the GOES signal in [0, 1].
            Use 0.0 when SoLEXS telemetry is absent / not yet calibrated.
        xgb_confidence : float
            Reliability of the XGBoost signal in [0, 1].
            Use 0.0 when SHARP data is stale or unavailable.

        Returns
        -------
        FusionResult
            Fully populated fusion result with contributions and alert message.
        """
        # Graceful None handling — missing input → zero risk, zero confidence
        if goes_patchtst_risk is None:
            goes_patchtst_risk = 0.0
            goes_confidence = 0.0
        if sdo_xgb_risk is None:
            sdo_xgb_risk = 0.0
            xgb_confidence = 0.0

        # Clip to valid probability range
        goes_patchtst_risk = float(np.clip(goes_patchtst_risk, 0.0, 1.0))
        sdo_xgb_risk       = float(np.clip(sdo_xgb_risk,       0.0, 1.0))
        goes_confidence    = float(np.clip(goes_confidence,    0.0, 1.0))
        xgb_confidence     = float(np.clip(xgb_confidence,    0.0, 1.0))

        # Effective weights (confidence-adjusted)
        eff_w_goes = self.w_goes * goes_confidence
        eff_w_xgb  = self.w_xgb  * xgb_confidence
        total_eff  = eff_w_goes + eff_w_xgb

        if total_eff < 1e-12:
            # Both sources unavailable — return zero risk with warning
            total_risk = 0.0
            goes_contribution = 0.0
            xgb_contribution  = 0.0
        else:
            total_risk = (
                eff_w_goes * goes_patchtst_risk + eff_w_xgb * sdo_xgb_risk
            ) / total_eff
            goes_contribution = eff_w_goes * goes_patchtst_risk / total_eff
            xgb_contribution  = eff_w_xgb  * sdo_xgb_risk       / total_eff

        total_risk = float(np.clip(total_risk, 0.0, 1.0))
        risk_lv = _risk_level(total_risk)

        dominant = (
            "GOES PatchTST (X-ray)"
            if goes_contribution >= xgb_contribution
            else "SDO XGBoost (magnetic)"
        )

        return FusionResult(
            total_risk=total_risk,
            risk_level=risk_lv,
            goes_patchtst_risk=goes_patchtst_risk,
            sdo_xgb_risk=sdo_xgb_risk,
            goes_contribution=goes_contribution,
            xgb_contribution=xgb_contribution,
            dominant_source=dominant,
            goes_confidence=goes_confidence,
            xgb_confidence=xgb_confidence,
            alert_message=_alert_message(risk_lv, total_risk),
        )

    def fuse_batch(
        self,
        goes_risks: np.ndarray,
        xgb_risks: np.ndarray,
        goes_confidences: Optional[np.ndarray] = None,
        xgb_confidences: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Vectorised fusion over arrays of predictions.

        Parameters
        ----------
        goes_risks : np.ndarray, shape (N,)
        xgb_risks  : np.ndarray, shape (N,)
        goes_confidences : np.ndarray or None, shape (N,). Defaults to all 1.0.
        xgb_confidences  : np.ndarray or None, shape (N,). Defaults to all 1.0.

        Returns
        -------
        np.ndarray, shape (N,)
            Fused total risk scores.
        """
        N = len(goes_risks)
        if goes_confidences is None:
            goes_confidences = np.ones(N)
        if xgb_confidences is None:
            xgb_confidences = np.ones(N)

        goes_risks  = np.clip(goes_risks,  0.0, 1.0)
        xgb_risks   = np.clip(xgb_risks,   0.0, 1.0)
        goes_conf   = np.clip(goes_confidences, 0.0, 1.0)
        xgb_conf    = np.clip(xgb_confidences,  0.0, 1.0)

        eff_w_goes = self.w_goes * goes_conf
        eff_w_xgb  = self.w_xgb  * xgb_conf
        total_eff  = eff_w_goes + eff_w_xgb
        total_eff  = np.where(total_eff < 1e-12, 1.0, total_eff)

        fused = (eff_w_goes * goes_risks + eff_w_xgb * xgb_risks) / total_eff
        return np.clip(fused, 0.0, 1.0).astype(np.float32)
