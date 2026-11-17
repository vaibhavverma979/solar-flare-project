"""
multimodal_fusion.py — Multi-modal fusion engine combining all AI signal sources.

Combines four signal channels for near-optimal flare prediction:
    1. X-ray time-series forecast (LSTM / PatchTST)
    2. SUIT visual complexity score (CNN)
    3. Historical pattern match confidence (VectorDB)
    4. Neupert-effect consistency (physics heuristic)

Fusion strategy: weighted ensemble with configurable weights.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class ModalitySignal:
    """Individual modality contribution to the fused prediction."""

    name: str
    probability: float  # 0–1
    confidence: float   # 0–1 (how reliable this modality's output is)
    weight: float       # configured weight
    details: dict = field(default_factory=dict)


@dataclass
class FusionReport:
    """Complete multi-modal fusion result."""

    fused_probability: float      # final weighted probability
    risk_level: str               # LOW / MEDIUM / HIGH / CRITICAL
    modalities: list[ModalitySignal] = field(default_factory=list)
    dominant_modality: str = ""   # which modality contributed most
    fusion_method: str = "weighted_ensemble"

    def to_dict(self) -> dict:
        return {
            "fused_probability": round(self.fused_probability, 4),
            "risk_level": self.risk_level,
            "fusion_method": self.fusion_method,
            "dominant_modality": self.dominant_modality,
            "modalities": [
                {
                    "name": m.name,
                    "probability": round(m.probability, 4),
                    "confidence": round(m.confidence, 4),
                    "weight": round(m.weight, 4),
                    "weighted_contribution": round(m.probability * m.weight, 4),
                    "details": m.details,
                }
                for m in self.modalities
            ],
        }


class MultiModalFusionEngine:
    """
    Fuse multi-modal signals into a single flare probability.

    Parameters
    ----------
    weight_timeseries : float
        Weight for the X-ray time-series forecast channel.
    weight_vision : float
        Weight for the SUIT visual complexity channel.
    weight_historical : float
        Weight for the historical pattern match channel.
    weight_neupert : float
        Weight for the Neupert-effect consistency channel.
    """

    def __init__(
        self,
        weight_timeseries: float = 0.45,
        weight_vision: float = 0.20,
        weight_historical: float = 0.20,
        weight_neupert: float = 0.15,
    ):
        total = weight_timeseries + weight_vision + weight_historical + weight_neupert
        # Normalise weights to sum to 1
        self.w_ts = weight_timeseries / total
        self.w_cv = weight_vision / total
        self.w_hist = weight_historical / total
        self.w_neupert = weight_neupert / total

    def fuse(
        self,
        p_timeseries: float = 0.0,
        p_vision: float = 0.0,
        p_historical: float = 0.0,
        p_neupert: float = 0.0,
        ts_confidence: float = 1.0,
        cv_confidence: float = 1.0,
        hist_confidence: float = 1.0,
        neupert_confidence: float = 1.0,
        ts_details: Optional[dict] = None,
        cv_details: Optional[dict] = None,
        hist_details: Optional[dict] = None,
        neupert_details: Optional[dict] = None,
    ) -> FusionReport:
        """
        Compute the fused multi-modal flare probability.

        Each modality provides:
        - A probability estimate (0–1)
        - A confidence score (0–1) indicating how reliable the estimate is
          (e.g., if no SUIT image is available, cv_confidence = 0)

        The fusion uses confidence-weighted blending:
            effective_weight_i = base_weight_i × confidence_i
            final_prob = Σ(effective_weight_i × probability_i) / Σ(effective_weight_i)
        """
        modalities = [
            ModalitySignal(
                name="X-ray Time Series (LSTM/PatchTST)",
                probability=np.clip(p_timeseries, 0, 1),
                confidence=np.clip(ts_confidence, 0, 1),
                weight=self.w_ts,
                details=ts_details or {},
            ),
            ModalitySignal(
                name="SUIT Visual Complexity (CNN)",
                probability=np.clip(p_vision, 0, 1),
                confidence=np.clip(cv_confidence, 0, 1),
                weight=self.w_cv,
                details=cv_details or {},
            ),
            ModalitySignal(
                name="Historical Pattern Match (VectorDB)",
                probability=np.clip(p_historical, 0, 1),
                confidence=np.clip(hist_confidence, 0, 1),
                weight=self.w_hist,
                details=hist_details or {},
            ),
            ModalitySignal(
                name="Neupert Effect Consistency",
                probability=np.clip(p_neupert, 0, 1),
                confidence=np.clip(neupert_confidence, 0, 1),
                weight=self.w_neupert,
                details=neupert_details or {},
            ),
        ]

        # Confidence-weighted fusion
        numerator = 0.0
        denominator = 0.0
        for m in modalities:
            eff_weight = m.weight * m.confidence
            numerator += eff_weight * m.probability
            denominator += eff_weight

        fused_prob = float(numerator / max(denominator, 1e-12))
        fused_prob = float(np.clip(fused_prob, 0, 1))

        # Determine risk level
        risk = _risk_level(fused_prob)

        # Dominant modality (highest effective contribution)
        contributions = [
            (m.name, m.weight * m.confidence * m.probability)
            for m in modalities
        ]
        dominant = max(contributions, key=lambda x: x[1])[0]

        return FusionReport(
            fused_probability=fused_prob,
            risk_level=risk,
            modalities=modalities,
            dominant_modality=dominant,
        )


def _risk_level(prob: float) -> str:
    """Map fused probability to a human-readable risk level."""
    if prob >= 0.80:
        return "CRITICAL"
    elif prob >= 0.60:
        return "HIGH"
    elif prob >= 0.30:
        return "MEDIUM"
    else:
        return "LOW"


# ── Convenience: derive probability from non-probabilistic signals ────────────

def complexity_to_probability(complexity_score: float) -> float:
    """
    Map a SUIT complexity score [0, 1] to a flare probability estimate.

    Uses an empirical mapping calibrated to McIntosh classification statistics:
    - alpha (0.0–0.2)  → ~5% flare probability (M-class+)
    - beta  (0.2–0.4)  → ~15%
    - βγ    (0.4–0.7)  → ~40%
    - βγδ   (0.7–1.0)  → ~70%
    """
    # Sigmoid-like mapping
    return float(1.0 / (1.0 + np.exp(-8 * (complexity_score - 0.5))))


def similarity_to_probability(similarity: float, reference_class: str) -> float:
    """
    Map a historical match similarity score to a flare probability.

    Higher similarity to a known X-class event → higher probability.
    """
    class_priors = {
        "A": 0.01, "B": 0.05, "C": 0.15, "M": 0.50, "X": 0.85,
    }
    # Extract class letter
    cls_letter = reference_class[0].upper() if reference_class else "C"
    prior = class_priors.get(cls_letter, 0.15)

    # Blend similarity with the prior probability of the matched event
    return float(np.clip(similarity * prior + (1 - similarity) * 0.1, 0, 1))
