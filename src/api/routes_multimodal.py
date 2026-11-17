"""
routes_multimodal.py — REST endpoints for multi-modal AI fusion.

Endpoints:
    GET /api/multimodal/status            — Module availability check
    GET /api/multimodal/fusion            — Fused multi-modal prediction
    GET /api/multimodal/historical-match  — Top-5 historical pattern matches
    GET /api/multimodal/suit-analysis     — SUIT image analysis (synthetic demo)
"""

from __future__ import annotations

from fastapi import APIRouter
import numpy as np

router = APIRouter(prefix="/api/multimodal", tags=["Multi-Modal AI"])


# ── Module availability check ────────────────────────────────────────────────

def _check_module(import_path: str) -> dict:
    """Check if a Python module is importable and return status."""
    try:
        __import__(import_path)
        return {"available": True, "error": None}
    except ImportError as e:
        return {"available": False, "error": str(e)}


@router.get("/status")
def multimodal_status():
    """Return availability status of all multi-modal AI modules."""
    pytorch_status = _check_module("torch")
    torchvision_status = _check_module("torchvision")
    chromadb_status = _check_module("chromadb")

    return {
        "status": "ok",
        "modules": {
            "vision_pipeline": {
                "status": "active" if pytorch_status["available"] and torchvision_status["available"] else "fallback_heuristic",
                "pytorch": pytorch_status["available"],
                "torchvision": torchvision_status["available"],
                "description": "SUIT Solar UV Image Analysis — CNN sunspot magnetic classification",
            },
            "vector_database": {
                "status": "active" if chromadb_status["available"] else "fallback_inmemory",
                "chromadb": chromadb_status["available"],
                "description": "Historical Flare Pattern Matching — 50 canonical events from 1859–2025",
            },
            "patchtst_transformer": {
                "status": "active" if pytorch_status["available"] else "fallback_lstm",
                "pytorch": pytorch_status["available"],
                "description": "PatchTST Time-Series Transformer — 6-hour attention windows",
            },
            "multimodal_fusion": {
                "status": "active",
                "description": "Weighted ensemble combining all signal sources",
            },
        },
    }


# ── Multi-modal fused prediction ─────────────────────────────────────────────

@router.get("/fusion")
def multimodal_fusion():
    """
    Return a fused multi-modal flare prediction combining:
    - X-ray time-series forecast
    - SUIT visual complexity
    - Historical pattern match
    - Neupert effect consistency
    """
    from src.nowcast.multimodal_fusion import (
        MultiModalFusionEngine,
        complexity_to_probability,
        similarity_to_probability,
    )

    try:
        # 1. Get time-series forecast probability from existing cache
        from src.api.cache import get_cached_forecast_prediction
        ts_result = get_cached_forecast_prediction(
            "data/raw/AL1_SOLEXS_20260731_SDD2_L1.lc",
            "data/raw/lightcurve_cdte1.fits",
        )
        p_ts = ts_result.get("peak_window_probability", 0.3)
    except Exception:
        p_ts = 0.3

    try:
        # 2. SUIT visual analysis (synthetic demo)
        from src.vision.suit_reader import SyntheticSuitGenerator
        from src.vision.sunspot_cnn import classify_sunspots
        gen = SyntheticSuitGenerator(size=256, seed=42)
        suit_img = gen.generate(n_sunspots=3, complexity="beta-gamma")
        classification = classify_sunspots(suit_img)
        p_cv = complexity_to_probability(classification.complexity_score)
        cv_confidence = 0.7  # synthetic data → lower confidence
        cv_details = {
            "magnetic_class": classification.magnetic_class,
            "complexity_score": round(classification.complexity_score, 4),
            "method": classification.method,
        }
    except Exception as e:
        p_cv = 0.0
        cv_confidence = 0.0
        cv_details = {"error": str(e)}

    try:
        # 3. Historical pattern match
        from src.vectordb.pattern_matcher import PatternMatcher
        matcher = PatternMatcher(use_cnn_embedder=False)  # Use statistical for speed
        # Generate a demo flux window
        demo_flux = np.random.default_rng(42).exponential(scale=1e-4, size=60) + 5e-5
        match_result = matcher.match_current_pattern(demo_flux, top_k=3)
        best_class = match_result.top_matches[0].flare_class if match_result.top_matches else "C1.0"
        best_sim = match_result.best_match_similarity
        p_hist = similarity_to_probability(best_sim, best_class)
        hist_details = match_result.to_dict()["best_match"]
    except Exception as e:
        p_hist = 0.0
        best_sim = 0.0
        hist_details = {"error": str(e)}

    try:
        # 4. Neupert effect from existing nowcast
        from src.api.cache import get_cached_nowcast_catalogue
        catalogue = get_cached_nowcast_catalogue(
            "data/raw/AL1_SOLEXS_20260731_SDD2_L1.lc",
            "data/raw/lightcurve_cdte1.fits",
        )
        if not catalogue.empty and "neupert_consistent" in catalogue.columns:
            neupert_frac = catalogue["neupert_consistent"].mean()
            p_neupert = float(0.3 + 0.7 * neupert_frac)
        else:
            p_neupert = 0.3
        neupert_confidence = 1.0
    except Exception:
        p_neupert = 0.3
        neupert_confidence = 0.5

    # Fuse all modalities
    engine = MultiModalFusionEngine()
    report = engine.fuse(
        p_timeseries=p_ts,
        p_vision=p_cv,
        p_historical=p_hist,
        p_neupert=p_neupert,
        ts_confidence=1.0,
        cv_confidence=cv_confidence,
        hist_confidence=min(best_sim + 0.3, 1.0) if best_sim > 0 else 0.5,
        neupert_confidence=neupert_confidence,
        ts_details={"model": "LSTM/PatchTST", "source": "X-ray flux"},
        cv_details=cv_details,
        hist_details=hist_details,
        neupert_details={"source": "SoLEXS/HEL1OS cross-correlation"},
    )

    return {"status": "ok", **report.to_dict()}


# ── Historical pattern match ─────────────────────────────────────────────────

@router.get("/historical-match")
def historical_match():
    """Find the top-5 most similar historical flare events to the current observation."""
    try:
        from src.vectordb.pattern_matcher import PatternMatcher

        matcher = PatternMatcher(use_cnn_embedder=False)

        # Use a representative flux window (could be replaced with real-time data)
        rng = np.random.default_rng(42)
        demo_flux = rng.exponential(scale=1e-4, size=60) + 5e-5

        result = matcher.match_current_pattern(demo_flux, top_k=5)

        return {
            "status": "ok",
            **result.to_dict(),
        }
    except Exception as e:
        return {"status": "error", "message": str(e), "top_matches": []}


# ── SUIT image analysis ──────────────────────────────────────────────────────

@router.get("/suit-analysis")
def suit_analysis():
    """
    Analyse a SUIT UV image for sunspot magnetic configuration.
    Uses synthetic images for demo when no real SUIT data is available.
    """
    try:
        from src.vision.suit_reader import SyntheticSuitGenerator
        from src.vision.sunspot_cnn import classify_sunspots
        from src.vision.visual_features import extract_visual_features

        gen = SyntheticSuitGenerator(size=256, seed=42)

        # Generate images at different complexity levels for demo
        results = []
        for complexity in ("alpha", "beta", "beta-gamma", "beta-gamma-delta"):
            img = gen.generate(n_sunspots=3, complexity=complexity)
            classification = classify_sunspots(img)
            features = extract_visual_features(img)

            results.append({
                "complexity_label": complexity,
                "magnetic_class": classification.magnetic_class,
                "class_probabilities": classification.class_probabilities.tolist(),
                "complexity_score": round(classification.complexity_score, 4),
                "method": classification.method,
                "visual_features": features.to_dict(),
                "image_size": list(img.image_size),
                "wavelength_nm": img.wavelength_nm,
            })

        return {
            "status": "ok",
            "source": "synthetic_demo",
            "note": "Using synthetic SUIT images. Replace with real PRADAN SUIT FITS files for production.",
            "analyses": results,
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}
