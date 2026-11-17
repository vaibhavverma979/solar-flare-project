from fastapi import APIRouter
from src.api.cache import get_cached_forecast_prediction

router = APIRouter(prefix="/api/forecast", tags=["Forecasting"])

@router.get("/predict")
def get_forecast_prediction(solexs: str = "data/raw/AL1_SOLEXS_20260731_SDD2_L1.lc", hel1os: str = "data/raw/lightcurve_cdte1.fits"):
    try:
        return get_cached_forecast_prediction(solexs, hel1os)
    except Exception as exc:
        return {"status": "error", "message": str(exc), "probability": 0.0}
