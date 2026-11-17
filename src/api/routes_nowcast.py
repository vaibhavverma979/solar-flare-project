from fastapi import APIRouter
import pandas as pd
import numpy as np
from src.api.cache import get_cached_nowcast_catalogue

router = APIRouter(prefix="/api/nowcast", tags=["Nowcasting"])

@router.get("/catalogue")
def get_nowcast_catalogue(solexs: str = "data/raw/AL1_SOLEXS_20260731_SDD2_L1.lc", hel1os: str = "data/raw/lightcurve_cdte1.fits"):
    try:
        master = get_cached_nowcast_catalogue(solexs, hel1os)
        if not master.empty:
            df_copy = master.copy()
            df_copy["start_time"] = df_copy["start_time"].astype(str)
            df_copy["peak_time"] = df_copy["peak_time"].astype(str)
            df_copy["end_time"] = df_copy["end_time"].astype(str)
            # Replace NaNs with None for JSON compliance
            records = df_copy.replace({np.nan: None}).to_dict(orient="records")
        else:
            records = []

        return {"status": "ok", "count": len(records), "catalogue": records}
    except Exception as exc:
        return {"status": "error", "message": str(exc), "catalogue": []}
