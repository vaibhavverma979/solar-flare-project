from fastapi import APIRouter
import numpy as np
from src.api.cache import get_cached_raw_data

router = APIRouter(prefix="/api/lightcurve", tags=["Lightcurve"])

@router.get("/data")
def get_lightcurve_data(solexs: str = "data/raw/AL1_SOLEXS_20260731_SDD2_L1.lc", hel1os: str = "data/raw/lightcurve_cdte1.fits", downsample: int = 50):
    try:
        sx_df, hx_df = get_cached_raw_data(solexs, hel1os)

        result = {}
        if not sx_df.empty:
            sub_sx = sx_df.iloc[::downsample]
            flux_vals = [float(x) if (x == x and not np.isnan(x)) else 0.0 for x in sub_sx["flux"]]
            result["solexs"] = {
                "timestamps": sub_sx["timestamp"].astype(str).tolist(),
                "flux": flux_vals,
                "instrument": "SoLEXS (1-30 keV)"
            }
        
        if not hx_df.empty:
            sub_hx = hx_df.iloc[::downsample]
            flux_col = "flux" if "flux" in sub_hx.columns else ("CTR" if "CTR" in sub_hx.columns else sub_hx.columns[1])
            flux_vals_hx = [float(x) if (x == x and not np.isnan(x)) else 0.0 for x in sub_hx[flux_col]]
            result["hel1os"] = {
                "timestamps": sub_hx["timestamp"].astype(str).tolist(),
                "flux": flux_vals_hx,
                "instrument": "HEL1OS (8-150 keV)"
            }

        return {"status": "ok", "data": result}
    except Exception as exc:
        return {"status": "error", "message": str(exc), "data": {}}
