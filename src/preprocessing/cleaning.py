"""
cleaning.py — Data quality filtering and dead-time correction (Phase 2).

Physics context
---------------
Space-based X-ray detectors suffer from several data quality issues:

1.  **Dead time**: When a very bright photon is detected, the electronics are
    briefly "dead" and cannot register subsequent photons.  For impulsive flares
    this can cause the measured flux to be *lower* than the true flux at peak.
    Dead-time correction restores the true rate: R_true = R_obs / (1 - R_obs * τ)
    where τ is the dead-time constant (instrument specific).

2.  **Particle contamination**: South Atlantic Anomaly (SAA) passages and cosmic
    ray showers produce spikes in the count rate that mimic flares.  Quality
    flags from the PRADAN pipeline mark these intervals.

3.  **Telemetry dropouts**: Missing packets produce NaN or zero-rate intervals.

TODO (Phase 2): Implement clean(df, dead_time_s) → pd.DataFrame
"""


import pandas as pd
import numpy as np


def clean(df: pd.DataFrame, dead_time_s: float = 2.5e-6) -> pd.DataFrame:
    """
    Apply quality filters and dead-time correction to a light-curve DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Aligned, background-subtracted light-curve DataFrame.
    dead_time_s : float
        Instrument dead-time constant in seconds (default: HEL1OS nominal).

    Returns
    -------
    pd.DataFrame
        Copy of *df* with:
        - Rows where quality_flag > 0 replaced with NaN flux values.
        - Dead-time corrected flux in a new column ``flux_deadtime_corr``.
        - A boolean ``is_valid`` column that downstream models can use as a mask.
    """
    if df.empty:
        df_copy = df.copy()
        df_copy["flux_deadtime_corr"] = np.nan
        df_copy["is_valid"] = pd.Series(dtype=bool)
        return df_copy

    df_copy = df.copy()

    # Mask bad quality rows (quality_flag > 0)
    if "quality_flag" in df_copy.columns:
        bad_mask = df_copy["quality_flag"] > 0
        if "flux" in df_copy.columns:
            df_copy.loc[bad_mask, "flux"] = np.nan
        if "flux_background_sub" in df_copy.columns:
            df_copy.loc[bad_mask, "flux_background_sub"] = np.nan

    # Target flux column for dead-time correction
    target_col = "flux_background_sub" if "flux_background_sub" in df_copy.columns else "flux"

    if target_col in df_copy.columns:
        r_obs = df_copy[target_col].astype(float)
        denom = (1.0 - r_obs * dead_time_s).clip(lower=0.01)
        r_true = r_obs / denom
        df_copy["flux_deadtime_corr"] = r_true

        # Determine validity boolean mask
        quality_ok = (df_copy["quality_flag"] == 0) if "quality_flag" in df_copy.columns else True
        flux_ok = df_copy[target_col].notna()
        df_copy["is_valid"] = quality_ok & flux_ok
    else:
        df_copy["flux_deadtime_corr"] = np.nan
        df_copy["is_valid"] = False

    return df_copy

