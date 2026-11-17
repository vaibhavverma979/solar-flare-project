"""
background.py — Rolling-median background subtraction (Phase 2).

Physics context
---------------
Solar X-ray detectors measure a baseline "quiescent sun" flux that evolves
on timescales of hours (driven by the active region count on the solar disk
and the orbital viewing geometry of Aditya-L1 at L1).  Before flare
detection, this slowly varying background must be subtracted so that the
detector signal we threshold on is purely the *excess* flux above quiet sun.

Rolling median is preferred over rolling mean because:
1.  It is robust to flare contamination within the window (a flare is an
    outlier; the median ignores it until it occupies >50% of the window).
2.  It preserves sharp onset edges better than a causal low-pass filter.
3.  It has no free parameters beyond window length.

Window lengths to implement (configurable):
- Short  (1 min)  — catches instrument gain drifts on very short timescales
- Medium (5 min)  — default for threshold detection
- Long   (15 min) — for slow background estimation during complex active periods

TODO (Phase 2): Implement subtract_background(df, window_s) → pd.DataFrame
"""


import pandas as pd
import numpy as np


def subtract_background(df: pd.DataFrame, window_s: int = 300) -> pd.DataFrame:
    """
    Subtract a rolling-median background from the flux column.

    Parameters
    ----------
    df : pd.DataFrame
        Aligned light-curve DataFrame (output of align.py).
    window_s : int
        Rolling window size in seconds (default: 5 minutes = 300 s).

    Returns
    -------
    pd.DataFrame
        Copy of *df* with two new columns:
        - ``background``        : estimated background flux
        - ``flux_background_sub`` : background-subtracted flux (negative values
                                    clipped to 0 — physical floor is zero counts)
    """
    if df.empty:
        df_copy = df.copy()
        df_copy["background"] = np.nan
        df_copy["flux_background_sub"] = np.nan
        return df_copy

    df_copy = df.copy()

    # Determine sorting and index for time-based rolling
    if "timestamp" in df_copy.columns:
        df_copy = df_copy.sort_values("timestamp")
        time_series = pd.to_datetime(df_copy["timestamp"])
    elif isinstance(df_copy.index, pd.DatetimeIndex):
        time_series = df_copy.index
    else:
        # Fall back to row-count window if no timestamp column/index
        time_series = None

    window_str = f"{window_s}s"

    def _calc_bg(series: pd.Series, times: pd.Series | None) -> pd.Series:
        if times is not None:
            temp_df = pd.DataFrame({"time": times, "val": series}).set_index("time")
            bg = temp_df["val"].rolling(window=window_str, min_periods=1).median()
            return bg.values
        else:
            return series.rolling(window=max(1, window_s), min_periods=1).median()

    if "flux" in df_copy.columns:
        if "instrument" in df_copy.columns:
            bg_series = pd.Series(index=df_copy.index, dtype=float)
            for _, grp in df_copy.groupby("instrument"):
                bg_val = _calc_bg(grp["flux"], grp["timestamp"] if "timestamp" in grp.columns else None)
                bg_series.loc[grp.index] = bg_val
            df_copy["background"] = bg_series
        else:
            df_copy["background"] = _calc_bg(df_copy["flux"], time_series)

        df_copy["flux_background_sub"] = (df_copy["flux"] - df_copy["background"]).clip(lower=0.0)
    
    return df_copy

