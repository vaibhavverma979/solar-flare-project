"""
features.py — Feature engineering for ML-based flare detection and forecasting (Phase 2).

Physics context
---------------
Raw count rates are poor direct inputs to ML models because:
- They have wildly varying scales (quiet sun vs. X-class flare: ~4 orders of magnitude).
- Temporal structure (rate of change) matters more than absolute value for
  *early* flare detection.

Feature set to implement
------------------------
1.  **Normalized flux** (log10 scale): log10(flux / background) — dimensionless,
    maps [quiet, A, B, C, M, X] → roughly [0, 1, 2, 3, 4, 5].

2.  **First derivative (slope)**: Δflux / Δt — the rate of rise is the key
    precursor signal.  Impulsive flares have ≥10× background rise in <2 min.

3.  **Second derivative (curvature)**: acceleration of the count rate.
    Positive curvature → onset phase; negative curvature → decay phase.

4.  **Spectral hardness ratio**: flux_HEL1OS / flux_SoLEXS — encodes the
    non-thermal fraction of the emission.  A rising hardness ratio before the
    SXR peak is diagnostic of particle acceleration (Neupert effect precursor).

5.  **Rolling statistics** over windows [1 min, 5 min, 15 min]:
    - rolling_mean_{window}
    - rolling_std_{window}
    - rolling_max_{window}
    (These capture multi-scale temporal context without requiring an RNN.)

TODO (Phase 2): Implement build_features(df) → pd.DataFrame
"""


import pandas as pd
import numpy as np


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add engineered feature columns to the aligned light-curve DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Wide-format aligned DataFrame (output of align.py with pivot=True),
        or standard aligned light curve.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with additional feature columns appended (in-place
        copy).  NaN values at the edges of derivative windows are left as NaN
        — the downstream model is responsible for handling them.
    """
    if df.empty:
        return df.copy()

    df_copy = df.copy()

    # Identify flux columns
    if "flux" in df_copy.columns:
        primary_flux_col = "flux"
    elif "flux_SoLEXS" in df_copy.columns:
        primary_flux_col = "flux_SoLEXS"
    else:
        # Pick first column starting with flux
        flux_cols = [c for c in df_copy.columns if c.startswith("flux")]
        primary_flux_col = flux_cols[0] if flux_cols else None

    if primary_flux_col is not None:
        flux = df_copy[primary_flux_col].astype(float)
        if "background" in df_copy.columns:
            bg = df_copy["background"].astype(float)
        else:
            bg = pd.Series(1.0, index=df_copy.index)

        # 1. Log normalized flux
        ratio = (flux / (bg.replace(0, np.nan).fillna(1.0))).clip(lower=1e-9)
        df_copy["log_flux"] = np.log10(ratio)

        # Time delta in seconds for derivatives
        if "timestamp" in df_copy.columns:
            ts = pd.to_datetime(df_copy["timestamp"])
            dt = ts.diff().dt.total_seconds().replace(0, np.nan)
        elif isinstance(df_copy.index, pd.DatetimeIndex):
            dt = df_copy.index.to_series().diff().dt.total_seconds().replace(0, np.nan)
        else:
            dt = 1.0

        # 2. First derivative (df/dt)
        df_copy["df_dt"] = flux.diff() / dt

        # 3. Second derivative (d2f/dt2)
        df_copy["d2f_dt2"] = df_copy["df_dt"].diff() / dt

        # 5. Rolling statistics [1 min, 5 min, 15 min]
        windows = [("1min", 60), ("5min", 300), ("15min", 900)]
        for w_label, w_sec in windows:
            if "timestamp" in df_copy.columns or isinstance(df_copy.index, pd.DatetimeIndex):
                # Time-based rolling
                if "timestamp" in df_copy.columns:
                    temp = pd.DataFrame({"time": pd.to_datetime(df_copy["timestamp"]), "val": flux}).set_index("time")
                else:
                    temp = pd.DataFrame({"val": flux}, index=df_copy.index)
                
                r = temp["val"].rolling(window=f"{w_sec}s", min_periods=1)
                df_copy[f"rolling_mean_{w_label}"] = r.mean().values
                df_copy[f"rolling_std_{w_label}"] = r.std().values
                df_copy[f"rolling_max_{w_label}"] = r.max().values
            else:
                # Row-count based rolling
                r = flux.rolling(window=max(1, w_sec), min_periods=1)
                df_copy[f"rolling_mean_{w_label}"] = r.mean()
                df_copy[f"rolling_std_{w_label}"] = r.std()
                df_copy[f"rolling_max_{w_label}"] = r.max()

    # 4. Spectral Hardness Ratio: flux_HEL1OS / flux_SoLEXS
    solexs_col = "flux_SoLEXS" if "flux_SoLEXS" in df_copy.columns else None
    hel1os_col = "flux_HEL1OS" if "flux_HEL1OS" in df_copy.columns else None

    if solexs_col and hel1os_col:
        s_flux = df_copy[solexs_col].astype(float)
        h_flux = df_copy[hel1os_col].astype(float)
        df_copy["hardness_ratio"] = h_flux / (s_flux.replace(0, np.nan).fillna(1e-9))
    elif "instrument" in df_copy.columns and "flux" in df_copy.columns:
        # Long format fallback
        piv = df_copy.pivot(index="timestamp", columns="instrument", values="flux")
        if "SoLEXS" in piv.columns and "HEL1OS" in piv.columns:
            hr = piv["HEL1OS"] / (piv["SoLEXS"].replace(0, np.nan).fillna(1e-9))
            df_copy["hardness_ratio"] = df_copy["timestamp"].map(hr)

    return df_copy

