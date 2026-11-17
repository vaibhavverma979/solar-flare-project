"""
threshold_detector.py — Sigma-above-background trigger (Phase 3).

Physics context
---------------
The simplest and most interpretable flare detection method is a sigma trigger:
a flare is flagged when the background-subtracted flux exceeds N×σ_background,
where σ_background is estimated from the noise in the pre-event window.

GOES uses a fixed threshold in absolute flux units (e.g., C1 = 10⁻⁶ W/m²).
For SoLEXS/HEL1OS we parameterize by sigma so the same code works regardless
of the instrument's absolute calibration (which may change after in-flight
calibration).

Key physics decisions encoded here:
1.  **Per-instrument thresholds**: SoLEXS and HEL1OS have very different
    background levels and noise statistics.  A 5σ threshold on SoLEXS means
    something physically different from 5σ on HEL1OS.
2.  **Refractory period**: After a flare peak, the count rate stays elevated
    for minutes (gradual phase).  Without a refractory period the detector
    fires repeated alerts for the same event.  Default: 10 minutes.
3.  **Minimum duration**: A 1-second spike above threshold is likely a cosmic
    ray, not a flare.  Require the threshold to be exceeded for ≥10 consecutive
    seconds (configurable).

TODO (Phase 3): Implement detect(df, sigma, min_duration_s, refractory_s) → pd.DataFrame
"""


import pandas as pd
import numpy as np


def detect(
    df: pd.DataFrame,
    sigma: float = 5.0,
    min_duration_s: int = 10,
    refractory_s: int = 600
) -> pd.DataFrame:
    """
    Flag time intervals where the background-subtracted flux exceeds *sigma*
    standard deviations above the estimated noise level.

    Parameters
    ----------
    df : pd.DataFrame
        Output of preprocessing.background.subtract_background().
    sigma : float
        Detection threshold in units of background noise (default: 5σ).
    min_duration_s : int
        Minimum consecutive seconds above threshold to declare a detection.
    refractory_s : int
        Minimum time (seconds) between successive detections per instrument.

    Returns
    -------
    pd.DataFrame
        Detection catalogue with columns:
        start_time, peak_time, end_time, peak_flux, sigma_excess, instrument.
    """
    catalogue_cols = ["start_time", "peak_time", "end_time", "peak_flux", "sigma_excess", "instrument"]
    if df.empty:
        return pd.DataFrame(columns=catalogue_cols)

    df_copy = df.copy()
    if "timestamp" in df_copy.columns:
        df_copy = df_copy.sort_values("timestamp")
    else:
        df_copy["timestamp"] = pd.date_range("2026-01-01", periods=len(df_copy), freq="1s")

    instruments = df_copy["instrument"].unique() if "instrument" in df_copy.columns else ["SoLEXS"]
    records = []

    for inst in instruments:
        if "instrument" in df_copy.columns:
            sub_df = df_copy[df_copy["instrument"] == inst].copy()
        else:
            sub_df = df_copy.copy()

        flux_col = "flux_background_sub" if "flux_background_sub" in sub_df.columns else "flux"
        flux_val = sub_df[flux_col].fillna(0.0).values
        times = pd.to_datetime(sub_df["timestamp"]).values

        if "flux_err" in sub_df.columns and sub_df["flux_err"].notna().any():
            noise_std = float(sub_df["flux_err"].median())
        else:
            noise_std = float(np.std(flux_val))
        
        if noise_std <= 0:
            noise_std = 1.0

        threshold = sigma * noise_std
        above_thresh = flux_val >= threshold

        # Group contiguous True blocks
        if not np.any(above_thresh):
            continue

        # Find transitions
        diff = np.diff(above_thresh.astype(int))
        starts = np.where(diff == 1)[0] + 1
        ends = np.where(diff == -1)[0]
        if above_thresh[0]:
            starts = np.r_[0, starts]
        if above_thresh[-1]:
            ends = np.r_[ends, len(above_thresh) - 1]

        last_detection_end_time = None

        for s_idx, e_idx in zip(starts, ends):
            start_t = pd.Timestamp(times[s_idx])
            end_t = pd.Timestamp(times[e_idx])
            duration = (end_t - start_t).total_seconds()

            if duration < min_duration_s and (e_idx - s_idx) < min_duration_s:
                continue

            if last_detection_end_time is not None:
                if (start_t - last_detection_end_time).total_seconds() < refractory_s:
                    continue

            # Identify peak in interval
            interval_flux = flux_val[s_idx : e_idx + 1]
            peak_relative_idx = int(np.argmax(interval_flux))
            peak_t = pd.Timestamp(times[s_idx + peak_relative_idx])
            peak_flux_val = float(interval_flux[peak_relative_idx])
            sigma_exc = peak_flux_val / noise_std

            records.append({
                "start_time": start_t,
                "peak_time": peak_t,
                "end_time": end_t,
                "peak_flux": peak_flux_val,
                "sigma_excess": sigma_exc,
                "instrument": str(inst),
            })
            last_detection_end_time = end_t

    return pd.DataFrame(records, columns=catalogue_cols)

