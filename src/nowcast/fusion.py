"""
fusion.py — Cross-instrument coincidence matching and Neupert-effect consistency (Phase 3).

Physics context — the Neupert effect
-------------------------------------
In 1968, Werner Neupert observed that the soft X-ray (SXR) time profile of a
solar flare is approximately the time-integral of the hard X-ray (HXR) or
microwave profile.  This arises because:

- HXR bremsstrahlung is produced by non-thermal electrons accelerated in
  the corona immediately during the impulsive phase.
- SXR emission comes from chromospheric plasma heated by these electrons
  (chromospheric evaporation) and persists much longer.

Consequences for fusion:
1.  **HXR onset precedes or coincides with SXR onset** — if SoLEXS triggers
    first, the event is more likely a thermal long-duration event (no Neupert),
    which has a lower probability of being an impulsive M/X class flare.
2.  **Coincidence window ±2 min** (configurable) — if both instruments trigger
    within this window, the detection confidence is higher.
3.  **Confidence score** = base_confidence + Δ_neupert_bonus + Δ_both_instruments_bonus

Key testable functions (required by the spec):
- ``neupert_consistent(sx_start, hx_start, tolerance_s=120) → bool``
  Named, testable, physics-justified function.
- ``match_detections(sx_catalogue, hx_catalogue, window_s=120) → pd.DataFrame``
  Cross-instrument event matching.
- ``build_master_catalogue(matched_df) → pd.DataFrame``
  Adds confidence scores, flare class (via classify.py), and metadata.

TODO (Phase 3): Implement all three functions above.
"""


import pandas as pd
import numpy as np
from src.nowcast.classify import classify_peak


def neupert_consistent(sx_start, hx_start, tolerance_s: int = 120) -> bool:
    """
    Return True if the SoLEXS and HEL1OS detection times are consistent with
    the Neupert effect (HXR onset precedes or coincides with SXR onset within
    *tolerance_s* seconds).
    """
    if pd.isna(sx_start) or pd.isna(hx_start):
        return False
    sx_t = pd.Timestamp(sx_start)
    hx_t = pd.Timestamp(hx_start)
    
    # Neupert: HXR starts before or close to SXR start
    # hx_t <= sx_t + tolerance_s AND hx_t >= sx_t - 600s
    max_lag = pd.Timedelta(seconds=tolerance_s)
    max_lead = pd.Timedelta(seconds=600)
    return (hx_t <= sx_t + max_lag) and (hx_t >= sx_t - max_lead)


def match_detections(sx_catalogue: pd.DataFrame, hx_catalogue: pd.DataFrame, window_s: int = 120) -> pd.DataFrame:
    """
    Match SoLEXS and HEL1OS single-instrument detections within *window_s*.
    """
    matched = []
    sx_used = set()
    hx_used = set()

    sx_cats = sx_catalogue.copy() if not sx_catalogue.empty else pd.DataFrame()
    hx_cats = hx_catalogue.copy() if not hx_catalogue.empty else pd.DataFrame()

    if not sx_cats.empty and not hx_cats.empty:
        for s_idx, s_row in sx_cats.iterrows():
            s_start = pd.Timestamp(s_row["start_time"])
            best_h_idx = None
            min_dt = float("inf")

            for h_idx, h_row in hx_cats.iterrows():
                if h_idx in hx_used:
                    continue
                h_start = pd.Timestamp(h_row["start_time"])
                dt = abs((s_start - h_start).total_seconds())
                if dt <= window_s and dt < min_dt:
                    min_dt = dt
                    best_h_idx = h_idx

            if best_h_idx is not None:
                h_row = hx_cats.loc[best_h_idx]
                sx_used.add(s_idx)
                hx_used.add(best_h_idx)

                h_start = pd.Timestamp(h_row["start_time"])
                is_neupert = neupert_consistent(s_start, h_start, tolerance_s=window_s)

                matched.append({
                    "sx_start": s_start,
                    "hx_start": h_start,
                    "start_time": min(s_start, h_start),
                    "peak_time": s_row["peak_time"],
                    "end_time": max(pd.Timestamp(s_row["end_time"]), pd.Timestamp(h_row["end_time"])),
                    "peak_flux_soft": s_row["peak_flux"],
                    "peak_flux_hard": h_row["peak_flux"],
                    "instruments_agreeing": 2,
                    "neupert_consistent": is_neupert,
                })

    # Unmatched SoLEXS
    if not sx_cats.empty:
        for s_idx, s_row in sx_cats.iterrows():
            if s_idx not in sx_used:
                matched.append({
                    "sx_start": s_row["start_time"],
                    "hx_start": pd.NaT,
                    "start_time": s_row["start_time"],
                    "peak_time": s_row["peak_time"],
                    "end_time": s_row["end_time"],
                    "peak_flux_soft": s_row["peak_flux"],
                    "peak_flux_hard": np.nan,
                    "instruments_agreeing": 1,
                    "neupert_consistent": False,
                })

    # Unmatched HEL1OS
    if not hx_cats.empty:
        for h_idx, h_row in hx_cats.iterrows():
            if h_idx not in hx_used:
                matched.append({
                    "sx_start": pd.NaT,
                    "hx_start": h_row["start_time"],
                    "start_time": h_row["start_time"],
                    "peak_time": h_row["peak_time"],
                    "end_time": h_row["end_time"],
                    "peak_flux_soft": np.nan,
                    "peak_flux_hard": h_row["peak_flux"],
                    "instruments_agreeing": 1,
                    "neupert_consistent": False,
                })

    res_df = pd.DataFrame(matched)
    if not res_df.empty:
        res_df = res_df.sort_values("start_time").reset_index(drop=True)
    return res_df


def build_master_catalogue(matched_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the master flare catalogue from matched detections.
    """
    cols = [
        "event_id", "start_time", "peak_time", "end_time", "class",
        "peak_flux_soft", "peak_flux_hard", "confidence",
        "instruments_agreeing", "neupert_consistent"
    ]
    if matched_df.empty:
        return pd.DataFrame(columns=cols)

    df_copy = matched_df.copy()
    records = []

    for idx, row in df_copy.iterrows():
        event_id = f"FLARE_{idx+1:04d}"
        peak_soft = float(row.get("peak_flux_soft", np.nan))
        flare_cls = classify_peak(peak_soft)

        agree = int(row.get("instruments_agreeing", 1))
        neupert = bool(row.get("neupert_consistent", False))

        # Confidence calculation
        conf = 0.5
        if agree > 1:
            conf += 0.3
        if neupert:
            conf += 0.2
        conf = min(1.0, conf)
        
        # Geomagnetic Storm Fusion Logic (ASPEX & MAG)
        # Default benign values if payload data is missing
        b_z = float(row.get("b_z", 5.0))
        wind_velocity = float(row.get("wind_velocity", 400.0))
        
        geomagnetic_warning = False
        arrival_time = None
        
        if flare_cls in ['M', 'X'] and b_z < -5.0 and wind_velocity > 500.0:
            geomagnetic_warning = True
            # Calculate dynamic arrival time
            # Delay in minutes = 1,500,000 km / wind_velocity / 60
            delay_mins = (1_500_000 / wind_velocity) / 60.0
            delay_td = pd.Timedelta(minutes=delay_mins)
            arrival_time = pd.Timestamp(row["peak_time"]) + delay_td

        records.append({
            "event_id": event_id,
            "start_time": row["start_time"],
            "peak_time": row["peak_time"],
            "end_time": row["end_time"],
            "class": flare_cls,
            "peak_flux_soft": peak_soft,
            "peak_flux_hard": float(row.get("peak_flux_hard", np.nan)),
            "confidence": conf,
            "instruments_agreeing": agree,
            "neupert_consistent": neupert,
            "geomagnetic_warning": geomagnetic_warning,
            "earth_arrival_time": arrival_time,
            "b_z": b_z,
            "wind_velocity": wind_velocity
        })

    return pd.DataFrame(records, columns=cols + ["geomagnetic_warning", "earth_arrival_time", "b_z", "wind_velocity"])

