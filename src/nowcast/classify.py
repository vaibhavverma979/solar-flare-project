"""
classify.py — Map peak flux to flare class (A/B/C/M/X analogue) (Phase 3).

Physics context
---------------
The GOES flare classification uses the peak 1–8 Å (1.5–12 keV) flux:

    A  : < 10⁻⁷ W/m²     (below C10 detection in quiet sun)
    B  : 10⁻⁷–10⁻⁶ W/m²
    C  : 10⁻⁶–10⁻⁵ W/m²
    M  : 10⁻⁵–10⁻⁴ W/m²
    X  : ≥ 10⁻⁴ W/m²

SoLEXS does not operate at the exact same energy band as GOES (0.5–4 Å and
1–8 Å), but after cross-calibration with GOES data a mapping from SoLEXS
counts/s to GOES flux equivalents can be established.

For the hackathon MVP we use a threshold table defined in counts/s
(empirically calibrated against GOES data from the same days when available,
otherwise using instrument-level flux estimates).

TODO (Phase 3): Implement classify_peak(peak_flux_cts_per_s) → str
"""


# Placeholder threshold table (counts/s for SoLEXS, to be calibrated with real data)
# Key = upper bound of class in counts/s; Value = class label
FLUX_CLASS_THRESHOLDS = {
    1e2: "A",
    1e3: "B",
    1e4: "C",
    1e5: "M",
    float("inf"): "X",
}


def classify_peak(peak_flux: float) -> str:
    """
    Map a peak flux value (counts/s) to the GOES-analogue flare class (A, B, C, M, X).
    """
    if pd_isna(peak_flux) or peak_flux < 0:
        return "A"

    for threshold, flare_class in sorted(FLUX_CLASS_THRESHOLDS.items()):
        if peak_flux < threshold:
            return flare_class
    return "X"


def pd_isna(val) -> bool:
    try:
        import pandas as pd
        return pd.isna(val)
    except Exception:
        return val != val

