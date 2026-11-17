"""
align.py — Resample and merge SoLEXS + HEL1OS DataFrames onto a common UTC timebase.

Physics context
---------------
SoLEXS and HEL1OS have different native cadences and may have different
observation gaps.  Before any joint analysis (cross-instrument coincidence
detection, spectral hardness ratio, Neupert-effect consistency check) the two
light curves must be co-registered onto the same time grid.

Design choices
--------------
* **Outer join** (default): preserves gaps as NaN.  This is important because
  detector dropouts look different from quiet-sun periods and downstream models
  need to distinguish them.  An inner join would silently discard data where
  one instrument was operating and the other was not.
* **Cadence selection**: if the user sets RESAMPLE_CADENCE_S=0 in .env we
  automatically pick the coarser (larger) of the two instruments' native
  cadences, so we never upsample (interpolate) a sparse channel.
* **Per-channel pivot (optional)**: the long-format merged DataFrame is the
  default output.  If `pivot=True` the function returns a wide-format
  DataFrame with one flux column per (instrument, energy_channel) pair, which
  is easier for direct ML feature engineering.
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Types ──────────────────────────────────────────────────────────────────────
JoinMode = Literal["outer", "inner"]


# ── Helpers ────────────────────────────────────────────────────────────────────

def _infer_cadence(df: pd.DataFrame) -> pd.Timedelta:
    """
    Estimate the native cadence of a light-curve DataFrame as the median
    gap between consecutive timestamps.

    We use the *median* rather than the minimum to be robust against
    sub-cadence jitter in the timestamp column.
    """
    if len(df) < 2:
        raise ValueError("Cannot infer cadence from fewer than 2 rows.")
    diffs = df["timestamp"].sort_values().diff().dropna()
    return diffs.median()


def _resample_instrument(
    df: pd.DataFrame,
    cadence: pd.Timedelta,
    instrument: str,
) -> pd.DataFrame:
    """
    Resample a single-instrument light curve to a regular *cadence*.

    Strategy per column:
    - flux       : mean (preserves energy; summing then dividing is correct for
                   photon counts as long as the original cadence is uniform)
    - flux_err   : quadrature sum of errors divided by number of samples (σ_combined)
    - quality_flag : max (worst quality wins — conservative)
    - energy_low, energy_high : first value (constant per channel)
    - energy_channel, instrument : first value (categorical)

    The DataFrame is resampled per (instrument, energy_channel) group so that
    a multi-band HEL1OS file is handled correctly.
    """
    df = df.copy()
    df = df.set_index("timestamp")

    groups: list[pd.DataFrame] = []
    for (instr, channel), grp in df.groupby(["instrument", "energy_channel"], sort=False):
        resampled = grp.resample(cadence).agg(
            flux=("flux", "mean"),
            flux_err=("flux_err", lambda x: np.sqrt(np.nansum(x**2)) / max(len(x), 1)),
            quality_flag=("quality_flag", "max"),
            energy_low=("energy_low", "first"),
            energy_high=("energy_high", "first"),
        )
        resampled["energy_channel"] = channel
        resampled["instrument"] = instr
        groups.append(resampled)

    if not groups:
        return pd.DataFrame()

    out = pd.concat(groups).reset_index()
    out = out.rename(columns={"timestamp": "timestamp"})
    return out


# ── Public API ─────────────────────────────────────────────────────────────────

def align(
    solexs_df: pd.DataFrame,
    hel1os_df: pd.DataFrame,
    cadence_s: int = 0,
    join: JoinMode = "outer",
    pivot: bool = False,
) -> pd.DataFrame:
    """
    Resample SoLEXS and HEL1OS light curves onto a shared UTC time grid and
    merge them into a single DataFrame.

    Parameters
    ----------
    solexs_df : pd.DataFrame
        Output of ``read_solexs()`` — must have a UTC-aware ``timestamp`` column.
    hel1os_df : pd.DataFrame
        Output of ``read_hel1os()`` — must have a UTC-aware ``timestamp`` column.
    cadence_s : int
        Target cadence in seconds.  If 0 (default), use the coarser of the two
        instruments' native cadences (never upsample).
    join : {"outer", "inner"}
        How to merge the two instruments' time grids.  "outer" keeps all
        timestamps from both instruments (gaps appear as NaN).  "inner" keeps
        only timestamps present in both.
    pivot : bool
        If True, return wide-format with one column per (instrument, channel).
        If False (default), return long-format.

    Returns
    -------
    pd.DataFrame
        Long format (default):
            timestamp, flux, flux_err, energy_low, energy_high,
            energy_channel, instrument, quality_flag
        Wide format (pivot=True):
            timestamp, flux_SoLEXS_1-30keV, flux_HEL1OS_8-150keV, …

    Notes
    -----
    Timestamps are rounded to the target cadence *before* resampling so that
    clock jitter between the two instruments does not produce misaligned bins.
    """
    if solexs_df.empty and hel1os_df.empty:
        raise ValueError("Both input DataFrames are empty.")

    # ── Determine cadence ──────────────────────────────────────────────────────
    if cadence_s > 0:
        target_cadence = pd.Timedelta(seconds=cadence_s)
    else:
        cadences: list[pd.Timedelta] = []
        if not solexs_df.empty:
            cadences.append(_infer_cadence(solexs_df))
        if not hel1os_df.empty:
            cadences.append(_infer_cadence(hel1os_df))
        # Coarser = larger timedelta
        target_cadence = max(cadences)
        logger.info("align: auto-selected cadence = %s", target_cadence)

    # ── Resample each instrument ───────────────────────────────────────────────
    frames: list[pd.DataFrame] = []
    if not solexs_df.empty:
        frames.append(_resample_instrument(solexs_df, target_cadence, "SoLEXS"))
    if not hel1os_df.empty:
        frames.append(_resample_instrument(hel1os_df, target_cadence, "HEL1OS"))

    merged_long = pd.concat(frames, ignore_index=True).sort_values(
        ["timestamp", "instrument", "energy_low"]
    ).reset_index(drop=True)

    if join == "inner":
        # Keep only timestamps present in ALL (instrument, channel) groups
        # This is conservative — use with caution when data has many gaps
        common_ts = None
        for (instr, ch), grp in merged_long.groupby(["instrument", "energy_channel"]):
            ts_set = set(grp["timestamp"])
            common_ts = ts_set if common_ts is None else common_ts & ts_set
        if common_ts is not None:
            merged_long = merged_long[merged_long["timestamp"].isin(common_ts)]
        logger.info(
            "align (inner join): retained %d rows across %d unique timestamps.",
            len(merged_long),
            merged_long["timestamp"].nunique(),
        )
    else:
        logger.info(
            "align (outer join): %d rows across %d unique timestamps.",
            len(merged_long),
            merged_long["timestamp"].nunique(),
        )

    if not pivot:
        return merged_long

    # ── Optional wide-format pivot ─────────────────────────────────────────────
    wide = merged_long.pivot_table(
        index="timestamp",
        columns=["instrument", "energy_channel"],
        values=["flux", "flux_err", "quality_flag"],
        aggfunc="first",
    )
    # Flatten MultiIndex columns: "flux_SoLEXS_1-30keV", etc.
    wide.columns = [
        f"{stat}_{instr}_{ch}" for stat, instr, ch in wide.columns
    ]
    wide = wide.reset_index().sort_values("timestamp")
    logger.info("align: pivoted to wide format with %d columns.", wide.shape[1])
    return wide


def clip_to_overlap(
    solexs_df: pd.DataFrame,
    hel1os_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Trim both DataFrames to their temporal overlap.

    Useful when one instrument has a longer observation window than the other
    and you want to avoid large one-sided NaN regions in the merged output.
    """
    start = max(solexs_df["timestamp"].min(), hel1os_df["timestamp"].min())
    end = min(solexs_df["timestamp"].max(), hel1os_df["timestamp"].max())

    if start >= end:
        logger.warning("clip_to_overlap: no temporal overlap found between instruments!")
        return solexs_df.iloc[0:0], hel1os_df.iloc[0:0]

    sx = solexs_df[solexs_df["timestamp"].between(start, end)].reset_index(drop=True)
    hx = hel1os_df[hel1os_df["timestamp"].between(start, end)].reset_index(drop=True)
    logger.info("clip_to_overlap: overlap = %s to %s", start, end)
    return sx, hx
