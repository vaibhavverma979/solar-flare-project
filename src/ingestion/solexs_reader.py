"""
solexs_reader.py — SoLEXS (Soft X-ray Spectrometer) FITS/CSV loader.

Physics context
---------------
SoLEXS is a Si-PIN diode spectrometer aboard Aditya-L1 covering the soft
X-ray band (~1–30 keV).  It measures the integrated solar X-ray flux (counts
per second) in one or more energy channels with a native time cadence of ~1 s.
The soft X-ray light curve is the primary diagnostic for flare detection
because the GOES flare classification (A/B/C/M/X) is defined in terms of
1–8 Å (~1.5–12 keV) peak flux.

Output schema (long format — one row per time × channel)
---------------------------------------------------------
timestamp       : datetime64[ns, UTC]   — event time, always UTC
flux            : float64               — counts s⁻¹ or photons cm⁻² s⁻¹
flux_err        : float64               — 1-σ Poisson uncertainty (NaN if unavailable)
energy_low      : float32               — lower bound of energy channel (keV)
energy_high     : float32               — upper bound of energy channel (keV)
energy_channel  : str                   — human label, e.g. "1-30keV"
instrument      : str                   — always "SoLEXS"
quality_flag    : int8                  — 0=good, 1=suspect, 2=bad
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Column-name aliases ────────────────────────────────────────────────────────
# Maps the standard internal name → list of candidate names in PRADAN files.
# Extended by inspect_fits.py findings before first use on real data.
_TIME_ALIASES: list[str] = ["TIME", "Date_Time", "UTC_TIME", "MJD", "TT", "TIMESTAMP"]
_FLUX_ALIASES: list[str] = ["RATE", "COUNTS", "FLUX", "COUNT_RATE", "SXR_FLUX"]
_FLUX_ERR_ALIASES: list[str] = ["RATE_ERR", "ERROR", "FLUX_ERR", "STAT_ERR", "ERR"]
_QUALITY_ALIASES: list[str] = ["QUALITY", "FLAG", "STATUS", "GOOD"]

# Default energy band when the file does not specify one
_DEFAULT_ENERGY_LOW: float = 1.0   # keV
_DEFAULT_ENERGY_HIGH: float = 30.0  # keV


# ── Internal helpers ───────────────────────────────────────────────────────────

def _find_column(available: list[str], aliases: list[str]) -> Optional[str]:
    """Return the first alias found in *available* (case-insensitive), or None."""
    upper = {c.upper(): c for c in available}
    for alias in aliases:
        if alias.upper() in upper:
            return upper[alias.upper()]
    return None


def _parse_time_column(series: pd.Series, header: dict | None = None) -> pd.Series:
    """
    Convert a raw time column to UTC datetime.

    Handles:
    - ISO-8601 strings
    - Unix seconds (values >= 1e8, e.g. 1_700_000_000)
    - FITS MJD (Modified Julian Date): values in the range 40 000–99 999
    - Seconds since a FITS epoch (MJDREFI + MJDREFF in header)

    Physics note: Aditya-L1 uses TT (Terrestrial Time) for its time axis, which
    is ~32.184 s ahead of UTC.  This offset is small enough that for flare
    detection (minute-scale features) it can be ignored, but we document it here
    for completeness.
    """
    if pd.api.types.is_string_dtype(series):
        return pd.to_datetime(series, utc=True, errors="coerce")

    median_val = float(series.median())

    # Unix epoch seconds: 1970-01-01 → today is roughly 0–2e9
    if median_val >= 1e8:
        return pd.to_datetime(series, unit="s", utc=True, errors="coerce")

    # MJD: Modified Julian Date counts days since 1858-11-17.
    # Reasonable range for modern data: 40 000 (≈1968) – 99 999 (≈2132).
    if 40_000 <= median_val <= 99_999:
        try:
            from astropy.time import Time
            t = Time(series.values.astype(float), format="mjd", scale="utc")
            return pd.DatetimeIndex(t.to_datetime(timezone=None)).tz_localize("UTC")
        except Exception as exc:
            logger.warning("SoLEXS: MJD conversion failed (%s); falling back to Unix seconds.", exc)
            return pd.to_datetime(series, unit="s", utc=True, errors="coerce")

    # Seconds since a FITS reference epoch
    mjdrefi = (header or {}).get("MJDREFI", 0)
    mjdreff = (header or {}).get("MJDREFF", 0.0)
    mjd_ref = mjdrefi + mjdreff
    if mjd_ref > 0:
        try:
            from astropy.time import Time
            epoch = Time(float(mjd_ref), format="mjd", scale="utc")
            t = epoch + pd.to_timedelta(series, unit="s")
            return pd.DatetimeIndex(t.to_datetime(timezone=None)).tz_localize("UTC")
        except Exception as exc:
            logger.warning("SoLEXS: FITS epoch conversion failed (%s); falling back.", exc)

    # Last resort: treat as Unix seconds
    logger.warning("SoLEXS: treating time column as Unix seconds (no FITS epoch found).")
    return pd.to_datetime(series, unit="s", utc=True, errors="coerce")


def _load_fits(filepath: Path) -> tuple[pd.DataFrame, dict]:
    """Load data from a FITS file; returns (raw_df, header_dict)."""
    from astropy.io import fits

    with fits.open(filepath, memmap=True, ignore_missing_simple=True) as hdul:
        # Prefer a table extension named RATE or EVENTS; fall back to first table
        target_hdu = None
        for hdu in hdul:
            if isinstance(hdu, (fits.BinTableHDU, fits.TableHDU)):
                if hdu.name.upper() in ("RATE", "EVENTS", "LIGHTCURVE", "SXR"):
                    target_hdu = hdu
                    break
        if target_hdu is None:
            for hdu in hdul:
                if isinstance(hdu, (fits.BinTableHDU, fits.TableHDU)):
                    target_hdu = hdu
                    break
        if target_hdu is None:
            raise ValueError(f"No table HDU found in {filepath.name}. Run inspect_fits.py first.")

        header = dict(target_hdu.header)
        from astropy.table import Table
        df = Table(target_hdu.data).to_pandas()
        logger.debug("SoLEXS FITS: loaded HDU %r with %d rows from %s", target_hdu.name, len(df), filepath.name)
    return df, header


def _load_csv(filepath: Path) -> tuple[pd.DataFrame, dict]:
    """Load data from a CSV file; returns (raw_df, empty_header)."""
    df = pd.read_csv(filepath, comment="#")
    logger.debug("SoLEXS CSV: loaded %d rows from %s", len(df), filepath.name)
    return df, {}


# ── Public API ─────────────────────────────────────────────────────────────────

def read_solexs(
    filepath: str | Path,
    energy_low: float = _DEFAULT_ENERGY_LOW,
    energy_high: float = _DEFAULT_ENERGY_HIGH,
    energy_channel: str | None = None,
) -> pd.DataFrame:
    """
    Load a SoLEXS FITS or CSV file from PRADAN into the standard schema.

    Parameters
    ----------
    filepath : str | Path
        Path to the .fits or .csv file.
    energy_low : float
        Lower bound of the energy channel in keV (used if the file does not
        specify it).  Default: 1.0 keV.
    energy_high : float
        Upper bound of the energy channel in keV.  Default: 30.0 keV.
    energy_channel : str | None
        Human-readable label for the energy band.  Defaults to "1-30keV".

    Returns
    -------
    pd.DataFrame
        Standard schema with columns:
        timestamp, flux, flux_err, energy_low, energy_high,
        energy_channel, instrument, quality_flag.
        Rows are sorted by timestamp.  Rows where timestamp could not be
        parsed are dropped with a warning.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file contains no table data at all.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"SoLEXS file not found: {filepath}")

    suffix = filepath.suffix.lower()
    if suffix in (".fits", ".fit", ".fz", ".lc", ".pi", ".gti"):
        raw, header = _load_fits(filepath)
    elif suffix in (".csv", ".txt", ".dat"):
        raw, header = _load_csv(filepath)
    else:
        # Try FITS first, then CSV
        try:
            raw, header = _load_fits(filepath)
        except Exception:
            raw, header = _load_csv(filepath)

    cols = list(raw.columns)

    # ── Time ──────────────────────────────────────────────────────────────────
    time_col = _find_column(cols, _TIME_ALIASES)
    if time_col is None:
        raise ValueError(
            f"Cannot find a time column in {filepath.name}. "
            f"Available columns: {cols}. Update _TIME_ALIASES after running inspect_fits.py."
        )
    timestamps = _parse_time_column(raw[time_col], header)
    bad_time_mask = pd.isna(timestamps)
    if bad_time_mask.any():
        logger.warning("SoLEXS: dropping %d rows with unparseable timestamps.", bad_time_mask.sum())

    # ── Flux ──────────────────────────────────────────────────────────────────
    flux_col = _find_column(cols, _FLUX_ALIASES)
    if flux_col is None:
        logger.warning(
            "SoLEXS: no flux column found in %s. Available: %s. Setting flux=NaN.", filepath.name, cols
        )
        flux = pd.Series(np.nan, index=raw.index)
    else:
        flux = pd.to_numeric(raw[flux_col], errors="coerce")

    # ── Flux error ────────────────────────────────────────────────────────────
    err_col = _find_column(cols, _FLUX_ERR_ALIASES)
    if err_col:
        flux_err = pd.to_numeric(raw[err_col], errors="coerce")
    else:
        # Poisson approximation: σ ≈ √(counts) — only meaningful if flux is in counts/s
        flux_err = np.where(flux >= 0, np.sqrt(np.abs(flux)), np.nan)
        flux_err = pd.Series(flux_err, index=raw.index)

    # ── Quality flag ──────────────────────────────────────────────────────────
    qual_col = _find_column(cols, _QUALITY_ALIASES)
    if qual_col:
        quality_flag = raw[qual_col].fillna(0).astype(np.int8)
    else:
        quality_flag = pd.Series(np.int8(0), index=raw.index)

    # ── Assemble output ───────────────────────────────────────────────────────
    if energy_channel is None:
        energy_channel = f"{energy_low:.4g}-{energy_high:.4g}keV"

    out = pd.DataFrame(
        {
            "timestamp": timestamps,
            "flux": flux.values,
            "flux_err": flux_err.values,
            "energy_low": np.float32(energy_low),
            "energy_high": np.float32(energy_high),
            "energy_channel": energy_channel,
            "instrument": "SoLEXS",
            "quality_flag": quality_flag.values,
        }
    )
    out = out.loc[~bad_time_mask].sort_values("timestamp").reset_index(drop=True)

    logger.info(
        "SoLEXS: loaded %d good rows from %s  [%s → %s]",
        len(out),
        filepath.name,
        out["timestamp"].iloc[0] if len(out) else "N/A",
        out["timestamp"].iloc[-1] if len(out) else "N/A",
    )
    return out
