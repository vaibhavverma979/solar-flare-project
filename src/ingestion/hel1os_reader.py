"""
hel1os_reader.py — HEL1OS (Hard X-ray Spectrometer) FITS/CSV loader.

Physics context
---------------
HEL1OS is a CdZnTe detector aboard Aditya-L1 covering the hard X-ray band
(~8–150 keV).  Hard X-ray emission during a solar flare is produced by
non-thermal electrons (bremsstrahlung) accelerated in the coronal magnetic
reconnection region.  The HXR light curve is crucial for two reasons:

1.  **Neupert effect**: In impulsive flares the soft X-ray (SXR) flux is
    approximately the time-integral of the hard X-ray flux (Neupert 1968).
    This means HEL1OS emission *precedes* SoLEXS emission and is therefore a
    strong precursor signal for forecasting.

2.  **Spectral hardness**: The ratio flux_HXR / flux_SXR encodes information
    about the non-thermal fraction of the flare energy — higher ratios indicate
    more energetic, impulsive events.

Output schema (same long format as SoLEXS reader)
--------------------------------------------------
timestamp       : datetime64[ns, UTC]   — event time, always UTC
flux            : float64               — counts s⁻¹
flux_err        : float64               — 1-σ uncertainty (NaN if unavailable)
energy_low      : float32               — lower bound of energy channel (keV)
energy_high     : float32               — upper bound of energy channel (keV)
energy_channel  : str                   — e.g. "8-150keV" or "8-25keV"
instrument      : str                   — always "HEL1OS"
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
# HEL1OS may use different column names than SoLEXS; update after inspect_fits.py
_TIME_ALIASES: list[str] = ["TIME", "Date_Time", "UTC_TIME", "MJD", "TT", "TIMESTAMP"]
_FLUX_ALIASES: list[str] = ["RATE", "COUNTS", "FLUX", "COUNT_RATE", "HXR_FLUX", "HEL1OS_RATE", "CTR"]
_FLUX_ERR_ALIASES: list[str] = ["RATE_ERR", "ERROR", "FLUX_ERR", "STAT_ERR", "ERR"]
_QUALITY_ALIASES: list[str] = ["QUALITY", "FLAG", "STATUS", "GOOD"]

# HEL1OS operates across several sub-bands; we store the full-band default here.
# When the file provides per-channel columns, read_hel1os() will return one row
# per channel per timestamp (long format), exactly like the SoLEXS reader.
_DEFAULT_ENERGY_LOW: float = 8.0    # keV
_DEFAULT_ENERGY_HIGH: float = 150.0  # keV

# Known HEL1OS sub-bands (populated from instrument documentation).
# Used to detect multi-channel files and build the long-format output.
_HEL1OS_SUBBANDS: list[tuple[float, float, str]] = [
    (8.0,   25.0,  "8-25keV"),
    (25.0,  50.0,  "25-50keV"),
    (50.0,  100.0, "50-100keV"),
    (100.0, 150.0, "100-150keV"),
]

# Column name prefixes that suggest per-band flux in HEL1OS files
_BAND_FLUX_PREFIXES = ("RATE_", "COUNTS_", "FLUX_", "CH_", "BAND_")


# ── Internal helpers (re-used from solexs_reader pattern) ─────────────────────

def _find_column(available: list[str], aliases: list[str]) -> Optional[str]:
    """Return the first alias found in *available* (case-insensitive)."""
    upper = {c.upper(): c for c in available}
    for alias in aliases:
        if alias.upper() in upper:
            return upper[alias.upper()]
    return None


def _parse_time_column(series: pd.Series, header: dict | None = None) -> pd.Series:
    """Convert raw time column to UTC datetime.

    Handles Unix epoch seconds (>= 1e8), MJD (40 000–99 999), FITS epoch offsets,
    and ISO-8601 strings.  Same logic as SoLEXS reader.
    """
    if pd.api.types.is_string_dtype(series):
        return pd.to_datetime(series, utc=True, errors="coerce")

    median_val = float(series.median())

    # Unix epoch seconds (1970-01-01 → today ≈ 1.7e9)
    if median_val >= 1e8:
        return pd.to_datetime(series, unit="s", utc=True, errors="coerce")

    # MJD: 40 000 (≈1968) – 99 999 (≈2132)
    if 40_000 <= median_val <= 99_999:
        try:
            from astropy.time import Time
            t = Time(series.values.astype(float), format="mjd", scale="utc")
            return pd.DatetimeIndex(t.to_datetime(timezone=None)).tz_localize("UTC")
        except Exception as exc:
            logger.warning("HEL1OS: MJD conversion failed (%s); falling back to Unix seconds.", exc)
            return pd.to_datetime(series, unit="s", utc=True, errors="coerce")

    # Seconds since FITS reference epoch
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
            logger.warning("HEL1OS: FITS epoch conversion failed (%s); falling back.", exc)

    logger.warning("HEL1OS: treating time column as Unix seconds (no FITS epoch found).")
    return pd.to_datetime(series, unit="s", utc=True, errors="coerce")


def _detect_band_columns(cols: list[str]) -> list[tuple[str, float, float, str]]:
    """
    Detect per-energy-band flux columns in a HEL1OS file.

    Returns a list of (col_name, energy_low, energy_high, label) tuples.
    Falls back to an empty list if no band structure is detected.
    """
    band_cols = []
    for col in cols:
        for prefix in _BAND_FLUX_PREFIXES:
            if col.upper().startswith(prefix):
                # Try to match against known sub-bands by index or keV range
                for i, (lo, hi, label) in enumerate(_HEL1OS_SUBBANDS):
                    if str(i) in col or label.replace("-", "_").upper() in col.upper():
                        band_cols.append((col, lo, hi, label))
                break
    return band_cols


def _load_fits(filepath: Path) -> tuple[pd.DataFrame, dict]:
    """Load data from a FITS file; returns (raw_df, header_dict)."""
    from astropy.io import fits

    with fits.open(filepath, memmap=True, ignore_missing_simple=True) as hdul:
        target_hdu = None
        for hdu in hdul:
            if isinstance(hdu, (fits.BinTableHDU, fits.TableHDU)):
                if hdu.name.upper() in ("RATE", "EVENTS", "HXR", "HEL1OS", "LIGHTCURVE"):
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
        logger.debug("HEL1OS FITS: loaded HDU %r with %d rows.", target_hdu.name, len(df))
    return df, header


def _load_csv(filepath: Path) -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(filepath, comment="#")
    logger.debug("HEL1OS CSV: loaded %d rows from %s.", len(df), filepath.name)
    return df, {}


# ── Public API ─────────────────────────────────────────────────────────────────

def read_hel1os(
    filepath: str | Path,
    energy_low: float = _DEFAULT_ENERGY_LOW,
    energy_high: float = _DEFAULT_ENERGY_HIGH,
    energy_channel: str | None = None,
) -> pd.DataFrame:
    """
    Load a HEL1OS FITS or CSV file from PRADAN into the standard schema.

    Parameters
    ----------
    filepath : str | Path
        Path to the .fits or .csv file.
    energy_low : float
        Lower bound of energy band (keV). Default: 10.0 keV.
    energy_high : float
        Upper bound of energy band (keV). Default: 150.0 keV.
    energy_channel : str | None
        Human-readable label for the energy band. Defaults to "10-150keV".

    Returns
    -------
    pd.DataFrame
        Standard schema (see module docstring).

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file contains no table data at all.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"HEL1OS file not found: {filepath}")

    suffix = filepath.suffix.lower()
    if suffix in (".fits", ".fit", ".fz"):
        raw, header = _load_fits(filepath)
    elif suffix in (".csv", ".txt", ".dat"):
        raw, header = _load_csv(filepath)
    else:
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
            f"Available columns: {cols}. Update _TIME_ALIASES after inspect_fits.py."
        )
    timestamps = _parse_time_column(raw[time_col], header)
    bad_time_mask = pd.isna(timestamps)
    if bad_time_mask.any():
        logger.warning("HEL1OS: dropping %d rows with unparseable timestamps.", bad_time_mask.sum())

    # ── Quality flag ──────────────────────────────────────────────────────────
    qual_col = _find_column(cols, _QUALITY_ALIASES)
    quality_flag = (
        raw[qual_col].fillna(0).astype(np.int8) if qual_col
        else pd.Series(np.int8(0), index=raw.index)
    )

    # ── Detect multi-band vs. single-band files ────────────────────────────────
    band_cols = _detect_band_columns(cols)

    if band_cols:
        # Multi-band: build long-format output by concatenating one frame per band
        logger.info("HEL1OS: detected %d energy sub-bands in %s.", len(band_cols), filepath.name)
        frames: list[pd.DataFrame] = []
        for col_name, lo, hi, label in band_cols:
            flux = pd.to_numeric(raw[col_name], errors="coerce")
            flux_err = np.where(flux >= 0, np.sqrt(np.abs(flux)), np.nan)
            frames.append(
                pd.DataFrame({
                    "timestamp": timestamps,
                    "flux": flux.values,
                    "flux_err": flux_err,
                    "energy_low": np.float32(lo),
                    "energy_high": np.float32(hi),
                    "energy_channel": label,
                    "instrument": "HEL1OS",
                    "quality_flag": quality_flag.values,
                })
            )
        out = pd.concat(frames, ignore_index=True)

    else:
        # Single (integrated) band
        flux_col = _find_column(cols, _FLUX_ALIASES)
        if flux_col is None:
            logger.warning(
                "HEL1OS: no flux column found in %s. Available: %s. Setting flux=NaN.", filepath.name, cols
            )
            flux = pd.Series(np.nan, index=raw.index)
        else:
            flux = pd.to_numeric(raw[flux_col], errors="coerce")

        err_col = _find_column(cols, _FLUX_ERR_ALIASES)
        flux_err = (
            pd.to_numeric(raw[err_col], errors="coerce") if err_col
            else pd.Series(np.where(flux >= 0, np.sqrt(np.abs(flux)), np.nan), index=raw.index)
        )

        if energy_channel is None:
            energy_channel = f"{energy_low:.4g}-{energy_high:.4g}keV"

        out = pd.DataFrame({
            "timestamp": timestamps,
            "flux": flux.values,
            "flux_err": flux_err.values,
            "energy_low": np.float32(energy_low),
            "energy_high": np.float32(energy_high),
            "energy_channel": energy_channel,
            "instrument": "HEL1OS",
            "quality_flag": quality_flag.values,
        })

    out = out.loc[~out["timestamp"].isna()].sort_values(["timestamp", "energy_low"]).reset_index(drop=True)

    logger.info(
        "HEL1OS: loaded %d good rows from %s  [%s → %s]",
        len(out),
        filepath.name,
        out["timestamp"].iloc[0] if len(out) else "N/A",
        out["timestamp"].iloc[-1] if len(out) else "N/A",
    )
    return out
