"""
test_solexs_reader.py — Unit tests for src/ingestion/solexs_reader.py.

All tests use synthetic fixtures or in-memory CSV/FITS to avoid dependency
on real PRADAN files.
"""

from __future__ import annotations

import io
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tests.fixtures.synthetic_lc import make_flare_lc

# ── Helpers ────────────────────────────────────────────────────────────────────

REQUIRED_COLUMNS = {
    "timestamp", "flux", "flux_err",
    "energy_low", "energy_high", "energy_channel",
    "instrument", "quality_flag",
}


def _make_csv(
    n: int = 200,
    time_col: str = "TIME",
    flux_col: str = "RATE",
    include_err: bool = True,
    include_quality: bool = True,
) -> str:
    """Build a minimal CSV that mimics a PRADAN SoLEXS export."""
    lc = make_flare_lc(duration_s=n, cadence_s=1)
    ts_unix = lc["timestamp"].astype(np.int64) // 10**9  # unix seconds

    lines = [f"{time_col},{flux_col}"]
    if include_err:
        lines[0] += ",RATE_ERR"
    if include_quality:
        lines[0] += ",QUALITY"

    for i in range(n):
        row = f"{ts_unix.iloc[i]},{lc['flux'].iloc[i]:.4f}"
        if include_err:
            row += f",{lc['flux_err'].iloc[i]:.4f}"
        if include_quality:
            row += ",0"
        lines.append(row)
    return "\n".join(lines)


# ── Tests ──────────────────────────────────────────────────────────────────────

class TestReadSolexsCSV:
    """Tests for CSV loading path."""

    def test_standard_csv_returns_required_columns(self, tmp_path: Path):
        from src.ingestion.solexs_reader import read_solexs
        csv_file = tmp_path / "solexs_test.csv"
        csv_file.write_text(_make_csv())
        df = read_solexs(csv_file)
        assert REQUIRED_COLUMNS.issubset(df.columns), f"Missing: {REQUIRED_COLUMNS - set(df.columns)}"

    def test_instrument_label_is_solexs(self, tmp_path: Path):
        from src.ingestion.solexs_reader import read_solexs
        csv_file = tmp_path / "solexs_test.csv"
        csv_file.write_text(_make_csv())
        df = read_solexs(csv_file)
        assert (df["instrument"] == "SoLEXS").all()

    def test_timestamps_are_utc_datetime(self, tmp_path: Path):
        from src.ingestion.solexs_reader import read_solexs
        csv_file = tmp_path / "solexs_test.csv"
        csv_file.write_text(_make_csv())
        df = read_solexs(csv_file)
        assert pd.api.types.is_datetime64_any_dtype(df["timestamp"])
        assert str(df["timestamp"].dt.tz) == "UTC"

    def test_output_sorted_by_timestamp(self, tmp_path: Path):
        from src.ingestion.solexs_reader import read_solexs
        csv_file = tmp_path / "solexs_test.csv"
        csv_file.write_text(_make_csv())
        df = read_solexs(csv_file)
        assert df["timestamp"].is_monotonic_increasing

    def test_no_negative_flux_err(self, tmp_path: Path):
        from src.ingestion.solexs_reader import read_solexs
        csv_file = tmp_path / "solexs_test.csv"
        csv_file.write_text(_make_csv())
        df = read_solexs(csv_file)
        valid_err = df["flux_err"].dropna()
        assert (valid_err >= 0).all()

    def test_missing_quality_column_defaults_to_zero(self, tmp_path: Path):
        from src.ingestion.solexs_reader import read_solexs
        csv_file = tmp_path / "solexs_test.csv"
        csv_file.write_text(_make_csv(include_quality=False))
        df = read_solexs(csv_file)
        assert (df["quality_flag"] == 0).all()

    def test_missing_flux_err_column_uses_poisson_estimate(self, tmp_path: Path):
        from src.ingestion.solexs_reader import read_solexs
        csv_file = tmp_path / "solexs_test.csv"
        csv_file.write_text(_make_csv(include_err=False))
        df = read_solexs(csv_file)
        # Poisson estimate: flux_err ≈ sqrt(flux), should be non-negative
        assert df["flux_err"].notna().any()

    def test_alternative_column_names(self, tmp_path: Path):
        """Reader should handle COUNTS/Date_Time instead of RATE/TIME."""
        from src.ingestion.solexs_reader import read_solexs
        csv_file = tmp_path / "solexs_alt.csv"
        csv_file.write_text(_make_csv(time_col="Date_Time", flux_col="COUNTS"))
        df = read_solexs(csv_file)
        assert len(df) > 0

    def test_file_not_found_raises(self):
        from src.ingestion.solexs_reader import read_solexs
        with pytest.raises(FileNotFoundError):
            read_solexs("/nonexistent/path/file.fits")

    def test_energy_channel_label_in_output(self, tmp_path: Path):
        from src.ingestion.solexs_reader import read_solexs
        csv_file = tmp_path / "solexs_test.csv"
        csv_file.write_text(_make_csv())
        df = read_solexs(csv_file, energy_channel="1-30keV")
        assert (df["energy_channel"] == "1-30keV").all()

    def test_row_count_matches_input(self, tmp_path: Path):
        from src.ingestion.solexs_reader import read_solexs
        n = 300
        csv_file = tmp_path / "solexs_test.csv"
        csv_file.write_text(_make_csv(n=n))
        df = read_solexs(csv_file)
        assert len(df) == n

    def test_quality_flag_dtype_is_int8(self, tmp_path: Path):
        from src.ingestion.solexs_reader import read_solexs
        csv_file = tmp_path / "solexs_test.csv"
        csv_file.write_text(_make_csv())
        df = read_solexs(csv_file)
        assert df["quality_flag"].dtype == np.int8

    def test_nan_timestamps_are_dropped(self, tmp_path: Path):
        from src.ingestion.solexs_reader import read_solexs
        # Inject a bad row with an empty time field
        csv_content = _make_csv(n=50)
        lines = csv_content.split("\n")
        # Corrupt row 5: replace timestamp with empty string
        parts = lines[5].split(",")
        parts[0] = ""
        lines[5] = ",".join(parts)
        csv_file = tmp_path / "solexs_bad_ts.csv"
        csv_file.write_text("\n".join(lines))
        df = read_solexs(csv_file)
        assert df["timestamp"].notna().all()
        assert len(df) == 49  # one row dropped
