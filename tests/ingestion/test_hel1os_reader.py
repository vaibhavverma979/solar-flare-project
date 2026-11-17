"""
test_hel1os_reader.py — Unit tests for src/ingestion/hel1os_reader.py.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tests.fixtures.synthetic_lc import make_flare_lc

REQUIRED_COLUMNS = {
    "timestamp", "flux", "flux_err",
    "energy_low", "energy_high", "energy_channel",
    "instrument", "quality_flag",
}


def _make_hel1os_csv(n: int = 200, multi_band: bool = False) -> str:
    """Generate a minimal HEL1OS-style CSV."""
    lc = make_flare_lc(duration_s=n, cadence_s=1)
    ts_unix = lc["timestamp"].astype(np.int64) // 10**9

    if multi_band:
        # Wide format with 4 sub-band columns
        header = "TIME,RATE_0,RATE_1,RATE_2,RATE_3,QUALITY"
        rows = [header]
        for i in range(n):
            f = lc["flux"].iloc[i]
            rows.append(f"{ts_unix.iloc[i]},{f*0.4:.4f},{f*0.3:.4f},{f*0.2:.4f},{f*0.1:.4f},0")
    else:
        header = "TIME,HEL1OS_RATE,RATE_ERR,QUALITY"
        rows = [header]
        for i in range(n):
            rows.append(
                f"{ts_unix.iloc[i]},{lc['flux'].iloc[i]:.4f},{lc['flux_err'].iloc[i]:.4f},0"
            )
    return "\n".join(rows)


class TestReadHel1osCSV:

    def test_standard_csv_has_required_columns(self, tmp_path: Path):
        from src.ingestion.hel1os_reader import read_hel1os
        f = tmp_path / "hel1os.csv"
        f.write_text(_make_hel1os_csv())
        df = read_hel1os(f)
        assert REQUIRED_COLUMNS.issubset(df.columns)

    def test_instrument_label_is_hel1os(self, tmp_path: Path):
        from src.ingestion.hel1os_reader import read_hel1os
        f = tmp_path / "hel1os.csv"
        f.write_text(_make_hel1os_csv())
        df = read_hel1os(f)
        assert (df["instrument"] == "HEL1OS").all()

    def test_timestamps_are_utc(self, tmp_path: Path):
        from src.ingestion.hel1os_reader import read_hel1os
        f = tmp_path / "hel1os.csv"
        f.write_text(_make_hel1os_csv())
        df = read_hel1os(f)
        assert pd.api.types.is_datetime64_any_dtype(df["timestamp"])
        assert str(df["timestamp"].dt.tz) == "UTC"

    def test_output_sorted(self, tmp_path: Path):
        from src.ingestion.hel1os_reader import read_hel1os
        f = tmp_path / "hel1os.csv"
        f.write_text(_make_hel1os_csv())
        df = read_hel1os(f)
        assert df["timestamp"].is_monotonic_increasing

    def test_default_energy_band_set_correctly(self, tmp_path: Path):
        from src.ingestion.hel1os_reader import read_hel1os
        f = tmp_path / "hel1os.csv"
        f.write_text(_make_hel1os_csv())
        df = read_hel1os(f, energy_low=8.0, energy_high=150.0, energy_channel="8-150keV")
        assert (df["energy_low"] == np.float32(8.0)).all()
        assert (df["energy_high"] == np.float32(150.0)).all()
        assert (df["energy_channel"] == "8-150keV").all()

    def test_file_not_found_raises(self):
        from src.ingestion.hel1os_reader import read_hel1os
        with pytest.raises(FileNotFoundError):
            read_hel1os("/nonexistent/hel1os.fits")

    def test_quality_flag_dtype_is_int8(self, tmp_path: Path):
        from src.ingestion.hel1os_reader import read_hel1os
        f = tmp_path / "hel1os.csv"
        f.write_text(_make_hel1os_csv())
        df = read_hel1os(f)
        assert df["quality_flag"].dtype == np.int8

    def test_no_negative_flux_err(self, tmp_path: Path):
        from src.ingestion.hel1os_reader import read_hel1os
        f = tmp_path / "hel1os.csv"
        f.write_text(_make_hel1os_csv())
        df = read_hel1os(f)
        assert (df["flux_err"].dropna() >= 0).all()

    def test_row_count_matches_input(self, tmp_path: Path):
        from src.ingestion.hel1os_reader import read_hel1os
        n = 150
        f = tmp_path / "hel1os.csv"
        f.write_text(_make_hel1os_csv(n=n))
        df = read_hel1os(f)
        assert len(df) == n
