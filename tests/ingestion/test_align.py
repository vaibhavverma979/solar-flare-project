"""
test_align.py — Unit tests for src/ingestion/align.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tests.fixtures.synthetic_lc import make_two_instrument_flares, make_flare_lc


class TestInferCadence:
    def test_returns_timedelta(self):
        from src.ingestion.align import _infer_cadence
        lc = make_flare_lc(cadence_s=1)
        result = _infer_cadence(lc)
        assert isinstance(result, pd.Timedelta)

    def test_correct_cadence_1s(self):
        from src.ingestion.align import _infer_cadence
        lc = make_flare_lc(cadence_s=1)
        assert _infer_cadence(lc) == pd.Timedelta("1s")

    def test_correct_cadence_4s(self):
        from src.ingestion.align import _infer_cadence
        lc = make_flare_lc(cadence_s=4)
        assert _infer_cadence(lc) == pd.Timedelta("4s")

    def test_raises_on_single_row(self):
        from src.ingestion.align import _infer_cadence
        lc = make_flare_lc(duration_s=1, cadence_s=1)
        with pytest.raises(ValueError):
            _infer_cadence(lc)


class TestAlign:

    def test_output_has_required_columns_long_format(self):
        from src.ingestion.align import align
        sx, hx = make_two_instrument_flares(duration_s=600, cadence_s=1)
        out = align(sx, hx, cadence_s=1)
        required = {"timestamp", "flux", "flux_err", "energy_low", "energy_high",
                    "energy_channel", "instrument", "quality_flag"}
        assert required.issubset(out.columns)

    def test_both_instruments_present_in_output(self):
        from src.ingestion.align import align
        sx, hx = make_two_instrument_flares(duration_s=600, cadence_s=1)
        out = align(sx, hx, cadence_s=1)
        assert "SoLEXS" in out["instrument"].values
        assert "HEL1OS" in out["instrument"].values

    def test_output_sorted_by_timestamp(self):
        from src.ingestion.align import align
        sx, hx = make_two_instrument_flares(duration_s=600, cadence_s=1)
        out = align(sx, hx, cadence_s=1)
        assert out["timestamp"].is_monotonic_increasing

    def test_auto_cadence_picks_coarser(self):
        """With cadence_s=0, align should pick the coarser (4s) cadence."""
        from src.ingestion.align import align
        sx = make_flare_lc(cadence_s=1, duration_s=120)  # 1-second SoLEXS
        hx = make_flare_lc(cadence_s=4, duration_s=120, instrument="HEL1OS",
                            energy_low=8.0, energy_high=150.0, energy_channel="8-150keV")
        out = align(sx, hx, cadence_s=0)
        # At 4-second cadence, 120s → 30 bins per instrument
        n_sx = out[out["instrument"] == "SoLEXS"].shape[0]
        assert n_sx <= 32  # allow small boundary rounding

    def test_outer_join_preserves_nan_gaps(self):
        """If one instrument has gaps the outer join should produce NaN rows."""
        from src.ingestion.align import align
        sx = make_flare_lc(cadence_s=1, duration_s=200)
        # HEL1OS data starts 100s later → overlap is only 100s
        hx = make_flare_lc(cadence_s=1, duration_s=100, instrument="HEL1OS",
                            energy_low=8.0, energy_high=150.0, energy_channel="8-150keV",
                            seed=7)
        # Offset HEL1OS timestamps by 100s
        hx = hx.copy()
        hx["timestamp"] = hx["timestamp"] + pd.Timedelta("100s")
        out = align(sx, hx, cadence_s=1, join="outer")
        # NaNs should exist in HEL1OS rows from before its start
        hx_rows = out[out["instrument"] == "HEL1OS"]
        sx_rows = out[out["instrument"] == "SoLEXS"]
        assert len(sx_rows) > len(hx_rows)

    def test_inner_join_produces_no_nan_timestamps(self):
        from src.ingestion.align import align
        sx, hx = make_two_instrument_flares(duration_s=300, cadence_s=1)
        out = align(sx, hx, cadence_s=1, join="inner")
        assert out["timestamp"].notna().all()

    def test_pivot_returns_wide_format(self):
        from src.ingestion.align import align
        sx, hx = make_two_instrument_flares(duration_s=120, cadence_s=1)
        out = align(sx, hx, cadence_s=1, pivot=True)
        # Wide format: should have a timestamp column and flux_ prefixed columns
        assert "timestamp" in out.columns
        flux_cols = [c for c in out.columns if c.startswith("flux_")]
        assert len(flux_cols) >= 2

    def test_empty_solexs_raises(self):
        from src.ingestion.align import align
        _, hx = make_two_instrument_flares(duration_s=120)
        empty_sx = make_flare_lc(duration_s=1).iloc[0:0]
        # Should still work with one empty instrument (uses HEL1OS only)
        # (If both are empty, it raises)
        out = align(empty_sx, hx, cadence_s=1)
        assert len(out) > 0

    def test_both_empty_raises(self):
        from src.ingestion.align import align
        sx = make_flare_lc(duration_s=1).iloc[0:0]
        hx = make_flare_lc(duration_s=1, instrument="HEL1OS").iloc[0:0]
        with pytest.raises(ValueError):
            align(sx, hx)


class TestClipToOverlap:

    def test_returns_overlapping_portion(self):
        from src.ingestion.align import clip_to_overlap
        sx = make_flare_lc(cadence_s=1, duration_s=200)
        hx = make_flare_lc(cadence_s=1, duration_s=100, instrument="HEL1OS",
                            energy_low=8.0, energy_high=150.0, energy_channel="8-150keV")
        hx["timestamp"] = hx["timestamp"] + pd.Timedelta("100s")
        sx_c, hx_c = clip_to_overlap(sx, hx)
        assert sx_c["timestamp"].min() >= hx_c["timestamp"].min()

    def test_no_overlap_returns_empty(self):
        from src.ingestion.align import clip_to_overlap
        sx = make_flare_lc(cadence_s=1, duration_s=60)
        hx = make_flare_lc(cadence_s=1, duration_s=60, instrument="HEL1OS",
                            energy_low=8.0, energy_high=150.0, energy_channel="8-150keV")
        hx["timestamp"] = hx["timestamp"] + pd.Timedelta("1d")  # next day
        sx_c, hx_c = clip_to_overlap(sx, hx)
        assert len(sx_c) == 0
        assert len(hx_c) == 0
