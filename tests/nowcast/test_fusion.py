import pytest
import pandas as pd
from src.nowcast.fusion import neupert_consistent, match_detections, build_master_catalogue
from tests.fixtures.synthetic_lc import make_two_instrument_flares
from src.preprocessing.background import subtract_background
from src.nowcast.threshold_detector import detect


def test_neupert_consistent():
    t0 = pd.Timestamp("2026-01-01T00:00:00Z")
    t_hxr = pd.Timestamp("2026-01-01T00:01:00Z")  # 60s lead
    t_sxr = pd.Timestamp("2026-01-01T00:02:00Z")
    assert neupert_consistent(sx_start=t_sxr, hx_start=t_hxr, tolerance_s=120)


def test_match_and_master_catalogue():
    sx_df, hx_df = make_two_instrument_flares(duration_s=1800)
    sx_bg = subtract_background(sx_df, window_s=300)
    hx_bg = subtract_background(hx_df, window_s=300)

    sx_cat = detect(sx_bg, sigma=3.0)
    hx_cat = detect(hx_bg, sigma=3.0)

    matched = match_detections(sx_cat, hx_cat, window_s=300)
    master = build_master_catalogue(matched)

    assert not master.empty
    assert "event_id" in master.columns
    assert "class" in master.columns
    assert "confidence" in master.columns
