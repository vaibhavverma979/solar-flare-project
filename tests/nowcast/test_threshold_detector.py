import pytest
import pandas as pd
import numpy as np
from src.nowcast.threshold_detector import detect
from src.preprocessing.background import subtract_background
from tests.fixtures.synthetic_lc import make_flare_lc


def test_detect_flags_flare():
    df = make_flare_lc(duration_s=1800, amplitude=500.0, background=50.0)
    df_bg = subtract_background(df, window_s=300)
    cat = detect(df_bg, sigma=3.0, min_duration_s=10, refractory_s=300)

    assert not cat.empty
    assert "start_time" in cat.columns
    assert "peak_time" in cat.columns
    assert "peak_flux" in cat.columns
    assert "sigma_excess" in cat.columns
    assert cat["sigma_excess"].iloc[0] > 3.0


def test_detect_empty_df():
    df = pd.DataFrame()
    cat = detect(df)
    assert cat.empty
