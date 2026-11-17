import pytest
import pandas as pd
import numpy as np
from src.preprocessing.cleaning import clean
from tests.fixtures.synthetic_lc import make_flare_lc


def test_clean_adds_columns_and_filters_quality():
    df = make_flare_lc(duration_s=300)
    df.loc[10:15, "quality_flag"] = 1  # mark bad quality
    result = clean(df, dead_time_s=2.5e-6)

    assert "flux_deadtime_corr" in result.columns
    assert "is_valid" in result.columns
    assert pd.isna(result.loc[12, "flux"])
    assert not result.loc[12, "is_valid"]
    assert result.loc[0, "is_valid"]


def test_clean_deadtime_correction_increases_flux():
    df = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=5, freq="1s"),
        "flux": [1000.0, 50000.0, 100000.0, 50000.0, 1000.0],
        "quality_flag": [0, 0, 0, 0, 0]
    })
    result = clean(df, dead_time_s=2.5e-6)
    assert result.loc[2, "flux_deadtime_corr"] > result.loc[2, "flux"]


def test_clean_empty_df():
    df = pd.DataFrame()
    result = clean(df)
    assert "flux_deadtime_corr" in result.columns
    assert "is_valid" in result.columns
    assert result.empty
