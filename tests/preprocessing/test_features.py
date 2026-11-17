import pytest
import pandas as pd
import numpy as np
from src.preprocessing.features import build_features
from tests.fixtures.synthetic_lc import make_flare_lc


def test_build_features_adds_expected_columns():
    df = make_flare_lc(duration_s=600)
    df["background"] = 100.0
    result = build_features(df)

    assert "log_flux" in result.columns
    assert "df_dt" in result.columns
    assert "d2f_dt2" in result.columns
    assert "rolling_mean_1min" in result.columns
    assert "rolling_std_5min" in result.columns
    assert "rolling_max_15min" in result.columns


def test_build_features_hardness_ratio():
    timestamps = pd.date_range("2026-01-01", periods=10, freq="1s")
    df = pd.DataFrame({
        "timestamp": timestamps,
        "flux_SoLEXS": [100.0] * 10,
        "flux_HEL1OS": [50.0] * 10,
    })
    result = build_features(df)
    assert "hardness_ratio" in result.columns
    np.testing.assert_allclose(result["hardness_ratio"].values, 0.5, rtol=1e-3)


def test_build_features_empty_df():
    df = pd.DataFrame()
    result = build_features(df)
    assert result.empty
