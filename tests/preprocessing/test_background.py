import pytest
import pandas as pd
import numpy as np
from src.preprocessing.background import subtract_background
from tests.fixtures.synthetic_lc import make_flare_lc


def test_subtract_background_adds_columns():
    df = make_flare_lc(duration_s=600)
    result = subtract_background(df, window_s=300)
    assert "background" in result.columns
    assert "flux_background_sub" in result.columns
    assert len(result) == len(df)


def test_subtract_background_non_negative():
    df = make_flare_lc(duration_s=600)
    result = subtract_background(df, window_s=300)
    assert (result["flux_background_sub"] >= 0).all()


def test_subtract_background_empty_df():
    df = pd.DataFrame()
    result = subtract_background(df)
    assert "background" in result.columns
    assert "flux_background_sub" in result.columns
    assert result.empty
