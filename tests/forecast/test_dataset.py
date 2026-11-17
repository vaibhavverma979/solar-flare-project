import pytest
import pandas as pd
import numpy as np
from src.forecast.dataset import build_sequences
from tests.fixtures.synthetic_lc import make_flare_lc
from src.preprocessing.features import build_features


def test_build_sequences():
    df = make_flare_lc(duration_s=3600)
    df_feat = build_features(df)
    catalogue = pd.DataFrame([{
        "event_id": "FLARE_0001",
        "start_time": pd.Timestamp("2024-01-01T00:15:00Z"),
        "peak_time": pd.Timestamp("2024-01-01T00:16:00Z"),
        "end_time": pd.Timestamp("2024-01-01T00:20:00Z"),
        "class": "M",
    }])

    X, y, ts = build_sequences(df_feat, catalogue, window_min=10, horizon_min=15, stride_min=2)

    assert isinstance(X, np.ndarray)
    assert len(X) == len(y) == len(ts)
    if len(y) > 0:
        assert set(np.unique(y)).issubset({0, 1})
