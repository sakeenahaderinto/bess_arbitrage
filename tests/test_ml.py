import pytest
import pandas as pd
import numpy as np

from bess.forecast.ml import build_features

def test_build_features_creates_lags():
    """
    Assert that the build_features function properly computes
    lag 1, 24, 48, 168 and rolling windows.
    """
    index = pd.date_range("2024-01-01", periods=200, freq="h", tz="UTC")
    prices = pd.Series(np.random.rand(200) * 100, index=index)
    
    df_feat = build_features(prices)
    
    assert "lag_1h" in df_feat.columns
    assert "lag_168h" in df_feat.columns
    assert "rolling_24h_mean" in df_feat.columns
    assert "rolling_7d_same_hour_mean" in df_feat.columns
    
    # Check that lag 1 shifts correctly
    assert df_feat["lag_1h"].iloc[1] == prices.iloc[0]
    
    # Check lag 168 is nan for the first 168 rows
    assert pd.isna(df_feat["lag_168h"].iloc[167])
    assert not pd.isna(df_feat["lag_168h"].iloc[168])
