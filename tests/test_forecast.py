import pytest
import pandas as pd
import numpy as np
from bess.forecast.naive import forecast_lag24, forecast_rolling7

def test_forecast_lag24_output_shape():
    """Test 1 - lag24 output shape"""
    start_dt = pd.to_datetime("2024-01-01", utc=True)
    index = pd.date_range(start=start_dt, periods=24 * 7, freq="h", tz="UTC")
    synthetic_prices = pd.Series(np.random.rand(24 * 7) * 100, index=index)
    
    forecast_date = "2024-01-08"
    forecast = forecast_lag24(synthetic_prices, forecast_date)
    
    assert len(forecast) == 24
    assert isinstance(forecast.index, pd.DatetimeIndex)
    assert forecast.index[0] == pd.to_datetime("2024-01-08 00:00:00", utc=True)

def test_forecast_lag24_correctness():
    """Test 2 - lag24 correctness: yesterday's prices are all 10.0"""
    start_dt = pd.to_datetime("2024-01-06", utc=True) 
    index = pd.date_range(start=start_dt, periods=48, freq="h", tz="UTC")
    
    prices_day1 = np.random.rand(24) * 100
    prices_day2 = np.array([10.0] * 24)
    data = np.concatenate([prices_day1, prices_day2])
    
    synthetic_prices = pd.Series(data, index=index)
    
    forecast_date = "2024-01-08" # Requires 2024-01-07 which is prices_day2
    forecast = forecast_lag24(synthetic_prices, forecast_date)
    
    assert (forecast == 10.0).all()

def test_forecast_rolling7_output_shape():
    """Test 3 - rolling7 output shape"""
    start_dt = pd.to_datetime("2024-01-01", utc=True)
    index = pd.date_range(start=start_dt, periods=24 * 7, freq="h", tz="UTC")
    synthetic_prices = pd.Series(np.random.rand(24 * 7) * 100, index=index)
    
    forecast_date = "2024-01-08"
    forecast = forecast_rolling7(synthetic_prices, forecast_date)
    
    assert len(forecast) == 24
    assert isinstance(forecast.index, pd.DatetimeIndex)
    assert forecast.index[0] == pd.to_datetime("2024-01-08 00:00:00", utc=True)

def test_forecast_rolling7_correctness():
    """Test 4 - rolling7 correctness: 7 days of 50.0 averages to 50.0"""
    start_dt = pd.to_datetime("2024-01-01", utc=True)
    index = pd.date_range(start=start_dt, periods=24 * 7, freq="h", tz="UTC")
    
    synthetic_prices = pd.Series(50.0, index=index)
    
    forecast_date = "2024-01-08"
    forecast = forecast_rolling7(synthetic_prices, forecast_date)
    
    assert (forecast == 50.0).all()
