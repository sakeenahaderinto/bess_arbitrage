import pytest
import pandas as pd
import numpy as np

from bess.backtest.engine import run_backtest
from bess.forecast.naive import forecast_lag24

def test_backtest_output_shape_and_columns():
    """
    Test 1 - output shape: Run on 7 days of synthetic prices.
    Test 2 - all columns present: Assert the output has expected columns.
    """
    start_dt = pd.to_datetime("2024-01-01", utc=True)
    index = pd.date_range(start=start_dt, periods=24 * 7, freq="h", tz="UTC")
    synthetic_prices = pd.Series(np.random.rand(24 * 7) * 100, index=index)
    
    battery_params = {
        "dt_hours": 1.0,
        "e_max_mwh": 1.0,
        "p_max_mw": 1.0,
        "roundtrip_eff": 0.9,
        "deg_cost_per_mwh": 5.0
    }
    
    df = run_backtest(synthetic_prices, None, battery_params)
    
    # Assert Test 1
    assert len(df) == 7, "Output DataFrame should have exactly 7 rows"
    
    # Assert Test 2
    expected_cols = {"profit", "throughput_MWh", "active_hours", "equiv_full_cycles", "avg_spread"}
    assert expected_cols.issubset(df.columns), f"Should contain all expected columns, got {df.columns}"

def test_backtest_profit_positive_on_favourable_prices():
    """
    Test 3 - profit is non-negative on favourable prices: Run on a price series
    with a clear daily pattern (cheap at night, expensive at peak). 
    Assert that at least 80% of days have profit > 0.
    """
    start_dt = pd.to_datetime("2024-01-01", utc=True)
    index = pd.date_range(start=start_dt, periods=24 * 10, freq="h", tz="UTC")
    
    # Creating a sinusoidal price that clearly supports arbitrage
    hours = np.arange(len(index))
    prices = 50 + 40 * np.sin(2 * np.pi * (hours - 6) / 24)
    synthetic_prices = pd.Series(prices, index=index)
    
    battery_params = {
        "dt_hours": 1.0,
        "e_max_mwh": 1.0,
        "p_max_mw": 1.0,
        "roundtrip_eff": 0.9,
        "deg_cost_per_mwh": 2.0  # low deg cost to ensure profit
    }
    
    df = run_backtest(synthetic_prices, None, battery_params)
    
    profit_days = len(df[df["profit"] > 0])
    assert profit_days / len(df) >= 0.8, "At least 80% of days should be profitable"

def test_backtest_forecast_mode():
    """
    Test 4 - forecast mode runs without error: Pass forecast_lag24 as
    forecast_fn and assert the backtest completes and returns a DataFrame.
    """
    start_dt = pd.to_datetime("2024-01-01", utc=True)
    index = pd.date_range(start=start_dt, periods=24 * 3, freq="h", tz="UTC")
    synthetic_prices = pd.Series(np.random.rand(24 * 3) * 100, index=index)
    
    battery_params = {
        "dt_hours": 1.0,
        "e_max_mwh": 1.0,
        "p_max_mw": 1.0,
        "roundtrip_eff": 0.9,
        "deg_cost_per_mwh": 5.0
    }
    
    df = run_backtest(synthetic_prices, forecast_lag24, battery_params)
    
    # Expected rows:
    # 2024-01-01 -> skipped by forecast_lag24 due to no prior data
    # 2024-01-02 -> runs successfully 
    # 2024-01-03 -> runs successfully
    
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0, "Forecast backtest should produce at least one day of results"
