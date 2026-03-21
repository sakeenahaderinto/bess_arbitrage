import pandas as pd
from datetime import timedelta

def forecast_lag24(historical_prices: pd.Series, forecast_date: str) -> pd.Series:
    """Generates a 24-hour baseline price forecast using a simple 24-hour lag.

    This naive approach assumes daily electricity price patterns repeat. While
    effective for contiguous weekdays, it exhibits degraded performance during
    weekend transitions, bank holidays, or sudden extreme weather events.

    Args:
        historical_prices: A pd.Series of past prices with a UTC DatetimeIndex.
                           Must contain the 24 hours immediately preceding `forecast_date`.
        forecast_date: The target forecast date in 'YYYY-MM-DD' format.

    Returns:
        pd.Series: A 24-hour sequence of hourly prices corresponding to the target date.
    """
    target_dt = pd.to_datetime(forecast_date, utc=True)
    prev_day_dt = target_dt - timedelta(days=1)
    
    # Extract historical prices for the preceding 24-hour period.
    start_prev = prev_day_dt
    end_prev = prev_day_dt + timedelta(hours=23)
    
    prev_day_prices = historical_prices.loc[start_prev:end_prev].copy()
    
    if len(prev_day_prices) != 24:
        raise ValueError(f"Historical data missing/incomplete for lag-24 on {forecast_date}.")
        
    # Align the time index forward by one day to represent the forecast horizon.
    prev_day_prices.index = prev_day_prices.index + timedelta(days=1)
    
    return prev_day_prices

def forecast_rolling7(historical_prices: pd.Series, forecast_date: str) -> pd.Series:
    """Generates a 24-hour baseline price forecast using a 7-day rolling average.

    Calculates the average price for every hour of the day across the preceding
    7 days. This approach smooths out single-day anomalies (e.g., freak plant
    outages) compared to the lag-24 method, though it reacts more slowly to
    persistent regime changes and averages weekend effects into weekdays.

    Args:
        historical_prices: A pd.Series of past prices with a UTC DatetimeIndex.
                           Must contain the equivalent of 7 full days prior to `forecast_date`.
        forecast_date: The target forecast date in 'YYYY-MM-DD' format.

    Returns:
        pd.Series: A 24-hour sequence of hourly prices corresponding to the target date.
    """
    target_dt = pd.to_datetime(forecast_date, utc=True)
    
    # Isolate the prior 7-day period up to the final minute of the preceding day.
    start_prev7 = target_dt - timedelta(days=7)
    end_prev7 = target_dt - timedelta(minutes=1)
    
    past_week = historical_prices.loc[start_prev7:end_prev7].copy()
    
    if len(past_week) < 24 * 7:
        raise ValueError(f"Historical data missing/incomplete for rolling-7 on {forecast_date}.")
        
    # Convert to DataFrame to facilitate hour-of-day grouping operations.
    df = past_week.to_frame(name="price")
    df['hour'] = df.index.hour
    
    # Compute the mean price for each of the 24 hours.
    hourly_avg = df.groupby('hour')['price'].mean()
    
    # Construct and align the final forecast series over the target diurnal period.
    forecast_index = pd.date_range(start=target_dt, periods=24, freq="h", tz="UTC")
    forecast_series = pd.Series(hourly_avg.values, index=forecast_index)
    
    return forecast_series
