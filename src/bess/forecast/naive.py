import pandas as pd
from datetime import timedelta

def forecast_lag24(historical_prices: pd.Series, forecast_date: str) -> pd.Series:
    """
    Returns a 24-hour baseline price forecast for the target day by taking the
    actual prices from exactly 24 hours prior.
    
    Why it works: Daily electricity consumption patterns (and thus prices) repeat.
    People wake up, go to work, come home, and sleep at similar times, especially
    on weekdays.
    
    When it breaks: Transitioning into or out of weekends, bank holidays, or
    sudden extreme weather events (e.g. storms changing wind generation).
    
    Args:
        historical_prices: A pd.Series of past prices with a UTC DatetimeIndex.
                           Must contain the 24 hours immediately preceding the forecast_date.
        forecast_date: A string 'YYYY-MM-DD' representing the day to forecast.
        
    Returns:
        pd.Series: 24 hourly prices with a DatetimeIndex for the target day.
    """
    target_dt = pd.to_datetime(forecast_date, utc=True)
    prev_day_dt = target_dt - timedelta(days=1)
    
    # Extract the previous day's prices
    start_prev = prev_day_dt
    end_prev = prev_day_dt + timedelta(hours=23)
    
    prev_day_prices = historical_prices.loc[start_prev:end_prev].copy()
    
    if len(prev_day_prices) != 24:
        raise ValueError(f"Historical data missing/incomplete for lag-24 on {forecast_date}.")
        
    # Shift the index forward by 1 day to represent the forecast day
    prev_day_prices.index = prev_day_prices.index + timedelta(days=1)
    
    return prev_day_prices

def forecast_rolling7(historical_prices: pd.Series, forecast_date: str) -> pd.Series:
    """
    Returns a 24-hour baseline price forecast by taking the average price for each
    hour across the 7 days immediately preceding the target day.
    
    Why it works vs lag-24: Smoothes out single-day anomalies. If yesterday had a 
    freak plant outage driving prices to $1000/MWh, rolling7 won't assume today 
    will too.
    
    When it breaks vs lag-24: Slower to react to persistent regime changes (e.g. 
    a shift in seasons or a sustained cold snap). It averages weekends into weekdays.
    
    Args:
        historical_prices: A pd.Series of past prices with a UTC DatetimeIndex.
                           Must contain the 7 full days preceding the forecast_date.
        forecast_date: A string 'YYYY-MM-DD' representing the day to forecast.
        
    Returns:
        pd.Series: 24 hourly prices with a DatetimeIndex for the target day.
    """
    target_dt = pd.to_datetime(forecast_date, utc=True)
    
    # We need the previous 7 days of data
    start_prev7 = target_dt - timedelta(days=7)
    end_prev7 = target_dt - timedelta(minutes=1) # up to the end of yesterday
    
    past_week = historical_prices.loc[start_prev7:end_prev7].copy()
    
    if len(past_week) < 24 * 7:
        raise ValueError(f"Historical data missing/incomplete for rolling-7 on {forecast_date}.")
        
    # Create a DataFrame to easily group by hour of day
    df = past_week.to_frame(name="price")
    df['hour'] = df.index.hour
    
    # Calculate the mean for each hour
    hourly_avg = df.groupby('hour')['price'].mean()
    
    # Construct the forecast series with the target day's dates
    # Assuming hour matching: index 0 is midnight
    forecast_index = pd.date_range(start=target_dt, periods=24, freq="h", tz="UTC")
    forecast_series = pd.Series(hourly_avg.values, index=forecast_index)
    
    return forecast_series
