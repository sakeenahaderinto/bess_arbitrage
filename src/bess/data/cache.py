import os
import pandas as pd
from pathlib import Path
from bess.data.em_client import get_historical_prices

def get_prices_cached(zone: str, start_date: str, end_date: str, force_refresh: bool = False) -> pd.Series:
    """
    Retrieves historical, day-ahead prices for the given zone and date range, using 
    a local Parquet cache to avoid redundant API calls.
    
    Args:
        zone: The zone code (e.g. 'DE').
        start_date: Start date string in 'YYYY-MM-DD' format.
        end_date: End date string in 'YYYY-MM-DD' format.
        force_refresh: If True, bypasses the cache and re-fetches from the API.
        
    Returns:
        pd.Series: Hourly prices with a UTC DatetimeIndex.
    """
    data_dir = Path("data")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    filename = data_dir / f"{zone}_{start_date}_{end_date}.parquet"
    
    if filename.exists() and not force_refresh:
        print(f"[cache] Loaded from disk: {filename}")
        # pyarrow engine preserves the DatetimeIndex reliably
        df = pd.read_parquet(filename, engine='pyarrow')
        # Squeeze the single-column dataframe back into a Series
        return df.squeeze("columns")
        
    # Fetch from API
    series = get_historical_prices(zone, start_date, end_date)
    
    # Save to parquet. Pandas to_parquet works best with DataFrames, so convert to frame.
    df = series.to_frame(name="price")
    df.to_parquet(filename, engine='pyarrow', index=True)
    
    print(f"[cache] Fetched from API and saved: {filename}")
    return series

if __name__ == "__main__":
    # Little manual verification script
    zone = "DE"
    start = "2024-01-01"
    end = "2024-01-07"
    
    print("--- First Call ---")
    s1 = get_prices_cached(zone, start, end)
    print(f"Items: {len(s1)}, Index: {type(s1.index)}")
    print(s1.head(3))
    
    print("\n--- Second Call (Should load from disk) ---")
    s2 = get_prices_cached(zone, start, end)
    print(f"Items: {len(s2)}, Index: {type(s2.index)}")
    print(s2.head(3))
    
    print("\n--- Third Call (Should force refresh) ---")
    s3 = get_prices_cached(zone, start, end, force_refresh=True)
    print(f"Items: {len(s3)}, Index: {type(s3.index)}")
    print(s3.head(3))
