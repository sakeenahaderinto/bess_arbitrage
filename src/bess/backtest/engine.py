import logging
import pandas as pd
from typing import Callable
from bess.optimiser import battery_solve_arbitrage

logger = logging.getLogger(__name__)

def run_backtest(
    prices: pd.Series, 
    forecast_fn: Callable | None, 
    battery_params: dict, 
    progress_callback: Callable | None = None
) -> pd.DataFrame:
    """
    Runs the battery optimiser over every day in the provided price series.
    
    Args:
        prices: Historical price Series with a UTC DatetimeIndex.
        forecast_fn: A function `(prices, date_str) -> pd.Series` that provides
                     a 24-hour forecast. If None, uses actual prices (perfect foresight).
        battery_params: Dict of parameters for `battery_solve_arbitrage`.
        progress_callback: Optional function `(current_day_int, total_days_int)` 
                           to report progress back to the caller/UI.
                           
    Returns:
        pd.DataFrame: A row per day with performance metrics.
    """
    if prices.empty:
        return pd.DataFrame()
        
    # 1. Get the sorted list of unique dates in `prices`.
    # Convert exactly to python dates (which handles the daily boundary).
    unique_dates = sorted(list(set(prices.index.date)))
    
    results = []
    total_days = len(unique_dates)
    
    # 2. For each date:
    for i, current_date in enumerate(unique_dates):
        date_str = current_date.strftime("%Y-%m-%d")
        
        try:
            # a. Slice the 24-hour price window for that date from `prices`.
            # A string slice like 'YYYY-MM-DD' on a DatetimeIndex gives all rows for that day.
            actual_prices = prices.loc[date_str]
            
            # Skip incomplete days if they don't have exactly 24 hours 
            # (Though depending on DST this could be 23 or 25, we assume 24 for standard simplicity)
            if len(actual_prices) != 24:
                logger.warning(f"Skipping {date_str} due to incomplete data ({len(actual_prices)} hours).")
                continue
                
            # b. If `forecast_fn` is None, use actual prices (perfect foresight).
            #    If `forecast_fn` is provided, call it to get forecast prices.
            if forecast_fn is None:
                solve_prices = actual_prices.tolist()
            else:
                forecast_series = forecast_fn(prices, date_str)
                solve_prices = forecast_series.tolist()
                
            # c. Call `battery_solve_arbitrage` with the prices and `battery_params`.
            _, _, df_dispatch, profit = battery_solve_arbitrage(solve_prices, **battery_params)
            
            # d. Record: date, total profit, total MWh throughput, number of cycles, average spread
            throughput = df_dispatch["throughput_MWh"].sum()
            n_cycles = (df_dispatch["net_MW"] != 0).sum()
            
            # Rough average spread calculation:
            # Income from discharge per MWh vs Cost to charge per MWh
            # Handling division by zero if it didn't cycle
            charge_volume = df_dispatch["p_charge_MW"].sum()
            discharge_volume = df_dispatch["p_discharge_MW"].sum()
            
            if charge_volume > 0 and discharge_volume > 0:
                avg_charge_price = (df_dispatch["p_charge_MW"] * df_dispatch["price_$perMWh"]).sum() / charge_volume
                avg_discharge_price = (df_dispatch["p_discharge_MW"] * df_dispatch["price_$perMWh"]).sum() / discharge_volume
                avg_spread = avg_discharge_price - avg_charge_price
            else:
                avg_spread = 0.0
                
            results.append({
                "date": current_date,
                "profit": profit,
                "throughput_MWh": throughput,
                "n_cycles": n_cycles,
                "avg_spread": avg_spread,
                "solve_failed": False
            })
            
        except PermissionError:
            # Re-raise authentication / permission errors immediately
            raise
        except Exception as e:
            # e. Wrap steps b-d in a try/except. Log the date/error, record profit=0, and continue.
            logger.error(f"Solve failed for {date_str}: {e}")
            results.append({
                "date": current_date,
                "profit": 0.0,
                "throughput_MWh": 0.0,
                "n_cycles": 0,
                "avg_spread": 0.0,
                "solve_failed": True
            })
            
        # f. If `progress_callback` is provided, call it with current progress.
        if progress_callback:
            progress_callback(i + 1, total_days)

    # 3. Return a DataFrame with one row per date and the columns from step 2d.
    if not results:
        return pd.DataFrame()
        
    df_results = pd.DataFrame(results)
    df_results.set_index("date", inplace=True)
    return df_results
