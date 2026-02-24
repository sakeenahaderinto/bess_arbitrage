import logging
import pandas as pd

logger = logging.getLogger(__name__)

def efficiency_gap(forecast_revenue: float, perfect_foresight_revenue: float) -> float | None:
    """
    Calculates the efficiency gap: how much value the forecast captured compared 
    to perfect foresight.
    
    A value of 1.0 means the forecast achieved the maximum mathematically possible 
    revenue exactly as perfect foresight would have. A value of 0.8 means it left 
    20% of the potential revenue on the table due to suboptimal dispatch decisions.
    
    Args:
        forecast_revenue: The total profit achieved using the forecasted prices.
        perfect_foresight_revenue: The total profit achieved using actual prices.
        
    Returns:
        float: A ratio (typically 0.0 to 1.0). Returns None if perfect foresight
               revenue is exactly zero, as division by zero is undefined.
    """
    if perfect_foresight_revenue == 0:
        logger.warning("perfect_foresight_revenue is 0. Returning None for efficiency gap.")
        return None
        
    return forecast_revenue / perfect_foresight_revenue


def daily_summary(backtest_df: pd.DataFrame) -> dict:
    """
    Summarizes the daily backtest results into a dictionary of key metrics.
    
    Args:
        backtest_df: A DataFrame containing at minimum a 'profit' column and 
                     indexed by date (one row per day).
                     
    Returns:
        dict: Summary statistics including total revenue, mean, std, sharpe, etc.
    """
    if backtest_df.empty or "profit" not in backtest_df.columns:
        return {}
        
    failed_days = 0
    if "solve_failed" in backtest_df.columns:
        failed_mask = backtest_df["solve_failed"] == True
        failed_days = int(failed_mask.sum())
        if failed_days > 0:
            logger.warning(f"Excluding {failed_days} days from summary where solve_failed was True.")
            backtest_df = backtest_df[~failed_mask]
            if backtest_df.empty:
                return {"failed_days": failed_days}
                
    # Sum of all daily profits
    total_revenue = float(backtest_df["profit"].sum())
    
    # Mean daily profit calculated directly
    daily_mean = backtest_df["profit"].mean()
    
    # Annualized based on the mean daily revenue * 365
    annualised_revenue = daily_mean * 365
    
    # Standard deviation of the daily profit
    daily_std = backtest_df["profit"].std()
    
    # Consistency ratio: mean daily profit divided by its standard deviation.
    # Note: This is NOT a true Sharpe ratio as there is no risk-free rate and it is not annualized.
    if daily_std == 0 or pd.isna(daily_std):
        consistency_ratio = 0.0
    else:
        consistency_ratio = daily_mean / daily_std
        
    # Row corresponding to the highest revenue
    best_row = backtest_df.loc[backtest_df["profit"].idxmax()]
    best_day = {"date": best_row.name, "profit": best_row["profit"]}
    
    # Row corresponding to the lowest revenue
    worst_row = backtest_df.loc[backtest_df["profit"].idxmin()]
    worst_day = {"date": worst_row.name, "profit": worst_row["profit"]}
    
    # Percentage of days that were strictly profitable
    num_days = len(backtest_df)
    profitable_days = len(backtest_df[backtest_df["profit"] > 0])
    pct_profitable = (profitable_days / num_days) * 100 if num_days > 0 else 0.0
    
    return {
        "total_revenue": total_revenue,
        "annualised_revenue": annualised_revenue,
        "daily_mean": daily_mean,
        "daily_std": daily_std,
        "consistency_ratio": consistency_ratio,
        "best_day": best_day,
        "worst_day": worst_day,
        "pct_profitable": pct_profitable,
        "failed_days": failed_days,
        "annualised_revenue_basis_days": num_days
    }


def payback_period(capital_cost_gbp: float, annualised_revenue: float) -> float | None:
    """
    Calculates the payback period in years.
    
    Args:
        capital_cost_gbp: Total baseline system cost.
        annualised_revenue: Expected yearly revenue based on historical backtests.
        
    Returns:
        float: Years to payback the capital cost. Returns None if expected 
               revenue is zero or negative.
    """
    if annualised_revenue <= 0:
        return None
        
    return capital_cost_gbp / annualised_revenue
