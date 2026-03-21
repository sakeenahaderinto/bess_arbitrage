import logging
import pandas as pd

logger = logging.getLogger(__name__)

def efficiency_gap(forecast_revenue: float, perfect_foresight_revenue: float) -> float | None:
    """Calculates the efficiency gap between forecasted and perfect foresight revenue.

    The efficiency gap quantifies the proportion of maximum mathematically possible
    revenue captured by the forecast. A ratio of 1.0 indicates perfect capture,
    whereas a ratio of 0.8 indicates 20% of potential revenue was lost to suboptimal
    dispatch decisions.

    Args:
        forecast_revenue: The cumulative profit achieved evaluating forecasted prices.
        perfect_foresight_revenue: The theoretical maximum profit using actual prices.

    Returns:
        float | None: The captured revenue ratio (typically 0.0 to 1.0), or None if
                      perfect foresight revenue is precisely zero (undefined division).
    """
    if perfect_foresight_revenue == 0:
        logger.warning("perfect_foresight_revenue is 0. Returning None for efficiency gap.")
        return None
        
    return forecast_revenue / perfect_foresight_revenue


def daily_summary(backtest_df: pd.DataFrame) -> dict:
    """Aggregates daily backtest statistics into a comprehensive summary dictionary.

    Args:
        backtest_df: A DataFrame indexed by date containing at least a 'profit' column.

    Returns:
        dict: Aggregated metrics including total revenue, standard deviation,
              consistency ratio, and extreme daily observations.
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
                
    # Compile total revenue aggregating all daily operations.
    total_revenue = float(backtest_df["profit"].sum())
    
    # Evaluate the arithmetic mean of daily profits.
    daily_mean = backtest_df["profit"].mean()
    
    # Extrapolate annualized revenue assuming constant 365-day operation.
    annualised_revenue = daily_mean * 365
    
    # Compute standard deviation representing daily revenue volatility.
    daily_std = backtest_df["profit"].std()
    
    # Compute the consistency ratio (mean divided by standard deviation).
    # This serves as a pseudo risk-adjusted metric, distinct from a formal Sharpe ratio.
    if daily_std == 0 or pd.isna(daily_std):
        consistency_ratio = 0.0
    else:
        consistency_ratio = daily_mean / daily_std
        
    # Identify the highest extreme daily revenue.
    best_row = backtest_df.loc[backtest_df["profit"].idxmax()]
    best_day = {"date": best_row.name, "profit": best_row["profit"]}
    
    # Identify the lowest extreme daily revenue limit.
    worst_row = backtest_df.loc[backtest_df["profit"].idxmin()]
    worst_day = {"date": worst_row.name, "profit": worst_row["profit"]}
    
    # Quantify the proportion of days yielding strictly positive profit.
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
    """Forecasts the capital payback period measured in years.

    Args:
        capital_cost_gbp: The total baseline system capital expenditure in GBP.
        annualised_revenue: The projected yearly revenue derived from historical backtests.

    Returns:
        float | None: The estimated time in years required to recoup initial costs,
                      or None if the annualized revenue is non-positive.
    """
    if annualised_revenue <= 0:
        return None
        
    return capital_cost_gbp / annualised_revenue
