import logging
import pandas as pd
from typing import Callable
from bess.optimiser import battery_solve_arbitrage

logger = logging.getLogger(__name__)

FAILED_DAY = {
    "profit": 0.0,
    "throughput_MWh": 0.0,
    "active_hours": 0,
    "equiv_full_cycles": 0.0,
    "avg_spread": 0.0,
    "solve_failed": True
}

def run_backtest(
    prices: pd.Series,
    forecast_fn: Callable | None,
    battery_params: dict,
    progress_callback: Callable | None = None
) -> pd.DataFrame:
    """
    Runs the battery optimiser over every day in the provided price series.

    Note: Days containing fewer or more than exactly 24 hours are skipped. This
    natively drops dates where Daylight Saving Time (DST) transitions produce
    23- or 25-hour days. The prices index is converted to UTC prior to processing
    to minimise these discrepancies. Days with NaN prices are also skipped.

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

    # Convert to UTC for consistent day boundaries
    if prices.index.tz is None:
        prices.index = prices.index.tz_localize("UTC")
    else:
        prices.index = prices.index.tz_convert("UTC")

    # Enforce strict hourly grid once on the full series before the loop.
    # This prevents asfreq('h') inside forecast functions from reindexing
    # a growing slice on every iteration, which is O(N) per day.
    prices = prices.asfreq('h')

    unique_dates = sorted(list(set(prices.index.date)))

    results = []
    total_days = len(unique_dates)

    for i, current_date in enumerate(unique_dates):
        date_str = current_date.strftime("%Y-%m-%d")

        try:
            actual_prices = prices.loc[date_str]

            # Skip days with NaN prices — these come from asfreq('h') filling
            # gaps in the raw data and would produce degenerate LP solutions.
            if actual_prices.isnull().any():
                logger.warning(f"Skipping {date_str} due to NaN prices.")
                continue

            # Skip incomplete days — DST transitions can produce 23 or 25 hour days
            if len(actual_prices) != 24:
                logger.warning(f"Skipping {date_str} due to incomplete data ({len(actual_prices)} hours).")
                continue

            if forecast_fn is None:
                solve_prices = actual_prices.tolist()
            else:
                forecast_series = forecast_fn(prices, date_str)

                # Validate forecast length and date alignment
                if len(forecast_series) != 24 or forecast_series.index[0].date() != current_date:
                    logger.warning(
                        f"Forecast for {date_str} returned invalid shape or date "
                        f"(length: {len(forecast_series)}). Marking as failed."
                    )
                    results.append({"date": current_date, **FAILED_DAY})
                    continue

                # Validate forecast has no NaN values
                if forecast_series.isnull().any():
                    logger.warning(f"Forecast for {date_str} contains NaN values. Marking as failed.")
                    results.append({"date": current_date, **FAILED_DAY})
                    continue

                solve_prices = forecast_series.tolist()

            logger.info(f"Solving {date_str} with prices: min={min(solve_prices):.2f} max={max(solve_prices):.2f}")

            _, _, df_dispatch, profit = battery_solve_arbitrage(solve_prices, **battery_params)

            logger.info(f"Solved {date_str}: profit={profit:.2f}")

            # Recalculate realised profit against actual prices when using a forecast.
            # The optimiser's returned profit is calculated against forecast prices,
            # not actual prices as this is what would really have been earned in the market.
            if forecast_fn is not None:
                dt_hours = battery_params.get("dt_hours", 1.0)
                profit = (
                    (df_dispatch["p_discharge_MW"] * actual_prices.values).sum() -
                    (df_dispatch["p_charge_MW"] * actual_prices.values).sum()
                ) * dt_hours

            throughput = df_dispatch["throughput_MWh"].sum()

            # Counts hours where net_MW is non-zero as a measure of activity, not true cycle count
            active_hours = int((df_dispatch["net_MW"] != 0).sum())

            # Equivalent full cycles: total throughput divided by twice the battery capacity
            equiv_full_cycles = throughput / (2 * battery_params["e_max_mwh"])

            # Average spread weighted by energy (MWh), not power (MW)
            dt_hours = battery_params.get("dt_hours", 1.0)
            charge_energy_mwh = (df_dispatch["p_charge_MW"] * dt_hours).sum()
            discharge_energy_mwh = (df_dispatch["p_discharge_MW"] * dt_hours).sum()

            if charge_energy_mwh > 0 and discharge_energy_mwh > 0:
                avg_charge_price = (
                    (df_dispatch["p_charge_MW"] * dt_hours * df_dispatch["price_per_mwh"]).sum()
                    / charge_energy_mwh
                )
                avg_discharge_price = (
                    (df_dispatch["p_discharge_MW"] * dt_hours * df_dispatch["price_per_mwh"]).sum()
                    / discharge_energy_mwh
                )
                avg_spread = avg_discharge_price - avg_charge_price
            else:
                avg_spread = 0.0

            results.append({
                "date": current_date,
                "profit": profit,
                "throughput_MWh": throughput,
                "active_hours": active_hours,
                "equiv_full_cycles": equiv_full_cycles,
                "avg_spread": avg_spread,
                "solve_failed": False
            })

        except PermissionError:
            raise
        except Exception as e:
            logger.error(f"Solve failed for {date_str}: {e}")
            results.append({"date": current_date, **FAILED_DAY})

        if progress_callback:
            progress_callback(i + 1, total_days)

    if not results:
        return pd.DataFrame()

    df_results = pd.DataFrame(results)
    df_results.set_index("date", inplace=True)
    return df_results