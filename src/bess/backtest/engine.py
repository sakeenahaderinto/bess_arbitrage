import gc
import logging
import pandas as pd
from typing import Callable
from bess.optimiser import build_backtest_solver, backtest_solve

logger = logging.getLogger(__name__)

FAILED_DAY = {
    "profit": 0.0,
    "throughput_MWh": 0.0,
    "active_hours": 0,
    "equiv_full_cycles": 0.0,
    "avg_spread": 0.0,
    "solve_failed": True,
}

# Collect garbage every N days rather than every day.
# gc.collect() on each of 365 days adds meaningful overhead on a slow shared CPU.
_GC_INTERVAL = 20


def run_backtest(
    prices: pd.Series,
    forecast_fn: Callable | None,
    battery_params: dict,
    progress_callback: Callable | None = None,
    start_date: str | None = None,
) -> pd.DataFrame:
    """
    Runs the battery optimiser over every day in the provided price series.

    The Pyomo ConcreteModel and HiGHS solver are built once before the loop and
    reused on every day via mutable price Params. This eliminates per-day model
    construction (RangeSet, Var, Constraint objects) which was the dominant
    source of slowness on constrained CPUs like Streamlit Cloud.

    Note: Days containing fewer or more than exactly 24 hours are skipped. This
    natively drops dates where Daylight Saving Time (DST) transitions produce
    23- or 25-hour days. Days with NaN prices are also skipped.

    Args:
        prices: Historical price Series with a UTC DatetimeIndex.
        forecast_fn: A function `(prices, date_str) -> pd.Series` that provides
                     a 24-hour forecast. If None, uses actual prices (perfect foresight).
        battery_params: Dict of parameters for build_backtest_solver / backtest_solve.
        progress_callback: Optional function `(current_day_int, total_days_int)`
                           to report progress back to the caller/UI.
        start_date: Optional date string 'YYYY-MM-DD'. If provided, only dates
                    on or after this date are included in the backtest.

    Returns:
        pd.DataFrame: A row per day with performance metrics.
    """
    if prices.empty:
        return pd.DataFrame()

    if prices.index.tz is None:
        prices.index = prices.index.tz_localize("UTC")
    else:
        prices.index = prices.index.tz_convert("UTC")

    prices = prices.asfreq("h")

    unique_dates = sorted(list(set(prices.index.date)))

    if start_date:
        backtest_start = pd.to_datetime(start_date).date()
        unique_dates = [d for d in unique_dates if d >= backtest_start]

    if not unique_dates:
        return pd.DataFrame()

    results = []
    total_days = len(unique_dates)
    dt_hours = battery_params.get("dt_hours", 1.0)

    # Build the Pyomo model and solver once — reused for every day in the loop.
    # On a 365-day backtest this avoids 364 redundant ConcreteModel constructions,
    # which is the dominant cost on a slow shared CPU (Streamlit Cloud free tier).
    m, opt = build_backtest_solver(battery_params)

    for i, current_date in enumerate(unique_dates):
        date_str = current_date.strftime("%Y-%m-%d")

        try:
            actual_prices = prices.loc[date_str]

            if actual_prices.isnull().any():
                logger.warning(f"Skipping {date_str} due to NaN prices.")
                continue

            if len(actual_prices) != 24:
                logger.warning(
                    f"Skipping {date_str} due to incomplete data ({len(actual_prices)} hours)."
                )
                continue

            if forecast_fn is None:
                solve_prices = actual_prices.tolist()
            else:
                forecast_series = forecast_fn(prices, date_str)

                if len(forecast_series) != 24 or forecast_series.index[0].date() != current_date:
                    logger.warning(
                        f"Forecast for {date_str} returned invalid shape or date "
                        f"(length: {len(forecast_series)}). Marking as failed."
                    )
                    results.append({"date": current_date, **FAILED_DAY})
                    continue

                if forecast_series.isnull().any():
                    logger.warning(
                        f"Forecast for {date_str} contains NaN values. Marking as failed."
                    )
                    results.append({"date": current_date, **FAILED_DAY})
                    continue

                solve_prices = forecast_series.tolist()

            logger.info(
                f"Solving {date_str}: min={min(solve_prices):.2f} max={max(solve_prices):.2f}"
            )

            df_dispatch, profit = backtest_solve(m, opt, solve_prices, battery_params)

            logger.info(f"Solved {date_str}: profit={profit:.2f}")

            if forecast_fn is not None:
                profit = (
                    (df_dispatch["p_discharge_MW"] * actual_prices.values).sum()
                    - (df_dispatch["p_charge_MW"] * actual_prices.values).sum()
                ) * dt_hours

            throughput = df_dispatch["throughput_MWh"].sum()
            active_hours = int((df_dispatch["net_MW"] != 0).sum())
            equiv_full_cycles = throughput / (2 * battery_params["e_max_mwh"])

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
                "solve_failed": False,
            })

        except PermissionError:
            raise
        except Exception as e:
            logger.error(f"Solve failed for {date_str}: {e}")
            results.append({"date": current_date, **FAILED_DAY})

        # Throttle GC — collecting every day adds overhead on slow CPUs.
        if i % _GC_INTERVAL == 0:
            gc.collect()

        if progress_callback:
            progress_callback(i + 1, total_days)

    del m
    gc.collect()

    if not results:
        return pd.DataFrame()

    df_results = pd.DataFrame(results)
    df_results.set_index("date", inplace=True)
    return df_results