# Energy Storage Arbitrage --> Where an energy storage operator captures economic value from price volatility

import math
from typing import Sequence, Optional, Dict, Any, Tuple
import io
import contextlib
import pyomo.environ as pyo
from pyomo.opt import TerminationCondition
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Internal helpers for backtest model reuse
# ============================================================

def _build_model(
    n: int,
    *,
    e_max_mwh: float,
    p_max_mw: float,
    roundtrip_eff: float,
    soc0_mwh: float,
    dt_hours: float,
    deg_cost_per_mwh: float,
    enforce_terminal_soc: bool,
    no_simultaneous: bool,
    return_duals: bool,
) -> Tuple[pyo.ConcreteModel, Any]:
    """
    Constructs the Pyomo ConcreteModel and persistent HiGHS solver instance
    with dummy zero prices. Called once per backtest run; prices are updated
    cheaply via mutable Params on each subsequent day using _update_prices().

    Returns:
        (model, solver) — both are reused across the entire backtest loop.
    """
    eta_c = math.sqrt(roundtrip_eff)
    eta_d = math.sqrt(roundtrip_eff)

    m = pyo.ConcreteModel(name="battery_arbitrage")
    m.T = pyo.RangeSet(0, n - 1)
    m.S = pyo.RangeSet(0, n)

    # Mutable=True is essential — it allows price updates without model rebuilds.
    m.price = pyo.Param(m.T, initialize={t: 0.0 for t in range(n)}, mutable=True)

    m.p_charge = pyo.Var(m.T, domain=pyo.NonNegativeReals, bounds=(0.0, p_max_mw))
    m.p_discharge = pyo.Var(m.T, domain=pyo.NonNegativeReals, bounds=(0.0, p_max_mw))
    m.soc = pyo.Var(m.S, domain=pyo.NonNegativeReals, bounds=(0.0, e_max_mwh))
    m.soc[0].fix(soc0_mwh)

    def soc_balance_rule(m, t):
        return m.soc[t + 1] == m.soc[t] + dt_hours * (
            eta_c * m.p_charge[t] - (1.0 / eta_d) * m.p_discharge[t]
        )
    m.soc_balance = pyo.Constraint(m.T, rule=soc_balance_rule)

    if enforce_terminal_soc:
        m.terminal_soc = pyo.Constraint(expr=m.soc[n] == soc0_mwh)

    if no_simultaneous:
        m.is_charging = pyo.Var(m.T, domain=pyo.Binary)
        m.charge_gate = pyo.Constraint(
            m.T, rule=lambda m, t: m.p_charge[t] <= p_max_mw * m.is_charging[t]
        )
        m.discharge_gate = pyo.Constraint(
            m.T, rule=lambda m, t: m.p_discharge[t] <= p_max_mw * (1 - m.is_charging[t])
        )

    m.profit = pyo.Objective(
        expr=sum(
            dt_hours * m.price[t] * (m.p_discharge[t] - m.p_charge[t])
            - deg_cost_per_mwh * dt_hours * (m.p_charge[t] + m.p_discharge[t])
            for t in m.T
        ),
        sense=pyo.maximize,
    )

    if return_duals:
        m.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)

    opt = pyo.SolverFactory("appsi_highs")
    if not opt.available():
        raise RuntimeError(
            "HiGHS solver not found. Install it with: pip install highspy"
        )

    return m, opt


def _update_prices(m: pyo.ConcreteModel, prices: Sequence[float]) -> None:
    """Update mutable price params in-place. O(n) with no model reconstruction."""
    for t, p in enumerate(prices):
        m.price[t].set_value(p)


def _extract_solution(
    m: pyo.ConcreteModel,
    prices: Sequence[float],
    dt_hours: float,
    return_duals: bool,
) -> Tuple[pd.DataFrame, float]:
    """Extract dispatch DataFrame and total profit from a solved model."""
    n = len(prices)
    rows = []
    for t in range(n):
        p_c = pyo.value(m.p_charge[t])
        p_d = pyo.value(m.p_discharge[t])
        soc_start = pyo.value(m.soc[t])
        soc_end = pyo.value(m.soc[t + 1])
        price = pyo.value(m.price[t])
        net_mw = p_d - p_c
        cash = dt_hours * price * net_mw
        throughput = dt_hours * (p_c + p_d)
        shadow_price = pyo.value(m.dual[m.soc_balance[t]]) if return_duals else None

        row = {
            "t": t,
            "price_per_mwh": price,
            "p_charge_MW": p_c,
            "p_discharge_MW": p_d,
            "soc_start_MWh": soc_start,
            "soc_end_MWh": soc_end,
            "net_MW": net_mw,
            "throughput_MWh": throughput,
            "cashflow_$": cash,
        }
        if return_duals:
            row["shadow_price"] = shadow_price
        rows.append(row)

    df = pd.DataFrame(rows).set_index("t")
    total_profit = float(df["cashflow_$"].sum())
    return df, total_profit


# ============================================================
# Public API
# ============================================================

def battery_solve_arbitrage(
        prices: Sequence[float],
        *,
        e_max_mwh: float = 1.0,
        p_max_mw: float = 0.5,
        roundtrip_eff: float = 0.90,
        soc0_mwh: float = 0.0,
        dt_hours: float = 1.0,
        deg_cost_per_mwh: float = 0.0,
        enforce_terminal_soc: bool = True,
        no_simultaneous: bool = False,
        return_duals: bool = False,
        tee: bool = False,
        solver_options: Optional[Dict[str, Any]] = None,
) -> Tuple[pyo.ConcreteModel, Any, pd.DataFrame, float]:
    """
    Solve the battery arbitrage problem for a single price sequence.

    Builds and solves a fresh model each call. For repeated solves over many
    days (e.g. backtesting), prefer build_backtest_solver() + backtest_solve()
    which reuse the model structure and avoid per-day reconstruction overhead.

    Return:
        model, results, dataframe, total_profit
    """
    prices = [float(p) for p in prices]
    if len(prices) == 0:
        raise ValueError("prices must contain at least 1 timestep")
    if e_max_mwh <= 0:
        raise ValueError("e_max_mwh must be > 0.")
    if p_max_mw <= 0:
        raise ValueError("p_max_mw must be > 0.")
    if dt_hours <= 0:
        raise ValueError("dt_hours must be > 0.")
    if not (0 < roundtrip_eff <= 1.0):
        raise ValueError("roundtrip_eff must be in (0, 1].")
    if not (0 <= soc0_mwh <= e_max_mwh):
        raise ValueError("soc0_mwh must be within [0, e_max_mwh].")
    if deg_cost_per_mwh < 0:
        raise ValueError("deg_cost_per_mwh must be >= 0.")
    if return_duals and no_simultaneous:
        raise ValueError(
            "Shadow prices (duals) are only available for pure LP models. "
            "Cannot use return_duals=True with no_simultaneous=True (MILP)."
        )

    n = len(prices)
    m, opt = _build_model(
        n,
        e_max_mwh=e_max_mwh,
        p_max_mw=p_max_mw,
        roundtrip_eff=roundtrip_eff,
        soc0_mwh=soc0_mwh,
        dt_hours=dt_hours,
        deg_cost_per_mwh=deg_cost_per_mwh,
        enforce_terminal_soc=enforce_terminal_soc,
        no_simultaneous=no_simultaneous,
        return_duals=return_duals,
    )
    _update_prices(m, prices)

    if solver_options:
        for k, v in solver_options.items():
            opt.options[k] = v

    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        results = opt.solve(m, tee=tee)

    if results.solver.termination_condition != TerminationCondition.optimal:
        raise RuntimeError(
            f"Optimisation failed. Solver termination condition: "
            f"{results.solver.termination_condition}"
        )

    df, total_profit = _extract_solution(m, prices, dt_hours, return_duals)
    return m, results, df, total_profit


def build_backtest_solver(
    battery_params: dict,
    n: int = 24,
) -> Tuple[pyo.ConcreteModel, Any]:
    """
    Build a reusable Pyomo model + HiGHS solver pair for backtest loops.

    The model structure (variables, constraints, objective) is constructed once.
    On each day, call backtest_solve(m, opt, prices, battery_params) to update
    the price params and re-solve without any model reconstruction overhead.

    Args:
        battery_params: Same dict passed to battery_solve_arbitrage.
        n: Number of timesteps per day (default 24 for hourly).

    Returns:
        (model, solver) — pass both to backtest_solve() on each iteration.
    """
    return _build_model(
        n,
        e_max_mwh=battery_params.get("e_max_mwh", 1.0),
        p_max_mw=battery_params.get("p_max_mw", 0.5),
        roundtrip_eff=battery_params.get("roundtrip_eff", 0.90),
        soc0_mwh=battery_params.get("soc0_mwh", 0.0),
        dt_hours=battery_params.get("dt_hours", 1.0),
        deg_cost_per_mwh=battery_params.get("deg_cost_per_mwh", 0.0),
        enforce_terminal_soc=battery_params.get("enforce_terminal_soc", True),
        no_simultaneous=battery_params.get("no_simultaneous", False),
        return_duals=battery_params.get("return_duals", False),
    )


def backtest_solve(
    m: pyo.ConcreteModel,
    opt: Any,
    prices: Sequence[float],
    battery_params: dict,
    tee: bool = False,
) -> Tuple[pd.DataFrame, float]:
    """
    Update prices on a pre-built model and re-solve. No model reconstruction.

    Args:
        m: ConcreteModel from build_backtest_solver().
        opt: Solver from build_backtest_solver().
        prices: 24-element price list for this day.
        battery_params: Same dict passed to build_backtest_solver().
        tee: Whether to print solver output.

    Returns:
        (df_dispatch, profit)

    Raises:
        RuntimeError: If the solver does not return an optimal solution.
    """
    _update_prices(m, prices)

    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        results = opt.solve(m, tee=tee)

    if results.solver.termination_condition != TerminationCondition.optimal:
        raise RuntimeError(
            f"Optimisation failed. Solver termination condition: "
            f"{results.solver.termination_condition}"
        )

    dt_hours = battery_params.get("dt_hours", 1.0)
    return_duals = battery_params.get("return_duals", False)
    return _extract_solution(m, prices, dt_hours, return_duals)


def plot_solution(df: pd.DataFrame, title: str = "Battery arbitrage (perfect foresight)") -> plt.Figure:
    fig, ax = plt.subplots(3, 1, sharex=True, figsize=(10, 7))

    ax[0].plot(df.index, df["price_per_mwh"], marker="o")
    ax[0].set_ylabel("Price ($/MWh)")
    ax[0].set_title(title)

    ax[1].bar(df.index, df["p_charge_MW"], alpha=0.6, label="Charge (MW)")
    ax[1].bar(df.index, -df["p_discharge_MW"], alpha=0.6, label="Discharge (MW)")
    ax[1].axhline(0, linewidth=0.8)
    ax[1].set_ylabel("Power (MW)")
    ax[1].legend()

    soc_series = [df.loc[t, "soc_start_MWh"] for t in df.index] + [df.iloc[-1]["soc_end_MWh"]]
    ax[2].step(range(len(soc_series)), soc_series, where="post")
    ax[2].set_ylabel("SoC (MWh)")
    ax[2].set_xlabel("Timestep")

    plt.tight_layout()
    return fig


if __name__ == "__main__":
    prices = [
        20, 15, 10, 12, 15, 25, 40, 60, 55, 40, 35, 30,
        25, 20, 18, 25, 60, 90, 85, 70, 50, 40, 30, 25
    ]

    _, _, df, profit = battery_solve_arbitrage(
        prices,
        e_max_mwh=1.0,
        p_max_mw=0.5,
        roundtrip_eff=0.90,
        soc0_mwh=0.0,
        dt_hours=1.0,
        enforce_terminal_soc=True,
        no_simultaneous=False,
        tee=False,
    )

    print(df.round(3))
    print(f"\nTotal arbitrage profit: ${profit:,.2f}")
    plot_solution(df)