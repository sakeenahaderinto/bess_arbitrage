import pytest
from bess.optimiser import battery_solve_arbitrage

def test_basic_run():
    """
    Test 1 — basic run: Call `battery_solve_arbitrage` with a 24-hour price curve.
    Assert that `total_profit > 0` and that the returned DataFrame has exactly 24 rows.
    """
    prices = [
        20, 15, 10, 12, 15, 25, 40, 60, 55, 40, 35, 30,
        25, 20, 18, 25, 60, 90, 85, 70, 50, 40, 30, 25
    ]
    _, _, df, profit = battery_solve_arbitrage(prices)

    assert profit > 0, "Total profit should be strictly positive."
    assert len(df) == 24, "DataFrame should have exactly 24 rows."

def test_soc_bounds():
    """
    Test 2 — SoC bounds: Assert that every value in `df["soc_start_MWh"]` 
    is >= 0 and <= `e_max_mwh`.
    """
    prices = [
        20, 15, 10, 12, 15, 25, 40, 60, 55, 40, 35, 30,
        25, 20, 18, 25, 60, 90, 85, 70, 50, 40, 30, 25
    ]
    e_max = 1.0
    _, _, df, _ = battery_solve_arbitrage(prices, e_max_mwh=e_max)

    assert (df["soc_start_MWh"] >= 0 - 1e-9).all(), "SoC must never be below 0."
    assert (df["soc_start_MWh"] <= e_max + 1e-9).all(), "SoC must never exceed e_max_mwh."

def test_terminal_soc():
    """
    Test 3 — terminal SoC: With `enforce_terminal_soc=True`, assert that
    `df.iloc[-1]["soc_end_MWh"]` equals `soc0_mwh`.
    """
    prices = [
        20, 15, 10, 12, 15, 25, 40, 60, 55, 40, 35, 30,
        25, 20, 18, 25, 60, 90, 85, 70, 50, 40, 30, 25
    ]
    soc0 = 0.0
    _, _, df, _ = battery_solve_arbitrage(
        prices,
        soc0_mwh=soc0,
        enforce_terminal_soc=True
    )

    # Use a small tolerance for floating point comparisons
    actual_terminal_soc = df.iloc[-1]["soc_end_MWh"]
    assert abs(actual_terminal_soc - soc0) < 1e-9, f"Terminal SoC ({actual_terminal_soc}) should equal soc0_mwh ({soc0})."

def test_degradation_suppresses_small_cycles():
    """
    Test 4 — degradation suppresses small cycles.
    Run the optimiser on a price series with only small spreads (20-25).
    With `deg_cost_per_mwh=10.0`, the battery shouldn't cycle.
    """
    prices = [20, 22, 25, 21, 24, 23, 22, 25] * 3  # 24 hours of small spreads
    _, _, df, profit = battery_solve_arbitrage(prices, deg_cost_per_mwh=10.0)
    
    assert abs(df["throughput_MWh"].sum()) < 1e-9, "Throughput should be 0 when spreads are smaller than wear cost."

def test_degradation_reduces_profit():
    """
    Test 5 — degradation reduces profit.
    Run a 24-hour price curve with 0.0 and 5.0 degradation cost.
    Assert profit is lower with the degradation cost.
    """
    prices = [
        20, 15, 10, 12, 15, 25, 40, 60, 55, 40, 35, 30,
        25, 20, 18, 25, 60, 90, 85, 70, 50, 40, 30, 25
    ]
    
    _, _, _, profit_no_deg = battery_solve_arbitrage(prices, deg_cost_per_mwh=0.0)
    _, _, _, profit_with_deg = battery_solve_arbitrage(prices, deg_cost_per_mwh=5.0)
    
    assert profit_with_deg <= profit_no_deg, "Degradation cost should reduce total profit."

