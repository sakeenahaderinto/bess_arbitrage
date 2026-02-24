import pytest
import pandas as pd
from bess.metrics import efficiency_gap, daily_summary, payback_period

def test_efficiency_gap_100():
    """Test 1 - efficiency gap at 100%: Assert efficiency_gap(100.0, 100.0) == 1.0"""
    gap = efficiency_gap(100.0, 100.0)
    assert gap == 1.0

def test_efficiency_gap_80():
    """Test 2 - efficiency gap at 80%: Assert efficiency_gap(80.0, 100.0) == 0.8"""
    gap = efficiency_gap(80.0, 100.0)
    assert gap == 0.8

def test_efficiency_gap_zero_denominator():
    """Test 3 - efficiency gap zero denominator: Assert efficiency_gap(50.0, 0.0) returns None"""
    gap = efficiency_gap(50.0, 0.0)
    assert gap is None

def test_daily_summary_keys():
    """
    Test 4 - daily summary keys: Pass a minimal DataFrame with 5 rows of known
    profit values and assert the returned dict contains all expected keys.
    """
    dates = pd.date_range("2024-01-01", periods=5)
    profits = [10.0, 20.0, -5.0, 15.0, 50.0]
    
    df = pd.DataFrame({"profit": profits}, index=dates)
    
    summary = daily_summary(df)
    
    expected_keys = {
        "total_revenue",
        "annualised_revenue",
        "daily_mean",
        "daily_std",
        "consistency_ratio",
        "best_day",
        "worst_day",
        "pct_profitable",
        "failed_days",
        "annualised_revenue_basis_days"
    }
    
    assert set(summary.keys()) == expected_keys
    
    assert summary["total_revenue"] == sum(profits)
    assert summary["daily_mean"] == sum(profits) / len(profits)
    assert summary["pct_profitable"] == 80.0  # 4 out of 5 days are > 0
    assert summary["best_day"]["profit"] == 50.0
    assert summary["worst_day"]["profit"] == -5.0

def test_payback_period():
    assert payback_period(100000.0, 20000.0) == 5.0
    assert payback_period(100000.0, 0.0) is None
    assert payback_period(100000.0, -100.0) is None
