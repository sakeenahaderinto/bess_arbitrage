import os
import pytest
import pandas as pd
from unittest.mock import patch, MagicMock

from bess.data.em_client import (
    get_zones,
    get_historical_prices,
    get_day_ahead_forecast,
    EUROPEAN_ZONES
)

def test_missing_api_key(monkeypatch):
    """
    Test 1 - missing API key: Temporarily unset EM_API_KEY and assert that
    calling get_historical_prices raises a ValueError.
    """
    monkeypatch.setattr("bess.data.em_client.os.getenv", lambda k: None)
    
    with pytest.raises(ValueError, match="missing or empty"):
        get_historical_prices("DE", "2024-01-01", "2024-01-02")

@patch("bess.data.em_client.requests.get")
def test_bad_api_key(mock_get, monkeypatch):
    """
    Test 2 - bad API key: Mock response to 401 and assert PermissionError is raised.
    """
    monkeypatch.setenv("EM_API_KEY", "invalid_key")
    
    mock_response = MagicMock()
    mock_response.status_code = 401
    mock_response.text = "Unauthorized"
    mock_get.return_value = mock_response
    
    with pytest.raises(PermissionError, match="status 401"):
        get_historical_prices("DE", "2024-01-01", "2024-01-02")

def test_invalid_zone(monkeypatch):
    """
    Test 3 - invalid zone: Call with an unsupported zone and catch ValueError.
    """
    monkeypatch.setenv("EM_API_KEY", "dummy_key")
    
    with pytest.raises(ValueError, match="not in the supported EUROPEAN_ZONES"):
        get_historical_prices("US-CAL", "2024-01-01", "2024-01-02")

def test_get_zones_returns_expected_content():
    """
    Test 4 - get_zones returns expected content.
    """
    zones = get_zones()
    assert isinstance(zones, list)
    assert "DE" in zones
    assert "GB" in zones
    assert len(zones) == len(set(zones))

@pytest.mark.integration
def test_live_fetch_historical_prices():
    """
    Test 5 - live fetch: With a real API key, fetch real data and check form.
    """
    if not os.getenv("EM_API_KEY"):
        pytest.skip("No EM_API_KEY found, skipping integration test.")
        
    series = get_historical_prices("DE", "2024-01-01", "2024-01-02")
    
    assert isinstance(series, pd.Series)
    assert isinstance(series.index, pd.DatetimeIndex)
    assert str(series.index.tz) == 'UTC'
    assert len(series) >= 24  # At least 24 hours of data
    assert pd.api.types.is_float_dtype(series)

@pytest.mark.skip(reason="API key does not have forecast access")
def test_live_fetch_forecast():
    """
    Live test for forecast endpoint.
    """
    if not os.getenv("EM_API_KEY"):
        pytest.skip("No EM_API_KEY found, skipping integration test.")
        
    series = get_day_ahead_forecast("DE")
    
    assert isinstance(series, pd.Series)
    assert isinstance(series.index, pd.DatetimeIndex)
    # The horizon is up to 72 hours, but sometimes they only have data up till tomorrow midnight.
    assert len(series) > 0
    assert pd.api.types.is_float_dtype(series)
