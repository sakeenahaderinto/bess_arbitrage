import os
import requests
import pandas as pd
from dotenv import load_dotenv

# Load .env file so EM_API_KEY is available via os.getenv()
load_dotenv()

# Hardcoded list of European zones with confirmed day-ahead price data on
# Electricitymaps. Scoped to Europe to keep the app focused — the full global
# zone list would add noise without adding value for this project.
# Source: Electricitymaps day-ahead price availability table.
# Update manually if coverage changes.
EUROPEAN_ZONES = [
    "AT", "AX", "BE", "BG", "CH", "CZ", "DE",
    "DK-BHM", "DK-DK1", "DK-DK2", "EE", "ES", "FI", "FR",
    "GB", "GB-NIR", "GR", "HR", "HU", "IE",
    "IT-CNO", "IT-CSO", "IT-NO", "IT-SAR", "IT-SIC", "IT-SO",
    "LT", "LU", "LV", "MD", "ME", "MK", "NL",
    "NO-NO1", "NO-NO2", "NO-NO3", "NO-NO4", "NO-NO5",
    "PL", "PT", "RO", "RS",
    "SE-SE1", "SE-SE2", "SE-SE3", "SE-SE4",
    "SI", "SK",
]

# Base URL for all Electricitymaps day-ahead price endpoints
BASE_URL = "https://api.electricitymaps.com/v3/price-day-ahead"


def _get_api_key() -> str:
    """
    Loads the Electricitymaps API key from the environment.
    Raises a clear ValueError if it is missing or empty, so the caller
    gets a useful message rather than a cryptic auth failure.
    """
    api_key = os.getenv("EM_API_KEY")
    if not api_key or not api_key.strip():
        raise ValueError(
            "EM_API_KEY environment variable is missing or empty. "
            "Add it to your .env file: EM_API_KEY=your_key_here"
        )
    return api_key


def _validate_zone(zone: str) -> None:
    """
    Checks that the requested zone is in our supported European zones list.
    Raises a ValueError early so we don't waste an API call on an unsupported zone.
    """
    if zone not in EUROPEAN_ZONES:
        raise ValueError(
            f"Zone '{zone}' is not in the supported EUROPEAN_ZONES list. "
            f"Call get_zones() to see available options."
        )


def get_zones() -> list[str]:
    """
    Returns the hardcoded list of supported European zone codes.

    No API call is made — this is purely a local lookup.
    Used by the Streamlit sidebar to populate the zone dropdown.
    """
    return EUROPEAN_ZONES


def get_historical_prices(zone: str, start_date: str, end_date: str, timeout: int = 10) -> pd.Series:
    """
    Fetches hourly day-ahead prices for a zone over a date range.

    The Electricitymaps API limits requests to 10 days at a time, so this
    function loops in 10-day chunks and stitches the results together.

    Args:
        zone:       Zone code, must be in EUROPEAN_ZONES (e.g. "DE", "GB").
        start_date: Start date in "YYYY-MM-DD" format (inclusive).
        end_date:   End date in "YYYY-MM-DD" format. One day is added internally 
                    so that all 24 hours of the end_date are captured. For example,
                    end_date="2024-01-31" fetches up to 2024-02-01 00:00 UTC.
        timeout:    Timeout in seconds for API requests.

    Returns:
        pd.Series with a UTC DatetimeIndex and float price values.
        Prices are in the local currency for that zone (e.g. EUR/MWh for DE).

    Raises:
        ValueError:   If the zone is unsupported, the API key is missing, or
                      the date format is invalid.
        RuntimeError: If the API returns a non-200 status code.
    """
    api_key = _get_api_key()
    _validate_zone(zone)

    # Parse the dates and add 1 day to end_dt so the full end_date is included.
    # Without +1 day, end_dt lands at midnight and misses the last 23 hours.
    try:
        start_dt = pd.to_datetime(start_date, utc=True)
        end_dt = pd.to_datetime(end_date, utc=True) + pd.Timedelta(days=1)
    except Exception as e:
        raise ValueError(f"Invalid date format. Expected YYYY-MM-DD. Error: {e}")

    url = f"{BASE_URL}/past-range"
    headers = {"auth-token": api_key}
    all_data = []

    # Loop in 10-day chunks — this is the API's maximum window per request
    current_start = start_dt
    while current_start < end_dt:
        current_end = min(current_start + pd.Timedelta(days=10), end_dt)

        # The API expects space-separated datetimes (e.g. "2024-01-01 00:00"),
        # not the ISO "T" format. requests handles URL-encoding automatically.
        params = {
            "zone": zone,
            "start": current_start.strftime("%Y-%m-%d %H:%M"),
            "end": current_end.strftime("%Y-%m-%d %H:%M"),
        }

        response = requests.get(url, headers=headers, params=params, timeout=timeout)

        if response.status_code == 401:
            raise PermissionError(
                f"Permission denied for zone '{zone}' "
                f"(status 401): {response.text}"
            )
        elif response.status_code == 429:
            raise RuntimeError(
                f"Rate limit exceeded for zone '{zone}' "
                f"(status 429): {response.text}"
            )
        elif response.status_code != 200:
            raise RuntimeError(
                f"API request failed for zone '{zone}' "
                f"(status {response.status_code}): {response.text}"
            )

        # Extract the list of price records from the response
        data_json = response.json()
        for entry in data_json.get("data", []):
            all_data.append({
                "datetime": entry["datetime"],
                "price": entry["value"],
            })

        # Advance to the next chunk
        current_start = current_end

    # Return an empty Series if no data came back
    if not all_data:
        return pd.Series(dtype=float, index=pd.DatetimeIndex([]))

    # Build a DataFrame then convert to a Series with a proper DatetimeIndex
    df = pd.DataFrame(all_data)
    datetime_index = pd.to_datetime(df["datetime"], utc=True)
    series = pd.Series(data=df["price"].values, index=datetime_index, dtype=float)

    # Remove any duplicates that could appear at chunk boundaries, then sort
    series = series[~series.index.duplicated(keep="first")]
    series = series.sort_index()

    return series


def get_day_ahead_forecast(zone: str, timeout: int = 10) -> pd.Series:
    """
    Fetches the 72-hour day-ahead price forecast for a zone, starting from now.

    Electricitymaps returns actual published prices where available, and falls
    back to their own model forecasts for hours not yet published. The 'source'
    field in the raw response indicates which is which.

    Args:
        zone: Zone code, must be in EUROPEAN_ZONES (e.g. "DE", "GB").

    Returns:
        pd.Series with a UTC DatetimeIndex and float price values (up to 72 hours).
        Prices are in the local currency for that zone (e.g. EUR/MWh for DE).

    Raises:
        ValueError:   If the zone is unsupported or the API key is missing.
        RuntimeError: If the API returns a non-200 status code.
    """
    api_key = _get_api_key()
    _validate_zone(zone)

    url = f"{BASE_URL}/forecast"
    headers = {"auth-token": api_key}

    # horizonHours=72 requests the maximum available forecast window
    params = {
        "zone": zone,
        "horizonHours": 72,
    }

    response = requests.get(url, headers=headers, params=params, timeout=timeout)

    if response.status_code == 401:
        raise PermissionError(
            f"Permission denied for zone '{zone}' "
            f"(status 401): {response.text}"
        )
    elif response.status_code == 429:
        raise RuntimeError(
            f"Rate limit exceeded for zone '{zone}' "
            f"(status 429): {response.text}"
        )
    elif response.status_code != 200:
        raise RuntimeError(
            f"API request failed for zone '{zone}' "
            f"(status {response.status_code}): {response.text}"
        )

    data_json = response.json()
    all_data = []
    for entry in data_json.get("data", []):
        all_data.append({
            "datetime": entry["datetime"],
            "price": entry["value"],
        })

    # Return an empty Series if no forecast data came back
    if not all_data:
        return pd.Series(dtype=float, index=pd.DatetimeIndex([]))

    df = pd.DataFrame(all_data)
    datetime_index = pd.to_datetime(df["datetime"], utc=True)
    series = pd.Series(data=df["price"].values, index=datetime_index, dtype=float)

    series = series.sort_index()
    return series