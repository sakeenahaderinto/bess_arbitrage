import os
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# Zones supported by the deployed app.
#
# Trimmed from the full Electricitymaps European list to the zones covered by
# one of the two pre-trained models (bess_gb.lgb and bess_europe.lgb).
# Exposing unsupported zones in the dropdown would let users select a market
# the model has never seen, silently producing misleading predictions.
#
# GB model  : GB, GB-NIR
# Europe model: DE, FR, NL, ES, IT-NO, SE-SE3, PL, BE
SUPPORTED_ZONES = [
    "BE",
    "DE",
    "ES",
    "FR",
    "GB",
    "GB-NIR",
    "IT-NO",
    "NL",
    "PL",
    "SE-SE3",
]

BASE_URL = "https://api.electricitymaps.com/v3/price-day-ahead"


def _get_api_key() -> str:
    api_key = os.getenv("EM_API_KEY")
    if not api_key or not api_key.strip():
        raise ValueError(
            "EM_API_KEY environment variable is missing or empty. "
            "Add it to your .env file: EM_API_KEY=your_key_here"
        )
    return api_key


def _validate_zone(zone: str) -> None:
    if zone not in SUPPORTED_ZONES:
        raise ValueError(
            f"Zone '{zone}' is not in the supported zones list. "
            f"Supported zones: {SUPPORTED_ZONES}"
        )


def get_zones() -> list[str]:
    """Returns the list of zones supported by the deployed models."""
    return SUPPORTED_ZONES


def get_historical_prices(zone: str, start_date: str, end_date: str, timeout: int = 10) -> pd.Series:
    """
    Fetches hourly day-ahead prices for a zone over a date range.

    The Electricitymaps API limits requests to 10 days at a time, so this
    function loops in 10-day chunks and stitches the results together.

    Args:
        zone:       Zone code, must be in SUPPORTED_ZONES (e.g. "DE", "GB").
        start_date: Start date in "YYYY-MM-DD" format (inclusive).
        end_date:   End date in "YYYY-MM-DD" format. One day is added internally
                    so that all 24 hours of the end_date are captured.
        timeout:    Timeout in seconds for API requests.

    Returns:
        pd.Series with a UTC DatetimeIndex and float price values.

    Raises:
        ValueError:     If the zone is unsupported or API key is missing.
        PermissionError: If the API returns 401.
        RuntimeError:   If the API returns any other non-200 status.
    """
    api_key = _get_api_key()
    _validate_zone(zone)

    try:
        start_dt = pd.to_datetime(start_date, utc=True)
        end_dt = pd.to_datetime(end_date, utc=True) + pd.Timedelta(days=1)
    except Exception as e:
        raise ValueError(f"Invalid date format. Expected YYYY-MM-DD. Error: {e}")

    url = f"{BASE_URL}/past-range"
    headers = {"auth-token": api_key}
    all_data = []

    current_start = start_dt
    while current_start < end_dt:
        current_end = min(current_start + pd.Timedelta(days=10), end_dt)

        params = {
            "zone": zone,
            "start": current_start.strftime("%Y-%m-%d %H:%M"),
            "end": current_end.strftime("%Y-%m-%d %H:%M"),
        }

        response = requests.get(url, headers=headers, params=params, timeout=timeout)

        if response.status_code == 401:
            raise PermissionError(
                f"Permission denied for zone '{zone}' (status 401): {response.text}"
            )
        elif response.status_code == 429:
            raise RuntimeError(
                f"Rate limit exceeded for zone '{zone}' (status 429): {response.text}"
            )
        elif response.status_code != 200:
            raise RuntimeError(
                f"API request failed for zone '{zone}' "
                f"(status {response.status_code}): {response.text}"
            )

        data_json = response.json()
        for entry in data_json.get("data", []):
            all_data.append({
                "datetime": entry["datetime"],
                "price": entry["value"],
            })

        current_start = current_end

    if not all_data:
        return pd.Series(dtype=float, index=pd.DatetimeIndex([]))

    df = pd.DataFrame(all_data)
    datetime_index = pd.to_datetime(df["datetime"], utc=True)
    series = pd.Series(data=df["price"].values, index=datetime_index, dtype=float)
    series = series[~series.index.duplicated(keep="first")]
    series = series.sort_index()

    return series


def get_day_ahead_forecast(zone: str, timeout: int = 10) -> pd.Series:
    """
    Fetches the 72-hour day-ahead price forecast for a zone.

    Args:
        zone: Zone code, must be in SUPPORTED_ZONES.

    Returns:
        pd.Series with a UTC DatetimeIndex and float price values (up to 72 hours).

    Raises:
        ValueError:     If the zone is unsupported or API key is missing.
        PermissionError: If the API returns 401.
        RuntimeError:   If the API returns any other non-200 status.
    """
    api_key = _get_api_key()
    _validate_zone(zone)

    url = f"{BASE_URL}/forecast"
    headers = {"auth-token": api_key}
    params = {"zone": zone, "horizonHours": 72}

    response = requests.get(url, headers=headers, params=params, timeout=timeout)

    if response.status_code == 401:
        raise PermissionError(
            f"Permission denied for zone '{zone}' (status 401): {response.text}"
        )
    elif response.status_code == 429:
        raise RuntimeError(
            f"Rate limit exceeded for zone '{zone}' (status 429): {response.text}"
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

    if not all_data:
        return pd.Series(dtype=float, index=pd.DatetimeIndex([]))

    df = pd.DataFrame(all_data)
    datetime_index = pd.to_datetime(df["datetime"], utc=True)
    series = pd.Series(data=df["price"].values, index=datetime_index, dtype=float)
    return series.sort_index()