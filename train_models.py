"""Script for local model training (not intended for production deployment).

Retrieves historical price data for predefined geographic zones, trains two distinct
LightGBM predictive models (a GB-specific model and a pooled continental European model),
and serializes the artifacts to the `models/` directory.

The generated `.lgb` artifacts should be committed to version control, enabling the
deployed application to load them instantaneously without incurring runtime training costs.

Usage:
    python train_models.py

Requirements:
    - The `EM_API_KEY` environment variable must be present in the local `.env` file.
    - Dependencies: `pip install lightgbm pandas pyarrow python-dotenv`
"""

import os
from pathlib import Path

import pandas as pd

from bess.data.cache import get_prices_cached
from bess.forecast.ml import train_model, train_model_pooled, GB_MODEL_PATH, EUROPE_MODEL_PATH


TRAIN_START = "2021-01-01"
TRAIN_END   = "2024-12-31"

GB_ZONES = ["GB"]  

EUROPE_ZONES = [
    "DE",       # Germany: Large, liquid market; gas-indexed with strong wind penetration.
    "FR",       # France: Nuclear-heavy; typically provides a lower-volatility baseline.
    "NL",       # Netherlands: Highly interconnected; effective proxy for Northwest Europe.
    "ES",       # Spain: Solar-heavy generation mix resulting in a distinct intraday profile.
    "IT-NO",    # Italy North: Highest liquidity Italian zone; robust hydro and import mix.
    "SE-SE3",   # Sweden (Stockholm): Hydro-dominated; operates under a distinct pricing regime.
    "PL",       # Poland: Coal-heavy; representative of Eastern European price dynamics.
    "BE",       # Belgium: Small but liquid; nuclear baseline with strong interconnection.
]


def fetch_zone(zone: str, start: str, end: str) -> pd.Series:
    print(f"  Fetching {zone} ({start} → {end})...")
    try:
        series = get_prices_cached(zone, start, end)
        if series.empty:
            print(f"  WARNING: No data returned for {zone} — skipping.")
            return pd.Series(dtype=float)
        n_missing = series.isna().sum()
        if n_missing > 0:
            print(f"  WARNING: {zone} has {n_missing} missing hours — they will be dropped during feature building.")
        print(f"  OK: {len(series)} hours fetched for {zone}.")
        return series
    except Exception as e:
        print(f"  ERROR fetching {zone}: {e} — skipping.")
        return pd.Series(dtype=float)


def fetch_all_zones(zones: list[str], start: str, end: str) -> list[pd.Series]:
    """Retrieves independent price series for multiple zones.

    Yields a list of `pd.Series` objects. Data is intentionally NOT concatenated here.
    Each zone's series is processed separately within `train_model_pooled()` to engineer
    zone-specific features prior to pooling.
    
    Attempting to concatenate raw price series sharing identical timestamps would produce
    duplicate DatetimeIndex labels, which causes subsequent structural operations (like `asfreq('h')`)
    to trace a `ValueError` during re-indexing.
    """
    all_series = []
    for zone in zones:
        s = fetch_zone(zone, start, end)
        if not s.empty:
            all_series.append(s)

    if not all_series:
        raise RuntimeError("No data fetched for any zone. Check your API key and zone list.")

    total_rows = sum(len(s) for s in all_series)
    print(f"  Ready: {total_rows} total hours across {len(all_series)} zones.\n")
    return all_series


def main():
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    # ── GB model ────────────────────────────────────────────────────────────
    print("=" * 60)
    print("Training GB model...")
    print("=" * 60)

    gb_zone_series = fetch_all_zones(GB_ZONES, TRAIN_START, TRAIN_END)
    # For a single zone, the standard `train_model` interface is appropriate.
    gb_model = train_model(gb_zone_series[0])
    gb_model.save_model(GB_MODEL_PATH)
    print(f"Saved GB model → {GB_MODEL_PATH}")
    print(f"  Trees: {gb_model.num_trees()} (best iteration: {gb_model.best_iteration})\n")

    # ── European pooled model ────────────────────────────────────────────────
    print("=" * 60)
    print("Training continental European pooled model...")
    print("=" * 60)

    europe_zone_series = fetch_all_zones(EUROPE_ZONES, TRAIN_START, TRAIN_END)
    # Feature engineering is iteratively applied per zone internally within `train_model_pooled`.
    # This prevents the creation of duplicate DatetimeIndex labels from naive concatenation.
    europe_model = train_model_pooled(europe_zone_series)
    europe_model.save_model(EUROPE_MODEL_PATH)
    print(f"Saved Europe model → {EUROPE_MODEL_PATH}")
    print(f"  Trees: {europe_model.num_trees()} (best iteration: {europe_model.best_iteration})\n")

    print("=" * 60)
    print("Done.")
    print("=" * 60)


if __name__ == "__main__":
    main()