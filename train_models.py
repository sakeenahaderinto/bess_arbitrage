"""
train_models.py — run locally, never deployed.

Pulls historical prices for the selected zones, trains two LightGBM models
(GB-specific and continental European pooled), and saves them to models/.
Commit both .lgb files to the repo so the deployed app can load them without
any training at runtime.

Usage:
    python train_models.py

Requirements:
    - EM_API_KEY set in your .env file
    - pip install lightgbm pandas pyarrow python-dotenv

Two models are trained:
    models/bess_gb.lgb      — GB and GB-NIR only
    models/bess_europe.lgb  — DE, FR, NL, ES, IT-NO, SE-SE3, PL, BE pooled

Why two models?
    GB is an island grid with no synchronous AC interconnection to continental
    Europe. Prices clear in £ on EPEX/N2EX, and the evening ramp is more
    pronounced because there's no cross-border smoothing via interconnectors.
    Training a single model on continental patterns and applying it to GB would
    systematically underestimate peak amplitudes.

    The continental model pools 8 zones to give the model a richer distribution
    of price regimes without requiring a separate model per zone. The zones were
    chosen to cover the main market structures: gas-indexed Central Europe (DE),
    nuclear-heavy (FR), highly interconnected (NL, BE), solar-heavy Southern
    Europe (ES, IT-NO), hydro-dominated Nordics (SE-SE3), and Eastern Europe (PL).
"""

import os
from pathlib import Path

import pandas as pd

from bess.data.cache import get_prices_cached
from bess.forecast.ml import train_model, train_model_pooled, GB_MODEL_PATH, EUROPE_MODEL_PATH

# ============================================================
# Config — edit these if you want a different training window
# ============================================================

TRAIN_START = "2021-01-01"
TRAIN_END   = "2024-12-31"

GB_ZONES = ["GB"]  # GB-NIR shares the model but isn't worth extra API calls

EUROPE_ZONES = [
    "DE",       # Germany — large, liquid, gas-indexed, strong wind
    "FR",       # France — nuclear-heavy, lower volatility baseline
    "NL",       # Netherlands — highly interconnected, good proxy for NW Europe
    "ES",       # Spain — solar-heavy, different intraday shape
    "IT-NO",    # Italy North — most liquid Italian zone, strong hydro + import mix
    "SE-SE3",   # Sweden (Stockholm) — hydro-dominated, very different regime
    "PL",       # Poland — coal-heavy, Eastern Europe price dynamics
    "BE",       # Belgium — small but liquid, nuclear + interconnected
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
    """
    Fetch price series for each zone independently, returning a list of Series.

    We intentionally do NOT concatenate here. Each zone's Series is passed
    separately to train_model_pooled(), which builds features per zone before
    pooling the feature DataFrames. Concatenating raw Series first produces
    duplicate DatetimeIndex labels (all zones share the same timestamps) which
    causes asfreq('h') to raise ValueError on reindex.
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
    # Single zone — train_model is fine here; train_model_pooled works too
    gb_model = train_model(gb_zone_series[0])
    gb_model.save_model(GB_MODEL_PATH)
    print(f"Saved GB model → {GB_MODEL_PATH}")
    print(f"  Trees: {gb_model.num_trees()} (best iteration: {gb_model.best_iteration})\n")

    # ── European pooled model ────────────────────────────────────────────────
    print("=" * 60)
    print("Training continental European pooled model...")
    print("=" * 60)

    europe_zone_series = fetch_all_zones(EUROPE_ZONES, TRAIN_START, TRAIN_END)
    # Features are built per zone inside train_model_pooled to avoid duplicate
    # DatetimeIndex labels that would break asfreq('h') on a naively concatenated Series.
    europe_model = train_model_pooled(europe_zone_series)
    europe_model.save_model(EUROPE_MODEL_PATH)
    print(f"Saved Europe model → {EUROPE_MODEL_PATH}")
    print(f"  Trees: {europe_model.num_trees()} (best iteration: {europe_model.best_iteration})\n")

    print("=" * 60)
    print("Done.")
    print("=" * 60)


if __name__ == "__main__":
    main()