import gc
import lightgbm as lgb
import pandas as pd
import numpy as np

# Canonical feature column order, defined once at module level.
# This eliminates the need to call build_features() just to get column names
# during iterative inference — which was rebuilding a full ~8,760-row DataFrame
# on every forecast day in the backtest loop.
FEATURE_COLS = [
    "hour", "dayofweek", "month",
    "lag_1h", "lag_24h", "lag_48h", "lag_168h",
    "rolling_24h_mean", "rolling_24h_std",
    "rolling_7d_same_hour_mean",
]


def build_features(prices: pd.Series) -> pd.DataFrame:
    """
    Builds a feature matrix from the price Series.
    Features include calendar features, lagged prices, and rolling statistics.

    Args:
        prices: A pd.Series with a UTC DatetimeIndex at hourly frequency.
                The index is sorted and resampled to a strict hourly grid
                internally. Any gaps introduced by asfreq are left as NaN
                and will be dropped downstream by the caller.

    Returns:
        pd.DataFrame with one row per hour and one column per feature,
        plus a 'price' column representing the target variable.
        Column order matches FEATURE_COLS (plus 'price').
    """
    # Defensively ensure chronological contiguous hourly data.
    # asfreq('h') enforces a strict hourly grid which means that any missing
    # hours become NaN rather than silently causing shift(24) to point to the
    # wrong timestamp.
    prices = prices.sort_index().asfreq("h")
    df = pd.DataFrame({"price": prices})

    # Calendar/Time features
    # Why: Electricity prices have strong daily, weekly, and seasonal patterns
    # due to human demand cycles and solar generation profiles.
    df["hour"] = df.index.hour
    df["dayofweek"] = df.index.dayofweek
    df["month"] = df.index.month

    # Lagged Prices
    # Why: Autoregressive effects. The price 1 hour ago (momentum), 24 hours ago
    # (yesterday's identical hour), 48h and 168h (last week) are strong predictors.
    df["lag_1h"] = df["price"].shift(1)
    df["lag_24h"] = df["price"].shift(24)
    df["lag_48h"] = df["price"].shift(48)
    df["lag_168h"] = df["price"].shift(168)

    # Rolling Statistics
    # Why: Captures recent volatility and baseline trends over the past day.
    # We shift(1) before rolling to ensure the window never includes the current
    # price, keeping the features strictly causal.
    df["rolling_24h_mean"] = df["price"].shift(1).rolling(window=24).mean()
    df["rolling_24h_std"] = df["price"].shift(1).rolling(window=24).std()

    # Rolling 7-day same-hour mean
    # Why: Captures structural weekday vs weekend differences at a specific hour
    # without relying on a single lag_168h point that might be an outlier.
    # We average the same hour across the previous 7 days using explicit shifts
    # to keep the feature strictly causal.
    # Accumulate with addition rather than pd.concat to avoid materialising
    # 7 full-length Series + a wide intermediate DataFrame simultaneously.
    # sum() then divide is equivalent and uses O(N) working memory instead of O(7N).
    same_hour_sum = sum(df["price"].shift(24 * d) for d in range(1, 8))
    df["rolling_7d_same_hour_mean"] = same_hour_sum / 7

    # Enforce FEATURE_COLS order so the returned DataFrame always matches
    # what the model expects, regardless of future column additions.
    return df[["price"] + FEATURE_COLS]


def train_model(historical_prices: pd.Series) -> lgb.Booster:
    """
    Trains a LightGBM model on historical price data with a 75/25 train/val split.

    Args:
        historical_prices: A pd.Series with a UTC DatetimeIndex containing
                           historical hourly prices. A practical minimum is
                           30 days — the 168-hour lag drops the first week of
                           rows entirely, and the remaining data needs to be
                           large enough for a meaningful 75/25 split.

    Returns:
        Trained lgb.Booster object with best_iteration set by early stopping.

    Raises:
        ValueError: If there is not enough data to produce any training rows
                    after feature engineering and NaN dropping.
    """
    df = build_features(historical_prices)

    # Drop NaNs introduced by lags — the 168h lag loses the first 7 days of rows.
    df = df.dropna()

    if df.empty:
        raise ValueError(
            "Not enough historical data to generate features. Building the 168-hour "
            "(7-day) lag drops the first week of rows entirely. A practical minimum "
            "is at least 30 days of data to have enough rows for a robust 75/25 split."
        )

    # Chronological 75/25 split — random splits are wrong for time series because
    # they leak future prices into the training set via lag features.
    split_idx = int(len(df) * 0.75)
    train = df.iloc[:split_idx]
    val = df.iloc[split_idx:]

    target = "price"

    train_data = lgb.Dataset(train[FEATURE_COLS], label=train[target])
    val_data = lgb.Dataset(val[FEATURE_COLS], label=val[target], reference=train_data)

    # Conservative defaults suitable for a moderate-sized tabular regression problem.
    # seed ensures repeatable results when bagging or other randomness is enabled.
    params = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "verbose": -1,
        "seed": 42,
    }

    booster = lgb.train(
        params,
        train_data,
        num_boost_round=500,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
    )

    # Free the training datasets from memory — they are not needed after training.
    del train_data, val_data, train, val, df
    gc.collect()

    return booster


def forecast_ml(model: lgb.Booster, historical_prices: pd.Series, forecast_date: str) -> pd.Series:
    """
    Forecasts 24 hourly prices for forecast_date using an iterative inference
    loop that prevents lookahead bias.

    In day-ahead forecasting, all 24 hours must be predicted at once (typically
    by 10am the day before). Because lag_1h for hour H depends on hour H-1 of
    the forecast day — which hasn't happened yet — we predict hour 0 first, feed
    that prediction back as lag_1h for hour 1, and continue iteratively.

    Feature rows are built incrementally (O(1) per step) using the module-level
    FEATURE_COLS constant rather than reconstructing the full feature DataFrame
    on every call, which previously cost one ~8,760-row allocation per forecast day.

    Args:
        model: Trained lgb.Booster returned by train_model.
        historical_prices: Full price history with a UTC DatetimeIndex.
                           The target date and any later dates are excluded
                           internally so the model cannot cheat.
        forecast_date: Target date in 'YYYY-MM-DD' format.

    Returns:
        pd.Series of 24 predicted prices with a UTC DatetimeIndex covering
        the forecast date.

    Falls back to forecast_lag24 if there is less than 168 hours of history,
        since the 168h lag feature cannot be constructed.
    """
    target_date_dt = pd.to_datetime(forecast_date).date()

    # Exclude the target date and anything after it — strict no-lookahead rule.
    past_prices = historical_prices[historical_prices.index.date < target_date_dt].copy()

    if len(past_prices) < 168:
        # Not enough history to construct the 168h lag feature.
        # Pass past_prices (not historical_prices) to preserve the no-lookahead rule.
        from bess.forecast.naive import forecast_lag24
        return forecast_lag24(past_prices, forecast_date)

    predictions = []

    tz = historical_prices.index.tz
    target_idx = pd.date_range(start=forecast_date, periods=24, freq="h", tz=tz)

    # Pre-allocate 24 NaN slots for the forecast window in one concat rather than
    # growing past_prices with 24 individual .loc assignments inside the loop.
    # Each .loc assignment to a new index key triggers a Series reallocation;
    # doing 24 of them per day across 365 days = ~8,760 unnecessary reallocations.
    forecast_slots = pd.Series(np.nan, index=target_idx)
    past_prices = pd.concat([past_prices, forecast_slots])

    for i in range(24):
        current_ts = target_idx[i]

        # --- Incremental feature construction (O(1) per step) ---

        # Calendar features
        hour = current_ts.hour
        dayofweek = current_ts.dayofweek
        month = current_ts.month

        # Lag features
        # lag_1h uses the previous prediction to propagate forecast uncertainty
        # forward correctly. For hour 0, there is no prior prediction so we use
        # the last known historical price (iloc[-25] because the 24 NaN slots
        # are now appended at the end).
        lag_1h = predictions[-1] if i > 0 else past_prices.iloc[-(25 - i)]
        lag_24h = past_prices.loc[current_ts - pd.Timedelta(hours=24)]
        lag_48h = past_prices.loc[current_ts - pd.Timedelta(hours=48)]
        lag_168h = past_prices.loc[current_ts - pd.Timedelta(hours=168)]

        # Rolling 24h statistics
        # The window combines the last (24 - i) known prices with the i predictions
        # made so far, ensuring the window always contains exactly 24 values and
        # reflects the most recent available information at each step.
        if i == 0:
            window_24h = past_prices.iloc[-(24 + 24):-(24)].values  # last 24 known prices
        else:
            known = past_prices.iloc[-(24 + 24 - i):-(24)].values
            window_24h = np.concatenate([known, predictions])

        rolling_24h_mean = float(np.mean(window_24h))
        # ddof=1 matches pandas rolling.std() default for an unbiased estimate
        rolling_24h_std = float(np.std(window_24h, ddof=1))

        # 7-day same-hour mean
        # Average the same hour across the 7 prior days for a robust structural baseline.
        same_hour_lags = [
            past_prices.loc[current_ts - pd.Timedelta(hours=24 * d)]
            for d in range(1, 8)
        ]
        rolling_7d_same_hour_mean = float(np.mean(same_hour_lags))

        # --- Predict ---
        row_dict = {
            "hour": hour,
            "dayofweek": dayofweek,
            "month": month,
            "lag_1h": lag_1h,
            "lag_24h": lag_24h,
            "lag_48h": lag_48h,
            "lag_168h": lag_168h,
            "rolling_24h_mean": rolling_24h_mean,
            "rolling_24h_std": rolling_24h_std,
            "rolling_7d_same_hour_mean": rolling_7d_same_hour_mean,
        }

        # Build a (1, n_features) numpy array in FEATURE_COLS order rather than
        # constructing a pd.DataFrame for each of the 24 × 365 = 8,760 predictions.
        # DataFrame construction has significant Python-level overhead at that scale;
        # LightGBM's predict() accepts numpy arrays directly with no accuracy change.
        pred_array = [[
            row_dict["hour"], row_dict["dayofweek"], row_dict["month"],
            row_dict["lag_1h"], row_dict["lag_24h"], row_dict["lag_48h"], row_dict["lag_168h"],
            row_dict["rolling_24h_mean"], row_dict["rolling_24h_std"],
            row_dict["rolling_7d_same_hour_mean"],
        ]]

        # Use best_iteration so we predict with the optimal tree count found
        # by early stopping, not all trained trees (which may include overfitted rounds).
        pred_value = float(model.predict(pred_array, num_iteration=model.best_iteration)[0])
        predictions.append(pred_value)

        # Write the prediction into the pre-allocated slot. Because the index key
        # already exists (from the concat above), this is an in-place update with
        # no Series reallocation.
        past_prices.loc[current_ts] = pred_value

    return pd.Series(predictions, index=target_idx)


# ============================================================
# Pre-trained model loading (deployed app)
# ============================================================

# Zones supported by the GB-specific model.
# GB-NIR trades separately from Great Britain but is grouped here because it
# shares enough structural similarity (island grid, sterling pricing) to be
# better served by the GB model than the continental European one.
_GB_ZONES = {"GB", "GB-NIR"}

# Zones supported by the continental European pooled model.
EUROPEAN_ZONES_SUPPORTED = [
    "DE", "FR", "NL", "ES", "IT-NO", "SE-SE3", "PL", "BE",
]

GB_MODEL_PATH = "models/bess_gb.lgb"
EUROPE_MODEL_PATH = "models/bess_europe.lgb"


def train_model_pooled(zone_series: list[pd.Series]) -> lgb.Booster:
    """
    Train a LightGBM model on price data pooled across multiple zones.

    Each zone's Series is processed through build_features() independently
    before concatenation. This avoids the duplicate DatetimeIndex problem
    that occurs when naively concatenating raw price Series from multiple
    zones — all zones share the same timestamps, so pd.concat produces
    duplicate index labels that asfreq('h') cannot reindex over.

    Args:
        zone_series: List of pd.Series, one per zone, each with a UTC
                     DatetimeIndex. At least one must produce non-empty
                     features after NaN dropping.

    Returns:
        Trained lgb.Booster with best_iteration set by early stopping.

    Raises:
        ValueError: If no valid feature rows are produced across all zones.
    """
    feature_frames = []
    for s in zone_series:
        df = build_features(s).dropna()
        if not df.empty:
            feature_frames.append(df)

    if not feature_frames:
        raise ValueError(
            "No valid feature rows produced across any zone. "
            "Check that each Series has at least 30 days of data."
        )

    # Reset index so the pooled DataFrame has no duplicate timestamp labels.
    # The DatetimeIndex is not used downstream — only the feature columns matter.
    pooled_df = pd.concat(feature_frames, ignore_index=True)

    split_idx = int(len(pooled_df) * 0.75)
    train = pooled_df.iloc[:split_idx]
    val = pooled_df.iloc[split_idx:]

    target = "price"
    train_data = lgb.Dataset(train[FEATURE_COLS], label=train[target])
    val_data = lgb.Dataset(val[FEATURE_COLS], label=val[target], reference=train_data)

    params = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "verbose": -1,
        "seed": 42,
    }

    booster = lgb.train(
        params,
        train_data,
        num_boost_round=500,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
    )

    del train_data, val_data, train, val, pooled_df, feature_frames
    gc.collect()

    return booster


def load_pretrained_model(zone: str) -> lgb.Booster:
    """
    Load the appropriate pre-trained LightGBM booster for the given zone.

    Two models are maintained:
      - bess_gb.lgb     : Trained on GB data only. GB is an island grid with
                          no synchronous AC interconnection to continental Europe,
                          prices in £, and a more pronounced evening ramp due to
                          the absence of cross-border smoothing via interconnectors.
                          A model trained on continental patterns would systematically
                          underestimate peak amplitudes for GB.
      - bess_europe.lgb : Trained on pooled DE/FR/NL/ES/IT-NO/SE-SE3/PL/BE data.
                          Pooling across zones gives the model a richer distribution
                          of price regimes (hydro-dominated Nordics, solar-heavy
                          Southern Europe, gas-indexed Central Europe) without
                          requiring a separate model per zone.

    Args:
        zone: Zone code from EUROPEAN_ZONES_SUPPORTED or _GB_ZONES.

    Returns:
        lgb.Booster ready for inference.

    Raises:
        FileNotFoundError: If the model file is missing — run train_models.py locally
                           and commit the .lgb files before deploying.
    """
    model_path = GB_MODEL_PATH if zone in _GB_ZONES else EUROPE_MODEL_PATH
    try:
        return lgb.Booster(model_file=model_path)
    except Exception:
        raise FileNotFoundError(
            f"Pre-trained model not found at '{model_path}'. "
            "Run train_models.py locally and commit the .lgb files to the repo."
        )