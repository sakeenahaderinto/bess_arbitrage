import gc
import lightgbm as lgb
import pandas as pd
import numpy as np

# Canonical feature column order, defined once at module level.
# This avoids repeatedly calling build_features() just to retrieve column names
# during iterative inference, which improves performance during backtesting.
FEATURE_COLS = [
    "hour", "dayofweek", "month",
    "lag_1h", "lag_24h", "lag_48h", "lag_168h",
    "rolling_24h_mean", "rolling_24h_std",
    "rolling_7d_same_hour_mean",
]


def build_features(prices: pd.Series) -> pd.DataFrame:
    """Constructs a feature matrix from the provided price Series.

    Features include calendar-based attributes, lagged prices, and rolling statistics.

    Args:
        prices: A pd.Series with an hourly UTC DatetimeIndex. The index is sorted
                and strictly resampled to an hourly frequency internally. Any missing
                gaps resulting from this are left as NaN for downstream handling.

    Returns:
        pd.DataFrame: A DataFrame containing one row per hour and one column per
        feature, along with a 'price' column as the target variable. The column
        order strictly matches `FEATURE_COLS` (plus 'price').
    """
    # Enforce a strict chronological hourly grid to prevent misalignment
    # when computing lagged features like shift(24).
    prices = prices.sort_index().asfreq("h")
    df = pd.DataFrame({"price": prices})

    # Calendar features to capture daily, weekly, and seasonal demand cycles.
    df["hour"] = df.index.hour
    df["dayofweek"] = df.index.dayofweek
    df["month"] = df.index.month

    # Lagged prices to capture autoregressive effects over multiple time horizons.
    df["lag_1h"] = df["price"].shift(1)
    df["lag_24h"] = df["price"].shift(24)
    df["lag_48h"] = df["price"].shift(48)
    df["lag_168h"] = df["price"].shift(168)

    # Rolling statistics to capture recent volatility and trends.
    # A shift(1) is applied prior to rolling to maintain strict causality.
    df["rolling_24h_mean"] = df["price"].shift(1).rolling(window=24).mean()
    df["rolling_24h_std"] = df["price"].shift(1).rolling(window=24).std()

    # Rolling average of the same hour across the preceding 7 days.
    # This mitigates reliance on a single lagged data point that might be an outlier.
    # Evaluated cumulatively to minimize memory allocation overhead.
    same_hour_sum = sum(df["price"].shift(24 * d) for d in range(1, 8))
    df["rolling_7d_same_hour_mean"] = same_hour_sum / 7

    # Enforce canonical column ordering for model compatibility.
    return df[["price"] + FEATURE_COLS]


def train_model(historical_prices: pd.Series) -> lgb.Booster:
    """Trains a LightGBM model on historical price data using a chronological split.

    Args:
        historical_prices: A pd.Series with an hourly UTC DatetimeIndex. A minimum
                           of 30 days is recommended, as the 168-hour lag feature
                           drops the initial week of data entirely.

    Returns:
        lgb.Booster: The trained model, with best_iteration configured via early stopping.

    Raises:
        ValueError: If insufficient data remains after feature preparation and filtering.
    """
    df = build_features(historical_prices)

    # Drop rows containing NaNs introduced by lagged features.
    df = df.dropna()

    if df.empty:
        raise ValueError(
            "Not enough historical data to generate features. Building the 168-hour "
            "(7-day) lag drops the first week of rows entirely. A practical minimum "
            "is at least 30 days of data to have enough rows for a robust 75/25 split."
        )

    # Apply a chronological 75/25 train-validation split to prevent temporal data leakage.
    split_idx = int(len(df) * 0.75)
    train = df.iloc[:split_idx]
    val = df.iloc[split_idx:]

    target = "price"

    train_data = lgb.Dataset(train[FEATURE_COLS], label=train[target])
    val_data = lgb.Dataset(val[FEATURE_COLS], label=val[target], reference=train_data)

    # Configure model hyperparameters for tabular regression with a fixed seed for reproducibility.
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

    # Explicitly release training datasets from memory.
    del train_data, val_data, train, val, df
    gc.collect()

    return booster


def forecast_ml(model: lgb.Booster, historical_prices: pd.Series, forecast_date: str) -> pd.Series:
    """Generates a 24-hour day-ahead price forecast using iterative inference.

    An iterative loop is employed to prevent lookahead bias. Predictions for the
    current hour are recursively fed back into the feature set (e.g., as `lag_1h`)
    for subsequent hourly predictions.

    Args:
        model: The trained `lgb.Booster` instance.
        historical_prices: The complete historical price Series. The target date
                           and all subsequent dates are strictly excluded internally.
        forecast_date: The target forecast date in 'YYYY-MM-DD' format.

    Returns:
        pd.Series: A 24-hour sequence of forecasted prices corresponding to the target date.
    """
    target_date_dt = pd.to_datetime(forecast_date).date()

    # Strictly isolate past data to enforce a no-lookahead constraint.
    past_prices = historical_prices[historical_prices.index.date < target_date_dt].copy()

    if len(past_prices) < 168:
        # Fall back to a naive forecast if insufficient history exists for 168h lags.
        from bess.forecast.naive import forecast_lag24
        return forecast_lag24(past_prices, forecast_date)

    predictions = []

    tz = historical_prices.index.tz
    target_idx = pd.date_range(start=forecast_date, periods=24, freq="h", tz=tz)

    # Pre-allocate the forecast window slots to avoid repeated Series reallocations inside the loop.
    forecast_slots = pd.Series(np.nan, index=target_idx)
    past_prices = pd.concat([past_prices, forecast_slots])

    for i in range(24):
        current_ts = target_idx[i]

        # --- Incremental feature construction (O(1) per step) ---

        # Calendar features
        hour = current_ts.hour
        dayofweek = current_ts.dayofweek
        month = current_ts.month

        # Construct lagged features, propagating previous predictions recursively for lag_1h.
        lag_1h = predictions[-1] if i > 0 else past_prices.iloc[-(25 - i)]
        lag_24h = past_prices.loc[current_ts - pd.Timedelta(hours=24)]
        lag_48h = past_prices.loc[current_ts - pd.Timedelta(hours=48)]
        lag_168h = past_prices.loc[current_ts - pd.Timedelta(hours=168)]

        # Compute sliding 24-hour statistics dynamically incorporating prior predictions.
        if i == 0:
            window_24h = past_prices.iloc[-(24 + 24):-(24)].values  # last 24 known prices
        else:
            known = past_prices.iloc[-(24 + 24 - i):-(24)].values
            window_24h = np.concatenate([known, predictions])

        rolling_24h_mean = float(np.mean(window_24h))
        # ddof=1 matches pandas rolling.std() default for an unbiased estimate
        rolling_24h_std = float(np.std(window_24h, ddof=1))

        # Compute the 7-day same-hour rolling average.
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

        # Format the feature array sequentially to match `FEATURE_COLS` expectations.
        pred_array = [[
            row_dict["hour"], row_dict["dayofweek"], row_dict["month"],
            row_dict["lag_1h"], row_dict["lag_24h"], row_dict["lag_48h"], row_dict["lag_168h"],
            row_dict["rolling_24h_mean"], row_dict["rolling_24h_std"],
            row_dict["rolling_7d_same_hour_mean"],
        ]]

        # Evaluate model predictions utilizing the best iteration index.
        pred_value = float(model.predict(pred_array, num_iteration=model.best_iteration)[0])
        predictions.append(pred_value)

        # Assign the prediction in-place to the pre-allocated slot.
        past_prices.loc[current_ts] = pred_value

    return pd.Series(predictions, index=target_idx)


# ============================================================
# Pre-trained model loading (deployed app)
# ============================================================

# Regional groupings for the GB-specific model.
_GB_ZONES = {"GB", "GB-NIR"}

# Zones supported by the continental European pooled model.
EUROPEAN_ZONES_SUPPORTED = [
    "DE", "FR", "NL", "ES", "IT-NO", "SE-SE3", "PL", "BE",
]

GB_MODEL_PATH = "models/bess_gb.lgb"
EUROPE_MODEL_PATH = "models/bess_europe.lgb"


def train_model_pooled(zone_series: list[pd.Series]) -> lgb.Booster:
    """Trains a LightGBM model on price data pooled across multiple geographic zones.

    Args:
        zone_series: A list of individual zone price Series with hourly UTC indexes.

    Returns:
        lgb.Booster: The globally trained model optimized via early stopping.

    Raises:
        ValueError: If no valid feature rows remain across all aggregated zones.
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

    # Reset indices to consolidate independently engineered zone datasets without collision.
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
    """Loads the pre-trained LightGBM booster specific to the supplied zone.

    Args:
        zone: The zone identifier string (e.g., 'GB', 'DE').

    Returns:
        lgb.Booster: The pre-trained predictive model.

    Raises:
        FileNotFoundError: If the relevant model binary is unavailable on disk.
    """
    model_path = GB_MODEL_PATH if zone in _GB_ZONES else EUROPE_MODEL_PATH
    try:
        return lgb.Booster(model_file=model_path)
    except Exception:
        raise FileNotFoundError(
            f"Pre-trained model not found at '{model_path}'. "
            "Run train_models.py locally and commit the .lgb files to the repo."
        )