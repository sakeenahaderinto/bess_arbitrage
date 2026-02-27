import lightgbm as lgb
import pandas as pd
import numpy as np


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
    """
    # Defensively ensure chronological contiguous hourly data.
    # asfreq('h') enforces a strict hourly grid which meanst that any missing hours become NaN
    # rather than silently causing shift(24) to point to the wrong timestamp.
    prices = prices.sort_index().asfreq('h')
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
    lags = [df["price"].shift(24 * d) for d in range(1, 8)]
    df["rolling_7d_same_hour_mean"] = pd.concat(lags, axis=1).mean(axis=1)

    return df


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
    features = [c for c in df.columns if c != target]

    train_data = lgb.Dataset(train[features], label=train[target])
    val_data = lgb.Dataset(val[features], label=val[target], reference=train_data)

    # Conservative defaults suitable for a moderate-sized tabular regression problem.
    # seed ensures repeatable results when bagging or other randomness is enabled.
    params = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "verbose": -1,
        "seed": 42
    }

    booster = lgb.train(
        params,
        train_data,
        num_boost_round=500,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
    )

    return booster


def forecast_ml(model: lgb.Booster, historical_prices: pd.Series, forecast_date: str) -> pd.Series:
    """
    Forecasts 24 hourly prices for forecast_date using an iterative inference
    loop that prevents lookahead bias.

    In day-ahead forecasting, all 24 hours must be predicted at once (typically
    by 10am the day before). Because lag_1h for hour H depends on hour H-1 of
    the forecast day — which hasn't happened yet — we predict hour 0 first, feed
    that prediction back as lag_1h for hour 1, and continue iteratively.

    Feature rows are built incrementally rather than rebuilding
    the full feature matrix on every iteration. This makes the function 
    suitable for production use cases where inference latency matters.

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

    # Build features once on past_prices to get the canonical column order.
    # We use the column names from this call to ensure pred_features always
    # matches the feature order the model was trained on.
    initial_features_df = build_features(past_prices)
    feature_cols = [c for c in initial_features_df.columns if c != "price"]

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
        # the last known price.
        lag_1h = predictions[-1] if i > 0 else past_prices.iloc[-1]
        lag_24h = past_prices.loc[current_ts - pd.Timedelta(hours=24)]
        lag_48h = past_prices.loc[current_ts - pd.Timedelta(hours=48)]
        lag_168h = past_prices.loc[current_ts - pd.Timedelta(hours=168)]

        # Rolling 24h statistics
        # The window combines the last (24 - i) known prices with the i predictions
        # made so far, ensuring the window always contains exactly 24 values and
        # reflects the most recent available information at each step.
        if i == 0:
            window_24h = past_prices.iloc[-24:].values
        else:
            window_24h = np.concatenate([past_prices.iloc[-(24 - i):].values, predictions])

        rolling_24h_mean = np.mean(window_24h)
        # ddof=1 matches pandas rolling.std() default for an unbiased estimate
        rolling_24h_std = np.std(window_24h, ddof=1)

        # 7-day same-hour mean
        # Average the same hour across the 7 prior days for a robust structural baseline.
        same_hour_lags = [
            past_prices.loc[current_ts - pd.Timedelta(hours=24 * d)]
            for d in range(1, 8)
        ]
        rolling_7d_same_hour_mean = np.mean(same_hour_lags)

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
            "rolling_7d_same_hour_mean": rolling_7d_same_hour_mean
        }

        # Enforce column order to match model training exactly
        pred_features = pd.DataFrame([row_dict], columns=feature_cols)

        # Use best_iteration so we predict with the optimal tree count found
        # by early stopping, not all trained trees (which may include overfitted rounds)
        pred_value = model.predict(pred_features, num_iteration=model.best_iteration)[0]
        predictions.append(pred_value)

        # Update past_prices with the prediction so that lag lookups for subsequent
        # hours resolve correctly. This is what makes lag_1h for hour H+1 equal to
        # the prediction for hour H rather than the last known historical price.
        # NOTE: The dummy value inserted here is safe because all features use
        # shifted prices ie. the current price is never used as a direct input.
        # If any unshifted feature is added in future, this approach would need
        # to be revisited.
        past_prices.loc[current_ts] = pred_value

    return pd.Series(predictions, index=target_idx)