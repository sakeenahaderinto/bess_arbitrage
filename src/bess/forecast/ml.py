import lightgbm as lgb
import pandas as pd
import numpy as np

def build_features(prices: pd.Series) -> pd.DataFrame:
    """
    Builds a feature matrix from the price Series.
    Features include calendar features, lagged prices, and rolling statistics.
    """
    df = pd.DataFrame({"price": prices})
    
    # 1. Calendar/Time features
    # Why: Electricity prices have strong daily, weekly, and seasonal patterns
    # due to human demand cycles and solar generation profiles.
    df["hour"] = df.index.hour
    df["dayofweek"] = df.index.dayofweek
    df["month"] = df.index.month
    
    # 2. Lagged Prices
    # Why: Autoregressive effects. The price 1 hour ago (momentum), 24 hours ago 
    # (yesterday's identical hour), 48h and 168h (last week) are strong predictors.
    df["lag_1h"] = df["price"].shift(1)
    df["lag_24h"] = df["price"].shift(24)
    df["lag_48h"] = df["price"].shift(48)
    df["lag_168h"] = df["price"].shift(168)
    
    # 3. Rolling Statistics
    # Why: Captures recent volatility and baseline trends over the past day.
    df["rolling_24h_mean"] = df["price"].shift(1).rolling(window=24).mean()
    df["rolling_24h_std"] = df["price"].shift(1).rolling(window=24).std()
    
    # Rolling 7-day same-hour mean
    # Why: Extremely powerful feature that captures structural weekday vs weekend
    # differences at a specific hour without relying on a single 'lag_168h' point that might be an outlier.
    # Since we need this to be strictly causal, we use the average of the last 7 instances of that specific hour.
    # We can do this cleanly by creating a shifted dataframe of 7 days:
    lags = [df["price"].shift(24 * d) for d in range(1, 8)]
    # Concatenate these aligned lags and take the mean along the column axis
    df["rolling_7d_same_hour_mean"] = pd.concat(lags, axis=1).mean(axis=1)
    
    return df

def train_model(historical_prices: pd.Series) -> lgb.Booster:
    """
    Trains a LightGBM model on historical price data with a 75/25 train/val split.
    
    Args:
        historical_prices: A pandas Series with a DateTimeIndex containing historical prices
        
    Returns:
        Trained lightgbm Booster object
    """
    # Build complete feature matrix
    df = build_features(historical_prices)
    
    # Drop NaNs that appear due to lags (specifically the 168h lag loses the first week)
    df = df.dropna()
    
    if df.empty:
        raise ValueError("Not enough historical data to generate features. Need at least > 168 hours.")
        
    # Standard Time Series split (75/25) 
    split_idx = int(len(df) * 0.75)
    
    train = df.iloc[:split_idx]
    val = df.iloc[split_idx:]
    
    # Define features and target 
    target = "price"
    features = [c for c in df.columns if c != target]
    
    # Create LightGBM datasets
    train_data = lgb.Dataset(train[features], label=train[target])
    val_data = lgb.Dataset(val[features], label=val[target], reference=train_data)
    
    # Basic robust hyperparameters for standard regression
    params = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "verbose": -1,
        "random_state": 42
    }
    
    # Train model
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
    Uses the trained LightGBM model to forecast 24 hours of prices for the target date,
    step-by-step to prevent lookahead bias (though as a proxy we can use true prior data
    if we assume we're predicting the whole day-ahead vector).
    
    The Prompt says: "Builds the feature row for each hour of forecast_date using 
    only data available before that date (no lookahead). Returns a 24-element Series."
    
    Because an ML auto-regressive model requires lag_1h, predicting hour 02 depends 
    on predicting hour 01 first (if we were strictly running live). But for Day-Ahead
    Market (DAM) forecasting, we must predict ALL 24 hours at e.g. 10:00 AM the day before.
    
    So for target = 2024-01-02, the last *known* price 
    is 2024-01-01 23:00. This means `lag_1h` for 08:00 on the forecast date is technically
    unavailable. To satisfy the prompt without an overly complex iterative inference loop, 
    we iteratively build the day.
    """
    target_date_dt = pd.to_datetime(forecast_date).date()
    
    # We only take the historical prices leading up strictly to the day before the target.
    # We exclude the target date itself so the ML has no way to cheat.
    past_prices = historical_prices[historical_prices.index.date < target_date_dt].copy()
    
    if len(past_prices) < 168:
        # We don't have enough history to construct the 168h lag feature.
        # Fall back to lag-24 naive to keep things running
        from bess.forecast.naive import forecast_lag24
        return forecast_lag24(historical_prices, forecast_date)
        
    predictions = []
    
    # Create the 24 timestamps for the forecast date
    # In strictly localized data, we'd use the same timezone.
    tz = historical_prices.index.tz
    target_idx = pd.date_range(start=forecast_date, periods=24, freq="h", tz=tz)
    
    for i in range(24):
        # We append a placeholder for the price we are about to predict.
        # The value doesn't matter for features because `build_features` shifts down for lags
        # so target `price` is structurally ignored when building features for *this* step.
        current_ts = target_idx[i]
        
        # Append a temporary dummy value to allow the dataframe to align exactly to the prediction hour
        # This will be popped off and replaced by the correct prediction at the end of the loop
        past_prices.loc[current_ts] = 0.0 
        
        # Re-build features up to this new dummy timestamp
        # (Optimisation: building the full series every loop is slow for production, but 
        # clean and robust for our current analytical scale).
        features_df = build_features(past_prices)
        
        # Extract just the last row (our prediction target row)
        # Drop the "price" column since we're inferencing
        pred_features = features_df.iloc[[-1]].drop(columns=["price"])
        
        # Predict using Booster
        pred_value = model.predict(pred_features)[0]
        predictions.append(pred_value)
        
        # Now replace the dummy value in `past_prices` with our actual PREDICTION.
        # This correctly propagates our prediction as the `lag_1h` constraint for the NEXT hour!
        past_prices.loc[current_ts] = pred_value
        
    return pd.Series(predictions, index=target_idx)
