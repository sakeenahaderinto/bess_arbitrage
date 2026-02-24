# BESS Arbitrage Optimiser

An analytical modelling and forecasting suite for evaluating Battery Energy Storage System (BESS) arbitrage revenue across European electricity markets.

Grid-scale batteries generate revenue by shifting energy temporally — charging during periods of excess renewable generation when prices are low, and discharging during peak demand when prices are high. This project uses real Day-Ahead Market (DAM) prices to calculate theoretical maximum revenue and compare it against naive and machine learning forecasting strategies.

[Streamlit Dashboard link](https://bessarbitragedemo.streamlit.app/)

## Features

- **Forecast vs Foresight** — Simulates a single day to calculate the efficiency gap between a chosen forecast model and mathematically optimal dispatch
- **Backtester** — Simulates multi-month BESS deployments with real market data, reporting annualised revenue, payback period, cumulative P&L, and consistency ratio
- **Region Comparison** — Compares arbitrage viability across European market zones by correlating price volatility to annualised yield
- **Robust Data Pipeline** — Handles API rate limiting, chunked date range fetching, missing data, and solver failures explicitly so backtest metrics stay clean

## Quick Start

1. Clone the repository
2. Install the `uv` package manager
3. Install dependencies: `uv sync`
4. Copy the environment template: `cp .env.example .env`
5. Add your ElectricityMaps API key to `.env`
6. Run the app: `uv run streamlit run app/main.py`

I used an academic-tier ElectricityMaps API key, which gives access to day-ahead price data for all supported European zones.


## Methodology

### Optimisation

The core optimiser uses Linear Programming (via the HiGHS solver) to find the mathematically optimal charge/discharge schedule for a given price series. The objective function maximises net revenue while respecting battery capacity, power limits, round-trip efficiency, and a per-MWh degradation cost.

Round-trip efficiency is split symmetrically between charge and discharge (`eta = sqrt(roundtrip_eff)`), reflecting the standard assumption when only a single round-trip figure is available. A terminal SOC constraint ensures the battery ends each day at its starting state of charge, preventing the solver from exploiting the end of the horizon by discharging everything regardless of future value.

Optionally, a MILP mode can be enabled which adds a binary variable to explicitly forbid simultaneous charge and discharge. This is more physically realistic but significantly slower to solve.

### Forecasting

Three forecast models are supported, each addressing a specific weakness of the previous:

| Model | Approach | Breaks when |
|---|---|---|
| Naive lag-24 | Yesterday's prices | Day-of-week transitions, bank holidays |
| Naive rolling-7 | Mean of same hour over 7 prior days | Persistent regime changes (cold snaps, seasonal shifts) |
| LightGBM | Gradient boosted trees on calendar features, lagged prices, and rolling statistics | Insufficient history (< 168 hours), structural market changes |

The ML model uses iterative inference for day-ahead forecasting where each hour's prediction is fed back as the `lag_1h` feature for the next hour, avoiding lookahead bias and trained on a chronological 75/25 split.

### Metrics

**Efficiency Gap** — ratio of forecast revenue to perfect foresight revenue. A value of 0.82 means the forecast captured 82% of the theoretical maximum arbitrage value available that day.

**Consistency Ratio** — mean daily profit divided by its standard deviation. A measure of revenue stability, not a Sharpe ratio (no risk-free rate, no annualisation).

**Annualised Revenue** — mean daily profit scaled to 365 days. The number of basis days used in the calculation is always surfaced so short backtests can be interpreted with appropriate caution.

>***Note that annualised figures extrapolated from short windows (< 3 months) should be interpreted cautiously, as a single high-volatility month can skew the projection significantly.***

Failed solver days are excluded from all metrics and reported separately.

## Data Source

Market data is sourced from the [ElectricityMaps API](https://electricitymaps.com/). The supported zone list is scoped to European markets with confirmed day-ahead price coverage.
