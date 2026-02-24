
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import timedelta

from bess.data.cache import get_prices_cached
from bess.forecast.naive import forecast_lag24, forecast_rolling7
from bess.forecast.ml import train_model, forecast_ml
from bess.backtest.engine import run_backtest
from bess.metrics import efficiency_gap, daily_summary, payback_period

st.set_page_config(page_title="Backtester", page_icon="📈", layout="wide")

@st.cache_data
def load_zones():
    return get_zones()

def render_sidebar():
    st.sidebar.header("Battery Parameters")
    
    # API Status Indicator
    try:
        zones = load_zones()
        st.sidebar.success("✓ API connected")
    except Exception as e:
        st.sidebar.error(f"✗ API error: {e}")
        zones = ["DE"] # Fallback just so the UI doesn't completely crash

    # Zone Selector
    zone = st.sidebar.selectbox(
        "Market Zone", 
        options=zones, 
        index=zones.index("GB") if "GB" in zones else 0
    )
    st.session_state["zone"] = zone
    
    # Battery capacity
    e_max_mwh = st.sidebar.number_input(
        "Battery capacity (MWh)", 
        min_value=0.1, max_value=1000.0, value=1.0, step=0.1
    )
    st.session_state["e_max_mwh"] = e_max_mwh
    
    # Power rating
    p_max_mw = st.sidebar.number_input(
        "Power rating (MW)", 
        min_value=0.1, max_value=1000.0, value=0.5, step=0.1
    )
    st.session_state["p_max_mw"] = p_max_mw
    
    # Efficiency
    eff_pct = st.sidebar.slider(
        "Round-trip efficiency (%)", 
        min_value=50, max_value=100, value=90, step=1
    )
    st.session_state["roundtrip_eff"] = eff_pct / 100.0
    
    # Degradation Cost
    deg_cost = st.sidebar.number_input(
        "Degradation cost (£/MWh)", 
        min_value=0.0, value=5.0, step=0.5
    )
    st.session_state["deg_cost_per_mwh"] = deg_cost
    
    # MILP Toggle
    enforce_milp = st.sidebar.checkbox(
        "Enforce no simultaneous charge/discharge",
        value=False
    )
    st.session_state["enforce_new_milp"] = enforce_milp  # Or standard simple modeling

def load_historical_prices(zone, start_date, end_date):
    """Fetch cached prices."""
    return get_prices_cached(zone, start_date, end_date)

@st.cache_resource
def load_ml_model(prices):
    with st.spinner("Training LightGBM model on historical data..."):
        return train_model(prices)

def main():
    st.title("📈 Backtester")
    st.markdown("Run the optimiser over an extended historical period to measure long-term value creation.")
    render_sidebar()
    
    # 1. Check dependencies
    required_keys = ["zone", "e_max_mwh", "p_max_mw", "roundtrip_eff", "deg_cost_per_mwh"]
    for key in required_keys:
        if key not in st.session_state:
            st.warning("Sidebar parameters missing. Please visit the main page first.")
            return

    zone = st.session_state["zone"]
    currency_symbol = "£" if zone.startswith("GB") else "€"
            
    # 2. Setup Backtesting Form UI
    with st.form("backtest_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            date_range = st.date_input(
                "Date Range",
                value=(pd.to_datetime("2024-05-01"), pd.to_datetime("2024-05-31"))
            )
            
        with col2:
            capital_cost = st.number_input(
                f"Battery Capital Cost ({currency_symbol})",
                min_value=0.0, value=150000.0, step=10000.0,
                help="Used to compute the payback period."
            )
            
        with col3:
            model_selection = st.radio(
                "Select Forecast Model",
                options=["Naive (lag-24)", "Naive (7-day avg)", "ML Model"],
                horizontal=True
            )
            
        submitted = st.form_submit_button("Run Backtest")
        
    if submitted:
        if isinstance(date_range, tuple) and len(date_range) == 2:
            start_date, end_date = date_range
        else:
            st.error("Please select both a start and end date.")
            return
            
        start_date_str = start_date.strftime("%Y-%m-%d")
        end_date_str = end_date.strftime("%Y-%m-%d")
        
        forecast_fn_map = {
            "Naive (lag-24)": (forecast_lag24, 1),
            "Naive (7-day avg)": (forecast_rolling7, 7),
            "ML Model": (forecast_ml, 365)
        }
        forecast_fn, lookback_days = forecast_fn_map[model_selection]
        
        # We need enough history for the forecast model. 7 days lookback means we need 
        # to fetch starting from target_date - 7 days.
        start_fetch = (start_date - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
        
        # UI Elements for progress
        st.markdown("---")
        progress_text = st.empty()
        progress_bar = st.progress(0)
        
        def update_progress(current, total):
            pct = int((current / total) * 100)
            progress_bar.progress(pct)
            progress_text.text(f"Running day {current} of {total}...")
            
        prices = load_historical_prices(zone, start_fetch, end_date_str)
        
        # check zero conditions where the API key might not permit pricing
        if prices.empty or sum(prices) == 0:
            st.warning("No price data found (or mock zeros). Using synthetic variance data to demonstrate Tab 2!")
            import numpy as np
            index = pd.date_range(start=start_fetch, end=end_date_str + " 23:00", freq="h", tz="UTC")
            hours = np.arange(len(index))
            dummy = 50 + 30 * np.sin(2 * np.pi * (hours - 8) / 24) + np.random.normal(0, 5, len(index))
            prices = pd.Series(dummy, index=index)
            
        battery_params = {
            "dt_hours": 1.0,
            "e_max_mwh": st.session_state["e_max_mwh"],
            "p_max_mw": st.session_state["p_max_mw"],
            "roundtrip_eff": st.session_state["roundtrip_eff"],
            "deg_cost_per_mwh": st.session_state["deg_cost_per_mwh"]
        }
        
        ml_model = None
        if forecast_fn == forecast_ml:
            ml_model = load_ml_model(prices)
            
            # The backtest_engine's run_backtest expects forecast_fn(prices, date_str).
            # But forecast_ml expects forecast_ml(model, prices, date_str).
            # We wrap it in a lambda to hide the model parameter from the engine.
            engine_forecast_fn = lambda p, d: forecast_ml(ml_model, p, d)
        else:
            engine_forecast_fn = forecast_fn
        
        # Run Perfect Foresight
        progress_text.text("Running Perfect Foresight benchmark...")
        df_pf = run_backtest(prices, None, battery_params, progress_callback=update_progress)
        
        # Run Forecast
        progress_text.text(f"Running {model_selection} forecast...")
        progress_bar.progress(0)
        df_fc = run_backtest(prices, engine_forecast_fn, battery_params, progress_callback=update_progress)
        
        progress_bar.empty()
        progress_text.empty()
        
        if df_pf.empty or df_fc.empty:
            st.error("Backtest failed to produce valid results. Check date ranges.")
            return

        # 3. Calculate metrics
        summary_pf = daily_summary(df_pf)
        summary_fc = daily_summary(df_fc)
        
        fc_revenue = summary_fc.get("total_revenue", 0)
        pf_revenue = summary_pf.get("total_revenue", 0)
        fc_annualised = summary_fc.get("annualised_revenue", 0)
        
        gap = efficiency_gap(fc_revenue, pf_revenue)
        payback_yrs = payback_period(capital_cost, fc_annualised)
        
        # 4. KPI Cards
        st.subheader("Backtest Results")
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        
        with kpi1:
            st.metric("Total Revenue", f"{currency_symbol}{fc_revenue:,.2f}")
        with kpi2:
            st.metric("Annualised Revenue", f"{currency_symbol}{fc_annualised:,.2f}")
        with kpi3:
            gap_str = f"{gap * 100:.1f}%" if gap is not None else "N/A"
            st.metric("Efficiency Gap", gap_str)
        with kpi4:
            pb_str = f"{payback_yrs:.1f} years" if payback_yrs is not None else "Never"
            st.metric("Payback Period", pb_str)
            
        # 5. Visuals
        st.markdown("---")
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            st.markdown(f"**Cumulative Revenue ({currency_symbol})**")
            fig1 = go.Figure()
            # Cumulative sums
            df_pf['cum_profit'] = df_pf['profit'].cumsum()
            df_fc['cum_profit'] = df_fc['profit'].cumsum()
            
            fig1.add_trace(go.Scatter(x=df_pf.index, y=df_pf['cum_profit'], name="Perfect Foresight", line=dict(color='green')))
            fig1.add_trace(go.Scatter(x=df_fc.index, y=df_fc['cum_profit'], name="Forecast", line=dict(color='blue')))
            fig1.update_layout(height=400, legend=dict(orientation="h", y=-0.2))
            st.plotly_chart(fig1, use_container_width=True)
            
        with chart_col2:
            st.markdown(f"**Monthly P&L ({currency_symbol})**")
            # Resample to monthly sums
            df_fc = df_fc.copy()
            df_fc.index = pd.to_datetime(df_fc.index)
            df_monthly = df_fc['profit'].resample('ME').sum()
            fig2 = go.Figure(data=[go.Bar(x=df_monthly.index.strftime('%Y-%B'), y=df_monthly.values)])
            fig2.update_layout(height=400)
            fig2.update_xaxes(tickangle=45)
            st.plotly_chart(fig2, use_container_width=True)
            
        # 6. Best and Worst Days
        st.markdown("---")
        st.markdown("**Best vs Worst Forecast Days**")
        bw1, bw2 = st.columns(2)
        
        sorted_days = df_fc.sort_values('profit', ascending=False)
        # Format the index dataframe to correctly render only the date portion (trimming off the zeroed timestamp)
        sorted_days = sorted_days.reset_index()
        sorted_days['date'] = pd.to_datetime(sorted_days['date']).dt.date
        bottom_5 = sorted_days[sorted_days['profit'] > 0].tail(5)
        
        with bw1:
            st.write("Top 5 Profitable Days")
            st.dataframe(sorted_days.head(5)[["date", "profit", "throughput_MWh", "n_cycles", "avg_spread"]], use_container_width=True)
        with bw2:
            st.write("Bottom 5 Days")
            st.dataframe(bottom_5[["date", "profit", "throughput_MWh", "n_cycles", "avg_spread"]].sort_values('profit'), use_container_width=True)
            
        # 7. CSV Export
        st.markdown("---")
        csv_data = df_fc.to_csv().encode('utf-8')
        st.download_button(
            label="Download Forecast Backtest Results (CSV)",
            data=csv_data,
            file_name="backtest_results.csv",
            mime="text/csv",
        )

if __name__ == "__main__":
    main()
