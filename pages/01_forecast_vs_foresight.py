
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import timedelta

from bess.data.cache import get_prices_cached
from bess.forecast.naive import forecast_lag24, forecast_rolling7
from bess.forecast.ml import train_model, forecast_ml
from bess.optimiser import battery_solve_arbitrage
from bess.metrics import efficiency_gap

st.set_page_config(page_title="Forecast vs Foresight", page_icon="⚖️", layout="wide")

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
        index=zones.index("DE") if "DE" in zones else 0
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
    """Trains the ML model once per run and caches it in memory."""
    with st.spinner("Training LightGBM model on historical data..."):
        return train_model(prices)

def run_single_day_scenario(prices, target_date, forecast_fn, battery_params, ml_model=None):
    """Runs a single day scenario either with perfect foresight (forecast_fn=None) or a forecast model."""
    try:
        actual_prices = prices.loc[target_date]
        if len(actual_prices) != 24:
            return None, None, 0.0

        if forecast_fn is None:
            solve_prices = actual_prices.copy()
        elif forecast_fn == forecast_ml:
            solve_prices = forecast_fn(ml_model, prices, target_date)
        else:
            solve_prices = forecast_fn(prices, target_date)
            
        model, solver_result, df_dispatch, profit = battery_solve_arbitrage(
            solve_prices.tolist(), **battery_params
        )
        
        
        df_dispatch.index = pd.date_range(start=pd.to_datetime(target_date, utc=True), periods=24, freq="h", tz="UTC")
        df_dispatch['actual_price'] = actual_prices.values
        df_dispatch['forecast_price'] = solve_prices.values
        
        # Recalculate actual profit by evaluating the forecast dispatch against real prices. Critical for evaluating forecast performance
        actual_revenue = (df_dispatch["p_discharge_MW"] * df_dispatch["actual_price"]).sum()
        actual_cost = (df_dispatch["p_charge_MW"] * df_dispatch["actual_price"]).sum()
        realised_profit = actual_revenue - actual_cost
        
        return df_dispatch, solve_prices, realised_profit
        
    except Exception as e:
        st.error(f"Error running scenario for {target_date}: {e}")
        return None, None, 0.0

def main():
    st.title("⚖️ Forecast vs Perfect Foresight")
    render_sidebar()
    
    # Check dependencies
    required_keys = ["zone", "e_max_mwh", "p_max_mw", "roundtrip_eff", "deg_cost_per_mwh"]
    for key in required_keys:
        if key not in st.session_state:
            st.warning("Sidebar parameters missing. Please visit the main page first.")
            return
            
    # Setup standard UI
    col1, col2 = st.columns([1, 2])
    with col1:
        target_date = st.date_input("Select Historical Date", value=pd.to_datetime("2024-06-08"))
        target_date_str = target_date.strftime("%Y-%m-%d")
        
    with col2:
        model_selection = st.radio(
            "Select Forecast Model",
            options=["Naive (lag-24)", "Naive (7-day avg)", "ML Model"],
            horizontal=True
        )
        
    if st.button("Run Comparison"):
        zone = st.session_state["zone"]
        
        forecast_fn_map = {
            "Naive (lag-24)": (forecast_lag24, 1),
            "Naive (7-day avg)": (forecast_rolling7, 7),
            "ML Model": (forecast_ml, 365) # Fetch a full year for ML features context
        }
        forecast_fn, lookback_days = forecast_fn_map[model_selection]
        
        # history for the forecast model. 7 days lookback means we need to fetch starting from target_date - 7 days.
        start_fetch = (target_date - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
        
        with st.spinner(f"Fetching data and solving for {target_date_str}..."):
            prices = load_historical_prices(zone, start_fetch, target_date_str)
            
            # check zero conditions where the API key might not permit pricing
            if prices.empty or sum(prices) == 0:
                st.warning("No price data found (or mock zeros). Using synthetic variance data to demonstrate Tab 2!")
                import numpy as np
                index = pd.date_range(start=start_fetch, periods=24 * (lookback_days + 1), freq="h", tz="UTC")
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
            
            # Scenario 1: Perfect Foresight
            df_pf, _, pf_profit = run_single_day_scenario(prices, target_date_str, None, battery_params)
            
            # Scenario 2: Forecast
            df_fc, _, fc_profit = run_single_day_scenario(prices, target_date_str, forecast_fn, battery_params, ml_model)

        if df_pf is None or df_fc is None:
            st.error("Failed to solve scenarios. Possibly not enough historical data for the requested date.")
            return
            
        # Metrics
        gap = efficiency_gap(fc_profit, pf_profit)
        
        st.markdown("---")
        
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        currency_symbol = "£" if zone.startswith("GB") else "€"
        
        with metric_col1:
            st.metric("Perfect Foresight Profit", f"{currency_symbol}{pf_profit:,.2f}")
        with metric_col2:
            st.metric("Forecast Realised Profit", f"{currency_symbol}{fc_profit:,.2f}")
        with metric_col3:
            if gap is not None:
                st.metric("Efficiency Gap", f"{gap * 100:.1f}%")
                if gap > 0.8:
                    st.success(f"**Your forecast captured {gap * 100:.1f}% of available value!**")
                else:
                    st.warning(f"Your forecast only captured {gap * 100:.1f}% of available value.")
            else:
                st.metric("Efficiency Gap", "N/A")
                
        st.caption("Efficiency Gap measures how much of the theoretically available arbitrage "
                   "value your forecast captured. A value of 85% means your forecast recovered "
                   "85p of every £1 that was available under perfect market knowledge.")
                
        # Charts
        st.markdown("### Dispatch Comparison")
        chart_col1, chart_col2 = st.columns(2)
        
        # Helper to plot nicely
        def plot_dispatch(df, title, forecast_overlay=False):
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Bar(x=df.index, y=df['p_discharge_MW'], name="Discharge", marker_color='green'), secondary_y=False)
            fig.add_trace(go.Bar(x=df.index, y=-df['p_charge_MW'], name="Charge", marker_color='blue'), secondary_y=False)
            
            fig.add_trace(go.Scatter(x=df.index, y=df['actual_price'], name="Actual Price", line=dict(color='red', width=2)), secondary_y=True)
            if forecast_overlay:
                fig.add_trace(go.Scatter(x=df.index, y=df['forecast_price'], name="Forecast Price", line=dict(color='orange', width=2, dash='dot')), secondary_y=True)
                
            fig.update_layout(title=title, barmode='relative', height=400, legend=dict(orientation="h", y=-0.2))
            fig.update_yaxes(title_text="Power (MW) — positive = discharge, negative = charge", secondary_y=False)
            fig.update_yaxes(title_text="Price (£/MWh)", secondary_y=True)
            return fig
            
        with chart_col1:
            st.plotly_chart(plot_dispatch(df_pf, "Perfect Foresight"), use_container_width=True)
            st.caption("Green bars = discharge (selling), Blue bars = charge (buying)")
            
        with chart_col2:
            st.plotly_chart(plot_dispatch(df_fc, f"Forecast: {model_selection}", forecast_overlay=True), use_container_width=True)
            st.caption("Green bars = discharge (selling), Blue bars = charge (buying)")
            
        # Table of decisions
        st.markdown("### Hourly Detailed Breakdown")
        
        def action_str(charge, discharge):
            if charge > 0: return f"Charge {charge:.2f} MW"
            if discharge > 0: return f"Discharge {discharge:.2f} MW"
            return "Idle"
            
        breakdown_data = []
        for i in range(24):
            hour_idx = df_pf.index[i]
            breakdown_data.append({
                "Hour": hour_idx.strftime("%H:00"),
                "Actual Price": round(df_pf.iloc[i]['actual_price'], 2),
                "Forecast Price": round(df_fc.iloc[i]['forecast_price'], 2),
                "Price Error": round(df_fc.iloc[i]['forecast_price'] - df_pf.iloc[i]['actual_price'], 2),
                "PF Action": action_str(df_pf.iloc[i]['p_charge_MW'], df_pf.iloc[i]['p_discharge_MW']),
                "Forecast Action": action_str(df_fc.iloc[i]['p_charge_MW'], df_fc.iloc[i]['p_discharge_MW'])
            })
            
        breakdown_df = pd.DataFrame(breakdown_data)
        
        # Highlight top 5 absolute price errors
        top_5_indices = breakdown_df["Price Error"].abs().nlargest(5).index
        def highlight_top_errors(row):
            return ['background-color: #fff3cd' if row.name in top_5_indices else ''] * len(row)
            
        styled_df = breakdown_df.style.apply(highlight_top_errors, axis=1)
        st.dataframe(styled_df, use_container_width=True)
        
        st.download_button(
            label="Download Hourly Breakdown as CSV",
            data=breakdown_df.to_csv(index=False),
            file_name=f"hourly_breakdown_{target_date_str}.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()
