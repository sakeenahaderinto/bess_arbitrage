
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from bess.data.em_client import get_zones
from bess.data.cache import get_prices_cached
from bess.backtest.engine import run_backtest
from bess.metrics import daily_summary

st.set_page_config(page_title="Region Comparison", page_icon="🌍", layout="wide")

@st.cache_data
def load_zones():
    # If the API blocks, just provide a fallback list
    try:
        return get_zones()
    except Exception:
        return ["GB", "DE", "FR", "NL", "BE", "ES"]

def load_historical_prices(zone, start_date, end_date):
    return get_prices_cached(zone, start_date, end_date)

def main():
    st.title("🌍 Region Comparison")
    st.markdown("Compare arbitrage potential across multiple electricity markets.")

    # 1. Check dependencies
    required_keys = ["e_max_mwh", "p_max_mw", "roundtrip_eff", "deg_cost_per_mwh"]
    for key in required_keys:
        if key not in st.session_state:
            st.warning("Sidebar parameters missing. Please visit the main page first.")
            return

    # 2. UI Setup
    all_zones = load_zones()
    
    with st.form("region_form"):
        col1, col2 = st.columns(2)
        with col1:
            selected_zones = st.multiselect(
                "Select Zones to Compare (2-4 recommended)",
                options=all_zones,
                default=["GB", "DE"] if "GB" in all_zones and "DE" in all_zones else all_zones[:2]
            )
        with col2:
            date_range = st.date_input(
                "Date Range",
                value=(pd.to_datetime("2024-05-01"), pd.to_datetime("2024-05-31"))
            )
            
        submitted = st.form_submit_button("Run Comparison")

    if submitted:
        if len(selected_zones) < 2:
            st.error("Please select at least 2 zones.")
            return
            
        if not isinstance(date_range, tuple) or len(date_range) != 2:
            st.error("Please select both a start and end date.")
            return

        start_date, end_date = date_range
        start_date_str = start_date.strftime("%Y-%m-%d")
        end_date_str = end_date.strftime("%Y-%m-%d")

        battery_params = {
            "dt_hours": 1.0,
            "e_max_mwh": st.session_state["e_max_mwh"],
            "p_max_mw": st.session_state["p_max_mw"],
            "roundtrip_eff": st.session_state["roundtrip_eff"],
            "deg_cost_per_mwh": st.session_state["deg_cost_per_mwh"]
        }

        # Progress elements
        st.markdown("---")
        progress_text = st.empty()
        progress_bar = st.progress(0)

        results_data = []

        # Iterate over zones and run backtests in Perfect Foresight
        for idx, z in enumerate(selected_zones):
            progress_text.text(f"Fetching data and running optimiser for {z} ({idx+1}/{len(selected_zones)})...")
            progress_bar.progress((idx) / len(selected_zones))
            
            prices = load_historical_prices(z, start_date_str, end_date_str)
            
            if prices.empty or sum(prices) == 0:
                st.error(f"Could not fetch prices for zone {z} — skipping.")
                continue
                
            # Run the engine
            df_bt = run_backtest(prices, None, battery_params)
            
            if not df_bt.empty:
                # calculate std dev of daily price spread
                # standard daily spread: max(price) - min(price) on each day
                # Group by date strings to get daily spread
                daily_max = prices.groupby(prices.index.date).max()
                daily_min = prices.groupby(prices.index.date).min()
                daily_spreads = daily_max - daily_min
                price_spread_std = daily_spreads.std()

                summary = daily_summary(df_bt)
                
                results_data.append({
                    "Zone": z,
                    "Total Revenue": summary.get("total_revenue", 0),
                    "Annualised Revenue": summary.get("annualised_revenue", 0),
                    "Price Volatility (Spread Std Dev)": price_spread_std,
                    "Daily Mean": summary.get("daily_mean", 0),
                    "Profitable Days (%)": summary.get("pct_profitable", 0)
                })

        # Clear progress elements completely
        progress_bar.empty()
        progress_text.empty()

        if not results_data:
            st.error("Failed to generate comparative data.")
            return

        df_results = pd.DataFrame(results_data)
        df_results.sort_values(by="Annualised Revenue", ascending=False, inplace=True)

        st.subheader("Comparison Results")
        
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            st.markdown("**Annualised Revenue per Zone**")
            fig1 = px.bar(
                df_results, 
                x="Zone", 
                y="Annualised Revenue",
                color="Zone",
                text_auto='.2s'
            )
            fig1.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig1, use_container_width=True)
            
        with col_chart2:
            st.markdown("**Revenue vs Price Volatility**")
            fig2 = px.scatter(
                df_results,
                x="Price Volatility (Spread Std Dev)",
                y="Annualised Revenue",
                text="Zone",
                color="Zone",
                labels={
                    "Price Volatility (Spread Std Dev)": "Daily Price Spread Std Dev (local currency/MWh)",
                    "Annualised Revenue": "Annualised Revenue (£)"
                }
            )
            fig2.update_traces(textposition='top center', marker=dict(size=12))
            fig2.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown("**Metrics Summary Table**")
        st.dataframe(
            df_results.style.format({
                "Total Revenue": "{:,.2f}",
                "Annualised Revenue": "{:,.2f}",
                "Price Volatility (Spread Std Dev)": "{:.2f}",
                "Daily Mean": "{:.2f}",
                "Profitable Days (%)": "{:.1f}%"
            }),
            use_container_width=True,
            hide_index=True
        )

if __name__ == "__main__":
    main()
