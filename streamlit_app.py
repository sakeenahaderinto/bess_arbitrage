
import streamlit as st
from bess.data.em_client import get_zones

# Page config must be the first Streamlit command
st.set_page_config(
    page_title="BESS Arbitrage Optimiser",
    page_icon="⚡",
    layout="wide"
)

# Cache the zone list so we don't fetch it repeatedly
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
        min_value=0.1, max_value=1000.0, value=st.session_state.get("e_max_mwh", 1.0), step=0.1,
        help="1.0 MWh is a small grid-scale unit. Commercial systems are typically 10-100+ MWh"
    )
    st.session_state["e_max_mwh"] = e_max_mwh
    
    # Power rating
    p_max_mw = st.sidebar.number_input(
        "Power rating (MW)", 
        min_value=0.1, max_value=1000.0, value=st.session_state.get("p_max_mw", 0.5), step=0.1,
        help="Typical C-rate for arbitrage is 0.5 (half the capacity in MW)"
    )
    st.session_state["p_max_mw"] = p_max_mw
    
    # Efficiency
    eff_pct = st.sidebar.slider(
        "Round-trip efficiency (%)", 
        min_value=50, max_value=100, value=int(st.session_state.get("roundtrip_eff", 0.9) * 100), step=1,
        help="Typical lithium-ion batteries are 85-95%"
    )
    st.session_state["roundtrip_eff"] = eff_pct / 100.0
    
    # Degradation Cost
    deg_cost = st.sidebar.number_input(
        "Degradation cost (£/MWh)", 
        min_value=0.0, value=st.session_state.get("deg_cost_per_mwh", 5.0), step=0.5,
        help="Typical range is £2-10/MWh depending on battery chemistry and cycle life assumptions"
    )
    st.session_state["deg_cost_per_mwh"] = deg_cost
    
    # MILP Toggle
    enforce_milp = st.sidebar.checkbox(
        "Enforce no simultaneous charge/discharge",
        value=st.session_state.get("enforce_new_milp", False)
    )
    st.session_state["enforce_new_milp"] = enforce_milp  # Or standard simple modeling
    
    st.sidebar.markdown("---")
    if st.sidebar.button("Reset to defaults"):
        st.session_state["e_max_mwh"] = 1.0
        st.session_state["p_max_mw"] = 0.5
        st.session_state["roundtrip_eff"] = 0.9
        st.session_state["deg_cost_per_mwh"] = 5.0
        st.session_state["enforce_new_milp"] = False
        st.rerun()

def main():
    st.title("⚡ BESS Arbitrage Optimiser")
    st.markdown("Welcome to the Battery Energy Storage System (BESS) Arbitrage modelling tool.")
    st.markdown("Use the sidebar to adjust your battery parameters, and navigate through the pages on the left to see live optimisations, backtests, and system efficiency analytics.")
    
    render_sidebar()

if __name__ == "__main__":
    main()
