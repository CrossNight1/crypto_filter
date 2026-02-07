"""
Correlation Matrix Page
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import sys
import os

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data import DataManager

st.set_page_config(page_title="Correlation Matrix", page_icon="🔗", layout="wide")

st.markdown("## 🔗 Correlation Matrix")

manager = DataManager()
inventory = manager.get_inventory()

if not inventory:
    st.warning("No data found. Please fetch data in 'Data Loader'.")
    st.stop()

# Settings
col1, col2, col3 = st.columns(3)

with col1:
    # 1. Select Timeframe
    # Gather all intervals
    all_intervals = set()
    for sym, intervals in inventory.items():
        all_intervals.update(intervals)
    
    selected_interval = st.selectbox("Timeframe", sorted(list(all_intervals)))

with col2:
    # 2. Select Method
    corr_method = st.selectbox("Correlation Method", ["pearson", "kendall", "spearman"])
    min_data_points = st.number_input("Min Overlapping Samples", min_value=10, max_value=2000, value=50)

with col3:
    # 3. Filter Symbols?
    # Get symbols for this interval
    available_symbols = [s for s, ints in inventory.items() if selected_interval in ints]
    # Default to top 20 to avoid overcrowding? Or all?
    # Let's select all but allow filtering
    selected_symbols = st.multiselect("Select Symbols", available_symbols, default=available_symbols[:50])

if not selected_symbols:
    st.info("Select symbols to generate matrix.")
    st.stop()

if st.button("Generate Matrix"):
    with st.spinner("Loading data and computing correlation..."):
        # Load and align data
        data_map = {}
        for sym in selected_symbols:
            df = manager.load_data(sym, selected_interval)
            if df is not None and not df.empty:
                # Calculate log returns
                # We need to align by time.
                df = df.set_index('open_time')['close']
                # Log returns for correlation usually better than prices
                # Handle potential infinite values or zeros
                with np.errstate(divide='ignore', invalid='ignore'):
                    ret_series = np.log(df / df.shift(1))
                ret_series = ret_series.replace([np.inf, -np.inf], np.nan).dropna()
                data_map[sym] = ret_series
        
        if not data_map:
            st.error("Could not load data for selected symbols.")
            st.stop()
            
        # Create wide dataframe (outer join by default)
        wide_df = pd.DataFrame(data_map)
        
        # User request: "just calculate on valid data"
        # We switch to Pairwise correlation. 
        # Pandas corr() automatically handles NaNs by using only valid overlapping rows for each pair.
        # min_periods ensures we don't return junk correlation from too few points.
        
        # Calculate Correlation
        corr_matrix = wide_df.corr(method=corr_method, min_periods=min_data_points)
        
        # Post-calculation cleanup: "dropit, do not show blank"
        # Iteratively drop symbols with the most NaNs in the correlation matrix
        dropped_corr_symbols = []
        
        while corr_matrix.isna().any().any():
            # Count NaNs per symbol (row)
            nan_counts = corr_matrix.isna().sum()
            
            # Find worst offender
            if nan_counts.max() == 0: break # Should be covered by while condition
            
            worst_symbol = nan_counts.idxmax()
            
            # Drop from matrix (both row and col)
            corr_matrix = corr_matrix.drop(index=worst_symbol, columns=worst_symbol)
            dropped_corr_symbols.append(worst_symbol)
            
            if len(corr_matrix) < 2:
                break
        
        if len(corr_matrix) < 2:
             st.error(f"After filtering blanks, not enough symbols remain to plot. Consider lowering 'Min Overlapping Samples'.")
             st.stop()
             
        if dropped_corr_symbols:
            st.warning(f"Hidden {len(dropped_corr_symbols)} symbols due to insufficient overlap (blanks): {', '.join(dropped_corr_symbols)}")
             
        # Plot
        fig = px.imshow(
            corr_matrix,
            text_auto=".2f",
            aspect="auto",
            color_continuous_scale="RdBu_r", # Red-Blue diverging (Red=Neg, Blue=Pos) or similar
            zmin=-1,
            zmax=1,
            title=f"{corr_method.capitalize()} Correlation Matrix ({selected_interval})"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        with st.expander("View Correlation Data"):
            st.dataframe(corr_matrix)
