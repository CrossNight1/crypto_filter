"""
Data Loader Page - Simplified & Compact
"""
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import sys
import os
import time

# Add parent directory to path to import src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data import BinanceFuturesFetcher, DataManager
from src.config import AVAILABLE_INTERVALS, DEFAULT_FETCH_INTERVALS, MANDATORY_CRYPTO

st.set_page_config(page_title="Data Loader", layout="wide")
st.markdown("### Data Loader")

# Initialize managers in session state
if 'fetcher' not in st.session_state:
    st.session_state.fetcher = BinanceFuturesFetcher()
if 'selected_symbols' not in st.session_state:
    st.session_state.selected_symbols = set(MANDATORY_CRYPTO)
if 'fetch_logs' not in st.session_state:
    st.session_state.fetch_logs = []

fetcher = st.session_state.fetcher
manager = DataManager()

# --- SIDEBAR: SYMBOL SELECTION (Compact) ---
with st.sidebar:
    st.subheader("Selection")
    
    # Quick Add Section
    with st.expander("Filter Tool", expanded=False):
        top_n = st.number_input("Top Liquidity N", min_value=0, max_value=200, value=50, step=10)
        if st.button("Filter"):
            with st.spinner(""):
                st.session_state.selected_symbols.update(fetcher.get_top_volume_symbols(top_n=top_n))
                st.rerun()

    # Manual Add Section
    manual_input = st.text_input("Add Ticker (Comma separated)", placeholder="BTCUSDT...")
    if st.button("Add", use_container_width=True):
        if manual_input:
            new_tickers = [s.strip().upper() for s in manual_input.split(',') if s.strip()]
            st.session_state.selected_symbols.update(new_tickers)
            st.rerun()
    
    st.markdown("---")
    # Current List Management
    sorted_symbols = sorted(list(st.session_state.selected_symbols))
    st.write(f"Active List: {len(sorted_symbols)}")
    
    updated_selection = st.multiselect(
        "Manage List",
        options=sorted_symbols,
        default=sorted_symbols,
        label_visibility="collapsed"
    )
    if len(updated_selection) != len(st.session_state.selected_symbols):
        st.session_state.selected_symbols = set(updated_selection)
        st.rerun()

    if st.button("Reset Selection", use_container_width=True):
        st.session_state.selected_symbols = set(MANDATORY_CRYPTO)
        st.rerun()

# --- MAIN PAGE: FETCH SETTINGS & LOGS ---
col1, col2 = st.columns([1, 2])

with col1:
    st.write("**Fetch Configuration**")
    intervals = st.multiselect(
        "Timeframes",
        options=AVAILABLE_INTERVALS,
        default=DEFAULT_FETCH_INTERVALS
    )
    
    fetch_mode = st.radio("Mode", ["Range", "Limit"], horizontal=True)
    params = {}
    if fetch_mode == "Range":
        days = st.number_input("Days Back", min_value=1, max_value=100_000, value=30)
        params['end'] = datetime.now()
        params['start'] = params['end'] - timedelta(days=days)
    else:
        params['limit'] = st.number_input("Candles", min_value=100, max_value=20_000, value=1000, step=100)
    
    fetch_btn = st.button("Execute Data Sync", type="primary", use_container_width=True)

    # Simple Inventory Preview
    with st.expander("Cache Info", expanded=False):
        inventory = manager.get_inventory()
        total_files = sum(len(v) for v in inventory.values())
        st.write(f"Files: {total_files}")
        if st.button("View Metadata"):
            st.session_state.show_meta = not st.session_state.get('show_meta', False)
        
        if st.session_state.get('show_meta', False):
            st.dataframe(pd.DataFrame(manager.get_cache_metadata()), hide_index=True)

with col2:
    st.write("**Activity Logs**")
    log_area = st.empty()
    
    if fetch_btn:
        if not intervals:
            st.error("Select interval")
        elif not st.session_state.selected_symbols:
            st.error("Select symbols")
        else:
            st.session_state.fetch_logs = []
            pbar = st.progress(0.0)
            all_syms = sorted(list(st.session_state.selected_symbols))
            total = len(all_syms) * len(intervals)
            count_done = 0
            
            for sym in all_syms:
                for inter in intervals:
                    st.session_state.fetch_logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] {sym} {inter}...")
                    log_area.code("\n".join(st.session_state.fetch_logs[-12:]))
                    
                    try:
                        if fetch_mode == "Range":
                            df = fetcher.fetch_history(sym, inter, start_time=params['start'], end_time=params['end'])
                        else:
                            df = fetcher.fetch_candles(sym, inter, limit=params['limit'])
                        
                        if not df.empty:
                            manager.save_data(df, sym, inter)
                            st.session_state.fetch_logs.append(f"  > Saved {len(df)} candles")
                        else:
                            st.session_state.fetch_logs.append(f"  ! Missing data")
                    except Exception as e:
                        st.session_state.fetch_logs.append(f"  ! Error: {str(e)}")
                    
                    count_done += 1
                    pbar.progress(count_done / total)
                    log_area.code("\n".join(st.session_state.fetch_logs[-12:]))
            
            st.success("Sync Complete")
