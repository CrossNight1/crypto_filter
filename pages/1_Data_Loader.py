"""
Data Loader Page
"""
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path to import src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data import BinanceFuturesFetcher, DataManager
from src.config import AVAILABLE_INTERVALS, DEFAULT_FETCH_INTERVALS, MANDATORY_CRYPTO

st.markdown("## Data Loader")

# Initialize managers
# Initialize managers
if 'fetcher' not in st.session_state or not hasattr(st.session_state.fetcher, 'get_all_symbols'):
    st.session_state.fetcher = BinanceFuturesFetcher()

fetcher = st.session_state.fetcher
manager = DataManager()

# Sidebar / Settings
with st.container():
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Configuration")
        
        top_n = st.number_input("Top Symbols by Volume", min_value=0, max_value=200, value=50, step=10)
        
        # Specific Ticker Selection (Formatted as comma-separated string)
        specific_tickers_input = st.text_area("Add Specific Tickers (Comma separated)", placeholder="BTCUSDT, ETHUSDT, SOLUSDT")
        
        intervals = st.multiselect(
            "Select Timeframes",
            options=AVAILABLE_INTERVALS,
            default=DEFAULT_FETCH_INTERVALS
        )
        
        # Data Fetch Mode
        fetch_mode = st.radio("Fetch Mode", ["Date Range", "Candle Count (Limit)"])
        
        limit_val = None
        start_date = None
        end_date = None
        
        if fetch_mode == "Date Range":
            days_back = st.slider("Days of History (Max 365)", min_value=1, max_value=365, value=14)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
        else:
            limit_val = st.number_input("Number of Candles (Limit)", min_value=100, max_value=10_000, value=1000, step=100)
        
        fetch_btn = st.button("Fetch Data", type="primary", use_container_width=True)

    with col2:
        st.subheader("Cache Status")
        # cached_files = manager.get_existing_symbols() # Legacy
        
        # We'll fetch detailed inventory when the page loads? 
        # Might be slow if many files. Let's do it inside the expander check?
        # Streamlit execution model runs top to bottom.
        
        # Let's just show count first
        inventory = manager.get_inventory()
        total_files = sum(len(v) for v in inventory.values())
        
        if total_files > 0:
            st.success(f"Found {total_files} cached files.")
            with st.expander("View Cached Files (Details)"):
                with st.spinner("Loading cache details..."):
                    meta = manager.get_cache_metadata()
                    if meta:
                        st.dataframe(pd.DataFrame(meta))
                    else:
                        st.write("No valid metadata found.")
        else:
            st.warning("No data found in cache.")

# Execution
if fetch_btn:
    if not intervals:
        st.error("Please select at least one timeframe.")
    else:
        status_container = st.container()
        progress_bar = st.progress(0)
        
        with status_container:
            st.info("Preparing Symbol List...")
            
            # 1. Fetch Top Crypto
            crypto_symbols = fetcher.get_top_volume_symbols(top_n=top_n)
            
            # 2. Add Specific Tickers
            if specific_tickers_input:
                # Parse comma-separated string
                specific_tickers = [s.strip().upper() for s in specific_tickers_input.split(',') if s.strip()]
                crypto_symbols.extend(specific_tickers)
            
            # 3. Mandatory Symbols
            mandatory_crypto = MANDATORY_CRYPTO
            
            # Merge Crypto
            final_crypto = list(set(crypto_symbols + mandatory_crypto))
            
            # Combine All
            all_symbols = final_crypto
            
            st.write(f"Targets: {len(final_crypto)} Crypto Symbols.")
            
            total_tasks = len(all_symbols) * len(intervals)
            completed = 0
            failed_symbols = []
            
            for symbol in all_symbols:
                for interval in intervals:
                    try:
                        # Fetch based on mode
                        if fetch_mode == "Date Range":
                            df = fetcher.fetch_history(symbol, interval, start_date, end_date)
                        else:
                            # Fetch by Limit
                            df = fetcher.fetch_klines(symbol, interval, limit=limit_val)
                            
                        if not df.empty:
                            manager.save_data(df, symbol, interval)
                        else:
                            if symbol not in failed_symbols:
                                failed_symbols.append(symbol)
                    except Exception as e:
                        if symbol not in failed_symbols:
                            failed_symbols.append(symbol)
                        print(f"Error fetching {symbol}: {e}")
                        
                    completed += 1
                    progress = completed / total_tasks
                    progress_bar.progress(progress)
            
            if failed_symbols:
                st.warning(f"Skipped {len(failed_symbols)} symbols: {', '.join(failed_symbols)}")
                    
            st.success("Analysis Data Ready!")
            st.rerun()
