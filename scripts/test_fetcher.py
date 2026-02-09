
import sys
import os
from datetime import datetime, timedelta
import pandas as pd

# Add parent directory to path to import src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data import BinanceFuturesFetcher

def test_fetcher():
    fetcher = BinanceFuturesFetcher()
    symbol = "BTCUSDT"
    interval = "1h"
    
    # Test 1: Fetch using years
    print(f"\n--- Test 1: Fetching 0.1 years of {symbol} {interval} ---")
    df1 = fetcher.fetch_history(symbol, interval, years=0.1)
    if not df1.empty:
        print(f"Fetched {len(df1)} candles. Start: {df1['open_time'].min()}, End: {df1['open_time'].max()}")
        
    # Test 2: Fetch using start_time string
    start_str = "2024-01-01"
    end_str = "2024-01-05"
    print(f"\n--- Test 2: Fetching {symbol} {interval} from {start_str} to {end_str} ---")
    df2 = fetcher.fetch_history(symbol, interval, start_time=start_str, end_time=end_str)
    if not df2.empty:
        print(f"Fetched {len(df2)} candles. Start: {df2['open_time'].min()}, End: {df2['open_time'].max()}")
        
    # Test 3: Fetch using fetch_candles (large count > 1500)
    limit_val = 2000
    print(f"\n--- Test 3: Fetching {limit_val} candles {symbol} {interval} ---")
    df3 = fetcher.fetch_candles(symbol, interval, limit=limit_val)
    if not df3.empty:
        print(f"Fetched {len(df3)} candles. Start: {df3['open_time'].min()}, End: {df3['open_time'].max()}")
        if len(df3) == limit_val:
            print(f"Success: Fetched exactly {limit_val} candles.")
        else:
            print(f"Warning: Expected {limit_val} candles, got {len(df3)}.")
    
    # Check for gaps in Test 3
    df3['diff'] = df3['open_time'].diff()
    expected_diff = pd.Timedelta(interval)
    gaps3 = df3[df3['diff'] > expected_diff]
    if len(gaps3) > 1:
        print(f"Warning: Found {len(gaps3)-1} gaps in large fetch!")
    else:
        print("Success: No gaps found in large fetch.")

if __name__ == "__main__":
    test_fetcher()
