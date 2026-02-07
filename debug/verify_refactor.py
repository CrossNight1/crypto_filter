"""
Verification Script for Crypto Filter Refactor
"""
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add current directory to path
sys.path.append(os.getcwd())

from src.data import BinanceFuturesFetcher, DataManager
from src.metrics import MetricsEngine

def verify_data_engine():
    print(">>> Verifying Data Engine...")
    fetcher = BinanceFuturesFetcher()
    
    # 1. Top Symbols
    print("Fetching top 5 symbols...")
    top_syms = fetcher.get_top_volume_symbols(top_n=5)
    print(f"Top symbols: {top_syms}")
    assert len(top_syms) == 5, "Failed to fetch top 5 symbols"
    
    # 2. Fetch History
    symbol = "BTCUSDT"
    print(f"Fetching 1h history for {symbol}...")
    end = datetime.now()
    start = end - timedelta(days=1)
    df = fetcher.fetch_history(symbol, '1h', start, end)
    print(f"Fetched {len(df)} rows.")
    assert not df.empty, "DataFrame is empty"
    assert 'close' in df.columns, "Missing close column"
    
    # 3. Cache
    print("Testing DataManager...")
    manager = DataManager(data_dir="test_cache")
    manager.save_data(df, symbol, '1h')
    
    loaded_df = manager.load_data(symbol, '1h')
    assert loaded_df is not None, "Failed to load data"
    assert len(loaded_df) == len(df), "Data length mismatch"
    
    # Verify inventory
    inventory = manager.get_inventory()
    print(f"Inventory: {inventory}")
    assert symbol in inventory, "Symbol not in inventory"
    assert '1h' in inventory[symbol], "Interval not in inventory"
    
    print("Data Engine Verified ✅")
    return loaded_df

def verify_metrics_engine(df):
    print("\n>>> Verifying Metrics Engine...")
    engine = MetricsEngine()
    
    # Create dummy dictionary input
    prices_data = {'BTCUSDT': df}
    
    # Compute metrics
    metrics = engine.compute_all_metrics(prices_data, benchmark_symbol='BTCUSDT')
    print("Metrics result:")
    print(metrics)
    
    assert not metrics.empty, "Metrics result is empty"
    assert 'beta' in metrics.columns, "Missing beta column"
    assert metrics.iloc[0]['beta'] == 1.0, "BTC beta should be 1.0"
    
    # Test volatility
    vol = metrics.iloc[0]['volatility']
    print(f"Volatility: {vol}")
    assert vol >= 0, "Volatility cannot be negative"
    
    print("Metrics Engine Verified ✅")

if __name__ == "__main__":
    try:
        df = verify_data_engine()
        verify_metrics_engine(df)
        print("\nAll systems operational! 🚀")
    except Exception as e:
        print(f"\n❌ Verification Failed: {e}")
        import traceback
        traceback.print_exc()
