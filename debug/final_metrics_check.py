
import os
import pandas as pd
from src.metrics import MetricsEngine
from src.data import DataManager

def final_metrics_check():
    manager = DataManager()
    engine = MetricsEngine()
    
    # Load all cached data
    inventory = manager.get_inventory()
    prices_data = {}
    
    # We only check 1h for simplicity here
    for symbol, intervals in inventory.items():
        if '1h' in intervals:
            df = manager.load_data(symbol, '1h')
            if df is not None:
                prices_data[symbol] = df
                
    print(f"Loaded {len(prices_data)} symbols for metric calculation.")
    
    results_df = engine.compute_all_metrics(prices_data, benchmark_symbol='BTCUSDT')
    
    # Check for zeros in volatility or adf_stat
    problematic = results_df[(results_df['volatility'] == 0) | (results_df['adf_stat'] == 0)]
    
    print("\nMetrics Calculation Summary:")
    print(f"Total symbols processed: {len(results_df)}")
    print(f"Symbols with 0 metrics (Flat Data): {len(problematic)}")
    
    if not problematic.empty:
        print("\nFlat Symbols (0 metrics):")
        print(problematic['symbol'].tolist())
        
    print("\nSample healthy metrics (first 5):")
    healthy = results_df[~results_df['symbol'].isin(problematic['symbol'])]
    print(healthy[['symbol', 'volatility', 'adf_stat', 'sharpe']].head())
    
    # Verify that healthy ones have non-zero metrics
    assert (healthy['volatility'] > 0).all(), "Healthy symbols should have non-zero volatility"

if __name__ == "__main__":
    final_metrics_check()
