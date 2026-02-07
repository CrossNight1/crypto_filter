
from src.data import BinanceFuturesFetcher, DataManager
import pandas as pd
import numpy as np

def analyze_top_symbols():
    fetcher = BinanceFuturesFetcher()
    manager = DataManager()
    
    top_symbols = fetcher.get_top_volume_symbols(top_n=20)
    print(f"Analyzing {len(top_symbols)} symbols...")
    
    results = []
    for sym in top_symbols:
        df = fetcher.fetch_klines(sym, '1h', limit=500)
        if df.empty:
            continue
            
        std = df['close'].std()
        unique_count = len(df['close'].unique())
        
        results.append({
            'symbol': sym,
            'std': std,
            'unique_prices': unique_count,
            'is_flat': std == 0
        })
        
        # Save to cache so we can test metrics later
        manager.save_data(df, sym, '1h')
        
    res_df = pd.DataFrame(results)
    print("\nSummary Results:")
    print(res_df)
    print(f"\nFlat Symbols Count: {res_df['is_flat'].sum()} out of {len(res_df)}")

if __name__ == "__main__":
    analyze_top_symbols()
