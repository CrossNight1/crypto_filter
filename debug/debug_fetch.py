
from src.data import BinanceFuturesFetcher
import pandas as pd

def debug_fetch():
    fetcher = BinanceFuturesFetcher()
    symbols = ['BTCUSDT', 'LINAUSDT', 'SOLUSDT']
    
    for sym in symbols:
        print(f"\n--- Fetching {sym} ---")
        df = fetcher.fetch_klines(sym, '1h', limit=10)
        if df.empty:
            print("Error: Empty DataFrame")
            continue
            
        print(df[['open_time', 'close']].head())
        std = df['close'].std()
        print(f"Standard Deviation: {std}")
        if std == 0:
            print("WARNING: Flat price data detected!")
        else:
            print("OK: Price data has variance.")

if __name__ == "__main__":
    debug_fetch()
