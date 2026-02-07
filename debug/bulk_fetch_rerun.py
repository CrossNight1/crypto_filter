
import os
from datetime import datetime, timedelta
from src.data import BinanceFuturesFetcher, DataManager, YFinanceFetcher

def bulk_fetch():
    fetcher = BinanceFuturesFetcher()
    yf_fetcher = YFinanceFetcher()
    manager = DataManager()
    
    # Configuration
    top_n = 50
    intervals = ['1h']
    
    print(f"Fetching top {top_n} symbols...")
    crypto_symbols = fetcher.get_top_volume_symbols(top_n=top_n)
    mandatory_crypto = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT']
    mandatory_tradfi = ['SPY', 'GC=F']
    
    all_symbols = list(set(crypto_symbols + mandatory_crypto)) + mandatory_tradfi
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    
    for symbol in all_symbols:
        is_tradfi = symbol in mandatory_tradfi
        current_fetcher = yf_fetcher if is_tradfi else fetcher
        
        for interval in intervals:
            try:
                print(f"Fetching {symbol} ({interval})...", end=' ')
                df = current_fetcher.fetch_history(symbol, interval, start_date, end_date)
                if not df.empty:
                    save_sym = 'GOLD' if symbol == 'GC=F' else symbol
                    manager.save_data(df, save_sym, interval)
                    print(f"OK ({len(df)} rows)")
                else:
                    print("SKIPPED (Empty)")
            except Exception as e:
                print(f"ERROR: {e}")

if __name__ == "__main__":
    bulk_fetch()
