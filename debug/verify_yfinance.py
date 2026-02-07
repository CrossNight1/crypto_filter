import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def test_fetch(symbol="SPY", interval="1h"):
    print(f"Testing fetch for {symbol} {interval}...")
    start_dt = datetime.now() - timedelta(days=30)
    end_dt = datetime.now()
    
    try:
        # Replicating src/data.py logic
        df = yf.download(symbol, start=start_dt, end=end_dt, interval=interval, progress=False, multi_level_index=False)
        print("Success!")
        print(df.head())
        print("Columns:", df.columns)
    except TypeError as e:
        print(f"TypeError caught: {e}")
        print("Retrying without multi_level_index...")
        try:
            df = yf.download(symbol, start=start_dt, end=end_dt, interval=interval, progress=False)
            print("Retry Success!")
            print(df.head())
            print("Columns:", df.columns)
        except Exception as e2:
             print(f"Retry Failed: {e2}")
    except Exception as e:
        print(f"General Error: {e}")

if __name__ == "__main__":
    test_fetch()
