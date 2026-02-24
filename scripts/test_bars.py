import pandas as pd
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from ml_engine.data.bars import construct_volume_bars, construct_dollar_bars

def test_bars():
    # Create sample data
    data = {
        'open': [100, 101, 102, 103, 104, 105],
        'high': [101, 102, 103, 104, 105, 106],
        'low': [99, 100, 101, 102, 103, 104],
        'close': [101, 102, 103, 104, 105, 106],
        'volume': [10, 10, 10, 10, 10, 10]
    }
    df = pd.DataFrame(data, index=pd.date_range('2023-01-01', periods=6, freq='H'))
    
    print("Original Data:")
    print(df)
    
    # Volume bars with threshold 25 (should yield 2 bars)
    vol_bars = construct_volume_bars(df, 25)
    print("\nVolume Bars (Threshold 25):")
    print(vol_bars)
    
    assert len(vol_bars) == 2
    assert vol_bars.iloc[0]['volume'] == 30
    assert vol_bars.iloc[1]['volume'] == 30
    
    # Dollar bars with threshold 2500 (101*10 + 102*10 + 103*10 = 3060)
    dollar_bars = construct_dollar_bars(df, 2500)
    print("\nDollar Bars (Threshold 2500):")
    print(dollar_bars)
    
    assert len(dollar_bars) == 2
    print("\nVerification Successful!")

if __name__ == "__main__":
    test_bars()
