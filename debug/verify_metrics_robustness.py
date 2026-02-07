
import numpy as np
import pandas as pd
import sys
import os

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from src.metrics import MetricsEngine

def test_robustness():
    engine = MetricsEngine()
    
    # 1. Short series with NaN
    prices = np.array([100, 101, np.nan, 102, 103, 104, 105, 106, 107, 108], dtype=float)
    
    print("Testing calculate_log_returns...")
    rets = engine.calculate_log_returns(prices)
    print(f"Returns: {rets}")
    assert not np.isnan(rets).any(), "Log returns should not contain NaNs"

    print("\nTesting calculate_custom_adf (short series)...")
    # Our new limit is lookback + 5. Default lookback is 20. 
    # Let's test with 25 points.
    long_prices = np.random.normal(100, 1, 30)
    long_prices[5] = np.nan
    hist, tau, smooth = engine.calculate_custom_adf(long_prices, lookback=10)
    print(f"ADF Results (len 30, lb 10): Hist={hist}, Tau={tau}")
    assert not np.isnan(hist), "ADF Hist should be a value"

    print("\nTesting rolling metrics with NaNs...")
    rolling_prices = pd.Series([100, 101, np.nan, 102, 103, 104, 105]).values
    res = engine.calculate_rolling_metric(rolling_prices, 'volatility', window=3)
    print(f"Rolling Volatility (window 3):\n{res}")
    # Should have values after enough points
    assert res.dropna().iloc[0] > 0, "Should have calculated volatility"

    print("\nRobustness tests passed!")

if __name__ == "__main__":
    test_robustness()
