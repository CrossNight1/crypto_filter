
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from src.metrics import MetricsEngine

def verify_fip():
    engine = MetricsEngine()
    
    # Example 1: Total Return > 0 (Up trend)
    # Returns: [0.1, 0.1, -0.05, -0.05, -0.05, -0.05]
    # Total = 0.05 > 0 -> sign = 1
    # 2 positive, 4 negative, total 6
    # pos_pct = 2/6 = 0.333
    # neg_pct = 4/6 = 0.666
    # FIP = 1 * (0.666 - 0.333) = 0.333
    returns = np.array([0.1, 0.1, -0.05, -0.05, -0.05, -0.05])
    fip = engine.calculate_fip(returns)
    print(f"Example 1 (Up trend, more neg days): {fip:.4f}")
    expected = 1 * (4/6 - 2/6)
    assert abs(fip - expected) < 1e-6
    
    # Example 2: Total Return < 0 (Down trend)
    # Returns: [-0.1, -0.1, 0.05, 0.05, 0.05, 0.05]
    # Total = -0.05 < 0 -> sign = -1
    # 4 positive, 2 negative, total 6
    # pos_pct = 4/6 = 0.666
    # neg_pct = 2/6 = 0.333
    # FIP = -1 * (0.333 - 0.666) = 0.333
    returns2 = np.array([-0.1, -0.1, 0.05, 0.05, 0.05, 0.05])
    fip2 = engine.calculate_fip(returns2)
    print(f"Example 2 (Down trend, more pos days): {fip2:.4f}")
    expected2 = -1 * (2/6 - 4/6)
    assert abs(fip2 - expected2) < 1e-6

    print("\nFIP Logic Verification Passed!")

if __name__ == "__main__":
    verify_fip()
