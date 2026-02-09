
import sys
import os
import pandas as pd
import numpy as np

# Add parent directory to path to import ml_engine
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ml_engine.modeling.factory import ModelFactory

def test_xgb_labels():
    # Dummy data with direction labels [-1, 0, 1]
    X = np.random.rand(100, 5)
    y_dirs = np.random.choice([-1, 0, 1], 100)
    
    print("Testing XGBClassifierWrapper with labels [-1, 0, 1]...")
    clf = ModelFactory.create_model('GBM Classifier')
    clf.fit(X, y_dirs)
    
    preds = clf.predict(X)
    unique_preds = np.unique(preds)
    print(f"✅ Fit successful. Unique predictions: {unique_preds}")
    
    if set(unique_preds).issubset({-1, 0, 1}):
        print("✅ Predictions matched original label space.")
    else:
        print(f"❌ Error: Predictions {unique_preds} outside expected space.")

if __name__ == "__main__":
    test_xgb_labels()
