
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

class FeatureSelector:
    @staticmethod
    def apply_vif_filter(X_input, threshold=10.0):
        """Iteratively remove features with VIF > threshold"""
        if X_input.shape[1] <= 1:
            return X_input.columns.tolist()
            
        cols = X_input.columns.tolist()
        while True:
            if len(cols) <= 1:
                break
                
            # Add constant for VIF calculation
            X_vif = sm.add_constant(X_input[cols])
            vifs = []
            for i in range(X_vif.shape[1]):
                col_name = X_vif.columns[i]
                if col_name == 'const':
                    vifs.append(0) # We don't drop the constant
                    continue
                try:
                    # variance_inflation_factor returns inf if perfect collinearity
                    v = variance_inflation_factor(X_vif.values, i)
                except Exception as e:
                    # If calculation fails (e.g. singular matrix), treat as extremely high VIF
                    v = 999.0
                vifs.append(v)
                
            vif_series = pd.Series(vifs, index=X_vif.columns)
            vif_no_const = vif_series.drop('const', errors='ignore')
            
            max_vif = vif_no_const.max()
            if max_vif > threshold or np.isnan(max_vif) or np.isinf(max_vif):
                exclude_col = vif_no_const.idxmax()
                cols.remove(exclude_col)
            else:
                break
        return cols
