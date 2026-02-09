
import pandas as pd
import numpy as np

class CorrelationEngine:
    """
    Modular engine for calculating correlation matrices between assets.
    """
    
    @staticmethod
    def calculate_matrix(data_map, method="pearson", min_periods=50):
        """
        Calculates correlation matrix from a dictionary of price series.
        
        Args:
            data_map (dict): {symbol: series}
            method (str): "pearson", "kendall", "spearman"
            min_periods (int): Min overlapping points.
            
        Returns:
            pd.DataFrame: Correlation matrix.
        """
        # Align data into a single wide DataFrame
        wide_df = pd.DataFrame(data_map)
        
        # Calculate correlation
        corr_matrix = wide_df.corr(method=method, min_periods=min_periods)
        
        return corr_matrix

    @staticmethod
    def filter_blanks(corr_matrix):
        """
        Iteratively drops symbols with the most NaNs in the correlation matrix.
        
        Returns:
            (filtered_matrix, dropped_symbols)
        """
        dropped_symbols = []
        matrix = corr_matrix.copy()
        
        while matrix.isna().any().any():
            nan_counts = matrix.isna().sum()
            if nan_counts.max() == 0:
                break
            
            worst_symbol = nan_counts.idxmax()
            matrix = matrix.drop(index=worst_symbol, columns=worst_symbol)
            dropped_symbols.append(worst_symbol)
            
            if len(matrix) < 2:
                break
                
        return matrix, dropped_symbols
