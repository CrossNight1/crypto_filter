"""
Metrics Engine for Crypto Filter
Optimized calculation using Numpy.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from scipy import stats
from statsmodels.tsa.stattools import adfuller

class MetricsEngine:
    """Calculates metrics from price data using vectorized operations."""
    TICKER_24H = "/fapi/v1/ticker/24hr"
    
    @staticmethod
    def get_annual_scaling(interval: str) -> float:
        """
        Returns number of periods per year for a given Binance interval.
        Assuming crypto markets 24/7 (365 days).
        """
        scales = {
            '1m': 525600,
            '3m': 175200,
            '5m': 105120,
            '15m': 35040,
            '30m': 17520,
            '1h': 8760,
            '2h': 4380,
            '4h': 2190,
            '6h': 1460,
            '8h': 1095,
            '12h': 730,
            '1d': 365,
            '3d': 121.66,
            '1w': 52.14,
            '1M': 12
        }
        return scales.get(interval, 8760) # Default to 1h if unknown

    @staticmethod
    def calculate_log_returns(prices: np.ndarray) -> np.ndarray:
        """Calculate log returns: ln(P_t / P_{t-1})"""
        with np.errstate(divide='ignore', invalid='ignore'):
            returns = np.log(prices[1:] / prices[:-1])
        return np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)

    @staticmethod
    def calculate_price_zscore(prices: np.ndarray, window: int = 30) -> float:
        """
        Calculates the latest Z-score of price relative to its rolling mean/std.
        Z = (Price - Mean) / Std
        """
        if len(prices) < window: return 0.0
        
        subset = prices[-window:]
        mean = np.mean(subset)
        std = np.std(subset)
        
        if std < 1e-12: return 0.0
        return float((prices[-1] - mean) / std)

    @staticmethod
    def calculate_price_sma_diff(prices: np.ndarray, window: int = 20) -> float:
        """
        Calculates (Price - SMA) / Price.
        """
        if len(prices) < window: return 0.0
        
        sma = np.mean(prices[-window:])
        current_price = prices[-1]
        
        if abs(current_price) < 1e-12: return 0.0
        return float((current_price - sma) / current_price)

    # --- TECHNICAL INDICATOR HELPERS ---
    @staticmethod
    def ema(series: pd.Series, span: int) -> pd.Series:
        return series.ewm(span=span, adjust=False).mean()

    @staticmethod
    def rsi(series: pd.Series, n: int = 14) -> pd.Series:
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=n).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=n).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=n).mean()

    @staticmethod
    def bollinger_bands(series: pd.Series, n: int = 20, k: float = 2.0) -> tuple:
        ma = series.rolling(window=n).mean()
        std = series.rolling(window=n).std()
        upper = ma + (k * std)
        lower = ma - (k * std)
        return upper, lower, ma

    @staticmethod
    def aroon(high: pd.Series, low: pd.Series, n: int = 25) -> tuple:
        aroon_up = high.rolling(window=n+1).apply(lambda x: float(np.argmax(x)) / n * 100, raw=True)
        aroon_down = low.rolling(window=n+1).apply(lambda x: float(np.argmin(x)) / n * 100, raw=True)
        return aroon_up, aroon_down

    @staticmethod
    def vwap(df: pd.DataFrame) -> pd.Series:
        tp = (df['high'] + df['low'] + df['close']) / 3
        return (tp * df['volume']).cumsum() / df['volume'].cumsum()

    @staticmethod
    def calculate_all_indicators(df: pd.DataFrame, benchmark_returns: pd.Series = None) -> pd.DataFrame:
        """
        Calculates all 11 advanced metrics requested by the user.
        Returns a DataFrame with the same index as the input df.
        """
        res = pd.DataFrame(index=df.index)
        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']
        
        # 1. EWVA: (EMA20 - EMA50) / STD(price)
        ema40 = MetricsEngine.ema(close, 40)
        ema100 = MetricsEngine.ema(close, 100)
        std_p = close.rolling(40).std()
        res['ewva'] = (ema40 - ema100) / std_p
        
        # 2. Aroon Oscillator: (AroonUp - AroonDown) / 100
        au, ad = MetricsEngine.aroon(high, low, 25)
        res['aroon_osc'] = (au - ad) / 100
        
        # 3. Bollinger Band Position (BBP): (Close - LowerBB) / (UpperBB - LowerBB)
        ubb, lbb, _ = MetricsEngine.bollinger_bands(close, 40, 2.0)
        bb_range = (ubb - lbb).replace(0, 1e-9)
        res['bbp'] = (close - lbb) / bb_range
        
        # 4. RSI (Normalized): RSI / 100
        res['rsi_norm'] = MetricsEngine.rsi(close, 40) / 100
        
        # 5. Return Z-Score: (Return - mean(Return)) / std(Return)
        ret = close.pct_change()
        res['return_z'] = (ret - ret.rolling(40).mean()) / ret.rolling(40).std()
        
        # 6. Normalized ATR: ATR / Close
        res['atr_norm'] = MetricsEngine.atr(high, low, close, 14) / close.replace(0, 1e-9)
        
        # 7. Chaikin Money Flow (CMF)
        # Formula: ((Close - Low) - (High - Close)) / (High - Low) * Volume -> then 20-period sum
        mfm = ((close - low) - (high - close)) / (high - low).replace(0, 1e-9)
        mfv = mfm * volume
        res['cmf'] = mfv.rolling(40).sum() / volume.rolling(40).sum()
        
        # 8. VWAP Deviation Z-Score: (Close - VWAP) / STD(price)
        vwp = MetricsEngine.vwap(df)
        res['vwap_z'] = (close - vwp) / std_p
        
        # 9. Relative Strength Z-Score (vs benchmark)
        if benchmark_returns is not None:
            # Align indices
            common_idx = ret.index.intersection(benchmark_returns.index)
            diff = ret.loc[common_idx] - benchmark_returns.loc[common_idx]
            res['rel_strength_z'] = (diff - diff.rolling(40).mean()) / diff.rolling(40).std()
        else:
            res['rel_strength_z'] = np.nan
            
        # 10. Volatility-Adjusted Momentum (VAM): ROC / (ATR / Close)
        roc = close.pct_change(14)
        atr_norm = res['atr_norm']
        res['vam'] = roc / atr_norm
        
        # 11. Return Skewness (Rolling)
        res['skewness'] = ret.rolling(40).skew()
        
        # 12. Lagged Returns
        res['return_lag1'] = ret.shift(1)
        res['return_lag2'] = ret.shift(2)
        res['return_lag3'] = ret.shift(3)
        
        # 13. Autocorrelation (5-window)
        # Using rolling apply for autocorrelation
        res['autocorr_5'] = ret.rolling(5).apply(lambda x: x.autocorr(lag=1) if len(x) == 5 else np.nan, raw=False)
        
        # 14. EWMA (Simple EMA of price normalized by price)
        res['ewma'] = MetricsEngine.ema(close, 20) / close
        
        return res

    @staticmethod
    def calculate_beta_alpha(asset_returns: np.ndarray, benchmark_returns: np.ndarray) -> tuple:
        """
        Calculate Alpha and Beta relative to benchmark.
        y = alpha + beta * x
        Using numpy linear algebra for speed.
        """
        if len(asset_returns) != len(benchmark_returns):
            # Align lengths
            min_len = min(len(asset_returns), len(benchmark_returns))
            asset_returns = asset_returns[-min_len:]
            benchmark_returns = benchmark_returns[-min_len:]
            
        # Standard OLS: beta = Cov(x,y) / Var(x)
        # alpha = mean(y) - beta * mean(x)
        
        # Stack x for linalg (add constant term for intercept/alpha)
        # A = [x, 1]
        x = benchmark_returns
        y = asset_returns
        
        # Manual calculation is often faster/simpler for 1D
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        
        covariance = np.mean((x - x_mean) * (y - y_mean))
        variance_x = np.mean((x - x_mean)**2)
        
        if variance_x == 0:
            return 0.0, 0.0, 0.0 # Beta, Alpha, R2
            
        beta = covariance / variance_x
        alpha = y_mean - beta * x_mean
        
        # R-squared
        # SS_res = sum((y - (alpha + beta*x))^2)
        # SS_tot = sum((y - mean(y))^2)
        y_pred = alpha + beta * x
        ss_res = np.sum((y - y_pred)**2)
        ss_tot = np.sum((y - y_mean)**2)
        
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
        
        return beta, alpha, r_squared

    @staticmethod
    def check_stationarity(prices: np.ndarray, max_lag: int = None) -> tuple:
        """
        ADF Test.
        Returns: (adf_stat, p_value, is_stationary)
        """
        # Use statsmodels adfuller, but ensure inputs are clean
        # Remove NaNs
        clean_prices = prices[~np.isnan(prices)]
        if len(clean_prices) < 20: # Not enough data
            return 0.0, 1.0, False
            
        try:
            # autolag='AIC' helps determine lags
            result = adfuller(clean_prices, maxlag=max_lag, autolag='AIC')
            stat = result[0]
            p_value = result[1]
            is_stationary = p_value < 0.05
            return stat, p_value, is_stationary
        except Exception:
            return 0.0, 1.0, False

    @staticmethod
    def calculate_volatility(log_returns: np.ndarray, interval: str = '1h') -> float:
        if len(log_returns) < 2: return 0.0
        ann_factor = MetricsEngine.get_annual_scaling(interval)
        return float(np.std(log_returns) * np.sqrt(ann_factor))

    @staticmethod
    def calculate_custom_adf(series: np.ndarray, lookback: int = 20) -> tuple:
        """
        Fast ADF implementation based on PineScript logic.
        Returns: (hist, tau_adf, tau_smooth)
        Effectively linear regression of diffs vs lagged values.
        """
    @staticmethod
    def calculate_custom_adf_series(series: np.ndarray, lookback: int = 20) -> tuple:
        """
        Fast ADF implementation returning full series for rolling usage.
        Returns: (hist_series, tau_sma_series, tau_smooth_series)
        """
        # Ensure prices is clean (replace inf/nan with 0 or last value?)
        # For ADF, dropping NaNs is usually better than filling with 0 if gaps are sparse.
        # But for constant spacing, ffill is better.
        series = pd.Series(series).ffill().fillna(0).values

        if len(series) < lookback + 5:
            nan_s = pd.Series(np.nan, index=range(len(series)))
            return nan_s, nan_s, nan_s

            
        # Optimization: We can vectorize the rolling window, but for simplicity/robustness match the logic first.
        # Logic: Run ADF for each window -> get series of tauADF -> SMA(10) -> EMA(50) -> Hist
        
        # However, computing rolling regression in pure python loop is slow. 
        # But lookback is small (defaults?), usually rolling window logic.
        # User logic seems to imply calculate ONE value for the series? 
        # "tauADF = adf_fast(src, lookback)" -> This usually implies a time series indicator in PineScript.
        # Meaning we generate a SERIES of tauADF values.
        
        # We need to implement vectorized rolling regression.
        # y = diff(src)
        # x = src_lagged
        # Model: dy = alpha + beta * x
        
        # Prepare arrays
        vals = series
        dy = np.diff(vals)
        x = vals[:-1]
        
        # Align lengths (dy has length N-1)
        n = len(dy)
        if n < lookback: 
            nan_s = pd.Series(np.nan, index=range(len(series)))
            return nan_s, nan_s, nan_s
        
        # Rolling window beta calculation
        # beta = (n*sum_xy - sum_x*sum_y) / (n*sum_xx - sum_x^2)
        # We can use stride_tricks or pandas rolling.
        # Let's use pandas for speed/convenience if possible, but we are in metrics.py using numpy.
        # Let's wrap in pandas Series for rolling apply or optimized rolling sum.
        
        dy_s = pd.Series(dy)
        x_s = pd.Series(x)
        
        # Rolling sums with min_periods=1 for robustness
        sum_x = x_s.rolling(window=lookback, min_periods=1).sum()
        sum_y = dy_s.rolling(window=lookback, min_periods=1).sum()
        sum_xx = (x_s**2).rolling(window=lookback, min_periods=1).sum()
        sum_xy = (x_s * dy_s).rolling(window=lookback, min_periods=1).sum()
        sum_yy = (dy_s**2).rolling(window=lookback, min_periods=1).sum()
        
        # Helper for vectorized beta, alpha
        denom = lookback * sum_xx - sum_x**2
        
        # Handle division by zero
        valid = np.abs(denom) > 1e-10
        beta = np.zeros_like(denom)
        alpha = np.zeros_like(denom)
        
        beta[valid] = (lookback * sum_xy[valid] - sum_x[valid] * sum_y[valid]) / denom[valid]
        alpha[valid] = (sum_y[valid] - beta[valid] * sum_x[valid]) / lookback
        
        # SE calculation requires RSS
        # RSS = sum( (dy - (alpha + beta*x))^2 )
        # Expand: sum(dy^2) + alpha^2*n + beta^2*sum_xx + 2*alpha*beta*sum_x - 2*alpha*sum_y - 2*beta*sum_xy
        
        rss = (sum_yy + 
               alpha**2 * lookback + 
               beta**2 * sum_xx + 
               2 * alpha * beta * sum_x - 
               2 * alpha * sum_y - 
               2 * beta * sum_xy)
               
        # Ensure positive RSS (floating point errors)
        rss = np.maximum(rss, 0)
        
        sigma2 = rss / max(lookback - 2, 1)
        
        # SE = sqrt(sigma2 * n / denom) = sqrt(sigma2 * lookback / denom)
        se = np.sqrt(sigma2 * lookback / np.maximum(denom, 1e-10))
        
        # t-stat (tau)
        tau = np.zeros_like(beta)
        mask = se > 1e-10
        tau[mask] = beta[mask] / se[mask]
        
        # Convert to series for smoothing
        tau_s = pd.Series(tau).fillna(0)
        
        # Verify: "tauADF := ta.sma(tauADF, 10)"
        tau_sma = tau_s.rolling(window=10, min_periods=1).mean()
        
        # "tauADF_smooth = ta.ema(tauADF, 50)"
        # EMA usually handles NaNs/start better than SMA
        tau_smooth = tau_sma.ewm(span=50, adjust=False, min_periods=1).mean()
        
        hist = tau_sma - tau_smooth
        
        # Pad beginning to match original length?
        # dy lost 1, but we want result aligned to series end.
        # tau_sma and hist are aligned to dy index.
        # Let's align back to series length by prepending NaNs or using original index if passed.
        # Since input is numpy, we return Series with default index.
        # We need to shift/pad to match original length T?
        # dy corresponds to t=1..T-1.
        # hist corresponds to t=1..T-1.
        
        # Add 1 NaN at start to match original length
        hist = pd.concat([pd.Series([np.nan]), hist], ignore_index=True)
        tau_sma = pd.concat([pd.Series([np.nan]), tau_sma], ignore_index=True)
        tau_smooth = pd.concat([pd.Series([np.nan]), tau_smooth], ignore_index=True)
        
        return hist, tau_sma, tau_smooth

        return hist, tau_sma, tau_smooth

    @staticmethod
    def calculate_custom_adf(series: np.ndarray, lookback: int = 20) -> tuple:
        """
        Wrapper for single value return.
        """
        hist, tau, smooth = MetricsEngine.calculate_custom_adf_series(series, lookback)
        # Return the *latest* value (last valid point)
        return hist.iloc[-1], tau.iloc[-1], smooth.iloc[-1]

    @staticmethod
    def calculate_fip(returns: np.ndarray) -> float:
        """
        Frog-in-the-Pan (FIP) momentum from Quantitative Momentum.
        FIP = sign(Past return) * [% negative - % positive]
        """
        if len(returns) < 2: return 0.0
        
        total_ret = np.sum(returns)
        sign_ret = np.sign(total_ret)
        
        n = len(returns)
        neg_pct = np.sum(returns < 0) / n
        pos_pct = np.sum(returns > 0) / n
        
        return sign_ret * (neg_pct - pos_pct)

    @staticmethod
    def calculate_sharpe_ratio(log_returns: np.ndarray, risk_free_rate: float = 0.0, interval: str = '1h') -> float:
        if len(log_returns) < 2: return 0.0
        
        mean_ret = np.mean(log_returns)
        std_ret = np.std(log_returns)
        
        if std_ret < 1e-9: return 0.0
        
        ann_factor = MetricsEngine.get_annual_scaling(interval)
        # Annualized Sharpe = (Mean * Factor) / (Std * Sqrt(Factor)) = (Mean / Std) * Sqrt(Factor)
        return float((mean_ret / std_ret) * np.sqrt(ann_factor))

    @staticmethod
    def calculate_rolling_metric(df: pd.DataFrame, metric_name: str, window: int = 30, step: int = 1, benchmark_returns: pd.Series = None, interval: str = '1h') -> pd.Series:
        """
        Calculates a metric over a rolling window.
        Returns a Series of values indexed by the original index.
        """
        if df.empty or len(df) < 2:
             return pd.Series()
             
        # Clean prices: ffill is safest for time series continuity
        df = df.copy()
        df['close'] = pd.to_numeric(df['close'], errors='coerce').ffill().fillna(method='bfill')
        df['high'] = pd.to_numeric(df['high'], errors='coerce').ffill().fillna(method='bfill')
        df['low'] = pd.to_numeric(df['low'], errors='coerce').ffill().fillna(method='bfill')
        df['volume'] = pd.to_numeric(df['volume'], errors='coerce').fillna(0)
        
        # New Metrics (Calculated via the all-in-one helper)
        new_metrics = [
            'ewva', 'aroon_osc', 'bbp', 'rsi_norm', 'return_z', 'atr_norm', 
            'cmf', 'vwap_z', 'rel_strength_z', 'vam', 'skewness',
            'return_lag1', 'return_lag2', 'return_lag3', 'autocorr_5', 'ewma'
        ]
        
        if metric_name in new_metrics:
            all_inds = MetricsEngine.calculate_all_indicators(df, benchmark_returns)
            res = all_inds[metric_name]
        
        # Legacy/Standard Metrics
        else:
            price_s = df['close']
            with np.errstate(divide='ignore', invalid='ignore'):
                ret_s = np.log(price_s / price_s.shift(1))
            ret_s = ret_s.replace([np.inf, -np.inf], np.nan).fillna(0)
            
            if metric_name == 'sharpe':
                roller = ret_s.rolling(window=window)
                ann_factor = MetricsEngine.get_annual_scaling(interval)
                res = (roller.mean() / roller.std()) * np.sqrt(ann_factor)
            elif metric_name == 'volatility':
                ann_factor = MetricsEngine.get_annual_scaling(interval)
                res = ret_s.rolling(window=window).std() * np.sqrt(ann_factor)
            elif metric_name == 'fip':
                def fip_func(x):
                    total_ret = np.sum(x)
                    sign_ret = np.sign(total_ret)
                    n = len(x)
                    neg_pct = np.sum(x < 0) / n
                    pos_pct = np.sum(x > 0) / n
                    return sign_ret * (neg_pct - pos_pct)
                res = ret_s.rolling(window=window).apply(fip_func, raw=True)
            elif metric_name == 'return':
                 res = ret_s.rolling(window=window, min_periods=1).sum()
            elif metric_name == 'adf_hist':
                hist, _, _ = MetricsEngine.calculate_custom_adf_series(df['close'].values, lookback=window)
                res = hist
            elif metric_name == 'adf_stat':
                _, tau, _ = MetricsEngine.calculate_custom_adf_series(df['close'].values, lookback=window)
                res = tau
            elif metric_name == 'price_zscore':
                close_s = pd.to_numeric(df['close'], errors='coerce').ffill()
                res = (close_s - close_s.rolling(window=window).mean()) / close_s.rolling(window=window).std()
            elif metric_name == 'price_sma_diff':
                close_s = pd.to_numeric(df['close'], errors='coerce').ffill()
                sma = close_s.rolling(window=window).mean()
                res = (close_s - sma) / close_s.replace(0, 1e-9)
            else:
                return pd.Series()
            
        # Apply Step (Stride)
        if step > 1:
            res = res.iloc[::step]
            
        return res

    def compute_all_metrics(self, prices_data: Dict[str, pd.DataFrame], interval: str = '1h', benchmark_symbol: str = 'BTCUSDT') -> pd.DataFrame:
        """
        Main pipeline to compute metrics for all symbols.
        """
        results = []
        
        # 1. Prepare Benchmark
        benchmark_returns = None
        if benchmark_symbol in prices_data:
            b_df = prices_data[benchmark_symbol]
            b_close = pd.to_numeric(b_df['close'], errors='coerce').ffill().fillna(0)
            benchmark_returns = b_close.pct_change().dropna()
            
        for symbol, df in prices_data.items():
            try:
                if df.empty: continue
                
                # Standardize and clean prices
                prices = pd.to_numeric(df['close'], errors='coerce').ffill().fillna(0).values.astype(float)
                if len(prices) < 2: continue
                
                # 1. Standard Metrics
                log_rets = self.calculate_log_returns(prices)
                vol = self.calculate_volatility(log_rets, interval=interval)
                sharpe = self.calculate_sharpe_ratio(log_rets, interval=interval)
                fip = self.calculate_fip(log_rets)
                total_ret = np.sum(log_rets)
                
                # 2. Beta/Alpha
                beta, alpha, r2 = 0.0, 0.0, 0.0
                if benchmark_returns is not None and symbol != benchmark_symbol:
                    sym_close = pd.to_numeric(df['close'], errors='coerce').ffill().fillna(0)
                    sym_rets = sym_close.pct_change().dropna()
                    
                    common_idx = sym_rets.index.intersection(benchmark_returns.index)
                    if len(common_idx) > 10:
                        y_vec = sym_rets.loc[common_idx].values
                        x_vec = benchmark_returns.loc[common_idx].values
                        beta, alpha, r2 = self.calculate_beta_alpha(y_vec, x_vec)
                elif symbol == benchmark_symbol:
                    beta, alpha, r2 = 1.0, 0.0, 1.0
                
                # 3. ADVANCED METRICS (The 11 new ones)
                # Calculate all and take the LATEST value for the snapshot
                adv_df = self.calculate_all_indicators(df, benchmark_returns)
                latest_adv = adv_df.iloc[-1].to_dict()
                
                # 4. Old Custom ADF (Keeping for compatibility)
                adf_hist, adf_stat, _ = self.calculate_custom_adf(prices, lookback=60)
                
                row = {
                    'symbol': symbol,
                    'volatility': vol,
                    'beta': beta,
                    'alpha': alpha,
                    'r_squared': r2,
                    'adf_hist': adf_hist,
                    'adf_stat': adf_stat,
                    'sharpe': sharpe,
                    'fip': fip,
                    'return': total_ret,
                    'metric_return': total_ret,
                    'price_zscore': self.calculate_price_zscore(prices, window=60),
                    'price_sma_diff': self.calculate_price_sma_diff(prices, window=200),
                    'count': len(prices)
                }
                # Add all new metrics to row
                row.update(latest_adv)
                results.append(row)
                
            except Exception as e:
                print(f"Error calculating metrics for {symbol}: {e}")
                
        return pd.DataFrame(results)
