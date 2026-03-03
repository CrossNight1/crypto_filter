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
    
    def __init__(self):
        self._results_cache = {} # Key: (symbol, interval, last_timestamp, length)
    
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
        gain = (delta.where(delta > 0, 0)).ewm(span=n, min_periods=n).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(span=n, min_periods=n).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.ewm(span=n, min_periods=n).mean()

    @staticmethod
    def bollinger_bands(series: pd.Series, n: int = 20, k: float = 2.0) -> tuple:
        ma = series.ewm(span=n).mean()
        std = series.ewm(span=n).std()
        upper = ma + (k * std)
        lower = ma - (k * std)
        return upper, lower, ma

    @staticmethod
    def aroon(high: pd.Series, low: pd.Series, n: int = 25) -> tuple:
        aroon_up = high.rolling(n+1).apply(lambda x: float(np.argmax(x)) / n * 100, raw=True)
        aroon_down = low.rolling(n+1).apply(lambda x: float(np.argmin(x)) / n * 100, raw=True)
        return aroon_up, aroon_down

    @staticmethod
    def vwap(df: pd.DataFrame) -> pd.Series:
        tp = (df['high'] + df['low'] + df['close']) / 3
        return (tp * df['volume']).cumsum() / df['volume'].cumsum()
    
    @staticmethod
    def vama(close: pd.Series, w_short: int = 40, w_slow: int = 120) -> pd.Series:
        rets = np.log(close / close.shift()).fillna(0)
        s_std = rets.ewm(span=w_short).std()
        l_std = rets.ewm(span=w_slow).std()
        alpha = ((l_std * 0.5) / s_std).clip(lower=0.05, upper=0.95).fillna(0.05)
        v_alpha = alpha.values
        v_close = close.values
        res = np.zeros_like(v_close)
        res[0] = v_close[0]
        for i in range(1, len(v_close)):
            res[i] = v_alpha[i] * v_close[i] + (1 - v_alpha[i]) * res[i-1]
        return pd.Series(res, index=close.index)

    @staticmethod
    def calculate_all_indicators(df: pd.DataFrame, window: int = 40, benchmark_returns: pd.Series = None, interval: str = '1h', include_metrics: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Calculates all 11 advanced metrics requested by the user.
        Returns a DataFrame with the same index as the input df.
        """
        res = pd.DataFrame(index=df.index)
        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']
        
        # Scaling factors for related windows
        w_slow = int(window * 2.5)
        w_short = max(2, int(window * 0.35))
        w_mid = max(3, int(window * 0.6))

        # Core returns calculation
        ret = close.pct_change()

        def should_calc(m):
            return include_metrics is None or m in include_metrics

        # 1. EWVA: (EMA_fast - EMA_slow) / STD(price)
        if should_calc('ewva') or should_calc('vol_rank') or should_calc('vwap_z'):
            std_p = close.ewm(span=window).std()
            
        if should_calc('ewva'):
            ema_f = MetricsEngine.ema(close, w_short)
            ema_s = MetricsEngine.ema(close, w_slow)
            res['ewva'] = (ema_f - ema_s) / std_p
        
        # 2. Aroon Oscillator: (AroonUp - AroonDown) / 100
        if should_calc('aroon_osc'):
            au, ad = MetricsEngine.aroon(high, low, w_mid)
            res['aroon_osc'] = (au - ad) / 100
        
        # 4. RSI (Normalized): RSI / 100
        if should_calc('rsi_norm'):
            res['rsi_norm'] = (MetricsEngine.rsi(close, window) - 50) / 100
                
        # 6. Normalized ATR: ATR / Close
        if should_calc('atr_norm') or should_calc('vam') or should_calc('vol_atr'):
            atr_s = MetricsEngine.atr(high, low, close, w_short)
            atr_norm = atr_s / close.replace(0, 1e-9)
            if should_calc('atr_norm'):
                res['atr_norm'] = atr_norm
        
        # 7. Chaikin Money Flow (CMF)
        if should_calc('cmf'):
            mfm = ((close - low) - (high - close)) / (high - low).replace(0, 1e-9)
            mfv = mfm * volume
            res['cmf'] = mfv.ewm(span=window).sum() / volume.ewm(span=window).sum()
        
        # 8. VWAP Deviation Z-Score: (Close - VWAP) / STD(price)
        if should_calc('vwap_z'):
            vwp = MetricsEngine.vwap(df)
            res['vwap_z'] = (close - vwp) / std_p
        
        # 9. Relative Strength Z-Score (vs benchmark)
        if should_calc('rel_strength_z'):
            if benchmark_returns is not None:
                # Align indices
                common_idx = ret.index.intersection(benchmark_returns.index)
                diff = ret.loc[common_idx] - benchmark_returns.loc[common_idx]
                res['rel_strength_z'] = (diff - diff.ewm(span=window).mean()) / diff.ewm(span=window).std()
            else:
                res['rel_strength_z'] = np.nan
            
        # 10. Volatility-Adjusted Momentum (VAM): ROC / (ATR / Close)
        if should_calc('vam'):
            roc = close.pct_change(w_short)
            res['vam'] = roc / atr_norm
        
        # 11. Return Skewness (Rolling)
        if should_calc('skewness'):
            res['skewness'] = ret.rolling(window).skew()
        
        # 12. Lagged Signs
        if should_calc('sign_lag1'): res['sign_lag1'] = np.sign(ret.shift(1))
        if should_calc('sign_lag2'): res['sign_lag2'] = np.sign(ret.shift(2))
        if should_calc('sign_lag3'): res['sign_lag3'] = np.sign(ret.shift(3))

        # 13. Rolling lagged sign
        if should_calc('rolling_sign_lag5'): res['rolling_sign_lag5'] = np.sign(ret.rolling(5).mean())
        if should_calc('rolling_sign_lag10'): res['rolling_sign_lag10'] = np.sign(ret.rolling(10).mean())
        if should_calc('rolling_sign_lag20'): res['rolling_sign_lag20'] = np.sign(ret.rolling(20).mean())

        # 14. Autocorrelation - Optimization: use pd.Series.autocorr only if explicitly needed and small window
        if should_calc('autocorr_1'):
            res['autocorr_1'] = ret.rolling(5).apply(lambda x: x.autocorr(lag=1) if len(x) == 5 else np.nan, raw=False)
        if should_calc('autocorr_5'):
            res['autocorr_5'] = ret.rolling(20).apply(lambda x: x.autocorr(lag=5) if len(x) == 20 else np.nan, raw=False)
        
        # 15. Imbalance Bar
        if should_calc('imbalance_bar'):
            body = (close - df['open']).abs()
            rng = (high - low).replace(0, 1e-9)
            res['imbalance_bar'] = (body / rng) * np.sign(close - df['open'])
        
        # 16. VAMA (Volatility Adjusted Moving Average)
        if should_calc('vama'):
            res['vama'] = MetricsEngine.vama(close, w_short, w_slow) / close
        
        # --- Standard/Legacy Metrics for Snapshot/Table use ---
        if should_calc('volatility') or should_calc('vol_atr'):
            ann_factor = MetricsEngine.get_annual_scaling(interval)
            vol_series = ret.ewm(span=window).std() * np.sqrt(ann_factor)
            if should_calc('volatility'): res['volatility'] = vol_series
            if should_calc('vol_atr'): res['vol_atr'] = vol_series / atr_norm

        if should_calc('vol_imbalance'):
            res['vol_imbalance'] = ret.ewm(span=w_short).std() / ret.ewm(span=w_slow).std()
        
        if should_calc('volume_imbalance'):
            res['volume_imbalance'] = volume.ewm(span=w_short).std() / volume.ewm(span=w_slow).std()
            
        if should_calc('liquidity_impact'):
            res['liquidity_impact'] = ret.abs().ewm(span=window).mean() / volume.ewm(span=window).mean()
        
        if should_calc('vol_rank'):
            vol = ret.rolling(window).std()
            res['vol_rank'] = vol.rolling(window).apply(lambda x: (x <= x.iloc[-1]).mean())
        
        # 228. FIP (Frog-in-the-Pan)
        if should_calc('fip'):
            def fip_func(x):
                total_ret = np.sum(x)
                sign_ret = np.sign(total_ret)
                n = len(x)
                neg_pct = np.sum(x < 0) / n
                pos_pct = np.sum(x > 0) / n
                return sign_ret * (neg_pct - pos_pct)
            res['fip'] = ret.rolling(window).apply(fip_func, raw=True)
        
        # Price Z-Score & SMA Diff
        if should_calc('price_zscore'):
            res['price_zscore'] = (close - close.rolling(window=window).mean()) / close.rolling(window=window).std()
        
        # ADF Statistics
        if should_calc('adf_hist') or should_calc('adf_stat'):
            hist, tau, _ = MetricsEngine.calculate_custom_adf_series(close.values, lookback=window)
            if should_calc('adf_hist'): res['adf_hist'] = hist.values
            if should_calc('adf_stat'): res['adf_stat'] = tau.values

        return res

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

        # Prepare arrays
        vals = series
        dy = np.diff(vals)
        x = vals[:-1]
        
        # Align lengths (dy has length N-1)
        n = len(dy)
        if n < lookback: 
            nan_s = pd.Series(np.nan, index=range(len(series)))
            return nan_s, nan_s, nan_s
                
        dy_s = pd.Series(dy)
        x_s = pd.Series(x)
        
        # Rolling sums with min_periods=1 for robustness
        sum_x = x_s.ewm(span=lookback, min_periods=1).sum()
        sum_y = dy_s.ewm(span=lookback, min_periods=1).sum()
        sum_xx = (x_s**2).ewm(span=lookback, min_periods=1).sum()
        sum_xy = (x_s * dy_s).ewm(span=lookback, min_periods=1).sum()
        sum_yy = (dy_s**2).ewm(span=lookback, min_periods=1).sum()
        
        # Helper for vectorized beta, alpha
        denom = lookback * sum_xx - sum_x**2
        
        # Handle division by zero
        valid = np.abs(denom) > 1e-10
        beta = np.zeros_like(denom)
        alpha = np.zeros_like(denom)
        
        beta[valid] = (lookback * sum_xy[valid] - sum_x[valid] * sum_y[valid]) / denom[valid]
        alpha[valid] = (sum_y[valid] - beta[valid] * sum_x[valid]) / lookback
        
        # SE calculation requires RSS
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
        tau_sma = tau_s.ewm(span=10, min_periods=1).mean()
        
        # "tauADF_smooth = ta.ema(tauADF, 50)"
        # EMA usually handles NaNs/start better than SMA
        tau_smooth = tau_sma.ewm(span=50, adjust=False, min_periods=1).mean()
        
        hist = tau_sma - tau_smooth
                
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
    def calculate_sortino_ratio(log_returns: np.ndarray, target: float = 0.0, interval: str = '1h') -> float:
        """
        Sortino Ratio = (Mean Return - Target) / Downside Deviation
        """
        if len(log_returns) < 2: return 0.0
        
        excess_returns = log_returns - target
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0:
            return float('inf') if np.mean(excess_returns) > 0 else 0.0
            
        downside_std = np.std(downside_returns)
        if downside_std < 1e-9: return 0.0
        
        mean_ret = np.mean(excess_returns)
        ann_factor = MetricsEngine.get_annual_scaling(interval)
        
        return float((mean_ret / downside_std) * np.sqrt(ann_factor))

    @staticmethod
    def calculate_max_drawdown(prices: np.ndarray) -> float:
        """
        Max Drawdown = Min((Price / RunningMax) - 1)
        """
        if len(prices) < 2: return 0.0
        
        running_max = np.maximum.accumulate(prices)
        drawdowns = (prices / running_max) - 1
        return float(np.min(drawdowns))

    @staticmethod
    def calculate_avg_drawdown(prices: np.ndarray) -> float:
        """
        Average of all drawdowns (where DD < 0)
        """
        if len(prices) < 2: return 0.0
        
        running_max = np.maximum.accumulate(prices)
        drawdowns = (prices / running_max) - 1
        
        # Filter for actual drawdowns
        negative_dds = drawdowns[drawdowns < 0]
        if len(negative_dds) == 0: return 0.0
        
        return float(np.mean(negative_dds))

    @staticmethod
    def calculate_win_rate(log_returns: np.ndarray) -> float:
        """
        Percentage of positive returns.
        """
        if len(log_returns) == 0: return 0.0
        
        wins = np.sum(log_returns > 0)
        return float(wins / len(log_returns))

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
        
        # Calculate all standard indicators
        all_inds = MetricsEngine.calculate_all_indicators(df, window=window, benchmark_returns=benchmark_returns, interval=interval)
        
        if metric_name in all_inds.columns:
            res = all_inds[metric_name]
        else:
            # Fallback for metrics not in the all-in-one helper (if any)
            if metric_name == 'count':
                res = pd.Series(len(df), index=df.index)
            else:
                return pd.Series()
            
        # Apply Step (Stride)
        if step > 1:
            res = res.iloc[::step]
            
        return res

    def compute_all_metrics(self, prices_data: Dict[str, pd.DataFrame], interval: str = '1h', benchmark_symbol: str = 'BTCUSDT', benchmark_returns: Optional[pd.Series] = None, window: int = 40) -> pd.DataFrame:
        """
        Main pipeline to compute metrics for all symbols.
        """
        results = []
        
        # 1. Prepare Benchmark if not provided
        if benchmark_returns is None and benchmark_symbol in prices_data:
            b_df = prices_data[benchmark_symbol]
            b_close = pd.to_numeric(b_df['close'], errors='coerce').ffill().fillna(0)
            benchmark_returns = b_close.pct_change().dropna()
            
        for symbol, df in prices_data.items():
            try:
                if df.empty: continue
                
                # Cache Check
                last_ts = df['open_time'].max() if 'open_time' in df.columns else None
                cache_key = (symbol, interval, last_ts, len(df))
                if cache_key in self._results_cache:
                    results.append(self._results_cache[cache_key])
                    continue

                # Standardize and clean prices
                prices = pd.to_numeric(df['close'], errors='coerce').ffill().fillna(0).values.astype(float)
                if len(prices) < 2: continue
                
                # Advanced Metrics (Standardized)
                adv_df = self.calculate_all_indicators(df, window=window, benchmark_returns=benchmark_returns)
                latest_adv = adv_df.iloc[-1].to_dict()
                
                row = {
                    'symbol': symbol,
                    'count': len(prices)
                }
                # Add all standard metrics to row
                row.update(latest_adv)
                
                # Store in cache
                self._results_cache[cache_key] = row
                results.append(row)
                
            except Exception as e:
                print(f"Error calculating metrics for {symbol}: {e}")
                
        return pd.DataFrame(results)
