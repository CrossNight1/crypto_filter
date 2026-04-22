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
        w_slow = int(window * 3)
        w_short = int(window * 0.5)
        w_mid = window

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
        if should_calc('atr_norm') or should_calc('vol_atr'):
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
            
        # 10. Volatility-Adjusted Momentum (VAM): ROC / (ATR / Vola)
        if should_calc('vam'):
            roc = close.pct_change(w_short)
            res['vam'] = roc / ret.ewm(span=window).std()
        
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
                pos = x[x > 0].sum()
                neg = np.abs(x[x < 0].sum())
                
                denom = pos + neg
                if denom == 0:
                    return 0.0
                
                return (pos - neg) / denom

            res['fip'] = ret.rolling(window).apply(fip_func, raw=True)
        
        # Price Z-Score & SMA Diff
        if should_calc('price_zscore'):
            res['price_zscore'] = (close - close.rolling(window=window).mean()) / close.rolling(window=window).std()
        
        # 271. ADF Statistics
        if should_calc('adf_hist') or should_calc('adf_stat'):
            hist, tau, _ = MetricsEngine.calculate_custom_adf_series(close.values, lookback=window)
            if should_calc('adf_hist'): res['adf_hist'] = hist.values
            if should_calc('adf_stat'): res['adf_stat'] = tau.values

        # 272. Rolling Drawdown Metrics
        if should_calc('max_drawdown') or should_calc('avg_drawdown'):
            # MDD(t) = (Price(t) / RollingMax(t)) - 1
            rolling_max = low.rolling(window=window, min_periods=1).max()
            drawdowns = (low / rolling_max) - 1.0
            
            if should_calc('max_drawdown'):
                # Max drawdown in the rolling window
                res['max_drawdown'] = drawdowns.rolling(window=window, min_periods=1).min()
            
            if should_calc('avg_drawdown'):
                # Average of drawdown events (where DD < 0)
                res['avg_drawdown'] = drawdowns.where(drawdowns < 0).rolling(window=window, min_periods=1).mean()

        # 273. Breakout Score
        if should_calc('breakout_score_up') or should_calc('breakout_score_down'):
            bku, bkd = MetricsEngine.calculate_breakout_score(df)
            if should_calc('breakout_score_up'): res['breakout_score_up'] = bku
            if should_calc('breakout_score_down'): res['breakout_score_down'] = bkd

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
    def calculate_breakout_score(df: pd.DataFrame, 
                                 len_up: int = 10, len_down: int = 10, 
                                 mult_base: float = 1.0, calcMethod: str = "Stdev",
                                 maMin: int = 10, period: int = 50,
                                 volFactor: float = 0.15, z_score_val: float = 0.5) -> tuple:
        """
        Calculates breakout trendline score.
        Returns: score_up, score_down (as pd.Series)
        """
        close = df['close'].values.astype(float)
        high = df['high'].values.astype(float)
        low = df['low'].values.astype(float)
        n = len(close)
        
        if n < max(len_up, len_down, period):
            return pd.Series(0.0, index=df.index), pd.Series(0.0, index=df.index)

        def rma(x, n_period):
            a = np.full_like(x, np.nan)
            if len(x) < n_period: return a
            start_idx = -1
            for i in range(len(x)):
                if not np.isnan(x[i]):
                    if start_idx == -1:
                        start_idx = i
                    if i - start_idx + 1 == n_period:
                        a[i] = np.nanmean(x[start_idx:i+1])
                        alpha = 1.0 / n_period
                        for j in range(i+1, len(x)):
                            a[j] = alpha * x[j] + (1 - alpha) * a[j-1]
                        break
            return a

        def stdev_func(x, n_period):
            return pd.Series(x).rolling(n_period).std(ddof=0).values

        def get_pivots(src, left, right, is_high):
            pivots = np.full(n, np.nan)
            for i in range(left + right, n):
                idx = i - right
                is_pivot = True
                
                for j in range(idx - left, idx):
                    if is_high and src[j] >= src[idx]:
                        is_pivot = False; break
                    if not is_high and src[j] <= src[idx]:
                        is_pivot = False; break
                if not is_pivot: continue
                
                for j in range(idx + 1, i + 1):
                    if is_high and src[j] >= src[idx]:
                        is_pivot = False; break
                    if not is_high and src[j] <= src[idx]:
                        is_pivot = False; break
                        
                if is_pivot:
                    pivots[i] = src[idx]
            return pivots

        tr = np.zeros_like(close)
        tr[0] = high[0] - low[0]
        for i in range(1, n):
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        atr_len_up = rma(tr, len_up)
        atr_len_down = rma(tr, len_down)
        atr_14 = rma(tr, 14)
        
        logret = np.zeros_like(close)
        with np.errstate(divide='ignore', invalid='ignore'):
            logret[1:] = np.log(close[1:] / close[:-1])
        logret[0] = 0.0
        logret = np.nan_to_num(logret)
        
        vol_vama = stdev_func(logret, period)
        targetVol = stdev_func(logret, period * 30) * volFactor
        
        vamaC = np.full(n, np.nan)
        
        span_val = float(maMin)
        for i in range(n):
            if not np.isnan(vol_vama[i]) and vol_vama[i] > 0 and not np.isnan(targetVol[i]):
                scale = min(targetVol[i] / vol_vama[i], 1.0)
                span_val = max(np.floor(maMin / scale), 1.0)
            
            alpha = 1.0 / (span_val + 1.0)
            
            if i == 0 or np.isnan(vamaC[i-1]):
                vamaC[i] = close[i]
            else:
                vamaC[i] = alpha * close[i] + (1 - alpha) * vamaC[i-1]

        vr = np.ones(n)
        for i in range(n):
            if not np.isnan(vol_vama[i]) and not np.isnan(targetVol[i]) and vol_vama[i] > 0 and targetVol[i] > 0:
                vr[i] = vol_vama[i] / targetVol[i]

        beta = 0.5
        mult_dyn = mult_base * np.power(vr, -beta)
        mult_dyn = np.clip(mult_dyn, mult_base * 0.5, mult_base * 2.0)
        
        ph = get_pivots(high, len_up, len_up, True)
        pl = get_pivots(low, len_down, len_down, False)
        
        slope_up = np.zeros(n)
        slope_down = np.zeros(n)
        
        if calcMethod == "Atr":
            slope_up = (atr_len_up / len_up) * mult_dyn
            slope_down = (atr_len_down / len_down) * mult_dyn
        elif calcMethod == "Stdev":
            slope_up = (stdev_func(close, len_up) / len_up) * mult_dyn
            slope_down = (stdev_func(close, len_down) / len_down) * mult_dyn
        elif calcMethod == "Linreg":
            def sma(x, p):
                return pd.Series(x).rolling(p).mean().values
            def variance(x, p):
                return pd.Series(x).rolling(p).var(ddof=0).values
                
            bar_index = np.arange(n)
            sma_src_n_up = sma(close * bar_index, len_up)
            sma_src_up = sma(close, len_up)
            sma_n_up = sma(bar_index, len_up)
            var_n_up = variance(bar_index, len_up)
            var_safe_up = np.where(var_n_up == 0, 1, var_n_up)
            slope_up = np.abs(sma_src_n_up - sma_src_up * sma_n_up) / var_safe_up / 2 * mult_dyn
            
            sma_src_n_down = sma(close * bar_index, len_down)
            sma_src_down = sma(close, len_down)
            sma_n_down = sma(bar_index, len_down)
            var_n_down = variance(bar_index, len_down)
            var_safe_down = np.where(var_n_down == 0, 1, var_n_down)
            slope_down = np.abs(sma_src_n_down - sma_src_down * sma_n_down) / var_safe_down / 2 * mult_dyn

        upper = np.zeros(n)
        lower = np.zeros(n)
        slope_ph = np.zeros(n)
        slope_pl = np.zeros(n)
        
        for i in range(1, n):
            is_ph = not np.isnan(ph[i])
            is_pl = not np.isnan(pl[i])
            
            c_slope_up = slope_up[i] if not np.isnan(slope_up[i]) else 0.0
            c_slope_down = slope_down[i] if not np.isnan(slope_down[i]) else 0.0
            
            slope_ph[i] = c_slope_up if is_ph else slope_ph[i-1]
            slope_pl[i] = c_slope_down if is_pl else slope_pl[i-1]
            
            upper[i] = ph[i] if is_ph else upper[i-1] - slope_ph[i]
            lower[i] = pl[i] if is_pl else lower[i-1] + slope_pl[i]
            
        d_up = np.zeros(n)
        d_down = np.zeros(n)
        
        for i in range(1, n):
            c = close[i]
            a = atr_14[i] if not np.isnan(atr_14[i]) and atr_14[i] != 0 else 1.0
            
            if c > upper[i]:
                d_up[i] = (c - (upper[i] - slope_ph[i])) / a
            else:
                d_up[i] = 0
                
            if c < lower[i]:
                d_down[i] = ((lower[i] + slope_pl[i]) - c) / a
            else:
                d_down[i] = 0
                
        k = 0.1
        bku_score = d_up * np.exp(-k * d_up * d_up)
        bkd_score = d_down * np.exp(-k * d_down * d_down)
        
        score_up = 0.5 * bku_score
        score_down = 0.5 * bkd_score
        
        return pd.Series(score_up, index=df.index).fillna(0), pd.Series(score_down, index=df.index).fillna(0)

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
             return pd.Series(dtype=float)
             
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
                return pd.Series(dtype=float)
            
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
            
        from src.data import BinanceFuturesFetcher
        import time
        fetcher = BinanceFuturesFetcher()
            
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
                
                # # Fetch orderbook metrics
                # book_stats = fetcher.get_books_status(symbol)
                # row['orderbook_imbalance'] = book_stats.get('orderbook_imbalance', np.nan)
                # row['spread'] = book_stats.get('impact_spread', np.nan)
                
                # Small sleep to respect rate limits (orderbook endpoint has weight 50)
                time.sleep(0.05)
                
                # Store in cache
                self._results_cache[cache_key] = row
                results.append(row)
                
            except Exception as e:
                print(f"Error calculating metrics for {symbol}: {e}")
                
        return pd.DataFrame(results)


import numpy as np
from scipy.stats import norm, t

def copula_cond_probs(u_hist, v_hist, u_curr, v_curr, method="gaussian", **kwargs):
    """
    Compute conditional probabilities:
        P(U <= u_curr | V=v_curr) and P(V <= v_curr | U=u_curr)
    
    Parameters
    ----------
    u_hist, v_hist : array-like
        Historical percentiles (0-1)
    u_curr, v_curr : float
        Current percentiles (0-1)
    method : str
        Copula type: "gaussian", "t", "clayton", "gumbel"
    kwargs : dict
        Extra parameters:
            - For "t": df = degrees of freedom
            - For "clayton"/"gumbel": theta = copula parameter (>0)

    Returns
    -------
    tuple
        (P(U <= u_curr | V=v_curr), P(V <= v_curr | U=u_curr))
    """
    u_hist = np.asarray(u_hist)
    v_hist = np.asarray(v_hist)
    
    if method.lower() == "gaussian":
        z1_hist = norm.ppf(u_hist)
        z2_hist = norm.ppf(v_hist)
        z1_curr = norm.ppf(u_curr)
        z2_curr = norm.ppf(v_curr)
        rho = np.corrcoef(z1_hist, z2_hist)[0,1]

        p_uv = norm.cdf((z1_curr - rho * z2_curr)/np.sqrt(1 - rho**2))
        p_vu = norm.cdf((z2_curr - rho * z1_curr)/np.sqrt(1 - rho**2))
        return p_uv, p_vu

    elif method.lower() == "t":
        df = kwargs.get("df", 5)
        z1_hist = t.ppf(u_hist, df)
        z2_hist = t.ppf(v_hist, df)
        z1_curr = t.ppf(u_curr, df)
        z2_curr = t.ppf(v_curr, df)
        rho = np.corrcoef(z1_hist, z2_hist)[0,1]

        cond_var_uv = ((df + z2_curr**2)/(df+1))*(1 - rho**2)
        cond_mean_uv = rho * z2_curr
        cond_var_vu = ((df + z1_curr**2)/(df+1))*(1 - rho**2)
        cond_mean_vu = rho * z1_curr

        p_uv = t.cdf(z1_curr, df+1, loc=cond_mean_uv, scale=np.sqrt(cond_var_uv))
        p_vu = t.cdf(z2_curr, df+1, loc=cond_mean_vu, scale=np.sqrt(cond_var_vu))
        return p_uv, p_vu

    elif method.lower() == "clayton":
        theta = kwargs.get("theta", 2)
        if theta <= 0:
            raise ValueError("Clayton copula θ must be > 0")

        base = np.maximum(u_curr**(-theta) + v_curr**(-theta) - 1, 1e-10)
        p_uv = (v_curr**(-theta - 1)) * (base**(-(1+1/theta)))
        p_vu = (u_curr**(-theta - 1)) * (base**(-(1+1/theta)))
        return p_uv, p_vu

    elif method.lower() == "gumbel":
        theta = kwargs.get("theta", 2)
        if theta < 1:
            raise ValueError("Gumbel copula θ must be ≥ 1")

        u_tilde = -np.log(u_curr)
        v_tilde = -np.log(v_curr)
        sum_t = np.maximum(u_tilde**theta + v_tilde**theta, 1e-10)
        
        C = np.exp(-(sum_t**(1/theta)))
        
        p_uv = C * (1.0 / v_curr) * (sum_t**(1/theta - 1)) * (v_tilde**(theta - 1))
        p_vu = C * (1.0 / u_curr) * (sum_t**(1/theta - 1)) * (u_tilde**(theta - 1))
        return p_uv, p_vu

    else:
        raise ValueError(f"Unknown copula method: {method}")