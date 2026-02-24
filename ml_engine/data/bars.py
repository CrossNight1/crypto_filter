import pandas as pd
import numpy as np
from numba import njit
from scipy.optimize import minimize
from scipy.stats import skew, kurtosis

@njit
def _volume_bar_core(open_, high_, low_, close_, volume_, threshold):
    n = len(close_)
    
    o_list = []
    h_list = []
    l_list = []
    c_list = []
    v_list = []
    t_list = []
    
    cum_vol = 0.0
    o = h = l = c = 0.0
    start_idx = 0
    
    for i in range(n):
        if cum_vol == 0.0:
            o = open_[i]
            h = high_[i]
            l = low_[i]
            start_idx = i
        
        cum_vol += volume_[i]
        if high_[i] > h: h = high_[i]
        if low_[i] < l: l = low_[i]
        c = close_[i]
        
        if cum_vol >= threshold:
            o_list.append(o)
            h_list.append(h)
            l_list.append(l)
            c_list.append(c)
            v_list.append(cum_vol)
            t_list.append(start_idx)
            cum_vol = 0.0
    
    return t_list, o_list, h_list, l_list, c_list, v_list


@njit
def _dollar_bar_core(open_, high_, low_, close_, volume_, threshold):
    n = len(close_)
    
    o_list = []
    h_list = []
    l_list = []
    c_list = []
    d_list = []
    v_list = []
    t_list = []
    
    cum_dollar = 0.0
    cum_vol = 0.0
    o = h = l = c = 0.0
    start_idx = 0
    
    for i in range(n):
        if cum_dollar == 0.0:
            o = open_[i]
            h = high_[i]
            l = low_[i]
            start_idx = i
        
        dollar_val = close_[i] * volume_[i]
        cum_dollar += dollar_val
        cum_vol += volume_[i]
        
        if high_[i] > h: h = high_[i]
        if low_[i] < l: l = low_[i]
        c = close_[i]
        
        if cum_dollar >= threshold:
            o_list.append(o)
            h_list.append(h)
            l_list.append(l)
            c_list.append(c)
            d_list.append(cum_dollar)
            v_list.append(cum_vol)
            t_list.append(start_idx)
            cum_dollar = 0.0
            cum_vol = 0.0
    
    return t_list, o_list, h_list, l_list, c_list, d_list, v_list


def construct_volume_bars(df, threshold):
    if df is None or df.empty:
        return pd.DataFrame()
        
    open_ = df["open"].values.astype(np.float64)
    high_ = df["high"].values.astype(np.float64)
    low_ = df["low"].values.astype(np.float64)
    close_ = df["close"].values.astype(np.float64)
    volume_ = df["volume"].values.astype(np.float64)
    
    t, o, h, l, c, v = _volume_bar_core(open_, high_, low_, close_, volume_, threshold)
    
    if len(o) == 0:
        return pd.DataFrame()
    
    idx = df.index[np.array(t)]
    out = pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "volume": v}, index=idx)
    return out


def construct_dollar_bars(df, threshold):
    if df is None or df.empty:
        return pd.DataFrame()

    open_ = df["open"].values.astype(np.float64)
    high_ = df["high"].values.astype(np.float64)
    low_ = df["low"].values.astype(np.float64)
    close_ = df["close"].values.astype(np.float64)
    volume_ = df["volume"].values.astype(np.float64)
    
    t, o, h, l, c, d, v = _dollar_bar_core(open_, high_, low_, close_, volume_, threshold)
    
    if len(o) == 0:
        return pd.DataFrame()
    
    idx = df.index[np.array(t)]
    out = pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "volume": v, "dollar_vol": d}, index=idx)
    return out


def calibrate_bar_threshold(df, bar_type="Dollar Bars", N_min=10000):
    """
    Auto-calibrate Volume or Dollar thresholds to minimize normality loss.
    Returns: optimal_threshold
    """
    if df is None or df.empty:
        return None

    def calibration_loss(returns):
        if len(returns) < 30:
            return 1e10
        N = len(returns)
        s = skew(returns)
        # k = kurtosis(returns) - 3 # Excess kurtosis
        return (s**2 + 1000 * max(0, (N_min - N)/N_min)**2)

    def objective_volume(vol_threshold):
        try:
            vol_df = construct_volume_bars(df, int(vol_threshold[0]))
            if vol_df.empty or len(vol_df) < 10:
                return 1e10
            returns = np.log(vol_df['close'] / vol_df['close'].shift(1)).dropna()
            return calibration_loss(returns)
        except:
            return 1e10

    def objective_dollar(dollar_threshold):
        try:
            dollar_df = construct_dollar_bars(df, int(dollar_threshold[0]))
            if dollar_df.empty or len(dollar_df) < 10:
                return 1e10
            returns = np.log(dollar_df['close'] / dollar_df['close'].shift(1)).dropna()
            return calibration_loss(returns)
        except:
            return 1e10

    if bar_type == "Volume Bars":
        best_loss = 1e10
        best_th = 10000
        for x0 in [1000, 5000, 10000, 50000]:
            res = minimize(
                lambda x: objective_volume([int(10**x[0])]),
                x0=[np.log10(x0)],
                bounds=[(np.log10(100), np.log10(1000000000))],
                method='Nelder-Mead',
                options={'maxiter': 100, 'xatol': 1e-2, 'fatol': 1e-2}
            )
            if res.fun < best_loss:
                best_loss = res.fun
                best_th = int(10**res.x[0])
        return best_th
    else:
        best_loss = 1e10
        best_th = 1000000000
        for x0 in [100000000, 500000000, 1000000000, 5000000000]:
            res = minimize(
                lambda x: objective_dollar([int(10**x[0])]),
                x0=[np.log10(x0)],
                bounds=[(np.log10(1000), np.log10(100000000000))],
                method='Nelder-Mead',
                options={'maxiter': 100, 'xatol': 1e-2, 'fatol': 1e-2}
            )
            if res.fun < best_loss:
                best_loss = res.fun
                best_th = int(10**res.x[0])
        return best_th
