"""
Application Configuration
Update this file to change titles, descriptions, symbols, and links.
"""


# --- APP SETTINGS ---
APP_TITLE = "Crypto Market Radar"
APP_ICON = "LEO"
APP_LAYOUT = "wide"
THEME = "quartz"
BG_COLOR = "#1a1a1a"

# --- TIMEZONE SETTING ---
# Global offset from UTC in hours (e.g., 7 for UTC+7, -5 for EST)
TIMEZONE_OFFSET = 7

# --- SIDEBAR & WELCOME ---
SIDEBAR_INFO = "Select a module to proceed"
WELCOME_TITLE = "Crypto Market Radar"
WELCOME_TEXT = """
### Welcome to the Advanced Metrics Dashboard

This application allows you to analyze Binance Perpetual Futures with advanced statistical metrics.

#### Modules:
1. **Data Loader**: Fetch and cache historical market data. 
2. **Market Radar**: Visualize and explore metrics.
3. **Regression**: Analyze predictive power of indicators.
"""

# --- DATA SETTINGS ---
# Symbols that are always fetched
MANDATORY_CRYPTO = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'METISUSDT', 'BTCDOMUSDT']

# Benchmark used for Relative Strength and Beta
BENCHMARK_SYMBOL = 'BTCUSDT'

# Default timeframes available in Data Loader
AVAILABLE_INTERVALS = ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d', '1w']
DEFAULT_FETCH_INTERVALS = ['1h', '4h', '1d']

# --- EXTERNAL LINKS ---
BINANCE_URL = "https://www.binance.com/en/futures/"

# --- METRIC LABELS ---
METRIC_LABELS = {
    'volatility': 'Volatility',
    'adf_hist': 'ADF Regime',
    'adf_stat': 'ADF Statistic',
    'fip': 'FIP',
    'count': 'Data Points',
    'ewva': 'EWVA',
    'aroon_osc': 'Aroon Oscillator',
    'rsi_norm': 'RSI (Normalized)',
    'atr_norm': 'Normalized ATR',
    'cmf': 'Chaikin Money Flow',
    'vwap_z': 'VWAP Z-Score',
    'rel_strength_z': 'Rel Strength Z-Score',
    'price_zscore': 'Price Z-Score',
    'vam': 'VAM (Vol-Adj Momentum)',
    'skewness': 'Return Skewness',
    'sign_lag1': 'Sign Lag 1',
    'sign_lag2': 'Sign Lag 2',
    'sign_lag3': 'Sign Lag 3',
    'rolling_sign_lag5': 'Rolling Sign (5)',
    'rolling_sign_lag10': 'Rolling Sign (10)',
    'rolling_sign_lag20': 'Rolling Sign (20)',
    'autocorr_1': 'Autocorr (1)',
    'autocorr_5': 'Autocorr (5)',
    'imbalance_bar': 'Imbalance Bar',
    'vol_imbalance': 'Volatility Imbalance',
    'volume_imbalance': 'Volume Imbalance',
    'vol_atr': 'Vol/ATR Ratio',
    'vama': 'VAMA',
    'liquidity_impact': 'Liquidity Impact',
    'vol_rank': 'Volatility Rank',
    'None': 'None'
}

# List of all available numeric metrics for axes
ALL_METRICS = [
    'volatility', 'fip', 
    'adf_hist', 'adf_stat', 'ewva', 'aroon_osc', 
    'rsi_norm', 'atr_norm', 'cmf', 'vwap_z', 'price_zscore', 'vam', 'skewness',
    'sign_lag1', 'sign_lag2', 'sign_lag3', 'rolling_sign_lag5', 'rolling_sign_lag10', 'rolling_sign_lag20',
    'autocorr_1', 'autocorr_5', 'imbalance_bar', 'vol_imbalance', 'volume_imbalance', 'vol_atr', 'vama', 'liquidity_impact', 'vol_rank'
]

DEFAULT_FEATURES = [
    'fip', 
    'adf_hist', 'ewva', 'aroon_osc', 
    'atr_norm', 'vol_atr', 'vama', 'cmf', 'price_zscore', 'vam', 'skewness',
    'sign_lag1', 'sign_lag2', 'sign_lag3',
    'autocorr_1', 'autocorr_5', 
    'imbalance_bar', 'vol_imbalance', 'volume_imbalance', 'liquidity_impact', 'vol_rank'
]
