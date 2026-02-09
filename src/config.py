"""
Application Configuration
Update this file to change titles, descriptions, symbols, and links.
"""

# --- APP SETTINGS ---
APP_TITLE = "Crypto Market Radar"
APP_ICON = "chart_with_upwards_trend" # Using a standard icon name if possible, or leave as text
APP_LAYOUT = "wide"

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
MANDATORY_CRYPTO = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'LINAUSDT']

# Benchmark used for Relative Strength and Beta
BENCHMARK_SYMBOL = 'BTCUSDT'

# Default timeframes available in Data Loader
AVAILABLE_INTERVALS = ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d', '1w']
DEFAULT_FETCH_INTERVALS = ['1h', '4h', '1d']

# --- EXTERNAL LINKS ---
BINANCE_URL = "https://www.binance.com/en/futures/"

# --- METRIC LABELS ---
METRIC_LABELS = {
    'beta': 'Beta',
    'alpha': 'Alpha',
    'volatility': 'Volatility',
    'r_squared': 'R-Squared',
    'adf_hist': 'ADF Regime',
    'adf_stat': 'ADF Statistic',
    'sharpe': 'Sharpe Ratio',
    'fip': 'FIP',
    'count': 'Data Points',
    'return': 'Return',
    'metric_return': 'Metric Return',
    'ewva': 'EWVA',
    'aroon_osc': 'Aroon Oscillator',
    'bbp': 'BBP (Bollinger Position)',
    'rsi_norm': 'RSI (Normalized)',
    'return_z': 'Return Z-Score',
    'atr_norm': 'Normalized ATR',
    'cmf': 'Chaikin Money Flow',
    'vwap_z': 'VWAP Z-Score',
    'rel_strength_z': 'Rel Strength Z-Score',
    'price_zscore': 'Price Z-Score',
    'price_sma_diff': 'Price vs SMA %',
    'vam': 'VAM (Vol-Adj Momentum)',
    'skewness': 'Return Skewness',
    'return_lag1': 'Return Lag 1',
    'return_lag2': 'Return Lag 2',
    'return_lag3': 'Return Lag 3',
    'autocorr_5': 'Autocorr (5)',
    'None': 'None'
}

# List of all available numeric metrics for axes
ALL_METRICS = [
    'beta', 'alpha', 'volatility', 'sharpe', 'fip', 'return', 
    'metric_return', 'adf_hist', 'adf_stat', 'ewva', 'aroon_osc', 
    'bbp', 'rsi_norm', 'return_z', 'atr_norm', 'cmf', 'vwap_z', 
    'rel_strength_z', 'price_zscore', 'price_sma_diff', 'vam', 'skewness',
    'return_lag1', 'return_lag2', 'return_lag3', 'autocorr_5', 'ewma'
]
