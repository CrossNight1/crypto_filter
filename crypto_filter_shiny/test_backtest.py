"""
Test script for backtest engine
"""
import sys
sys.path.append('/Users/leoinv/Documents/CODE/crypto_filter/crypto_filter_shiny')

import pandas as pd
import numpy as np
from src.backtest import BacktestEngine
from src.data import DataManager

# Load some sample data
manager = DataManager()
df = manager.load_data('BTCUSDT', '1h')

if df is not None and not df.empty:
    # Ensure datetime index
    if 'open_time' in df.columns:
        df = df.set_index(pd.to_datetime(df['open_time']))
    
    # Take last 500 bars for testing
    df = df.tail(500).copy()
    
    # Generate some simple signals (for testing)
    # Simple momentum strategy: buy when price > MA, sell when price < MA
    df['ma'] = df['close'].rolling(20).mean()
    signals = pd.Series(0, index=df.index)
    signals[df['close'] > df['ma']] = 1
    signals[df['close'] < df['ma']] = -1
    
    # Calculate volatility
    returns = np.log(df['close'] / df['close'].shift(1))
    volatility = returns.rolling(20).std()
    
    # Run backtest
    print("Running backtest...")
    engine = BacktestEngine(
        initial_cash=10000,
        commission=0.001,
        tp_multiplier=2.0,
        sl_multiplier=1.0
    )
    
    trade_results, metrics = engine.run(df, signals, volatility)
    
    print("\n=== Backtest Results ===")
    print(f"Total Trades: {metrics['total_trades']}")
    print(f"Winning Trades: {metrics['winning_trades']}")
    print(f"Losing Trades: {metrics['losing_trades']}")
    print(f"Win Rate: {metrics['win_rate']:.2%}")
    print(f"Total Return: {metrics['total_return']:.2%}")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
    
    if len(trade_results) > 0:
        print("\n=== Sample Trades ===")
        print(trade_results.head(10))
        
        # Generate meta labels
        print("\n=== Generating Meta Labels ===")
        meta_labels = engine.generate_meta_labels(df, signals, volatility)
        print(f"Meta Labels Distribution:")
        print(meta_labels.value_counts())
    
else:
    print("Failed to load data")
