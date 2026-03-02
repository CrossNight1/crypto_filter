import backtrader as bt
import pandas as pd
import numpy as np
from typing import Dict, Tuple


class SignalStrategy(bt.Strategy):

    params = (
        ('signals', None),
        ('tp_multiplier', 2.0),
        ('sl_multiplier', 2.0),
        ('volatility', None),
        ('position_size', None),
        ('min_holding_bar', 2),
        ('max_holding_bar', 10),
        ('max_positions', 10)  # max simultaneous positions
    )

    def __init__(self):
        self.open_positions = []  # list of dicts: each dict represents an open position
        self.trade_results = []
        self.signal_history = {}
        self.size_history = {}
        self.equity_history = {}

        self.signal_dict = self.params.signals['signal'].to_dict() if self.params.signals is not None else {}
        self.vol_dict = self.params.volatility.to_dict() if self.params.volatility is not None else {}
        self.position_size_dict = self.params.position_size.to_dict() if self.params.position_size is not None else {}

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status == order.Completed:
            # Check if this was an entry order
            for pos in self.open_positions:
                if pos['status'] == 'PENDING_ENTRY' and pos['entry_ref'] == order.ref:
                    pos['entry_price'] = order.executed.price
                    pos['entry_bar'] = len(self)
                    pos['entry_comm'] = order.executed.comm
                    pos['status'] = 'OPEN'
                    # logger.log("Backtest", "DEBUG", f"Entry Completed: {pos['signal']} @ {pos['entry_price']}")
                    return

            # Check if this was an exit order
            for pos in self.open_positions:
                if pos['status'] == 'CLOSING' and pos['exit_ref'] == order.ref:
                    exit_price = order.executed.price
                    pnl = (exit_price - pos['entry_price']) * pos['size'] * pos['signal']
                    pnl_comm = pnl - order.executed.comm - pos['entry_comm']
                    
                    trade_value = pos['entry_price'] * pos['size']
                    pnl_pct = pnl_comm / trade_value if trade_value != 0 else 0

                    result = {
                        'entry_bar': pos['entry_bar'],
                        'exit_bar': len(self),
                        'entry_price': pos['entry_price'],
                        'exit_price': exit_price,
                        'pnl': pnl_comm,
                        'pnl_pct': pnl_pct,
                        'signal': pos['signal'],
                        'outcome': 1 if pnl_comm > 0 else -1
                    }
                    self.trade_results.append(result)
                    self.open_positions.remove(pos)
                    # logger.log("Backtest", "DEBUG", f"Exit Completed: PnL {pnl_comm:.2f}")
                    return
        
        elif order.status in [order.Canceled, order.Margin, order.Rejected, order.Expired]:
            # Order failed, clean up the position record
            # logger.log("Backtest", "WARNING", f"Order failed: {order.status} ref {order.ref}")
            for pos in self.open_positions:
                if pos.get('entry_ref') == order.ref or pos.get('exit_ref') == order.ref:
                    if pos['status'] == 'PENDING_ENTRY':
                        self.open_positions.remove(pos)
                    elif pos['status'] == 'CLOSING':
                        pos['status'] = 'OPEN' # Revert to open because exit failed
                        pos['exit_ref'] = None
                    break

    def notify_trade(self, trade):
        # We now track trade results manually in notify_order for precision with sub-positions
        pass

    def next(self):
        current_idx = self.datas[0].datetime.datetime(0)

        # Record state for plotting - include CLOSING positions as they still affect equity
        self.signal_history[current_idx] = sum([p['signal'] for p in self.open_positions if p['status'] in ['OPEN', 'CLOSING']]) if self.open_positions else 0
        self.size_history[current_idx] = sum([p['size'] * p['signal'] for p in self.open_positions if p['status'] in ['OPEN', 'CLOSING']]) if self.open_positions else 0
        self.equity_history[current_idx] = self.broker.getvalue()

        signal = self.signal_dict.get(current_idx, 0)
        vol = self.vol_dict.get(current_idx, None)
        position_size = self.position_size_dict.get(current_idx, 1.0)
        price = self.data.close[0]
        high = self.data.high[0]
        low = self.data.low[0]

        # Update existing positions
        for pos in self.open_positions:
            if pos['status'] != 'OPEN':
                continue
                
            pos['holding_bar'] += 1
            should_close = False
            
            if pos['holding_bar'] >= self.params.max_holding_bar:
                should_close = True
            elif pos['signal'] == 1 and (high >= pos['tp_level'] or low <= pos['sl_level']):
                should_close = True
            elif pos['signal'] == -1 and (low <= pos['tp_level'] or high >= pos['sl_level']):
                should_close = True

            if should_close:
                order = self.close(size=abs(pos['size']))
                if order:
                    pos['status'] = 'CLOSING'
                    pos['exit_ref'] = order.ref

        # Open new position if signal exists and max_positions not reached
        if signal != 0 and vol is not None and vol > 0:
            active_pos_count = len([p for p in self.open_positions if p['status'] in ['OPEN', 'PENDING_ENTRY']])
            if active_pos_count < self.params.max_positions:
                equity = self.broker.getvalue()
                size = abs(position_size) * equity / price 
                signal_signed = int(np.sign(signal) * np.sign(position_size))

                tp_level = price * (1 + vol * self.params.tp_multiplier * signal_signed)
                sl_level = price * (1 - vol * self.params.sl_multiplier * signal_signed)

                order = None
                if signal_signed > 0:
                    order = self.buy(size=size)
                elif signal_signed < 0:
                    order = self.sell(size=size)

                if order:
                    self.open_positions.append({
                        'entry_ref': order.ref,
                        'signal': signal_signed,
                        'size': size,
                        'tp_level': tp_level,
                        'sl_level': sl_level,
                        'holding_bar': 0,
                        'entry_price': None,
                        'entry_bar': None,
                        'entry_comm': 0, # To be filled in notify_order
                        'status': 'PENDING_ENTRY'
                    })

class BacktestEngine:

    def __init__(self, initial_cash=10_000_000.0, commission=0.0, tp_multiplier=2.0, sl_multiplier=2.0, min_holding_bar=2, max_holding_bar=10, max_positions=10):
        self.initial_cash = initial_cash
        self.commission = commission
        self.tp_multiplier = tp_multiplier
        self.sl_multiplier = sl_multiplier
        self.min_holding_bar = min_holding_bar
        self.max_holding_bar = max_holding_bar
        self.max_positions = max_positions

    def run(self, df, signals, volatility, position_size):

        if not isinstance(df.index, pd.DatetimeIndex):
            df = df.copy()
            df.index = pd.to_datetime(df.index)

        signals = signals.reindex(df.index, fill_value=0)
        volatility = volatility.reindex(df.index)
        position_size = position_size.reindex(df.index, fill_value=1.0)

        signal_df = pd.DataFrame({'signal': signals}, index=df.index)

        cerebro = bt.Cerebro()

        data = bt.feeds.PandasData(
            dataname=df,
            open='open',
            high='high',
            low='low',
            close='close',
            volume='volume'
        )

        cerebro.adddata(data)

        cerebro.addstrategy(
            SignalStrategy,
            signals=signal_df,
            volatility=volatility,
            tp_multiplier=self.tp_multiplier,
            sl_multiplier=self.sl_multiplier,
            position_size=position_size,
            min_holding_bar=self.min_holding_bar,
            max_holding_bar=self.max_holding_bar,
            max_positions=self.max_positions
        )

        cerebro.broker.setcash(self.initial_cash)
        cerebro.broker.setcommission(commission=self.commission)

        results = cerebro.run()
        strat = results[0]
        final_value = cerebro.broker.getvalue()

        trade_results = pd.DataFrame(strat.trade_results) if strat.trade_results else pd.DataFrame()

        equity_curve = pd.Series(strat.equity_history).sort_index()
        equity_curve = equity_curve.reindex(df.index).ffill().fillna(self.initial_cash)

        buy_hold_returns = df['close'].pct_change().fillna(0)
        buy_hold_equity = (1 + buy_hold_returns).cumprod() * self.initial_cash
        buy_hold_equity = buy_hold_equity.reindex(df.index).ffill().fillna(self.initial_cash)

        sig_hist = pd.Series(strat.signal_history).sort_index()
        sig_hist = sig_hist.reindex(df.index, fill_value=0)
        
        size_hist = pd.Series(strat.size_history).sort_index()
        size_hist = size_hist.reindex(df.index, fill_value=0)

        # Manual Metrics Calculation
        # 1. Sharpe Ratio
        eq_returns = equity_curve.pct_change().fillna(0)
        mean_ret = eq_returns.mean()
        std_ret = eq_returns.std()
        
        # Determine frequency for annualization (defaulting to 1h if not inferrable)
        from src.metrics import MetricsEngine
        # We try to infer interval from index if possible, otherwise use a default or pass it?
        # For now, let's assume 1h as a fallback or if we can get it from df.
        # Often interval is available in the index frequency
        interval = '1h' 
        if hasattr(df.index, 'freq') and df.index.freq:
             # Map pandas freq to binance interval if possible
             pass 

        ann_factor = MetricsEngine.get_annual_scaling(interval)
        sharpe = (mean_ret / std_ret * np.sqrt(ann_factor)) if std_ret > 1e-9 else 0.0

        # 2. Max Drawdown
        running_max = equity_curve.cummax()
        drawdown = (equity_curve / running_max) - 1
        max_dd = drawdown.min() # This will be negative, e.g. -0.052

        metrics = {
            'initial_value': self.initial_cash,
            'final_value': final_value,
            'buy_hold_final_value': buy_hold_equity.iloc[-1],
            'total_return': (final_value - self.initial_cash) / self.initial_cash,
            'buy_hold_return': (buy_hold_equity.iloc[-1] - self.initial_cash) / self.initial_cash,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd,
            'total_trades': len(trade_results),
            'winning_trades': len(trade_results[trade_results['outcome'] == 1]) if len(trade_results) > 0 else 0,
            'losing_trades': len(trade_results[trade_results['outcome'] == -1]) if len(trade_results) > 0 else 0,
        }

        metrics['win_rate'] = metrics['winning_trades'] / metrics['total_trades'] if metrics['total_trades'] > 0 else 0

        return trade_results, metrics, equity_curve, buy_hold_equity, sig_hist, size_hist

    def generate_meta_labels(self, df, primary_signals, volatility, position_size):

        trade_results, _, _, _, _, _ = self.run(df, primary_signals, volatility, position_size)

        meta_labels = pd.Series(0, index=df.index)

        if len(trade_results) > 0:
            for _, trade in trade_results.iterrows():
                bar_idx = int(trade['entry_bar']) - 1
                if 0 <= bar_idx < len(df):
                    entry_idx = df.index[bar_idx]
                    meta_labels.loc[entry_idx] = trade['outcome']

        return meta_labels
