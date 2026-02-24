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
        ('max_holding_bar', 10)
    )

    def __init__(self):
        self.order = None
        self.entry_price = None
        self.entry_bar = None
        self.tp_level = None
        self.sl_level = None
        self.holding_bar = 0
        self.current_signal = 0
        self.current_size = 0
        self.trade_results = []
        self.signal_history = {}
        self.size_history = {}

        self.signal_dict = self.params.signals['signal'].to_dict() if self.params.signals is not None else {}
        self.vol_dict = self.params.volatility.to_dict() if self.params.volatility is not None else {}
        self.position_size_dict = self.params.position_size.to_dict() if self.params.position_size is not None else {}

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status == order.Completed:
            self.entry_price = order.executed.price
            self.entry_bar = len(self)

        self.order = None

    def notify_trade(self, trade):
        if trade.isclosed:
            trade_value = abs(trade.size * trade.price) if trade.price else 1
            pnl_pct = trade.pnlcomm / trade_value if trade_value != 0 else 0

            result = {
                'entry_bar': self.entry_bar,
                'exit_bar': len(self),
                'entry_price': self.entry_price,
                'exit_price': self.data.close[0],
                'pnl': trade.pnlcomm,
                'pnl_pct': pnl_pct,
                'signal': self.current_signal,
                'outcome': 1 if trade.pnlcomm > 0 else -1
            }

            self.trade_results.append(result)

    def next(self):
        current_idx = self.datas[0].datetime.datetime(0)
        # Record state for plotting EVERY bar
        self.signal_history[current_idx] = self.current_signal if self.position else 0
        self.size_history[current_idx] = self.current_size if self.current_size else 0

        if self.order:
            return

        current_idx = self.datas[0].datetime.datetime(0)

        signal = self.signal_dict.get(current_idx, 0)
        vol = self.vol_dict.get(current_idx, None)
        position_size = self.position_size_dict.get(current_idx, 1.0)

        if vol is None or vol <= 0:
            return

        price = self.data.close[0]
        high = self.data.high[0]
        low = self.data.low[0]

        if self.position:
            self.holding_bar += 1

            if self.holding_bar >= self.params.max_holding_bar:
                self.order = self.close()
                return

            if self.current_signal == 1:
                if high >= self.tp_level or low <= self.sl_level:
                    self.order = self.close()
                    return

            elif self.current_signal == -1:
                if low <= self.tp_level or high >= self.sl_level:
                    self.order = self.close()
                    return

            if signal != self.current_signal and signal != 0 and self.holding_bar >= self.params.min_holding_bar:
                self.order = self.close()
                self.current_signal = 0
                self.current_size = 0
                return

        if not self.position and signal != 0:
            equity = self.broker.getvalue()
            size = abs(position_size) * equity / price
            signal *= np.sign(position_size)
            self.current_size = size * np.sign(signal)

            if signal == 1:
                self.order = self.buy(size=size)
                self.current_signal = 1
                self.tp_level = price * (1 + vol * self.params.tp_multiplier)
                self.sl_level = price * (1 - vol * self.params.sl_multiplier)

            elif signal == -1:
                self.order = self.sell(size=size)
                self.current_signal = -1
                self.tp_level = price * (1 - vol * self.params.tp_multiplier)
                self.sl_level = price * (1 + vol * self.params.sl_multiplier)


class BacktestEngine:

    def __init__(self, initial_cash=10_000_000.0, commission=0.0, tp_multiplier=2.0, sl_multiplier=2.0, min_holding_bar=2, max_holding_bar=10):
        self.initial_cash = initial_cash
        self.commission = commission
        self.tp_multiplier = tp_multiplier
        self.sl_multiplier = sl_multiplier
        self.min_holding_bar = min_holding_bar
        self.max_holding_bar = max_holding_bar

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
            max_holding_bar=self.max_holding_bar
        )

        cerebro.broker.setcash(self.initial_cash)
        cerebro.broker.setcommission(commission=self.commission)

        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
        cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
        cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='timereturn')

        results = cerebro.run()
        strat = results[0]
        final_value = cerebro.broker.getvalue()

        trade_results = pd.DataFrame(strat.trade_results) if strat.trade_results else pd.DataFrame()

        timereturn = strat.analyzers.timereturn.get_analysis()
        equity_curve = pd.Series(timereturn).sort_index()
        equity_curve = (1 + equity_curve).cumprod() * self.initial_cash
        equity_curve = equity_curve.reindex(df.index).fillna(method='ffill').fillna(self.initial_cash)

        buy_hold_returns = df['close'].pct_change().fillna(0)
        buy_hold_equity = (1 + buy_hold_returns).cumprod() * self.initial_cash
        buy_hold_equity = buy_hold_equity.reindex(df.index).fillna(method='ffill').fillna(self.initial_cash)

        sig_hist = pd.Series(strat.signal_history).sort_index()
        sig_hist = sig_hist.reindex(df.index, fill_value=0)
        
        size_hist = pd.Series(strat.size_history).sort_index()
        size_hist = size_hist.reindex(df.index, fill_value=0)

        metrics = {
            'initial_value': self.initial_cash,
            'final_value': final_value,
            'buy_hold_final_value': buy_hold_equity.iloc[-1],
            'total_return': (final_value - self.initial_cash) / self.initial_cash,
            'buy_hold_return': (buy_hold_equity.iloc[-1] - self.initial_cash) / self.initial_cash,
            'sharpe_ratio': strat.analyzers.sharpe.get_analysis().get('sharperatio', 0),
            'max_drawdown': strat.analyzers.drawdown.get_analysis().get('max', {}).get('drawdown', 0),
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
