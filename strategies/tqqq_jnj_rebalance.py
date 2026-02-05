#!/usr/bin/env python3
"""
TQQQ-JNJ Rebalancing Strategy Backtest

Strategy:
1. Buy TQQQ and JNJ with $100k each initially
2. When TQQQ goes up N% (default 5%), sell N/2% (2.5%) of current TQQQ position to buy JNJ
3. If TQQQ drops 10% from recent high, pause trading for 24 hours
4. If it drops another 10% from that point, pause 24 hours again
5. Continue pausing until drop slows for 24 hours, then rebalance
"""

import backtrader as bt
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, List, Dict
import matplotlib.pyplot as plt


class TQQQJNJRebalanceStrategy(bt.Strategy):
    """
    Rebalancing strategy between TQQQ and JNJ with pause logic during drawdowns
    """
    
    params = (
        ('rebalance_threshold', 0.05),  # N% - default 5%
        ('rebalance_ratio', 0.5),       # N/2% - default 2.5%
        ('drop_threshold', 0.5),       # 10% drop threshold
        ('pause_hours', 24),            # Hours to pause after drop
        ('initial_tqqq', 100000),       # Initial TQQQ allocation
        ('initial_jnj', 100000),        # Initial JNJ allocation
        ('printlog', True),
    )
    
    def __init__(self):
        # Data feeds
        self.tqqq = self.datas[0]
        self.jnj = self.datas[1]
        
        # State tracking
        self.tqqq_high = None  # Recent high price for TQQQ
        self.paused_until = None  # Timestamp when pause ends
        self.last_rebalance_price = None  # TQQQ price at last rebalance
        self.last_drop_check = None  # Last time we checked for drops
        self.drop_check_price = None  # Price at last drop check
        self.initial_purchases_done = False  # Track if initial purchases are done
        
        # Position tracking
        self.tqqq_value = 0
        self.jnj_value = 0
        self.trades_log = []  # Log of all trades/rebalances
        
        # Order tracking
        self.order = None
        self.tqqq_order = None
        self.jnj_order = None
        
    def log(self, txt, dt=None):
        """Logging function"""
        if self.params.printlog:
            dt = dt or self.datas[0].datetime.date(0)
            print(f'{dt.isoformat()}: {txt}')
    
    def notify_order(self, order):
        """Notification of order status"""
        if order.status in [order.Submitted, order.Accepted]:
            return
        
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'BUY EXECUTED: {order.data._name} - Price: {order.executed.price:.2f}, '
                        f'Size: {order.executed.size:.2f}, Cost: {order.executed.value:.2f}')
            elif order.issell():
                self.log(f'SELL EXECUTED: {order.data._name} - Price: {order.executed.price:.2f}, '
                        f'Size: {order.executed.size:.2f}, Cost: {order.executed.value:.2f}')
        
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log(f'Order {order.status} for {order.data._name}')
        
        # Reset order tracking
        if order == self.tqqq_order:
            self.tqqq_order = None
        if order == self.jnj_order:
            self.jnj_order = None
        if order == self.order:
            self.order = None
    
    def start(self):
        """Called when strategy starts"""
        self.log('=' * 50)
        self.log('Strategy Starting')
        self.log(f'Initial TQQQ: ${self.params.initial_tqqq:,.2f}')
        self.log(f'Initial JNJ: ${self.params.initial_jnj:,.2f}')
        self.log(f'Rebalance Threshold: {self.params.rebalance_threshold*100:.1f}%')
        self.log(f'Rebalance Ratio: {self.params.rebalance_ratio*100:.1f}%')
        self.log(f'Drop Threshold: {self.params.drop_threshold*100:.1f}%')
        self.log(f'Pause Hours: {self.params.pause_hours}')
        self.log('=' * 50)
    
    def nextstart(self):
        """Called once when strategy starts - make initial purchases"""
        # Make initial purchases
        tqqq_price = self.tqqq.close[0]
        jnj_price = self.jnj.close[0]
        current_date = self.datas[0].datetime.date(0)
        
        # Calculate shares to buy
        tqqq_shares = int(self.params.initial_tqqq / tqqq_price)
        jnj_shares = int(self.params.initial_jnj / jnj_price)
        
        # Buy TQQQ
        if tqqq_shares > 0:
            self.tqqq_order = self.buy(data=self.tqqq, size=tqqq_shares)
            self.log(f'Initial purchase: {tqqq_shares} TQQQ shares at ${tqqq_price:.2f}')
        
        # Buy JNJ
        if jnj_shares > 0:
            self.jnj_order = self.buy(data=self.jnj, size=jnj_shares)
            self.log(f'Initial purchase: {jnj_shares} JNJ shares at ${jnj_price:.2f}')
        
        # Set initial state
        self.tqqq_high = tqqq_price
        self.last_rebalance_price = tqqq_price
        self.initial_purchases_done = True
        
        self.trades_log.append({
            'date': current_date,
            'type': 'initial_purchase',
            'tqqq_price': tqqq_price,
            'jnj_price': jnj_price,
            'tqqq_shares': tqqq_shares,
            'jnj_shares': jnj_shares,
            'tqqq_value': tqqq_shares * tqqq_price,
            'jnj_value': jnj_shares * jnj_price
        })
    
    def next(self):
        """Called for each bar"""
        # Skip if we don't have enough data or initial purchases not done
        if len(self.tqqq) < 2 or len(self.jnj) < 2 or not self.initial_purchases_done:
            return
        
        # Skip if we have pending orders
        if self.tqqq_order or self.jnj_order:
            return
        
        # Get current prices
        tqqq_price = self.tqqq.close[0]
        jnj_price = self.jnj.close[0]
        current_date = self.datas[0].datetime.date(0)
        
        # Update recent high for TQQQ (only when not paused)
        if self.paused_until is None:
            if self.tqqq_high is None:
                self.tqqq_high = tqqq_price
            else:
                self.tqqq_high = max(self.tqqq_high, tqqq_price)
        
        # Check if we're paused
        if self.paused_until is not None:
            # Check if pause period has ended
            pause_end_date = self.paused_until.date() if isinstance(self.paused_until, datetime) else self.paused_until
            if current_date >= pause_end_date:
                # Pause period ended, check if drop has slowed
                if self.drop_check_price is not None and tqqq_price > 0:
                    drop_since_check = (self.drop_check_price - tqqq_price) / self.drop_check_price
                    if drop_since_check < self.params.drop_threshold:
                        # Drop has slowed (less than 10% further drop), resume
                        self.log(f'Pause ended. Drop has slowed ({drop_since_check*100:.2f}% since pause start). Resuming trading.')
                        self.paused_until = None
                        self.last_drop_check = None
                        self.drop_check_price = None
                        self.last_rebalance_price = tqqq_price
                    else:
                        # Still dropping, extend pause
                        pause_duration = timedelta(hours=self.params.pause_hours)
                        self.paused_until = current_date + pause_duration
                        self.last_drop_check = current_date
                        self.log(f'Drop continues ({drop_since_check*100:.2f}% since pause start). Extending pause until {self.paused_until}.')
                        return
                else:
                    # Resume if we can't check
                    self.log(f'Pause ended. Resuming trading.')
                    self.paused_until = None
                    self.last_drop_check = None
                    self.drop_check_price = None
                    self.last_rebalance_price = tqqq_price
            else:
                # Still paused, check for further drops
                if self.drop_check_price is not None and tqqq_price > 0:
                    drop_since_check = (self.drop_check_price - tqqq_price) / self.drop_check_price
                    if drop_since_check >= self.params.drop_threshold:
                        # Another 10% drop, extend pause
                        pause_duration = timedelta(hours=self.params.pause_hours)
                        self.paused_until = current_date + pause_duration
                        self.last_drop_check = current_date
                        self.drop_check_price = tqqq_price
                        self.log(f'TQQQ dropped another {drop_since_check*100:.2f}% from pause start ${self.drop_check_price:.2f}. Extending pause until {self.paused_until}.')
                return
        
        # Check for initial drop (only when not paused)
        if self.paused_until is None and self.tqqq_high is not None and tqqq_price > 0:
            drop_from_high = (self.tqqq_high - tqqq_price) / self.tqqq_high
            
            if drop_from_high >= self.params.drop_threshold:
                # First drop detected, start pause
                pause_duration = timedelta(hours=self.params.pause_hours)
                self.paused_until = current_date + pause_duration
                self.last_drop_check = current_date
                self.drop_check_price = tqqq_price
                self.log(f'TQQQ dropped {drop_from_high*100:.2f}% from high ${self.tqqq_high:.2f}. Pausing until {self.paused_until}.')
                return
        
        # Check for rebalancing opportunity (TQQQ gain)
        if self.last_rebalance_price is not None and tqqq_price > 0:
            gain_from_rebalance = (tqqq_price - self.last_rebalance_price) / self.last_rebalance_price
            
            if gain_from_rebalance >= self.params.rebalance_threshold:
                # Time to rebalance
                self._rebalance_on_gain(tqqq_price, jnj_price, current_date, gain_from_rebalance)
                self.last_rebalance_price = tqqq_price
    
    def _rebalance_on_gain(self, tqqq_price: float, jnj_price: float, date, gain_pct: float):
        """Rebalance when TQQQ gains: sell N/2% of current TQQQ position value, buy JNJ"""
        # Calculate current TQQQ position value
        tqqq_position = self.broker.getposition(self.tqqq)
        tqqq_value = tqqq_position.size * tqqq_price
        
        if tqqq_value <= 0 or tqqq_position.size <= 0:
            return
        
        # Calculate amount to sell: N/2% of current TQQQ position value
        # When TQQQ goes up N%, we sell N/2% of the current position value
        # rebalance_ratio is 0.5, so N/2% = 0.5 * N% = 0.5 * 0.05 = 0.025 (2.5%)
        sell_ratio = self.params.rebalance_ratio * self.params.rebalance_threshold
        sell_value = tqqq_value * sell_ratio
        
        # Calculate shares to sell
        shares_to_sell = int(sell_value / tqqq_price)
        
        # Don't sell more than we have
        shares_to_sell = min(shares_to_sell, tqqq_position.size)
        
        if shares_to_sell > 0:
            # Sell TQQQ
            self.log(f'Rebalancing: TQQQ gained {gain_pct*100:.2f}%. Selling {shares_to_sell} TQQQ shares (${shares_to_sell * tqqq_price:.2f})')
            self.tqqq_order = self.sell(data=self.tqqq, size=shares_to_sell)
            
            # Calculate JNJ shares to buy (approximate proceeds after commission)
            # Commission is typically small, so we'll use the sell value directly
            # Backtrader will handle commission automatically
            jnj_shares_to_buy = int(sell_value / jnj_price)
            
            if jnj_shares_to_buy > 0:
                # Buy JNJ
                self.log(f'Buying {jnj_shares_to_buy} JNJ shares with proceeds (${sell_value:.2f})')
                self.jnj_order = self.buy(data=self.jnj, size=jnj_shares_to_buy)
            
            # Log the trade
            self.trades_log.append({
                'date': date,
                'type': 'rebalance_gain',
                'tqqq_price': tqqq_price,
                'jnj_price': jnj_price,
                'tqqq_shares_sold': shares_to_sell,
                'jnj_shares_bought': jnj_shares_to_buy,
                'value': shares_to_sell * tqqq_price,
                'gain_pct': gain_pct * 100
            })
    
    def _rebalance_after_pause(self, tqqq_price: float, jnj_price: float, date):
        """Rebalance after pause period ends"""
        # Log that we're resuming after pause
        self.log(f'Resuming after pause. TQQQ: ${tqqq_price:.2f}, JNJ: ${jnj_price:.2f}')
        # Update rebalance price to current price
        self.last_rebalance_price = tqqq_price
        
        self.trades_log.append({
            'date': date,
            'type': 'resume_after_pause',
            'tqqq_price': tqqq_price,
            'jnj_price': jnj_price,
            'tqqq_shares_sold': 0,
            'jnj_shares_bought': 0,
            'value': 0
        })
    
    def stop(self):
        """Called when strategy stops"""
        # Calculate final values
        tqqq_position = self.broker.getposition(self.tqqq)
        jnj_position = self.broker.getposition(self.jnj)
        
        final_tqqq_value = tqqq_position.size * self.tqqq.close[0]
        final_jnj_value = jnj_position.size * self.jnj.close[0]
        total_value = self.broker.getvalue()
        
        self.log('=' * 50)
        self.log('Strategy Stopped')
        self.log(f'Final TQQQ Value: ${final_tqqq_value:,.2f}')
        self.log(f'Final JNJ Value: ${final_jnj_value:,.2f}')
        self.log(f'Total Portfolio Value: ${total_value:,.2f}')
        self.log(f'Total Return: {(total_value - (self.params.initial_tqqq + self.params.initial_jnj)) / (self.params.initial_tqqq + self.params.initial_jnj) * 100:.2f}%')
        self.log(f'Number of Rebalances: {len(self.trades_log)}')
        self.log('=' * 50)


def fetch_data(tickers: List[str], start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
    """Fetch historical data for tickers using yfinance"""
    data = {}
    for ticker in tickers:
        print(f"Downloading data for {ticker}...")
        df = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)
        if df.empty:
            raise ValueError(f"No data found for {ticker}")
        
        # Flatten multi-index columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        # Ensure column names are standard
        df.columns = [col.capitalize() for col in df.columns]
        
        # Ensure we have the required columns
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns for {ticker}: {missing}")
        
        print(f"  {ticker}: {len(df)} days of data from {df.index[0].date()} to {df.index[-1].date()}")
        data[ticker] = df
    return data


def create_backtrader_feeds(data: Dict[str, pd.DataFrame]) -> List[bt.feeds.PandasData]:
    """Create backtrader data feeds from pandas DataFrames"""
    feeds = []
    for ticker, df in data.items():
        # Ensure index is datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
        # Make a clean copy with proper columns
        clean_df = pd.DataFrame({
            'open': df['Open'],
            'high': df['High'],
            'low': df['Low'],
            'close': df['Close'],
            'volume': df['Volume'],
        }, index=df.index)
        
        # Create feed with lowercase column names
        feed = bt.feeds.PandasData(
            dataname=clean_df,
            datetime=None,  # Use index
            open='open',
            high='high',
            low='low',
            close='close',
            volume='volume',
            openinterest=-1
        )
        feed._name = ticker
        feeds.append(feed)
    return feeds


def run_backtest(
    start_date: str = '2020-01-01',
    end_date: str = None,
    rebalance_threshold: float = 0.05,
    rebalance_ratio: float = 0.5,
    drop_threshold: float = 0.10,
    pause_hours: int = 24,
    initial_cash: float = 200000,
    initial_tqqq: float = 100000,
    initial_jnj: float = 100000,
    commission: float = 0.001,  # 0.1%
    printlog: bool = True
):
    """
    Run the TQQQ-JNJ rebalancing strategy backtest
    
    Args:
        start_date: Start date for backtest (YYYY-MM-DD)
        end_date: End date for backtest (YYYY-MM-DD), None for today
        rebalance_threshold: Percentage gain to trigger rebalance (default 0.05 = 5%)
        rebalance_ratio: Ratio of position to rebalance (default 0.5 = N/2%)
        drop_threshold: Percentage drop to trigger pause (default 0.10 = 10%)
        pause_hours: Hours to pause after drop (default 24)
        initial_cash: Initial cash (default 200000)
        initial_tqqq: Initial TQQQ allocation (default 100000)
        initial_jnj: Initial JNJ allocation (default 100000)
        commission: Commission rate (default 0.001 = 0.1%)
        printlog: Print log messages (default True)
    """
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    print("=" * 70)
    print("TQQQ-JNJ Rebalancing Strategy Backtest")
    print("=" * 70)
    print(f"Period: {start_date} to {end_date}")
    print(f"Initial Capital: ${initial_cash:,.2f}")
    print(f"  - TQQQ: ${initial_tqqq:,.2f}")
    print(f"  - JNJ: ${initial_jnj:,.2f}")
    print(f"Rebalance Threshold: {rebalance_threshold*100:.1f}%")
    print(f"Rebalance Ratio: {rebalance_ratio*100:.1f}%")
    print(f"Drop Threshold: {drop_threshold*100:.1f}%")
    print(f"Pause Hours: {pause_hours}")
    print("=" * 70)
    
    # Fetch data
    print("\nFetching market data...")
    tickers = ['TQQQ', 'JNJ']
    data = fetch_data(tickers, start_date, end_date)
    
    # Create Cerebro engine
    cerebro = bt.Cerebro()
    
    # Add data feeds
    feeds = create_backtrader_feeds(data)
    for feed in feeds:
        cerebro.adddata(feed)
    
    # Add strategy
    cerebro.addstrategy(
        TQQQJNJRebalanceStrategy,
        rebalance_threshold=rebalance_threshold,
        rebalance_ratio=rebalance_ratio,
        drop_threshold=drop_threshold,
        pause_hours=pause_hours,
        initial_tqqq=initial_tqqq,
        initial_jnj=initial_jnj,
        printlog=printlog
    )
    
    # Set initial cash
    cerebro.broker.setcash(initial_cash)
    
    # Set commission
    cerebro.broker.setcommission(commission=commission)
    
    # Add analyzers
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
    
    # Run backtest
    print("\nRunning backtest...")
    results = cerebro.run()
    strategy = results[0]
    
    # Get analyzers
    sharpe = strategy.analyzers.sharpe.get_analysis()
    drawdown = strategy.analyzers.drawdown.get_analysis()
    returns = strategy.analyzers.returns.get_analysis()
    trades = strategy.analyzers.trades.get_analysis()
    
    # Print results
    print("\n" + "=" * 70)
    print("BACKTEST RESULTS")
    print("=" * 70)
    
    final_value = cerebro.broker.getvalue()
    total_return = (final_value - initial_cash) / initial_cash * 100
    
    print(f"\nPortfolio Performance:")
    print(f"  Initial Value: ${initial_cash:,.2f}")
    print(f"  Final Value: ${final_value:,.2f}")
    print(f"  Total Return: {total_return:.2f}%")
    
    if returns.get('rnorm100', None) is not None:
        print(f"  Annualized Return: {returns['rnorm100']:.2f}%")
    if sharpe.get('sharperatio', None) is not None:
        print(f"  Sharpe Ratio: {sharpe['sharperatio']:.3f}")
    if drawdown.get('max', {}).get('drawdown', None) is not None:
        print(f"  Max Drawdown: {drawdown['max']['drawdown']:.2f}%")
        print(f"  Max Drawdown Period: {drawdown['max']['len']} days")
    
    print(f"\nTrading Statistics:")
    if trades.get('total', {}).get('total', None) is not None:
        total_trades = trades['total']['total']
        print(f"  Total Trades: {total_trades}")
        
        won_trades = trades.get('won', {}).get('total', 0)
        lost_trades = trades.get('lost', {}).get('total', 0)
        
        print(f"  Won: {won_trades}")
        print(f"  Lost: {lost_trades}")
        
        if total_trades > 0 and won_trades > 0:
            win_rate = won_trades / total_trades * 100
            print(f"  Win Rate: {win_rate:.2f}%")
    
    print(f"  Rebalances: {len(strategy.trades_log)}")
    
    # Save trade log
    if strategy.trades_log:
        trades_df = pd.DataFrame(strategy.trades_log)
        output_file = 'data/results/tqqq_jnj_rebalance_trades.csv'
        trades_df.to_csv(output_file, index=False)
        print(f"\nTrade log saved to: {output_file}")
    
    # Plot results
    try:
        print("\nGenerating plots...")
        cerebro.plot(style='candlestick', barup='green', bardown='red', volume=False)
        print("Plots displayed (close window to continue)")
    except Exception as e:
        print(f"Could not generate plots: {e}")
    
    return {
        'strategy': strategy,
        'cerebro': cerebro,
        'final_value': final_value,
        'total_return': total_return,
        'sharpe': sharpe,
        'drawdown': drawdown,
        'returns': returns,
        'trades': trades,
        'trades_log': strategy.trades_log
    }


if __name__ == '__main__':
    # Run backtest with default parameters
    results = run_backtest(
        start_date='2020-01-01',
        end_date='2025-12-01',  # Today
        rebalance_threshold=0.01,  # 5%
        rebalance_ratio=0.5,  # N/2% = 2.5%
        drop_threshold=0.10,  # 10%
        pause_hours=24,
        initial_cash=200000,
        initial_tqqq=100000,
        initial_jnj=100000,
        commission=0.0001,  # 0.1%
        printlog=True
    )

