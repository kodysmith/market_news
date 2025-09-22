#!/usr/bin/env python3
"""
Options Backtesting Engine using Backtrader

Event-driven backtesting for complex options strategies with position management.
"""

import backtrader as bt
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import pyarrow.parquet as pq
from prefect import flow, task
from prefect.tasks import task_input_hash
import mlflow
import warnings
warnings.filterwarnings('ignore')

# Numba-accelerated Black-Scholes pricing (same as original)
from numba import njit
import numba as nb

@njit(fastmath=True)
def norm_cdf(x):
    """Fast normal CDF approximation"""
    a1, a2, a3, a4, a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
    p = 0.3275911
    y = abs(x)
    t = 1.0 / (1.0 + p * y)
    z = t * (a1 + t * (a2 + t * (a3 + t * (a4 + t * a5))))
    return np.sign(x) * (1.0 - z * np.exp(-y * y)) * 0.5 + 0.5

@njit(fastmath=True)
def bs_put_price(S, K, sigma, T, r=0.02):
    """Black-Scholes put pricing"""
    if T <= 0 or sigma <= 0:
        return 0.01
    d1 = (np.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    put = K * np.exp(-r * T) * norm_cdf(-d2) - S * norm_cdf(-d1)
    return max(put, 0.01)

@njit(fastmath=True)
def bs_call_price(S, K, sigma, T, r=0.02):
    """Black-Scholes call pricing"""
    if T <= 0 or sigma <= 0:
        return 0.01
    d1 = (np.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call = S * norm_cdf(d1) - K * np.exp(-r * T) * norm_cdf(d2)
    return max(call, 0.01)

class OptionsDataFeed(bt.feeds.PandasData):
    """Custom data feed for options data"""
    lines = ('iv', 'delta', 'gamma', 'theta', 'vega', 'rho', 'dte', 'moneyness')

    params = (
        ('iv', 'iv'),
        ('delta', 'delta'),
        ('gamma', 'gamma'),
        ('theta', 'theta'),
        ('vega', 'vega'),
        ('rho', 'rho'),
        ('dte', 'dte'),
        ('moneyness', 'moneyness'),
    )

class OptionsStrategy(bt.Strategy):
    """Base options strategy with common functionality"""

    params = (
        ('risk_regime_max', 2),
        ('hedge_budget_pct', 0.01),
        ('min_hedge_ratio', 0.4),
        ('max_options_per_trade', 10),
    )

    def __init__(self):
        # Track positions
        self.equity_positions = {}
        self.options_positions = {}
        self.hedge_spend = {}
        self.portfolio_values = []

        # Risk metrics
        self.current_risk_regime = 1
        self.current_trend = True

        # Initialize analyzers
        self.add_analyzer(bt.analyzers.Returns, _name='returns')
        self.add_analyzer(bt.analyzers.DrawDown, _name='drawdown')
        self.add_analyzer(bt.analyzers.SharpeRatio, _name='sharpe')
        self.add_analyzer(bt.analyzers.TradeAnalyzer, _name='trades')

    def get_risk_regime(self):
        """Get current risk regime from features"""
        if hasattr(self.datas[0], 'risk_regime'):
            return self.datas[0].risk_regime[0]
        return 1

    def get_trend_signal(self):
        """Get current trend signal"""
        if hasattr(self.datas[0], 'trend_up'):
            return self.datas[0].trend_up[0] > 0
        return True

    def can_hedge(self, cost):
        """Check if hedging is allowed within budget"""
        current_date = self.datas[0].datetime.date(0)
        month_key = (current_date.year, current_date.month)

        # Get portfolio value for budget calculation
        portfolio_value = self.broker.getvalue()

        # Monthly budget
        monthly_budget = portfolio_value * self.params.hedge_budget_pct

        # Current spend this month
        current_spend = self.hedge_spend.get(month_key, 0)

        return (current_spend + cost) <= monthly_budget

    def update_hedge_spend(self, cost):
        """Update hedge spending tracker"""
        current_date = self.datas[0].datetime.date(0)
        month_key = (current_date.year, current_date.month)

        if month_key not in self.hedge_spend:
            self.hedge_spend[month_key] = 0

        self.hedge_spend[month_key] += cost

    def calculate_hedge_ratio(self):
        """Calculate current hedge ratio"""
        if not self.equity_positions:
            return 0.0

        total_equity = sum(pos.size for pos in self.equity_positions.values())
        total_hedged = sum(pos.size * 100 for pos in self.options_positions.values()
                          if hasattr(pos, 'is_protection') and pos.is_protection)

        return total_hedged / max(total_equity, 1) if total_equity > 0 else 0.0

    def buy_equity(self, symbol, size, price=None):
        """Buy equity shares"""
        if price is None:
            price = self.datas[0].close[0]

        cost = size * price
        if cost <= self.broker.getcash():
            order = self.buy(size=size, price=price)
            self.equity_positions[symbol] = order
            return order
        return None

    def sell_equity(self, symbol, size, price=None):
        """Sell equity shares"""
        if price is None:
            price = self.datas[0].close[0]

        if symbol in self.equity_positions:
            order = self.sell(size=size, price=price)
            return order
        return None

class CoveredCallsStrategy(OptionsStrategy):
    """Covered calls strategy with dynamic position sizing"""

    params = (
        ('call_delta_target', 0.15),
        ('call_dte_min', 30),
        ('call_dte_max', 60),
        ('cover_ratio_max', 0.6),
    )

    def next(self):
        current_price = self.datas[0].close[0]
        current_date = self.datas[0].datetime.date(0)

        # Update risk regime
        self.current_risk_regime = self.get_risk_regime()
        self.current_trend = self.get_trend_signal()

        # Buy equity if not fully invested and conditions are good
        if (self.current_trend and
            self.current_risk_regime <= self.params.risk_regime_max and
            len(self.equity_positions) < 5):  # Max 5 positions

            # Calculate position size based on risk regime
            base_size = 100
            risk_multiplier = max(0.5, 2 - self.current_risk_regime)  # Reduce size in high risk
            position_size = int(base_size * risk_multiplier)

            self.buy_equity(f"equity_{current_date}", position_size, current_price)

        # Sell covered calls if we have equity positions
        if self.equity_positions:
            hedge_ratio = self.calculate_hedge_ratio()

            if hedge_ratio < self.params.cover_ratio_max:
                # Find suitable call options
                call_strike = self.find_call_strike(current_price)
                call_premium = self.estimate_call_premium(call_strike, current_price)

                if call_premium > 0:
                    # Calculate how many calls to sell
                    total_equity = sum(pos.size for pos in self.equity_positions.values())
                    max_covered = int(total_equity * self.params.cover_ratio_max / 100)
                    contracts_to_sell = min(max_covered, self.params.max_options_per_trade)

                    if contracts_to_sell > 0:
                        total_premium = contracts_to_sell * call_premium * 100
                        self.sell_options('call', call_strike, contracts_to_sell, total_premium)

    def find_call_strike(self, current_price):
        """Find appropriate call strike based on delta target"""
        # Simplified strike selection - in practice would use options data
        return current_price * 1.05  # 5% OTM

    def estimate_call_premium(self, strike, spot_price):
        """Estimate call premium using Black-Scholes"""
        # Simplified IV estimate - in practice would use market data
        iv = 0.25
        dte = 45  # days
        return bs_call_price(spot_price, strike, iv, dte/365.0)

    def sell_options(self, option_type, strike, contracts, premium):
        """Sell options contracts"""
        if self.can_hedge(premium):
            # Create synthetic options position
            option_pos = bt.Position(self.datas[0], size=-contracts)
            option_pos.strike = strike
            option_pos.option_type = option_type
            option_pos.is_protection = False  # This is income generation

            self.options_positions[f"{option_type}_{strike}"] = option_pos
            self.update_hedge_spend(premium)

            self.broker.add_cash(premium)  # Credit from selling options

class ProtectivePutsStrategy(OptionsStrategy):
    """Protective puts strategy with dynamic hedging"""

    params = (
        ('put_delta_target', -0.20),
        ('put_dte_target', 90),
        ('protection_ratio_min', 0.4),
        ('protection_ratio_max', 0.8),
    )

    def next(self):
        current_price = self.datas[0].close[0]
        current_date = self.datas[0].datetime.date(0)

        # Update risk regime
        self.current_risk_regime = self.get_risk_regime()
        self.current_trend = self.get_trend_signal()

        # Buy equity if conditions are good
        if (self.current_trend and
            self.current_risk_regime <= self.params.risk_regime_max and
            not self.equity_positions):

            position_size = 1000  # Fixed size for simplicity
            self.buy_equity(f"equity_{current_date}", position_size, current_price)

        # Buy protective puts based on risk regime
        if self.equity_positions:
            hedge_ratio = self.calculate_hedge_ratio()

            # Adjust protection ratio based on risk regime
            if self.current_risk_regime <= 1:  # Low risk
                target_ratio = self.params.protection_ratio_min
            else:  # High risk
                target_ratio = self.params.protection_ratio_max

            if hedge_ratio < target_ratio:
                # Find suitable put options
                put_strike = self.find_put_strike(current_price)
                put_premium = self.estimate_put_premium(put_strike, current_price)

                if put_premium > 0:
                    # Calculate contracts needed
                    total_equity = sum(pos.size for pos in self.equity_positions.values())
                    shortfall = int(total_equity * (target_ratio - hedge_ratio) / 100)
                    contracts_needed = min(shortfall, self.params.max_options_per_trade)

                    if contracts_needed > 0:
                        total_cost = contracts_needed * put_premium * 100

                        if self.can_hedge(total_cost):
                            self.buy_options('put', put_strike, contracts_needed, total_cost)

    def find_put_strike(self, current_price):
        """Find appropriate put strike based on delta target"""
        # Simplified - in practice would use options data and delta targeting
        return current_price * 0.95  # 5% OTM

    def estimate_put_premium(self, strike, spot_price):
        """Estimate put premium using Black-Scholes"""
        # Simplified IV estimate
        iv = 0.25
        dte = self.params.put_dte_target
        return bs_put_price(spot_price, strike, iv, dte/365.0)

    def buy_options(self, option_type, strike, contracts, cost):
        """Buy options contracts"""
        if cost <= self.broker.getcash():
            # Create synthetic options position
            option_pos = bt.Position(self.datas[0], size=contracts)
            option_pos.strike = strike
            option_pos.option_type = option_type
            option_pos.is_protection = True

            self.options_positions[f"{option_type}_{strike}"] = option_pos
            self.update_hedge_spend(cost)

            self.broker.add_cash(-cost)  # Debit from buying options

class OptionsBacktestEngine:
    """Options backtesting engine using Backtrader"""

    def __init__(self, symbol: str, data_path: str = "./data"):
        self.symbol = symbol
        self.data_path = data_path
        self.price_data = None
        self.features = None
        self.cerebro = None

    def load_data(self, start_date: str = "2010-01-01", end_date: str = None) -> bool:
        """Load data for backtesting"""
        print(f"📊 Loading options backtest data for {self.symbol}")

        try:
            # Load price data
            price_path = os.path.join(self.data_path, 'equity', self.symbol.lower(), '**', '*.parquet')
            if os.path.exists(os.path.dirname(price_path)):
                price_df = pq.read_table(price_path).to_pandas()
                price_df['date'] = pd.to_datetime(price_df['date'])
                price_df = price_df.set_index('date').sort_index()

                # Filter date range
                if end_date is None:
                    end_date = datetime.now().strftime('%Y-%m-%d')
                price_df = price_df.loc[start_date:end_date]

                self.price_data = price_df
                print(f"✅ Loaded {len(price_df)} price observations")
            else:
                print(f"❌ Price data not found for {self.symbol}")
                return False

            # Load features
            features_path = os.path.join(self.data_path, 'features', self.symbol.lower(), '**', '*.parquet')
            if os.path.exists(os.path.dirname(features_path)):
                features_df = pq.read_table(features_path).to_pandas()
                features_df['date'] = pd.to_datetime(features_df['date'])
                features_df = features_df.set_index('date').sort_index()
                features_df = features_df.loc[start_date:end_date]

                self.features = features_df
                print(f"✅ Loaded {len(features_df)} feature observations")
            else:
                print(f"⚠️ Features not found, using basic price data only")
                self.features = pd.DataFrame(index=self.price_data.index)

            return True

        except Exception as e:
            print(f"❌ Failed to load data: {e}")
            return False

    def prepare_data_feed(self) -> bt.feeds.PandasData:
        """Prepare data feed for Backtrader"""
        # Merge price and features data
        combined_data = self.price_data.copy()

        if not self.features.empty:
            # Add feature columns to price data
            for col in self.features.columns:
                if col not in combined_data.columns:
                    combined_data[col] = self.features[col]

        # Ensure required columns exist
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in combined_data.columns:
                # Create from close if missing
                combined_data[col] = combined_data.get('close', 0)

        # Reset index for Backtrader
        combined_data = combined_data.reset_index()

        # Create data feed
        data_feed = OptionsDataFeed(dataname=combined_data)

        return data_feed

    def run_backtest(self, strategy_class: type, strategy_params: Dict[str, Any] = None) -> bt.Cerebro:
        """
        Run options backtest

        Args:
            strategy_class: Backtrader strategy class
            strategy_params: Strategy parameters

        Returns:
            Backtrader Cerebro instance
        """
        print(f"🚀 Running {strategy_class.__name__} backtest for {self.symbol}")

        # Initialize Cerebro
        self.cerebro = bt.Cerebro()

        # Add strategy
        if strategy_params:
            self.cerebro.addstrategy(strategy_class, **strategy_params)
        else:
            self.cerebro.addstrategy(strategy_class)

        # Add data feed
        data_feed = self.prepare_data_feed()
        self.cerebro.adddata(data_feed)

        # Set broker settings
        self.cerebro.broker.setcash(1000000.0)
        self.cerebro.broker.setcommission(commission=0.001)  # 10 bps

        # Add analyzers
        self.cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
        self.cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
        self.cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
        self.cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
        self.cerebro.addanalyzer(bt.analyzers.PyFolio, _name='pyfolio')

        # Run backtest
        results = self.cerebro.run()

        print(f"✅ Backtest completed for {strategy_class.__name__}")
        return results[0]

    def analyze_results(self, strategy_result) -> Dict[str, Any]:
        """Analyze backtest results"""
        print("📊 Analyzing options backtest results")

        # Extract analyzer results
        returns_analyzer = strategy_result.analyzers.returns.get_analysis()
        drawdown_analyzer = strategy_result.analyzers.drawdown.get_analysis()
        sharpe_analyzer = strategy_result.analyzers.sharpe.get_analysis()
        trades_analyzer = strategy_result.analyzers.trades.get_analysis()

        # Basic metrics
        total_return = returns_analyzer.get('rtot', 0)
        sharpe_ratio = sharpe_analyzer.get('sharperatio', 0)
        max_drawdown = drawdown_analyzer.get('max', {}).get('drawdown', 0)

        # Trade metrics
        total_trades = trades_analyzer.get('total', {}).get('total', 0)
        winning_trades = trades_analyzer.get('won', {}).get('total', 0)
        losing_trades = trades_analyzer.get('lost', {}).get('total', 0)

        win_rate = winning_trades / max(total_trades, 1)

        # PnL metrics
        pnl_net = trades_analyzer.get('pnl', {}).get('net', {}).get('total', 0)
        avg_win = trades_analyzer.get('won', {}).get('pnl', {}).get('average', 0)
        avg_loss = trades_analyzer.get('lost', {}).get('pnl', {}).get('average', 0)

        results = {
            'total_return': float(total_return),
            'sharpe_ratio': float(sharpe_ratio),
            'max_drawdown': float(max_drawdown),
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': float(win_rate),
            'pnl_net': float(pnl_net),
            'avg_win': float(avg_win),
            'avg_loss': float(avg_loss),
        }

        print(f"📈 Total Return: {results['total_return']:.2%}")
        print(f"📊 Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        print(f"📉 Max Drawdown: {results['max_drawdown']:.2%}")
        print(f"🎯 Win Rate: {results['win_rate']:.2%}")

        return results

@task(cache_key_fn=task_input_hash, cache_expiration=timedelta(minutes=30))
def run_options_backtest(
    symbol: str,
    strategy_class: str,
    strategy_params: Dict[str, Any] = None,
    start_date: str = "2015-01-01",
    end_date: str = None,
    data_path: str = "./data"
) -> Tuple[Any, Dict[str, Any]]:
    """
    Run options backtest for a symbol

    Args:
        symbol: Stock ticker symbol
        strategy_class: Name of strategy class ('CoveredCallsStrategy' or 'ProtectivePutsStrategy')
        strategy_params: Strategy parameters
        start_date: Backtest start date
        end_date: Backtest end date
        data_path: Path to data directory

    Returns:
        Tuple of (strategy_result, results)
    """
    # Map strategy names to classes
    strategy_map = {
        'CoveredCallsStrategy': CoveredCallsStrategy,
        'ProtectivePutsStrategy': ProtectivePutsStrategy,
    }

    if strategy_class not in strategy_map:
        raise ValueError(f"Unknown strategy: {strategy_class}")

    strategy_cls = strategy_map[strategy_class]

    engine = OptionsBacktestEngine(symbol, data_path)

    # Load data
    if not engine.load_data(start_date, end_date):
        raise ValueError(f"Failed to load data for {symbol}")

    # Run backtest
    strategy_result = engine.run_backtest(strategy_cls, strategy_params)

    # Analyze results
    results = engine.analyze_results(strategy_result)

    return strategy_result, results

@flow(name="options-backtest-flow")
def options_backtest_flow(
    symbols: List[str],
    strategies: List[Dict[str, Any]],
    start_date: str = "2015-01-01",
    end_date: str = None,
    data_path: str = "./data",
    experiment_name: str = "options_backtest"
) -> Dict[str, Any]:
    """
    Run options backtests for multiple symbols and strategies

    Args:
        symbols: List of stock ticker symbols
        strategies: List of strategy configurations with 'name' and 'params' keys
        start_date: Backtest start date
        end_date: Backtest end date
        data_path: Path to data directory
        experiment_name: MLflow experiment name

    Returns:
        Dictionary with backtest results
    """
    print(f"🚀 Starting options backtest flow for {len(symbols)} symbols, {len(strategies)} strategies")

    # Start MLflow experiment
    mlflow.set_experiment(experiment_name)

    all_results = {}

    for symbol in symbols:
        symbol_results = {}

        for strategy_config in strategies:
            strategy_name = strategy_config.get('name', 'unknown')
            strategy_params = strategy_config.get('params', {})

            try:
                with mlflow.start_run(run_name=f"{symbol}_{strategy_name}"):
                    # Log parameters
                    mlflow.log_params({
                        'symbol': symbol,
                        'strategy': strategy_name,
                        'start_date': start_date,
                        'end_date': end_date,
                        **strategy_params
                    })

                    # Run backtest
                    strategy_result, results = run_options_backtest(
                        symbol, strategy_name, strategy_params, start_date, end_date, data_path
                    )

                    # Log metrics
                    mlflow.log_metrics(results)

                    symbol_results[strategy_name] = results
                    print(f"✅ Completed {symbol} {strategy_name}: {results['total_return']:.2%}")

            except Exception as e:
                print(f"❌ Failed {symbol} {strategy_name}: {e}")
                symbol_results[strategy_name] = {'error': str(e)}

        all_results[symbol] = symbol_results

    print("🎉 Options backtest flow completed")
    return all_results

if __name__ == "__main__":
    # Example usage
    symbols = ["SPY", "QQQ"]

    # Define strategy configurations
    strategies = [
        {
            'name': 'CoveredCallsStrategy',
            'params': {
                'call_delta_target': 0.15,
                'risk_regime_max': 2,
            }
        },
        {
            'name': 'ProtectivePutsStrategy',
            'params': {
                'put_delta_target': -0.20,
                'risk_regime_max': 2,
            }
        }
    ]

    results = options_backtest_flow(symbols, strategies)
    print("Options backtest results:", results)

