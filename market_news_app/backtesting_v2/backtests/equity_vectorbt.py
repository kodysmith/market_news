#!/usr/bin/env python3
"""
Equity Backtesting Engine using VectorBT

Vectorized backtesting for equity strategies with regime-aware signals.
"""

import vectorbt as vbt
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import pyarrow.parquet as pq
from prefect import flow, task
from prefect.tasks import task_input_hash
import mlflow
# MLflow integration for experiment tracking
import warnings
warnings.filterwarnings('ignore')

class EquityBacktestEngine:
    """Vectorized equity backtesting engine using VectorBT"""

    def __init__(self, symbol: str, data_path: str = "./data"):
        """
        Initialize the equity backtest engine

        Args:
            symbol: Stock ticker symbol
            data_path: Path to data directory
        """
        self.symbol = symbol
        self.data_path = data_path
        self.price_data = None
        self.features = None

    def load_data(self, start_date: str = "2010-01-01", end_date: str = None) -> bool:
        """
        Load price data and features

        Args:
            start_date: Start date for backtest
            end_date: End date for backtest

        Returns:
            True if data loaded successfully
        """
        print(f"📊 Loading data for {self.symbol}")

        try:
            # Load price data (year-partitioned)
            price_base_path = os.path.join(self.data_path, 'equity', self.symbol.lower())
            if os.path.exists(price_base_path):
                try:
                    # Read partitioned parquet dataset
                    price_df = pd.read_parquet(price_base_path)
                except Exception:
                    # Fallback: try to read individual files
                    import glob
                    parquet_files = glob.glob(os.path.join(price_base_path, '**', '*.parquet'), recursive=True)
                    if parquet_files:
                        price_df = pd.read_parquet(parquet_files[0])
                        for file in parquet_files[1:]:
                            temp_df = pd.read_parquet(file)
                            price_df = pd.concat([price_df, temp_df])
                        price_df = price_df.sort_values('date')
                price_df['date'] = pd.to_datetime(price_df['date'])
                price_df = price_df.set_index('date').sort_index()

                # Filter date range
                if end_date is None:
                    end_date = datetime.now().strftime('%Y-%m-%d')
                price_df = price_df.loc[start_date:end_date]

                # Convert to VectorBT format (OHLCV)
                ohlcv = price_df[['Open', 'High', 'Low', 'Close', 'Volume']]
                ohlcv.columns = ['open', 'high', 'low', 'close', 'volume']

                self.price_data = ohlcv
                print(f"✅ Loaded {len(ohlcv)} price observations")
            else:
                print(f"❌ Price data not found for {self.symbol}")
                return False

            # Load features
            features_path = os.path.join(self.data_path, 'features', self.symbol.lower(), '**', '*.parquet')
            if os.path.exists(os.path.dirname(features_path)):
                features_df = pq.read_table(features_path).to_pandas()
                features_df['date'] = pd.to_datetime(features_df['date'])
                features_df = features_df.set_index('date').sort_index()

                # Filter date range
                features_df = features_df.loc[start_date:end_date]

                self.features = features_df
                print(f"✅ Loaded {len(features_df)} feature observations")
            else:
                print(f"⚠️ Features not found for {self.symbol}, using price data only")
                self.features = pd.DataFrame(index=self.price_data.index)

            return True

        except Exception as e:
            print(f"❌ Failed to load data: {e}")
            return False

    def generate_signals(self, strategy_config: Dict[str, Any]) -> pd.DataFrame:
        """
        Generate trading signals based on strategy configuration

        Args:
            strategy_config: Dictionary with strategy parameters

        Returns:
            DataFrame with signals
        """
        print("🎯 Generating trading signals")

        signals = pd.DataFrame(index=self.price_data.index)

        # Momentum signals
        if 'momentum' in strategy_config:
            momentum_config = strategy_config['momentum']

            # Simple moving average crossover
            if 'sma_crossover' in momentum_config:
                fast_period = momentum_config['sma_crossover'].get('fast', 20)
                slow_period = momentum_config['sma_crossover'].get('slow', 50)

                fast_sma = self.price_data['close'].rolling(fast_period).mean()
                slow_sma = self.price_data['close'].rolling(slow_period).mean()

                signals['sma_crossover'] = (fast_sma > slow_sma).astype(int)

            # RSI signals
            if 'rsi' in momentum_config:
                rsi_period = momentum_config['rsi'].get('period', 14)
                rsi_overbought = momentum_config['rsi'].get('overbought', 70)
                rsi_oversold = momentum_config['rsi'].get('oversold', 30)

                def calculate_rsi(data, period):
                    delta = data.diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
                    rs = gain / loss
                    return 100 - (100 / (1 + rs))

                rsi = calculate_rsi(self.price_data['close'], rsi_period)
                signals['rsi_oversold'] = (rsi < rsi_oversold).astype(int)
                signals['rsi_overbought'] = (rsi > rsi_overbought).astype(int)

        # Mean reversion signals
        if 'mean_reversion' in strategy_config:
            mr_config = strategy_config['mean_reversion']

            if 'bollinger_bands' in mr_config:
                bb_period = mr_config['bollinger_bands'].get('period', 20)
                bb_std = mr_config['bollinger_bands'].get('std', 2)

                sma = self.price_data['close'].rolling(bb_period).mean()
                std = self.price_data['close'].rolling(bb_period).std()
                upper_band = sma + (std * bb_std)
                lower_band = sma - (std * bb_std)

                signals['bb_lower'] = (self.price_data['close'] < lower_band).astype(int)
                signals['bb_upper'] = (self.price_data['close'] > upper_band).astype(int)

        # Regime-aware signals
        if 'regime_aware' in strategy_config and not self.features.empty:
            regime_config = strategy_config['regime_aware']

            # Risk regime filter
            if 'risk_regime_filter' in regime_config:
                max_risk_level = regime_config['risk_regime_filter'].get('max_level', 2)
                signals['risk_filter'] = (self.features.get('risk_regime', 1) <= max_risk_level).astype(int)

            # Trend regime filter
            if 'trend_filter' in regime_config:
                signals['trend_filter'] = self.features.get('trend_up', 1).astype(int)

            # Volatility regime adjustment
            if 'vol_regime_adjust' in regime_config:
                vol_regime = self.features.get('vol_regime', 1)
                # Reduce position size in high vol regimes
                signals['vol_adjustment'] = 1.0 / (1.0 + vol_regime)

        # Combine signals into final entry/exit signals
        signals['long_entry'] = 0
        signals['long_exit'] = 0

        # Simple combination logic (can be made more sophisticated)
        if 'momentum' in strategy_config:
            # Enter long on bullish momentum signals
            momentum_signals = (
                signals.get('sma_crossover', 0) |
                signals.get('rsi_oversold', 0)
            )
            signals['long_entry'] = momentum_signals.astype(int)

        if 'mean_reversion' in strategy_config:
            # Enter long on oversold signals
            mr_signals = signals.get('bb_lower', 0)
            signals['long_entry'] = mr_signals.astype(int)

        # Apply regime filters
        if 'regime_aware' in strategy_config:
            regime_filter = (
                signals.get('risk_filter', 1) &
                signals.get('trend_filter', 1)
            )
            signals['long_entry'] = (signals['long_entry'] & regime_filter).astype(int)

        # Generate exit signals (opposite of entry or time-based)
        signals['long_exit'] = (signals['long_entry'].shift(1) == 1).astype(int)

        print(f"✅ Generated signals: {signals['long_entry'].sum()} entries, {signals['long_exit'].sum()} exits")
        return signals

    def run_backtest(self, strategy_config: Dict[str, Any], **kwargs) -> vbt.Portfolio:
        """
        Run vectorized backtest

        Args:
            strategy_config: Strategy configuration
            **kwargs: Additional VectorBT parameters

        Returns:
            VectorBT Portfolio object
        """
        print(f"🚀 Running backtest for {self.symbol}")

        # Generate signals
        signals = self.generate_signals(strategy_config)

        # Create VectorBT portfolio
        entries = signals['long_entry'] == 1
        exits = signals['long_exit'] == 1

        # Default portfolio settings
        portfolio_kwargs = {
            'freq': 'D',
            'fees': 0.001,  # 10 bps round trip
            'slippage': 0.001,  # 10 bps slippage
        }
        portfolio_kwargs.update(kwargs)

        # Run backtest
        portfolio = vbt.Portfolio.from_signals(
            close=self.price_data['close'],
            entries=entries,
            exits=exits,
            **portfolio_kwargs
        )

        print(f"✅ Backtest completed: {len(portfolio.trades)} trades")
        return portfolio

    def analyze_results(self, portfolio: vbt.Portfolio) -> Dict[str, Any]:
        """
        Analyze backtest results

        Args:
            portfolio: VectorBT Portfolio object

        Returns:
            Dictionary with performance metrics
        """
        print("📊 Analyzing backtest results")

        # Basic metrics
        total_return = portfolio.total_return()
        sharpe_ratio = portfolio.sharpe_ratio()
        max_drawdown = portfolio.max_drawdown()
        win_rate = portfolio.trades.win_rate()

        # Advanced metrics
        calmar_ratio = total_return / abs(max_drawdown) if max_drawdown != 0 else 0
        sortino_ratio = portfolio.sortino_ratio()

        # Risk metrics
        returns_series = portfolio.returns()
        volatility = returns_series.std() * np.sqrt(252)

        # Calculate VaR manually since VectorBT API might vary
        try:
            var_95 = portfolio.value_at_risk()
            cvar_95 = portfolio.conditional_value_at_risk()
        except:
            # Fallback: calculate manually
            var_95 = np.percentile(returns_series, 5)
            cvar_95 = returns_series[returns_series <= var_95].mean()

        # Trade analysis - handle VectorBT API variations
        try:
            avg_trade = portfolio.trades.PnL.mean()
            winning_trades = portfolio.trades.PnL[portfolio.trades.PnL > 0]
            losing_trades = portfolio.trades.PnL[portfolio.trades.PnL < 0]
            avg_winning_trade = winning_trades.mean() if len(winning_trades) > 0 else 0
            avg_losing_trade = losing_trades.mean() if len(losing_trades) > 0 else 0
        except:
            # Fallback for different VectorBT versions
            try:
                trades_df = portfolio.trades.records_readable
                avg_trade = trades_df['PnL'].mean()
                avg_winning_trade = trades_df[trades_df['PnL'] > 0]['PnL'].mean()
                avg_losing_trade = trades_df[trades_df['PnL'] < 0]['PnL'].mean()
            except:
                # Basic fallback
                avg_trade = 0
                avg_winning_trade = 0
                avg_losing_trade = 0

        results = {
            'total_return': float(total_return),
            'sharpe_ratio': float(sharpe_ratio),
            'sortino_ratio': float(sortino_ratio),
            'calmar_ratio': float(calmar_ratio),
            'max_drawdown': float(max_drawdown),
            'volatility': float(volatility),
            'var_95': float(var_95),
            'cvar_95': float(cvar_95),
            'win_rate': float(win_rate),
            'total_trades': len(portfolio.trades),
            'avg_trade_pnl': float(avg_trade) if not np.isnan(avg_trade) else 0,
            'avg_winning_trade': float(avg_winning_trade) if not np.isnan(avg_winning_trade) else 0,
            'avg_losing_trade': float(avg_losing_trade) if not np.isnan(avg_losing_trade) else 0,
        }

        print(f"📈 Total Return: {results['total_return']:.2%}")
        print(f"📊 Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        print(f"📉 Max Drawdown: {results['max_drawdown']:.2%}")
        print(f"🎯 Win Rate: {results['win_rate']:.2%}")

        return results

@task(cache_key_fn=task_input_hash, cache_expiration=timedelta(minutes=30))
def run_equity_backtest(
    symbol: str,
    strategy_config: Dict[str, Any],
    start_date: str = "2015-01-01",
    end_date: str = None,
    data_path: str = "./data"
) -> Tuple[vbt.Portfolio, Dict[str, Any]]:
    """
    Run equity backtest for a symbol

    Args:
        symbol: Stock ticker symbol
        strategy_config: Strategy configuration
        start_date: Backtest start date
        end_date: Backtest end date
        data_path: Path to data directory

    Returns:
        Tuple of (portfolio, results)
    """
    engine = EquityBacktestEngine(symbol, data_path)

    # Load data
    if not engine.load_data(start_date, end_date):
        raise ValueError(f"Failed to load data for {symbol}")

    # Run backtest
    portfolio = engine.run_backtest(strategy_config)

    # Analyze results
    results = engine.analyze_results(portfolio)

    return portfolio, results

@flow(name="equity-backtest-flow")
def equity_backtest_flow(
    symbols: List[str],
    strategy_configs: List[Dict[str, Any]],
    start_date: str = "2015-01-01",
    end_date: str = None,
    data_path: str = "./data",
    experiment_name: str = "equity_backtest"
) -> Dict[str, Any]:
    """
    Run equity backtests for multiple symbols and strategies

    Args:
        symbols: List of stock ticker symbols
        strategy_configs: List of strategy configurations
        start_date: Backtest start date
        end_date: Backtest end date
        data_path: Path to data directory
        experiment_name: MLflow experiment name

    Returns:
        Dictionary with backtest results
    """
    print(f"🚀 Starting equity backtest flow for {len(symbols)} symbols, {len(strategy_configs)} strategies")

    # Start MLflow experiment
    mlflow.set_experiment(experiment_name)

    all_results = {}

    for symbol in symbols:
        symbol_results = {}

        for i, strategy_config in enumerate(strategy_configs):
            strategy_name = strategy_config.get('name', f'strategy_{i+1}')

            try:
                with mlflow.start_run(run_name=f"{symbol}_{strategy_name}"):
                    # Log parameters
                    mlflow.log_params({
                        'symbol': symbol,
                        'strategy': strategy_name,
                        'start_date': start_date,
                        'end_date': end_date,
                        **strategy_config
                    })

                    # Run backtest
                    portfolio, results = run_equity_backtest(
                        symbol, strategy_config, start_date, end_date, data_path
                    )

                    # Log metrics
                    mlflow.log_metrics(results)

                    # Log portfolio as artifacts
                    portfolio_stats = portfolio.stats()
                    portfolio_stats.to_csv("portfolio_stats.csv")
                    mlflow.log_artifact("portfolio_stats.csv")

                    # Log equity curve
                    equity_df = pd.DataFrame({
                        'equity': portfolio.value().values,
                        'returns': portfolio.returns().values,
                        'date': portfolio.value().index
                    })
                    equity_df.to_csv("equity_curve.csv", index=False)
                    mlflow.log_artifact("equity_curve.csv")

                    symbol_results[strategy_name] = results
                    print(f"✅ Completed {symbol} {strategy_name}: {results['total_return']:.2%}")

            except Exception as e:
                print(f"❌ Failed {symbol} {strategy_name}: {e}")
                symbol_results[strategy_name] = {'error': str(e)}

        all_results[symbol] = symbol_results

    print("🎉 Equity backtest flow completed")
    return all_results

if __name__ == "__main__":
    # Example usage
    symbols = ["SPY", "QQQ"]

    # Define strategy configurations
    strategies = [
        {
            'name': 'momentum_sma',
            'momentum': {
                'sma_crossover': {'fast': 20, 'slow': 50}
            },
            'regime_aware': {
                'trend_filter': True
            }
        },
        {
            'name': 'mean_reversion_bb',
            'mean_reversion': {
                'bollinger_bands': {'period': 20, 'std': 2}
            },
            'regime_aware': {
                'risk_regime_filter': {'max_level': 2}
            }
        }
    ]

    results = equity_backtest_flow(symbols, strategies)
    print("Backtest results:", results)
