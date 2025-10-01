"""
Vectorized Backtest Engine for AI Quant Trading System

Core backtesting functionality with support for:
- Vectorized strategy execution
- Cost modeling (commissions, slippage, borrowing)
- Options overlays and hedging
- Walk-forward optimization
- Performance analytics
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging
from pathlib import Path
import numba as nb

import sys
from pathlib import Path

# Add QuantEngine root to path for imports
quant_engine_root = Path(__file__).parent.parent.parent
if str(quant_engine_root) not in sys.path:
    sys.path.insert(0, str(quant_engine_root))

from utils.strategy_dsl import StrategySpec, SignalDefinition, SignalType

logger = logging.getLogger(__name__)


class BacktestResult:
    """Container for backtest results"""

    def __init__(self, returns: pd.Series, positions: pd.DataFrame, trades: pd.DataFrame,
                 metrics: Dict[str, float], signals: pd.DataFrame = None):
        self.returns = returns
        self.positions = positions
        self.trades = trades
        self.metrics = metrics
        self.signals = signals

    def __repr__(self):
        return f"BacktestResult(returns={len(self.returns)}, trades={len(self.trades)}, sharpe={self.metrics.get('sharpe', 'N/A'):.2f})"


class VectorizedBacktester:
    """Vectorized backtesting engine with cost modeling"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.commission_bps = config.get('commission_bps', 2.0)
        self.slippage_bps = config.get('slippage_bps', 1.0)
        self.initial_capital = config.get('initial_capital', 1000000)

    def run_backtest(self, strategy_spec: StrategySpec, market_data: Dict[str, pd.DataFrame],
                    signals_df: pd.DataFrame = None) -> BacktestResult:
        """
        Run backtest for a strategy specification

        Args:
            strategy_spec: Validated strategy specification
            market_data: Dictionary of price dataframes keyed by ticker
            signals_df: Pre-computed signals dataframe (optional)

        Returns:
            BacktestResult with returns, positions, trades, and metrics
        """

        # Get price data for universe
        universe_prices = self._prepare_universe_data(strategy_spec.universe, market_data)

        # Generate signals if not provided
        if signals_df is None:
            signals_df = self._generate_signals(strategy_spec, universe_prices)

        # Generate positions
        positions_df = self._generate_positions(strategy_spec, signals_df, universe_prices)

        # Apply sizing and risk management
        positions_df = self._apply_position_sizing(strategy_spec, positions_df, universe_prices)

        # Handle options overlays if specified
        if strategy_spec.overlays:
            positions_df = self._apply_options_overlays(strategy_spec, positions_df, universe_prices)

        # Calculate returns with costs
        returns, trades_df = self._calculate_returns_with_costs(positions_df, universe_prices, strategy_spec)

        # Calculate performance metrics
        metrics = self._calculate_metrics(returns, positions_df, trades_df)

        return BacktestResult(returns, positions_df, trades_df, metrics, signals_df)

    def _prepare_universe_data(self, universe: List[str], market_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Prepare aligned price data for the trading universe"""
        price_dfs = []

        for ticker in universe:
            if ticker not in market_data:
                logger.warning(f"Missing data for {ticker}, skipping")
                continue

            df = market_data[ticker].copy()
            df[f'price_{ticker}'] = df['close']  # Assume OHLCV format
            df[f'returns_{ticker}'] = df[f'price_{ticker}'].pct_change()
            price_dfs.append(df[[f'price_{ticker}', f'returns_{ticker}']])

        if not price_dfs:
            raise ValueError("No valid price data for universe")

        # Align all dataframes on datetime index
        combined = pd.concat(price_dfs, axis=1, join='inner')
        
        # Flatten MultiIndex columns if they exist
        if isinstance(combined.columns, pd.MultiIndex):
            combined.columns = [col[0] if isinstance(col, tuple) else col for col in combined.columns]
        
        return combined

    def _generate_signals(self, strategy_spec: StrategySpec, prices_df: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals based on strategy specification"""
        signals = {}

        for signal_def in strategy_spec.signals:
            signal_name = signal_def.name or f"{signal_def.type}_{len(signals)}"

            if signal_def.type == SignalType.MA_CROSS:
                signals[signal_name] = self._ma_cross_signal(
                    prices_df, signal_def.params['fast'], signal_def.params['slow']
                )
            elif signal_def.type == SignalType.IV_PROXY:
                signals[signal_name] = self._iv_proxy_signal(prices_df, signal_def.params)
            elif signal_def.type == SignalType.RSI:
                signals[signal_name] = self._rsi_signal(prices_df, signal_def.params)
            else:
                logger.warning(f"Unsupported signal type: {signal_def.type}")
                continue

        return pd.DataFrame(signals, index=prices_df.index)

    def _ma_cross_signal(self, prices_df: pd.DataFrame, fast_period: int, slow_period: int) -> pd.Series:
        """Generate moving average crossover signal"""
        # Use first ticker for signal generation (could be enhanced)
        price_col = prices_df.columns[0]  # price_{ticker}
        prices = prices_df[price_col]

        fast_ma = prices.rolling(fast_period).mean()
        slow_ma = prices.rolling(slow_period).mean()

        return (fast_ma > slow_ma).astype(int)

    def _iv_proxy_signal(self, prices_df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
        """Generate implied volatility proxy signal"""
        # Simple realized volatility proxy
        # Handle both string columns and MultiIndex columns
        if isinstance(prices_df.columns, pd.MultiIndex):
            returns_col = [col for col in prices_df.columns if isinstance(col, tuple) and any(str(c).startswith('returns_') for c in col)][0]
        else:
            returns_col = [col for col in prices_df.columns if str(col).startswith('returns_')][0]
        
        returns = prices_df[returns_col]

        window = params.get('window', 20)
        rv = returns.rolling(window).std() * np.sqrt(252)  # Annualized

        threshold = params.get('low_thresh', 0.3)
        return (rv < threshold).astype(int)

    def _rsi_signal(self, prices_df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
        """Generate RSI-based signal"""
        # Simplified RSI implementation
        price_col = prices_df.columns[0]
        prices = prices_df[price_col]

        period = params.get('period', 14)
        overbought = params.get('overbought', 70)
        oversold = params.get('oversold', 30)

        # Calculate RSI
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        # Generate signals
        signal = pd.Series(0, index=prices.index)
        signal[rsi < oversold] = 1  # Buy signal
        signal[rsi > overbought] = -1  # Sell signal

        return signal

    def _generate_positions(self, strategy_spec: StrategySpec, signals_df: pd.DataFrame,
                          prices_df: pd.DataFrame) -> pd.DataFrame:
        """Generate position signals based on entry/exit conditions"""
        positions = pd.DataFrame(index=signals_df.index)

        # Evaluate entry conditions
        entry_signals = self._evaluate_conditions(strategy_spec.entry, signals_df)

        # Evaluate exit conditions if specified
        if strategy_spec.exit:
            exit_signals = self._evaluate_conditions(strategy_spec.exit, signals_df)
        else:
            exit_signals = pd.Series(False, index=signals_df.index)

        # Generate position series
        position_signal = pd.Series(0, index=signals_df.index)

        # Entry logic
        position_signal[entry_signals] = 1

        # Exit logic (close positions)
        position_signal[exit_signals] = 0

        # Forward fill positions (hold until exit)
        position_signal = position_signal.replace(0, np.nan).ffill().fillna(0)

        positions['target_weight'] = position_signal.astype(float)
        return positions

    def _evaluate_conditions(self, conditions, signals_df: pd.DataFrame) -> pd.Series:
        """Evaluate entry/exit conditions"""
        if conditions.all:
            # All conditions must be true
            result = pd.Series(True, index=signals_df.index)
            for condition in conditions.all:
                if '.' in condition:
                    signal_name, rule = condition.split('.', 1)
                    if signal_name in signals_df.columns:
                        result = result & (signals_df[signal_name] == 1)
                else:
                    # Handle expressions like "vol_regime<low_thresh"
                    if '<' in condition or '>' in condition or '==' in condition or '!=' in condition:
                        # This is an expression, not a signal name
                        logger.warning(f"Expression condition not yet supported: {condition}")
                        continue
                    else:
                        result = result & signals_df[condition].astype(bool)
        elif conditions.any:
            # Any condition must be true
            result = pd.Series(False, index=signals_df.index)
            for condition in conditions.any:
                if '.' in condition:
                    signal_name, rule = condition.split('.', 1)
                    if signal_name in signals_df.columns:
                        result = result | (signals_df[signal_name] == 1)
                else:
                    # Handle expressions like "vol_regime<low_thresh"
                    if '<' in condition or '>' in condition or '==' in condition or '!=' in condition:
                        # This is an expression, not a signal name
                        logger.warning(f"Expression condition not yet supported: {condition}")
                        continue
                    else:
                        result = result | signals_df[condition].astype(bool)
        else:
            result = pd.Series(False, index=signals_df.index)

        return result

    def _apply_position_sizing(self, strategy_spec: StrategySpec, positions_df: pd.DataFrame,
                             prices_df: pd.DataFrame) -> pd.DataFrame:
        """Apply position sizing and risk management"""
        sizing = strategy_spec.sizing

        # Apply volatility targeting
        if sizing.vol_target_ann > 0:
            positions_df = self._apply_vol_targeting(positions_df, prices_df, sizing.vol_target_ann)

        # Apply position limits
        positions_df['target_weight'] = positions_df['target_weight'].clip(
            sizing.min_weight, sizing.max_weight
        )

        return positions_df

    def _apply_vol_targeting(self, positions_df: pd.DataFrame, prices_df: pd.DataFrame,
                           vol_target: float) -> pd.DataFrame:
        """Apply volatility targeting to position sizes"""
        # Simple vol targeting based on recent realized vol
        returns_col = [col for col in prices_df.columns if col.startswith('returns_')][0]
        realized_vol = prices_df[returns_col].rolling(60).std() * np.sqrt(252)

        # Scale positions by (target_vol / realized_vol)
        vol_scalar = vol_target / realized_vol

        # Apply Kelly fraction adjustment
        kelly_fraction = 0.5  # Could be parameterized
        vol_scalar = vol_scalar * kelly_fraction

        # Apply to positions
        positions_df = positions_df.copy()
        positions_df['target_weight'] = positions_df['target_weight'] * vol_scalar

        return positions_df

    def _apply_options_overlays(self, strategy_spec: StrategySpec, positions_df: pd.DataFrame,
                               prices_df: pd.DataFrame) -> pd.DataFrame:
        """Apply options overlays for hedging"""
        # Placeholder for options overlay logic
        # This would integrate with options pricing models
        logger.info("Options overlays not yet implemented - using base positions")
        return positions_df

    def _calculate_returns_with_costs(self, positions_df: pd.DataFrame, prices_df: pd.DataFrame,
                                    strategy_spec: StrategySpec) -> Tuple[pd.Series, pd.DataFrame]:
        """Calculate strategy returns including trading costs"""
        # Get price returns
        returns_col = [col for col in prices_df.columns if col.startswith('returns_')][0]
        asset_returns = prices_df[returns_col]

        # Calculate position changes for transaction costs
        position_changes = positions_df['target_weight'].diff().abs()
        position_changes.iloc[0] = positions_df['target_weight'].iloc[0]  # Initial position

        # Calculate commissions and slippage
        commission_cost = position_changes * (self.commission_bps / 10000)
        slippage_cost = position_changes * (self.slippage_bps / 10000)

        # Total costs
        total_costs = commission_cost + slippage_cost

        # Strategy returns = position_weight * asset_returns - costs
        strategy_returns = (positions_df['target_weight'].shift(1).fillna(0) * asset_returns) - total_costs

        # Create trades dataframe
        trades_df = pd.DataFrame({
            'position_change': position_changes,
            'commission_cost': commission_cost,
            'slippage_cost': slippage_cost,
            'total_cost': total_costs,
            'weight': positions_df['target_weight']
        })

        return strategy_returns, trades_df

    def _calculate_metrics(self, returns: pd.Series, positions_df: pd.DataFrame,
                          trades_df: pd.DataFrame) -> Dict[str, float]:
        """Calculate comprehensive performance metrics"""
        # Basic metrics
        cumulative_returns = (1 + returns).cumprod()
        total_return = cumulative_returns.iloc[-1] - 1

        # Annualized metrics
        days = len(returns)
        years = days / 252
        ann_return = (1 + total_return) ** (1 / years) - 1

        # Volatility and Sharpe
        ann_vol = returns.std() * np.sqrt(252)
        sharpe = ann_return / ann_vol if ann_vol > 0 else 0

        # Drawdown
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_dd = drawdown.min()

        # Win rate and profit factor
        winning_trades = (returns > 0).sum()
        total_trades = (returns != 0).sum()
        win_rate = winning_trades / total_trades if total_trades > 0 else 0

        gross_profit = returns[returns > 0].sum()
        gross_loss = abs(returns[returns < 0].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # Turnover
        turnover = positions_df['target_weight'].diff().abs().sum()

        return {
            'total_return': total_return,
            'ann_return': ann_return,
            'ann_vol': ann_vol,
            'sharpe': sharpe,
            'max_dd': max_dd,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'turnover': turnover,
            'total_trades': total_trades,
            'avg_trade': returns[returns != 0].mean(),
        }


class WalkForwardBacktester:
    """Walk-forward optimization and testing"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.base_backtester = VectorizedBacktester(config)

    def run_walk_forward(self, strategy_spec: StrategySpec, market_data: Dict[str, pd.DataFrame],
                        train_window_months: int = 24, test_window_months: int = 6,
                        step_months: int = 3) -> List[BacktestResult]:
        """
        Run walk-forward analysis

        Args:
            strategy_spec: Strategy to test
            market_data: Price data
            train_window_months: Training window length
            test_window_months: Testing window length
            step_months: Step size for rolling windows

        Returns:
            List of backtest results for each test period
        """

        results = []

        # Get date range from data
        all_dates = sorted(list(market_data.values())[0].index)
        start_date = pd.Timestamp(all_dates[0])
        end_date = pd.Timestamp(all_dates[-1])

        current_train_end = start_date + pd.DateOffset(months=train_window_months)

        while current_train_end + pd.DateOffset(months=test_window_months) <= end_date:
            # Define train/test periods
            train_start = current_train_end - pd.DateOffset(months=train_window_months)
            test_end = current_train_end + pd.DateOffset(months=test_window_months)

            # Split data
            train_data = self._filter_data_by_date(market_data, train_start, current_train_end)
            test_data = self._filter_data_by_date(market_data, current_train_end, test_end)

            if not train_data or not test_data:
                break

            # Run backtest on test data
            result = self.base_backtester.run_backtest(strategy_spec, test_data)
            results.append(result)

            # Move window
            current_train_end += pd.DateOffset(months=step_months)

        return results

    def _filter_data_by_date(self, market_data: Dict[str, pd.DataFrame],
                           start_date: pd.Timestamp, end_date: pd.Timestamp) -> Dict[str, pd.DataFrame]:
        """Filter market data by date range"""
        filtered = {}
        for ticker, df in market_data.items():
            mask = (df.index >= start_date) & (df.index <= end_date)
            if mask.sum() > 0:
                filtered[ticker] = df[mask].copy()
        return filtered
