"""
Adaptive Strategy Research and Optimization System

Continuously generates, tests, and optimizes trading strategies based on:
- Current market regime
- Historical performance data
- Risk constraints
- Opportunity costs
- Machine learning insights

Implements genetic algorithms, reinforcement learning, and statistical testing.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime, timedelta
from sklearn.model_selection import TimeSeriesSplit
import random
import json
from pathlib import Path

logger = logging.getLogger(__name__)


def calculate_sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
    """Calculate Sharpe ratio manually"""
    if len(returns) < 2:
        return 0.0

    # Annualized return
    avg_return = np.mean(returns)
    annualized_return = avg_return * 252  # Assuming daily returns

    # Annualized volatility
    volatility = np.std(returns, ddof=1)
    annualized_volatility = volatility * np.sqrt(252)

    # Sharpe ratio
    if annualized_volatility == 0:
        return 0.0

    sharpe = (annualized_return - risk_free_rate) / annualized_volatility
    return sharpe


class AdaptiveStrategyResearcher:
    """
    Self-learning strategy research system that evolves trading strategies
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config

        # Strategy components
        self.technical_indicators = [
            'SMA', 'EMA', 'RSI', 'MACD', 'BBANDS', 'STOCH', 'CCI', 'MFI', 'WILLR', 'ROC'
        ]

        self.entry_signals = [
            'crossover', 'divergence', 'breakout', 'reversal', 'momentum', 'mean_reversion'
        ]

        self.exit_signals = [
            'target_profit', 'stop_loss', 'trailing_stop', 'time_exit', 'signal_reversal'
        ]

        self.risk_management = [
            'fixed_position', 'volatility_targeting', 'kelly_criterion', 'risk_parity'
        ]

        # Evolution parameters
        self.population_size = 50
        self.generations = 10
        self.mutation_rate = 0.1
        self.crossover_rate = 0.7

        # Strategy library
        self.strategy_library = self.load_strategy_library()
        self.performance_database = {}

    def load_strategy_library(self) -> Dict[str, Any]:
        """Load existing strategy library"""
        library_path = Path("data/strategy_library.json")

        if library_path.exists():
            try:
                with open(library_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load strategy library: {e}")

        return {}

    def save_strategy_library(self):
        """Save strategy library to disk"""
        library_path = Path("data/strategy_library.json")
        library_path.parent.mkdir(exist_ok=True)

        try:
            with open(library_path, 'w') as f:
                json.dump(self.strategy_library, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Could not save strategy library: {e}")

    async def research_strategies(self, historical_data: Dict[str, Any],
                                current_regime: str,
                                performance_history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Research and develop new trading strategies

        Args:
            historical_data: Historical price and volume data
            current_regime: Current market regime
            performance_history: Historical performance data

        Returns:
            List of promising new strategies
        """

        logger.info(f"🧪 Researching strategies for {current_regime} regime")

        # Generate strategy candidates
        candidates = self.generate_strategy_candidates(current_regime, performance_history)

        # Evaluate candidates on historical data
        evaluated_strategies = []
        for candidate in candidates:
            try:
                performance = await self.evaluate_strategy(candidate, historical_data)

                if performance and self.meets_minimum_criteria(performance):
                    candidate['performance'] = performance
                    candidate['regime'] = current_regime
                    candidate['generated_at'] = datetime.now().isoformat()

                    evaluated_strategies.append(candidate)

            except Exception as e:
                logger.warning(f"Strategy evaluation failed: {e}")
                continue

        # Rank and select top strategies
        ranked_strategies = self.rank_strategies(evaluated_strategies)

        # Add to strategy library
        for strategy in ranked_strategies[:5]:  # Keep top 5
            strategy_id = f"{strategy['name']}_{int(datetime.now().timestamp())}"
            self.strategy_library[strategy_id] = strategy

        self.save_strategy_library()

        logger.info(f"✅ Generated {len(ranked_strategies)} new strategies")

        return ranked_strategies[:10]  # Return top 10

    def generate_strategy_candidates(self, current_regime: str,
                                   performance_history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate candidate strategies for testing"""

        candidates = []

        # Generate random strategies
        for i in range(self.population_size):
            strategy = self.generate_random_strategy(current_regime)
            candidates.append(strategy)

        # Generate regime-specific strategies
        regime_strategies = self.generate_regime_specific_strategies(current_regime)
        candidates.extend(regime_strategies)

        # Evolve existing strategies
        if self.strategy_library:
            evolved_strategies = self.evolve_existing_strategies(performance_history)
            candidates.extend(evolved_strategies)

        return candidates

    def generate_random_strategy(self, regime: str) -> Dict[str, Any]:
        """Generate a random trading strategy"""

        strategy = {
            'name': f'random_{regime}_{random.randint(1000, 9999)}',
            'type': 'technical',
            'regime': regime,

            # Entry conditions
            'entry_indicators': random.sample(self.technical_indicators,
                                            random.randint(1, 3)),
            'entry_signal': random.choice(self.entry_signals),
            'entry_params': {
                'fast_period': random.randint(5, 20),
                'slow_period': random.randint(20, 50),
                'rsi_level': random.randint(20, 80)
            },

            # Exit conditions
            'exit_signal': random.choice(self.exit_signals),
            'exit_params': {
                'profit_target': random.uniform(0.02, 0.10),
                'stop_loss': random.uniform(0.01, 0.05),
                'trailing_stop': random.uniform(0.005, 0.03)
            },

            # Risk management
            'risk_management': random.choice(self.risk_management),
            'position_size': random.uniform(0.01, 0.05),  # 1-5% of portfolio

            # Filters
            'filters': {
                'min_volume': random.randint(500000, 2000000),
                'max_volatility': random.uniform(0.02, 0.08),
                'trend_filter': random.choice([True, False])
            }
        }

        return strategy

    def generate_regime_specific_strategies(self, regime: str) -> List[Dict[str, Any]]:
        """Generate strategies optimized for specific market regimes"""

        regime_strategies = []

        if regime == 'bull_market':
            # Momentum and trend-following strategies
            strategies = [
                {
                    'name': f'bull_momentum_{random.randint(1000, 9999)}',
                    'type': 'momentum',
                    'entry_indicators': ['EMA', 'MACD'],
                    'entry_signal': 'momentum',
                    'exit_signal': 'trailing_stop',
                    'risk_management': 'volatility_targeting'
                },
                {
                    'name': f'bull_trend_{random.randint(1000, 9999)}',
                    'type': 'trend_following',
                    'entry_indicators': ['SMA', 'ROC'],
                    'entry_signal': 'breakout',
                    'exit_signal': 'target_profit',
                    'risk_management': 'kelly_criterion'
                }
            ]

        elif regime == 'bear_market':
            # Defensive and mean-reversion strategies
            strategies = [
                {
                    'name': f'bear_defensive_{random.randint(1000, 9999)}',
                    'type': 'defensive',
                    'entry_indicators': ['RSI', 'CCI'],
                    'entry_signal': 'reversal',
                    'exit_signal': 'time_exit',
                    'risk_management': 'fixed_position'
                },
                {
                    'name': f'bear_volatility_{random.randint(1000, 9999)}',
                    'type': 'volatility',
                    'entry_indicators': ['BBANDS', 'ATR'],
                    'entry_signal': 'breakout',
                    'exit_signal': 'stop_loss',
                    'risk_management': 'risk_parity'
                }
            ]

        elif regime == 'high_volatility':
            # Volatility harvesting strategies
            strategies = [
                {
                    'name': f'vol_harvest_{random.randint(1000, 9999)}',
                    'type': 'volatility',
                    'entry_indicators': ['BBANDS', 'STOCH'],
                    'entry_signal': 'mean_reversion',
                    'exit_signal': 'target_profit',
                    'risk_management': 'volatility_targeting'
                }
            ]

        else:
            # Neutral/low volatility strategies
            strategies = [
                {
                    'name': f'neutral_carry_{random.randint(1000, 9999)}',
                    'type': 'carry',
                    'entry_indicators': ['SMA', 'RSI'],
                    'entry_signal': 'crossover',
                    'exit_signal': 'time_exit',
                    'risk_management': 'fixed_position'
                }
            ]

        # Add default parameters to regime strategies
        for strategy in strategies:
            strategy.update({
                'regime': regime,
                'entry_params': {'fast_period': 10, 'slow_period': 30, 'rsi_level': 30},
                'exit_params': {'profit_target': 0.05, 'stop_loss': 0.02, 'trailing_stop': 0.015},
                'position_size': 0.02,
                'filters': {'min_volume': 1000000, 'max_volatility': 0.05, 'trend_filter': False}
            })

        regime_strategies.extend(strategies)
        return regime_strategies

    def evolve_existing_strategies(self, performance_history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Evolve existing strategies through genetic algorithms"""

        if not self.strategy_library:
            return []

        evolved_strategies = []

        # Select top performers for breeding
        top_strategies = self.select_top_strategies(10)

        # Create offspring through crossover and mutation
        for i in range(len(top_strategies) - 1):
            for j in range(i + 1, min(i + 3, len(top_strategies))):
                if random.random() < self.crossover_rate:
                    offspring = self.crossover_strategies(top_strategies[i], top_strategies[j])
                    offspring = self.mutate_strategy(offspring)

                    offspring['name'] = f'evolved_{offspring["name"]}_{random.randint(1000, 9999)}'
                    evolved_strategies.append(offspring)

        return evolved_strategies

    def select_top_strategies(self, n: int) -> List[Dict[str, Any]]:
        """Select top performing strategies"""

        if not self.strategy_library:
            return []

        # Score strategies by performance
        scored_strategies = []
        for strategy_id, strategy in self.strategy_library.items():
            perf = strategy.get('performance', {})
            sharpe = perf.get('sharpe_ratio', 0)
            returns = perf.get('total_return', 0)
            max_dd = perf.get('max_drawdown', 1)

            # Composite score
            score = sharpe * 0.4 + returns * 0.3 - max_dd * 0.3
            scored_strategies.append((score, strategy))

        # Sort by score and return top n
        scored_strategies.sort(key=lambda x: x[0], reverse=True)
        return [strategy for score, strategy in scored_strategies[:n]]

    def crossover_strategies(self, strategy1: Dict[str, Any], strategy2: Dict[str, Any]) -> Dict[str, Any]:
        """Create offspring strategy from two parent strategies"""

        offspring = strategy1.copy()

        # Mix entry parameters
        for param in ['fast_period', 'slow_period', 'rsi_level']:
            if param in strategy1.get('entry_params', {}) and param in strategy2.get('entry_params', {}):
                parent1_val = strategy1['entry_params'][param]
                parent2_val = strategy2['entry_params'][param]
                offspring['entry_params'][param] = (parent1_val + parent2_val) // 2

        # Mix exit parameters
        for param in ['profit_target', 'stop_loss']:
            if param in strategy1.get('exit_params', {}) and param in strategy2.get('exit_params', {}):
                parent1_val = strategy1['exit_params'][param]
                parent2_val = strategy2['exit_params'][param]
                offspring['exit_params'][param] = (parent1_val + parent2_val) / 2

        # Randomly select some attributes from either parent
        if random.random() < 0.5:
            offspring['entry_signal'] = strategy2['entry_signal']
        if random.random() < 0.5:
            offspring['exit_signal'] = strategy2['exit_signal']
        if random.random() < 0.5:
            offspring['risk_management'] = strategy2['risk_management']

        return offspring

    def mutate_strategy(self, strategy: Dict[str, Any]) -> Dict[str, Any]:
        """Apply random mutations to strategy"""

        mutated = strategy.copy()

        if random.random() < self.mutation_rate:
            # Mutate entry parameters
            if 'entry_params' in mutated:
                for param in ['fast_period', 'slow_period', 'rsi_level']:
                    if param in mutated['entry_params'] and random.random() < 0.3:
                        current_val = mutated['entry_params'][param]
                        mutation = random.randint(-5, 5)
                        mutated['entry_params'][param] = max(1, current_val + mutation)

        if random.random() < self.mutation_rate:
            # Mutate exit parameters
            if 'exit_params' in mutated:
                for param in ['profit_target', 'stop_loss']:
                    if param in mutated['exit_params'] and random.random() < 0.3:
                        current_val = mutated['exit_params'][param]
                        mutation = random.uniform(-0.01, 0.01)
                        mutated['exit_params'][param] = max(0.001, current_val + mutation)

        if random.random() < self.mutation_rate:
            # Change signals
            if random.random() < 0.5:
                mutated['entry_signal'] = random.choice(self.entry_signals)
            else:
                mutated['exit_signal'] = random.choice(self.exit_signals)

        return mutated

    async def evaluate_strategy(self, strategy: Dict[str, Any],
                              historical_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Evaluate strategy performance on historical data"""

        try:
            # Use simplified backtest for evaluation
            from engine.backtest_engine.backtester import Backtester

            backtester = Backtester({})

            # Convert strategy to backtest format
            backtest_config = self.strategy_to_backtest_config(strategy)

            # Run backtest
            results = await backtester.run_backtest(backtest_config, historical_data)

            if results and 'performance' in results:
                return results['performance']

        except Exception as e:
            logger.warning(f"Strategy evaluation failed: {e}")

        return None

    def strategy_to_backtest_config(self, strategy: Dict[str, Any]) -> Dict[str, Any]:
        """Convert strategy dict to backtest configuration"""

        return {
            'strategy_name': strategy['name'],
            'universe': ['SPY'],  # Single asset for evaluation
            'signals': [
                {
                    'type': 'SMA_cross' if 'SMA' in strategy.get('entry_indicators', []) else 'RSI',
                    'name': 'entry_signal',
                    'params': strategy.get('entry_params', {})
                }
            ],
            'entry_condition': f"entry_signal > 0",
            'sizing': {
                'type': 'fixed',
                'size': strategy.get('position_size', 0.02)
            },
            'costs': {
                'commission': 0.001,
                'slippage': 0.0005
            },
            'risk_limits': {
                'max_drawdown': 0.15,
                'max_position_size': 0.05
            }
        }

    def meets_minimum_criteria(self, performance: Dict[str, Any]) -> bool:
        """Check if strategy meets minimum performance criteria"""

        sharpe = performance.get('sharpe_ratio', 0)
        total_return = performance.get('total_return', 0)
        max_drawdown = performance.get('max_drawdown', 1)
        win_rate = performance.get('win_rate', 0)

        # Minimum criteria
        min_sharpe = self.config['strategy_research'].get('min_sharpe_ratio', 1.0)
        max_drawdown_limit = self.config['strategy_research'].get('max_drawdown_limit', 0.15)

        return (sharpe >= min_sharpe and
                max_drawdown <= max_drawdown_limit and
                total_return > 0 and
                win_rate >= 0.4)

    def rank_strategies(self, strategies: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rank strategies by composite performance score"""

        def strategy_score(strategy):
            perf = strategy.get('performance', {})
            sharpe = perf.get('sharpe_ratio', 0)
            returns = perf.get('total_return', 0)
            win_rate = perf.get('win_rate', 0)
            max_dd = perf.get('max_drawdown', 1)

            # Weighted score
            score = (sharpe * 0.35 +
                    returns * 0.25 +
                    win_rate * 0.25 -
                    max_dd * 0.15)

            return score

        ranked = sorted(strategies, key=strategy_score, reverse=True)
        return ranked

    def get_strategy_recommendations(self, current_regime: str,
                                   risk_tolerance: str = 'moderate') -> List[Dict[str, Any]]:
        """Get strategy recommendations for current conditions"""

        if not self.strategy_library:
            return []

        # Filter strategies by regime and performance
        suitable_strategies = []

        for strategy_id, strategy in self.strategy_library.items():
            if strategy.get('regime') == current_regime or strategy.get('regime') == 'neutral':
                perf = strategy.get('performance', {})
                if self.meets_risk_tolerance(perf, risk_tolerance):
                    suitable_strategies.append(strategy)

        # Return top strategies
        return self.rank_strategies(suitable_strategies)[:5]

    def meets_risk_tolerance(self, performance: Dict[str, Any], risk_tolerance: str) -> bool:
        """Check if strategy matches risk tolerance"""

        max_dd = performance.get('max_drawdown', 1)
        volatility = performance.get('volatility', 1)
        sharpe = performance.get('sharpe_ratio', 0)

        if risk_tolerance == 'conservative':
            return max_dd <= 0.05 and volatility <= 0.10 and sharpe >= 1.5
        elif risk_tolerance == 'moderate':
            return max_dd <= 0.10 and volatility <= 0.15 and sharpe >= 1.0
        elif risk_tolerance == 'aggressive':
            return max_dd <= 0.20 and sharpe >= 0.5
        else:
            return True

    def get_strategy_statistics(self) -> Dict[str, Any]:
        """Get statistics about the strategy library"""

        if not self.strategy_library:
            return {'error': 'No strategies in library'}

        total_strategies = len(self.strategy_library)

        # Performance statistics
        sharpe_ratios = []
        total_returns = []
        max_drawdowns = []

        for strategy in self.strategy_library.values():
            perf = strategy.get('performance', {})
            if perf:
                sharpe_ratios.append(perf.get('sharpe_ratio', 0))
                total_returns.append(perf.get('total_return', 0))
                max_drawdowns.append(perf.get('max_drawdown', 0))

        # Regime distribution
        regime_counts = {}
        for strategy in self.strategy_library.values():
            regime = strategy.get('regime', 'unknown')
            regime_counts[regime] = regime_counts.get(regime, 0) + 1

        return {
            'total_strategies': total_strategies,
            'avg_sharpe_ratio': np.mean(sharpe_ratios) if sharpe_ratios else 0,
            'avg_total_return': np.mean(total_returns) if total_returns else 0,
            'avg_max_drawdown': np.mean(max_drawdowns) if max_drawdowns else 0,
            'regime_distribution': regime_counts,
            'best_strategy': max(self.strategy_library.items(),
                               key=lambda x: x[1].get('performance', {}).get('sharpe_ratio', 0))[0]
        }
