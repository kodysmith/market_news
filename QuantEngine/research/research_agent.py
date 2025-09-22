"""
Research Agent for AI Quant Trading System

Generates strategy hypotheses, validates them through backtesting,
and produces research notes for promising strategies.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
from pathlib import Path
import json
import time
from datetime import datetime

import sys
from pathlib import Path

# Add QuantEngine root to path for imports
quant_engine_root = Path(__file__).parent.parent
if str(quant_engine_root) not in sys.path:
    sys.path.insert(0, str(quant_engine_root))

from utils.strategy_dsl import StrategySpec, StrategyValidator, EXAMPLE_TQQQ_STRATEGY
from engine.data_ingestion.data_manager import DataManager
from engine.feature_builder.feature_builder import FeatureBuilder
from engine.backtest_engine.backtester import VectorizedBacktester, WalkForwardBacktester
from engine.robustness_lab.robustness_tester import RobustnessTester
from engine.reporting_notes.report_generator import ReportGenerator

logger = logging.getLogger(__name__)


class ResearchAgent:
    """AI-powered strategy research and generation agent"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

        # Initialize all modules
        self.data_manager = DataManager(config)
        self.feature_builder = FeatureBuilder(config)
        self.backtester = VectorizedBacktester(config)
        self.walk_forward_backtester = WalkForwardBacktester(config)
        self.robustness_tester = RobustnessTester(config)
        self.report_generator = ReportGenerator(config)

        # Load feature catalog
        self.feature_catalog = self._load_feature_catalog()

        # Research state
        self.research_history = []
        self.approved_strategies = []

    def _load_feature_catalog(self) -> Dict[str, Any]:
        """Load the catalog of available features and signals"""

        catalog_path = Path("config/feature_catalog.yaml")
        if catalog_path.exists():
            # Would load from YAML - for now use hardcoded catalog
            pass

        # Default feature catalog
        return {
            'price_features': [
                'returns_1d', 'returns_5d', 'returns_20d', 'returns_60d',
                'vol_5d', 'vol_20d', 'vol_60d',
                'sma_20', 'sma_50', 'sma_200',
                'rsi_14', 'macd', 'bb_upper', 'bb_lower'
            ],
            'microstructure': [
                'gap_up', 'gap_down', 'range_compression_20',
                'rv_proxy_20', 'vrp_20'
            ],
            'signal_types': [
                'MA_cross', 'IV_proxy', 'RSI', 'MACD', 'bollinger',
                'momentum', 'mean_reversion', 'sentiment'
            ],
            'trading_objectives': [
                'trend_following', 'mean_reversion', 'carry', 'volatility_targeting',
                'risk_parity', 'momentum', 'quality', 'value'
            ]
        }

    def run_research_cycle(self, objectives: List[str] = None) -> Dict[str, Any]:
        """
        Run a complete research cycle: generate hypotheses -> test -> validate -> report

        Args:
            objectives: List of trading objectives to focus on

        Returns:
            Research cycle results
        """

        if objectives is None:
            objectives = ['trend_following', 'volatility_targeting']

        logger.info(f"Starting research cycle with objectives: {objectives}")

        cycle_results = {
            'cycle_id': datetime.now().strftime("%Y%m%d_%H%M%S"),
            'objectives': objectives,
            'hypotheses_generated': 0,
            'strategies_tested': 0,
            'strategies_approved': 0,
            'reports_generated': [],
            'start_time': datetime.now()
        }

        # Get market data
        universe = self.config.get('universe', ['SPY', 'QQQ', 'TQQQ'])
        market_data = self.data_manager.get_universe_data('default')

        if not market_data:
            logger.error("No market data available for research")
            return cycle_results

        # Generate hypotheses
        hypotheses = self._generate_hypotheses(objectives, market_data)
        cycle_results['hypotheses_generated'] = len(hypotheses)

        logger.info(f"Generated {len(hypotheses)} strategy hypotheses")

        # Test each hypothesis
        approved_strategies = []

        for hypothesis in hypotheses:
            try:
                # Convert hypothesis to strategy spec
                strategy_spec = self._compile_hypothesis_to_strategy(hypothesis, universe)

                if strategy_spec:
                    # Backtest the strategy
                    backtest_result = self.backtester.run_backtest(strategy_spec, market_data)

                    # Run robustness tests
                    robustness_report = self.robustness_tester.run_full_robustness_suite(
                        [backtest_result], market_data
                    )

                    cycle_results['strategies_tested'] += 1

                    # Check if strategy passes green light criteria
                    if robustness_report.get('green_light', {}).get('approved', False):
                        approved_strategies.append({
                            'spec': strategy_spec,
                            'backtest': backtest_result,
                            'robustness': robustness_report
                        })

                        # Generate research note
                        report_path = self.report_generator.generate_research_note(
                            strategy_spec, backtest_result, robustness_report, market_data
                        )

                        cycle_results['reports_generated'].append(report_path)
                        cycle_results['strategies_approved'] += 1

                        logger.info(f"Strategy {strategy_spec.name} approved and documented")

            except Exception as e:
                logger.error(f"Failed to test hypothesis: {e}")
                continue

        cycle_results['end_time'] = datetime.now()
        cycle_results['duration_minutes'] = (cycle_results['end_time'] - cycle_results['start_time']).total_seconds() / 60

        # Save research history
        self.research_history.append(cycle_results)

        logger.info(f"Research cycle completed: {cycle_results['strategies_approved']}/{cycle_results['strategies_tested']} strategies approved")

        return cycle_results

    def _generate_hypotheses(self, objectives: List[str], market_data: Dict[str, pd.DataFrame]) -> List[Dict[str, Any]]:
        """
        Generate strategy hypotheses based on objectives and market data

        This is a simplified version. In a full implementation, this would use
        LLM generation with the feature catalog as context.
        """

        hypotheses = []

        # Template-based hypothesis generation
        templates = {
            'trend_following': [
                {
                    'name': 'ma_cross_trend',
                    'description': 'Trend following with moving average crossovers',
                    'signals': [
                        {'type': 'MA_cross', 'fast': 20, 'slow': 200, 'rule': 'fast>slow'}
                    ],
                    'entry': ['trend_filter.rule'],
                    'universe': ['SPY', 'QQQ']
                },
                {
                    'name': 'momentum_trend',
                    'description': 'Momentum-based trend following',
                    'signals': [
                        {'type': 'momentum', 'period': 20, 'threshold': 0.05}
                    ],
                    'entry': ['momentum.signal'],
                    'universe': ['SPY', 'QQQ', 'IWM']
                }
            ],
            'volatility_targeting': [
                {
                    'name': 'low_vol_carry',
                    'description': 'Carry strategy during low volatility periods',
                    'signals': [
                        {'type': 'IV_proxy', 'method': 'rv20_scaled', 'low_thresh': 0.4}
                    ],
                    'entry': ['vol_regime<low_thresh'],
                    'universe': ['SPY']
                }
            ],
            'mean_reversion': [
                {
                    'name': 'rsi_reversion',
                    'description': 'Mean reversion using RSI extremes',
                    'signals': [
                        {'type': 'RSI', 'period': 14, 'overbought': 70, 'oversold': 30}
                    ],
                    'entry': ['rsi_oversold'],
                    'universe': ['SPY', 'QQQ']
                }
            ]
        }

        # Generate hypotheses for each objective
        for objective in objectives:
            if objective in templates:
                for template in templates[objective]:
                    # Customize template with current market conditions
                    hypothesis = self._customize_template_for_market(template, market_data)
                    hypotheses.append(hypothesis)

        # Add some variation to existing approved strategies
        if self.approved_strategies:
            for approved in self.approved_strategies[-2:]:  # Last 2 strategies
                variations = self._generate_strategy_variations(approved['spec'])
                hypotheses.extend(variations)

        return hypotheses[:self.config.get('hypotheses_per_run', 10)]  # Limit number

    def _customize_template_for_market(self, template: Dict[str, Any], market_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Customize a hypothesis template based on current market conditions"""

        # This is a simplified customization - in practice would analyze current market regime
        customized = template.copy()

        # Adjust parameters based on recent volatility
        if market_data:
            sample_ticker = list(market_data.keys())[0]
            recent_vol = market_data[sample_ticker]['close'].pct_change().rolling(20).std().iloc[-1] * np.sqrt(252)

            # Adjust position sizing based on volatility
            if recent_vol > 0.25:  # High vol environment
                customized['vol_target'] = 0.10  # Reduce vol target
            elif recent_vol < 0.15:  # Low vol environment
                customized['vol_target'] = 0.18  # Increase vol target
            else:
                customized['vol_target'] = 0.15  # Standard target

        return customized

    def _generate_strategy_variations(self, base_spec: StrategySpec) -> List[Dict[str, Any]]:
        """Generate variations of an existing approved strategy"""

        variations = []

        # Parameter variations
        param_variations = [
            {'fast_ma': base_spec.signals[0].params.get('fast', 20) + 5},
            {'slow_ma': base_spec.signals[0].params.get('slow', 200) - 20},
            {'vol_target': base_spec.sizing.vol_target_ann * 0.9},
            {'max_weight': min(base_spec.sizing.max_weight * 1.1, 1.0)},
        ]

        for i, variation in enumerate(param_variations):
            var_dict = base_spec.dict()
            var_dict['name'] = f"{base_spec.name}_var_{i+1}"

            # Apply variation
            if 'fast_ma' in variation:
                var_dict['signals'][0]['params']['fast'] = variation['fast_ma']
            elif 'slow_ma' in variation:
                var_dict['signals'][0]['params']['slow'] = variation['slow_ma']
            elif 'vol_target' in variation:
                var_dict['sizing']['vol_target_ann'] = variation['vol_target']
            elif 'max_weight' in variation:
                var_dict['sizing']['max_weight'] = variation['max_weight']

            variations.append(var_dict)

        return variations

    def _compile_hypothesis_to_strategy(self, hypothesis: Dict[str, Any], universe: List[str]) -> Optional[StrategySpec]:
        """Convert a hypothesis dictionary to a validated StrategySpec"""

        try:
            # Build strategy spec from hypothesis
            strategy_dict = {
                'name': hypothesis.get('name', f"hypothesis_{int(time.time())}"),
                'description': hypothesis.get('description', ''),
                'universe': hypothesis.get('universe', universe),
                'signals': hypothesis.get('signals', []),
                'entry': {'all': hypothesis.get('entry', [])},
                'sizing': {
                    'vol_target_ann': hypothesis.get('vol_target', 0.15),
                    'max_weight': hypothesis.get('max_weight', 1.0),
                    'kelly_fraction': 0.5
                },
                'costs': {'commission_bps': 2.0, 'slippage_bps': 1.0, 'fee_per_option': 0.65},
                'risk': {
                    'max_dd_pct': 0.25,
                    'max_gross_exposure': 1.2,
                    'circuit_breaker_dd': 0.05
                }
            }

            # Add overlays if specified
            if hypothesis.get('overlays'):
                strategy_dict['overlays'] = hypothesis['overlays']

            # Validate and return
            return StrategyValidator.validate_spec(strategy_dict)

        except Exception as e:
            logger.error(f"Failed to compile hypothesis to strategy: {e}")
            return None

    def get_research_summary(self) -> Dict[str, Any]:
        """Get summary of research activities"""

        if not self.research_history:
            return {'message': 'No research cycles completed'}

        total_cycles = len(self.research_history)
        total_hypotheses = sum(cycle['hypotheses_generated'] for cycle in self.research_history)
        total_tested = sum(cycle['strategies_tested'] for cycle in self.research_history)
        total_approved = sum(cycle['strategies_approved'] for cycle in self.research_history)

        success_rate = total_approved / total_tested if total_tested > 0 else 0

        return {
            'total_research_cycles': total_cycles,
            'total_hypotheses_generated': total_hypotheses,
            'total_strategies_tested': total_tested,
            'total_strategies_approved': total_approved,
            'approval_success_rate': success_rate,
            'approved_strategies': [s['spec'].name for s in self.approved_strategies],
            'recent_cycle': self.research_history[-1] if self.research_history else None
        }

    def run_tqqq_reference_implementation(self) -> Dict[str, Any]:
        """Run the TQQQ reference strategy as a test"""

        logger.info("Running TQQQ reference strategy implementation")

        # Get market data
        market_data = self.data_manager.get_market_data(['TQQQ'], '2020-01-01', '2024-01-01')

        if not market_data:
            return {'error': 'No market data available'}

        # Use the example TQQQ strategy
        strategy_spec = StrategyValidator.validate_spec(EXAMPLE_TQQQ_STRATEGY)

        # Run backtest
        backtest_result = self.backtester.run_backtest(strategy_spec, market_data)

        # Run robustness tests
        robustness_report = self.robustness_tester.run_full_robustness_suite([backtest_result], market_data)

        # Generate report
        report_path = self.report_generator.generate_research_note(
            strategy_spec, backtest_result, robustness_report, market_data
        )

        result = {
            'strategy_name': strategy_spec.name,
            'backtest_metrics': backtest_result.metrics,
            'robustness_report': robustness_report,
            'report_path': report_path,
            'approved': robustness_report.get('green_light', {}).get('approved', False)
        }

        logger.info(f"TQQQ reference strategy {'approved' if result['approved'] else 'rejected'}")

        return result
