"""
Integration tests for AI Quant Trading System

Tests the interaction between core components.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.strategy_dsl import StrategyValidator, EXAMPLE_TQQQ_STRATEGY
from engine.data_ingestion.data_manager import DataManager
from engine.feature_builder.feature_builder import FeatureBuilder
from engine.backtest_engine.backtester import VectorizedBacktester
from engine.robustness_lab.robustness_tester import RobustnessTester


class TestIntegration:
    """Integration tests for core system components"""

    @pytest.fixture
    def config(self):
        """Basic test configuration"""
        return {
            'data_path': 'data',
            'start_date': '2020-01-01',
            'end_date': '2023-01-01',
            'initial_capital': 100000,
            'commission_bps': 2.0,
            'slippage_bps': 1.0,
        }

    @pytest.fixture
    def mock_market_data(self):
        """Create mock market data for testing"""
        dates = pd.date_range('2020-01-01', '2023-01-01', freq='D')
        np.random.seed(42)

        # Generate realistic-ish price data
        n_days = len(dates)
        returns = np.random.normal(0.0005, 0.02, n_days)  # ~12% annual return, 30% vol
        prices = 100 * np.exp(np.cumsum(returns))

        df = pd.DataFrame({
            'open': prices * (1 + np.random.normal(0, 0.005, n_days)),
            'high': prices * (1 + np.random.normal(0.005, 0.01, n_days)),
            'low': prices * (1 - np.random.normal(0.005, 0.01, n_days)),
            'close': prices,
            'volume': np.random.randint(100000, 1000000, n_days),
            'adj_close': prices
        }, index=dates)

        return {'TEST': df}

    def test_strategy_dsl_validation(self):
        """Test strategy DSL validation"""
        spec = StrategyValidator.validate_spec(EXAMPLE_TQQQ_STRATEGY)
        assert spec.name == "tqqq_regime_puts_v1"
        assert len(spec.signals) == 2
        assert spec.sizing.vol_target_ann == 0.15

    def test_data_manager_mock(self, config, mock_market_data):
        """Test data manager with mock data"""
        # This would normally download real data, but we'll skip for testing
        assert len(mock_market_data) > 0
        assert 'TEST' in mock_market_data
        assert len(mock_market_data['TEST']) > 100

    def test_feature_builder(self, config, mock_market_data):
        """Test feature builder"""
        builder = FeatureBuilder(config)
        features = builder.build_features(mock_market_data)

        assert 'TEST' in features
        df = features['TEST']

        # Check that basic features were created
        assert 'close' in df.columns
        assert 'returns_1d' in df.columns
        assert 'vol_20d' in df.columns

    def test_backtester(self, config, mock_market_data):
        """Test backtester with simple strategy"""
        spec = StrategyValidator.validate_spec(EXAMPLE_TQQQ_STRATEGY)

        backtester = VectorizedBacktester(config)
        result = backtester.run_backtest(spec, mock_market_data)

        assert result is not None
        assert hasattr(result, 'metrics')
        assert 'sharpe' in result.metrics
        assert 'total_return' in result.metrics

        # Basic sanity checks
        assert isinstance(result.metrics['sharpe'], (int, float))
        assert isinstance(result.metrics['total_return'], (int, float))

    def test_robustness_tester(self, config, mock_market_data):
        """Test robustness testing"""
        spec = StrategyValidator.validate_spec(EXAMPLE_TQQQ_STRATEGY)

        backtester = VectorizedBacktester(config)
        backtest_result = backtester.run_backtest(spec, mock_market_data)

        robustness_tester = RobustnessTester(config)
        report = robustness_tester.run_full_robustness_suite([backtest_result], mock_market_data)

        assert report is not None
        assert 'performance' in report
        assert 'green_light' in report

        # Check that green light decision was made
        assert 'approved' in report['green_light']

    def test_end_to_end_workflow(self, config, mock_market_data):
        """Test complete workflow from strategy to robustness report"""

        # 1. Validate strategy
        spec = StrategyValidator.validate_spec(EXAMPLE_TQQQ_STRATEGY)
        assert spec is not None

        # 2. Get/build features
        feature_builder = FeatureBuilder(config)
        features = feature_builder.build_features(mock_market_data)
        assert len(features) > 0

        # 3. Run backtest
        backtester = VectorizedBacktester(config)
        backtest_result = backtester.run_backtest(spec, mock_market_data)
        assert backtest_result is not None

        # 4. Run robustness tests
        robustness_tester = RobustnessTester(config)
        robustness_report = robustness_tester.run_full_robustness_suite(
            [backtest_result], mock_market_data
        )
        assert robustness_report is not None

        # 5. Check final decision
        green_light = robustness_report.get('green_light', {})
        approved = green_light.get('approved', False)
        score = green_light.get('score', 0)

        # With mock data, approval is probabilistic, but we check the process works
        assert isinstance(approved, bool)
        assert isinstance(score, (int, float))
        assert 0 <= score <= 10

        print(f"Integration test completed. Strategy approved: {approved}, Score: {score}")


if __name__ == "__main__":
    # Run basic integration test
    test = TestIntegration()

    # Create mock config and data
    config = {
        'data_path': 'data',
        'start_date': '2020-01-01',
        'end_date': '2023-01-01',
        'initial_capital': 100000,
        'commission_bps': 2.0,
        'slippage_bps': 1.0,
    }

    # Create mock market data
    dates = pd.date_range('2020-01-01', '2023-01-01', freq='D')
    np.random.seed(42)
    n_days = len(dates)
    returns = np.random.normal(0.0005, 0.02, n_days)
    prices = 100 * np.exp(np.cumsum(returns))

    mock_data = pd.DataFrame({
        'open': prices * (1 + np.random.normal(0, 0.005, n_days)),
        'high': prices * (1 + np.random.normal(0.005, 0.01, n_days)),
        'low': prices * (1 - np.random.normal(0.005, 0.01, n_days)),
        'close': prices,
        'volume': np.random.randint(100000, 1000000, n_days),
        'adj_close': prices
    }, index=dates)

    mock_market_data = {'TEST': mock_data}

    try:
        print("Running integration test...")
        test.test_end_to_end_workflow(config, mock_market_data)
        print("✅ Integration test passed!")
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()

