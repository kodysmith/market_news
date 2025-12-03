"""
Test script for trading system

Tests the modular components in backtest mode.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from trading.runner import StrategyRunner
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_backtest_mode():
    """Test backtest mode"""
    logger.info("Testing backtest mode...")

    try:
        runner = StrategyRunner('backtest', 'config')

        # Test single day run
        test_date = datetime(2024, 1, 15)
        result = runner.run(test_date)

        logger.info(f"Backtest run successful!")
        logger.info(f"Regime: {result['regime']['regime']}")
        logger.info(f"Portfolio value: ${result['portfolio_value']:,.2f}")
        logger.info(f"Positions: {result['final_positions']}")

        return True

    except Exception as e:
        logger.error(f"Backtest test failed: {e}", exc_info=True)
        return False


def test_components():
    """Test individual components"""
    logger.info("Testing individual components...")

    from trading.core.regime_detector import RegimeDetector
    from trading.core.portfolio_engine import PortfolioEngine
    from trading.core.risk_manager import RiskManager

    config = {
        'target_assets': ['SPY', 'QQQ', 'TQQQ'],
        'base_weights': {'SPY': 0.33, 'QQQ': 0.33, 'TQQQ': 0.34}
    }

    # Test regime detector
    try:
        detector = RegimeDetector(config)
        market_data = {
            'spy': {
                'prices': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110,
                          111, 112, 113, 114, 115, 116, 117, 118, 119, 120]
            },
            'vix': {'level': 20, 'change': 0}
        }
        regime = detector.detect_regime(market_data)
        logger.info(f"Regime detection: {regime['regime']} (confidence: {regime['confidence']:.1%})")
    except Exception as e:
        logger.error(f"Regime detector test failed: {e}")
        return False

    # Test portfolio engine
    try:
        engine = PortfolioEngine(config)
        weights = engine.calculate_weights(regime)
        logger.info(f"Portfolio weights: {weights}")
    except Exception as e:
        logger.error(f"Portfolio engine test failed: {e}")
        return False

    # Test risk manager
    try:
        risk_manager = RiskManager(config)
        validation = risk_manager.validate_weights(weights)
        logger.info(f"Risk validation: {validation['is_valid']}")
    except Exception as e:
        logger.error(f"Risk manager test failed: {e}")
        return False

    return True


if __name__ == '__main__':
    print("="*70)
    print("TRADING SYSTEM TEST")
    print("="*70)
    print()

    # Test components
    print("Testing individual components...")
    if test_components():
        print("✅ Component tests passed")
    else:
        print("❌ Component tests failed")
        sys.exit(1)

    print()

    # Test backtest mode
    print("Testing backtest mode...")
    if test_backtest_mode():
        print("✅ Backtest mode test passed")
    else:
        print("❌ Backtest mode test failed")
        sys.exit(1)

    print()
    print("="*70)
    print("ALL TESTS PASSED")
    print("="*70)
