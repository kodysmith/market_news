#!/usr/bin/env python3
"""
Simplified Phase 0 Test - No external dependencies

Tests core logic without pandas/yfinance to validate architecture.
"""

import sys
import json
import random
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict

# Add QuantEngine to path
sys.path.insert(0, str(Path(__file__).parent))

def test_strategy_dsl():
    """Test strategy DSL functionality"""
    print("🧪 Testing Strategy DSL...")

    try:
        from utils.strategy_dsl import StrategyValidator, EXAMPLE_TQQQ_STRATEGY

        # Test validation
        spec = StrategyValidator.validate_spec(EXAMPLE_TQQQ_STRATEGY)
        print("✅ Strategy validation successful")
        print(f"   Strategy: {spec.name}")
        print(f"   Universe: {spec.universe}")
        print(f"   Signals: {len(spec.signals)}")

        return True
    except Exception as e:
        print(f"❌ Strategy DSL test failed: {e}")
        return False

def create_mock_market_data():
    """Create mock market data for testing"""
    print("📊 Creating mock market data...")

    # Generate 4 years of daily data (2020-2024)
    start_date = datetime(2020, 1, 1)
    dates = []
    current_date = start_date

    while current_date <= datetime(2024, 1, 1):
        if current_date.weekday() < 5:  # Monday-Friday
            dates.append(current_date)
        current_date += timedelta(days=1)

    # Mock prices for different assets
    assets = {
        'SPY': {'start_price': 300, 'trend': 0.08, 'vol': 0.15},
        'QQQ': {'start_price': 280, 'trend': 0.10, 'vol': 0.18},
        'TQQQ': {'start_price': 50, 'trend': 0.25, 'vol': 0.35},  # Leveraged ETF
        'XLE': {'start_price': 60, 'trend': 0.05, 'vol': 0.20},
        'XLF': {'start_price': 30, 'trend': 0.06, 'vol': 0.22},
        'XLK': {'start_price': 90, 'trend': 0.12, 'vol': 0.16}
    }

    market_data = {}

    for ticker, params in assets.items():
        prices = []
        current_price = params['start_price']

        for i, date in enumerate(dates):
            # Random walk with drift
            daily_return = random.gauss(params['trend']/252, params['vol']/np.sqrt(252))
            current_price *= (1 + daily_return)
            prices.append(current_price)

        # Create mock OHLCV data
        data = []
        for i, (date, price) in enumerate(zip(dates, prices)):
            # Simple OHLC simulation
            high = price * (1 + abs(random.gauss(0, 0.02)))
            low = price * (1 - abs(random.gauss(0, 0.02)))
            open_price = prices[i-1] if i > 0 else price
            volume = random.randint(1000000, 50000000)

            data.append({
                'date': date,
                'open': round(open_price, 2),
                'high': round(high, 2),
                'low': round(low, 2),
                'close': round(price, 2),
                'volume': volume,
                'adj_close': round(price, 2)
            })

        market_data[ticker] = data
        print(f"   ✅ {ticker}: {len(data)} days of data")

    return market_data

def simple_backtester(strategy_spec, market_data, config):
    """Simple backtester implementation"""

    ticker = strategy_spec.get('universe', ['SPY'])[0]
    if ticker not in market_data:
        return None

    data = market_data[ticker]

    # Extract close prices and calculate returns
    closes = [d['close'] for d in data]
    returns = []
    for i in range(1, len(closes)):
        ret = (closes[i] - closes[i-1]) / closes[i-1]
        returns.append(ret)

    # Generate signals
    signals = generate_signals(strategy_spec, data)

    # Calculate positions (simplified)
    positions = [0] * len(data)
    for i in range(len(signals)):
        if signals[i] > 0:
            positions[i] = 1.0  # Long position

    # Calculate strategy returns with costs
    strategy_returns = []
    commission_bps = config.get('commission_bps', 2.0)
    slippage_bps = config.get('slippage_bps', 1.0)

    for i in range(1, len(positions)):
        prev_position = positions[i-1]
        current_position = positions[i]

        # Transaction costs
        position_change = abs(current_position - prev_position)
        costs = position_change * (commission_bps + slippage_bps) / 10000

        # Strategy return
        asset_return = returns[i-1] if i-1 < len(returns) else 0
        strategy_return = (prev_position * asset_return) - costs
        strategy_returns.append(strategy_return)

    # Calculate metrics
    if strategy_returns:
        total_return = sum(strategy_returns)
        ann_return = total_return * 252 / len(strategy_returns)  # Rough annualization

        # Volatility and Sharpe (simplified)
        if len(strategy_returns) > 1:
            avg_return = sum(strategy_returns) / len(strategy_returns)
            variance = sum((r - avg_return)**2 for r in strategy_returns) / len(strategy_returns)
            ann_vol = (variance ** 0.5) * (252 ** 0.5)
            sharpe = ann_return / ann_vol if ann_vol > 0 else 0
        else:
            ann_vol = 0
            sharpe = 0

        # Max drawdown (simplified)
        cumulative = 1
        max_cumulative = 1
        max_dd = 0

        for ret in strategy_returns:
            cumulative *= (1 + ret)
            max_cumulative = max(max_cumulative, cumulative)
            dd = (cumulative - max_cumulative) / max_cumulative
            max_dd = min(max_dd, dd)

        # Win rate
        winning_trades = sum(1 for r in strategy_returns if r > 0)
        win_rate = winning_trades / len(strategy_returns) if strategy_returns else 0

        metrics = {
            'total_return': total_return,
            'ann_return': ann_return,
            'ann_vol': ann_vol,
            'sharpe': sharpe,
            'max_dd': max_dd,
            'win_rate': win_rate,
            'num_trades': len([p for p in positions if p != 0])
        }
    else:
        metrics = {'error': 'No returns generated'}

    return {
        'returns': strategy_returns,
        'positions': positions,
        'metrics': metrics,
        'data_length': len(data)
    }

def generate_signals(strategy_spec, data):
    """Generate trading signals from strategy spec"""
    signals = [0] * len(data)

    for signal_def in strategy_spec.get('signals', []):
        signal_type = signal_def.get('type')

        if signal_type == 'MA_cross':
            # Simple moving average crossover
            params = signal_def.get('params', {})
            fast_period = params.get('fast', 20)
            slow_period = params.get('slow', 200)

            closes = [d['close'] for d in data]

            for i in range(max(fast_period, slow_period), len(data)):
                fast_sum = sum(closes[i-fast_period:i])
                fast_ma = fast_sum / fast_period

                slow_sum = sum(closes[i-slow_period:i])
                slow_ma = slow_sum / slow_period

                if fast_ma > slow_ma:
                    signals[i] = 1

        elif signal_type == 'IV_proxy':
            # Simplified volatility proxy
            params = signal_def.get('params', {})
            threshold = params.get('low_thresh', 0.45)

            closes = [d['close'] for d in data]

            for i in range(20, len(data)):
                # Calculate rolling volatility
                window_returns = []
                for j in range(i-20, i):
                    if j > 0:
                        ret = (closes[j] - closes[j-1]) / closes[j-1]
                        window_returns.append(ret)

                if window_returns:
                    vol = sum(r**2 for r in window_returns) / len(window_returns)
                    vol = vol ** 0.5  # Standard deviation proxy

                    if vol < threshold:
                        signals[i] = 1

    return signals

def generate_report(result, strategy_spec, output_path):
    """Generate performance report"""

    if not result or 'metrics' not in result:
        print("❌ No results to report")
        return

    metrics = result['metrics']

    report = f"""# {strategy_spec.get('name', 'Strategy')} - Phase 0 Performance Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Strategy Overview
- **Universe:** {', '.join(strategy_spec.get('universe', ['Unknown']))}
- **Signals:** {len(strategy_spec.get('signals', []))}
- **Data Points:** {result.get('data_length', 0)}

## Performance Metrics

| Metric | Value |
|--------|-------|
| Total Return | {metrics.get('total_return', 0):.4f} |
| Annualized Return | {metrics.get('ann_return', 0):.4f} |
| Annualized Volatility | {metrics.get('ann_vol', 0):.4f} |
| Sharpe Ratio | {metrics.get('sharpe', 0):.2f} |
| Max Drawdown | {metrics.get('max_dd', 0):.4f} |
| Win Rate | {metrics.get('win_rate', 0):.1%} |
| Number of Trades | {metrics.get('num_trades', 0)} |

## Analysis

"""

    # Performance commentary
    sharpe = metrics.get('sharpe', 0)
    ann_return = metrics.get('ann_return', 0)
    max_dd = metrics.get('max_dd', 0)

    if sharpe > 1.0:
        report += f"**Strong risk-adjusted performance** with Sharpe ratio of {sharpe:.2f}.\n\n"
    elif sharpe > 0.5:
        report += f"**Moderate risk-adjusted performance** with Sharpe ratio of {sharpe:.2f}.\n\n"
    else:
        report += f"**Weak risk-adjusted performance** with Sharpe ratio of {sharpe:.2f}.\n\n"

    if ann_return > 0.10:
        report += f"**Good absolute returns** of {ann_return:.1%} annualized.\n\n"
    elif ann_return > 0.05:
        report += f"**Moderate absolute returns** of {ann_return:.1%} annualized.\n\n"
    else:
        report += f"**Poor absolute returns** of {ann_return:.1%} annualized.\n\n"

    if abs(max_dd) < 0.20:
        report += f"**Acceptable risk** with maximum drawdown of {max_dd:.1%}.\n\n"
    else:
        report += f"**High risk** with maximum drawdown of {max_dd:.1%}.\n\n"

    report += "## Conclusion\n\n"
    if sharpe > 0.8 and abs(max_dd) < 0.25:
        report += "✅ **Strategy passes Phase 0 validation**\n\n"
    else:
        report += "❌ **Strategy needs improvement before Phase 1**\n\n"

    # Save report
    with open(output_path, 'w') as f:
        f.write(report)

    print(f"✅ Report saved to {output_path}")

def run_phase0_test():
    """Run Phase 0 validation test"""
    print("🚀 Starting Phase 0 Validation Test")
    print("=" * 50)

    # Test 1: Strategy DSL
    if not test_strategy_dsl():
        print("❌ Cannot proceed without working DSL")
        return

    # Test 2: Mock data creation
    market_data = create_mock_market_data()
    if not market_data:
        print("❌ Cannot proceed without market data")
        return

    # Test 3: Backtesting
    print("🔄 Running backtest...")

    config = {
        'commission_bps': 2.0,
        'slippage_bps': 1.0,
    }

    from utils.strategy_dsl import EXAMPLE_TQQQ_STRATEGY
    strategy_spec = EXAMPLE_TQQQ_STRATEGY

    result = simple_backtester(strategy_spec, market_data, config)

    if result and 'metrics' in result:
        print("✅ Backtest completed")
        print(".2%")

        # Test 4: Report generation
        reports_dir = Path('reports')
        reports_dir.mkdir(exist_ok=True)

        report_path = reports_dir / f"{strategy_spec['name']}_phase0_report.md"
        generate_report(result, strategy_spec, str(report_path))

        print("✅ Phase 0 validation completed!")
        print(f"📊 Check the report at: {report_path}")

        # Summary
        metrics = result['metrics']
        print("\n📈 Key Metrics:")
        print(".2%")
        print(".2f")
        print(".1%")

    else:
        print("❌ Backtest failed")

if __name__ == "__main__":
    # Mock numpy for the test
    class np:
        @staticmethod
        def sqrt(x):
            return x ** 0.5

    # Run the test
    run_phase0_test()
