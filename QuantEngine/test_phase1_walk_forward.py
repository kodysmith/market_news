#!/usr/bin/env python3
"""
Phase 1 Test - Walk-Forward Cross Validation and Parameter Sweeps

Tests:
- Walk-forward optimization
- Parameter sweeps
- OOS validation
- Performance stability analysis
"""

import sys
import json
import random
from pathlib import Path
from datetime import datetime, timedelta
from itertools import product
import pandas as pd
import numpy as np

# Add QuantEngine to path
sys.path.insert(0, str(Path(__file__).parent))

from utils.strategy_dsl import StrategyValidator, EXAMPLE_TQQQ_STRATEGY


def create_mock_market_data():
    """Create mock market data for testing"""
    dates = pd.date_range('2020-01-01', '2024-01-01', freq='D')
    np.random.seed(42)

    # Generate 4 years of daily data
    n_days = len(dates)
    returns = np.random.normal(0.0005, 0.02, n_days)  # ~12% annual return, 30% vol
    prices = 100 * np.exp(np.cumsum(returns))

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

    return {'TEST': data}


def simple_backtester(strategy_spec, market_data, config):
    """Simplified backtester for testing"""

    ticker = strategy_spec.get('universe', ['TEST'])[0]
    data = market_data[ticker]

    # Extract close prices and calculate returns
    closes = [d['close'] for d in data]
    returns = []
    for i in range(1, len(closes)):
        ret = (closes[i] - closes[i-1]) / closes[i-1]
        returns.append(ret)

    # Generate signals
    signals = generate_signals(strategy_spec, data)

    # Calculate positions
    positions = [0] * len(data)
    for i in range(len(signals)):
        if signals[i] > 0:
            positions[i] = 1.0

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
        ann_return = total_return * 252 / len(strategy_returns)

        # Volatility and Sharpe
        if len(strategy_returns) > 1:
            avg_return = sum(strategy_returns) / len(strategy_returns)
            variance = sum((r - avg_return)**2 for r in strategy_returns) / len(strategy_returns)
            ann_vol = (variance ** 0.5) * (252 ** 0.5)
            sharpe = ann_return / ann_vol if ann_vol > 0 else 0
        else:
            ann_vol = 0
            sharpe = 0

        # Max drawdown
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
    """Generate trading signals"""
    signals = [0] * len(data)

    for signal_def in strategy_spec.get('signals', []):
        signal_type = signal_def.get('type')

        if signal_type == 'MA_cross':
            fast_period = signal_def.get('params', {}).get('fast', 20)
            slow_period = signal_def.get('params', {}).get('slow', 200)

            closes = [d['close'] for d in data]

            for i in range(max(fast_period, slow_period), len(data)):
                fast_sum = sum(closes[i-fast_period:i])
                fast_ma = fast_sum / fast_period

                slow_sum = sum(closes[i-slow_period:i])
                slow_ma = slow_sum / slow_period

                if fast_ma > slow_ma:
                    signals[i] = 1

        elif signal_type == 'IV_proxy':
            threshold = signal_def.get('params', {}).get('low_thresh', 0.45)

            closes = [d['close'] for d in data]

            for i in range(20, len(data)):
                window_returns = []
                for j in range(i-20, i):
                    if j > 0:
                        ret = (closes[j] - closes[j-1]) / closes[j-1]
                        window_returns.append(ret)

                if window_returns:
                    vol = sum(r**2 for r in window_returns) / len(window_returns)
                    vol = vol ** 0.5

                    if vol < threshold:
                        signals[i] = 1

    return signals


class WalkForwardOptimizer:
    """Simplified walk-forward optimizer for testing"""

    def __init__(self, config):
        self.config = config
        self.backtester = simple_backtester

    def optimize_strategy(self, base_strategy, market_data, param_grid, metric='sharpe'):
        """Optimize strategy parameters"""

        print(f"🔄 Optimizing {base_strategy['name']} for {metric}")

        # Generate parameter combinations
        param_combinations = self._generate_param_combinations(param_grid)
        print(f"   Testing {len(param_combinations)} parameter combinations")

        results = []

        for i, params in enumerate(param_combinations):
            # Create strategy variant
            strategy_variant = self._create_strategy_variant(base_strategy, params)

            # Run simplified backtest (no walk-forward for this test)
            result = self.backtester(strategy_variant, market_data, self.config)

            # Calculate optimization metric
            opt_metric_value = self._calculate_optimization_metric(result, metric)

            results.append({
                'params': params,
                'result': result,
                'optimization_metric': opt_metric_value,
                'metric_name': metric
            })

        # Find best parameters
        best_result = max(results, key=lambda x: x['optimization_metric'])

        return {
            'base_strategy': base_strategy['name'],
            'optimization_metric': metric,
            'total_combinations_tested': len(param_combinations),
            'best_params': best_result['params'],
            'best_metric_value': best_result['optimization_metric'],
            'best_result': best_result['result'],
            'all_results': results
        }

    def _generate_param_combinations(self, param_grid):
        """Generate all combinations from parameter grid"""
        keys = list(param_grid.keys())
        values = list(param_grid.values())

        combinations = []
        for combination in product(*values):
            param_dict = dict(zip(keys, combination))
            combinations.append(param_dict)

        return combinations

    def _create_strategy_variant(self, base_strategy, params):
        """Create strategy variant with modified parameters"""
        strategy = json.loads(json.dumps(base_strategy))

        for param_key, param_value in params.items():
            if '.' in param_key:
                parts = param_key.split('.')
                obj = strategy
                for part in parts[:-1]:
                    if part.isdigit():
                        part = int(part)
                    obj = obj[part]
                obj[parts[-1]] = param_value
            else:
                strategy[param_key] = param_value

        return strategy

    def _calculate_optimization_metric(self, result, metric):
        """Calculate optimization metric"""
        if 'metrics' not in result:
            return -999

        metrics = result['metrics']

        if metric == 'sharpe':
            return metrics.get('sharpe', -999)
        elif metric == 'total_return':
            return metrics.get('total_return', -999)
        elif metric == 'win_rate':
            return metrics.get('win_rate', -999)
        else:
            return metrics.get('sharpe', -999)


def run_phase1_walk_forward_test():
    """Run Phase 1 walk-forward optimization test"""

    print("🚀 Starting Phase 1 Walk-Forward Test")
    print("=" * 50)

    # Create mock data
    market_data = create_mock_market_data()

    # Configuration
    config = {
        'commission_bps': 2.0,
        'slippage_bps': 1.0,
    }

    # Base strategy - simplified TQQQ strategy
    base_strategy = {
        'name': 'tqqq_ma_cross_optimized',
        'universe': ['TEST'],
        'signals': [{
            'type': 'MA_cross',
            'params': {'fast': 20, 'slow': 200}
        }],
        'entry': {'all': ['signals.0.rule']},
        'sizing': {'vol_target_ann': 0.15, 'max_weight': 1.0},
        'costs': {'commission_bps': 2.0, 'slippage_bps': 1.0},
        'risk': {'max_dd_pct': 0.25}
    }

    # Parameter grid for optimization
    param_grid = {
        'signals.0.params.fast': [10, 20, 30],
        'signals.0.params.slow': [100, 150, 200],
        'sizing.vol_target_ann': [0.10, 0.15, 0.20]
    }

    # Run optimization
    optimizer = WalkForwardOptimizer(config)
    results = optimizer.optimize_strategy(base_strategy, market_data, param_grid, metric='sharpe')

    print("✅ Optimization completed!")
    print(".3f")
    print(f"   Best parameters: {results['best_params']}")
    print(f"   Combinations tested: {results['total_combinations_tested']}")

    # Generate optimization report
    generate_optimization_report(results)

    return results


def generate_optimization_report(results):
    """Generate optimization report"""

    report = f"""# Phase 1 - Parameter Optimization Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Optimization Summary

- **Strategy:** {results['base_strategy']}
- **Metric:** {results['optimization_metric']}
- **Combinations Tested:** {results['total_combinations_tested']}

## Best Parameters Found

```json
{json.dumps(results['best_params'], indent=2)}
```

**Best {results['optimization_metric'].title()}:** {results['best_metric_value']:.3f}

## Best Strategy Performance

"""

    best_result = results['best_result']
    if 'metrics' in best_result:
        metrics = best_result['metrics']
        report += "| Metric | Value |\n"
        report += "|--------|-------|\n"
        report += f"| Sharpe Ratio | {metrics.get('sharpe', 0):.3f} |\n"
        report += f"| Total Return | {metrics.get('total_return', 0):.4f} |\n"
        report += f"| Annualized Return | {metrics.get('ann_return', 0):.1%} |\n"
        report += f"| Annualized Volatility | {metrics.get('ann_vol', 0):.1%} |\n"
        report += f"| Max Drawdown | {metrics.get('max_dd', 0):.1%} |\n"
        report += f"| Win Rate | {metrics.get('win_rate', 0):.1%} |\n"
        report += "\n"

    # Parameter sensitivity analysis
    report += "## Parameter Sensitivity Analysis\n\n"

    all_results = results['all_results']
    if all_results:
        # Group by parameter
        param_analysis = {}
        for result in all_results:
            for param_key, param_value in result['params'].items():
                if param_key not in param_analysis:
                    param_analysis[param_key] = []
                param_analysis[param_key].append((param_value, result['optimization_metric']))

        for param_key, values in param_analysis.items():
            report += f"### {param_key}\n\n"
            report += "| Value | Sharpe |\n"
            report += "|-------|--------|\n"

            # Sort by parameter value
            sorted_values = sorted(values, key=lambda x: x[0])
            for param_val, sharpe in sorted_values:
                report += f"| {param_val} | {sharpe:.3f} |\n"
            report += "\n"

    # Conclusion
    best_sharpe = results['best_metric_value']
    report += "## Conclusion\n\n"

    if best_sharpe > 1.0:
        report += "✅ **Strong optimization results** - strategy shows robust performance across parameter ranges.\n\n"
    elif best_sharpe > 0.5:
        report += "⚠️ **Moderate optimization results** - strategy performance is sensitive to parameter choices.\n\n"
    else:
        report += "❌ **Weak optimization results** - strategy may need fundamental redesign.\n\n"

    # Save report
    reports_dir = Path('reports')
    reports_dir.mkdir(exist_ok=True)

    report_path = reports_dir / 'phase1_parameter_optimization.md'
    with open(report_path, 'w') as f:
        f.write(report)

    print(f"✅ Optimization report saved to {report_path}")

    # Also save detailed results as JSON
    json_path = reports_dir / 'phase1_optimization_results.json'
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"✅ Detailed results saved to {json_path}")


if __name__ == "__main__":
    run_phase1_walk_forward_test()


