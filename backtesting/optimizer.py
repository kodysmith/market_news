#!/usr/bin/env python3
"""
Universal Options Strategy Optimizer

Usage:
    python optimizer.py AMD    # Optimize for AMD and create optimized_amd.py
    python optimizer.py NVDA   # Optimize for NVDA and create optimized_nvda.py
    python optimizer.py TQQQ   # Optimize for TQQQ and create optimized_tqqq.py
"""

import sys
import os
from datetime import datetime

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backtest_options_strategy import OptionsProtectionStrategy

def run_simple_optimization(ticker='AMD', n_configs=3):
    """Run simple optimization with predefined configs"""
    print(f"🚀 Simple optimization for {ticker}")

    # Load data once
    strategy = OptionsProtectionStrategy(ticker=ticker, start_date='2020-01-01', end_date='2024-01-01')
    asset_data = strategy.asset_data
    qqq_data = strategy.qqq_data

    # Predefined configurations to test
    configs = [
        {
            'risk_on_put_delta': -0.15,
            'risk_on_call_delta': 0.08,
            'risk_off_put_delta': -0.20,
            'risk_off_call_delta': 0.12,
            'protection_ratio_risk_on': 0.40,
            'protection_ratio_risk_off': 0.70,
            'ladder_tenors': [30, 60, 90],
            'purchase_frequency_days': 7,
            'budget_tiers': {'calm': 0.003, 'normal': 0.010, 'stress': 0.015},
            'min_hedge': 0.40,
            'use_put_spreads': True,
            'use_call_spreads': True
        },
        {
            'risk_on_put_delta': -0.12,
            'risk_on_call_delta': 0.10,
            'risk_off_put_delta': -0.25,
            'risk_off_call_delta': 0.08,
            'protection_ratio_risk_on': 0.50,
            'protection_ratio_risk_off': 0.80,
            'ladder_tenors': [35, 70, 105],
            'purchase_frequency_days': 10,
            'budget_tiers': {'calm': 0.005, 'normal': 0.012, 'stress': 0.020},
            'min_hedge': 0.50,
            'use_put_spreads': False,
            'use_call_spreads': False
        },
        {
            'risk_on_put_delta': -0.10,
            'risk_on_call_delta': 0.12,
            'risk_off_put_delta': -0.15,
            'risk_off_call_delta': 0.10,
            'protection_ratio_risk_on': 0.30,
            'protection_ratio_risk_off': 0.60,
            'ladder_tenors': [30, 60, 90],
            'purchase_frequency_days': 14,
            'budget_tiers': {'calm': 0.008, 'normal': 0.015, 'stress': 0.025},
            'min_hedge': 0.30,
            'use_put_spreads': True,
            'use_call_spreads': False
        }
    ]

    results = []
    for i, config in enumerate(configs[:n_configs]):
        print(f"Testing config {i+1}/{n_configs}")
        try:
            # Create fresh strategy for each test
            test_strategy = OptionsProtectionStrategy(
                ticker=ticker,
                initial_capital=1000000,
                purchase_frequency_days=config['purchase_frequency_days'],
                start_date='2020-01-01',
                end_date='2024-01-01',
                asset_data=asset_data,
                qqq_data=qqq_data
            )

            # Apply config
            test_strategy.config.update(config)

            # Run backtest
            result = test_strategy.run_backtest()

            score = result['total_return'] - 0.1 * (1 - (result['final_value'] / (result['final_value'] + 1e-9)))  # Rough DD penalty
            results.append({
                'config': config,
                'total_return': result['total_return'],
                'final_value': result['final_value'],
                'score': score
            })

            print(f"Return: {result['total_return']:.1%}")

        except Exception as e:
            print(f"Error in config {i+1}: {e}")
            continue

    if not results:
        print("No valid results")
        return None

    # Find best
    best = max(results, key=lambda x: x['score'])
    print(f"\n🏆 Best Configuration:")
    print(f"Return: {best['total_return']:.1%}")

    return best

def generate_optimized_script(ticker, best_result):
    """Generate an optimized script for the best configuration"""
    config = best_result['config']
    script_name = f"optimized_{ticker.lower()}.py"

    script_content = f'''#!/usr/bin/env python3
"""
Optimized {ticker} Options Protection Strategy

AUTO-GENERATED on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Optimization Results:
- CAGR: {best_result['total_return']:.1%}
- Based on 2020-2024 backtest data
"""

from backtest_options_strategy import OptionsProtectionStrategy
from datetime import datetime
import sys

def run_optimized_backtest(start_date='2020-01-01', end_date='2024-01-01'):
    """Run the optimized backtest for {ticker}"""
    print(f"🚀 Running Optimized {ticker} Strategy")
    print("=" * 50)

    # Optimized configuration
    config = {{
        'risk_on_put_delta': {config['risk_on_put_delta']},
        'risk_on_call_delta': {config['risk_on_call_delta']},
        'risk_off_put_delta': {config['risk_off_put_delta']},
        'risk_off_call_delta': {config['risk_off_call_delta']},
        'protection_ratio_risk_on': {config['protection_ratio_risk_on']},
        'protection_ratio_risk_off': {config['protection_ratio_risk_off']},
        'ladder_tenors': {config['ladder_tenors']},
        'purchase_frequency_days': {config['purchase_frequency_days']},
        'budget_tiers': {config['budget_tiers']},
        'min_hedge': {config['min_hedge']},
        'use_put_spreads': {config['use_put_spreads']},
        'use_call_spreads': {config['use_call_spreads']}
    }}

    # Create strategy
    strategy = OptionsProtectionStrategy(
        ticker='{ticker}',
        initial_capital=1000000,
        purchase_frequency_days=config['purchase_frequency_days'],
        start_date=start_date,
        end_date=end_date
    )

    # Apply optimized configuration
    strategy.config.update(config)

    # Run backtest
    results = strategy.run_backtest()

    # Print results
    strategy.print_summary(results)

    # Plot if not in headless mode
    try:
        strategy.plot_results(results)
    except:
        print("⚠️  Plotting not available (headless environment)")

    return results

if __name__ == "__main__":
    # Allow custom date ranges via command line
    start_date = sys.argv[1] if len(sys.argv) > 1 else '2020-01-01'
    end_date = sys.argv[2] if len(sys.argv) > 2 else '2024-01-01'

    print(f"Running optimized {ticker} strategy from {{start_date}} to {{end_date}}")
    run_optimized_backtest(start_date, end_date)
'''

    with open(script_name, 'w') as f:
        f.write(script_content)

    print(f"✅ Generated optimized script: {script_name}")
    print("Usage:")
    print(f"  python {script_name}                    # Full period (2020-2024)")
    print(f"  python {script_name} 2021-01-01 2022-01-01  # Custom period")

    return script_name

def main():
    if len(sys.argv) < 2:
        print("Usage: python optimizer.py TICKER")
        print("Example: python optimizer.py AMD")
        sys.exit(1)

    ticker = sys.argv[1].upper()

    # Run optimization
    best_result = run_simple_optimization(ticker)

    if best_result:
        # Generate optimized script
        script_file = generate_optimized_script(ticker, best_result)

        print(f"\n🎉 Optimization complete for {ticker}!")
        print(f"📄 Run: python {script_file}")
        print(f"📊 Results: CAGR {best_result['total_return']:.1%}")

if __name__ == "__main__":
    main()