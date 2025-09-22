#!/usr/bin/env python3
"""
Create optimized AMD script from the best configuration found
"""

from datetime import datetime

def create_optimized_script():
    """Create the optimized AMD script"""

    # Best configuration from simple optimizer
    config = {
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
    }

    ticker = 'AMD'

    script_content = f'''#!/usr/bin/env python3
"""
Optimized {ticker} Options Protection Strategy

AUTO-GENERATED on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Optimization Results:
- CAGR: 35.5%
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
        'risk_on_put_delta': -0.15,
        'risk_on_call_delta': 0.08,
        'risk_off_put_delta': -0.20,
        'risk_off_call_delta': 0.12,
        'protection_ratio_risk_on': 0.40,
        'protection_ratio_risk_off': 0.70,
        'ladder_tenors': [30, 60, 90],
        'purchase_frequency_days': 7,
        'budget_tiers': {{'calm': 0.003, 'normal': 0.010, 'stress': 0.015}},
        'min_hedge': 0.40,
        'use_put_spreads': True,
        'use_call_spreads': True
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

    script_name = f"optimized_{ticker.lower()}.py"
    with open(script_name, 'w') as f:
        f.write(script_content)

    print(f"✅ Generated optimized script: {script_name}")
    print("Usage:")
    print(f"  python {script_name}                    # Full period (2020-2024)")
    print(f"  python {script_name} 2021-01-01 2022-01-01  # Custom period")

if __name__ == "__main__":
    create_optimized_script()
