#!/usr/bin/env python3
"""
Simple sequential optimizer for testing
"""

import pandas as pd
import numpy as np
import sys
from backtest_options_strategy import OptionsProtectionStrategy

def simple_optimize(ticker='AMD', n_trials=3):
    """Simple sequential optimization"""
    print(f"🚀 Simple optimization for {ticker}")

    # Load data once
    strategy = OptionsProtectionStrategy(ticker=ticker, start_date='2020-01-01', end_date='2024-01-01')
    asset_data = strategy.asset_data
    qqq_data = strategy.qqq_data

    # Simple parameter grid
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
    for i, config in enumerate(configs[:n_trials]):
        print(f"Testing config {i+1}/{n_trials}")
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
    print("\n🏆 Best Configuration:")
    print(f"Return: {best['total_return']:.1%}")

    return best

if __name__ == "__main__":
    ticker = sys.argv[1] if len(sys.argv) > 1 else 'AMD'
    best_result = simple_optimize(ticker)
    if best_result:
        print(f"Best config: {best_result['config']}")