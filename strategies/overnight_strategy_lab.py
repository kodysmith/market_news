#!/usr/bin/env python3
"""
Overnight Walk-Forward Strategy Lab for TQQQ Options Strategy

Walk-forward backtesting: Train on in-sample, test on out-of-sample windows.
Sweeps parameters to find robust configs across market regimes.

Usage:
    python overnight_strategy_lab.py --start 2020-01-01 --end 2024-01-01 --wf 18 6 --trials 200 --min-hedge 0.6
"""

import pandas as pd
import numpy as np
from itertools import product
from datetime import datetime, timedelta
from typing import Dict, Any
import argparse
import json
import sys
import os

# Import your strategy
from tqqq_options_strategy import TQQQOptionsStrategy


def compute_metrics(dates, values, initial_capital, start_date, end_date):
    series = pd.Series(values, index=pd.to_datetime(dates))
    start = pd.to_datetime(start_date).tz_localize(series.index.tz)
    end = pd.to_datetime(end_date).tz_localize(series.index.tz)
    series = series[(series.index >= start) & (series.index <= end)]
    if len(series) == 0:
        return {'final_value': initial_capital, 'total_return': 0.0, 'cagr': 0.0, 'max_drawdown': 0.0, 'vol': 0.0, 'sharpe': 0.0}
    years = (series.index[-1] - series.index[0]).days / 365.0
    final_value = float(series.iloc[-1])
    total_return = (final_value - initial_capital) / initial_capital
    cagr = (final_value / initial_capital) ** (1.0 / max(years, 1e-6)) - 1.0
    rolling_max = series.expanding().max()
    drawdown = (series - rolling_max) / rolling_max
    max_dd = float(drawdown.min()) if not drawdown.empty else 0.0
    vol = series.pct_change().std() * np.sqrt(252) if len(series) > 1 else 0.0
    returns = series.pct_change().dropna()
    sharpe = (returns.mean() / (returns.std() + 1e-9)) * np.sqrt(252) if len(returns) and returns.std() > 0 else 0.0
    return {'final_value': final_value, 'total_return': total_return, 'cagr': cagr, 'max_drawdown': abs(max_dd), 'vol': float(vol), 'sharpe': float(sharpe)}


def objective(metrics: Dict[str, Any], dd_cap: float = 0.20, penalty: float = 2.0) -> float:
    cagr = metrics['cagr']
    max_dd = metrics['max_drawdown']
    over = max(0.0, max_dd - dd_cap)
    return cagr - penalty * over


def walk_forward_windows(start_date: str, end_date: str, in_sample_months: int, out_sample_months: int):
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    windows = []
    current_start = start
    while current_start + pd.DateOffset(months=in_sample_months + out_sample_months) <= end:
        in_end = current_start + pd.DateOffset(months=in_sample_months)
        out_end = in_end + pd.DateOffset(months=out_sample_months)
        windows.append((current_start.strftime('%Y-%m-%d'), in_end.strftime('%Y-%m-%d'), out_end.strftime('%Y-%m-%d')))
        current_start = out_end
    return windows


def main():
    parser = argparse.ArgumentParser(description='Overnight Walk-Forward Strategy Lab')
    parser.add_argument('--start', type=str, default='2020-01-01', help='Start date')
    parser.add_argument('--end', type=str, default='2024-01-01', help='End date')
    parser.add_argument('--wf', type=int, nargs=2, default=[18, 6], help='Walk-forward: in-sample months, out-sample months')
    parser.add_argument('--trials', type=int, default=200, help='Trials per window')
    parser.add_argument('--min-hedge', type=float, default=0.6, help='Min hedge ratio')
    parser.add_argument('--out', type=str, default='overnight_trials.csv', help='Output CSV')
    args = parser.parse_args()

    print(f"🔎 Starting walk-forward lab: {args.start} to {args.end}, WF {args.wf[0]}/{args.wf[1]} months, {args.trials} trials/window")

    # Pre-fetch data once
    base = TQQQOptionsStrategy(initial_capital=1_000_000, start_date=args.start, end_date=args.end)
    tqqq_data = base.tqqq_data.copy()
    qqq_data = base.qqq_data.copy()

    # HIGH-RETURN OPTIMIZATION GRID - Target 20% CAGR with protection
    grid = {
        # Aggressive put protection for tail risk
        'risk_on_put_delta': [-0.15, -0.12, -0.10],  # Shallower puts = less drag
        'risk_on_call_delta': [0.08, 0.12, 0.15],   # Higher delta calls = more premium
        'risk_off_put_delta': [-0.30, -0.25, -0.20], # Deeper puts when needed
        'risk_off_call_delta': [0.08, 0.12],         # Higher calls in OFF

        # Reduce protection ratios for more capital in shares
        'protection_ratio_risk_on': [0.30, 0.40, 0.50], # Less hedging = more upside
        'protection_ratio_risk_off': [0.70, 0.80],      # Still protected in downtrends

        # Optimize ladder for better premium capture
        'ladder_tenors': [(30,60,90), (35,70,105)],

        # More frequent DCA for compounding
        'purchase_frequency_days': [7, 10, 14],  # Weekly/bi-weekly buys

        # More aggressive budgets to allow higher premium spend
        'budget_tiers_calm': [0.003, 0.005, 0.008],  # 3x higher budgets
        'budget_tiers_normal': [0.008, 0.010],
        'budget_tiers_stress': [0.015, 0.020],

        # Mix naked vs spreads for higher returns
        'use_put_spreads': [False, True],   # Naked puts for deeper protection when needed
        'use_call_spreads': [False, True],  # Naked calls for unlimited upside potential

        # Lower minimum hedge for more aggressive exposure
        'min_hedge': [0.40, 0.50],  # Allow more unhedged exposure
    }

    keys = list(grid.keys())
    combos = list(product(*[grid[k] for k in keys]))
    np.random.seed(42)
    selected_combos = np.random.choice(len(combos), size=min(args.trials, len(combos)), replace=False)
    combos = [combos[i] for i in selected_combos]

    windows = walk_forward_windows(args.start, args.end, args.wf[0], args.wf[1])
    print(f"🧮 {len(windows)} windows, {len(combos)} combos/window")

    all_results = []
    for win_idx, (train_start, train_end, test_start) in enumerate(windows):
        print(f"\n=== Window {win_idx+1}/{len(windows)}: {train_start} → {train_end} (train), {test_start} (test) ===")

        window_results = []
        for combo in combos:
            params = dict(zip(keys, combo))
            budget_tiers = {
                'calm': params.pop('budget_tiers_calm'),
                'normal': params.pop('budget_tiers_normal'),
                'stress': params.pop('budget_tiers_stress'),
            }
            ladder = tuple(params['ladder_tenors'])

            # Train on in-sample
            strat_train = TQQQOptionsStrategy(
                initial_capital=1_000_000, start_date=train_start, end_date=train_end,
                purchase_frequency_days=params['purchase_frequency_days'],
                tqqq_data=tqqq_data, qqq_data=qqq_data,
            )
            overrides = params.copy()
            overrides['ladder_tenors'] = list(ladder)
            overrides['budget_tiers'] = budget_tiers
            strat_train.set_config(**overrides)

            # Run backtest on train data
            results_train = strat_train.run_backtest()
            metrics_train = compute_metrics(results_train['dates'], results_train['portfolio_values'], results_train['initial_capital'], train_start, train_end)

            # Test on out-of-sample
            strat_test = TQQQOptionsStrategy(
                initial_capital=1_000_000, start_date=train_start, end_date=test_start,
                purchase_frequency_days=params['purchase_frequency_days'],
                tqqq_data=tqqq_data, qqq_data=qqq_data,
            )
            strat_test.set_config(**overrides)
            results_test = strat_test.run_backtest()
            metrics_test = compute_metrics(results_test['dates'], results_test['portfolio_values'], results_test['initial_capital'], train_end, test_start)

            # Score on test (out-of-sample performance)
            score = objective(metrics_test)
            window_results.append({
                'window': win_idx,
                'train_start': train_start,
                'train_end': train_end,
                'test_start': test_start,
                **overrides,
                'budget_tiers': str(budget_tiers),
                'train_cagr': metrics_train['cagr'],
                'train_max_dd': metrics_train['max_drawdown'],
                'test_cagr': metrics_test['cagr'],
                'test_max_dd': metrics_test['max_drawdown'],
                'test_vol': metrics_test['vol'],
                'test_sharpe': metrics_test['sharpe'],
                'score': score,
            })

        # Best in this window
        best_in_window = max(window_results, key=lambda x: x['score'])
        print(f"Best score: {best_in_window['score']:.4f}")

        all_results.extend(window_results)

    # Overall top-5
    df = pd.DataFrame(all_results)
    df_sorted = df.sort_values('score', ascending=False)
    top5 = df_sorted.head(5)

    print("\n🏁 Top-5 Overall (by OOS score):")
    display_cols = ['score', 'test_cagr', 'test_max_dd', 'test_sharpe', 'risk_on_put_delta', 'risk_on_call_delta', 'protection_ratio_risk_on', 'purchase_frequency_days', 'use_put_spreads', 'use_call_spreads']
    print(top5[display_cols].to_string(index=False))

    print("\n📋 Params of Top-5:")
    for i, row in top5.iterrows():
        print(f"Top {i+1}:")
        params_dict = {k: row[k] for k in ['risk_on_put_delta', 'risk_on_call_delta', 'risk_off_put_delta', 'risk_off_call_delta', 'protection_ratio_risk_on', 'protection_ratio_risk_off', 'ladder_tenors', 'purchase_frequency_days', 'budget_tiers', 'use_put_spreads', 'use_call_spreads', 'min_hedge']}
        print(json.dumps(params_dict, indent=2))

    # Save all
    df.to_csv(args.out, index=False)
    print(f"\n💾 Saved {len(df)} trials to {args.out}")


if __name__ == '__main__':
    main()