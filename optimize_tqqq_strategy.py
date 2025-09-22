#!/usr/bin/env python3
import pandas as pd
import numpy as np
from itertools import product
from datetime import datetime
from typing import Dict, Any

from tqqq_options_strategy import TQQQOptionsStrategy


def compute_metrics(dates, values, initial_capital):
    series = pd.Series(values, index=pd.to_datetime(dates))
    returns = series.pct_change().dropna()
    if len(series) == 0:
        return {
            'final_value': initial_capital,
            'total_return': 0.0,
            'cagr': 0.0,
            'max_drawdown': 0.0,
            'vol': 0.0,
            'sharpe': 0.0,
        }
    years = (series.index[-1] - series.index[0]).days / 365.0
    final_value = float(series.iloc[-1])
    total_return = (final_value - initial_capital) / initial_capital
    cagr = (final_value / initial_capital) ** (1.0 / max(years, 1e-6)) - 1.0
    rolling_max = series.expanding().max()
    drawdown = (series - rolling_max) / rolling_max
    max_dd = float(drawdown.min()) if not drawdown.empty else 0.0
    vol = returns.std() * np.sqrt(252) if len(returns) else 0.0
    sharpe = (returns.mean() / (returns.std() + 1e-9)) * np.sqrt(252) if len(returns) and returns.std() > 0 else 0.0
    return {
        'final_value': final_value,
        'total_return': total_return,
        'cagr': cagr,
        'max_drawdown': abs(max_dd),
        'vol': float(vol),
        'sharpe': float(sharpe),
    }


def objective(metrics: Dict[str, Any], dd_cap: float = 0.20, penalty: float = 2.0) -> float:
    # Maximize CAGR with a penalty if max drawdown exceeds dd_cap
    cagr = metrics['cagr']
    max_dd = metrics['max_drawdown']
    over = max(0.0, max_dd - dd_cap)
    return cagr - penalty * over


def main():
    print("🔎 Preparing baseline data for optimization...")
    # Warm instance to fetch data once
    base = TQQQOptionsStrategy(initial_capital=1_000_000,
                               start_date='2020-01-01', end_date='2024-01-01')
    tqqq_data = base.tqqq_data.copy()
    qqq_data = base.qqq_data.copy()

    # Parameter grid (keep modest for speed)
    grid = {
        'risk_on_put_delta': [-0.20, -0.15],
        'risk_on_call_delta': [0.10, 0.15],
        'risk_off_put_delta': [-0.35, -0.30],
        'risk_off_call_delta': [0.05, 0.07],
        'protection_ratio_risk_on': [0.50, 0.60],
        'protection_ratio_risk_off': [0.80, 0.90],
        'ladder_tenors': [(35, 70, 105), (30, 60, 90)],
        'budget_tiers_calm': [0.001, 0.002],
        'budget_tiers_normal': [0.003],
        'budget_tiers_stress': [0.0075, 0.0100],
        'purchase_frequency_days': [14, 30],
    }

    keys = list(grid.keys())
    combos = list(product(*[grid[k] for k in keys]))
    print(f"🧮 Running grid search over {len(combos)} configurations...")

    rows = []
    for combo in combos:
        params = dict(zip(keys, combo))
        # Build budget_tiers dict from parts
        budget_tiers = {
            'calm': params.pop('budget_tiers_calm'),
            'normal': params.pop('budget_tiers_normal'),
            'stress': params.pop('budget_tiers_stress'),
        }
        ladder = tuple(params['ladder_tenors'])

        strat = TQQQOptionsStrategy(
            initial_capital=1_000_000,
            start_date='2020-01-01', end_date='2024-01-01',
            purchase_frequency_days=params['purchase_frequency_days'],
            tqqq_data=tqqq_data, qqq_data=qqq_data,
        )
        overrides = params.copy()
        overrides['ladder_tenors'] = list(ladder)
        overrides['budget_tiers'] = budget_tiers
        strat.set_config(**overrides)

        results = strat.run_backtest()
        metrics = compute_metrics(results['dates'], results['portfolio_values'], results['initial_capital'])
        score = objective(metrics, dd_cap=0.20, penalty=2.0)

        rows.append({
            **overrides,
            'purchase_frequency_days': params['purchase_frequency_days'],
            **metrics,
            'score': score,
            'final_value': results['final_value'],
            'total_return': results['total_return'],
            'shares_owned': results['shares_owned'],
            'total_trades': results['total_trades'],
        })

    df = pd.DataFrame(rows)
    df_sorted = df.sort_values('score', ascending=False)
    df_sorted.to_csv('tqqq_param_sweep.csv', index=False)
    print("\n🏁 Top 10 configurations by score (CAGR penalized for DD > 20%):")
    cols = [
        'score', 'cagr', 'max_drawdown', 'sharpe', 'final_value',
        'risk_on_put_delta', 'risk_on_call_delta',
        'risk_off_put_delta', 'risk_off_call_delta',
        'protection_ratio_risk_on', 'protection_ratio_risk_off',
        'ladder_tenors', 'purchase_frequency_days',
        'budget_tiers'
    ]
    display_cols = [c for c in cols if c in df_sorted.columns]
    print(df_sorted[display_cols].head(10).to_string(index=False))
    print("\n💾 Full sweep saved to tqqq_param_sweep.csv")


if __name__ == '__main__':
    main()



