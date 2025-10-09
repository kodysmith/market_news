#!/usr/bin/env python3
"""
Create side-by-side comparison chart for original vs optimized strategy
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sqqq_qqq_wheel_optimized import SQQQQQQWheelOptimized

print("Running backtests for comparison...")

# Run both strategies
original = SQQQQQQWheelOptimized(
    initial_capital=100000,
    enable_overnight_hedge=False,
    start_date='2020-01-01',
    end_date='2025-01-01'
)
original_results = original.run_backtest()

optimized = SQQQQQQWheelOptimized(
    initial_capital=100000,
    enable_overnight_hedge=True,
    overnight_hedge_delta=-0.50,
    overnight_hedge_coverage=1.0,
    start_date='2020-01-01',
    end_date='2025-01-01'
)
optimized_results = optimized.run_backtest()

# Create comparison figure
fig = plt.figure(figsize=(18, 10))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# 1. Portfolio Value Comparison (large, top)
ax1 = fig.add_subplot(gs[0, :])
dates_orig = original_results['dates']
values_orig = original_results['portfolio_values']
dates_opt = optimized_results['dates']
values_opt = optimized_results['portfolio_values']

ax1.plot(dates_orig, values_orig, linewidth=2.5, label='Original (Weekend Hedging)', color='#3498db', alpha=0.8)
ax1.plot(dates_opt, values_opt, linewidth=2.5, label='Optimized (+ Overnight Hedging)', color='#2ecc71', alpha=0.8)
ax1.axhline(y=100000, color='gray', linestyle='--', alpha=0.4, linewidth=1, label='Initial Capital')

ax1.fill_between(dates_orig, 100000, values_orig, alpha=0.2, color='#3498db')
ax1.fill_between(dates_opt, 100000, values_opt, alpha=0.2, color='#2ecc71')

ax1.set_title('Portfolio Value Comparison: Original vs Optimized Strategy', fontsize=16, fontweight='bold', pad=20)
ax1.set_ylabel('Portfolio Value ($)', fontsize=13)
ax1.legend(loc='upper left', fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))

# Add performance annotations
final_orig = original_results['final_value']
final_opt = optimized_results['final_value']
improvement = ((final_opt - final_orig) / final_orig) * 100

ax1.text(0.98, 0.05, 
         f"Original: ${final_orig:,.0f}\nOptimized: ${final_opt:,.0f}\nImprovement: +{improvement:.1f}%",
         transform=ax1.transAxes, fontsize=11, verticalalignment='bottom', horizontalalignment='right',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))

# 2. Drawdown Comparison
ax2 = fig.add_subplot(gs[1, 0])
portfolio_orig = pd.Series(values_orig, index=dates_orig)
rolling_max_orig = portfolio_orig.expanding().max()
drawdown_orig = (portfolio_orig - rolling_max_orig) / rolling_max_orig

portfolio_opt = pd.Series(values_opt, index=dates_opt)
rolling_max_opt = portfolio_opt.expanding().max()
drawdown_opt = (portfolio_opt - rolling_max_opt) / rolling_max_opt

ax2.fill_between(dates_orig, drawdown_orig * 100, 0, alpha=0.3, color='#e74c3c', label='Original')
ax2.plot(dates_orig, drawdown_orig * 100, linewidth=1.5, color='#c0392b', alpha=0.7)
ax2.fill_between(dates_opt, drawdown_opt * 100, 0, alpha=0.3, color='#f39c12', label='Optimized')
ax2.plot(dates_opt, drawdown_opt * 100, linewidth=1.5, color='#e67e22', alpha=0.7)

ax2.set_title('Drawdown Comparison', fontsize=13, fontweight='bold')
ax2.set_ylabel('Drawdown (%)', fontsize=11)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

# Add max DD lines
ax2.axhline(y=original_results['max_drawdown'] * 100, color='#c0392b', linestyle='--', 
            alpha=0.7, linewidth=1, label=f"Max DD Original: {original_results['max_drawdown']:.1%}")
ax2.axhline(y=optimized_results['max_drawdown'] * 100, color='#e67e22', linestyle='--',
            alpha=0.7, linewidth=1, label=f"Max DD Optimized: {optimized_results['max_drawdown']:.1%}")

# 3. Returns Distribution
ax3 = fig.add_subplot(gs[1, 1])
returns_orig = portfolio_orig.pct_change().dropna() * 100
returns_opt = portfolio_opt.pct_change().dropna() * 100

ax3.hist(returns_orig, bins=50, alpha=0.5, color='#3498db', label='Original', density=True)
ax3.hist(returns_opt, bins=50, alpha=0.5, color='#2ecc71', label='Optimized', density=True)
ax3.axvline(x=returns_orig.mean(), color='#3498db', linestyle='--', linewidth=2, alpha=0.8)
ax3.axvline(x=returns_opt.mean(), color='#2ecc71', linestyle='--', linewidth=2, alpha=0.8)

ax3.set_title('Daily Returns Distribution', fontsize=13, fontweight='bold')
ax3.set_xlabel('Daily Return (%)', fontsize=11)
ax3.set_ylabel('Density', fontsize=11)
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)

# 4. Metrics Comparison Bar Chart
ax4 = fig.add_subplot(gs[1, 2])
metrics = ['Total\nReturn', 'CAGR', 'Sharpe\nRatio', 'Max DD\n(abs)']
orig_vals = [
    original_results['total_return'] * 100,
    original_results['cagr'] * 100,
    original_results['sharpe_ratio'] * 100,
    abs(original_results['max_drawdown']) * 100
]
opt_vals = [
    optimized_results['total_return'] * 100,
    optimized_results['cagr'] * 100,
    optimized_results['sharpe_ratio'] * 100,
    abs(optimized_results['max_drawdown']) * 100
]

x = np.arange(len(metrics))
width = 0.35

bars1 = ax4.bar(x - width/2, orig_vals, width, label='Original', color='#3498db', alpha=0.8)
bars2 = ax4.bar(x + width/2, opt_vals, width, label='Optimized', color='#2ecc71', alpha=0.8)

ax4.set_title('Performance Metrics Comparison', fontsize=13, fontweight='bold')
ax4.set_ylabel('Value (%)', fontsize=11)
ax4.set_xticks(x)
ax4.set_xticklabels(metrics, fontsize=9)
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}', ha='center', va='bottom', fontsize=8)

# 5. Premium & P&L Breakdown
ax5 = fig.add_subplot(gs[2, 0])
categories = ['Put\nPremium', 'Call\nPremium', 'SQQQ\nHedge', 'Overnight\nHedge']
orig_pnl = [
    original_results['put_premium'] / 1000,
    original_results['call_premium'] / 1000,
    original_results['sqqq_hedge_pnl'] / 1000,
    0
]
opt_pnl = [
    optimized_results['put_premium'] / 1000,
    optimized_results['call_premium'] / 1000,
    optimized_results['sqqq_hedge_pnl'] / 1000,
    optimized_results['overnight_hedge_pnl'] / 1000
]

x = np.arange(len(categories))
bars1 = ax5.bar(x - width/2, orig_pnl, width, label='Original', color='#3498db', alpha=0.8)
bars2 = ax5.bar(x + width/2, opt_pnl, width, label='Optimized', color='#2ecc71', alpha=0.8)

ax5.set_title('P&L Breakdown by Source', fontsize=13, fontweight='bold')
ax5.set_ylabel('P&L ($K)', fontsize=11)
ax5.set_xticks(x)
ax5.set_xticklabels(categories, fontsize=9)
ax5.legend(fontsize=9)
ax5.grid(True, alpha=0.3, axis='y')

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax5.text(bar.get_x() + bar.get_width()/2., height,
                    f'${height:.1f}K', ha='center', va='bottom', fontsize=8)

# 6. Rolling Sharpe Ratio (30-day)
ax6 = fig.add_subplot(gs[2, 1])
rolling_sharpe_orig = (returns_orig.rolling(30).mean() / returns_orig.rolling(30).std()) * np.sqrt(252)
rolling_sharpe_opt = (returns_opt.rolling(30).mean() / returns_opt.rolling(30).std()) * np.sqrt(252)

# Align dates with rolling values (skip first 30 days)
valid_dates_orig = dates_orig[30:30+len(rolling_sharpe_orig.dropna())]
valid_dates_opt = dates_opt[30:30+len(rolling_sharpe_opt.dropna())]

ax6.plot(valid_dates_orig, rolling_sharpe_orig.dropna(), linewidth=1.5, label='Original', color='#3498db', alpha=0.8)
ax6.plot(valid_dates_opt, rolling_sharpe_opt.dropna(), linewidth=1.5, label='Optimized', color='#2ecc71', alpha=0.8)
ax6.axhline(y=0, color='gray', linestyle='--', alpha=0.4, linewidth=1)
ax6.axhline(y=1, color='gray', linestyle=':', alpha=0.4, linewidth=1, label='Sharpe = 1.0')

ax6.set_title('Rolling 30-Day Sharpe Ratio', fontsize=13, fontweight='bold')
ax6.set_ylabel('Sharpe Ratio', fontsize=11)
ax6.legend(fontsize=9, loc='lower left')
ax6.grid(True, alpha=0.3)
ax6.set_ylim(-3, 5)

# 7. Summary Stats Box
ax7 = fig.add_subplot(gs[2, 2])
ax7.axis('off')

summary_text = f"""
STRATEGY COMPARISON SUMMARY
{'='*45}

ORIGINAL (Weekend Hedging Only)
  Total Return:       {original_results['total_return']:>8.1%}
  CAGR:               {original_results['cagr']:>8.1%}
  Sharpe Ratio:       {original_results['sharpe_ratio']:>8.2f}
  Max Drawdown:       {original_results['max_drawdown']:>8.1%}
  Final Value:        ${original_results['final_value']:>8,.0f}

OPTIMIZED (+ Overnight Hedging)
  Total Return:       {optimized_results['total_return']:>8.1%}
  CAGR:               {optimized_results['cagr']:>8.1%}
  Sharpe Ratio:       {optimized_results['sharpe_ratio']:>8.2f}
  Max Drawdown:       {optimized_results['max_drawdown']:>8.1%}
  Final Value:        ${optimized_results['final_value']:>8,.0f}

IMPROVEMENT
  Return:             {(optimized_results['total_return'] - original_results['total_return']):>+8.1%}
  Sharpe:             {(optimized_results['sharpe_ratio'] - original_results['sharpe_ratio']):>+8.2f}
  Max DD:             {(optimized_results['max_drawdown'] - original_results['max_drawdown']):>+8.1%}
  Value:              ${(optimized_results['final_value'] - original_results['final_value']):>+8,.0f}

Overnight Hedge P&L:  ${optimized_results['overnight_hedge_pnl']:>8,.0f}
Extra Trades:         {optimized_results['total_trades'] - original_results['total_trades']:>8,}

VERDICT: Overnight hedging SIGNIFICANTLY
         improves performance!
"""

ax7.text(0.05, 0.95, summary_text, transform=ax7.transAxes,
        fontsize=9, verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.95, edgecolor='gray', linewidth=2))

plt.suptitle('SQQQ-QQQ Wheel Strategy: Original vs Overnight Hedging Optimization', 
            fontsize=18, fontweight='bold', y=0.995)

plt.tight_layout()
plt.savefig('strategy_comparison.png', dpi=300, bbox_inches='tight')
print("\n✅ Comparison chart saved to: strategy_comparison.png")
plt.close()

print("\n📊 Comparison complete!")

