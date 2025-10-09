#!/usr/bin/env python3
"""
Runner script for SQQQ-QQQ Wheel Strategy Backtest

This script runs the backtest and generates comprehensive performance reports,
visualizations, and trade logs.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sqqq_qqq_wheel_strategy import SQQQQQQWheelStrategy

def plot_results(results: dict, save_path: str = None):
    """
    Create comprehensive visualization of backtest results
    
    Args:
        results: Results dictionary from backtest
        save_path: Optional path to save figure
    """
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # 1. Portfolio Value Over Time
    ax1 = fig.add_subplot(gs[0, :])
    dates = results['dates']
    portfolio_values = results['portfolio_values']
    
    ax1.plot(dates, portfolio_values, linewidth=2, label='Strategy Portfolio', color='#2E86AB')
    ax1.axhline(y=results['initial_capital'], color='gray', linestyle='--', 
                alpha=0.5, label='Initial Capital')
    ax1.fill_between(dates, results['initial_capital'], portfolio_values, 
                     where=np.array(portfolio_values) >= results['initial_capital'],
                     alpha=0.3, color='green', label='Profit')
    ax1.fill_between(dates, results['initial_capital'], portfolio_values,
                     where=np.array(portfolio_values) < results['initial_capital'],
                     alpha=0.3, color='red', label='Loss')
    
    ax1.set_title('Portfolio Value Over Time', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Portfolio Value ($)', fontsize=12)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    # 2. Drawdown Chart
    ax2 = fig.add_subplot(gs[1, 0])
    portfolio_series = pd.Series(portfolio_values, index=dates)
    rolling_max = portfolio_series.expanding().max()
    drawdown = (portfolio_series - rolling_max) / rolling_max
    
    ax2.fill_between(dates, drawdown * 100, 0, alpha=0.3, color='red')
    ax2.plot(dates, drawdown * 100, linewidth=1.5, color='darkred')
    ax2.set_title('Drawdown', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Drawdown (%)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=results['max_drawdown'] * 100, color='red', linestyle='--', 
                alpha=0.7, label=f"Max DD: {results['max_drawdown']:.1%}")
    ax2.legend()
    
    # 3. Monthly Returns Heatmap
    ax3 = fig.add_subplot(gs[1, 1])
    returns = portfolio_series.pct_change().dropna()
    monthly_returns = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
    
    # Create monthly returns matrix
    monthly_df = pd.DataFrame({
        'Year': monthly_returns.index.year,
        'Month': monthly_returns.index.month,
        'Return': monthly_returns.values * 100
    })
    
    if len(monthly_df) > 0:
        pivot_table = monthly_df.pivot(index='Month', columns='Year', values='Return')
        
        sns.heatmap(pivot_table, annot=True, fmt='.1f', cmap='RdYlGn', center=0,
                   ax=ax3, cbar_kws={'label': 'Return (%)'})
        ax3.set_title('Monthly Returns Heatmap (%)', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Month', fontsize=12)
        ax3.set_xlabel('Year', fontsize=12)
    
    # 4. Trade Distribution
    ax4 = fig.add_subplot(gs[2, 0])
    trades_df = pd.DataFrame(results['trades'])
    
    if len(trades_df) > 0:
        action_counts = trades_df['action'].value_counts()
        colors = plt.cm.Set3(range(len(action_counts)))
        
        wedges, texts, autotexts = ax4.pie(action_counts.values, labels=action_counts.index,
                                           autopct='%1.1f%%', colors=colors, startangle=90)
        ax4.set_title('Trade Distribution', fontsize=14, fontweight='bold')
        
        # Make percentage text more readable
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
    
    # 5. Performance Metrics Box
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.axis('off')
    
    metrics_text = f"""
    PERFORMANCE METRICS
    {'='*40}
    
    Total Return:           {results['total_return']:>10.1%}
    CAGR:                   {results['cagr']:>10.1%}
    Sharpe Ratio:           {results['sharpe_ratio']:>10.2f}
    Max Drawdown:           {results['max_drawdown']:>10.1%}
    
    TRADING SUMMARY
    {'='*40}
    
    Total Trades:           {results['total_trades']:>10,}
    Put Assignments:        {results['put_assignments']:>10,}
    Call Assignments:       {results['call_assignments']:>10,}
    
    PREMIUM & P&L
    {'='*40}
    
    Total Premium:          ${results['total_premium_collected']:>10,.0f}
    Put Premium:            ${results['put_premium']:>10,.0f}
    Call Premium:           ${results['call_premium']:>10,.0f}
    SQQQ Hedge P&L:         ${results['sqqq_hedge_pnl']:>10,.0f}
    
    FINAL POSITIONS
    {'='*40}
    
    Final Value:            ${results['final_value']:>10,.0f}
    Cash:                   ${results['final_cash']:>10,.0f}
    QQQ Shares:             {results['final_qqq_shares']:>10,}
    """
    
    ax5.text(0.1, 0.95, metrics_text, transform=ax5.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.suptitle('SQQQ-QQQ Wheel Strategy Backtest Results', 
                fontsize=16, fontweight='bold', y=0.995)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Chart saved to: {save_path}")
    
    plt.tight_layout()
    plt.show()


def export_trades_csv(results: dict, filepath: str = 'sqqq_qqq_trades.csv'):
    """
    Export detailed trade log to CSV
    
    Args:
        results: Results dictionary from backtest
        filepath: Path to save CSV file
    """
    trades_df = pd.DataFrame(results['trades'])
    
    if len(trades_df) > 0:
        # Convert date column to string for better CSV formatting
        trades_df['date'] = pd.to_datetime(trades_df['date']).dt.strftime('%Y-%m-%d')
        
        # Reorder columns for better readability
        column_order = ['date', 'action', 'symbol']
        other_cols = [col for col in trades_df.columns if col not in column_order]
        trades_df = trades_df[column_order + other_cols]
        
        trades_df.to_csv(filepath, index=False)
        print(f"📝 Trade log exported to: {filepath}")
        print(f"   Total trades: {len(trades_df)}")
    else:
        print("⚠️  No trades to export")


def export_portfolio_values_csv(results: dict, filepath: str = 'sqqq_qqq_portfolio.csv'):
    """
    Export daily portfolio values to CSV
    
    Args:
        results: Results dictionary from backtest
        filepath: Path to save CSV file
    """
    portfolio_df = pd.DataFrame({
        'date': results['dates'],
        'portfolio_value': results['portfolio_values']
    })
    
    # Calculate daily returns and cumulative returns
    portfolio_df['daily_return'] = portfolio_df['portfolio_value'].pct_change()
    portfolio_df['cumulative_return'] = (portfolio_df['portfolio_value'] / results['initial_capital'] - 1)
    
    # Calculate rolling metrics
    portfolio_df['rolling_sharpe_30d'] = (
        portfolio_df['daily_return'].rolling(30).mean() / 
        portfolio_df['daily_return'].rolling(30).std() * np.sqrt(252)
    )
    
    # Calculate drawdown
    rolling_max = portfolio_df['portfolio_value'].expanding().max()
    portfolio_df['drawdown'] = (portfolio_df['portfolio_value'] - rolling_max) / rolling_max
    
    portfolio_df['date'] = pd.to_datetime(portfolio_df['date']).dt.strftime('%Y-%m-%d')
    
    portfolio_df.to_csv(filepath, index=False)
    print(f"💼 Portfolio values exported to: {filepath}")
    print(f"   Total days: {len(portfolio_df)}")


def generate_performance_summary(results: dict, filepath: str = 'sqqq_qqq_summary.txt'):
    """
    Generate a comprehensive text summary of backtest results
    
    Args:
        results: Results dictionary from backtest
        filepath: Path to save summary file
    """
    portfolio_series = pd.Series(results['portfolio_values'], index=results['dates'])
    returns = portfolio_series.pct_change().dropna()
    
    # Calculate additional metrics
    volatility = returns.std() * np.sqrt(252)
    positive_days = (returns > 0).sum()
    negative_days = (returns < 0).sum()
    win_rate = positive_days / (positive_days + negative_days) if (positive_days + negative_days) > 0 else 0
    
    # Best and worst days
    best_day = returns.max()
    best_day_date = returns.idxmax()
    worst_day = returns.min()
    worst_day_date = returns.idxmin()
    
    # Monthly statistics
    monthly_returns = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
    avg_monthly_return = monthly_returns.mean()
    positive_months = (monthly_returns > 0).sum()
    negative_months = (monthly_returns < 0).sum()
    
    # Sortino ratio (using only downside deviation)
    downside_returns = returns[returns < 0]
    downside_std = downside_returns.std() * np.sqrt(252)
    sortino = returns.mean() * 252 / downside_std if downside_std > 0 else 0
    
    # Calmar ratio (CAGR / Max Drawdown)
    calmar = abs(results['cagr'] / results['max_drawdown']) if results['max_drawdown'] != 0 else 0
    
    summary = f"""
{'='*70}
SQQQ-QQQ WHEEL STRATEGY BACKTEST SUMMARY
{'='*70}

BACKTEST PARAMETERS
{'-'*70}
Initial Capital:                ${results['initial_capital']:,.0f}
Backtest Period:                {results['dates'][0].strftime('%Y-%m-%d')} to {results['dates'][-1].strftime('%Y-%m-%d')}
Total Days:                     {len(results['dates'])}
Years:                          {(results['dates'][-1] - results['dates'][0]).days / 365.25:.2f}

PERFORMANCE METRICS
{'-'*70}
Final Portfolio Value:          ${results['final_value']:,.2f}
Total Return:                   {results['total_return']:.2%}
CAGR (Annualized):             {results['cagr']:.2%}
Annualized Volatility:          {volatility:.2%}

Risk-Adjusted Returns:
  Sharpe Ratio:                 {results['sharpe_ratio']:.3f}
  Sortino Ratio:                {sortino:.3f}
  Calmar Ratio:                 {calmar:.3f}

Maximum Drawdown:               {results['max_drawdown']:.2%}

DAILY STATISTICS
{'-'*70}
Best Day:                       {best_day:.2%} on {best_day_date.strftime('%Y-%m-%d')}
Worst Day:                      {worst_day:.2%} on {worst_day_date.strftime('%Y-%m-%d')}
Positive Days:                  {positive_days} ({win_rate:.1%})
Negative Days:                  {negative_days} ({(1-win_rate):.1%})

MONTHLY STATISTICS
{'-'*70}
Average Monthly Return:         {avg_monthly_return:.2%}
Positive Months:                {positive_months}
Negative Months:                {negative_months}

TRADING ACTIVITY
{'-'*70}
Total Trades:                   {results['total_trades']}
Put Assignments:                {results['put_assignments']}
Call Assignments:               {results['call_assignments']}

Trade Breakdown:
"""
    
    # Add trade breakdown
    trades_df = pd.DataFrame(results['trades'])
    if len(trades_df) > 0:
        action_counts = trades_df['action'].value_counts()
        for action, count in action_counts.items():
            summary += f"  {action:30s} {count:>6}\n"
    
    summary += f"""
PREMIUM & PROFIT ANALYSIS
{'-'*70}
Total Premium Collected:        ${results['total_premium_collected']:,.2f}
  Put Premium:                  ${results['put_premium']:,.2f}
  Call Premium:                 ${results['call_premium']:,.2f}

SQQQ Hedge P&L:                 ${results['sqqq_hedge_pnl']:,.2f}

FINAL POSITIONS
{'-'*70}
Cash:                           ${results['final_cash']:,.2f}
QQQ Shares Held:                {results['final_qqq_shares']:,}
Open Put Positions:             {len([t for t in results['trades'] if t['action'] == 'SELL_PUT']) - results['put_assignments']}
Open Call Positions:            {len([t for t in results['trades'] if t['action'] == 'SELL_CALL']) - results['call_assignments']}

{'='*70}
Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*70}
"""
    
    # Write to file
    with open(filepath, 'w') as f:
        f.write(summary)
    
    print(f"📄 Performance summary saved to: {filepath}")
    
    # Also print to console
    print(summary)


def main():
    """Main execution function"""
    print("\n" + "="*70)
    print("SQQQ-QQQ WHEEL STRATEGY BACKTESTER")
    print("="*70 + "\n")
    
    # Create strategy instance
    print("🔧 Initializing strategy...")
    strategy = SQQQQQQWheelStrategy(
        initial_capital=100000,
        max_qqq_allocation=0.75,
        weekly_capital_pct=0.01,
        put_delta_target=-0.30,
        put_dte=14,
        call_delta_target=0.30,
        call_dte=30,
        sqqq_call_dte=30,
        start_date='2020-01-01',
        end_date='2025-01-01'
    )
    
    # Run backtest
    print("\n🚀 Running backtest...\n")
    results = strategy.run_backtest()
    
    # Print summary
    strategy.print_summary(results)
    
    # Generate detailed performance summary
    print("\n📊 Generating performance reports...\n")
    generate_performance_summary(results)
    
    # Export trades
    export_trades_csv(results)
    
    # Export portfolio values
    export_portfolio_values_csv(results)
    
    # Plot results
    print("\n📈 Generating visualizations...\n")
    try:
        plot_results(results, save_path='sqqq_qqq_backtest_results.png')
    except Exception as e:
        print(f"⚠️  Could not generate plot: {e}")
        print("   (This is normal in headless environments)")
    
    print("\n✅ Backtest complete! All reports generated.\n")
    print("Generated files:")
    print("  - sqqq_qqq_summary.txt (Performance summary)")
    print("  - sqqq_qqq_trades.csv (Trade log)")
    print("  - sqqq_qqq_portfolio.csv (Daily portfolio values)")
    print("  - sqqq_qqq_backtest_results.png (Visualizations)\n")


if __name__ == "__main__":
    main()

