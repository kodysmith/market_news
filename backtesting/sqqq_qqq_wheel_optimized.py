#!/usr/bin/env python3
"""
SQQQ-QQQ Wheel Strategy with Overnight Hedge Optimization

Enhancement: Buy protective puts EOD, sell at open to eliminate overnight risk
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sqqq_qqq_wheel_strategy import SQQQQQQWheelStrategy, bs_put_price, put_delta, strike_for_put_delta
import pandas as pd
import numpy as np


class SQQQQQQWheelOptimized(SQQQQQQWheelStrategy):
    """
    Enhanced wheel strategy with overnight put hedging
    
    New Features:
    - Buy ATM/OTM puts before close to hedge overnight risk
    - Sell puts at open next morning
    - Tracks overnight hedge P&L separately
    """
    
    def __init__(self, 
                 enable_overnight_hedge=True,
                 overnight_hedge_delta=-0.50,  # ATM-ish puts
                 overnight_hedge_dte=30,  # 30 DTE for lower theta decay
                 overnight_hedge_coverage=1.0,  # 100% of exposure
                 **kwargs):
        """
        Initialize optimized strategy
        
        Args:
            enable_overnight_hedge: Enable daily overnight put protection
            overnight_hedge_delta: Delta for overnight puts (-0.50 = ATM)
            overnight_hedge_dte: Days to expiration for overnight puts (30 = lower theta)
            overnight_hedge_coverage: % of QQQ exposure to hedge (1.0 = 100%)
            **kwargs: Pass through to parent class
        """
        super().__init__(**kwargs)
        
        self.enable_overnight_hedge = enable_overnight_hedge
        self.overnight_hedge_delta = overnight_hedge_delta
        self.overnight_hedge_dte = overnight_hedge_dte
        self.overnight_hedge_coverage = overnight_hedge_coverage
        
        # Track overnight hedges separately
        self.overnight_puts = []
        self.overnight_hedge_pnl = 0.0
        
    def _buy_overnight_hedge(self, date, current_price: float, iv: float) -> dict:
        """
        Buy protective puts before close to hedge overnight risk
        Uses 30 DTE options for lower theta decay
        Size based on current QQQ delta exposure
        """
        # Calculate net QQQ delta exposure
        qqq_delta = self._get_net_qqq_delta(current_price, iv, date)
        
        if qqq_delta <= 0:
            return None  # No long exposure to hedge
        
        # Calculate contracts needed (based on coverage ratio)
        delta_to_hedge = qqq_delta * self.overnight_hedge_coverage
        contracts_needed = int(abs(delta_to_hedge) / 100)
        
        if contracts_needed < 1:
            return None
        
        # Solve for strike at target delta (ATM-ish) with 30 DTE
        T = self.overnight_hedge_dte / 365.0
        strike = strike_for_put_delta(current_price, iv, T, self.overnight_hedge_delta)
        
        # Get premium
        premium = bs_put_price(current_price, strike, iv, T)
        
        # Calculate cost
        total_cost = premium * contracts_needed * 100
        commission = self.option_commission * contracts_needed
        
        # Check if we have cash (use small portion of hedge reserve)
        max_overnight_cost = self.initial_capital * 0.02  # Max 2% for overnight hedges
        if total_cost + commission > max_overnight_cost or total_cost + commission > self.cash:
            # Reduce size if needed
            contracts_needed = max(1, int(max_overnight_cost / (premium * 100 + self.option_commission)))
            total_cost = premium * contracts_needed * 100
            commission = self.option_commission * contracts_needed
        
        if total_cost + commission > self.cash:
            return None
        
        # Execute trade
        self.cash -= (total_cost + commission)
        
        position = {
            'date_opened': date,
            'strike': strike,
            'premium': premium,
            'contracts': contracts_needed,
            'dte': self.overnight_hedge_dte,
            'type': 'overnight_put',
            'cost': total_cost + commission
        }
        
        self.overnight_puts.append(position)
        
        trade = {
            'date': date,
            'action': 'BUY_OVERNIGHT_PUT',
            'symbol': 'QQQ',
            'strike': strike,
            'premium': premium,
            'contracts': contracts_needed,
            'dte': self.overnight_hedge_dte,
            'cost': total_cost,
            'commission': commission,
            'delta': self.overnight_hedge_delta,
            'qqq_delta_hedged': qqq_delta
        }
        
        self.trades.append(trade)
        
        return trade
    
    def _sell_overnight_hedge(self, date, current_price: float, iv: float):
        """
        Sell overnight puts at open next morning
        With 30 DTE options, we lose ~1 day of time value
        """
        positions_to_remove = []
        
        for i, p in enumerate(self.overnight_puts):
            days_held = (date - p['date_opened']).days
            
            # Sell at open next day (or if held longer due to weekend)
            if days_held >= 1:
                # Mark to market with remaining time (30 DTE → 29 DTE after 1 day)
                remaining_dte = max(p['dte'] - days_held, 1)
                T = remaining_dte / 365.0
                
                current_value = bs_put_price(current_price, p['strike'], iv, T)
                
                # Sell
                proceeds = current_value * p['contracts'] * 100
                commission = self.option_commission * p['contracts']
                
                self.cash += proceeds - commission
                
                # Calculate P&L
                pnl = proceeds - p['cost']
                self.overnight_hedge_pnl += pnl
                
                self.trades.append({
                    'date': date,
                    'action': 'SELL_OVERNIGHT_PUT',
                    'symbol': 'QQQ',
                    'strike': p['strike'],
                    'contracts': p['contracts'],
                    'proceeds': proceeds,
                    'commission': commission,
                    'pnl': pnl
                })
                
                positions_to_remove.append(i)
        
        # Remove sold positions
        for i in reversed(positions_to_remove):
            self.overnight_puts.pop(i)
    
    def _calculate_portfolio_value(self, date, qqq_price: float, qqq_iv: float, 
                                   sqqq_price: float, sqqq_iv: float) -> float:
        """
        Override to include overnight hedge positions
        """
        # Get base portfolio value
        total_value = super()._calculate_portfolio_value(date, qqq_price, qqq_iv, 
                                                         sqqq_price, sqqq_iv)
        
        # Add overnight put value (these are long, so they're assets)
        for p in self.overnight_puts:
            days_held = (date - p['date_opened']).days
            remaining_dte = max(p['dte'] - days_held, 0.1)
            T = remaining_dte / 365.0
            current_value = bs_put_price(qqq_price, p['strike'], qqq_iv, T)
            total_value += current_value * p['contracts'] * 100
        
        return total_value
    
    def run_backtest(self) -> dict:
        """
        Run backtest with overnight hedging
        """
        print(f"\n🚀 Running OPTIMIZED SQQQ-QQQ Wheel Strategy Backtest")
        print(f"{'='*60}")
        print(f"Period: {self.start_date} to {self.end_date}")
        print(f"Initial Capital: ${self.initial_capital:,.0f}")
        print(f"Max QQQ Allocation: {self.max_qqq_allocation:.0%}")
        print(f"Weekly Capital Deployment: {self.weekly_capital_pct:.1%}")
        print(f"Overnight Hedging: {'ENABLED' if self.enable_overnight_hedge else 'DISABLED'}")
        if self.enable_overnight_hedge:
            print(f"  - Hedge Coverage: {self.overnight_hedge_coverage:.0%}")
            print(f"  - Hedge Delta: {self.overnight_hedge_delta:.0%}")
            print(f"  - Hedge DTE: {self.overnight_hedge_dte} days")
        print(f"{'='*60}\n")
        
        # Track last trade day
        last_put_sale_date = None
        
        # Iterate through each trading day
        for i, date in enumerate(self.qqq_data.index):
            qqq_price = self.qqq_data.loc[date, 'Close']
            qqq_iv = self.qqq_data.loc[date, 'IV']
            
            # Get SQQQ data for this date
            if date in self.sqqq_data.index:
                sqqq_price = self.sqqq_data.loc[date, 'Close']
                sqqq_iv = self.sqqq_data.loc[date, 'IV']
            else:
                sqqq_price = self.sqqq_data['Close'].iloc[-1] if len(self.sqqq_data) > 0 else 10
                sqqq_iv = self.sqqq_data['IV'].iloc[-1] if len(self.sqqq_data) > 0 else 0.5
            
            day_of_week = date.dayofweek  # 0=Monday, 4=Friday
            
            # MORNING: Sell overnight hedges from previous day
            if self.enable_overnight_hedge and len(self.overnight_puts) > 0:
                self._sell_overnight_hedge(date, qqq_price, qqq_iv)
            
            # FRIDAY: Buy SQQQ calls to hedge weekend
            if day_of_week == 4 and len(self.short_puts) > 0:  # Friday
                qqq_delta = self._get_net_qqq_delta(qqq_price, qqq_iv, date)
                if qqq_delta > 0:
                    self._buy_sqqq_calls(date, sqqq_price, sqqq_iv, qqq_delta)
            
            # MONDAY: Sell SQQQ calls
            if day_of_week == 0 and len(self.sqqq_calls) > 0:  # Monday
                self._sell_sqqq_calls(date, sqqq_price, sqqq_iv)
            
            # WEEKLY PUT SALES (every Monday)
            if day_of_week == 0:  # Monday
                if last_put_sale_date is None or (date - last_put_sale_date).days >= 7:
                    self._sell_qqq_put(date, qqq_price, qqq_iv)
                    last_put_sale_date = date
            
            # Check for assignments and expirations (daily)
            self._check_put_assignments(date, qqq_price, qqq_iv)
            self._check_call_assignments(date, qqq_price)
            
            # EVENING: Buy overnight puts (if not Friday - already have SQQQ calls)
            # Only hedge weeknights, not weekends (we have SQQQ for that)
            if self.enable_overnight_hedge and day_of_week < 4:  # Mon-Thu
                qqq_delta = self._get_net_qqq_delta(qqq_price, qqq_iv, date)
                if qqq_delta > 0:
                    self._buy_overnight_hedge(date, qqq_price, qqq_iv)
            
            # Calculate portfolio value
            portfolio_value = self._calculate_portfolio_value(date, qqq_price, qqq_iv, 
                                                             sqqq_price, sqqq_iv)
            
            self.portfolio_values.append(portfolio_value)
            self.dates.append(date)
        
        # Calculate final results (same as parent)
        final_value = self.portfolio_values[-1]
        total_return = (final_value - self.initial_capital) / self.initial_capital
        
        years = (self.dates[-1] - self.dates[0]).days / 365.25
        cagr = (final_value / self.initial_capital) ** (1 / years) - 1
        
        portfolio_series = pd.Series(self.portfolio_values, index=self.dates)
        returns = portfolio_series.pct_change().dropna()
        sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        
        rolling_max = portfolio_series.expanding().max()
        drawdown = (portfolio_series - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        put_premium = sum(t.get('premium_received', 0) for t in self.trades if t['action'] == 'SELL_PUT')
        call_premium = sum(t.get('premium_received', 0) for t in self.trades if t['action'] == 'SELL_CALL')
        total_premium = put_premium + call_premium
        
        put_assignments = sum(1 for t in self.trades if t['action'] == 'PUT_ASSIGNED')
        call_assignments = sum(1 for t in self.trades if t['action'] == 'CALL_ASSIGNED')
        
        sqqq_pnl = sum(t.get('pnl', 0) for t in self.trades if t['action'] == 'SELL_SQQQ_CALL')
        
        # Count overnight hedge trades
        overnight_trades = sum(1 for t in self.trades if 'OVERNIGHT' in t['action'])
        
        results = {
            'initial_capital': self.initial_capital,
            'final_value': final_value,
            'total_return': total_return,
            'cagr': cagr,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'total_trades': len(self.trades),
            'put_assignments': put_assignments,
            'call_assignments': call_assignments,
            'total_premium_collected': total_premium,
            'put_premium': put_premium,
            'call_premium': call_premium,
            'sqqq_hedge_pnl': sqqq_pnl,
            'overnight_hedge_pnl': self.overnight_hedge_pnl,
            'overnight_trades': overnight_trades,
            'final_qqq_shares': self.qqq_shares,
            'final_cash': self.cash,
            'portfolio_values': self.portfolio_values,
            'dates': self.dates,
            'trades': self.trades
        }
        
        return results
    
    def print_summary(self, results: dict):
        """Enhanced summary with overnight hedge stats"""
        print(f"\n{'='*60}")
        print(f"📊 OPTIMIZED BACKTEST RESULTS SUMMARY")
        print(f"{'='*60}")
        print(f"Initial Capital:        ${results['initial_capital']:>15,.0f}")
        print(f"Final Value:            ${results['final_value']:>15,.0f}")
        print(f"Total Return:           {results['total_return']:>15.1%}")
        print(f"CAGR:                   {results['cagr']:>15.1%}")
        print(f"Sharpe Ratio:           {results['sharpe_ratio']:>15.2f}")
        print(f"Max Drawdown:           {results['max_drawdown']:>15.1%}")
        print(f"\n{'='*60}")
        print(f"TRADING ACTIVITY")
        print(f"{'='*60}")
        print(f"Total Trades:           {results['total_trades']:>15,}")
        print(f"  - Overnight Hedges:   {results['overnight_trades']:>15,}")
        print(f"Put Assignments:        {results['put_assignments']:>15,}")
        print(f"Call Assignments:       {results['call_assignments']:>15,}")
        print(f"\n{'='*60}")
        print(f"PREMIUM & P&L")
        print(f"{'='*60}")
        print(f"Put Premium Collected:  ${results['put_premium']:>15,.0f}")
        print(f"Call Premium Collected: ${results['call_premium']:>15,.0f}")
        print(f"Total Premium:          ${results['total_premium_collected']:>15,.0f}")
        print(f"SQQQ Hedge P&L:         ${results['sqqq_hedge_pnl']:>15,.0f}")
        print(f"Overnight Hedge P&L:    ${results['overnight_hedge_pnl']:>15,.0f}")
        print(f"\n{'='*60}")
        print(f"FINAL POSITIONS")
        print(f"{'='*60}")
        print(f"Cash:                   ${results['final_cash']:>15,.0f}")
        print(f"QQQ Shares:             {results['final_qqq_shares']:>15,}")
        print(f"{'='*60}\n")


def compare_strategies():
    """
    Compare original vs optimized strategy with overnight hedging
    """
    print("\n" + "="*70)
    print("STRATEGY COMPARISON: ORIGINAL vs OVERNIGHT HEDGING")
    print("="*70 + "\n")
    
    # Run original strategy
    print("🔷 Running ORIGINAL strategy (weekend hedges only)...\n")
    original = SQQQQQQWheelOptimized(
        initial_capital=100000,
        enable_overnight_hedge=False,  # Disable overnight hedging
        start_date='2020-01-01',
        end_date='2025-01-01'
    )
    original_results = original.run_backtest()
    
    # Run optimized strategy
    print("\n🔶 Running OPTIMIZED strategy (with overnight hedging)...\n")
    optimized = SQQQQQQWheelOptimized(
        initial_capital=100000,
        enable_overnight_hedge=True,   # Enable overnight hedging
        overnight_hedge_delta=-0.50,   # ATM puts
        overnight_hedge_dte=30,        # 30 DTE for realistic theta decay
        overnight_hedge_coverage=1.0,  # 100% coverage
        start_date='2020-01-01',
        end_date='2025-01-01'
    )
    optimized_results = optimized.run_backtest()
    
    # Print comparison
    print("\n" + "="*70)
    print("📊 PERFORMANCE COMPARISON")
    print("="*70)
    
    metrics = [
        ('Total Return', 'total_return', '%'),
        ('CAGR', 'cagr', '%'),
        ('Sharpe Ratio', 'sharpe_ratio', ''),
        ('Max Drawdown', 'max_drawdown', '%'),
        ('Final Value', 'final_value', '$'),
        ('Total Premium', 'total_premium_collected', '$'),
        ('SQQQ Hedge P&L', 'sqqq_hedge_pnl', '$'),
        ('Overnight Hedge P&L', 'overnight_hedge_pnl', '$'),
        ('Total Trades', 'total_trades', '#'),
    ]
    
    print(f"\n{'Metric':<25} {'Original':>15} {'Optimized':>15} {'Difference':>15}")
    print("-" * 70)
    
    for metric_name, metric_key, fmt in metrics:
        orig_val = original_results[metric_key]
        opt_val = optimized_results[metric_key]
        diff = opt_val - orig_val
        
        if fmt == '%':
            print(f"{metric_name:<25} {orig_val:>14.1%} {opt_val:>14.1%} {diff:>+14.1%}")
        elif fmt == '$':
            print(f"{metric_name:<25} ${orig_val:>13,.0f} ${opt_val:>13,.0f} ${diff:>+13,.0f}")
        elif fmt == '#':
            print(f"{metric_name:<25} {orig_val:>15,.0f} {opt_val:>15,.0f} {diff:>+15,.0f}")
        else:
            print(f"{metric_name:<25} {orig_val:>15.2f} {opt_val:>15.2f} {diff:>+15.2f}")
    
    print("="*70)
    
    # Calculate improvement
    return_improvement = optimized_results['total_return'] - original_results['total_return']
    sharpe_improvement = optimized_results['sharpe_ratio'] - original_results['sharpe_ratio']
    dd_improvement = optimized_results['max_drawdown'] - original_results['max_drawdown']
    
    print(f"\n💡 KEY INSIGHTS:")
    print(f"  • Return improvement: {return_improvement:+.1%}")
    print(f"  • Sharpe improvement: {sharpe_improvement:+.2f}")
    print(f"  • Drawdown improvement: {dd_improvement:+.1%} (lower is better)")
    print(f"  • Overnight hedge cost: ${optimized_results['overnight_hedge_pnl']:,.0f}")
    print(f"  • Extra trades: {optimized_results['total_trades'] - original_results['total_trades']:,}")
    
    if return_improvement > 0 and sharpe_improvement > 0:
        print(f"\n✅ VERDICT: Overnight hedging IMPROVES performance!")
    elif return_improvement > 0:
        print(f"\n⚠️  VERDICT: Higher returns but mixed risk-adjusted performance")
    else:
        print(f"\n❌ VERDICT: Overnight hedging reduces returns (hedge drag)")
    
    print("="*70 + "\n")
    
    return original_results, optimized_results


if __name__ == "__main__":
    # Run comparison
    original_results, optimized_results = compare_strategies()

