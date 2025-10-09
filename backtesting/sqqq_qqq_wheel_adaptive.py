#!/usr/bin/env python3
"""
SQQQ-QQQ Wheel Strategy with ADAPTIVE Overnight Hedging

Smart hedging based on holding period:
- Weeknights (Mon-Thu): 1-2 DTE puts (minimal theta, just overnight)
- Weekends (Fri): 7-14 DTE puts (cover 3-day weekend)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sqqq_qqq_wheel_strategy import SQQQQQQWheelStrategy, bs_put_price, put_delta, strike_for_put_delta
import pandas as pd
import numpy as np


class SQQQQQQWheelAdaptive(SQQQQQQWheelStrategy):
    """
    Enhanced wheel strategy with ADAPTIVE overnight hedging
    
    - Weeknights: Short-dated options (1-2 DTE) for low theta
    - Weekends: Longer-dated options (7-14 DTE) for multi-day coverage
    """
    
    def __init__(self, 
                 enable_overnight_hedge=True,
                 overnight_hedge_delta=-0.50,
                 weeknight_hedge_dte=2,  # 1-2 DTE for Mon-Thu nights
                 weekend_hedge_dte=14,   # 7-14 DTE for Fri-Mon
                 overnight_hedge_coverage=1.0,
                 **kwargs):
        """
        Initialize adaptive strategy
        
        Args:
            enable_overnight_hedge: Enable daily overnight put protection
            overnight_hedge_delta: Delta for overnight puts (-0.50 = ATM)
            weeknight_hedge_dte: DTE for weeknight hedges (1-2 days)
            weekend_hedge_dte: DTE for weekend hedges (7-14 days)
            overnight_hedge_coverage: % of QQQ exposure to hedge (1.0 = 100%)
        """
        super().__init__(**kwargs)
        
        self.enable_overnight_hedge = enable_overnight_hedge
        self.overnight_hedge_delta = overnight_hedge_delta
        self.weeknight_hedge_dte = weeknight_hedge_dte
        self.weekend_hedge_dte = weekend_hedge_dte
        self.overnight_hedge_coverage = overnight_hedge_coverage
        
        # Track overnight hedges separately
        self.overnight_puts = []
        self.overnight_hedge_pnl = 0.0
        
    def _buy_overnight_hedge(self, date, current_price: float, iv: float, is_friday: bool = False) -> dict:
        """
        Buy protective puts before close
        Adaptive DTE based on holding period:
        - Weeknights: 1-2 DTE (sell next morning)
        - Weekends: 7-14 DTE (sell Monday)
        """
        # Calculate net QQQ delta exposure
        qqq_delta = self._get_net_qqq_delta(current_price, iv, date)
        
        if qqq_delta <= 0:
            return None
        
        # Calculate contracts needed
        delta_to_hedge = qqq_delta * self.overnight_hedge_coverage
        contracts_needed = int(abs(delta_to_hedge) / 100)
        
        if contracts_needed < 1:
            return None
        
        # Adaptive DTE: longer for weekends
        dte = self.weekend_hedge_dte if is_friday else self.weeknight_hedge_dte
        T = dte / 365.0
        
        # Solve for strike at target delta
        strike = strike_for_put_delta(current_price, iv, T, self.overnight_hedge_delta)
        
        # Get premium
        premium = bs_put_price(current_price, strike, iv, T)
        
        # Calculate cost
        total_cost = premium * contracts_needed * 100
        commission = self.option_commission * contracts_needed
        
        # Check if we have cash (use hedge reserve)
        max_overnight_cost = self.initial_capital * 0.03  # Max 3% for overnight hedges
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
            'dte': dte,
            'type': 'overnight_put',
            'cost': total_cost + commission,
            'is_weekend': is_friday
        }
        
        self.overnight_puts.append(position)
        
        trade = {
            'date': date,
            'action': 'BUY_OVERNIGHT_PUT',
            'symbol': 'QQQ',
            'strike': strike,
            'premium': premium,
            'contracts': contracts_needed,
            'dte': dte,
            'cost': total_cost,
            'commission': commission,
            'delta': self.overnight_hedge_delta,
            'qqq_delta_hedged': qqq_delta,
            'is_weekend': is_friday
        }
        
        self.trades.append(trade)
        
        hedge_type = "WEEKEND" if is_friday else "OVERNIGHT"
        print(f"🛡️  {date.date()}: {hedge_type} hedge - bought {contracts_needed} QQQ {strike:.2f}P ({dte}DTE) for ${total_cost:,.0f}")
        
        return trade
    
    def _sell_overnight_hedge(self, date, current_price: float, iv: float):
        """
        Sell overnight puts at open
        """
        positions_to_remove = []
        
        for i, p in enumerate(self.overnight_puts):
            days_held = (date - p['date_opened']).days
            
            # Sell if held for 1+ days (overnight or through weekend)
            if days_held >= 1:
                # Calculate remaining DTE
                remaining_dte = max(p['dte'] - days_held, 0.1)
                T = remaining_dte / 365.0
                
                current_value = bs_put_price(current_price, p['strike'], iv, T)
                
                # Sell
                proceeds = current_value * p['contracts'] * 100
                commission = self.option_commission * p['contracts']
                
                self.cash += proceeds - commission
                
                # Calculate P&L
                pnl = proceeds - p['cost']
                self.overnight_hedge_pnl += pnl
                
                hedge_type = "WEEKEND" if p.get('is_weekend', False) else "OVERNIGHT"
                
                self.trades.append({
                    'date': date,
                    'action': 'SELL_OVERNIGHT_PUT',
                    'symbol': 'QQQ',
                    'strike': p['strike'],
                    'contracts': p['contracts'],
                    'proceeds': proceeds,
                    'commission': commission,
                    'pnl': pnl,
                    'days_held': days_held,
                    'is_weekend': p.get('is_weekend', False)
                })
                
                if abs(pnl) > 100:  # Only print significant P&Ls
                    print(f"💵 {date.date()}: {hedge_type} hedge P&L: ${pnl:,.0f} ({days_held} days)")
                
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
        
        # Add overnight put value (long positions)
        for p in self.overnight_puts:
            days_held = (date - p['date_opened']).days
            remaining_dte = max(p['dte'] - days_held, 0.1)
            T = remaining_dte / 365.0
            current_value = bs_put_price(qqq_price, p['strike'], qqq_iv, T)
            total_value += current_value * p['contracts'] * 100
        
        return total_value
    
    def run_backtest(self) -> dict:
        """
        Run backtest with adaptive overnight hedging
        """
        print(f"\n🚀 Running ADAPTIVE SQQQ-QQQ Wheel Strategy Backtest")
        print(f"{'='*60}")
        print(f"Period: {self.start_date} to {self.end_date}")
        print(f"Initial Capital: ${self.initial_capital:,.0f}")
        print(f"Max QQQ Allocation: {self.max_qqq_allocation:.0%}")
        print(f"Weekly Capital Deployment: {self.weekly_capital_pct:.1%}")
        print(f"Overnight Hedging: {'ENABLED' if self.enable_overnight_hedge else 'DISABLED'}")
        if self.enable_overnight_hedge:
            print(f"  - Weeknight DTE: {self.weeknight_hedge_dte} days (Mon-Thu)")
            print(f"  - Weekend DTE: {self.weekend_hedge_dte} days (Fri-Mon)")
            print(f"  - Hedge Coverage: {self.overnight_hedge_coverage:.0%}")
            print(f"  - Hedge Delta: {self.overnight_hedge_delta:.0%}")
        print(f"{'='*60}\n")
        
        # Track last trade day
        last_put_sale_date = None
        
        # Iterate through each trading day
        for i, date in enumerate(self.qqq_data.index):
            qqq_price = self.qqq_data.loc[date, 'Close']
            qqq_iv = self.qqq_data.loc[date, 'IV']
            
            # Get SQQQ data
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
            
            # WEEKLY PUT SALES (every Monday)
            if day_of_week == 0:  # Monday
                if last_put_sale_date is None or (date - last_put_sale_date).days >= 7:
                    self._sell_qqq_put(date, qqq_price, qqq_iv)
                    last_put_sale_date = date
            
            # Check for assignments and expirations (daily)
            self._check_put_assignments(date, qqq_price, qqq_iv)
            self._check_call_assignments(date, qqq_price)
            
            # EVENING: Buy overnight hedges
            if self.enable_overnight_hedge:
                qqq_delta = self._get_net_qqq_delta(qqq_price, qqq_iv, date)
                if qqq_delta > 0:
                    # Friday evening: Use longer-dated options for weekend
                    # Mon-Thu evening: Use short-dated options
                    is_friday = (day_of_week == 4)
                    self._buy_overnight_hedge(date, qqq_price, qqq_iv, is_friday=is_friday)
            
            # Calculate portfolio value
            portfolio_value = self._calculate_portfolio_value(date, qqq_price, qqq_iv, 
                                                             sqqq_price, sqqq_iv)
            
            self.portfolio_values.append(portfolio_value)
            self.dates.append(date)
        
        # Calculate final results
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
        
        # Count overnight hedge trades
        overnight_trades = sum(1 for t in self.trades if 'OVERNIGHT' in t['action'])
        weeknight_hedges = sum(1 for t in self.trades if t['action'] == 'BUY_OVERNIGHT_PUT' and not t.get('is_weekend', False))
        weekend_hedges = sum(1 for t in self.trades if t['action'] == 'BUY_OVERNIGHT_PUT' and t.get('is_weekend', False))
        
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
            'overnight_hedge_pnl': self.overnight_hedge_pnl,
            'overnight_trades': overnight_trades,
            'weeknight_hedges': weeknight_hedges,
            'weekend_hedges': weekend_hedges,
            'final_qqq_shares': self.qqq_shares,
            'final_cash': self.cash,
            'portfolio_values': self.portfolio_values,
            'dates': self.dates,
            'trades': self.trades
        }
        
        return results
    
    def print_summary(self, results: dict):
        """Enhanced summary with adaptive hedge stats"""
        print(f"\n{'='*60}")
        print(f"📊 ADAPTIVE BACKTEST RESULTS SUMMARY")
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
        print(f"  - Weeknight Hedges:   {results['weeknight_hedges']:>15,}")
        print(f"  - Weekend Hedges:     {results['weekend_hedges']:>15,}")
        print(f"Put Assignments:        {results['put_assignments']:>15,}")
        print(f"Call Assignments:       {results['call_assignments']:>15,}")
        print(f"\n{'='*60}")
        print(f"PREMIUM & P&L")
        print(f"{'='*60}")
        print(f"Put Premium Collected:  ${results['put_premium']:>15,.0f}")
        print(f"Call Premium Collected: ${results['call_premium']:>15,.0f}")
        print(f"Total Premium:          ${results['total_premium_collected']:>15,.0f}")
        print(f"Overnight Hedge P&L:    ${results['overnight_hedge_pnl']:>15,.0f}")
        print(f"\n{'='*60}")
        print(f"FINAL POSITIONS")
        print(f"{'='*60}")
        print(f"Cash:                   ${results['final_cash']:>15,.0f}")
        print(f"QQQ Shares:             {results['final_qqq_shares']:>15,}")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("ADAPTIVE OVERNIGHT HEDGING BACKTEST")
    print("="*70 + "\n")
    
    strategy = SQQQQQQWheelAdaptive(
        initial_capital=100000,
        enable_overnight_hedge=True,
        overnight_hedge_delta=-0.50,
        weeknight_hedge_dte=2,   # 1-2 DTE for Mon-Thu
        weekend_hedge_dte=14,    # 7-14 DTE for Fri-Mon
        overnight_hedge_coverage=1.0,
        start_date='2020-01-01',
        end_date='2025-01-01'
    )
    
    results = strategy.run_backtest()
    strategy.print_summary(results)

