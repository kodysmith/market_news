#!/usr/bin/env python3
"""
SQQQ-QQQ Wheel Strategy with EVENT-DRIVEN Hedging

Smart hedging only around real risk events:
1. Holidays (market closed next day)
2. Major tech earnings (NVDA, MSFT, AAPL, META, GOOGL, AMZN)
3. Weekends (SQQQ calls - already implemented)

This dramatically reduces hedge cost while protecting against actual gap risk.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sqqq_qqq_wheel_strategy import SQQQQQQWheelStrategy, bs_put_price, strike_for_put_delta
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


class SQQQQQQWheelEvents(SQQQQQQWheelStrategy):
    """
    Event-driven wheel strategy
    
    Only hedge before:
    - Holidays
    - Major tech earnings
    - Not every night (eliminates daily theta bleed)
    """
    
    def __init__(self, 
                 enable_event_hedge=True,
                 event_hedge_delta=-0.50,
                 event_hedge_dte=7,  # 7 DTE for events
                 event_hedge_coverage=1.0,
                 **kwargs):
        """
        Initialize event-driven strategy
        
        Args:
            enable_event_hedge: Enable hedging before events
            event_hedge_delta: Delta for event puts
            event_hedge_dte: DTE for event hedges (7 days default)
            event_hedge_coverage: % of exposure to hedge
        """
        super().__init__(**kwargs)
        
        self.enable_event_hedge = enable_event_hedge
        self.event_hedge_delta = event_hedge_delta
        self.event_hedge_dte = event_hedge_dte
        self.event_hedge_coverage = event_hedge_coverage
        
        # Track event hedges
        self.event_puts = []
        self.event_hedge_pnl = 0.0
        
        # Define US market holidays (approximate for 2020-2025)
        self._define_holidays()
        
    def _define_holidays(self):
        """Define major US market holidays for 2020-2025"""
        # Simplified holiday list (would use pandas.market_calendars in production)
        self.holidays = pd.to_datetime([
            # 2020
            '2020-01-01', '2020-01-20', '2020-02-17', '2020-04-10', '2020-05-25',
            '2020-07-03', '2020-09-07', '2020-11-26', '2020-12-25',
            # 2021
            '2021-01-01', '2021-01-18', '2021-02-15', '2021-04-02', '2021-05-31',
            '2021-07-05', '2021-09-06', '2021-11-25', '2021-12-24',
            # 2022
            '2022-01-17', '2022-02-21', '2022-04-15', '2022-05-30', '2022-06-20',
            '2022-07-04', '2022-09-05', '2022-11-24', '2022-12-26',
            # 2023
            '2023-01-02', '2023-01-16', '2023-02-20', '2023-04-07', '2023-05-29',
            '2023-06-19', '2023-07-04', '2023-09-04', '2023-11-23', '2023-12-25',
            # 2024
            '2024-01-01', '2024-01-15', '2024-02-19', '2024-03-29', '2024-05-27',
            '2024-06-19', '2024-07-04', '2024-09-02', '2024-11-28', '2024-12-25',
        ])
        
    def _is_before_holiday(self, date) -> bool:
        """Check if tomorrow is a holiday"""
        tomorrow = date + timedelta(days=1)
        # Check if tomorrow is holiday or weekend
        return tomorrow in self.holidays.values or tomorrow.weekday() >= 5
    
    def _is_earnings_week(self, date) -> bool:
        """
        Check if we're in a major earnings week
        
        Major tech earnings typically cluster:
        - Late January (MSFT, AAPL, META)
        - Late April (GOOGL, MSFT, META)
        - Late July (AAPL, META, AMZN)
        - Late October (GOOGL, MSFT, META)
        
        For simplicity, hedge the week before these periods
        """
        month = date.month
        day = date.day
        
        # Late January earnings
        if month == 1 and 20 <= day <= 31:
            return True
        
        # Late April earnings
        if month == 4 and 20 <= day <= 30:
            return True
        
        # Late July earnings
        if month == 7 and 20 <= day <= 31:
            return True
        
        # Late October earnings
        if month == 10 and 20 <= day <= 31:
            return True
        
        # NVIDIA earnings (typically Feb, May, Aug, Nov)
        # Hedge last week of these months
        if month in [2, 5, 8, 11] and day >= 20:
            return True
        
        return False
    
    def _should_hedge_today(self, date) -> tuple:
        """
        Determine if we should hedge today and why
        Returns: (should_hedge: bool, reason: str)
        """
        # Check for holidays
        if self._is_before_holiday(date):
            return True, "HOLIDAY"
        
        # Check for earnings season
        if self._is_earnings_week(date):
            return True, "EARNINGS"
        
        # Check for Friday (weekend - but we use SQQQ calls for this)
        # Don't double-hedge with puts
        # if date.weekday() == 4:  # Friday
        #     return True, "WEEKEND"
        
        return False, None
    
    def _buy_event_hedge(self, date, current_price: float, iv: float, reason: str) -> dict:
        """
        Buy protective puts before an event
        """
        # Calculate net QQQ delta exposure
        qqq_delta = self._get_net_qqq_delta(current_price, iv, date)
        
        if qqq_delta <= 0:
            return None
        
        # Calculate contracts needed
        delta_to_hedge = qqq_delta * self.event_hedge_coverage
        contracts_needed = int(abs(delta_to_hedge) / 100)
        
        if contracts_needed < 1:
            return None
        
        # Use 7 DTE for events (balance between cost and protection)
        T = self.event_hedge_dte / 365.0
        
        # Solve for strike
        strike = strike_for_put_delta(current_price, iv, T, self.event_hedge_delta)
        
        # Get premium
        premium = bs_put_price(current_price, strike, iv, T)
        
        # Calculate cost
        total_cost = premium * contracts_needed * 100
        commission = self.option_commission * contracts_needed
        
        # Check cash
        max_hedge_cost = self.initial_capital * 0.05  # Max 5% for event hedges
        if total_cost + commission > max_hedge_cost or total_cost + commission > self.cash:
            contracts_needed = max(1, int(max_hedge_cost / (premium * 100 + self.option_commission)))
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
            'dte': self.event_hedge_dte,
            'type': 'event_put',
            'cost': total_cost + commission,
            'reason': reason
        }
        
        self.event_puts.append(position)
        
        trade = {
            'date': date,
            'action': 'BUY_EVENT_HEDGE',
            'symbol': 'QQQ',
            'strike': strike,
            'premium': premium,
            'contracts': contracts_needed,
            'dte': self.event_hedge_dte,
            'cost': total_cost,
            'commission': commission,
            'delta': self.event_hedge_delta,
            'qqq_delta_hedged': qqq_delta,
            'reason': reason
        }
        
        self.trades.append(trade)
        
        print(f"🛡️  {date.date()}: {reason} hedge - bought {contracts_needed} QQQ {strike:.2f}P ({self.event_hedge_dte}DTE) for ${total_cost:,.0f}")
        
        return trade
    
    def _sell_event_hedge(self, date, current_price: float, iv: float):
        """
        Sell event hedges after event passes (or at profit target)
        """
        positions_to_remove = []
        
        for i, p in enumerate(self.event_puts):
            days_held = (date - p['date_opened']).days
            
            # Sell after event (1+ days) or at 50% profit
            if days_held >= 1:
                remaining_dte = max(p['dte'] - days_held, 0.1)
                T = remaining_dte / 365.0
                
                current_value = bs_put_price(current_price, p['strike'], iv, T)
                
                # Calculate profit %
                profit_pct = (current_value - p['premium']) / p['premium']
                
                # Sell if: held 1+ days OR at 50% profit
                if days_held >= 1 or profit_pct >= 0.50:
                    # Sell
                    proceeds = current_value * p['contracts'] * 100
                    commission = self.option_commission * p['contracts']
                    
                    self.cash += proceeds - commission
                    
                    # Calculate P&L
                    pnl = proceeds - p['cost']
                    self.event_hedge_pnl += pnl
                    
                    self.trades.append({
                        'date': date,
                        'action': 'SELL_EVENT_HEDGE',
                        'symbol': 'QQQ',
                        'strike': p['strike'],
                        'contracts': p['contracts'],
                        'proceeds': proceeds,
                        'commission': commission,
                        'pnl': pnl,
                        'days_held': days_held,
                        'reason': p['reason']
                    })
                    
                    if abs(pnl) > 100:
                        print(f"💵 {date.date()}: {p['reason']} hedge P&L: ${pnl:,.0f} ({days_held} days, {profit_pct:+.1%})")
                    
                    positions_to_remove.append(i)
        
        # Remove sold positions
        for i in reversed(positions_to_remove):
            self.event_puts.pop(i)
    
    def _calculate_portfolio_value(self, date, qqq_price: float, qqq_iv: float, 
                                   sqqq_price: float, sqqq_iv: float) -> float:
        """
        Override to include event hedge positions
        """
        # Get base portfolio value
        total_value = super()._calculate_portfolio_value(date, qqq_price, qqq_iv, 
                                                         sqqq_price, sqqq_iv)
        
        # Add event put value
        for p in self.event_puts:
            days_held = (date - p['date_opened']).days
            remaining_dte = max(p['dte'] - days_held, 0.1)
            T = remaining_dte / 365.0
            current_value = bs_put_price(qqq_price, p['strike'], qqq_iv, T)
            total_value += current_value * p['contracts'] * 100
        
        return total_value
    
    def run_backtest(self) -> dict:
        """
        Run backtest with event-driven hedging
        """
        print(f"\n🚀 Running EVENT-DRIVEN SQQQ-QQQ Wheel Strategy Backtest")
        print(f"{'='*60}")
        print(f"Period: {self.start_date} to {self.end_date}")
        print(f"Initial Capital: ${self.initial_capital:,.0f}")
        print(f"Max QQQ Allocation: {self.max_qqq_allocation:.0%}")
        print(f"Weekly Capital Deployment: {self.weekly_capital_pct:.1%}")
        print(f"Event Hedging: {'ENABLED' if self.enable_event_hedge else 'DISABLED'}")
        if self.enable_event_hedge:
            print(f"  - Hedge before: Holidays, Major Earnings")
            print(f"  - Hedge DTE: {self.event_hedge_dte} days")
            print(f"  - Hedge Coverage: {self.event_hedge_coverage:.0%}")
            print(f"  - Hedge Delta: {self.event_hedge_delta:.0%}")
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
            
            # MORNING: Sell event hedges from previous events
            if self.enable_event_hedge and len(self.event_puts) > 0:
                self._sell_event_hedge(date, qqq_price, qqq_iv)
            
            # FRIDAY: Buy SQQQ calls for weekend (keep existing logic)
            if day_of_week == 4 and len(self.short_puts) > 0:
                qqq_delta = self._get_net_qqq_delta(qqq_price, qqq_iv, date)
                if qqq_delta > 0:
                    self._buy_sqqq_calls(date, sqqq_price, sqqq_iv, qqq_delta)
            
            # MONDAY: Sell SQQQ calls
            if day_of_week == 0 and len(self.sqqq_calls) > 0:
                self._sell_sqqq_calls(date, sqqq_price, sqqq_iv)
            
            # WEEKLY PUT SALES (every Monday)
            if day_of_week == 0:
                if last_put_sale_date is None or (date - last_put_sale_date).days >= 7:
                    self._sell_qqq_put(date, qqq_price, qqq_iv)
                    last_put_sale_date = date
            
            # Check for assignments and expirations
            self._check_put_assignments(date, qqq_price, qqq_iv)
            self._check_call_assignments(date, qqq_price)
            
            # EVENING: Check if we should hedge for upcoming event
            if self.enable_event_hedge:
                should_hedge, reason = self._should_hedge_today(date)
                if should_hedge:
                    qqq_delta = self._get_net_qqq_delta(qqq_price, qqq_iv, date)
                    if qqq_delta > 0:
                        self._buy_event_hedge(date, qqq_price, qqq_iv, reason)
            
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
        
        sqqq_pnl = sum(t.get('pnl', 0) for t in self.trades if t['action'] == 'SELL_SQQQ_CALL')
        
        # Count event hedges
        event_hedges = sum(1 for t in self.trades if t['action'] == 'BUY_EVENT_HEDGE')
        holiday_hedges = sum(1 for t in self.trades if t['action'] == 'BUY_EVENT_HEDGE' and t.get('reason') == 'HOLIDAY')
        earnings_hedges = sum(1 for t in self.trades if t['action'] == 'BUY_EVENT_HEDGE' and t.get('reason') == 'EARNINGS')
        
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
            'event_hedge_pnl': self.event_hedge_pnl,
            'event_hedges': event_hedges,
            'holiday_hedges': holiday_hedges,
            'earnings_hedges': earnings_hedges,
            'final_qqq_shares': self.qqq_shares,
            'final_cash': self.cash,
            'portfolio_values': self.portfolio_values,
            'dates': self.dates,
            'trades': self.trades
        }
        
        return results
    
    def print_summary(self, results: dict):
        """Enhanced summary with event hedge stats"""
        print(f"\n{'='*60}")
        print(f"📊 EVENT-DRIVEN BACKTEST RESULTS SUMMARY")
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
        print(f"  - Event Hedges:       {results['event_hedges']:>15,}")
        print(f"    • Holiday Hedges:   {results['holiday_hedges']:>15,}")
        print(f"    • Earnings Hedges:  {results['earnings_hedges']:>15,}")
        print(f"Put Assignments:        {results['put_assignments']:>15,}")
        print(f"Call Assignments:       {results['call_assignments']:>15,}")
        print(f"\n{'='*60}")
        print(f"PREMIUM & P&L")
        print(f"{'='*60}")
        print(f"Put Premium Collected:  ${results['put_premium']:>15,.0f}")
        print(f"Call Premium Collected: ${results['call_premium']:>15,.0f}")
        print(f"Total Premium:          ${results['total_premium_collected']:>15,.0f}")
        print(f"SQQQ Hedge P&L:         ${results['sqqq_hedge_pnl']:>15,.0f}")
        print(f"Event Hedge P&L:        ${results['event_hedge_pnl']:>15,.0f}")
        print(f"\n{'='*60}")
        print(f"FINAL POSITIONS")
        print(f"{'='*60}")
        print(f"Cash:                   ${results['final_cash']:>15,.0f}")
        print(f"QQQ Shares:             {results['final_qqq_shares']:>15,}")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("EVENT-DRIVEN HEDGING BACKTEST")
    print("Hedge only before: Holidays, Major Earnings, Weekends")
    print("="*70 + "\n")
    
    strategy = SQQQQQQWheelEvents(
        initial_capital=100000,
        enable_event_hedge=True,
        event_hedge_delta=-0.50,
        event_hedge_dte=7,  # 7 DTE for events
        event_hedge_coverage=1.0,
        start_date='2020-01-01',
        end_date='2025-01-01'
    )
    
    results = strategy.run_backtest()
    strategy.print_summary(results)

