#!/usr/bin/env python3
"""
SQQQ-QQQ Wheel Strategy Backtester

Strategy:
1. Sell QQQ puts weekly at 30% delta, 14 DTE (1% of capital per trade)
2. Deploy up to 75% of capital in QQQ positions
3. If assigned, hold shares and sell covered calls (wheel)
4. Buy SQQQ ATM calls Friday, sell Monday (delta hedge)
5. Reserve 25% for hedges and margin

Capital: $100,000 initial
Period: 2020-2025 (5 years)
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Numba for fast BS pricing (reuse from existing code)
from numba import njit

@njit(fastmath=True)
def erf_approx(x):
    """Abramowitz & Stegun approximation for erf"""
    a1, a2, a3, a4, a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
    p = 0.3275911
    y = abs(x)
    t = 1.0 / (1.0 + p * y)
    z = t * (a1 + t * (a2 + t * (a3 + t * (a4 + t * a5))))
    return np.sign(x) * (1.0 - z * np.exp(-y * y))

@njit(fastmath=True)
def norm_cdf(x):
    """Normal CDF approximation"""
    return 0.5 * (1.0 + erf_approx(x / np.sqrt(2.0)))

@njit(fastmath=True)
def norm_ppf(p):
    """Inverse normal CDF (simplified rational approximation)"""
    p = max(1e-9, min(1 - 1e-9, p))
    if p < 0.5:
        # Lower tail
        q = np.sqrt(-2.0 * np.log(p))
        z = ((((-0.00778 * q - 0.322) * q - 2.4) * q - 2.55) * q + 4.37) / \
            ((((0.00778 * q + 0.322) * q + 2.45) * q + 3.75) * q + 1)
        return -z
    else:
        # Upper tail
        q = np.sqrt(-2.0 * np.log(1 - p))
        z = ((((-0.00778 * q - 0.322) * q - 2.4) * q - 2.55) * q + 4.37) / \
            ((((0.00778 * q + 0.322) * q + 2.45) * q + 3.75) * q + 1)
        return z

@njit(fastmath=True)
def bs_put_price(S, K, sigma, T, r=0.02):
    """Black-Scholes put pricing"""
    if T <= 0 or sigma <= 0:
        return max(K - S, 0.01)
    d1 = (np.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    put = K * np.exp(-r * T) * norm_cdf(-d2) - S * norm_cdf(-d1)
    return max(put, 0.01)

@njit(fastmath=True)
def bs_call_price(S, K, sigma, T, r=0.02):
    """Black-Scholes call pricing"""
    if T <= 0 or sigma <= 0:
        return max(S - K, 0.01)
    d1 = (np.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call = S * norm_cdf(d1) - K * np.exp(-r * T) * norm_cdf(d2)
    return max(call, 0.01)

@njit(fastmath=True)
def put_delta(S, K, sigma, T, r=0.02):
    """Put delta calculation"""
    if T <= 0 or sigma <= 0:
        return -1.0 if S < K else 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * np.sqrt(T))
    return norm_cdf(d1) - 1.0

@njit(fastmath=True)
def call_delta(S, K, sigma, T, r=0.02):
    """Call delta calculation"""
    if T <= 0 or sigma <= 0:
        return 1.0 if S > K else 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * np.sqrt(T))
    return norm_cdf(d1)

@njit(fastmath=True)
def strike_for_put_delta(S, sigma, T, target_delta=-0.30, r=0.02):
    """Solve for strike given target put delta"""
    if T <= 0 or sigma <= 0:
        return S * 0.95
    # Put delta = N(d1) - 1 = target_delta
    # So N(d1) = target_delta + 1
    d1 = norm_ppf(target_delta + 1.0)
    K = S * np.exp(-d1 * sigma * np.sqrt(T) + (r + 0.5 * sigma**2) * T)
    return K

@njit(fastmath=True)
def strike_for_call_delta(S, sigma, T, target_delta=0.30, r=0.02):
    """Solve for strike given target call delta"""
    if T <= 0 or sigma <= 0:
        return S * 1.05
    # Call delta = N(d1) = target_delta
    d1 = norm_ppf(target_delta)
    K = S * np.exp(-d1 * sigma * np.sqrt(T) + (r + 0.5 * sigma**2) * T)
    return K


class SQQQQQQWheelStrategy:
    """
    SQQQ-QQQ Wheel Strategy with Delta Hedging
    
    Strategy Components:
    1. Sell QQQ puts (30% delta, 14 DTE, weekly)
    2. On assignment: wheel to covered calls
    3. SQQQ calls (ATM, Friday-Monday, hedge QQQ delta)
    """
    
    def __init__(self,
                 initial_capital: float = 100000,
                 max_qqq_allocation: float = 0.75,
                 weekly_capital_pct: float = 0.01,
                 put_delta_target: float = -0.30,
                 put_dte: int = 14,
                 call_delta_target: float = 0.30,
                 call_dte: int = 30,
                 sqqq_call_dte: int = 30,
                 start_date: str = '2020-01-01',
                 end_date: str = '2025-01-01'):
        """
        Initialize the wheel strategy
        
        Args:
            initial_capital: Starting capital ($100k default)
            max_qqq_allocation: Max % of capital for QQQ (0.75 = 75%)
            weekly_capital_pct: % of capital per weekly trade (0.01 = 1%)
            put_delta_target: Target delta for sold puts (-0.30 = 30%)
            put_dte: Days to expiration for puts (14 days)
            call_delta_target: Target delta for covered calls (0.30 = 30%)
            call_dte: Days to expiration for covered calls (30 days)
            sqqq_call_dte: Days to expiration for SQQQ calls (30 days)
            start_date: Backtest start date
            end_date: Backtest end date
        """
        self.initial_capital = initial_capital
        self.max_qqq_allocation = max_qqq_allocation
        self.weekly_capital_pct = weekly_capital_pct
        self.put_delta_target = put_delta_target
        self.put_dte = put_dte
        self.call_delta_target = call_delta_target
        self.call_dte = call_dte
        self.sqqq_call_dte = sqqq_call_dte
        self.start_date = start_date
        self.end_date = end_date
        
        # Portfolio state
        self.cash = initial_capital
        self.qqq_shares = 0
        self.short_puts = []  # List of sold put positions
        self.covered_calls = []  # List of sold call positions
        self.sqqq_calls = []  # List of long SQQQ call positions
        
        # Tracking
        self.trades = []
        self.portfolio_values = []
        self.dates = []
        
        # Commission structure
        self.option_commission = 0.65  # per contract
        self.stock_commission = 0.01  # per share
        
        # Load data
        self.qqq_data = None
        self.sqqq_data = None
        self._load_data()
        
    def _load_data(self):
        """Download QQQ and SQQQ price data"""
        print(f"📊 Downloading QQQ and SQQQ data...")
        
        # Download QQQ
        qqq = yf.Ticker("QQQ")
        self.qqq_data = qqq.history(start=self.start_date, end=self.end_date)
        
        # Calculate volatility (20-day rolling for IV proxy)
        returns = self.qqq_data['Close'].pct_change()
        self.qqq_data['Returns'] = returns
        self.qqq_data['IV'] = returns.rolling(20).std() * np.sqrt(252)
        self.qqq_data['IV'] = self.qqq_data['IV'].fillna(method='bfill').fillna(0.25)
        self.qqq_data['IV'] = self.qqq_data['IV'].clip(0.10, 0.80)  # Reasonable bounds
        
        # Download SQQQ
        sqqq = yf.Ticker("SQQQ")
        self.sqqq_data = sqqq.history(start=self.start_date, end=self.end_date)
        
        # Calculate SQQQ volatility
        sqqq_returns = self.sqqq_data['Close'].pct_change()
        self.sqqq_data['Returns'] = sqqq_returns
        self.sqqq_data['IV'] = sqqq_returns.rolling(20).std() * np.sqrt(252)
        self.sqqq_data['IV'] = self.sqqq_data['IV'].fillna(method='bfill').fillna(0.50)
        self.sqqq_data['IV'] = self.sqqq_data['IV'].clip(0.30, 1.50)  # SQQQ is more volatile
        
        print(f"✅ Loaded {len(self.qqq_data)} days of QQQ data")
        print(f"✅ Loaded {len(self.sqqq_data)} days of SQQQ data")
        
    def _get_capital_deployed(self) -> float:
        """Calculate % of capital currently deployed in QQQ positions"""
        # Value of short puts (margin requirement estimate: ~20% of notional)
        put_exposure = sum(p['strike'] * p['contracts'] * 100 * 0.20 for p in self.short_puts)
        
        # Value of QQQ shares
        current_qqq_price = self.qqq_data.loc[self.dates[-1], 'Close'] if self.dates else 0
        share_value = self.qqq_shares * current_qqq_price
        
        total_deployed = put_exposure + share_value
        return total_deployed / self.initial_capital
    
    def _get_net_qqq_delta(self, current_qqq_price: float, current_iv: float, date) -> float:
        """
        Calculate net delta exposure from all QQQ positions
        Returns: net delta (positive = long exposure)
        """
        total_delta = 0.0
        
        # Delta from shares (always +1 per share)
        total_delta += self.qqq_shares
        
        # Delta from short puts (negative delta = short exposure when we sell puts)
        for p in self.short_puts:
            days_held = (date - p['date_opened']).days
            remaining_dte = max(p['dte'] - days_held, 1)
            T = remaining_dte / 365.0
            p_delta = put_delta(current_qqq_price, p['strike'], current_iv, T)
            # We're short the put, so multiply by -1
            total_delta += -1 * p_delta * p['contracts'] * 100
        
        # Delta from covered calls (reduces upside exposure)
        for c in self.covered_calls:
            days_held = (date - c['date_opened']).days
            remaining_dte = max(c['dte'] - days_held, 1)
            T = remaining_dte / 365.0
            c_delta = call_delta(current_qqq_price, c['strike'], current_iv, T)
            # We're short the call, so multiply by -1
            total_delta += -1 * c_delta * c['contracts'] * 100
        
        return total_delta
    
    def _sell_qqq_put(self, date, current_price: float, iv: float) -> Optional[Dict]:
        """
        Sell a QQQ put at 30% delta, 14 DTE
        Use 1% of capital per trade, respecting 75% max allocation
        """
        # Check if we have room to deploy more capital
        deployed = self._get_capital_deployed()
        if deployed >= self.max_qqq_allocation:
            return None
        
        # Calculate position size
        trade_capital = self.initial_capital * self.weekly_capital_pct
        
        # Solve for strike at target delta
        T = self.put_dte / 365.0
        strike = strike_for_put_delta(current_price, iv, T, self.put_delta_target)
        
        # Get premium
        premium = bs_put_price(current_price, strike, iv, T)
        
        # Calculate contracts: try to use $1k capital
        # Margin requirement is ~20% of notional for cash-secured puts
        notional_per_contract = strike * 100
        margin_per_contract = notional_per_contract * 0.20
        
        contracts = max(1, int(trade_capital / margin_per_contract))
        
        # Check if we have enough cash
        total_margin = margin_per_contract * contracts
        commission = self.option_commission * contracts
        
        if total_margin + commission > self.cash * 0.8:  # Keep 20% buffer
            return None
        
        # Execute trade
        premium_received = premium * contracts * 100
        self.cash += premium_received - commission
        
        position = {
            'date_opened': date,
            'strike': strike,
            'premium': premium,
            'contracts': contracts,
            'dte': self.put_dte,
            'type': 'put'
        }
        
        self.short_puts.append(position)
        
        trade = {
            'date': date,
            'action': 'SELL_PUT',
            'symbol': 'QQQ',
            'strike': strike,
            'premium': premium,
            'contracts': contracts,
            'dte': self.put_dte,
            'premium_received': premium_received,
            'commission': commission,
            'delta': self.put_delta_target
        }
        
        self.trades.append(trade)
        print(f"💰 {date.date()}: Sold {contracts} QQQ {strike:.2f}P for ${premium:.2f} (${premium_received:.0f} received)")
        
        return trade
    
    def _check_put_assignments(self, date, current_price: float, iv: float):
        """
        Check if any puts should be assigned
        Assignment happens if price < strike at expiration
        """
        positions_to_remove = []
        
        for i, p in enumerate(self.short_puts):
            days_held = (date - p['date_opened']).days
            remaining_dte = p['dte'] - days_held
            
            # Check for assignment at expiration
            if remaining_dte <= 0:
                if current_price < p['strike']:
                    # ASSIGNED! Buy shares at strike price
                    shares = p['contracts'] * 100
                    cost = shares * p['strike']
                    commission = shares * self.stock_commission
                    
                    if cost + commission <= self.cash:
                        self.cash -= (cost + commission)
                        self.qqq_shares += shares
                        
                        self.trades.append({
                            'date': date,
                            'action': 'PUT_ASSIGNED',
                            'symbol': 'QQQ',
                            'strike': p['strike'],
                            'shares': shares,
                            'cost': cost,
                            'commission': commission
                        })
                        
                        print(f"📦 {date.date()}: Put assigned - bought {shares} QQQ @ ${p['strike']:.2f}")
                        
                        # Immediately sell covered call (wheel)
                        self._sell_covered_call(date, current_price, iv, shares)
                    else:
                        print(f"⚠️  {date.date()}: Insufficient cash for assignment!")
                    
                    positions_to_remove.append(i)
                else:
                    # Expired OTM - keep premium
                    print(f"✅ {date.date()}: Put expired worthless - kept ${p['premium'] * p['contracts'] * 100:.2f}")
                    positions_to_remove.append(i)
            
            # Close at 50% profit target
            elif remaining_dte > 0:
                T = remaining_dte / 365.0
                current_value = bs_put_price(current_price, p['strike'], iv, T)
                profit_pct = (p['premium'] - current_value) / p['premium']
                
                if profit_pct >= 0.50:
                    # Close for 50% profit
                    close_cost = current_value * p['contracts'] * 100
                    commission = self.option_commission * p['contracts']
                    
                    self.cash -= (close_cost + commission)
                    
                    self.trades.append({
                        'date': date,
                        'action': 'CLOSE_PUT',
                        'symbol': 'QQQ',
                        'strike': p['strike'],
                        'contracts': p['contracts'],
                        'close_cost': close_cost,
                        'commission': commission,
                        'profit_pct': profit_pct
                    })
                    
                    print(f"💚 {date.date()}: Closed put for {profit_pct:.1%} profit")
                    positions_to_remove.append(i)
        
        # Remove closed/assigned positions
        for i in reversed(positions_to_remove):
            self.short_puts.pop(i)
    
    def _sell_covered_call(self, date, current_price: float, iv: float, shares_to_cover: int):
        """
        Sell covered call after assignment
        Target: 30% delta, 30-45 DTE
        """
        if shares_to_cover < 100:
            return None
        
        contracts = shares_to_cover // 100
        
        # Solve for strike at target delta
        T = self.call_dte / 365.0
        strike = strike_for_call_delta(current_price, iv, T, self.call_delta_target)
        
        # Get premium
        premium = bs_call_price(current_price, strike, iv, T)
        
        # Execute trade
        premium_received = premium * contracts * 100
        commission = self.option_commission * contracts
        
        self.cash += premium_received - commission
        
        position = {
            'date_opened': date,
            'strike': strike,
            'premium': premium,
            'contracts': contracts,
            'dte': self.call_dte,
            'type': 'call'
        }
        
        self.covered_calls.append(position)
        
        trade = {
            'date': date,
            'action': 'SELL_CALL',
            'symbol': 'QQQ',
            'strike': strike,
            'premium': premium,
            'contracts': contracts,
            'dte': self.call_dte,
            'premium_received': premium_received,
            'commission': commission,
            'delta': self.call_delta_target
        }
        
        self.trades.append(trade)
        print(f"📞 {date.date()}: Sold {contracts} QQQ {strike:.2f}C for ${premium:.2f} (${premium_received:.0f} received)")
        
        return trade
    
    def _check_call_assignments(self, date, current_price: float):
        """
        Check if any covered calls should be assigned
        Assignment happens if price >= strike at expiration
        """
        positions_to_remove = []
        
        for i, c in enumerate(self.covered_calls):
            days_held = (date - c['date_opened']).days
            remaining_dte = c['dte'] - days_held
            
            # Check for assignment at expiration
            if remaining_dte <= 0:
                if current_price >= c['strike']:
                    # ASSIGNED! Sell shares at strike price
                    shares = c['contracts'] * 100
                    proceeds = shares * c['strike']
                    commission = shares * self.stock_commission
                    
                    if self.qqq_shares >= shares:
                        self.cash += proceeds - commission
                        self.qqq_shares -= shares
                        
                        self.trades.append({
                            'date': date,
                            'action': 'CALL_ASSIGNED',
                            'symbol': 'QQQ',
                            'strike': c['strike'],
                            'shares': shares,
                            'proceeds': proceeds,
                            'commission': commission
                        })
                        
                        print(f"📤 {date.date()}: Call assigned - sold {shares} QQQ @ ${c['strike']:.2f}")
                    
                    positions_to_remove.append(i)
                else:
                    # Expired OTM - keep premium and shares
                    print(f"✅ {date.date()}: Call expired worthless - kept ${c['premium'] * c['contracts'] * 100:.2f}")
                    positions_to_remove.append(i)
        
        # Remove closed/assigned positions
        for i in reversed(positions_to_remove):
            self.covered_calls.pop(i)
    
    def _buy_sqqq_calls(self, date, sqqq_price: float, sqqq_iv: float, qqq_delta: float):
        """
        Buy SQQQ ATM calls on Friday to hedge QQQ delta exposure
        SQQQ is inverse 3x QQQ, so each dollar of SQQQ hedges ~$3 of QQQ
        """
        if qqq_delta <= 0:
            return None  # No long exposure to hedge
        
        # Calculate hedge notional
        # If we have 1000 delta on QQQ, we want ~333 delta on SQQQ (since SQQQ is -3x)
        target_sqqq_delta = abs(qqq_delta) / 3.0
        
        # ATM calls have ~0.5 delta
        contracts_needed = int(target_sqqq_delta / (0.50 * 100))
        contracts = max(1, min(contracts_needed, 10))  # Limit to 10 contracts max
        
        # ATM strike
        strike = sqqq_price
        
        # Get premium
        T = self.sqqq_call_dte / 365.0
        premium = bs_call_price(sqqq_price, strike, sqqq_iv, T)
        
        # Calculate cost
        total_cost = premium * contracts * 100
        commission = self.option_commission * contracts
        
        # Check cash (use hedge reserve)
        hedge_reserve = self.initial_capital * (1 - self.max_qqq_allocation)
        if total_cost + commission > hedge_reserve * 0.5:  # Use max 50% of hedge reserve
            return None
        
        if total_cost + commission > self.cash:
            return None
        
        # Execute trade
        self.cash -= (total_cost + commission)
        
        position = {
            'date_opened': date,
            'strike': strike,
            'premium': premium,
            'contracts': contracts,
            'dte': self.sqqq_call_dte,
            'type': 'sqqq_call'
        }
        
        self.sqqq_calls.append(position)
        
        trade = {
            'date': date,
            'action': 'BUY_SQQQ_CALL',
            'symbol': 'SQQQ',
            'strike': strike,
            'premium': premium,
            'contracts': contracts,
            'dte': self.sqqq_call_dte,
            'cost': total_cost,
            'commission': commission,
            'qqq_delta_hedged': qqq_delta
        }
        
        self.trades.append(trade)
        print(f"🛡️  {date.date()}: Bought {contracts} SQQQ {strike:.2f}C for ${premium:.2f} (hedging {qqq_delta:.0f} QQQ delta)")
        
        return trade
    
    def _sell_sqqq_calls(self, date, sqqq_price: float, sqqq_iv: float):
        """
        Sell all SQQQ calls on Monday
        """
        positions_to_remove = []
        
        for i, sc in enumerate(self.sqqq_calls):
            days_held = (date - sc['date_opened']).days
            remaining_dte = max(sc['dte'] - days_held, 1)
            T = remaining_dte / 365.0
            
            # Mark to market
            current_value = bs_call_price(sqqq_price, sc['strike'], sqqq_iv, T)
            
            # Sell
            proceeds = current_value * sc['contracts'] * 100
            commission = self.option_commission * sc['contracts']
            
            self.cash += proceeds - commission
            
            pnl = proceeds - (sc['premium'] * sc['contracts'] * 100)
            
            self.trades.append({
                'date': date,
                'action': 'SELL_SQQQ_CALL',
                'symbol': 'SQQQ',
                'strike': sc['strike'],
                'contracts': sc['contracts'],
                'proceeds': proceeds,
                'commission': commission,
                'pnl': pnl
            })
            
            print(f"💵 {date.date()}: Sold {sc['contracts']} SQQQ {sc['strike']:.2f}C for ${current_value:.2f} (P&L: ${pnl:.2f})")
            positions_to_remove.append(i)
        
        # Remove all positions
        for i in reversed(positions_to_remove):
            self.sqqq_calls.pop(i)
    
    def _calculate_portfolio_value(self, date, qqq_price: float, qqq_iv: float, 
                                   sqqq_price: float, sqqq_iv: float) -> float:
        """
        Calculate total portfolio value (mark-to-market)
        """
        total_value = self.cash
        
        # Value of QQQ shares
        total_value += self.qqq_shares * qqq_price
        
        # Mark-to-market short puts (liability)
        for p in self.short_puts:
            days_held = (date - p['date_opened']).days
            remaining_dte = max(p['dte'] - days_held, 1)
            T = remaining_dte / 365.0
            current_value = bs_put_price(qqq_price, p['strike'], qqq_iv, T)
            # We're short, so this is a liability
            total_value -= current_value * p['contracts'] * 100
        
        # Mark-to-market covered calls (liability)
        for c in self.covered_calls:
            days_held = (date - c['date_opened']).days
            remaining_dte = max(c['dte'] - days_held, 1)
            T = remaining_dte / 365.0
            current_value = bs_call_price(qqq_price, c['strike'], qqq_iv, T)
            # We're short, so this is a liability
            total_value -= current_value * c['contracts'] * 100
        
        # Mark-to-market SQQQ calls (asset)
        for sc in self.sqqq_calls:
            days_held = (date - sc['date_opened']).days
            remaining_dte = max(sc['dte'] - days_held, 1)
            T = remaining_dte / 365.0
            current_value = bs_call_price(sqqq_price, sc['strike'], sqqq_iv, T)
            # We're long, so this is an asset
            total_value += current_value * sc['contracts'] * 100
        
        return total_value
    
    def run_backtest(self) -> Dict:
        """
        Run the complete backtest
        """
        print(f"\n🚀 Running SQQQ-QQQ Wheel Strategy Backtest")
        print(f"{'='*60}")
        print(f"Period: {self.start_date} to {self.end_date}")
        print(f"Initial Capital: ${self.initial_capital:,.0f}")
        print(f"Max QQQ Allocation: {self.max_qqq_allocation:.0%}")
        print(f"Weekly Capital Deployment: {self.weekly_capital_pct:.1%}")
        print(f"{'='*60}\n")
        
        # Track last trade day
        last_put_sale_date = None
        
        # Iterate through each trading day
        for date in self.qqq_data.index:
            qqq_price = self.qqq_data.loc[date, 'Close']
            qqq_iv = self.qqq_data.loc[date, 'IV']
            
            # Get SQQQ data for this date
            if date in self.sqqq_data.index:
                sqqq_price = self.sqqq_data.loc[date, 'Close']
                sqqq_iv = self.sqqq_data.loc[date, 'IV']
            else:
                # Use closest available date
                sqqq_price = self.sqqq_data['Close'].iloc[-1] if len(self.sqqq_data) > 0 else 10
                sqqq_iv = self.sqqq_data['IV'].iloc[-1] if len(self.sqqq_data) > 0 else 0.5
            
            day_of_week = date.dayofweek  # 0=Monday, 4=Friday
            
            # FRIDAY: Buy SQQQ calls to hedge
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
            
            # Calculate portfolio value
            portfolio_value = self._calculate_portfolio_value(date, qqq_price, qqq_iv, 
                                                             sqqq_price, sqqq_iv)
            
            self.portfolio_values.append(portfolio_value)
            self.dates.append(date)
        
        # Calculate final results
        final_value = self.portfolio_values[-1]
        total_return = (final_value - self.initial_capital) / self.initial_capital
        
        # Calculate CAGR
        years = (self.dates[-1] - self.dates[0]).days / 365.25
        cagr = (final_value / self.initial_capital) ** (1 / years) - 1
        
        # Calculate Sharpe ratio
        portfolio_series = pd.Series(self.portfolio_values, index=self.dates)
        returns = portfolio_series.pct_change().dropna()
        sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        
        # Calculate max drawdown
        rolling_max = portfolio_series.expanding().max()
        drawdown = (portfolio_series - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # Premium collected
        put_premium = sum(t.get('premium_received', 0) for t in self.trades if t['action'] == 'SELL_PUT')
        call_premium = sum(t.get('premium_received', 0) for t in self.trades if t['action'] == 'SELL_CALL')
        total_premium = put_premium + call_premium
        
        # Assignments
        put_assignments = sum(1 for t in self.trades if t['action'] == 'PUT_ASSIGNED')
        call_assignments = sum(1 for t in self.trades if t['action'] == 'CALL_ASSIGNED')
        
        # SQQQ hedge P&L
        sqqq_pnl = sum(t.get('pnl', 0) for t in self.trades if t['action'] == 'SELL_SQQQ_CALL')
        
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
            'final_qqq_shares': self.qqq_shares,
            'final_cash': self.cash,
            'portfolio_values': self.portfolio_values,
            'dates': self.dates,
            'trades': self.trades
        }
        
        return results
    
    def print_summary(self, results: Dict):
        """Print backtest summary"""
        print(f"\n{'='*60}")
        print(f"📊 BACKTEST RESULTS SUMMARY")
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
        print(f"Put Assignments:        {results['put_assignments']:>15,}")
        print(f"Call Assignments:       {results['call_assignments']:>15,}")
        print(f"\n{'='*60}")
        print(f"PREMIUM & P&L")
        print(f"{'='*60}")
        print(f"Put Premium Collected:  ${results['put_premium']:>15,.0f}")
        print(f"Call Premium Collected: ${results['call_premium']:>15,.0f}")
        print(f"Total Premium:          ${results['total_premium_collected']:>15,.0f}")
        print(f"SQQQ Hedge P&L:         ${results['sqqq_hedge_pnl']:>15,.0f}")
        print(f"\n{'='*60}")
        print(f"FINAL POSITIONS")
        print(f"{'='*60}")
        print(f"Cash:                   ${results['final_cash']:>15,.0f}")
        print(f"QQQ Shares:             {results['final_qqq_shares']:>15,}")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    # Run backtest
    strategy = SQQQQQQWheelStrategy(
        initial_capital=100000,
        start_date='2020-01-01',
        end_date='2025-01-01'
    )
    
    results = strategy.run_backtest()
    strategy.print_summary(results)

