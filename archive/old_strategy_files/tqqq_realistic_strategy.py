#!/usr/bin/env python3
"""
Realistic TQQQ Options Protection Strategy Backtest
Implements delta-targeted strikes, regime awareness, and budget discipline

Strategy:
- DCA only when price > 200DMA (trend filter)
- Delta-targeted put strikes (10-25Δ based on regime)
- Covered call selling to finance puts (collar strategy)
- Monthly hedge budget discipline (1% of portfolio)
- Smart rolling based on delta and IV spikes
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import warnings
from scipy.stats import norm
from scipy.optimize import brentq
warnings.filterwarnings('ignore')

class RealisticTQQQStrategy:
    def __init__(self, 
                 initial_capital: float = 1000000,  # Increased for realistic testing
                 share_purchase_amount: int = 100,
                 purchase_frequency_days: int = 14,
                 protection_ratio: float = 0.60,  # 60% notional hedge
                 put_dte: int = 90,
                 roll_dte: int = 25,  # Roll when 25 DTE
                 monthly_hedge_budget_pct: float = 0.01,  # 1% of portfolio per month
                 start_date: str = '2020-01-01',
                 end_date: str = '2024-01-01'):
        """
        Initialize the realistic TQQQ options protection strategy
        """
        self.initial_capital = initial_capital
        self.share_purchase_amount = share_purchase_amount
        self.purchase_frequency_days = purchase_frequency_days
        self.protection_ratio = protection_ratio
        self.put_dte = put_dte
        self.roll_dte = roll_dte
        self.monthly_hedge_budget_pct = monthly_hedge_budget_pct
        self.start_date = start_date
        self.end_date = end_date
        
        # Strategy state
        self.cash = initial_capital
        self.shares_owned = 0
        self.total_invested = 0
        self.options_positions = []  # Both puts and calls
        self.trades = []
        self.portfolio_values = []
        self.dates = []
        self.hedge_spend_month = {}  # Track monthly hedge spending
        
        # Download TQQQ data
        self.tqqq_data = self._download_tqqq_data()
        
    def _download_tqqq_data(self) -> pd.DataFrame:
        """Download TQQQ historical data with regime indicators"""
        print("📊 Downloading TQQQ data...")
        ticker = yf.Ticker("TQQQ")
        data = ticker.history(start=self.start_date, end=self.end_date)
        
        if data.empty:
            raise ValueError("No TQQQ data found for the specified date range")
        
        # Calculate daily returns and volatility
        data['Returns'] = data['Close'].pct_change()
        data['Volatility'] = data['Returns'].rolling(window=20).std() * np.sqrt(252)
        data['20d_Vol'] = data['Volatility'].fillna(method='bfill')
        
        # Add regime indicators
        data['MA200'] = data['Close'].rolling(200).mean()
        data['MA100'] = data['Close'].rolling(100).mean()
        data['Regime'] = np.where(data['Close'] > data['MA200'], 'RISK_ON', 'RISK_OFF')
        
        # Add IV proxy with bounds
        data['IV_Proxy'] = np.clip(1.2 * data['20d_Vol'], 0.35, 1.10)
        
        print(f"✅ Downloaded {len(data)} days of TQQQ data")
        print(f"📈 RISK_ON periods: {len(data[data['Regime'] == 'RISK_ON'])} days")
        print(f"📉 RISK_OFF periods: {len(data[data['Regime'] == 'RISK_OFF'])} days")
        
        return data
    
    def _strike_for_put_delta(self, S: float, sigma: float, r: float, T: float, target_put_delta: float = -0.20) -> float:
        """Calculate strike price for target put delta using Black-Scholes"""
        if sigma <= 0 or T <= 0:
            return S * 0.8
        
        def f(K):
            d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
            return (norm.cdf(d1) - 1.0) - target_put_delta
        
        try:
            # Solve for strike between 40% and 99% of spot
            K = brentq(f, S*0.4, S*0.99, maxiter=100)
            return K
        except:
            # Fallback to simple OTM calculation
            return S * (1 - abs(target_put_delta) * 0.5)
    
    def _strike_for_call_delta(self, S: float, sigma: float, r: float, T: float, target_call_delta: float = 0.10) -> float:
        """Calculate strike price for target call delta using Black-Scholes"""
        if sigma <= 0 or T <= 0:
            return S * 1.2
        
        def f(K):
            d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
            return norm.cdf(d1) - target_call_delta
        
        try:
            # Solve for strike between 101% and 200% of spot
            K = brentq(f, S*1.01, S*2.0, maxiter=100)
            return K
        except:
            # Fallback to simple OTM calculation
            return S * (1 + target_call_delta * 0.5)
    
    def _skewed_put_iv(self, base_iv: float, put_delta: float) -> float:
        """Add volatility skew for puts (higher IV for lower deltas)"""
        extra = np.interp(abs(put_delta), [0.05, 0.25], [0.10, 0.02])
        return min(base_iv + extra, 1.20)
    
    def _skewed_call_iv(self, base_iv: float, call_delta: float) -> float:
        """Add volatility skew for calls (slightly higher IV for higher deltas)"""
        extra = np.interp(call_delta, [0.05, 0.25], [0.02, 0.05])
        return min(base_iv + extra, 1.20)
    
    def _estimate_option_premium(self, S: float, K: float, sigma: float, T: float, option_type: str = 'put') -> float:
        """Estimate option premium using Black-Scholes with realistic factors"""
        if T <= 0 or sigma <= 0:
            return 0.01
        
        r = 0.02  # 2% risk-free rate
        
        d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        
        if option_type == 'put':
            price = K * np.exp(-r*T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        else:  # call
            price = S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)
        
        # Add realistic factors
        price = max(price, 0.01)  # Minimum $0.01
        price *= 1.1  # 10% spread
        price *= (1 + sigma * 0.1)  # Volatility premium
        
        # Cap premium at reasonable levels
        if option_type == 'put':
            price = min(price, S * 0.15)  # Max 15% of stock price
        else:
            price = min(price, S * 0.20)  # Max 20% of stock price
        
        return price
    
    def _get_hedge_budget(self, date: datetime) -> float:
        """Get available hedge budget for the month"""
        mkey = (date.year, date.month)
        self.hedge_spend_month.setdefault(mkey, 0.0)
        
        current_value = self.portfolio_values[-1] if self.portfolio_values else self.initial_capital
        monthly_budget = current_value * self.monthly_hedge_budget_pct
        spent_this_month = self.hedge_spend_month[mkey]
        
        return max(0.0, monthly_budget - spent_this_month)
    
    def _buy_shares(self, date: datetime, price: float) -> Dict:
        """Buy shares and record transaction"""
        cost = self.share_purchase_amount * price
        
        if cost > self.cash:
            print(f"⚠️  Insufficient cash to buy shares on {date.strftime('%Y-%m-%d')}")
            return None
        
        self.cash -= cost
        self.shares_owned += self.share_purchase_amount
        self.total_invested += cost
        
        trade = {
            'date': date,
            'action': 'BUY_SHARES',
            'quantity': self.share_purchase_amount,
            'price': price,
            'cost': cost,
            'shares_owned': self.shares_owned,
            'cash_remaining': self.cash
        }
        
        self.trades.append(trade)
        print(f"📈 Bought {self.share_purchase_amount} TQQQ shares at ${price:.2f} on {date.strftime('%Y-%m-%d')}")
        
        return trade
    
    def _sell_covered_calls(self, date: datetime, current_price: float, iv: float, dte: int, target_delta: float = 0.10) -> Optional[Dict]:
        """Sell covered calls to finance puts"""
        if self.shares_owned == 0:
            return None
        
        # Calculate strike and premium
        strike = self._strike_for_call_delta(current_price, iv, 0.02, dte/365, target_delta)
        premium = self._estimate_option_premium(current_price, strike, iv, dte/365, 'call')
        
        # Size based on shares available
        shares_to_cover = int(self.shares_owned * 0.4)  # Cover 40% of shares
        contracts = max(1, int(np.ceil(shares_to_cover / 100)))
        
        # Calculate total premium received
        total_premium = contracts * premium * 100
        
        if total_premium > 0:
            self.cash += total_premium
            
            # Create call position
            call_position = {
                'date_opened': date,
                'strike': strike,
                'premium': premium,
                'contracts': contracts,
                'total_premium': total_premium,
                'dte': dte,
                'shares_covered': contracts * 100,
                'type': 'call'
            }
            
            self.options_positions.append(call_position)
            
            trade = {
                'date': date,
                'action': 'SELL_CALLS',
                'strike': strike,
                'premium': premium,
                'contracts': contracts,
                'premium_received': total_premium,
                'shares_covered': contracts * 100,
                'dte': dte
            }
            
            self.trades.append(trade)
            print(f"📞 Sold {contracts} call contracts at ${strike:.2f} strike, ${premium:.2f} premium on {date.strftime('%Y-%m-%d')}")
            
            return trade
        
        return None
    
    def _buy_put_protection(self, date: datetime, current_price: float, iv: float, dte: int, target_delta: float = -0.20) -> Optional[Dict]:
        """Buy put options for protection with budget discipline"""
        if self.shares_owned == 0:
            return None
        
        # Add skew to IV
        skewed_iv = self._skewed_put_iv(iv, target_delta)
        
        # Calculate strike and premium
        strike = self._strike_for_put_delta(current_price, skewed_iv, 0.02, dte/365, target_delta)
        premium = self._estimate_option_premium(current_price, strike, skewed_iv, dte/365, 'put')
        
        # Calculate number of contracts needed
        shares_to_protect = int(self.shares_owned * self.protection_ratio)
        contracts = max(1, int(np.ceil(shares_to_protect / 100)))
        
        # Calculate total cost
        total_cost = contracts * premium * 100
        
        # Check budget constraint
        budget_available = self._get_hedge_budget(date)
        available_cash = min(budget_available, self.cash)
        
        if total_cost > available_cash:
            # Try put spread to reduce cost
            strike_short = max(current_price * 0.4, strike * 0.8)
            premium_short = self._estimate_option_premium(current_price, strike_short, skewed_iv, dte/365, 'put')
            net_cost = (premium - premium_short) * contracts * 100
            
            if net_cost > available_cash:
                print(f"⚠️  Insufficient budget for put protection on {date.strftime('%Y-%m-%d')}")
                return None
            
            # Create put spread
            self.cash -= net_cost
            mkey = (date.year, date.month)
            self.hedge_spend_month[mkey] += net_cost
            
            spread_position = {
                'date_opened': date,
                'strike_long': strike,
                'strike_short': strike_short,
                'premium_long': premium,
                'premium_short': premium_short,
                'contracts': contracts,
                'total_cost': net_cost,
                'dte': dte,
                'shares_protected': contracts * 100,
                'type': 'put_spread'
            }
            
            self.options_positions.append(spread_position)
            
            trade = {
                'date': date,
                'action': 'BUY_PUT_SPREAD',
                'strike_long': strike,
                'strike_short': strike_short,
                'premium_long': premium,
                'premium_short': premium_short,
                'contracts': contracts,
                'cost': net_cost,
                'shares_protected': contracts * 100,
                'dte': dte
            }
            
            self.trades.append(trade)
            print(f"🛡️  Bought {contracts} put spread contracts ({strike:.2f}/{strike_short:.2f}) for ${net_cost:.2f} on {date.strftime('%Y-%m-%d')}")
            
            return trade
        else:
            # Buy straight puts
            self.cash -= total_cost
            mkey = (date.year, date.month)
            self.hedge_spend_month[mkey] += total_cost
            
            put_position = {
                'date_opened': date,
                'strike': strike,
                'premium': premium,
                'contracts': contracts,
                'total_cost': total_cost,
                'dte': dte,
                'shares_protected': contracts * 100,
                'type': 'put'
            }
            
            self.options_positions.append(put_position)
            
            trade = {
                'date': date,
                'action': 'BUY_PUTS',
                'strike': strike,
                'premium': premium,
                'contracts': contracts,
                'cost': total_cost,
                'shares_protected': contracts * 100,
                'dte': dte
            }
            
            self.trades.append(trade)
            print(f"🛡️  Bought {contracts} put contracts at ${strike:.2f} strike, ${premium:.2f} premium on {date.strftime('%Y-%m-%d')}")
            
            return trade
    
    def _roll_options_smart(self, date: datetime, current_price: float, iv: float) -> List[Dict]:
        """Smart option rolling based on delta and DTE"""
        rolled_trades = []
        positions_to_remove = []
        max_rolls_per_day = 5  # Prevent infinite rolling
        rolls_today = 0
        
        for i, position in enumerate(self.options_positions):
            if rolls_today >= max_rolls_per_day:
                break
            days_held = (date - position['date_opened']).days
            current_dte = position['dte'] - days_held
            
            # Skip if position is too new (less than 7 days old)
            if days_held < 7:
                continue
            
            # Determine if we should roll
            should_roll = False
            roll_reason = ""
            
            if current_dte <= self.roll_dte:
                should_roll = True
                roll_reason = "DTE"
            elif position['type'] in ['put', 'call']:
                # Check delta for single options
                if position['type'] == 'put':
                    current_delta = self._calculate_put_delta(current_price, position['strike'], iv, current_dte/365)
                    if abs(current_delta) < 0.06:  # Too far OTM
                        should_roll = True
                        roll_reason = "Delta"
                else:  # call
                    current_delta = self._calculate_call_delta(current_price, position['strike'], iv, current_dte/365)
                    if current_delta < 0.06:  # Too far OTM
                        should_roll = True
                        roll_reason = "Delta"
            
            if should_roll:
                # Close current position (simplified mark-to-market)
                if position['type'] == 'put':
                    remaining_value = self._estimate_option_premium(current_price, position['strike'], iv, current_dte/365, 'put') * position['contracts'] * 100
                elif position['type'] == 'call':
                    remaining_value = self._estimate_option_premium(current_price, position['strike'], iv, current_dte/365, 'call') * position['contracts'] * 100
                else:  # spread
                    remaining_value = position['total_cost'] * (current_dte / position['dte']) * 0.3
                
                self.cash += remaining_value
                
                # Determine new position parameters based on regime
                regime = self.tqqq_data.loc[date, 'Regime']
                if regime == 'RISK_ON':
                    target_put_delta = -0.20
                    target_call_delta = 0.10
                else:
                    target_put_delta = -0.10
                    target_call_delta = 0.05
                
                # Create new position
                if position['type'] == 'put':
                    new_strike = self._strike_for_put_delta(current_price, iv, 0.02, self.put_dte/365, target_put_delta)
                    
                    # Skip if new strike is too close to old strike (prevent infinite rolling)
                    if abs(new_strike - position['strike']) / position['strike'] < 0.02:  # Less than 2% difference
                        continue
                    
                    new_premium = self._estimate_option_premium(current_price, new_strike, iv, self.put_dte/365, 'put')
                    new_cost = position['contracts'] * new_premium * 100
                    
                    if new_cost <= self.cash:
                        self.cash -= new_cost
                        mkey = (date.year, date.month)
                        self.hedge_spend_month.setdefault(mkey, 0.0)
                        self.hedge_spend_month[mkey] += new_cost
                        
                        new_position = {
                            'date_opened': date,
                            'strike': new_strike,
                            'premium': new_premium,
                            'contracts': position['contracts'],
                            'total_cost': new_cost,
                            'dte': self.put_dte,
                            'shares_protected': position['shares_protected'],
                            'type': 'put'
                        }
                        
                        self.options_positions.append(new_position)
                        positions_to_remove.append(i)
                        
                        trade = {
                            'date': date,
                            'action': 'ROLL_PUTS',
                            'old_strike': position['strike'],
                            'new_strike': new_strike,
                            'old_premium': position['premium'],
                            'new_premium': new_premium,
                            'contracts': position['contracts'],
                            'net_cost': new_cost - remaining_value,
                            'dte': self.put_dte,
                            'reason': roll_reason
                        }
                        
                        self.trades.append(trade)
                        rolled_trades.append(trade)
                        rolls_today += 1
                        print(f"🔄 Rolled {position['contracts']} put contracts from ${position['strike']:.2f} to ${new_strike:.2f} on {date.strftime('%Y-%m-%d')} ({roll_reason})")
                
                elif position['type'] == 'call':
                    new_strike = self._strike_for_call_delta(current_price, iv, 0.02, self.put_dte/365, target_call_delta)
                    
                    # Skip if new strike is too close to old strike (prevent infinite rolling)
                    if abs(new_strike - position['strike']) / position['strike'] < 0.02:  # Less than 2% difference
                        continue
                    
                    new_premium = self._estimate_option_premium(current_price, new_strike, iv, self.put_dte/365, 'call')
                    new_premium_received = position['contracts'] * new_premium * 100
                    
                    self.cash += new_premium_received
                    
                    new_position = {
                        'date_opened': date,
                        'strike': new_strike,
                        'premium': new_premium,
                        'contracts': position['contracts'],
                        'total_premium': new_premium_received,
                        'dte': self.put_dte,
                        'shares_covered': position['shares_covered'],
                        'type': 'call'
                    }
                    
                    self.options_positions.append(new_position)
                    positions_to_remove.append(i)
                    
                    trade = {
                        'date': date,
                        'action': 'ROLL_CALLS',
                        'old_strike': position['strike'],
                        'new_strike': new_strike,
                        'old_premium': position['premium'],
                        'new_premium': new_premium,
                        'contracts': position['contracts'],
                        'net_premium': new_premium_received - remaining_value,
                        'dte': self.put_dte,
                        'reason': roll_reason
                    }
                    
                    self.trades.append(trade)
                    rolled_trades.append(trade)
                    rolls_today += 1
                    print(f"🔄 Rolled {position['contracts']} call contracts from ${position['strike']:.2f} to ${new_strike:.2f} on {date.strftime('%Y-%m-%d')} ({roll_reason})")
        
        # Remove rolled positions
        for i in reversed(positions_to_remove):
            self.options_positions.pop(i)
        
        return rolled_trades
    
    def _calculate_put_delta(self, S: float, K: float, sigma: float, T: float) -> float:
        """Calculate put delta"""
        if T <= 0 or sigma <= 0:
            return 0.0
        
        d1 = (np.log(S/K) + (0.02 + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        return norm.cdf(d1) - 1.0
    
    def _calculate_call_delta(self, S: float, K: float, sigma: float, T: float) -> float:
        """Calculate call delta"""
        if T <= 0 or sigma <= 0:
            return 0.0
        
        d1 = (np.log(S/K) + (0.02 + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        return norm.cdf(d1)
    
    def _calculate_portfolio_value(self, date: datetime, current_price: float) -> float:
        """Calculate total portfolio value with option values"""
        shares_value = self.shares_owned * current_price
        
        # Calculate options value
        options_value = 0
        for position in self.options_positions:
            days_held = (date - position['date_opened']).days
            current_dte = position['dte'] - days_held
            
            if current_dte > 0:
                if position['type'] == 'put':
                    # Long put value
                    intrinsic_value = max(0, position['strike'] - current_price)
                    time_value = position['premium'] * (current_dte / position['dte']) * 0.5
                    option_value = (intrinsic_value + time_value) * position['contracts'] * 100
                    options_value += option_value
                
                elif position['type'] == 'call':
                    # Short call value (negative)
                    intrinsic_value = max(0, current_price - position['strike'])
                    time_value = position['premium'] * (current_dte / position['dte']) * 0.5
                    option_value = -(intrinsic_value + time_value) * position['contracts'] * 100
                    options_value += option_value
                
                elif position['type'] == 'put_spread':
                    # Put spread value
                    long_value = max(0, position['strike_long'] - current_price)
                    short_value = max(0, position['strike_short'] - current_price)
                    spread_value = (long_value - short_value) * position['contracts'] * 100
                    options_value += spread_value
        
        return self.cash + shares_value + options_value
    
    def run_backtest(self) -> Dict:
        """Run the complete backtest with regime awareness"""
        print("🚀 Starting Realistic TQQQ Options Protection Strategy Backtest")
        print("=" * 60)
        
        # Get trading dates
        trading_dates = self.tqqq_data.index.tolist()
        last_purchase_date = None
        
        for i, date in enumerate(trading_dates):
            current_price = self.tqqq_data.loc[date, 'Close']
            iv = self.tqqq_data.loc[date, 'IV_Proxy']
            regime = self.tqqq_data.loc[date, 'Regime']
            
            # Check if it's time to buy shares (only in RISK_ON)
            if (last_purchase_date is None or 
                (date - last_purchase_date).days >= self.purchase_frequency_days):
                
                if regime == 'RISK_ON':
                    # Buy shares in RISK_ON regime
                    self._buy_shares(date, current_price)
                    
                    # Set hedge parameters for RISK_ON
                    target_put_delta = -0.20
                    target_call_delta = 0.10
                    self.protection_ratio = 0.60
                else:
                    # No new shares in RISK_OFF, increase hedge
                    target_put_delta = -0.10
                    target_call_delta = 0.05
                    self.protection_ratio = 0.80
                
                # Sell covered calls to finance puts
                if regime == 'RISK_ON':
                    self._sell_covered_calls(date, current_price, iv, self.put_dte, target_call_delta)
                
                # Buy put protection
                self._buy_put_protection(date, current_price, iv, self.put_dte, target_put_delta)
                
                last_purchase_date = date
            
            # Smart option rolling
            self._roll_options_smart(date, current_price, iv)
            
            # Calculate portfolio value
            portfolio_value = self._calculate_portfolio_value(date, current_price)
            self.portfolio_values.append(portfolio_value)
            self.dates.append(date)
        
        # Calculate final results
        final_value = self.portfolio_values[-1]
        total_return = (final_value - self.initial_capital) / self.initial_capital
        
        # Calculate benchmark (buy and hold TQQQ)
        initial_price = self.tqqq_data['Close'].iloc[0]
        final_price = self.tqqq_data['Close'].iloc[-1]
        benchmark_return = (final_price - initial_price) / initial_price
        
        results = {
            'initial_capital': self.initial_capital,
            'final_value': final_value,
            'total_return': total_return,
            'benchmark_return': benchmark_return,
            'excess_return': total_return - benchmark_return,
            'shares_owned': self.shares_owned,
            'total_invested': self.total_invested,
            'cash_remaining': self.cash,
            'total_trades': len(self.trades),
            'options_positions': len(self.options_positions),
            'portfolio_values': self.portfolio_values,
            'dates': self.dates,
            'trades': self.trades
        }
        
        return results
    
    def plot_results(self, results: Dict):
        """Plot backtest results"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Portfolio value over time
        ax1.plot(results['dates'], results['portfolio_values'], label='Strategy', linewidth=2)
        
        # Benchmark (buy and hold TQQQ)
        initial_price = self.tqqq_data['Close'].iloc[0]
        benchmark_values = [initial_price * (price / initial_price) * (self.initial_capital / initial_price) 
                           for price in self.tqqq_data['Close']]
        ax1.plot(self.tqqq_data.index, benchmark_values, label='Buy & Hold TQQQ', linewidth=2, alpha=0.7)
        
        # Add regime shading
        risk_off_periods = self.tqqq_data[self.tqqq_data['Regime'] == 'RISK_OFF'].index
        for period in risk_off_periods:
            ax1.axvspan(period, period, alpha=0.3, color='red', label='RISK_OFF' if period == risk_off_periods[0] else "")
        
        ax1.set_title('Portfolio Value Over Time')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Drawdown analysis
        portfolio_series = pd.Series(results['portfolio_values'], index=results['dates'])
        rolling_max = portfolio_series.expanding().max()
        drawdown = (portfolio_series - rolling_max) / rolling_max
        
        ax2.fill_between(results['dates'], drawdown, 0, alpha=0.3, color='red')
        ax2.set_title('Portfolio Drawdown')
        ax2.set_ylabel('Drawdown (%)')
        ax2.grid(True, alpha=0.3)
        
        # Trade analysis
        trades_df = pd.DataFrame(results['trades'])
        if not trades_df.empty:
            trade_counts = trades_df['action'].value_counts()
            ax3.pie(trade_counts.values, labels=trade_counts.index, autopct='%1.1f%%')
            ax3.set_title('Trade Distribution')
        
        # Performance metrics
        metrics = [
            f"Total Return: {results['total_return']:.1%}",
            f"Benchmark Return: {results['benchmark_return']:.1%}",
            f"Excess Return: {results['excess_return']:.1%}",
            f"Final Value: ${results['final_value']:,.0f}",
            f"Shares Owned: {results['shares_owned']:,}",
            f"Total Trades: {results['total_trades']}"
        ]
        
        ax4.text(0.1, 0.9, '\n'.join(metrics), transform=ax4.transAxes, 
                fontsize=12, verticalalignment='top', fontfamily='monospace')
        ax4.set_title('Performance Metrics')
        ax4.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def print_summary(self, results: Dict):
        """Print backtest summary"""
        print("\n" + "="*60)
        print("📊 REALISTIC TQQQ OPTIONS PROTECTION STRATEGY RESULTS")
        print("="*60)
        print(f"Initial Capital: ${results['initial_capital']:,.0f}")
        print(f"Final Value: ${results['final_value']:,.0f}")
        print(f"Total Return: {results['total_return']:.1%}")
        print(f"Benchmark Return: {results['benchmark_return']:.1%}")
        print(f"Excess Return: {results['excess_return']:.1%}")
        print(f"Shares Owned: {results['shares_owned']:,}")
        print(f"Total Invested: ${results['total_invested']:,.0f}")
        print(f"Cash Remaining: ${results['cash_remaining']:,.0f}")
        print(f"Total Trades: {results['total_trades']}")
        print(f"Active Options Positions: {results['options_positions']}")
        
        # Calculate some additional metrics
        portfolio_series = pd.Series(results['portfolio_values'], index=results['dates'])
        returns = portfolio_series.pct_change().dropna()
        
        if len(returns) > 0:
            volatility = returns.std() * np.sqrt(252)
            sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
            max_drawdown = (portfolio_series / portfolio_series.expanding().max() - 1).min()
            
            print(f"\nRisk Metrics:")
            print(f"Annualized Volatility: {volatility:.1%}")
            print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
            print(f"Maximum Drawdown: {max_drawdown:.1%}")
            
            # Calculate hedge spending
            total_hedge_spend = sum(self.hedge_spend_month.values())
            print(f"Total Hedge Spending: ${total_hedge_spend:,.0f}")
            print(f"Average Monthly Hedge Spend: ${total_hedge_spend / len(self.hedge_spend_month):,.0f}")

def main():
    """Run the realistic TQQQ options strategy backtest"""
    print("🏦 Realistic TQQQ Options Protection Strategy Backtest")
    print("=" * 60)
    
    # Strategy parameters - realistic and profitable
    strategy = RealisticTQQQStrategy(
        initial_capital=1000000,  # $1M for realistic testing
        share_purchase_amount=100,
        purchase_frequency_days=14,  # Every 2 weeks
        protection_ratio=0.60,  # 60% notional hedge
        put_dte=90,  # 90 days to expiration
        roll_dte=25,  # Roll when 25 days left
        monthly_hedge_budget_pct=0.01,  # 1% of portfolio per month
        start_date='2020-01-01',
        end_date='2024-01-01'
    )
    
    # Run backtest
    results = strategy.run_backtest()
    
    # Print results
    strategy.print_summary(results)
    
    # Plot results
    strategy.plot_results(results)
    
    # Save detailed results
    results_df = pd.DataFrame({
        'Date': results['dates'],
        'Portfolio_Value': results['portfolio_values']
    })
    results_df.to_csv('tqqq_realistic_strategy_results.csv', index=False)
    print(f"\n💾 Detailed results saved to 'tqqq_realistic_strategy_results.csv'")

if __name__ == "__main__":
    main()
