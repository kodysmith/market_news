#!/usr/bin/env python3
"""
Multi-Asset Wheel Strategy with Adaptive Hedging

Diversify across major US indices:
- SPY (S&P 500) - Largest, most liquid
- QQQ (Nasdaq-100) - Tech-heavy
- DIA (Dow Jones) - Blue chips
- IWM (Russell 2000) - Small caps

Benefits:
- Reduced single-asset concentration
- Lower correlation = smoother returns
- More trading opportunities
- Better risk-adjusted returns
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

# Reuse pricing functions
from sqqq_qqq_wheel_strategy import (
    bs_put_price, bs_call_price, 
    strike_for_put_delta, strike_for_call_delta,
    put_delta, call_delta
)


class MultiAssetWheelStrategy:
    """
    Multi-asset wheel strategy across SPY, QQQ, DIA, IWM
    """
    
    def __init__(self,
                 initial_capital: float = 100000,
                 assets: List[str] = ['SPY', 'QQQ', 'DIA', 'IWM'],
                 allocation_method: str = 'equal',  # 'equal', 'vol_weighted', 'liquidity_weighted'
                 enable_adaptive_hedge: bool = True,
                 put_delta_target: float = -0.30,
                 put_dte: int = 14,
                 call_delta_target: float = 0.30,
                 call_dte: int = 30,
                 weeknight_hedge_dte: int = 2,
                 weekend_hedge_dte: int = 14,
                 start_date: str = '2020-01-01',
                 end_date: str = '2025-01-01'):
        """
        Initialize multi-asset strategy
        
        Args:
            initial_capital: Starting capital
            assets: List of ETF tickers to trade
            allocation_method: How to allocate capital across assets
            enable_adaptive_hedge: Use overnight hedging
            put_delta_target: Target delta for puts
            put_dte: Days to expiration for puts
            call_delta_target: Target delta for calls
            call_dte: Days to expiration for calls
            weeknight_hedge_dte: DTE for Mon-Thu hedges
            weekend_hedge_dte: DTE for Friday hedges
        """
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.assets = assets
        self.allocation_method = allocation_method
        self.enable_adaptive_hedge = enable_adaptive_hedge
        self.put_delta_target = put_delta_target
        self.put_dte = put_dte
        self.call_delta_target = call_delta_target
        self.call_dte = call_dte
        self.weeknight_hedge_dte = weeknight_hedge_dte
        self.weekend_hedge_dte = weekend_hedge_dte
        self.start_date = start_date
        self.end_date = end_date
        
        # Commission structure
        self.option_commission = 0.65
        self.stock_commission = 0.01
        
        # Portfolio state (per asset)
        self.shares = {asset: 0 for asset in assets}
        self.short_puts = {asset: [] for asset in assets}
        self.covered_calls = {asset: [] for asset in assets}
        self.overnight_puts = {asset: [] for asset in assets}
        
        # Tracking
        self.trades = []
        self.portfolio_values = []
        self.dates = []
        self.overnight_hedge_pnl = 0.0
        
        # Load data for all assets
        self.asset_data = {}
        self._load_all_data()
        
        # Calculate allocations
        self.allocations = self._calculate_allocations()
        
    def _load_all_data(self):
        """Download data for all assets"""
        print(f"📊 Downloading data for {len(self.assets)} assets...")
        
        for asset in self.assets:
            print(f"  - Loading {asset}...")
            ticker = yf.Ticker(asset)
            data = ticker.history(start=self.start_date, end=self.end_date)
            
            if data.empty:
                raise ValueError(f"No data for {asset}")
            
            # Calculate metrics
            data['Returns'] = data['Close'].pct_change()
            data['IV'] = data['Returns'].rolling(20).std() * np.sqrt(252)
            data['IV'] = data['IV'].fillna(method='bfill').fillna(0.25)
            data['IV'] = data['IV'].clip(0.10, 0.80)
            
            # Regime
            data['MA200'] = data['Close'].rolling(200).mean()
            data['Regime'] = np.where(data['Close'] > data['MA200'], 'RISK_ON', 'RISK_OFF')
            
            self.asset_data[asset] = data
            print(f"    ✅ {len(data)} days")
        
        print(f"✅ All data loaded\n")
    
    def _calculate_allocations(self) -> Dict[str, float]:
        """Calculate capital allocation per asset"""
        if self.allocation_method == 'equal':
            # Equal weight
            weight = 1.0 / len(self.assets)
            return {asset: weight for asset in self.assets}
        
        elif self.allocation_method == 'vol_weighted':
            # Inverse volatility weighting (lower vol = higher weight)
            vols = {}
            for asset in self.assets:
                vol = self.asset_data[asset]['IV'].mean()
                vols[asset] = vol
            
            # Inverse weights
            inv_vols = {asset: 1.0/vol for asset, vol in vols.items()}
            total_inv = sum(inv_vols.values())
            return {asset: w/total_inv for asset, w in inv_vols.items()}
        
        elif self.allocation_method == 'liquidity_weighted':
            # Weight by typical liquidity (SPY > QQQ > DIA > IWM)
            liquidity_scores = {
                'SPY': 0.40,  # Most liquid
                'QQQ': 0.30,
                'DIA': 0.20,
                'IWM': 0.10   # Least liquid (but more volatile)
            }
            return {asset: liquidity_scores.get(asset, 0.25) for asset in self.assets}
        
        else:
            # Default to equal
            weight = 1.0 / len(self.assets)
            return {asset: weight for asset in self.assets}
    
    def _get_net_delta(self, asset: str, current_price: float, iv: float, date) -> float:
        """Calculate net delta for an asset"""
        total_delta = 0.0
        
        # Shares
        total_delta += self.shares[asset]
        
        # Short puts
        for p in self.short_puts[asset]:
            days_held = (date - p['date_opened']).days
            remaining_dte = max(p['dte'] - days_held, 1)
            T = remaining_dte / 365.0
            p_delta = put_delta(current_price, p['strike'], iv, T)
            total_delta += -1 * p_delta * p['contracts'] * 100
        
        # Covered calls
        for c in self.covered_calls[asset]:
            days_held = (date - c['date_opened']).days
            remaining_dte = max(c['dte'] - days_held, 1)
            T = remaining_dte / 365.0
            c_delta = call_delta(current_price, c['strike'], iv, T)
            total_delta += -1 * c_delta * c['contracts'] * 100
        
        return total_delta
    
    def _sell_put(self, asset: str, date, current_price: float, iv: float) -> Optional[Dict]:
        """Sell a put on specified asset"""
        # Calculate position size based on allocation
        allocated_capital = self.initial_capital * self.allocations[asset]
        trade_size = allocated_capital * 0.01  # 1% per week per asset
        
        # Solve for strike
        T = self.put_dte / 365.0
        strike = strike_for_put_delta(current_price, iv, T, self.put_delta_target)
        premium = bs_put_price(current_price, strike, iv, T)
        
        # Size position
        margin_per_contract = strike * 100 * 0.20
        contracts = max(1, int(trade_size / margin_per_contract))
        
        total_margin = margin_per_contract * contracts
        commission = self.option_commission * contracts
        
        if total_margin + commission > self.cash * 0.8:
            return None
        
        # Execute
        premium_received = premium * contracts * 100
        self.cash += premium_received - commission
        
        position = {
            'date_opened': date,
            'strike': strike,
            'premium': premium,
            'contracts': contracts,
            'dte': self.put_dte,
            'type': 'put',
            'asset': asset
        }
        
        self.short_puts[asset].append(position)
        
        trade = {
            'date': date,
            'action': 'SELL_PUT',
            'asset': asset,
            'strike': strike,
            'premium': premium,
            'contracts': contracts,
            'dte': self.put_dte,
            'premium_received': premium_received,
            'commission': commission
        }
        
        self.trades.append(trade)
        return trade
    
    def _check_put_assignments(self, asset: str, date, current_price: float, iv: float):
        """Check for put assignments on asset"""
        positions_to_remove = []
        
        for i, p in enumerate(self.short_puts[asset]):
            days_held = (date - p['date_opened']).days
            remaining_dte = p['dte'] - days_held
            
            if remaining_dte <= 0:
                if current_price < p['strike']:
                    # Assigned
                    shares = p['contracts'] * 100
                    cost = shares * p['strike']
                    commission = shares * self.stock_commission
                    
                    if cost + commission <= self.cash:
                        self.cash -= (cost + commission)
                        self.shares[asset] += shares
                        
                        self.trades.append({
                            'date': date,
                            'action': 'PUT_ASSIGNED',
                            'asset': asset,
                            'strike': p['strike'],
                            'shares': shares,
                            'cost': cost
                        })
                        
                        # Sell covered call
                        self._sell_covered_call(asset, date, current_price, iv, shares)
                    
                    positions_to_remove.append(i)
                else:
                    # Expired worthless
                    positions_to_remove.append(i)
            
            # Close at 50% profit
            elif remaining_dte > 0:
                T = remaining_dte / 365.0
                current_value = bs_put_price(current_price, p['strike'], iv, T)
                profit_pct = (p['premium'] - current_value) / p['premium']
                
                if profit_pct >= 0.50:
                    close_cost = current_value * p['contracts'] * 100
                    commission = self.option_commission * p['contracts']
                    self.cash -= (close_cost + commission)
                    
                    self.trades.append({
                        'date': date,
                        'action': 'CLOSE_PUT',
                        'asset': asset,
                        'strike': p['strike'],
                        'contracts': p['contracts'],
                        'profit_pct': profit_pct
                    })
                    
                    positions_to_remove.append(i)
        
        for i in reversed(positions_to_remove):
            self.short_puts[asset].pop(i)
    
    def _sell_covered_call(self, asset: str, date, current_price: float, iv: float, shares: int):
        """Sell covered call after assignment"""
        if shares < 100:
            return None
        
        contracts = shares // 100
        T = self.call_dte / 365.0
        strike = strike_for_call_delta(current_price, iv, T, self.call_delta_target)
        premium = bs_call_price(current_price, strike, iv, T)
        
        premium_received = premium * contracts * 100
        commission = self.option_commission * contracts
        self.cash += premium_received - commission
        
        position = {
            'date_opened': date,
            'strike': strike,
            'premium': premium,
            'contracts': contracts,
            'dte': self.call_dte,
            'type': 'call',
            'asset': asset
        }
        
        self.covered_calls[asset].append(position)
        
        self.trades.append({
            'date': date,
            'action': 'SELL_CALL',
            'asset': asset,
            'strike': strike,
            'premium': premium,
            'contracts': contracts,
            'premium_received': premium_received
        })
        
        return position
    
    def _check_call_assignments(self, asset: str, date, current_price: float):
        """Check for call assignments"""
        positions_to_remove = []
        
        for i, c in enumerate(self.covered_calls[asset]):
            days_held = (date - c['date_opened']).days
            remaining_dte = c['dte'] - days_held
            
            if remaining_dte <= 0:
                if current_price >= c['strike']:
                    # Assigned
                    shares = c['contracts'] * 100
                    proceeds = shares * c['strike']
                    commission = shares * self.stock_commission
                    
                    if self.shares[asset] >= shares:
                        self.cash += proceeds - commission
                        self.shares[asset] -= shares
                        
                        self.trades.append({
                            'date': date,
                            'action': 'CALL_ASSIGNED',
                            'asset': asset,
                            'shares': shares,
                            'proceeds': proceeds
                        })
                    
                    positions_to_remove.append(i)
                else:
                    positions_to_remove.append(i)
        
        for i in reversed(positions_to_remove):
            self.covered_calls[asset].pop(i)
    
    def _buy_overnight_hedge(self, asset: str, date, current_price: float, iv: float, is_friday: bool):
        """Buy overnight hedge for specific asset"""
        delta = self._get_net_delta(asset, current_price, iv, date)
        
        if delta <= 0:
            return None
        
        contracts_needed = max(1, int(abs(delta) / 100))
        
        dte = self.weekend_hedge_dte if is_friday else self.weeknight_hedge_dte
        T = dte / 365.0
        
        strike = strike_for_put_delta(current_price, iv, T, -0.50)
        premium = bs_put_price(current_price, strike, iv, T)
        
        total_cost = premium * contracts_needed * 100
        commission = self.option_commission * contracts_needed
        
        max_cost = self.initial_capital * 0.01
        if total_cost + commission > max_cost or total_cost + commission > self.cash:
            return None
        
        self.cash -= (total_cost + commission)
        
        position = {
            'date_opened': date,
            'strike': strike,
            'premium': premium,
            'contracts': contracts_needed,
            'dte': dte,
            'cost': total_cost + commission,
            'asset': asset
        }
        
        self.overnight_puts[asset].append(position)
        
        self.trades.append({
            'date': date,
            'action': 'BUY_OVERNIGHT_HEDGE',
            'asset': asset,
            'strike': strike,
            'premium': premium,
            'contracts': contracts_needed,
            'dte': dte,
            'cost': total_cost
        })
        
        return position
    
    def _sell_overnight_hedge(self, asset: str, date, current_price: float, iv: float):
        """Sell overnight hedge"""
        positions_to_remove = []
        
        for i, p in enumerate(self.overnight_puts[asset]):
            days_held = (date - p['date_opened']).days
            
            if days_held >= 1:
                remaining_dte = max(p['dte'] - days_held, 0.1)
                T = remaining_dte / 365.0
                current_value = bs_put_price(current_price, p['strike'], iv, T)
                
                proceeds = current_value * p['contracts'] * 100
                commission = self.option_commission * p['contracts']
                self.cash += proceeds - commission
                
                pnl = proceeds - p['cost']
                self.overnight_hedge_pnl += pnl
                
                self.trades.append({
                    'date': date,
                    'action': 'SELL_OVERNIGHT_HEDGE',
                    'asset': asset,
                    'pnl': pnl
                })
                
                positions_to_remove.append(i)
        
        for i in reversed(positions_to_remove):
            self.overnight_puts[asset].pop(i)
    
    def _calculate_portfolio_value(self, date) -> float:
        """Calculate total portfolio value across all assets"""
        total_value = self.cash
        
        for asset in self.assets:
            if date not in self.asset_data[asset].index:
                continue
            
            price = self.asset_data[asset].loc[date, 'Close']
            iv = self.asset_data[asset].loc[date, 'IV']
            
            # Shares
            total_value += self.shares[asset] * price
            
            # Short puts (liability)
            for p in self.short_puts[asset]:
                days_held = (date - p['date_opened']).days
                remaining_dte = max(p['dte'] - days_held, 1)
                T = remaining_dte / 365.0
                value = bs_put_price(price, p['strike'], iv, T)
                total_value -= value * p['contracts'] * 100
            
            # Covered calls (liability)
            for c in self.covered_calls[asset]:
                days_held = (date - c['date_opened']).days
                remaining_dte = max(c['dte'] - days_held, 1)
                T = remaining_dte / 365.0
                value = bs_call_price(price, c['strike'], iv, T)
                total_value -= value * c['contracts'] * 100
            
            # Overnight puts (asset)
            for op in self.overnight_puts[asset]:
                days_held = (date - op['date_opened']).days
                remaining_dte = max(op['dte'] - days_held, 0.1)
                T = remaining_dte / 365.0
                value = bs_put_price(price, op['strike'], iv, T)
                total_value += value * op['contracts'] * 100
        
        return total_value
    
    def run_backtest(self) -> Dict:
        """Run multi-asset backtest"""
        print(f"\n🚀 Running MULTI-ASSET Wheel Strategy Backtest")
        print(f"{'='*60}")
        print(f"Assets: {', '.join(self.assets)}")
        print(f"Initial Capital: ${self.initial_capital:,.0f}")
        print(f"Allocation Method: {self.allocation_method}")
        print(f"Allocations:")
        for asset, alloc in self.allocations.items():
            print(f"  - {asset}: {alloc:.1%}")
        print(f"Adaptive Hedging: {'ENABLED' if self.enable_adaptive_hedge else 'DISABLED'}")
        print(f"{'='*60}\n")
        
        # Get common dates (intersection of all assets)
        common_dates = self.asset_data[self.assets[0]].index
        for asset in self.assets[1:]:
            common_dates = common_dates.intersection(self.asset_data[asset].index)
        
        common_dates = sorted(common_dates)
        last_put_sale = {asset: None for asset in self.assets}
        
        for date in common_dates:
            day_of_week = date.dayofweek
            
            # Morning: Sell overnight hedges
            if self.enable_adaptive_hedge:
                for asset in self.assets:
                    if len(self.overnight_puts[asset]) > 0:
                        price = self.asset_data[asset].loc[date, 'Close']
                        iv = self.asset_data[asset].loc[date, 'IV']
                        self._sell_overnight_hedge(asset, date, price, iv)
            
            # Weekly put sales (Monday for each asset)
            if day_of_week == 0:
                for asset in self.assets:
                    if last_put_sale[asset] is None or (date - last_put_sale[asset]).days >= 7:
                        price = self.asset_data[asset].loc[date, 'Close']
                        iv = self.asset_data[asset].loc[date, 'IV']
                        self._sell_put(asset, date, price, iv)
                        last_put_sale[asset] = date
            
            # Check assignments daily
            for asset in self.assets:
                price = self.asset_data[asset].loc[date, 'Close']
                iv = self.asset_data[asset].loc[date, 'IV']
                self._check_put_assignments(asset, date, price, iv)
                self._check_call_assignments(asset, date, price)
            
            # Evening: Buy overnight hedges
            if self.enable_adaptive_hedge:
                is_friday = (day_of_week == 4)
                for asset in self.assets:
                    price = self.asset_data[asset].loc[date, 'Close']
                    iv = self.asset_data[asset].loc[date, 'IV']
                    delta = self._get_net_delta(asset, price, iv, date)
                    if delta > 0:
                        self._buy_overnight_hedge(asset, date, price, iv, is_friday)
            
            # Calculate portfolio value
            portfolio_value = self._calculate_portfolio_value(date)
            self.portfolio_values.append(portfolio_value)
            self.dates.append(date)
        
        # Calculate results
        return self._calculate_results()
    
    def _calculate_results(self) -> Dict:
        """Calculate final results"""
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
        
        # Premium by asset
        put_premium_by_asset = {}
        call_premium_by_asset = {}
        for asset in self.assets:
            put_premium_by_asset[asset] = sum(
                t.get('premium_received', 0) for t in self.trades 
                if t['action'] == 'SELL_PUT' and t.get('asset') == asset
            )
            call_premium_by_asset[asset] = sum(
                t.get('premium_received', 0) for t in self.trades 
                if t['action'] == 'SELL_CALL' and t.get('asset') == asset
            )
        
        total_premium = sum(put_premium_by_asset.values()) + sum(call_premium_by_asset.values())
        
        results = {
            'initial_capital': self.initial_capital,
            'final_value': final_value,
            'total_return': total_return,
            'cagr': cagr,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'total_trades': len(self.trades),
            'total_premium_collected': total_premium,
            'put_premium_by_asset': put_premium_by_asset,
            'call_premium_by_asset': call_premium_by_asset,
            'overnight_hedge_pnl': self.overnight_hedge_pnl,
            'final_shares': self.shares.copy(),
            'final_cash': self.cash,
            'portfolio_values': self.portfolio_values,
            'dates': self.dates,
            'trades': self.trades,
            'allocations': self.allocations
        }
        
        return results
    
    def print_summary(self, results: Dict):
        """Print results summary"""
        print(f"\n{'='*60}")
        print(f"📊 MULTI-ASSET STRATEGY RESULTS")
        print(f"{'='*60}")
        print(f"Initial Capital:        ${results['initial_capital']:>15,.0f}")
        print(f"Final Value:            ${results['final_value']:>15,.0f}")
        print(f"Total Return:           {results['total_return']:>15.1%}")
        print(f"CAGR:                   {results['cagr']:>15.1%}")
        print(f"Sharpe Ratio:           {results['sharpe_ratio']:>15.2f}")
        print(f"Max Drawdown:           {results['max_drawdown']:>15.1%}")
        print(f"\n{'='*60}")
        print(f"PREMIUM BY ASSET")
        print(f"{'='*60}")
        for asset in self.assets:
            put_prem = results['put_premium_by_asset'].get(asset, 0)
            call_prem = results['call_premium_by_asset'].get(asset, 0)
            total = put_prem + call_prem
            print(f"{asset:>6}: ${total:>12,.0f} (Put: ${put_prem:>10,.0f}, Call: ${call_prem:>10,.0f})")
        print(f"\nTotal Premium:          ${results['total_premium_collected']:>15,.0f}")
        print(f"Overnight Hedge P&L:    ${results['overnight_hedge_pnl']:>15,.0f}")
        print(f"\n{'='*60}")
        print(f"FINAL POSITIONS")
        print(f"{'='*60}")
        print(f"Cash:                   ${results['final_cash']:>15,.0f}")
        for asset in self.assets:
            shares = results['final_shares'].get(asset, 0)
            if shares > 0:
                print(f"{asset} Shares:             {shares:>15,}")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    # Run multi-asset comparison
    print("\n" + "="*70)
    print("MULTI-ASSET DIVERSIFICATION TEST")
    print("="*70)
    
    # Test different allocation methods
    for method in ['equal', 'liquidity_weighted', 'vol_weighted']:
        print(f"\n🔷 Testing {method.upper()} allocation...")
        
        strategy = MultiAssetWheelStrategy(
            initial_capital=100000,
            assets=['SPY', 'QQQ', 'DIA', 'IWM'],
            allocation_method=method,
            enable_adaptive_hedge=True,
            weeknight_hedge_dte=2,
            weekend_hedge_dte=14,
            start_date='2020-01-01',
            end_date='2025-01-01'
        )
        
        results = strategy.run_backtest()
        strategy.print_summary(results)

