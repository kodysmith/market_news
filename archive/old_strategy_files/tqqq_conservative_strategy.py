#!/usr/bin/env python3
"""
Conservative TQQQ Options Protection Strategy Backtest
Very conservative approach with minimal options costs

Strategy:
- Buy 25 shares of TQQQ every 2 weeks (very small position)
- Buy put options to protect 25% of position size (minimal protection)
- Put options 90 days out, 1 standard deviation OTM (closer to money)
- Roll options when they reach ~30 days to expiration
- Very conservative option pricing
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class ConservativeTQQQStrategy:
    def __init__(self, 
                 initial_capital: float = 100000,
                 share_purchase_amount: int = 25,  # Very small position
                 purchase_frequency_days: int = 14,
                 protection_ratio: float = 0.25,  # Minimal protection
                 put_dte: int = 90,
                 roll_dte: int = 30,
                 otm_deviations: float = 1.0,  # Closer to money
                 start_date: str = '2020-01-01',
                 end_date: str = '2024-01-01'):
        """
        Initialize the conservative TQQQ options protection strategy
        """
        self.initial_capital = initial_capital
        self.share_purchase_amount = share_purchase_amount
        self.purchase_frequency_days = purchase_frequency_days
        self.protection_ratio = protection_ratio
        self.put_dte = put_dte
        self.roll_dte = roll_dte
        self.otm_deviations = otm_deviations
        self.start_date = start_date
        self.end_date = end_date
        
        # Strategy state
        self.cash = initial_capital
        self.shares_owned = 0
        self.total_invested = 0
        self.options_positions = []
        self.trades = []
        self.portfolio_values = []
        self.dates = []
        
        # Download TQQQ data
        self.tqqq_data = self._download_tqqq_data()
        
    def _download_tqqq_data(self) -> pd.DataFrame:
        """Download TQQQ historical data"""
        print("📊 Downloading TQQQ data...")
        ticker = yf.Ticker("TQQQ")
        data = ticker.history(start=self.start_date, end=self.end_date)
        
        if data.empty:
            raise ValueError("No TQQQ data found for the specified date range")
        
        # Calculate daily returns and volatility
        data['Returns'] = data['Close'].pct_change()
        data['Volatility'] = data['Returns'].rolling(window=20).std() * np.sqrt(252)
        data['20d_Vol'] = data['Volatility'].fillna(method='bfill')
        
        print(f"✅ Downloaded {len(data)} days of TQQQ data")
        return data
    
    def _calculate_put_strike(self, current_price: float, volatility: float, dte: int) -> float:
        """Calculate put strike price using very conservative approach"""
        if volatility <= 0 or dte <= 0:
            return current_price * 0.9  # Fallback to 10% OTM
        
        # Very conservative strike calculation
        # Strike = S * (1 - volatility * sqrt(dte/365) * deviations * 0.2)
        strike_factor = 1 - (volatility * np.sqrt(dte/365) * self.otm_deviations * 0.2)
        strike = current_price * max(strike_factor, 0.8)  # At least 20% OTM
        
        # Ensure strike is reasonable
        min_strike = current_price * 0.6  # At least 40% of current price
        max_strike = current_price * 0.95  # At most 5% OTM
        
        return max(min_strike, min(strike, max_strike))
    
    def _estimate_put_premium(self, current_price: float, strike: float, volatility: float, dte: int) -> float:
        """Very conservative put option premium estimation"""
        if dte <= 0 or volatility <= 0:
            return 0.01
        
        S = current_price
        K = strike
        r = 0.02  # 2% risk-free rate
        sigma = volatility
        T = dte / 365.0
        
        # Very conservative Black-Scholes approximation
        d1 = (np.log(S/K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        # Use normal CDF approximation
        def norm_cdf(x):
            return 0.5 * (1 + np.tanh(0.8 * x))
        
        # Put option value
        put_price = K * np.exp(-r * T) * norm_cdf(-d2) - S * norm_cdf(-d1)
        
        # Very conservative factors
        put_price = max(put_price, 0.01)  # Minimum $0.01
        put_price *= 0.5  # 50% discount for conservative estimate
        put_price *= (1 + volatility * 0.05)  # Small volatility premium
        
        # Cap the premium at 2% of stock price
        put_price = min(put_price, S * 0.02)
        
        return put_price
    
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
    
    def _buy_put_protection(self, date: datetime, current_price: float, volatility: float) -> Optional[Dict]:
        """Buy put options for protection"""
        if self.shares_owned == 0:
            return None
        
        # Calculate strike price
        strike = self._calculate_put_strike(current_price, volatility, self.put_dte)
        
        # Estimate premium
        premium = self._estimate_put_premium(current_price, strike, volatility, self.put_dte)
        
        # Calculate number of contracts needed
        shares_to_protect = int(self.shares_owned * self.protection_ratio)
        contracts_needed = max(1, int(np.ceil(shares_to_protect / 100)))
        
        # Calculate total cost
        total_cost = contracts_needed * premium * 100
        
        if total_cost > self.cash:
            print(f"⚠️  Insufficient cash to buy put protection on {date.strftime('%Y-%m-%d')}")
            return None
        
        self.cash -= total_cost
        
        # Create option position
        option_position = {
            'date_opened': date,
            'strike': strike,
            'premium': premium,
            'contracts': contracts_needed,
            'total_cost': total_cost,
            'dte': self.put_dte,
            'shares_protected': contracts_needed * 100
        }
        
        self.options_positions.append(option_position)
        
        trade = {
            'date': date,
            'action': 'BUY_PUTS',
            'strike': strike,
            'premium': premium,
            'contracts': contracts_needed,
            'cost': total_cost,
            'shares_protected': contracts_needed * 100,
            'dte': self.put_dte
        }
        
        self.trades.append(trade)
        print(f"🛡️  Bought {contracts_needed} put contracts at ${strike:.2f} strike, ${premium:.2f} premium on {date.strftime('%Y-%m-%d')}")
        
        return trade
    
    def _roll_options(self, date: datetime, current_price: float, volatility: float) -> List[Dict]:
        """Roll options that are close to expiration"""
        rolled_trades = []
        positions_to_remove = []
        
        for i, position in enumerate(self.options_positions):
            days_held = (date - position['date_opened']).days
            current_dte = position['dte'] - days_held
            
            if current_dte <= self.roll_dte:
                # Close current position (simplified - assume we get back some value)
                remaining_value = position['total_cost'] * (current_dte / position['dte']) * 0.1
                self.cash += remaining_value
                
                # Buy new protection
                new_strike = self._calculate_put_strike(current_price, volatility, self.put_dte)
                new_premium = self._estimate_put_premium(current_price, new_strike, volatility, self.put_dte)
                new_contracts = position['contracts']
                new_cost = new_contracts * new_premium * 100
                
                if new_cost <= self.cash:
                    self.cash -= new_cost
                    
                    # Create new position
                    new_position = {
                        'date_opened': date,
                        'strike': new_strike,
                        'premium': new_premium,
                        'contracts': new_contracts,
                        'total_cost': new_cost,
                        'dte': self.put_dte,
                        'shares_protected': new_contracts * 100
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
                        'contracts': new_contracts,
                        'net_cost': new_cost - remaining_value,
                        'dte': self.put_dte
                    }
                    
                    self.trades.append(trade)
                    rolled_trades.append(trade)
                    print(f"🔄 Rolled {new_contracts} put contracts from ${position['strike']:.2f} to ${new_strike:.2f} on {date.strftime('%Y-%m-%d')}")
        
        # Remove rolled positions
        for i in reversed(positions_to_remove):
            self.options_positions.pop(i)
        
        return rolled_trades
    
    def _calculate_portfolio_value(self, date: datetime, current_price: float) -> float:
        """Calculate total portfolio value"""
        shares_value = self.shares_owned * current_price
        
        # Calculate options value (simplified)
        options_value = 0
        for position in self.options_positions:
            days_held = (date - position['date_opened']).days
            current_dte = position['dte'] - days_held
            
            if current_dte > 0:
                # Simplified option value calculation
                intrinsic_value = max(0, position['strike'] - current_price)
                time_value = position['premium'] * (current_dte / position['dte']) * 0.1
                option_value = (intrinsic_value + time_value) * position['contracts'] * 100
                options_value += option_value
        
        return self.cash + shares_value + options_value
    
    def run_backtest(self) -> Dict:
        """Run the complete backtest"""
        print("🚀 Starting Conservative TQQQ Options Protection Strategy Backtest")
        print("=" * 60)
        
        # Get trading dates
        trading_dates = self.tqqq_data.index.tolist()
        last_purchase_date = None
        
        for i, date in enumerate(trading_dates):
            current_price = self.tqqq_data.loc[date, 'Close']
            volatility = self.tqqq_data.loc[date, '20d_Vol']
            
            # Check if it's time to buy shares
            if (last_purchase_date is None or 
                (date - last_purchase_date).days >= self.purchase_frequency_days):
                
                # Buy shares
                self._buy_shares(date, current_price)
                
                # Buy put protection
                self._buy_put_protection(date, current_price, volatility)
                
                last_purchase_date = date
            
            # Check if we need to roll options
            self._roll_options(date, current_price, volatility)
            
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
        print("📊 CONSERVATIVE TQQQ OPTIONS PROTECTION STRATEGY RESULTS")
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

def main():
    """Run the conservative TQQQ options strategy backtest"""
    print("🏦 Conservative TQQQ Options Protection Strategy Backtest")
    print("=" * 60)
    
    # Strategy parameters - very conservative
    strategy = ConservativeTQQQStrategy(
        initial_capital=100000,
        share_purchase_amount=25,  # Very small position
        purchase_frequency_days=14,  # Every 2 weeks
        protection_ratio=0.25,  # Protect only 25% of position
        put_dte=90,  # 90 days to expiration
        roll_dte=30,  # Roll when 30 days left
        otm_deviations=1.0,  # 1 standard deviation OTM (closer to money)
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
    results_df.to_csv('tqqq_conservative_strategy_results.csv', index=False)
    print(f"\n💾 Detailed results saved to 'tqqq_conservative_strategy_results.csv'")

if __name__ == "__main__":
    main()
