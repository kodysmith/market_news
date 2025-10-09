#!/usr/bin/env python3
"""
Production System Backtester
Run the actual trading engine on historical data to validate performance

This uses the REAL production code with historical data from Alpaca
"""

import sys
import os
from datetime import datetime, timedelta, date
from decimal import Decimal
import pandas as pd
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.main.trading_engine import TradingEngine
from src.common.config import load_config
from src.data.market_data_service import MarketDataService

# Try to import Alpaca for historical data
try:
    from alpaca.data import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False
    print("⚠️  alpaca-py not installed, using yfinance for historical data")

import yfinance as yf


class HistoricalDataProvider:
    """Provides historical market data for backtesting"""
    
    def __init__(self, config):
        self.config = config
        self.alpaca_available = False
        
        # Use yfinance as primary (more reliable for historical data)
        print("✅ Using yfinance for historical data (most reliable)")
        
        if ALPACA_AVAILABLE and config.broker.alpaca_api_key:
            try:
                self.alpaca = StockHistoricalDataClient(
                    config.broker.alpaca_api_key,
                    config.broker.alpaca_secret_key
                )
                self.alpaca_available = True
                print("   (Alpaca available as backup)")
            except:
                pass
        
        # Cache for historical data
        self.price_cache: Dict[str, pd.DataFrame] = {}
    
    def load_data(self, symbols: List[str], start_date: date, end_date: date):
        """Load historical data for backtesting"""
        print(f"\n📊 Loading historical data...")
        print(f"   Symbols: {symbols}")
        print(f"   Period: {start_date} to {end_date}")
        
        for symbol in symbols:
            # Try yfinance first (more reliable for historical data)
            self.price_cache[symbol] = self._load_yfinance(symbol, start_date, end_date)
            
            # If yfinance fails and Alpaca available, try Alpaca
            if self.price_cache[symbol].empty and self.alpaca_available:
                print(f"   Trying Alpaca for {symbol}...")
                self.price_cache[symbol] = self._load_alpaca(symbol, start_date, end_date)
            
            if not self.price_cache[symbol].empty:
                print(f"   ✅ {symbol}: {len(self.price_cache[symbol])} days loaded")
                # Debug: Show date range
                first_date = self.price_cache[symbol].index[0]
                last_date = self.price_cache[symbol].index[-1]
                first_price = self.price_cache[symbol]['Close'].iloc[0]
                print(f"      Range: {first_date.date()} (${first_price:.2f}) to {last_date.date()}")
            else:
                print(f"   ❌ {symbol}: No data!")
        
        print()
    
    def _load_alpaca(self, symbol: str, start_date: date, end_date: date) -> pd.DataFrame:
        """Load from Alpaca"""
        try:
            request = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame.Day,
                start=start_date,
                end=end_date
            )
            
            bars = self.alpaca.get_stock_bars(request)
            
            if symbol in bars:
                df = bars[symbol].df
                df = df.rename(columns={
                    'open': 'Open',
                    'high': 'High', 
                    'low': 'Low',
                    'close': 'Close',
                    'volume': 'Volume'
                })
                return df
            
            return pd.DataFrame()
        except Exception as e:
            print(f"   Alpaca failed for {symbol}: {e}")
            return pd.DataFrame()
    
    def _load_yfinance(self, symbol: str, start_date: date, end_date: date) -> pd.DataFrame:
        """Load from yfinance"""
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date)
            return df
        except Exception as e:
            print(f"   yfinance failed for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_price(self, symbol: str, trade_date: date) -> Decimal:
        """Get historical price for a specific date"""
        if symbol not in self.price_cache:
            return Decimal('0.0')
        
        df = self.price_cache[symbol]
        
        if df.empty:
            return Decimal('0.0')
        
        try:
            import pandas as pd
            
            # Ensure we're working with DatetimeIndex
            if not isinstance(df.index, pd.DatetimeIndex):
                return Decimal('0.0')
            
            # Convert trade_date to string for iloc-based lookup (more reliable)
            trade_date_str = str(trade_date)
            
            # Find closest date using asof (most reliable method)
            try:
                price_series = df['Close'].asof(pd.Timestamp(trade_date))
                
                # Extract scalar value
                if pd.isna(price_series):
                    # No data before this date, use first available
                    price_val = float(df['Close'].iloc[0])
                else:
                    price_val = float(price_series)
                
                return Decimal(str(round(price_val, 2)))
            
            except:
                # Fallback: Just get closest by converting index to dates
                df_dates = [idx.date() for idx in df.index]
                
                if trade_date in df_dates:
                    idx = df_dates.index(trade_date)
                    price_val = float(df['Close'].iloc[idx])
                    return Decimal(str(round(price_val, 2)))
                else:
                    # Use first available
                    price_val = float(df['Close'].iloc[0])
                    return Decimal(str(round(price_val, 2)))
        
        except Exception as e:
            # More detailed error
            import traceback
            print(f"   ⚠️  Price lookup error for {symbol}: {type(e).__name__}")
            return Decimal('0.0')
    
    def get_iv(self, symbol: str) -> Decimal:
        """Get IV estimate (simplified for backtesting)"""
        return Decimal('0.25')  # Use constant 25% IV for backtesting


class MockMarketDataService:
    """Mock market data service for backtesting"""
    
    def __init__(self, historical_provider, current_date):
        self.historical = historical_provider
        self.current_date = current_date
    
    def get_price(self, symbol: str) -> Decimal:
        """Get price for current backtest date"""
        price = self.historical.get_price(symbol, self.current_date)
        
        # Debug first call
        if not hasattr(self, '_price_debug_done'):
            print(f"      🔍 DEBUG MockMarketData.get_price:")
            print(f"         Symbol={symbol}, Date={self.current_date}, Price returned=${price}")
            self._price_debug_done = True
        
        # Ensure price is never 0
        if price == 0 or price is None:
            print(f"   ⚠️  No price for {symbol} on {self.current_date}, using fallback")
            return Decimal('100.0')  # Fallback price
        return price
    
    def get_iv(self, symbol: str) -> Decimal:
        """Get IV estimate"""
        iv = self.historical.get_iv(symbol)
        # Ensure IV is never 0
        if iv == 0 or iv is None:
            return Decimal('0.25')
        return iv
    
    def health_check(self) -> bool:
        return True


def run_backtest(start_date: str, end_date: str, initial_capital: float = 100000.0):
    """
    Run production trading engine on historical data
    
    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        initial_capital: Starting capital
    """
    print("="*70)
    print("🚀 PRODUCTION SYSTEM BACKTEST")
    print("="*70)
    print(f"Period: {start_date} to {end_date}")
    print(f"Initial Capital: ${initial_capital:,.0f}")
    print()
    
    # Load config
    config = load_config()
    config.initial_capital = initial_capital
    
    # Convert dates
    start = datetime.strptime(start_date, '%Y-%m-%d').date()
    end = datetime.strptime(end_date, '%Y-%m-%d').date()
    
    # Load historical data
    historical = HistoricalDataProvider(config)
    historical.load_data(config.strategy.assets, start, end)
    
    # Initialize trading engine (with modified market data)
    print("Initializing trading engine...")
    engine = TradingEngine(config)
    
    # IMPORTANT: Set initial cash in position manager
    engine.position_manager.cash = Decimal(str(initial_capital))
    engine.current_nav = Decimal(str(initial_capital))
    engine.peak_nav = Decimal(str(initial_capital))
    
    print(f"✅ Initial cash set: ${initial_capital:,.0f}")
    
    # Track results
    results = []
    trade_count = 0
    
    # Run through each trading day
    current_date = start
    trading_days = 0
    
    print("\n" + "="*70)
    print("RUNNING BACKTEST...")
    print("="*70)
    
    while current_date <= end:
        # Skip weekends
        if current_date.weekday() >= 5:
            current_date += timedelta(days=1)
            continue
        
        trading_days += 1
        
        # Create mock market data for this date
        mock_market_data = MockMarketDataService(historical, current_date)
        
        # Replace the engine's market data service
        engine.market_data = mock_market_data
        
        # Run daily cycle
        timestamp = datetime.combine(current_date, datetime.min.time())
        
        try:
            # Debug: Check portfolio state before cycle
            portfolio_before = engine.position_manager.get_portfolio_state()
            if trading_days == 1:  # First day
                print(f"  DEBUG - First day portfolio:")
                print(f"    Cash: ${portfolio_before['cash']:,.2f}")
                print(f"    NAV: ${portfolio_before['nav']:,.2f}")
            
            cycle_result = engine.run_daily_cycle(timestamp)
            
            # Track results
            results.append({
                'date': current_date,
                'nav': float(cycle_result['nav']),
                'signals': cycle_result['signals_generated'],
                'trades': cycle_result['trades_executed'],
                'status': cycle_result['status']
            })
            
            trade_count += cycle_result['trades_executed']
            
            # Print progress every week
            if trading_days % 5 == 0:
                print(f"  {current_date}: NAV=${cycle_result['nav']:,.2f}, "
                      f"Trades={trade_count}, Status={cycle_result['status']}")
        
        except Exception as e:
            print(f"  ❌ Error on {current_date}: {e}")
        
        current_date += timedelta(days=1)
    
    # Calculate performance metrics
    print("\n" + "="*70)
    print("📊 BACKTEST RESULTS")
    print("="*70)
    
    if results:
        df = pd.DataFrame(results)
        
        initial_nav = df['nav'].iloc[0]
        final_nav = df['nav'].iloc[-1]
        total_return = (final_nav - initial_nav) / initial_nav
        
        # Calculate Sharpe ratio
        df['daily_return'] = df['nav'].pct_change()
        sharpe = df['daily_return'].mean() / df['daily_return'].std() * (252 ** 0.5) if df['daily_return'].std() > 0 else 0
        
        # Calculate max drawdown
        df['peak'] = df['nav'].cummax()
        df['drawdown'] = (df['nav'] - df['peak']) / df['peak']
        max_dd = df['drawdown'].min()
        
        # Print results
        print(f"\n📈 Performance Metrics:")
        print(f"   Initial NAV: ${initial_nav:,.2f}")
        print(f"   Final NAV: ${final_nav:,.2f}")
        print(f"   Total Return: {total_return:.2%}")
        print(f"   Trading Days: {trading_days}")
        print(f"   Total Trades: {trade_count}")
        print(f"   Sharpe Ratio: {sharpe:.2f}")
        print(f"   Max Drawdown: {max_dd:.2%}")
        
        # Annualized metrics
        years = (end - start).days / 365.0
        annual_return = (1 + total_return) ** (1/years) - 1 if years > 0 else 0
        print(f"\n📊 Annualized Metrics:")
        print(f"   Annual Return: {annual_return:.2%}")
        print(f"   Years: {years:.2f}")
        
        # Save results
        output_file = f"backtest_results_{start_date}_to_{end_date}.csv"
        df.to_csv(output_file, index=False)
        print(f"\n💾 Results saved to: {output_file}")
        
        print("\n" + "="*70)
        print("✅ BACKTEST COMPLETE!")
        print("="*70)
        
        return df
    else:
        print("❌ No results generated")
        return None


if __name__ == "__main__":
    """Run backtest"""
    
    # Get date range from user or use defaults
    import sys
    
    if len(sys.argv) >= 3:
        start_date = sys.argv[1]
        end_date = sys.argv[2]
    else:
        # Default: Last 3 months
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
        
        print(f"Using default dates: {start_date} to {end_date}")
        print("(To specify dates: python backtest_with_production.py YYYY-MM-DD YYYY-MM-DD)")
        print()
    
    # Run backtest
    results = run_backtest(start_date, end_date)
    
    if results is not None:
        print("\n💡 Next Steps:")
        print("   1. Review the results CSV")
        print("   2. Compare to original backtests")
        print("   3. If similar, system is validated!")
        print("   4. If different, investigate discrepancies")

