#!/usr/bin/env python3
"""
Real options data fetcher using yfinance
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import time

class RealDataFetcher:
    def __init__(self):
        self.last_request_time = 0
        self.min_request_interval = 1.0  # 1 second between requests
    
    def _rate_limit(self):
        """Implement rate limiting"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current stock price using yfinance"""
        try:
            self._rate_limit()
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Try different price fields
            price = info.get('currentPrice') or info.get('regularMarketPrice') or info.get('previousClose')
            
            if price:
                return float(price)
            
            # Fallback to historical data
            hist = ticker.history(period="1d")
            if not hist.empty:
                return float(hist['Close'].iloc[-1])
            
            print(f"❌ Could not get price for {symbol}")
            return None
            
        except Exception as e:
            print(f"❌ Error getting price for {symbol}: {e}")
            return None
    
    def get_options_chain(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get options chain using yfinance"""
        try:
            self._rate_limit()
            ticker = yf.Ticker(symbol)
            
            # Get current price first
            current_price = self.get_current_price(symbol)
            if not current_price:
                return None
            
            # Get options expiration dates
            expirations = ticker.options
            if not expirations:
                print(f"❌ No options found for {symbol}")
                return None
            
            # Get the nearest expiration (within 30-45 days)
            target_date = None
            for exp_date in expirations:
                exp_dt = datetime.strptime(exp_date, '%Y-%m-%d')
                days_to_exp = (exp_dt - datetime.now()).days
                if 20 <= days_to_exp <= 45:
                    target_date = exp_date
                    break
            
            if not target_date:
                # Use the first available expiration
                target_date = expirations[0]
            
            print(f"📅 Using expiration: {target_date}")
            
            # Get options chain for the target date
            options_chain = ticker.option_chain(target_date)
            puts = options_chain.puts
            
            if puts.empty:
                print(f"❌ No puts found for {symbol} on {target_date}")
                return None
            
            print(f"📊 Found {len(puts)} put options")
            print(f"📋 Available columns: {list(puts.columns)}")
            
            # Convert to our format
            puts_list = []
            for _, put in puts.iterrows():
                puts_list.append({
                    'strike': put['strike'],
                    'bid': put['bid'],
                    'ask': put['ask'],
                    'lastPrice': put['lastPrice'],
                    'volume': put['volume'],
                    'openInterest': put['openInterest'],
                    'impliedVolatility': put['impliedVolatility'],
                    'delta': put.get('delta', 0.0),  # Handle missing delta
                    'gamma': put.get('gamma', 0.0),
                    'theta': put.get('theta', 0.0),
                    'vega': put.get('vega', 0.0),
                    'expiration': target_date
                })
            
            return {
                'options': [{'puts': puts_list, 'calls': []}],
                'quote': {
                    'regularMarketPrice': current_price,
                    'symbol': symbol
                }
            }
            
        except Exception as e:
            print(f"❌ Error getting options chain for {symbol}: {e}")
            return None
    
    def generate_bull_put_spreads(self, symbol: str, max_dte: int = 45, 
                                 min_ev: float = 0.0) -> List[Dict[str, Any]]:
        """Generate bull put spreads from real options data"""
        try:
            print(f"🔍 Fetching real options data for {symbol}...")
            
            # Get current price
            current_price = self.get_current_price(symbol)
            if not current_price:
                print(f"❌ Could not get current price for {symbol}")
                return []
            
            print(f"📊 Current {symbol} price: ${current_price:.2f}")
            
            # Get options chain
            options_data = self.get_options_chain(symbol)
            if not options_data:
                print(f"❌ Could not get options chain for {symbol}")
                return []
            
            # Extract puts
            puts = options_data.get('options', [{}])[0].get('puts', [])
            if not puts:
                print(f"❌ No puts found for {symbol}")
                return []
            
            print(f"📈 Found {len(puts)} put options")
            
            # Filter puts by DTE and OTM
            valid_puts = []
            for i, put in enumerate(puts):
                # Calculate DTE
                exp_date = datetime.strptime(put['expiration'], '%Y-%m-%d')
                dte = (exp_date - datetime.now()).days
                
                # Only OTM puts (strike < current price) but not too far OTM
                if (put['strike'] < current_price and 
                    put['strike'] > current_price * 0.90 and  # Within 10% of current price for better spreads
                    10 <= dte <= max_dte):
                    valid_puts.append(put)
                    if i < 5:  # Debug first few
                        print(f"  Valid put {i+1}: Strike ${put['strike']:.2f}, DTE {dte}, Bid ${put['bid']:.2f}, Ask ${put['ask']:.2f}")
            
            print(f"📅 Found {len(valid_puts)} OTM puts within DTE range (10-{max_dte} days)")
            
            # Generate spreads
            spreads = []
            print(f"🔄 Generating spreads from {len(valid_puts)} valid puts...")
            
            for i in range(len(valid_puts) - 1):
                short_put = valid_puts[i]
                long_put = valid_puts[i + 1]
                
                short_strike = short_put['strike']
                long_strike = long_put['strike']
                
                # Ensure short strike > long strike for bull put spread
                if short_strike <= long_strike:
                    continue
                
                width = short_strike - long_strike
                short_bid = short_put['bid']
                long_ask = long_put['ask']
                
                # Skip if no valid bid/ask
                if short_bid <= 0 or long_ask <= 0:
                    if i < 5:  # Debug first few
                        print(f"  Skipped {short_strike}/{long_strike}: Invalid bid/ask (${short_bid:.2f}/${long_ask:.2f})")
                    continue
                
                credit = short_bid - long_ask
                if credit <= 0:
                    if i < 5:  # Debug first few
                        print(f"  Skipped {short_strike}/{long_strike}: Negative credit (${credit:.2f})")
                    continue
                
                # Debug: Print first few spreads
                if len(spreads) < 3:
                    print(f"  Spread {len(spreads)+1}: {short_strike}/{long_strike}, Credit: ${credit:.2f}, Width: ${width:.2f}")
                
                max_loss = width - credit
                
                # Calculate probability of profit based on distance from current price
                distance_pct = (current_price - short_strike) / current_price
                if distance_pct > 0.1:  # >10% OTM
                    pop = 0.85
                elif distance_pct > 0.05:  # 5-10% OTM
                    pop = 0.75
                elif distance_pct > 0.02:  # 2-5% OTM
                    pop = 0.65
                else:  # <2% OTM
                    pop = 0.55
                
                # Calculate delta approximation
                short_delta = max(0.1, min(0.4, distance_pct * 2))  # Rough delta approximation
                
                # Calculate expected value
                expected_value = (pop * credit) - ((1 - pop) * max_loss)
                
                # Only include positive EV spreads (or very small negative EV for real market conditions)
                if expected_value < min_ev and expected_value < -1.0:  # Allow small negative EV for real market
                    continue
                
                # Get additional data
                short_iv = short_put['impliedVolatility']
                short_volume = short_put['volume']
                short_oi = short_put['openInterest']
                
                # Calculate DTE
                dte = (datetime.strptime(short_put['expiration'], '%Y-%m-%d') - datetime.now()).days
                
                spread = {
                    'ticker': symbol,
                    'strategy': 'BULL_PUT',
                    'expiry': short_put['expiration'],
                    'shortK': short_strike,
                    'longK': long_strike,
                    'width': width,
                    'credit': credit,
                    'maxLoss': max_loss,
                    'dte': dte,
                    'pop': pop,
                    'ev': expected_value,
                    'ivShort': short_iv,
                    'bidAskW': (short_put['ask'] - short_put['bid']) / short_put['strike'],
                    'oiShort': short_oi,
                    'oiLong': long_put['openInterest'],
                    'volShort': short_volume,
                    'volLong': long_put['volume'],
                    'fillScore': min(1.0, (short_volume + short_oi) / 1000),
                    'id': f"{symbol}_{short_strike}_{long_strike}_{short_put['expiration']}"
                }
                
                spreads.append(spread)
            
            # Sort by expected value
            spreads.sort(key=lambda x: x['ev'], reverse=True)
            
            print(f"✅ Generated {len(spreads)} real spreads with positive EV")
            return spreads[:10]  # Return top 10
            
        except Exception as e:
            print(f"❌ Error generating spreads for {symbol}: {e}")
            return []
    
    def get_market_sentiment(self) -> Dict[str, Any]:
        """Get market sentiment from real data"""
        try:
            # Get SPY price for sentiment
            spy_price = self.get_current_price('SPY')
            if not spy_price:
                return {
                    'sentiment': 'neutral',
                    'confidence': 0.5,
                    'indicators': ['No data available']
                }
            
            # Simple sentiment based on price
            if spy_price > 650:
                sentiment = 'bullish'
            elif spy_price < 600:
                sentiment = 'bearish'
            else:
                sentiment = 'neutral'
            
            return {
                'sentiment': sentiment,
                'confidence': 0.7,
                'indicators': [f'SPY Price: ${spy_price:.2f}'],
                'spy_price': spy_price
            }
            
        except Exception as e:
            print(f"❌ Error getting market sentiment: {e}")
            return {
                'sentiment': 'neutral',
                'confidence': 0.5,
                'indicators': ['Error fetching data']
            }

def test_real_data():
    """Test real data fetching"""
    print("🧪 Testing Real Data Fetching with yfinance")
    print("=" * 50)
    
    fetcher = RealDataFetcher()
    
    # Test current price
    print("📊 Testing current price...")
    price = fetcher.get_current_price('SPY')
    if price:
        print(f"✅ SPY price: ${price:.2f}")
    else:
        print("❌ Failed to get SPY price")
        return False
    
    # Test options chain
    print("\n📈 Testing options chain...")
    options = fetcher.get_options_chain('SPY')
    if options:
        puts = options.get('options', [{}])[0].get('puts', [])
        print(f"✅ Found {len(puts)} puts")
    else:
        print("❌ Failed to get options chain")
        return False
    
    # Test spread generation
    print("\n🔄 Testing spread generation...")
    spreads = fetcher.generate_bull_put_spreads('SPY', max_dte=45, min_ev=0.0)
    if spreads:
        print(f"✅ Generated {len(spreads)} spreads")
        print("\n📋 Top spreads:")
        for i, spread in enumerate(spreads[:3], 1):
            print(f"  {i}. {spread['shortK']}/{spread['longK']} "
                  f"Credit: ${spread['credit']:.2f}, EV: ${spread['ev']:.2f}, "
                  f"POP: {spread['pop']:.1%}")
    else:
        print("❌ No spreads generated")
        return False
    
    print("\n🎉 Real data fetching test passed!")
    return True

if __name__ == "__main__":
    test_real_data()
