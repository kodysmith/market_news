#!/usr/bin/env python3
"""
Realistic mock data fetcher that simulates real options data
"""

import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import random

class MockRealisticFetcher:
    def __init__(self):
        # Current market prices (as of Sept 2025)
        self.current_prices = {
            'SPY': 663.9,
            'QQQ': 520.0,
            'IWM': 240.0,
            'NVDA': 950.0,
            'TSLA': 280.0,
            'AAPL': 240.0,
            'MSFT': 450.0,
            'GOOGL': 190.0,
            'AMZN': 210.0,
            'META': 580.0,
        }
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current stock price (mock)"""
        return self.current_prices.get(symbol, 660.0)
    
    def generate_bull_put_spreads(self, symbol: str, max_dte: int = 45, 
                                 min_ev: float = 0.0) -> List[Dict[str, Any]]:
        """Generate realistic bull put spreads"""
        try:
            print(f"🔍 Generating realistic spreads for {symbol}...")
            
            current_price = self.get_current_price(symbol)
            print(f"📊 Current {symbol} price: ${current_price:.2f}")
            
            spreads = []
            
            # Generate 3-5 realistic spreads
            num_spreads = random.randint(3, 5)
            
            for i in range(num_spreads):
                # Generate realistic strikes
                otm_percent = 0.08 + i * 0.02  # 8% to 16% OTM
                short_strike = current_price * (1 - otm_percent)
                long_strike = short_strike - (5 + i * 2)  # 5-11 point spreads
                
                # Ensure short strike > long strike
                if short_strike <= long_strike:
                    continue
                
                width = short_strike - long_strike
                
                # Generate realistic credit based on distance from current price
                distance_factor = (current_price - short_strike) / current_price
                base_credit = width * (0.25 + distance_factor * 0.2)  # 25-45% of width
                credit = base_credit + random.uniform(-0.1, 0.1)  # Add some randomness
                credit = max(0.1, credit)  # Minimum credit
                
                max_loss = width - credit
                
                # Calculate probability of profit based on distance
                pop = 0.6 + distance_factor * 0.3  # 60-90% based on distance
                pop = min(0.9, max(0.5, pop))  # Clamp between 50-90%
                
                # Calculate expected value
                expected_value = (pop * credit) - ((1 - pop) * max_loss)
                
                # Ensure positive EV by adjusting credit if needed
                if expected_value < min_ev:
                    # Increase credit to make it positive EV
                    min_credit_for_positive_ev = max_loss * (1 - pop) / pop
                    if min_credit_for_positive_ev < width:  # Only if it's possible
                        credit = min_credit_for_positive_ev + 0.5  # Add small buffer
                        max_loss = width - credit
                        expected_value = (pop * credit) - ((1 - pop) * max_loss)
                    else:
                        continue  # Skip if impossible to make positive EV
                
                # Generate additional realistic data
                short_delta = 0.15 + i * 0.05  # 0.15 to 0.35
                short_iv = 0.20 + i * 0.03  # 20% to 35%
                short_volume = max(1, int(50 + i * 20))
                short_oi = max(10, int(200 + i * 100))
                
                # Calculate DTE
                dte = 30 + i * 5  # 30-50 days
                expiration = datetime.now() + timedelta(days=dte)
                
                spread = {
                    'ticker': symbol,
                    'strategy': 'BULL_PUT',
                    'expiry': expiration.strftime('%Y-%m-%d'),
                    'shortK': round(short_strike, 2),
                    'longK': round(long_strike, 2),
                    'width': round(width, 2),
                    'credit': round(credit, 2),
                    'maxLoss': round(max_loss, 2),
                    'dte': dte,
                    'pop': round(pop, 3),
                    'ev': round(expected_value, 2),
                    'ivShort': round(short_iv, 3),
                    'bidAskW': round(0.05 + i * 0.01, 2),  # 0.05 to 0.09
                    'oiShort': short_oi,
                    'oiLong': max(10, int(short_oi * 0.8)),
                    'volShort': short_volume,
                    'volLong': max(1, int(short_volume * 0.7)),
                    'fillScore': round(0.7 + i * 0.05, 2),  # 0.7 to 0.9
                    'id': f"{symbol}_{short_strike}_{long_strike}_{int(expiration.timestamp())}"
                }
                
                spreads.append(spread)
            
            # Sort by expected value
            spreads.sort(key=lambda x: x['ev'], reverse=True)
            
            print(f"✅ Generated {len(spreads)} realistic spreads with positive EV")
            return spreads
            
        except Exception as e:
            print(f"❌ Error generating spreads for {symbol}: {e}")
            return []
    
    def get_market_sentiment(self) -> Dict[str, Any]:
        """Get market sentiment (mock)"""
        spy_price = self.get_current_price('SPY')
        
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

def test_mock_realistic():
    """Test realistic mock data generation"""
    print("🧪 Testing Realistic Mock Data Generation")
    print("=" * 50)
    
    fetcher = MockRealisticFetcher()
    
    # Test current price
    print("📊 Testing current price...")
    price = fetcher.get_current_price('SPY')
    print(f"✅ SPY price: ${price:.2f}")
    
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
    
    print("\n🎉 Realistic mock data generation test passed!")
    return True

if __name__ == "__main__":
    test_mock_realistic()
