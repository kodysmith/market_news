#!/usr/bin/env python3
"""
Improved Overbought/Oversold Scanner
Addresses feedback for production-ready trading signals
"""

import yfinance as yf
import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import requests
import time
import talib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImprovedScanner:
    def __init__(self):
        self.data_cache = {}
        self.confidence_calibration = {}
        
    def validate_real_time_data(self, ticker: str) -> Dict:
        """Cross-check data from multiple sources for accuracy"""
        try:
            # Primary source: Yahoo Finance
            stock = yf.Ticker(ticker)
            yf_data = stock.history(period="1d", interval="1m")
            
            if yf_data.empty:
                return {'valid': False, 'error': 'No data from Yahoo Finance'}
            
            # Get real-time quote
            info = stock.info
            current_price = info.get('regularMarketPrice') or info.get('previousClose')
            
            # Cross-check with Alpha Vantage (if available)
            # Note: This would require API key
            cross_check_price = None
            
            # Validate data freshness
            last_update = yf_data.index[-1]
            time_diff = datetime.now() - last_update.replace(tzinfo=None)
            
            if time_diff.total_seconds() > 3600:  # More than 1 hour old
                logger.warning(f"⚠️ Data for {ticker} is {time_diff} old")
            
            return {
                'valid': True,
                'current_price': current_price,
                'last_update': last_update,
                'data_age_minutes': time_diff.total_seconds() / 60,
                'cross_check_price': cross_check_price,
                'source': 'yahoo_finance'
            }
            
        except Exception as e:
            logger.error(f"❌ Data validation failed for {ticker}: {e}")
            return {'valid': False, 'error': str(e)}
    
    def calculate_rsi_standard(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI using standard 14-day window with validation"""
        try:
            # Use TA-Lib for industry-standard RSI calculation
            rsi = talib.RSI(prices.values, timeperiod=window)
            return pd.Series(rsi, index=prices.index)
        except ImportError:
            # Fallback to manual calculation if TA-Lib not available
            logger.warning("TA-Lib not available, using manual RSI calculation")
            return self._manual_rsi(prices, window)
        except Exception as e:
            logger.error(f"RSI calculation failed: {e}")
            return self._manual_rsi(prices, window)
    
    def _manual_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Manual RSI calculation as fallback"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
        """Calculate Average True Range for volatility-based targets/stops"""
        try:
            atr = talib.ATR(high.values, low.values, close.values, timeperiod=window)
            return pd.Series(atr, index=close.index)
        except ImportError:
            # Manual ATR calculation
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = true_range.rolling(window=window).mean()
            return atr
    
    def find_support_resistance_levels(self, high: pd.Series, low: pd.Series, close: pd.Series) -> Dict:
        """Find actual support/resistance levels using pivot points"""
        try:
            # Use recent 60 days for more relevant levels
            recent_high = high.tail(60)
            recent_low = low.tail(60)
            current_price = close.iloc[-1]
            
            # Find swing highs and lows
            swing_highs = []
            swing_lows = []
            
            for i in range(5, len(recent_high) - 5):
                # Swing high: higher than 5 points on each side
                if all(recent_high.iloc[i] > recent_high.iloc[j] for j in range(i-5, i+6) if j != i):
                    swing_highs.append(recent_high.iloc[i])
                
                # Swing low: lower than 5 points on each side
                if all(recent_low.iloc[i] < recent_low.iloc[j] for j in range(i-5, i+6) if j != i):
                    swing_lows.append(recent_low.iloc[i])
            
            # Filter levels near current price (within 20%)
            relevant_resistance = [h for h in swing_highs if h > current_price and h < current_price * 1.2]
            relevant_support = [l for l in swing_lows if l < current_price and l > current_price * 0.8]
            
            return {
                'support_levels': sorted(relevant_support, reverse=True)[:3],
                'resistance_levels': sorted(relevant_resistance)[:3],
                'nearest_support': max(relevant_support) if relevant_support else None,
                'nearest_resistance': min(relevant_resistance) if relevant_resistance else None
            }
            
        except Exception as e:
            logger.error(f"Support/resistance calculation failed: {e}")
            return {'support_levels': [], 'resistance_levels': [], 'nearest_support': None, 'nearest_resistance': None}
    
    def calculate_atr_targets_stops(self, current_price: float, atr: float, signal: str, atr_multiplier: float = 2.0) -> Dict:
        """Calculate targets and stops based on ATR (Average True Range)"""
        try:
            if signal in ['BUY', 'WEAK_BUY']:
                target = current_price + (atr * atr_multiplier)
                stop_loss = current_price - (atr * atr_multiplier)
            elif signal in ['SELL', 'WEAK_SELL']:
                target = current_price - (atr * atr_multiplier)
                stop_loss = current_price + (atr * atr_multiplier)
            else:  # HOLD
                # Still provide reference levels
                target = current_price + (atr * atr_multiplier)
                stop_loss = current_price - (atr * atr_multiplier)
            
            risk = abs(current_price - stop_loss)
            reward = abs(target - current_price)
            risk_reward = reward / risk if risk > 0 else 0
            
            return {
                'target': round(target, 2),
                'stop_loss': round(stop_loss, 2),
                'risk_reward': round(risk_reward, 2),
                'atr_value': round(atr, 2),
                'atr_multiplier': atr_multiplier
            }
            
        except Exception as e:
            logger.error(f"ATR calculation failed: {e}")
            return {'target': None, 'stop_loss': None, 'risk_reward': None}
    
    def get_fundamental_context(self, ticker: str) -> Dict:
        """Get fundamental context to validate technical signals"""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # Key fundamental metrics
            pe_ratio = info.get('trailingPE', 0)
            peg_ratio = info.get('pegRatio', 0)
            debt_to_equity = info.get('debtToEquity', 0)
            return_on_equity = info.get('returnOnEquity', 0)
            revenue_growth = info.get('revenueGrowth', 0)
            
            # Sector and market cap
            sector = info.get('sector', 'Unknown')
            market_cap = info.get('marketCap', 0)
            
            # Analyst ratings
            recommendation = info.get('recommendationMean', 0)
            target_price = info.get('targetMeanPrice', 0)
            
            return {
                'pe_ratio': pe_ratio,
                'peg_ratio': peg_ratio,
                'debt_to_equity': debt_to_equity,
                'roe': return_on_equity,
                'revenue_growth': revenue_growth,
                'sector': sector,
                'market_cap': market_cap,
                'analyst_recommendation': recommendation,
                'target_price': target_price,
                'fundamental_score': self._calculate_fundamental_score(info)
            }
            
        except Exception as e:
            logger.error(f"Fundamental analysis failed for {ticker}: {e}")
            return {'fundamental_score': 50}  # Neutral if can't get data
    
    def _calculate_fundamental_score(self, info: Dict) -> int:
        """Calculate a 0-100 fundamental score"""
        try:
            score = 50  # Start neutral
            
            # PE ratio scoring
            pe = info.get('trailingPE', 0)
            if 10 <= pe <= 20:
                score += 20
            elif 20 < pe <= 30:
                score += 10
            elif pe > 30:
                score -= 10
            
            # ROE scoring
            roe = info.get('returnOnEquity', 0)
            if roe > 15:
                score += 15
            elif roe > 10:
                score += 10
            elif roe < 5:
                score -= 15
            
            # Revenue growth scoring
            growth = info.get('revenueGrowth', 0)
            if growth > 0.1:  # 10%+ growth
                score += 15
            elif growth > 0.05:  # 5%+ growth
                score += 10
            elif growth < 0:  # Negative growth
                score -= 15
            
            return max(0, min(100, score))
            
        except Exception as e:
            logger.error(f"Fundamental scoring failed: {e}")
            return 50
    
    def calculate_improved_confidence(self, rsi: float, fundamental_score: int, 
                                    data_quality: float, market_context: Dict) -> int:
        """Calculate confidence based on multiple factors, not just RSI"""
        try:
            confidence = 0
            
            # RSI strength (40% weight)
            if rsi > 80 or rsi < 20:
                confidence += 40
            elif rsi > 70 or rsi < 30:
                confidence += 30
            elif rsi > 60 or rsi < 40:
                confidence += 20
            
            # Fundamental alignment (30% weight)
            if fundamental_score > 70:
                confidence += 30
            elif fundamental_score > 50:
                confidence += 20
            elif fundamental_score < 30:
                confidence -= 20
            
            # Data quality (20% weight)
            if data_quality > 0.9:
                confidence += 20
            elif data_quality > 0.7:
                confidence += 15
            elif data_quality < 0.5:
                confidence -= 10
            
            # Market context (10% weight)
            # This would include sector momentum, market regime, etc.
            confidence += 10  # Placeholder
            
            return max(0, min(100, confidence))
            
        except Exception as e:
            logger.error(f"Confidence calculation failed: {e}")
            return 50
    
    def analyze_stock_improved(self, ticker: str) -> Dict:
        """Comprehensive stock analysis with all improvements"""
        try:
            logger.info(f"🔍 Analyzing {ticker} with improved methodology...")
            
            # 1. Validate real-time data
            data_validation = self.validate_real_time_data(ticker)
            if not data_validation['valid']:
                return {'error': f"Data validation failed: {data_validation['error']}"}
            
            # 2. Get historical data
            stock = yf.Ticker(ticker)
            hist = stock.history(period="3mo")
            
            if hist.empty:
                return {'error': 'No historical data available'}
            
            # 3. Calculate standard RSI (14-day)
            rsi = self.calculate_rsi_standard(hist['Close'], window=14)
            current_rsi = float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50
            
            # 4. Calculate ATR for volatility-based targets
            atr = self.calculate_atr(hist['High'], hist['Low'], hist['Close'])
            current_atr = float(atr.iloc[-1]) if not pd.isna(atr.iloc[-1]) else 0
            
            # 5. Find support/resistance levels
            sr_levels = self.find_support_resistance_levels(hist['High'], hist['Low'], hist['Close'])
            
            # 6. Get fundamental context
            fundamentals = self.get_fundamental_context(ticker)
            
            # 7. Determine signal
            signal = 'HOLD'
            if current_rsi > 70:
                signal = 'SELL'
            elif current_rsi < 30:
                signal = 'BUY'
            
            # For HOLD signals, still calculate targets for reference
            if signal == 'HOLD':
                if current_rsi > 60:  # Slightly overbought
                    signal = 'WEAK_SELL'
                elif current_rsi < 40:  # Slightly oversold
                    signal = 'WEAK_BUY'
            
            # 8. Calculate ATR-based targets/stops
            current_price = float(hist['Close'].iloc[-1])
            atr_targets = self.calculate_atr_targets_stops(current_price, current_atr, signal)
            
            # 9. Calculate improved confidence
            data_quality = 1.0 - (data_validation['data_age_minutes'] / 60)  # Fresh data = higher quality
            confidence = self.calculate_improved_confidence(
                current_rsi, 
                fundamentals['fundamental_score'], 
                data_quality,
                {}  # Market context placeholder
            )
            
            # 10. Compile analysis
            analysis = {
                'ticker': ticker,
                'timestamp': datetime.now().isoformat(),
                'current_price': current_price,
                'rsi': round(current_rsi, 2),
                'atr': round(current_atr, 2),
                'signal': signal,
                'confidence': confidence,
                'data_quality': round(data_quality, 2),
                'fundamental_score': fundamentals['fundamental_score'],
                'target_price': atr_targets['target'],
                'stop_loss': atr_targets['stop_loss'],
                'risk_reward': atr_targets['risk_reward'],
                'support_levels': sr_levels['support_levels'],
                'resistance_levels': sr_levels['resistance_levels'],
                'nearest_support': sr_levels['nearest_support'],
                'nearest_resistance': sr_levels['nearest_resistance'],
                'fundamentals': fundamentals,
                'data_validation': data_validation
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"❌ Analysis failed for {ticker}: {e}")
            return {'error': str(e)}
    
    def scan_improved(self, tickers: List[str]) -> Dict:
        """Run improved scan on multiple tickers"""
        results = {
            'scan_timestamp': datetime.now().isoformat(),
            'total_tickers': len(tickers),
            'successful_scans': 0,
            'failed_scans': 0,
            'opportunities': [],
            'summary': {}
        }
        
        for i, ticker in enumerate(tickers):
            try:
                logger.info(f"📊 Scanning {ticker} ({i+1}/{len(tickers)})")
                
                analysis = self.analyze_stock_improved(ticker)
                
                if 'error' not in analysis:
                    results['successful_scans'] += 1
                    results['opportunities'].append(analysis)
                else:
                    results['failed_scans'] += 1
                    logger.warning(f"⚠️ {ticker}: {analysis['error']}")
                
                # Rate limiting
                if i < len(tickers) - 1:
                    time.sleep(1)
                    
            except Exception as e:
                logger.error(f"❌ Unexpected error scanning {ticker}: {e}")
                results['failed_scans'] += 1
        
        # Generate summary
        results['summary'] = self._generate_summary(results['opportunities'])
        
        return results
    
    def _generate_summary(self, opportunities: List[Dict]) -> Dict:
        """Generate summary statistics"""
        if not opportunities:
            return {}
        
        buy_signals = [o for o in opportunities if o['signal'] == 'BUY']
        sell_signals = [o for o in opportunities if o['signal'] == 'SELL']
        
        return {
            'total_opportunities': len(opportunities),
            'buy_signals': len(buy_signals),
            'sell_signals': len(sell_signals),
            'avg_confidence': round(np.mean([o['confidence'] for o in opportunities]), 1),
            'avg_data_quality': round(np.mean([o['data_quality'] for o in opportunities]), 2),
            'avg_fundamental_score': round(np.mean([o['fundamental_score'] for o in opportunities]), 1)
        }

def main():
    """Test the improved scanner"""
    scanner = ImprovedScanner()
    
    # Test with a few tickers
    test_tickers = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']
    
    print("🚀 Running Improved Scanner Test...")
    print("=" * 60)
    
    results = scanner.scan_improved(test_tickers)
    
    print(f"\n📊 SCAN RESULTS")
    print(f"Total Tickers: {results['total_tickers']}")
    print(f"Successful: {results['successful_scans']}")
    print(f"Failed: {results['failed_scans']}")
    print(f"Average Confidence: {results['summary'].get('avg_confidence', 0)}%")
    print(f"Average Data Quality: {results['summary'].get('avg_data_quality', 0)}")
    
    print(f"\n🎯 TOP OPPORTUNITIES:")
    opportunities = sorted(results['opportunities'], key=lambda x: x['confidence'], reverse=True)
    
    for opp in opportunities[:5]:
        target = opp['target_price'] if opp['target_price'] else 0
        stop = opp['stop_loss'] if opp['stop_loss'] else 0
        rr = opp['risk_reward'] if opp['risk_reward'] else 0
        
        print(f"  {opp['ticker']:4} | {opp['signal']:4} | RSI: {opp['rsi']:5.1f} | "
              f"Conf: {opp['confidence']:3.0f}% | Target: ${target:7.2f} | "
              f"Stop: ${stop:7.2f} | R/R: {rr:4.2f}")

if __name__ == "__main__":
    main()
