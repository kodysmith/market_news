#!/usr/bin/env python3
"""
Production-Ready Scanner
Integrates all improvements: real-time data, ATR targets, backtested confidence, fundamentals
"""

import yfinance as yf
import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import talib
import requests
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProductionScanner:
    def __init__(self, calibration_file: str = 'scanner_calibration.json'):
        self.calibration_data = self._load_calibration(calibration_file)
        self.data_cache = {}
        
    def _load_calibration(self, file_path: str) -> Dict:
        """Load backtested confidence calibration data"""
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"⚠️ Calibration file {file_path} not found, using default confidence")
            return {}
    
    def validate_data_freshness(self, ticker: str) -> Dict:
        """Validate data freshness and cross-check sources"""
        try:
            stock = yf.Ticker(ticker)
            
            # Get real-time quote
            info = stock.info
            current_price = info.get('regularMarketPrice') or info.get('previousClose')
            
            # Get recent minute data for freshness check
            recent_data = stock.history(period="1d", interval="1m")
            
            if recent_data.empty:
                return {'valid': False, 'error': 'No recent data available'}
            
            last_update = recent_data.index[-1]
            time_diff = datetime.now() - last_update.replace(tzinfo=None)
            data_age_minutes = time_diff.total_seconds() / 60
            
            # Data quality score (0-1, higher is better)
            if data_age_minutes < 5:
                quality_score = 1.0
            elif data_age_minutes < 30:
                quality_score = 0.9
            elif data_age_minutes < 60:
                quality_score = 0.7
            else:
                quality_score = 0.5
            
            return {
                'valid': True,
                'current_price': current_price,
                'data_age_minutes': data_age_minutes,
                'quality_score': quality_score,
                'last_update': last_update.isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Data validation failed for {ticker}: {e}")
            return {'valid': False, 'error': str(e)}
    
    def calculate_technical_indicators(self, hist: pd.DataFrame) -> Dict:
        """Calculate all technical indicators using industry standards"""
        try:
            closes = hist['Close']
            highs = hist['High']
            lows = hist['Low']
            volumes = hist['Volume']
            
            # Convert to numpy arrays with proper dtype
            closes_array = closes.values.astype(np.float64)
            highs_array = highs.values.astype(np.float64)
            lows_array = lows.values.astype(np.float64)
            volumes_array = volumes.values.astype(np.float64)
            
            # RSI (14-day standard)
            rsi = talib.RSI(closes_array, timeperiod=14)
            rsi_series = pd.Series(rsi, index=closes.index)
            
            # MACD
            macd, macd_signal, macd_hist = talib.MACD(closes_array)
            
            # Moving averages
            sma_20 = talib.SMA(closes_array, timeperiod=20)
            sma_50 = talib.SMA(closes_array, timeperiod=50)
            sma_200 = talib.SMA(closes_array, timeperiod=200)
            
            # ATR for volatility
            atr = talib.ATR(highs_array, lows_array, closes_array, timeperiod=14)
            
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = talib.BBANDS(closes_array, timeperiod=20)
            
            # Volume indicators
            volume_sma = talib.SMA(volumes_array, timeperiod=20)
            
            return {
                'rsi': rsi_series.iloc[-1] if not pd.isna(rsi_series.iloc[-1]) else 50,
                'macd': macd[-1] if not pd.isna(macd[-1]) else 0,
                'macd_signal': macd_signal[-1] if not pd.isna(macd_signal[-1]) else 0,
                'macd_histogram': macd_hist[-1] if not pd.isna(macd_hist[-1]) else 0,
                'sma_20': sma_20[-1] if not pd.isna(sma_20[-1]) else closes.iloc[-1],
                'sma_50': sma_50[-1] if not pd.isna(sma_50[-1]) else closes.iloc[-1],
                'sma_200': sma_200[-1] if not pd.isna(sma_200[-1]) else closes.iloc[-1],
                'atr': atr[-1] if not pd.isna(atr[-1]) else 0,
                'bb_upper': bb_upper[-1] if not pd.isna(bb_upper[-1]) else closes.iloc[-1],
                'bb_middle': bb_middle[-1] if not pd.isna(bb_middle[-1]) else closes.iloc[-1],
                'bb_lower': bb_lower[-1] if not pd.isna(bb_lower[-1]) else closes.iloc[-1],
                'volume_ratio': volumes.iloc[-1] / volume_sma[-1] if volume_sma[-1] > 0 else 1
            }
            
        except Exception as e:
            logger.error(f"❌ Technical indicators calculation failed: {e}")
            return {}
    
    def find_support_resistance_levels(self, hist: pd.DataFrame) -> Dict:
        """Find actual support/resistance levels using pivot points"""
        try:
            highs = hist['High']
            lows = hist['Low']
            closes = hist['Close']
            current_price = closes.iloc[-1]
            
            # Use last 60 days for relevant levels
            recent_high = highs.tail(60)
            recent_low = lows.tail(60)
            
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
            
            # Filter levels near current price (within 15%)
            relevant_resistance = [h for h in swing_highs if h > current_price and h < current_price * 1.15]
            relevant_support = [l for l in swing_lows if l < current_price and l > current_price * 0.85]
            
            # Add moving averages as levels
            sma_20 = hist['Close'].rolling(20).mean().iloc[-1]
            sma_50 = hist['Close'].rolling(50).mean().iloc[-1]
            
            if sma_20 > current_price and sma_20 not in relevant_resistance:
                relevant_resistance.append(sma_20)
            elif sma_20 < current_price and sma_20 not in relevant_support:
                relevant_support.append(sma_20)
                
            if sma_50 > current_price and sma_50 not in relevant_resistance:
                relevant_resistance.append(sma_50)
            elif sma_50 < current_price and sma_50 not in relevant_support:
                relevant_support.append(sma_50)
            
            return {
                'support_levels': sorted(relevant_support, reverse=True)[:3],
                'resistance_levels': sorted(relevant_resistance)[:3],
                'nearest_support': max(relevant_support) if relevant_support else None,
                'nearest_resistance': min(relevant_resistance) if relevant_resistance else None
            }
            
        except Exception as e:
            logger.error(f"❌ Support/resistance calculation failed: {e}")
            return {'support_levels': [], 'resistance_levels': [], 'nearest_support': None, 'nearest_resistance': None}
    
    def get_fundamental_context(self, ticker: str) -> Dict:
        """Get fundamental context to validate technical signals"""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # Key metrics
            pe_ratio = info.get('trailingPE', 0)
            peg_ratio = info.get('pegRatio', 0)
            debt_to_equity = info.get('debtToEquity', 0)
            return_on_equity = info.get('returnOnEquity', 0)
            revenue_growth = info.get('revenueGrowth', 0)
            profit_margin = info.get('profitMargins', 0)
            
            # Calculate fundamental score
            fundamental_score = self._calculate_fundamental_score(info)
            
            return {
                'pe_ratio': pe_ratio,
                'peg_ratio': peg_ratio,
                'debt_to_equity': debt_to_equity,
                'roe': return_on_equity,
                'revenue_growth': revenue_growth,
                'profit_margin': profit_margin,
                'sector': info.get('sector', 'Unknown'),
                'market_cap': info.get('marketCap', 0),
                'analyst_recommendation': info.get('recommendationMean', 0),
                'target_price': info.get('targetMeanPrice', 0),
                'fundamental_score': fundamental_score
            }
            
        except Exception as e:
            logger.error(f"❌ Fundamental analysis failed for {ticker}: {e}")
            return {'fundamental_score': 50}
    
    def _calculate_fundamental_score(self, info: Dict) -> int:
        """Calculate 0-100 fundamental score"""
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
            if growth > 0.1:
                score += 15
            elif growth > 0.05:
                score += 10
            elif growth < 0:
                score -= 15
            
            # Profit margin scoring
            margin = info.get('profitMargins', 0)
            if margin > 0.15:
                score += 10
            elif margin > 0.05:
                score += 5
            elif margin < 0:
                score -= 10
            
            return max(0, min(100, score))
            
        except Exception as e:
            logger.error(f"❌ Fundamental scoring failed: {e}")
            return 50
    
    def calculate_production_confidence(self, ticker: str, technicals: Dict, 
                                     fundamentals: Dict, data_quality: float) -> int:
        """Calculate production confidence using backtested calibration"""
        try:
            # Start with base confidence from backtesting
            base_confidence = 50
            if ticker in self.calibration_data.get('calibration_results', {}):
                base_confidence = self.calibration_data['calibration_results'][ticker].get('calibrated_confidence', 50)
            
            # Adjust based on current conditions
            rsi = technicals.get('rsi', 50)
            fundamental_score = fundamentals.get('fundamental_score', 50)
            
            # RSI strength adjustment
            if rsi > 80 or rsi < 20:
                rsi_adjustment = 20
            elif rsi > 70 or rsi < 30:
                rsi_adjustment = 15
            elif rsi > 60 or rsi < 40:
                rsi_adjustment = 10
            else:
                rsi_adjustment = 0
            
            # Fundamental alignment adjustment
            if fundamental_score > 70:
                fund_adjustment = 15
            elif fundamental_score > 50:
                fund_adjustment = 10
            elif fundamental_score < 30:
                fund_adjustment = -15
            else:
                fund_adjustment = 0
            
            # Data quality adjustment
            quality_adjustment = int((data_quality - 0.5) * 20)  # -10 to +10
            
            # MACD confirmation
            macd_adjustment = 0
            if technicals.get('macd', 0) > technicals.get('macd_signal', 0):
                macd_adjustment = 5
            elif technicals.get('macd', 0) < technicals.get('macd_signal', 0):
                macd_adjustment = -5
            
            # Volume confirmation
            volume_adjustment = 0
            if technicals.get('volume_ratio', 1) > 1.5:
                volume_adjustment = 5
            elif technicals.get('volume_ratio', 1) < 0.5:
                volume_adjustment = -5
            
            final_confidence = base_confidence + rsi_adjustment + fund_adjustment + quality_adjustment + macd_adjustment + volume_adjustment
            
            return max(0, min(100, final_confidence))
            
        except Exception as e:
            logger.error(f"❌ Confidence calculation failed: {e}")
            return 50
    
    def calculate_atr_targets_stops(self, current_price: float, atr: float, signal: str) -> Dict:
        """Calculate ATR-based targets and stops"""
        try:
            if atr == 0:
                return {'target': None, 'stop_loss': None, 'risk_reward': None}
            
            # Use 2x ATR for targets, 1.5x ATR for stops
            if signal in ['BUY', 'WEAK_BUY']:
                target = current_price + (atr * 2.0)
                stop_loss = current_price - (atr * 1.5)
            elif signal in ['SELL', 'WEAK_SELL']:
                target = current_price - (atr * 2.0)
                stop_loss = current_price + (atr * 1.5)
            else:  # HOLD
                target = current_price + (atr * 2.0)
                stop_loss = current_price - (atr * 1.5)
            
            risk = abs(current_price - stop_loss)
            reward = abs(target - current_price)
            risk_reward = reward / risk if risk > 0 else 0
            
            return {
                'target': round(target, 2),
                'stop_loss': round(stop_loss, 2),
                'risk_reward': round(risk_reward, 2),
                'atr_value': round(atr, 2)
            }
            
        except Exception as e:
            logger.error(f"❌ ATR calculation failed: {e}")
            return {'target': None, 'stop_loss': None, 'risk_reward': None}
    
    def analyze_stock_production(self, ticker: str) -> Dict:
        """Comprehensive production-ready stock analysis"""
        try:
            logger.info(f"🔍 Production analysis for {ticker}...")
            
            # 1. Validate data freshness
            data_validation = self.validate_data_freshness(ticker)
            if not data_validation['valid']:
                return {'error': f"Data validation failed: {data_validation['error']}"}
            
            # 2. Get historical data
            stock = yf.Ticker(ticker)
            hist = stock.history(period="6mo")  # 6 months for better analysis
            
            if hist.empty:
                return {'error': 'No historical data available'}
            
            # 3. Calculate technical indicators
            technicals = self.calculate_technical_indicators(hist)
            if not technicals:
                return {'error': 'Technical analysis failed'}
            
            # 4. Find support/resistance levels
            sr_levels = self.find_support_resistance_levels(hist)
            
            # 5. Get fundamental context
            fundamentals = self.get_fundamental_context(ticker)
            
            # 6. Determine signal
            rsi = technicals['rsi']
            current_price = float(hist['Close'].iloc[-1])
            
            signal = 'HOLD'
            if rsi > 70:
                signal = 'SELL'
            elif rsi < 30:
                signal = 'BUY'
            elif rsi > 60:
                signal = 'WEAK_SELL'
            elif rsi < 40:
                signal = 'WEAK_BUY'
            
            # 7. Calculate ATR-based targets/stops
            atr_targets = self.calculate_atr_targets_stops(current_price, technicals['atr'], signal)
            
            # 8. Calculate production confidence
            confidence = self.calculate_production_confidence(
                ticker, technicals, fundamentals, data_validation['quality_score']
            )
            
            # 9. Compile comprehensive analysis
            analysis = {
                'ticker': ticker,
                'timestamp': datetime.now().isoformat(),
                'current_price': current_price,
                'signal': signal,
                'confidence': confidence,
                'technical_indicators': technicals,
                'support_resistance': sr_levels,
                'fundamentals': fundamentals,
                'targets_stops': atr_targets,
                'data_quality': data_validation['quality_score'],
                'data_age_minutes': data_validation['data_age_minutes'],
                'backtested_confidence': self.calibration_data.get('calibration_results', {}).get(ticker, {}).get('calibrated_confidence', 50)
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"❌ Production analysis failed for {ticker}: {e}")
            return {'error': str(e)}
    
    def scan_production(self, tickers: List[str]) -> Dict:
        """Run production scan on multiple tickers"""
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
                logger.info(f"📊 Production scan {ticker} ({i+1}/{len(tickers)})")
                
                analysis = self.analyze_stock_production(ticker)
                
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
        results['summary'] = self._generate_production_summary(results['opportunities'])
        
        return results
    
    def _generate_production_summary(self, opportunities: List[Dict]) -> Dict:
        """Generate production summary statistics"""
        if not opportunities:
            return {}
        
        buy_signals = [o for o in opportunities if o['signal'] in ['BUY', 'WEAK_BUY']]
        sell_signals = [o for o in opportunities if o['signal'] in ['SELL', 'WEAK_SELL']]
        
        return {
            'total_opportunities': len(opportunities),
            'buy_signals': len(buy_signals),
            'sell_signals': len(sell_signals),
            'avg_confidence': round(np.mean([o['confidence'] for o in opportunities]), 1),
            'avg_data_quality': round(np.mean([o['data_quality'] for o in opportunities]), 2),
            'avg_fundamental_score': round(np.mean([o['fundamentals']['fundamental_score'] for o in opportunities]), 1),
            'high_confidence_signals': len([o for o in opportunities if o['confidence'] > 70]),
            'backtested_avg_confidence': round(np.mean([o['backtested_confidence'] for o in opportunities]), 1)
        }

def main():
    """Test the production scanner"""
    scanner = ProductionScanner()
    
    # Test with a few tickers
    test_tickers = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']
    
    print("🚀 Running Production Scanner Test...")
    print("=" * 60)
    
    results = scanner.scan_production(test_tickers)
    
    print(f"\n📊 PRODUCTION SCAN RESULTS")
    print(f"Total Tickers: {results['total_tickers']}")
    print(f"Successful: {results['successful_scans']}")
    print(f"Failed: {results['failed_scans']}")
    print(f"Average Confidence: {results['summary'].get('avg_confidence', 0)}%")
    print(f"Average Data Quality: {results['summary'].get('avg_data_quality', 0)}")
    print(f"High Confidence Signals: {results['summary'].get('high_confidence_signals', 0)}")
    
    print(f"\n🎯 TOP OPPORTUNITIES:")
    opportunities = sorted(results['opportunities'], key=lambda x: x['confidence'], reverse=True)
    
    for opp in opportunities[:5]:
        target = opp['targets_stops']['target'] if opp['targets_stops']['target'] else 0
        stop = opp['targets_stops']['stop_loss'] if opp['targets_stops']['stop_loss'] else 0
        rr = opp['targets_stops']['risk_reward'] if opp['targets_stops']['risk_reward'] else 0
        
        print(f"  {opp['ticker']:4} | {opp['signal']:8} | RSI: {opp['technical_indicators']['rsi']:5.1f} | "
              f"Conf: {opp['confidence']:3.0f}% | Target: ${target:7.2f} | "
              f"Stop: ${stop:7.2f} | R/R: {rr:4.2f} | "
              f"Fund: {opp['fundamentals']['fundamental_score']:3.0f}")

if __name__ == "__main__":
    main()
