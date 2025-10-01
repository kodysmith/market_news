#!/usr/bin/env python3
"""
Enhanced Research Chat Interface for QuantEngine

This version includes improved confidence scoring, better technical analysis,
more sophisticated scenario generation, and comprehensive risk assessment.
"""

import json
import sys
import logging
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import yfinance as yf
from scipy import stats

# Add QuantEngine root to path
quant_engine_root = Path(__file__).parent
if str(quant_engine_root) not in sys.path:
    sys.path.insert(0, str(quant_engine_root))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedResearchChat:
    """
    Enhanced chat interface with improved analysis quality and confidence
    """
    
    def __init__(self):
        self.research_history = []
        self.data_cache = {}
        print("🤖 Enhanced Research QuantEngine Chat initialized")
    
    def parse_question(self, question: str) -> Dict[str, Any]:
        """Parse research question to extract key components"""
        
        question_lower = question.lower()
        
        # Extract asset/sector mentions
        assets = []
        sectors = []
        
        # Check for specific tickers
        ticker_patterns = ['spy', 'qqq', 'iwm', 'vti', 'tqqq', 'nflx', 'aapl', 'googl', 'msft', 'tsla', 'nvda', 'amd']
        for ticker in ticker_patterns:
            if ticker in question_lower:
                assets.append(ticker.upper())
        
        # Check for sector mentions
        sector_keywords = {
            'tech': ['technology', 'tech', 'software', 'semiconductor'],
            'financial': ['bank', 'financial', 'finance', 'banking'],
            'energy': ['oil', 'energy', 'gas', 'renewable'],
            'healthcare': ['health', 'pharma', 'biotech', 'medical'],
            'real_estate': ['housing', 'real estate', 'reit', 'property']
        }
        
        for sector, keywords in sector_keywords.items():
            if any(keyword in question_lower for keyword in keywords):
                sectors.append(sector)
        
        # Extract time horizon
        time_horizon = "3 months"  # default
        if "30 days" in question_lower or "1 month" in question_lower:
            time_horizon = "30 days"
        elif "6 months" in question_lower or "6 month" in question_lower:
            time_horizon = "6 months"
        elif "1 year" in question_lower or "12 months" in question_lower:
            time_horizon = "1 year"
        elif "3 months" in question_lower or "3 month" in question_lower:
            time_horizon = "3 months"
        
        # Extract event mentions
        events = []
        if "fed" in question_lower or "federal reserve" in question_lower:
            events.append("fed_decision")
        if "earnings" in question_lower:
            events.append("earnings")
        if "inflation" in question_lower:
            events.append("inflation")
        if "recession" in question_lower:
            events.append("recession")
        
        return {
            'original_question': question,
            'assets': assets,
            'sectors': sectors,
            'time_horizon': time_horizon,
            'events': events,
            'question_type': self.classify_question_type(question_lower)
        }
    
    def classify_question_type(self, question_lower: str) -> str:
        """Classify the type of research question"""
        
        if any(word in question_lower for word in ['impact', 'affect', 'influence']):
            return 'impact_analysis'
        elif any(word in question_lower for word in ['outlook', 'forecast', 'prediction', 'expect', 'evaluate', 'price movement']):
            return 'outlook_analysis'
        elif any(word in question_lower for word in ['research', 'analyze', 'analysis']):
            return 'general_research'
        else:
            return 'general_research'
    
    def fetch_asset_data(self, asset: str, days: int = 30) -> Dict[str, Any]:
        """Fetch real data for an asset with enhanced analysis"""
        
        cache_key = f"{asset}_{days}"
        if cache_key in self.data_cache:
            logger.info(f"📊 Using cached data for {asset}")
            return self.data_cache[cache_key]
        
        logger.info(f"🔍 Fetching real data for {asset}...")
        
        try:
            # Use yfinance to get real data
            ticker = yf.Ticker(asset)
            hist = ticker.history(period=f"{days}d")
            
            if hist.empty:
                logger.warning(f"⚠️ No data found for {asset}")
                return None
            
            # Get current info
            info = ticker.info
            
            # Calculate enhanced technical indicators
            data = self.calculate_enhanced_indicators(asset, hist, info)
            
            # Cache the data
            self.data_cache[cache_key] = data
            logger.info(f"✅ Successfully fetched enhanced data for {asset}")
            return data
            
        except Exception as e:
            logger.error(f"❌ Error fetching data for {asset}: {e}")
            return None
    
    def calculate_enhanced_indicators(self, asset: str, hist, info: Dict) -> Dict[str, Any]:
        """Calculate enhanced technical indicators and analysis"""
        
        # Basic data
        current_price = hist['Close'].iloc[-1]
        returns = hist['Close'].pct_change().dropna()
        
        # Enhanced volatility calculation
        volatility = returns.std() * (252 ** 0.5)  # Annualized
        volatility_30d = returns.tail(30).std() * (252 ** 0.5) if len(returns) >= 30 else volatility
        
        # Trend analysis
        sma_20 = hist['Close'].rolling(20).mean().iloc[-1] if len(hist) >= 20 else current_price
        sma_50 = hist['Close'].rolling(50).mean().iloc[-1] if len(hist) >= 50 else current_price
        trend_strength = (current_price - sma_20) / sma_20
        
        # Momentum indicators
        rsi = self.calculate_rsi(hist['Close'], 14)
        macd_line, macd_signal = self.calculate_macd(hist['Close'])
        
        # Support and resistance levels
        support_level = hist['Low'].tail(20).min() if len(hist) >= 20 else hist['Low'].min()
        resistance_level = hist['High'].tail(20).max() if len(hist) >= 20 else hist['High'].max()
        
        # Volume analysis
        avg_volume = hist['Volume'].mean()
        current_volume = hist['Volume'].iloc[-1]
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
        
        # Price position analysis
        high_52w = info.get('fiftyTwoWeekHigh', hist['High'].max())
        low_52w = info.get('fiftyTwoWeekLow', hist['Low'].min())
        price_position = (current_price - low_52w) / (high_52w - low_52w) if high_52w != low_52w else 0.5
        
        # Risk metrics
        max_drawdown = self.calculate_max_drawdown(hist['Close'])
        sharpe_ratio = self.calculate_sharpe_ratio(returns)
        
        # Market regime indicators
        regime_score = self.calculate_regime_score(hist, returns, volatility)
        
        return {
            'symbol': asset,
            'current_price': current_price,
            'price_change': returns.iloc[-1] if len(returns) > 0 else 0,
            'volume': hist['Volume'].iloc[-1],
            'high_52w': high_52w,
            'low_52w': low_52w,
            'market_cap': info.get('marketCap', 0),
            'pe_ratio': info.get('trailingPE', 0),
            'beta': info.get('beta', 1.0),
            'volatility': volatility,
            'volatility_30d': volatility_30d,
            'returns': returns.tolist(),
            'prices': hist['Close'].tolist(),
            'dates': hist.index.strftime('%Y-%m-%d').tolist(),
            
            # Enhanced indicators
            'trend_strength': trend_strength,
            'sma_20': sma_20,
            'sma_50': sma_50,
            'rsi': rsi,
            'macd_line': macd_line,
            'macd_signal': macd_signal,
            'support_level': support_level,
            'resistance_level': resistance_level,
            'volume_ratio': volume_ratio,
            'price_position': price_position,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'regime_score': regime_score,
            
            # Data quality metrics
            'data_quality': self.assess_data_quality(hist, returns),
            'confidence_factors': self.calculate_confidence_factors(hist, returns, volatility)
        }
    
    def calculate_rsi(self, prices, period=14):
        """Calculate RSI indicator"""
        if len(prices) < period + 1:
            return 50  # Neutral RSI
        
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1] if not rsi.empty else 50
    
    def calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """Calculate MACD indicator"""
        if len(prices) < slow:
            return 0, 0
        
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        macd_signal = macd_line.ewm(span=signal).mean()
        
        return macd_line.iloc[-1], macd_signal.iloc[-1]
    
    def calculate_max_drawdown(self, prices):
        """Calculate maximum drawdown"""
        peak = prices.expanding().max()
        drawdown = (prices - peak) / peak
        return drawdown.min()
    
    def calculate_sharpe_ratio(self, returns, risk_free_rate=0.02):
        """Calculate Sharpe ratio"""
        if len(returns) == 0 or returns.std() == 0:
            return 0
        excess_returns = returns.mean() - risk_free_rate / 252
        return excess_returns / returns.std() * (252 ** 0.5)
    
    def calculate_regime_score(self, hist, returns, volatility):
        """Calculate market regime score (-1 to 1)"""
        # Trend component
        trend_score = (hist['Close'].iloc[-1] - hist['Close'].iloc[-20]) / hist['Close'].iloc[-20] if len(hist) >= 20 else 0
        
        # Volatility component (inverted - lower vol = higher score)
        vol_score = max(0, 1 - volatility / 0.5)  # Normalize to 0.5 max vol
        
        # Volume component
        volume_score = min(1, hist['Volume'].iloc[-1] / hist['Volume'].mean()) if hist['Volume'].mean() > 0 else 0.5
        
        # Combine components
        regime_score = (trend_score * 0.4 + vol_score * 0.3 + volume_score * 0.3)
        return max(-1, min(1, regime_score))
    
    def assess_data_quality(self, hist, returns):
        """Assess the quality of the data"""
        quality_score = 1.0
        
        # Check for missing data
        if hist.isnull().any().any():
            quality_score -= 0.2
        
        # Check for sufficient data points
        if len(hist) < 20:
            quality_score -= 0.3
        elif len(hist) < 50:
            quality_score -= 0.1
        
        # Check for extreme outliers
        if len(returns) > 0:
            outlier_threshold = 3 * returns.std()
            outliers = abs(returns) > outlier_threshold
            if outliers.sum() > len(returns) * 0.05:  # More than 5% outliers
                quality_score -= 0.2
        
        return max(0, quality_score)
    
    def calculate_confidence_factors(self, hist, returns, volatility):
        """Calculate factors that affect confidence in analysis"""
        factors = {
            'data_completeness': min(1.0, len(hist) / 50),  # More data = higher confidence
            'volatility_stability': max(0, 1 - abs(volatility - 0.2) / 0.3),  # Moderate vol = higher confidence
            'trend_clarity': abs(returns.mean()) / returns.std() if returns.std() > 0 else 0,
            'volume_consistency': 1 - abs(hist['Volume'].std() / hist['Volume'].mean()) if hist['Volume'].mean() > 0 else 0.5
        }
        return factors
    
    def generate_enhanced_scenarios(self, parsed_question: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate enhanced scenarios based on comprehensive analysis"""
        
        scenarios = []
        
        # Get market context
        market_data = self.fetch_market_data()
        
        # Process each asset
        for asset in parsed_question['assets']:
            asset_data = self.fetch_asset_data(asset, 30)
            if not asset_data:
                continue
            
            # Generate enhanced scenarios based on comprehensive data
            scenarios.extend(self.generate_asset_scenarios_enhanced(asset, asset_data, market_data))
        
        # If no specific assets, generate general scenarios
        if not scenarios:
            scenarios = self.generate_market_scenarios_enhanced(market_data)
        
        return scenarios
    
    def generate_asset_scenarios_enhanced(self, asset: str, asset_data: Dict[str, Any], 
                                        market_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate enhanced scenarios for a specific asset"""
        
        scenarios = []
        current_price = asset_data['current_price']
        volatility = asset_data['volatility']
        trend_strength = asset_data['trend_strength']
        rsi = asset_data['rsi']
        regime_score = asset_data['regime_score']
        support_level = asset_data['support_level']
        resistance_level = asset_data['resistance_level']
        price_position = asset_data['price_position']
        
        # Calculate scenario probabilities based on multiple factors
        bullish_signals = 0
        bearish_signals = 0
        neutral_signals = 0
        
        # Trend analysis
        if trend_strength > 0.02:
            bullish_signals += 1
        elif trend_strength < -0.02:
            bearish_signals += 1
        else:
            neutral_signals += 1
        
        # RSI analysis
        if rsi < 30:  # Oversold
            bullish_signals += 1
        elif rsi > 70:  # Overbought
            bearish_signals += 1
        else:
            neutral_signals += 1
        
        # Price position analysis
        if price_position < 0.3:  # Near support
            bullish_signals += 1
        elif price_position > 0.7:  # Near resistance
            bearish_signals += 1
        else:
            neutral_signals += 1
        
        # Regime analysis
        if regime_score > 0.2:
            bullish_signals += 1
        elif regime_score < -0.2:
            bearish_signals += 1
        else:
            neutral_signals += 1
        
        # Calculate probabilities
        total_signals = bullish_signals + bearish_signals + neutral_signals
        if total_signals > 0:
            bullish_prob = (bullish_signals / total_signals) * 0.8  # Cap at 80%
            bearish_prob = (bearish_signals / total_signals) * 0.8
            neutral_prob = 1 - bullish_prob - bearish_prob
        else:
            bullish_prob = 0.3
            bearish_prob = 0.3
            neutral_prob = 0.4
        
        # Scenario 1: Bullish
        price_target_up = current_price * (1 + volatility * 1.2)
        scenarios.append({
            'name': f'{asset} Bullish Scenario',
            'description': f'Strong upward movement based on technical and fundamental analysis',
            'probability': bullish_prob,
            'price_target': f'${price_target_up:.2f} - ${price_target_up * 1.15:.2f}',
            'key_drivers': [
                f'Trend strength: {trend_strength:.1%}',
                f'RSI: {rsi:.1f} ({"Oversold" if rsi < 30 else "Neutral" if rsi < 70 else "Overbought"})',
                f'Price position: {price_position:.1%} of 52-week range',
                f'Regime score: {regime_score:.2f}',
                f'Resistance level: ${resistance_level:.2f}'
            ],
            'risks': [
                'Technical resistance at key levels',
                'Market correction risk',
                'Earnings disappointment',
                'Sector rotation'
            ],
            'confidence': min(0.9, 0.6 + (bullish_signals / 4) * 0.3)
        })
        
        # Scenario 2: Range-bound
        range_low = max(support_level, current_price * (1 - volatility * 0.4))
        range_high = min(resistance_level, current_price * (1 + volatility * 0.4))
        scenarios.append({
            'name': f'{asset} Range-bound Scenario',
            'description': f'Price consolidates within support and resistance levels',
            'probability': neutral_prob,
            'price_target': f'${range_low:.2f} - ${range_high:.2f}',
            'key_drivers': [
                f'Support: ${support_level:.2f}, Resistance: ${resistance_level:.2f}',
                f'Current volatility: {volatility:.1%}',
                f'RSI: {rsi:.1f} (neutral zone)',
                'Mixed technical signals',
                'Consolidation pattern'
            ],
            'risks': [
                'Breakout risk in either direction',
                'Volume decline',
                'Time decay for options',
                'Range compression'
            ],
            'confidence': min(0.9, 0.7 + (neutral_signals / 4) * 0.2)
        })
        
        # Scenario 3: Bearish
        price_target_down = current_price * (1 - volatility * 1.2)
        scenarios.append({
            'name': f'{asset} Bearish Scenario',
            'description': f'Downward pressure due to technical and fundamental headwinds',
            'probability': bearish_prob,
            'price_target': f'${price_target_down * 0.9:.2f} - ${price_target_down:.2f}',
            'key_drivers': [
                f'Trend weakness: {trend_strength:.1%}',
                f'RSI: {rsi:.1f} ({"Oversold" if rsi < 30 else "Neutral" if rsi < 70 else "Overbought"})',
                f'Price position: {price_position:.1%} of 52-week range',
                f'Regime score: {regime_score:.2f}',
                f'Support level: ${support_level:.2f}'
            ],
            'risks': [
                'Technical support breakdown',
                'Further downside risk',
                'Market selloff',
                'Sector weakness'
            ],
            'confidence': min(0.9, 0.6 + (bearish_signals / 4) * 0.3)
        })
        
        return scenarios
    
    def generate_market_scenarios_enhanced(self, market_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate enhanced general market scenarios"""
        
        scenarios = []
        sentiment = market_data.get('sentiment', 0)
        
        # Get VIX level for volatility assessment
        vix_data = market_data.get('VIX', {})
        vix_level = vix_data.get('current_price', 20)
        
        # Enhanced probability calculation
        bullish_prob = 0.4
        if sentiment > 0.3 and vix_level < 18:
            bullish_prob = 0.6
        elif sentiment < -0.3 or vix_level > 25:
            bullish_prob = 0.2
        
        neutral_prob = 0.4
        bearish_prob = 1 - bullish_prob - neutral_prob
        
        scenarios.extend([
            {
                'name': 'Market Bullish Scenario',
                'description': 'Positive market momentum with strong fundamentals',
                'probability': bullish_prob,
                'key_drivers': [
                    f'Market sentiment: {sentiment:.1f}',
                    f'VIX level: {vix_level:.1f} (low fear)',
                    'Strong earnings growth',
                    'Supportive monetary policy',
                    'Technical breakout patterns'
                ],
                'risks': [
                    'Valuation concerns',
                    'Interest rate sensitivity',
                    'Geopolitical risks',
                    'Profit-taking pressure'
                ],
                'confidence': 0.75
            },
            {
                'name': 'Market Neutral Scenario',
                'description': 'Mixed signals with sideways movement',
                'probability': neutral_prob,
                'key_drivers': [
                    f'Balanced sentiment: {sentiment:.1f}',
                    f'Moderate volatility: {vix_level:.1f}',
                    'Mixed economic data',
                    'Policy uncertainty',
                    'Technical consolidation'
                ],
                'risks': [
                    'Directional uncertainty',
                    'Low conviction',
                    'Time decay',
                    'Range-bound trading'
                ],
                'confidence': 0.8
            },
            {
                'name': 'Market Bearish Scenario',
                'description': 'Negative momentum with headwinds',
                'probability': bearish_prob,
                'key_drivers': [
                    f'Negative sentiment: {sentiment:.1f}',
                    f'High volatility: {vix_level:.1f}',
                    'Economic headwinds',
                    'Policy concerns',
                    'Technical breakdown patterns'
                ],
                'risks': [
                    'Further downside',
                    'Risk-off sentiment',
                    'Liquidity concerns',
                    'Capitulation selling'
                ],
                'confidence': 0.75
            }
        ])
        
        return scenarios
    
    def fetch_market_data(self) -> Dict[str, Any]:
        """Fetch current market data and sentiment"""
        
        if 'market_data' in self.data_cache:
            return self.data_cache['market_data']
        
        logger.info("🔍 Fetching enhanced market data...")
        
        try:
            # Fetch major indices
            indices = ['SPY', 'QQQ', 'IWM', 'VIX']
            market_data = {}
            
            for index in indices:
                data = self.fetch_asset_data(index, 30)
                if data:
                    market_data[index] = data
            
            # Calculate enhanced market sentiment
            if 'SPY' in market_data and 'VIX' in market_data:
                spy_data = market_data['SPY']
                vix_level = market_data['VIX']['current_price']
                
                # Multi-factor sentiment calculation
                price_momentum = spy_data['trend_strength']
                volatility_factor = max(0, 1 - vix_level / 30)  # Lower VIX = higher sentiment
                regime_factor = spy_data['regime_score']
                
                sentiment = (price_momentum * 0.4 + volatility_factor * 0.3 + regime_factor * 0.3)
                sentiment = max(-1, min(1, sentiment))
            else:
                sentiment = 0.0
            
            market_data['sentiment'] = sentiment
            market_data['timestamp'] = datetime.now().isoformat()
            
            self.data_cache['market_data'] = market_data
            return market_data
            
        except Exception as e:
            logger.error(f"❌ Error fetching market data: {e}")
            return {}
    
    def calculate_enhanced_confidence(self, scenarios: List[Dict[str, Any]], 
                                    market_data: Dict[str, Any]) -> float:
        """Calculate enhanced confidence score"""
        
        if not scenarios:
            return 0.0
        
        # Base confidence from scenario confidence scores
        scenario_confidences = [s.get('confidence', 0.5) for s in scenarios]
        avg_scenario_confidence = np.mean(scenario_confidences)
        
        # Market data quality factor
        data_quality = 1.0
        if 'SPY' in market_data:
            spy_data = market_data['SPY']
            data_quality = spy_data.get('data_quality', 0.8)
        
        # Market stability factor
        stability_factor = 1.0
        if 'VIX' in market_data:
            vix_level = market_data['VIX']['current_price']
            if vix_level > 30:  # High volatility
                stability_factor = 0.8
            elif vix_level < 15:  # Very low volatility
                stability_factor = 0.9
        
        # Overall confidence
        overall_confidence = (
            avg_scenario_confidence * 0.6 +
            data_quality * 0.2 +
            stability_factor * 0.2
        )
        
        return min(overall_confidence, 0.95)  # Cap at 95%
    
    def generate_enhanced_trading_strategy(self, asset: str, scenarios: List[Dict[str, Any]], 
                                         asset_data: Dict[str, Any]) -> List[str]:
        """Generate enhanced trading strategies with risk management"""
        
        strategies = []
        current_price = asset_data['current_price']
        volatility = asset_data['volatility']
        support_level = asset_data['support_level']
        resistance_level = asset_data['resistance_level']
        
        # Find the highest probability scenario
        highest_prob_scenario = max(scenarios, key=lambda x: x.get('probability', 0))
        scenario_name = highest_prob_scenario['name'].lower()
        
        # Calculate position sizing based on volatility
        position_size = min(0.1, 0.05 / volatility) if volatility > 0 else 0.05
        
        if 'bullish' in scenario_name:
            strategies.extend([
                f"🟢 ENHANCED BULLISH STRATEGY for {asset}:",
                f"• Entry: Buy calls at ${current_price * 0.98:.2f} strike",
                f"• Stop Loss: ${current_price * 0.95:.2f} (3% risk)",
                f"• Target 1: ${current_price * 1.03:.2f} (3% gain)",
                f"• Target 2: ${resistance_level:.2f} (resistance level)",
                f"• Timeframe: 30-45 days",
                f"• Position Size: {position_size:.1%} of portfolio",
                f"• Risk/Reward: 1:2.5",
                f"• Volatility: {volatility:.1%} (adjust position size accordingly)"
            ])
        elif 'bearish' in scenario_name:
            strategies.extend([
                f"🔴 ENHANCED BEARISH STRATEGY for {asset}:",
                f"• Entry: Buy puts at ${current_price * 1.02:.2f} strike",
                f"• Stop Loss: ${current_price * 1.05:.2f} (3% risk)",
                f"• Target 1: ${current_price * 0.97:.2f} (3% gain)",
                f"• Target 2: ${support_level:.2f} (support level)",
                f"• Timeframe: 30-45 days",
                f"• Position Size: {position_size:.1%} of portfolio",
                f"• Risk/Reward: 1:2.5",
                f"• Volatility: {volatility:.1%} (adjust position size accordingly)"
            ])
        else:  # Range-bound
            strategies.extend([
                f"🟡 ENHANCED RANGE-BOUND STRATEGY for {asset}:",
                f"• Sell calls at ${resistance_level:.2f} strike",
                f"• Sell puts at ${support_level:.2f} strike",
                f"• Collect premium on both sides",
                f"• Range: ${support_level:.2f} - ${resistance_level:.2f}",
                f"• Timeframe: 30-45 days",
                f"• Position Size: {position_size:.1%} of portfolio",
                f"• Profit if price stays in range",
                f"• Volatility: {volatility:.1%} (adjust strikes accordingly)"
            ])
        
        # Add risk management notes
        strategies.extend([
            "",
            "📊 RISK MANAGEMENT:",
            f"• Max loss per trade: {position_size * 100:.1f}% of portfolio",
            f"• Volatility-adjusted position sizing",
            f"• Technical levels: Support ${support_level:.2f}, Resistance ${resistance_level:.2f}",
            "• Monitor for breakout/breakdown signals",
            "• Consider hedging with opposite positions"
        ])
        
        return strategies
    
    def ask_question(self, question: str) -> Dict[str, Any]:
        """Main interface for asking research questions with enhanced analysis"""
        logger.info(f"🔍 Processing question: {question}")
        
        # Parse the question
        parsed_question = self.parse_question(question)
        
        # Generate enhanced scenarios
        scenarios = self.generate_enhanced_scenarios(parsed_question)
        
        # Get market context
        market_data = self.fetch_market_data()
        
        # Calculate enhanced confidence
        confidence = self.calculate_enhanced_confidence(scenarios, market_data)
        
        # Generate recommendations
        recommendations = self.generate_recommendations(scenarios)
        
        # Generate enhanced trading strategies if requested
        trading_strategies = []
        if any(word in question.lower() for word in ['strategy', 'trade', 'position', 'option']):
            for asset in parsed_question['assets']:
                asset_data = self.fetch_asset_data(asset, 30)
                if asset_data:
                    strategies = self.generate_enhanced_trading_strategy(asset, scenarios, asset_data)
                    trading_strategies.extend(strategies)
        
        # Create enhanced response
        response = {
            'question': question,
            'analysis_type': parsed_question['question_type'],
            'time_horizon': parsed_question['time_horizon'],
            'scenarios': scenarios,
            'recommendations': recommendations,
            'trading_strategies': trading_strategies,
            'confidence_score': confidence,
            'data_sources': 'Real-time market data via yfinance with enhanced technical analysis',
            'analysis_quality': self.assess_analysis_quality(scenarios, market_data),
            'timestamp': datetime.now().isoformat()
        }
        
        # Store in research history
        self.research_history.append({
            'question': question,
            'response': response,
            'timestamp': datetime.now().isoformat()
        })
        
        return response
    
    def assess_analysis_quality(self, scenarios: List[Dict[str, Any]], market_data: Dict[str, Any]) -> str:
        """Assess the overall quality of the analysis"""
        
        if not scenarios:
            return "Low - No scenarios generated"
        
        # Check data quality
        data_quality = "High"
        if 'SPY' in market_data:
            spy_quality = market_data['SPY'].get('data_quality', 0.8)
            if spy_quality < 0.6:
                data_quality = "Low"
            elif spy_quality < 0.8:
                data_quality = "Medium"
        
        # Check scenario confidence
        avg_confidence = np.mean([s.get('confidence', 0.5) for s in scenarios])
        if avg_confidence < 0.6:
            confidence_level = "Low"
        elif avg_confidence < 0.8:
            confidence_level = "Medium"
        else:
            confidence_level = "High"
        
        # Check market stability
        stability = "Stable"
        if 'VIX' in market_data:
            vix_level = market_data['VIX']['current_price']
            if vix_level > 30:
                stability = "Volatile"
            elif vix_level < 15:
                stability = "Very Stable"
        
        return f"{data_quality} data quality, {confidence_level} confidence, {stability} market"
    
    def generate_recommendations(self, scenarios: List[Dict[str, Any]]) -> List[str]:
        """Generate enhanced recommendations based on scenarios"""
        
        recommendations = []
        
        if not scenarios:
            return ["Insufficient data for recommendations"]
        
        # Get highest probability scenario
        highest_prob_scenario = max(scenarios, key=lambda x: x.get('probability', 0))
        prob = highest_prob_scenario.get('probability', 0)
        name = highest_prob_scenario.get('name', '')
        
        # Enhanced recommendations based on probability and confidence
        if prob > 0.6:  # High probability scenario
            if 'bullish' in name.lower():
                recommendations.extend([
                    f"Strong bullish signal ({prob:.1%} probability) - Consider increasing equity exposure",
                    "Focus on growth-oriented positions with momentum",
                    "Monitor for technical breakout confirmation",
                    "Consider options strategies for leverage"
                ])
            elif 'bearish' in name.lower():
                recommendations.extend([
                    f"Strong bearish signal ({prob:.1%} probability) - Consider defensive positioning",
                    "Increase cash allocation and quality holdings",
                    "Consider hedging strategies",
                    "Monitor for technical breakdown confirmation"
                ])
            else:
                recommendations.extend([
                    f"High probability range-bound scenario ({prob:.1%}) - Consider income strategies",
                    "Sell premium with options strategies",
                    "Focus on support/resistance levels",
                    "Monitor for breakout signals"
                ])
        else:  # Moderate probability scenarios
            recommendations.extend([
                "Mixed signals - maintain balanced portfolio allocation",
                "Consider tactical adjustments based on market signals",
                "Monitor key technical levels and economic indicators",
                "Use smaller position sizes due to uncertainty"
            ])
        
        return recommendations
    
    def format_response(self, response: Dict[str, Any]) -> str:
        """Format enhanced research response for display"""
        
        output = []
        output.append(f"# Enhanced Research Analysis: {response['question']}")
        output.append(f"**Analysis Type:** {response['analysis_type'].replace('_', ' ').title()}")
        output.append(f"**Time Horizon:** {response['time_horizon']}")
        output.append(f"**Confidence Score:** {response['confidence_score']:.1%}")
        output.append(f"**Analysis Quality:** {response['analysis_quality']}")
        output.append(f"**Data Sources:** {response['data_sources']}")
        output.append("")
        
        # Scenarios
        output.append("## Enhanced Scenario Analysis")
        for i, scenario in enumerate(response['scenarios'], 1):
            output.append(f"### Scenario {i}: {scenario['name']}")
            output.append(f"**Probability:** {scenario['probability']:.1%}")
            output.append(f"**Description:** {scenario['description']}")
            
            if 'price_target' in scenario:
                output.append(f"**Price Target:** {scenario['price_target']}")
            
            if 'key_drivers' in scenario:
                output.append("**Key Drivers:**")
                for driver in scenario['key_drivers']:
                    output.append(f"- {driver}")
            
            if 'risks' in scenario:
                output.append("**Risks:**")
                for risk in scenario['risks']:
                    output.append(f"- {risk}")
            
            output.append(f"**Confidence:** {scenario.get('confidence', 0):.1%}")
            output.append("")
        
        # Trading Strategies
        if response.get('trading_strategies'):
            output.append("## Enhanced Trading Strategies")
            for strategy in response['trading_strategies']:
                output.append(strategy)
            output.append("")
        
        # Recommendations
        output.append("## Enhanced Recommendations")
        for i, rec in enumerate(response['recommendations'], 1):
            output.append(f"{i}. {rec}")
        
        output.append("")
        output.append(f"*Enhanced analysis generated on {response['timestamp']}*")
        
        return "\n".join(output)

def main():
    """Main entry point for enhanced research chat"""
    print("🤖 Enhanced Research QuantEngine Chat")
    print("=" * 50)
    print("Ask research questions - I'll do comprehensive analysis with enhanced confidence!")
    print("Type 'quit' to exit.")
    print("")
    
    chat = EnhancedResearchChat()
    
    while True:
        try:
            # Get user input
            question = input("\n💬 Your question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not question:
                continue
            
            # Process question
            response = chat.ask_question(question)
            formatted_response = chat.format_response(response)
            print("\n" + "="*80)
            print(formatted_response)
            print("="*80)
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()

