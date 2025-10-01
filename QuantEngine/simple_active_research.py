#!/usr/bin/env python3
"""
Simple Active Research Chat Interface

This version actively fetches real data using yfinance without requiring aiohttp.
"""

import json
import sys
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import yfinance as yf

# Add QuantEngine root to path
quant_engine_root = Path(__file__).parent
if str(quant_engine_root) not in sys.path:
    sys.path.insert(0, str(quant_engine_root))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleActiveResearchChat:
    """
    Simple chat interface that actively researches and fetches data when needed
    """
    
    def __init__(self):
        self.research_history = []
        self.data_cache = {}
        print("🤖 Simple Active Research QuantEngine Chat initialized")
    
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
        """Fetch real data for an asset"""
        
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
            
            # Calculate technical indicators
            data = {
                'symbol': asset,
                'current_price': hist['Close'].iloc[-1],
                'price_change': hist['Close'].pct_change().iloc[-1],
                'volume': hist['Volume'].iloc[-1],
                'high_52w': info.get('fiftyTwoWeekHigh', hist['High'].max()),
                'low_52w': info.get('fiftyTwoWeekLow', hist['Low'].min()),
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('trailingPE', 0),
                'beta': info.get('beta', 1.0),
                'volatility': hist['Close'].pct_change().std() * (252 ** 0.5),  # Annualized
                'returns': hist['Close'].pct_change().dropna().tolist(),
                'prices': hist['Close'].tolist(),
                'dates': hist.index.strftime('%Y-%m-%d').tolist()
            }
            
            # Cache the data
            self.data_cache[cache_key] = data
            logger.info(f"✅ Successfully fetched data for {asset}")
            return data
            
        except Exception as e:
            logger.error(f"❌ Error fetching data for {asset}: {e}")
            return None
    
    def fetch_market_data(self) -> Dict[str, Any]:
        """Fetch current market data and sentiment"""
        
        if 'market_data' in self.data_cache:
            return self.data_cache['market_data']
        
        logger.info("🔍 Fetching market data...")
        
        try:
            # Fetch major indices
            indices = ['SPY', 'QQQ', 'IWM', 'VIX']
            market_data = {}
            
            for index in indices:
                data = self.fetch_asset_data(index, 30)
                if data:
                    market_data[index] = data
            
            # Calculate market sentiment
            if 'SPY' in market_data and 'VIX' in market_data:
                spy_change = market_data['SPY']['price_change']
                vix_level = market_data['VIX']['current_price']
                
                # Simple sentiment calculation
                if spy_change > 0.01 and vix_level < 20:
                    sentiment = 0.7  # Bullish
                elif spy_change < -0.01 and vix_level > 25:
                    sentiment = -0.7  # Bearish
                else:
                    sentiment = 0.0  # Neutral
            else:
                sentiment = 0.0
            
            market_data['sentiment'] = sentiment
            market_data['timestamp'] = datetime.now().isoformat()
            
            self.data_cache['market_data'] = market_data
            return market_data
            
        except Exception as e:
            logger.error(f"❌ Error fetching market data: {e}")
            return {}
    
    def generate_data_driven_scenarios(self, parsed_question: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate scenarios based on real data analysis"""
        
        scenarios = []
        
        # Get market context
        market_data = self.fetch_market_data()
        
        # Process each asset
        for asset in parsed_question['assets']:
            asset_data = self.fetch_asset_data(asset, 30)
            if not asset_data:
                continue
            
            # Generate scenarios based on actual data
            scenarios.extend(self.generate_asset_scenarios(asset, asset_data, market_data))
        
        # If no specific assets, generate general scenarios
        if not scenarios:
            scenarios = self.generate_market_scenarios(market_data)
        
        return scenarios
    
    def generate_asset_scenarios(self, asset: str, asset_data: Dict[str, Any], 
                               market_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate scenarios for a specific asset based on real data"""
        
        scenarios = []
        current_price = asset_data['current_price']
        volatility = asset_data['volatility']
        returns = asset_data['returns']
        
        # Calculate recent performance
        recent_return = sum(returns[-5:]) if len(returns) >= 5 else 0  # Last 5 days
        
        # Scenario 1: Bullish (based on momentum and market conditions)
        bullish_prob = 0.3
        if recent_return > 0.02:  # Strong recent performance
            bullish_prob = 0.4
        elif recent_return < -0.02:  # Poor recent performance
            bullish_prob = 0.2
        
        price_target_up = current_price * (1 + volatility * 1.5)
        scenarios.append({
            'name': f'{asset} Bullish Scenario',
            'description': f'Strong upward movement based on momentum and market conditions',
            'probability': bullish_prob,
            'price_target': f'${price_target_up:.2f} - ${price_target_up * 1.1:.2f}',
            'key_drivers': [
                f'Recent momentum: {recent_return:.1%} over last 5 days',
                f'Market sentiment: {market_data.get("sentiment", 0):.1f}',
                f'Volatility level: {volatility:.1%}',
                'Positive earnings outlook'
            ],
            'risks': [
                'Market correction risk',
                'Earnings disappointment',
                'Sector rotation'
            ],
            'confidence': 0.7
        })
        
        # Scenario 2: Neutral/Range-bound
        neutral_prob = 0.5
        if volatility > 0.3:  # High volatility
            neutral_prob = 0.3
        elif volatility < 0.15:  # Low volatility
            neutral_prob = 0.6
        
        range_low = current_price * (1 - volatility * 0.5)
        range_high = current_price * (1 + volatility * 0.5)
        scenarios.append({
            'name': f'{asset} Range-bound Scenario',
            'description': f'Price consolidates within a range',
            'probability': neutral_prob,
            'price_target': f'${range_low:.2f} - ${range_high:.2f}',
            'key_drivers': [
                f'Current volatility: {volatility:.1%}',
                'Mixed market signals',
                'Consolidation after recent moves',
                'Support and resistance levels'
            ],
            'risks': [
                'Breakout risk',
                'Volume decline',
                'Time decay'
            ],
            'confidence': 0.75
        })
        
        # Scenario 3: Bearish
        bearish_prob = 1 - bullish_prob - neutral_prob
        price_target_down = current_price * (1 - volatility * 1.5)
        scenarios.append({
            'name': f'{asset} Bearish Scenario',
            'description': f'Downward pressure due to headwinds',
            'probability': bearish_prob,
            'price_target': f'${price_target_down * 0.9:.2f} - ${price_target_down:.2f}',
            'key_drivers': [
                f'Recent weakness: {recent_return:.1%} over last 5 days',
                f'Market headwinds',
                f'High volatility: {volatility:.1%}',
                'Earnings concerns'
            ],
            'risks': [
                'Further downside risk',
                'Market selloff',
                'Sector weakness'
            ],
            'confidence': 0.7
        })
        
        return scenarios
    
    def generate_market_scenarios(self, market_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate general market scenarios based on real data"""
        
        scenarios = []
        sentiment = market_data.get('sentiment', 0)
        
        # Get VIX level for volatility assessment
        vix_data = market_data.get('VIX', {})
        vix_level = vix_data.get('current_price', 20)
        
        # Scenario 1: Bullish
        bullish_prob = 0.4
        if sentiment > 0.3 and vix_level < 18:
            bullish_prob = 0.5
        elif sentiment < -0.3 or vix_level > 25:
            bullish_prob = 0.2
        
        scenarios.append({
            'name': 'Market Bullish Scenario',
            'description': 'Positive market momentum continues',
            'probability': bullish_prob,
            'key_drivers': [
                f'Market sentiment: {sentiment:.1f}',
                f'VIX level: {vix_level:.1f} (low fear)',
                'Strong earnings growth',
                'Supportive monetary policy'
            ],
            'risks': [
                'Valuation concerns',
                'Interest rate sensitivity',
                'Geopolitical risks'
            ],
            'confidence': 0.7
        })
        
        # Scenario 2: Neutral
        neutral_prob = 0.4
        scenarios.append({
            'name': 'Market Neutral Scenario',
            'description': 'Mixed signals with sideways movement',
            'probability': neutral_prob,
            'key_drivers': [
                f'Balanced sentiment: {sentiment:.1f}',
                f'Moderate volatility: {vix_level:.1f}',
                'Mixed economic data',
                'Policy uncertainty'
            ],
            'risks': [
                'Directional uncertainty',
                'Low conviction',
                'Time decay'
            ],
            'confidence': 0.75
        })
        
        # Scenario 3: Bearish
        bearish_prob = 1 - bullish_prob - neutral_prob
        scenarios.append({
            'name': 'Market Bearish Scenario',
            'description': 'Negative momentum and headwinds',
            'probability': bearish_prob,
            'key_drivers': [
                f'Negative sentiment: {sentiment:.1f}',
                f'High volatility: {vix_level:.1f}',
                'Economic headwinds',
                'Policy concerns'
            ],
            'risks': [
                'Further downside',
                'Risk-off sentiment',
                'Liquidity concerns'
            ],
            'confidence': 0.7
        })
        
        return scenarios
    
    def generate_trading_strategy(self, asset: str, scenarios: List[Dict[str, Any]], 
                                asset_data: Dict[str, Any]) -> List[str]:
        """Generate specific trading strategies based on scenarios and data"""
        
        strategies = []
        current_price = asset_data['current_price']
        volatility = asset_data['volatility']
        
        # Find the highest probability scenario
        highest_prob_scenario = max(scenarios, key=lambda x: x.get('probability', 0))
        scenario_name = highest_prob_scenario['name'].lower()
        
        if 'bullish' in scenario_name:
            strategies.extend([
                f"🟢 BULLISH STRATEGY for {asset}:",
                f"• Buy calls at ${current_price * 0.98:.2f} strike",
                f"• Set stop loss at ${current_price * 0.95:.2f}",
                f"• Target: ${current_price * 1.05:.2f} (5% gain)",
                f"• Timeframe: 30 days",
                f"• Risk/Reward: 1:2.5"
            ])
        elif 'bearish' in scenario_name:
            strategies.extend([
                f"🔴 BEARISH STRATEGY for {asset}:",
                f"• Buy puts at ${current_price * 1.02:.2f} strike",
                f"• Set stop loss at ${current_price * 1.05:.2f}",
                f"• Target: ${current_price * 0.95:.2f} (5% gain)",
                f"• Timeframe: 30 days",
                f"• Risk/Reward: 1:2.5"
            ])
        else:  # Range-bound
            strategies.extend([
                f"🟡 RANGE-BOUND STRATEGY for {asset}:",
                f"• Sell calls at ${current_price * 1.03:.2f} strike",
                f"• Sell puts at ${current_price * 0.97:.2f} strike",
                f"• Collect premium on both sides",
                f"• Range: ${current_price * 0.97:.2f} - ${current_price * 1.03:.2f}",
                f"• Timeframe: 30 days",
                f"• Profit if price stays in range"
            ])
        
        # Add volatility-based adjustments
        if volatility > 0.25:  # High volatility
            strategies.append("⚠️ High volatility detected - consider smaller position sizes")
        elif volatility < 0.15:  # Low volatility
            strategies.append("💡 Low volatility - consider longer timeframes for options")
        
        return strategies
    
    def ask_question(self, question: str) -> Dict[str, Any]:
        """
        Main interface for asking research questions with active data fetching
        """
        logger.info(f"🔍 Processing question: {question}")
        
        # Parse the question
        parsed_question = self.parse_question(question)
        
        # Generate data-driven scenarios
        scenarios = self.generate_data_driven_scenarios(parsed_question)
        
        # Calculate confidence
        confidence = self.calculate_confidence(scenarios)
        
        # Generate recommendations
        recommendations = self.generate_recommendations(scenarios)
        
        # Generate trading strategies if requested
        trading_strategies = []
        if any(word in question.lower() for word in ['strategy', 'trade', 'position', 'option']):
            for asset in parsed_question['assets']:
                asset_data = self.fetch_asset_data(asset, 30)
                if asset_data:
                    strategies = self.generate_trading_strategy(asset, scenarios, asset_data)
                    trading_strategies.extend(strategies)
        
        # Create response
        response = {
            'question': question,
            'analysis_type': parsed_question['question_type'],
            'time_horizon': parsed_question['time_horizon'],
            'scenarios': scenarios,
            'recommendations': recommendations,
            'trading_strategies': trading_strategies,
            'confidence_score': confidence,
            'data_sources': 'Real-time market data via yfinance',
            'timestamp': datetime.now().isoformat()
        }
        
        # Store in research history
        self.research_history.append({
            'question': question,
            'response': response,
            'timestamp': datetime.now().isoformat()
        })
        
        return response
    
    def calculate_confidence(self, scenarios: List[Dict[str, Any]]) -> float:
        """Calculate overall confidence in the analysis"""
        
        if not scenarios:
            return 0.0
        
        # Average confidence from scenarios
        confidences = [s.get('confidence', 0.5) for s in scenarios]
        return sum(confidences) / len(confidences)
    
    def generate_recommendations(self, scenarios: List[Dict[str, Any]]) -> List[str]:
        """Generate actionable recommendations based on scenarios"""
        
        recommendations = []
        
        if not scenarios:
            return ["Insufficient data for recommendations"]
        
        # Get highest probability scenario
        highest_prob_scenario = max(scenarios, key=lambda x: x.get('probability', 0))
        
        # Generate recommendations based on scenario
        if 'bullish' in highest_prob_scenario.get('name', '').lower():
            recommendations.extend([
                "Consider increasing equity exposure",
                "Focus on growth-oriented positions",
                "Monitor for signs of market rotation"
            ])
        elif 'bearish' in highest_prob_scenario.get('name', '').lower():
            recommendations.extend([
                "Consider defensive positioning",
                "Increase cash allocation",
                "Focus on quality and dividend stocks"
            ])
        else:
            recommendations.extend([
                "Maintain balanced portfolio allocation",
                "Consider tactical adjustments based on market signals",
                "Monitor key economic indicators"
            ])
        
        return recommendations
    
    def format_response(self, response: Dict[str, Any]) -> str:
        """Format research response for display"""
        
        output = []
        output.append(f"# Research Analysis: {response['question']}")
        output.append(f"**Analysis Type:** {response['analysis_type'].replace('_', ' ').title()}")
        output.append(f"**Time Horizon:** {response['time_horizon']}")
        output.append(f"**Confidence Score:** {response['confidence_score']:.1%}")
        output.append(f"**Data Sources:** {response['data_sources']}")
        output.append("")
        
        # Scenarios
        output.append("## Scenario Analysis")
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
            output.append("## Trading Strategies")
            for strategy in response['trading_strategies']:
                output.append(strategy)
            output.append("")
        
        # Recommendations
        output.append("## Recommendations")
        for i, rec in enumerate(response['recommendations'], 1):
            output.append(f"{i}. {rec}")
        
        output.append("")
        output.append(f"*Analysis generated on {response['timestamp']}*")
        
        return "\n".join(output)

def main():
    """Main entry point for simple active research chat"""
    print("🤖 Simple Active Research QuantEngine Chat")
    print("=" * 50)
    print("Ask research questions - I'll fetch real data and do active research!")
    print("Type 'quit' to exit.")
    print("")
    
    chat = SimpleActiveResearchChat()
    
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

