#!/usr/bin/env python3
"""
QuantEngine Conversational Research Interface

A conversational AI interface that leverages the QuantEngine to answer research questions
and generate comprehensive reports with scenario analysis and probabilities.

Example questions:
- "How will the Fed interest rate decision impact housing prices in 6 months?"
- "Research NFLX company for stock price outlook 3 months from now"
- "What's the outlook for tech stocks given current market conditions?"
"""

import asyncio
import logging
import json
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np

# Add QuantEngine root to path
quant_engine_root = Path(__file__).parent
if str(quant_engine_root) not in sys.path:
    sys.path.insert(0, str(quant_engine_root))

from engine.data_ingestion.live_data_manager import LiveDataManager
from engine.research.regime_detector import MarketRegimeDetector
from engine.research.opportunity_scanner import OpportunityScanner
from engine.research.strategy_optimizer import AdaptiveStrategyResearcher
from engine.feature_builder.feature_builder import FeatureBuilder
from engine.feature_builder.sentiment_signals import SentimentSignalGenerator
from engine.backtest_engine.backtester import VectorizedBacktester
from engine.robustness_lab.robustness_tester import RobustnessTester
from engine.reporting_notes.report_generator import ReportGenerator
from data_broker import QuantBotDataBroker

logger = logging.getLogger(__name__)

class QuantResearchChat:
    """
    Conversational interface for QuantEngine research capabilities
    """
    
    def __init__(self, config_path: str = "config/bot_config.json"):
        self.config = self.load_config(config_path)
        self.data_manager = None
        self.regime_detector = None
        self.opportunity_scanner = None
        self.strategy_researcher = None
        self.feature_builder = None
        self.sentiment_generator = None
        self.backtester = None
        self.robustness_tester = None
        self.report_generator = None
        self.data_broker = None
        
        # Research state
        self.current_research = {}
        self.research_history = []
        
        logger.info("🤖 QuantResearchChat initialized")
    
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration"""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            # Default config
            return {
                "data_sources": {
                    "yahoo_finance": {"enabled": True, "update_interval": 60},
                    "news_feeds": {"enabled": True, "update_interval": 300},
                    "economic_data": {"enabled": True, "update_interval": 3600}
                },
                "universe": {
                    "equities": ["SPY", "QQQ", "IWM", "VTI", "BND"],
                    "sectors": ["XLE", "XLF", "XLK", "XLV", "XLY", "XLI", "XLC", "XLU", "XLB", "XLRE"],
                    "leveraged": ["TQQQ", "SQQQ", "UVXY", "SVXY"]
                }
            }
    
    async def initialize(self):
        """Initialize all QuantEngine components"""
        logger.info("🔧 Initializing QuantEngine components...")
        
        try:
            # Initialize components
            self.data_manager = LiveDataManager(self.config)
            self.regime_detector = MarketRegimeDetector(self.config)
            self.opportunity_scanner = OpportunityScanner(self.config)
            self.strategy_researcher = AdaptiveStrategyResearcher(self.config)
            self.feature_builder = FeatureBuilder(self.config)
            self.sentiment_generator = SentimentSignalGenerator(self.config)
            self.backtester = VectorizedBacktester(self.config)
            self.robustness_tester = RobustnessTester(self.config)
            self.report_generator = ReportGenerator(self.config)
            self.data_broker = QuantBotDataBroker()
            
            # Initialize data manager
            await self.data_manager.initialize()
            
            logger.info("✅ All components initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Component initialization failed: {e}")
            raise
    
    async def ask_question(self, question: str) -> Dict[str, Any]:
        """
        Main interface for asking research questions
        
        Args:
            question: Research question (e.g., "How will Fed rates impact housing?")
            
        Returns:
            Comprehensive research response with scenarios and probabilities
        """
        logger.info(f"🔍 Processing question: {question}")
        
        # Parse the question to extract key components
        parsed_question = self.parse_question(question)
        
        # Generate research response
        response = await self.generate_research_response(parsed_question)
        
        # Store in research history
        self.research_history.append({
            'question': question,
            'parsed': parsed_question,
            'response': response,
            'timestamp': datetime.now().isoformat()
        })
        
        return response
    
    def parse_question(self, question: str) -> Dict[str, Any]:
        """Parse research question to extract key components"""
        
        question_lower = question.lower()
        
        # Extract asset/sector mentions
        assets = []
        sectors = []
        
        # Check for specific tickers
        ticker_patterns = ['spy', 'qqq', 'iwm', 'vti', 'tqqq', 'nflx', 'aapl', 'googl', 'msft', 'tsla']
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
        if "6 months" in question_lower or "6 month" in question_lower:
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
        elif any(word in question_lower for word in ['outlook', 'forecast', 'prediction', 'expect']):
            return 'outlook_analysis'
        elif any(word in question_lower for word in ['research', 'analyze', 'analysis']):
            return 'general_research'
        else:
            return 'general_research'
    
    async def generate_research_response(self, parsed_question: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive research response"""
        
        response = {
            'question': parsed_question['original_question'],
            'analysis_type': parsed_question['question_type'],
            'time_horizon': parsed_question['time_horizon'],
            'scenarios': [],
            'market_context': {},
            'recommendations': [],
            'confidence_score': 0.0,
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            # Get current market context
            market_context = await self.get_market_context()
            response['market_context'] = market_context
            
            # Generate scenarios based on question type
            if parsed_question['question_type'] == 'impact_analysis':
                scenarios = await self.generate_impact_scenarios(parsed_question, market_context)
            elif parsed_question['question_type'] == 'outlook_analysis':
                scenarios = await self.generate_outlook_scenarios(parsed_question, market_context)
            else:
                scenarios = await self.generate_general_scenarios(parsed_question, market_context)
            
            response['scenarios'] = scenarios
            
            # Calculate overall confidence
            response['confidence_score'] = self.calculate_confidence(scenarios, market_context)
            
            # Generate recommendations
            response['recommendations'] = self.generate_recommendations(scenarios, market_context)
            
        except Exception as e:
            logger.error(f"Error generating research response: {e}")
            response['error'] = str(e)
        
        return response
    
    async def get_market_context(self) -> Dict[str, Any]:
        """Get current market context and regime"""
        
        try:
            # Get market overview
            market_overview = await self.data_manager.get_market_overview()
            
            # Detect current regime
            regime_info = await self.regime_detector.detect_regime(market_overview)
            
            # Get recent news sentiment
            market_sentiment = self.data_manager.get_recent_news_sentiment(hours=24)
            
            return {
                'regime': regime_info.get('regime', 'unknown'),
                'regime_confidence': regime_info.get('confidence', 0.0),
                'market_sentiment': market_sentiment,
                'volatility_level': market_overview.get('volatility_level', 20.0),
                'economic_indicators': market_overview.get('economic_indicators', {}),
                'data_quality': market_overview.get('data_quality', 'unknown')
            }
            
        except Exception as e:
            logger.error(f"Error getting market context: {e}")
            return {
                'regime': 'unknown',
                'regime_confidence': 0.0,
                'market_sentiment': 0.0,
                'volatility_level': 20.0,
                'economic_indicators': {},
                'data_quality': 'error'
            }
    
    async def generate_impact_scenarios(self, parsed_question: Dict[str, Any], 
                                      market_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate scenarios for impact analysis questions"""
        
        scenarios = []
        
        # Example: Fed rate decision impact on housing
        if 'fed_decision' in parsed_question['events'] and 'real_estate' in parsed_question['sectors']:
            
            # Scenario 1: Rate Hike
            scenarios.append({
                'name': 'Rate Hike Scenario',
                'description': 'Fed raises rates by 0.25-0.5%',
                'probability': 0.35,
                'impact': {
                    'housing_prices': 'Decline 2-5% over 6 months',
                    'mortgage_rates': 'Increase 0.3-0.6%',
                    'housing_demand': 'Decrease due to higher borrowing costs',
                    'reit_performance': 'Negative impact on REITs'
                },
                'key_drivers': [
                    'Higher borrowing costs reduce affordability',
                    'Reduced demand from first-time buyers',
                    'Potential slowdown in construction activity'
                ],
                'confidence': 0.75
            })
            
            # Scenario 2: Rate Hold
            scenarios.append({
                'name': 'Rate Hold Scenario',
                'description': 'Fed maintains current rates',
                'probability': 0.45,
                'impact': {
                    'housing_prices': 'Stable to slight increase 1-3%',
                    'mortgage_rates': 'Remain relatively stable',
                    'housing_demand': 'Maintain current levels',
                    'reit_performance': 'Neutral to slightly positive'
                },
                'key_drivers': [
                    'Continued demand from demographic trends',
                    'Limited supply supporting prices',
                    'Stable financing environment'
                ],
                'confidence': 0.80
            })
            
            # Scenario 3: Rate Cut
            scenarios.append({
                'name': 'Rate Cut Scenario',
                'description': 'Fed cuts rates by 0.25-0.5%',
                'probability': 0.20,
                'impact': {
                    'housing_prices': 'Increase 3-7% over 6 months',
                    'mortgage_rates': 'Decrease 0.2-0.4%',
                    'housing_demand': 'Increase due to lower borrowing costs',
                    'reit_performance': 'Positive impact on REITs'
                },
                'key_drivers': [
                    'Lower borrowing costs increase affordability',
                    'Increased demand from buyers',
                    'Potential boost to construction activity'
                ],
                'confidence': 0.70
            })
        
        return scenarios
    
    async def generate_outlook_scenarios(self, parsed_question: Dict[str, Any], 
                                       market_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate scenarios for outlook analysis questions"""
        
        scenarios = []
        
        # Example: NFLX stock outlook
        if 'NFLX' in parsed_question['assets']:
            
            # Get current market data for NFLX
            try:
                quotes = await self.data_manager.get_live_quotes(['NFLX'])
                nflx_quote = quotes.get('NFLX')
                current_price = nflx_quote.price if nflx_quote else 0
            except:
                current_price = 400  # Fallback price
            
            # Scenario 1: Bullish
            scenarios.append({
                'name': 'Bullish Scenario',
                'description': 'Strong subscriber growth and content success',
                'probability': 0.30,
                'price_target': f"${current_price * 1.15:.0f} - ${current_price * 1.25:.0f}",
                'key_drivers': [
                    'Strong Q4 subscriber additions',
                    'Successful new content releases',
                    'International expansion success',
                    'Ad-supported tier growth'
                ],
                'risks': [
                    'Increased competition',
                    'Content production costs',
                    'Market saturation'
                ],
                'confidence': 0.70
            })
            
            # Scenario 2: Neutral
            scenarios.append({
                'name': 'Neutral Scenario',
                'description': 'Steady performance with moderate growth',
                'probability': 0.50,
                'price_target': f"${current_price * 0.95:.0f} - ${current_price * 1.10:.0f}",
                'key_drivers': [
                    'Stable subscriber base',
                    'Moderate content investment',
                    'Competitive market position',
                    'Steady revenue growth'
                ],
                'risks': [
                    'Market competition',
                    'Content cost pressures',
                    'Economic headwinds'
                ],
                'confidence': 0.75
            })
            
            # Scenario 3: Bearish
            scenarios.append({
                'name': 'Bearish Scenario',
                'description': 'Subscriber losses and increased competition',
                'probability': 0.20,
                'price_target': f"${current_price * 0.75:.0f} - ${current_price * 0.90:.0f}",
                'key_drivers': [
                    'Subscriber churn increases',
                    'Intense competition from Disney+, HBO Max',
                    'Content production challenges',
                    'Economic pressure on discretionary spending'
                ],
                'risks': [
                    'Market share loss',
                    'Higher content costs',
                    'Regulatory challenges'
                ],
                'confidence': 0.65
            })
        
        return scenarios
    
    async def generate_general_scenarios(self, parsed_question: Dict[str, Any], 
                                       market_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate scenarios for general research questions"""
        
        scenarios = []
        current_regime = market_context.get('regime', 'unknown')
        
        # Base scenarios on current market regime
        if current_regime == 'bull_market':
            scenarios.extend([
                {
                    'name': 'Continued Bull Market',
                    'description': 'Current bull market continues',
                    'probability': 0.60,
                    'key_factors': ['Strong earnings growth', 'Low interest rates', 'Positive sentiment'],
                    'confidence': 0.70
                },
                {
                    'name': 'Market Correction',
                    'description': '10-15% correction within 3 months',
                    'probability': 0.30,
                    'key_factors': ['Valuation concerns', 'Rate hike fears', 'Geopolitical risks'],
                    'confidence': 0.65
                },
                {
                    'name': 'Bear Market',
                    'description': '20%+ decline into bear market',
                    'probability': 0.10,
                    'key_factors': ['Recession fears', 'Aggressive Fed tightening', 'Earnings decline'],
                    'confidence': 0.60
                }
            ])
        else:
            scenarios.extend([
                {
                    'name': 'Market Recovery',
                    'description': 'Gradual recovery from current conditions',
                    'probability': 0.40,
                    'key_factors': ['Policy support', 'Earnings stabilization', 'Risk appetite return'],
                    'confidence': 0.65
                },
                {
                    'name': 'Continued Weakness',
                    'description': 'Current conditions persist or worsen',
                    'probability': 0.45,
                    'key_factors': ['Economic headwinds', 'Policy uncertainty', 'Risk aversion'],
                    'confidence': 0.70
                },
                {
                    'name': 'Volatile Range',
                    'description': 'Sideways trading with high volatility',
                    'probability': 0.15,
                    'key_factors': ['Mixed signals', 'Uncertainty', 'Trading opportunities'],
                    'confidence': 0.60
                }
            ])
        
        return scenarios
    
    def calculate_confidence(self, scenarios: List[Dict[str, Any]], 
                           market_context: Dict[str, Any]) -> float:
        """Calculate overall confidence in the analysis"""
        
        if not scenarios:
            return 0.0
        
        # Base confidence from scenario confidence scores
        scenario_confidences = [s.get('confidence', 0.5) for s in scenarios]
        avg_scenario_confidence = np.mean(scenario_confidences)
        
        # Adjust based on market context
        regime_confidence = market_context.get('regime_confidence', 0.5)
        data_quality = market_context.get('data_quality', 'unknown')
        
        data_quality_multiplier = {
            'excellent': 1.0,
            'good': 0.9,
            'fair': 0.8,
            'poor': 0.6,
            'error': 0.4,
            'unknown': 0.7
        }.get(data_quality, 0.7)
        
        # Weighted average
        overall_confidence = (
            avg_scenario_confidence * 0.5 +
            regime_confidence * 0.3 +
            data_quality_multiplier * 0.2
        )
        
        return min(overall_confidence, 1.0)
    
    def generate_recommendations(self, scenarios: List[Dict[str, Any]], 
                               market_context: Dict[str, Any]) -> List[str]:
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
        
        # Add regime-specific recommendations
        regime = market_context.get('regime', 'unknown')
        if regime == 'high_volatility':
            recommendations.append("Consider volatility strategies and hedging")
        elif regime == 'low_volatility':
            recommendations.append("Look for volatility expansion opportunities")
        
        return recommendations
    
    def format_response(self, response: Dict[str, Any]) -> str:
        """Format research response for display"""
        
        output = []
        output.append(f"# Research Analysis: {response['question']}")
        output.append(f"**Analysis Type:** {response['analysis_type'].replace('_', ' ').title()}")
        output.append(f"**Time Horizon:** {response['time_horizon']}")
        output.append(f"**Confidence Score:** {response['confidence_score']:.1%}")
        output.append("")
        
        # Market Context
        output.append("## Market Context")
        context = response['market_context']
        output.append(f"- **Current Regime:** {context.get('regime', 'Unknown')} (confidence: {context.get('regime_confidence', 0):.1%})")
        output.append(f"- **Market Sentiment:** {context.get('market_sentiment', 0):.2f}")
        output.append(f"- **Volatility Level:** {context.get('volatility_level', 0):.1f}")
        output.append("")
        
        # Scenarios
        output.append("## Scenario Analysis")
        for i, scenario in enumerate(response['scenarios'], 1):
            output.append(f"### Scenario {i}: {scenario['name']}")
            output.append(f"**Probability:** {scenario['probability']:.1%}")
            output.append(f"**Description:** {scenario['description']}")
            
            if 'price_target' in scenario:
                output.append(f"**Price Target:** {scenario['price_target']}")
            
            if 'impact' in scenario:
                output.append("**Impact:**")
                for key, value in scenario['impact'].items():
                    output.append(f"- {key.replace('_', ' ').title()}: {value}")
            
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
        
        # Recommendations
        output.append("## Recommendations")
        for i, rec in enumerate(response['recommendations'], 1):
            output.append(f"{i}. {rec}")
        
        output.append("")
        output.append(f"*Analysis generated on {response['timestamp']}*")
        
        return "\n".join(output)

# Example usage and testing
async def main():
    """Example usage of the QuantResearchChat interface"""
    
    # Initialize the chat interface
    chat = QuantResearchChat()
    await chat.initialize()
    
    # Example questions
    questions = [
        "How will the Fed interest rate decision impact housing prices in 6 months?",
        "Research NFLX company for stock price outlook 3 months from now",
        "What's the outlook for tech stocks given current market conditions?"
    ]
    
    print("🤖 QuantEngine Research Chat Interface")
    print("=" * 50)
    
    for question in questions:
        print(f"\n🔍 Question: {question}")
        print("-" * 50)
        
        # Get research response
        response = await chat.ask_question(question)
        
        # Format and display
        formatted_response = chat.format_response(response)
        print(formatted_response)
        print("\n" + "=" * 50)

if __name__ == "__main__":
    asyncio.run(main())

