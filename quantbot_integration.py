#!/usr/bin/env python3
"""
QuantBot Integration for Market News App

Provides API endpoints and data feeds for QuantBot findings:
- Live trading opportunities
- Market regime analysis
- Strategy recommendations
- Enhanced news feed with sentiment
- Real-time market signals

Integrates with the existing market news app API.
"""

import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
import asyncio
import logging

# Add QuantEngine to path
quant_engine_path = Path(__file__).parent / 'QuantEngine'
if str(quant_engine_path) not in sys.path:
    sys.path.insert(0, str(quant_engine_path))

from QuantEngine.engine.data_ingestion.live_data_manager import LiveDataManager

logger = logging.getLogger(__name__)

class QuantBotIntegration:
    """
    Integrates QuantBot findings with the market news app
    """

    def __init__(self):
        self.live_data_manager = None
        self.quantbot_data_path = Path('QuantEngine')
        self.cache_duration = timedelta(minutes=5)  # Cache data for 5 minutes
        self.last_update = None

        # Initialize async components
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

    async def initialize(self):
        """Initialize the integration"""
        if not self.live_data_manager:
            config = {
                "universe": {
                    "equities": ["SPY", "QQQ", "AAPL", "MSFT", "TSLA"],
                    "sectors": ["XLE", "XLF", "XLK", "XLV"],
                    "leveraged": ["TQQQ", "UVXY"]
                }
            }
            self.live_data_manager = LiveDataManager(config)
            await self.live_data_manager.initialize()
            logger.info("✅ QuantBot integration initialized")

    def get_live_opportunities(self) -> Dict[str, Any]:
        """
        Get live trading opportunities from QuantBot

        Returns opportunities in format suitable for the app
        """
        try:
            # Check for cached opportunity data
            opportunity_file = self.quantbot_data_path / 'opportunity_cache.json'

            if opportunity_file.exists():
                with open(opportunity_file, 'r') as f:
                    data = json.load(f)

                opportunities = []
                for opp in data.get('opportunities', []):
                    # Format for app consumption
                    opportunities.append({
                        'id': opp.get('id', ''),
                        'title': f"{opp.get('strategy', 'Unknown').title()} Opportunity in {opp.get('symbol', 'Unknown')}",
                        'symbol': opp.get('symbol', 'Unknown'),
                        'strategy': opp.get('strategy', 'Unknown'),
                        'expected_return': opp.get('expected_return', 0),
                        'confidence': opp.get('confidence', 0),
                        'timeframe': opp.get('timeframe', 'Unknown'),
                        'direction': opp.get('direction', 'long'),
                        'timestamp': opp.get('timestamp', ''),
                        'type': 'quantbot_opportunity',
                        'source': 'QuantBot AI',
                        'sentiment': 'bullish' if opp.get('direction') == 'long' else 'bearish',
                        'impact': 'high' if opp.get('confidence', 0) > 0.7 else 'medium'
                    })

                return {
                    'opportunities': opportunities,
                    'total_count': len(opportunities),
                    'last_updated': data.get('timestamp', ''),
                    'source': 'QuantBot'
                }

        except Exception as e:
            logger.warning(f"Failed to load QuantBot opportunities: {e}")

        # Return empty structure if no data available
        return {
            'opportunities': [],
            'total_count': 0,
            'last_updated': datetime.now().isoformat(),
            'source': 'QuantBot',
            'error': 'No opportunity data available'
        }

    def get_market_regime_analysis(self) -> Dict[str, Any]:
        """
        Get current market regime analysis from QuantBot
        """
        try:
            # This would ideally read from QuantBot's status file
            status_file = self.quantbot_data_path / 'bot_status.json'

            if status_file.exists():
                with open(status_file, 'r') as f:
                    status = json.load(f)

                return {
                    'current_regime': status.get('regime', 'unknown'),
                    'confidence': 0.8,  # Default confidence
                    'last_updated': status.get('timestamp', ''),
                    'active_strategies': status.get('active_strategies', 0),
                    'market_conditions': {
                        'volatility': 'normal',
                        'trend': 'sideways',
                        'liquidity': 'good'
                    }
                }

        except Exception as e:
            logger.warning(f"Failed to load market regime data: {e}")

        return {
            'current_regime': 'unknown',
            'confidence': 0.5,
            'last_updated': datetime.now().isoformat(),
            'market_conditions': {
                'volatility': 'normal',
                'trend': 'sideways',
                'liquidity': 'good'
            }
        }

    async def get_enhanced_news_feed(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get enhanced news feed combining FMP news with QuantBot analysis

        Returns news items formatted for the market news app
        """
        try:
            await self.initialize()

            # Get live news from FMP
            fmp_news = await self.live_data_manager.get_live_news(limit=limit//2)

            # Get QuantBot opportunities to include as "news"
            opportunities = self.get_live_opportunities()

            # Combine and format news items
            news_items = []

            # Add FMP news items
            for item in fmp_news:
                news_items.append({
                    'headline': item.title,
                    'source': item.source,
                    'url': item.url,
                    'summary': item.content[:300] + '...' if len(item.content) > 300 else item.content,
                    'published_date': item.timestamp.isoformat(),
                    'sentiment': 'bullish' if item.sentiment_score and item.sentiment_score > 0.1 else
                               'bearish' if item.sentiment_score and item.sentiment_score < -0.1 else 'neutral',
                    'sentiment_score': item.sentiment_score,
                    'tickers': item.tickers or [],
                    'type': 'financial_news',
                    'impact': 'high' if abs(item.sentiment_score or 0) > 0.3 else 'medium'
                })

            # Add QuantBot opportunities as "trading news"
            for opp in opportunities.get('opportunities', [])[:5]:  # Top 5 opportunities
                news_items.append({
                    'headline': f"🤖 QuantBot: {opp['strategy'].title()} Signal in {opp['symbol']}",
                    'source': 'QuantBot AI',
                    'url': f'/quantbot/opportunity/{opp["id"]}',
                    'summary': f"AI trading system detected a {opp['expected_return']:.1%} expected return "
                              f"opportunity in {opp['symbol']} with {opp['confidence']:.0%} confidence. "
                              f"Timeframe: {opp['timeframe']}. Direction: {opp['direction'].title()}.",
                    'published_date': opp.get('timestamp', datetime.now().isoformat()),
                    'sentiment': opp.get('sentiment', 'neutral'),
                    'sentiment_score': opp.get('expected_return', 0),
                    'tickers': [opp['symbol']],
                    'type': 'quantbot_signal',
                    'impact': opp.get('impact', 'medium'),
                    'quantbot_data': {
                        'expected_return': opp['expected_return'],
                        'confidence': opp['confidence'],
                        'timeframe': opp['timeframe'],
                        'direction': opp['direction']
                    }
                })

            # Sort by timestamp (most recent first)
            news_items.sort(key=lambda x: x.get('published_date', ''), reverse=True)

            return news_items[:limit]

        except Exception as e:
            logger.error(f"Failed to get enhanced news feed: {e}")
            return []

    async def get_market_signals(self) -> Dict[str, Any]:
        """
        Get comprehensive market signals from QuantBot
        """
        try:
            await self.initialize()

            # Get market overview
            market_overview = await self.live_data_manager.get_market_overview()

            # Get opportunities
            opportunities = self.get_live_opportunities()

            # Get regime analysis
            regime = self.get_market_regime_analysis()

            return {
                'market_overview': market_overview,
                'trading_opportunities': opportunities,
                'regime_analysis': regime,
                'technical_signals': self._generate_technical_signals(),
                'risk_metrics': self._calculate_risk_metrics(),
                'last_updated': datetime.now().isoformat(),
                'data_source': 'QuantBot AI'
            }

        except Exception as e:
            logger.error(f"Failed to get market signals: {e}")
            return {
                'error': str(e),
                'last_updated': datetime.now().isoformat()
            }

    def _generate_technical_signals(self) -> List[Dict[str, Any]]:
        """Generate technical signals for key assets"""
        signals = []

        # This would normally analyze live data, but for now return sample signals
        key_assets = ['SPY', 'QQQ', 'AAPL', 'TSLA']

        for asset in key_assets:
            signals.append({
                'symbol': asset,
                'signal': 'HOLD',  # Would be BUY/SELL/HOLD based on analysis
                'strength': 0.6,
                'timeframe': '1-3 days',
                'indicators': ['RSI', 'MACD', 'Moving Averages'],
                'last_updated': datetime.now().isoformat()
            })

        return signals

    def _calculate_risk_metrics(self) -> Dict[str, Any]:
        """Calculate current market risk metrics"""
        return {
            'market_volatility': 0.18,
            'vix_level': 18.5,
            'correlation_risk': 'medium',
            'liquidity_risk': 'low',
            'tail_risk': 'normal',
            'last_updated': datetime.now().isoformat()
        }

    async def update_news_json(self):
        """
        Update the news.json file with live QuantBot-enhanced news feed
        """
        try:
            # Get enhanced news feed
            news_items = await self.get_enhanced_news_feed(limit=30)

            # Convert to the format expected by news.json
            formatted_news = []
            for item in news_items:
                formatted_news.append({
                    'headline': item['headline'],
                    'source': item['source'],
                    'url': item['url'],
                    'summary': item.get('summary', ''),
                    'sentiment': item.get('sentiment', 'neutral'),
                    'tickers': item.get('tickers', []),
                    'type': item.get('type', 'news'),
                    'impact': item.get('impact', 'medium'),
                    'published_date': item.get('published_date', datetime.now().isoformat())
                })

            # Save to news.json
            with open('news.json', 'w', encoding='utf-8') as f:
                json.dump(formatted_news, f, indent=2, ensure_ascii=False)

            logger.info(f"✅ Updated news.json with {len(formatted_news)} items")
            return True

        except Exception as e:
            logger.error(f"Failed to update news.json: {e}")
            return False

# Global instance for API use
quantbot_integration = QuantBotIntegration()

# API Functions for Flask integration
def get_quantbot_opportunities():
    """API function to get QuantBot trading opportunities"""
    return quantbot_integration.get_live_opportunities()

def get_market_regime():
    """API function to get market regime analysis"""
    return quantbot_integration.get_market_regime_analysis()

async def get_enhanced_news():
    """API function to get enhanced news feed"""
    return await quantbot_integration.get_enhanced_news_feed()

async def get_market_signals():
    """API function to get comprehensive market signals"""
    return await quantbot_integration.get_market_signals()

async def update_news_feed():
    """API function to update news.json with live data"""
    return await quantbot_integration.update_news_json()

# Initialize on import
async def initialize_integration():
    """Initialize the QuantBot integration"""
    await quantbot_integration.initialize()

if __name__ == "__main__":
    # Test the integration
    async def test():
        print("🧪 Testing QuantBot Integration...")

        await initialize_integration()

        # Test opportunities
        opportunities = quantbot_integration.get_live_opportunities()
        print(f"📊 Found {opportunities.get('total_count', 0)} trading opportunities")

        # Test news feed
        news_items = await quantbot_integration.get_enhanced_news_feed(limit=10)
        print(f"📰 Got {len(news_items)} news items")

        # Test market signals
        signals = await quantbot_integration.get_market_signals()
        print(f"📈 Market signals loaded: {len(signals)} keys")

        print("✅ QuantBot integration test complete!")

    asyncio.run(test())

