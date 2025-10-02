#!/usr/bin/env python3
"""
Sector Research Chat Interface for QuantEngine

A conversational AI interface specifically designed for researching sectors and groups of stocks.
Integrates with LLM for intelligent analysis and provides comprehensive sector insights.

Features:
- Sector-specific analysis and research
- Group/portfolio analysis capabilities
- Real-time data integration
- LLM-powered insights and recommendations
- Interactive chat interface
"""

import asyncio
import json
import sys
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import yfinance as yf
from scipy import stats
import requests

# Add QuantEngine root to path
quant_engine_root = Path(__file__).parent
if str(quant_engine_root) not in sys.path:
    sys.path.insert(0, str(quant_engine_root))

# Import QuantEngine components
try:
    from llm_integration import OllamaLLM
    from data_broker import QuantBotDataBroker
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    print("⚠️ Ollama integration not available")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SectorResearchChat:
    """
    Advanced chat interface for sector and group stock research
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._get_default_config()
        self.research_history = []
        self.data_cache = {}
        self.llm_client = None
        self.data_broker = None
        
        # Initialize LLM if available
        if OLLAMA_AVAILABLE:
            try:
                self.llm_client = OllamaLLM(model="qwen2.5:72b")
                logger.info("✅ LLM integration initialized")
            except Exception as e:
                logger.warning(f"⚠️ LLM not available: {e}")
        
        # Initialize data broker
        try:
            self.data_broker = QuantBotDataBroker()
            logger.info("✅ Data broker initialized")
        except Exception as e:
            logger.warning(f"⚠️ Data broker not available: {e}")
        
        # Sector definitions
        self.sector_definitions = self._get_sector_definitions()
        
        logger.info("🤖 Sector Research Chat initialized")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            "llm": {
                "enabled": True,
                "model": "qwen2.5:72b",
                "temperature": 0.3,
                "max_tokens": 1500
            },
            "data_sources": {
                "yahoo_finance": True,
                "quantbot": True,
                "cache_duration": 300  # 5 minutes
            },
            "analysis": {
                "default_lookback_days": 30,
                "confidence_threshold": 0.6,
                "max_scenarios": 5
            }
        }
    
    def _get_sector_definitions(self) -> Dict[str, Dict[str, Any]]:
        """Get comprehensive sector definitions with ETFs and key stocks"""
        return {
            'technology': {
                'etf': 'XLK',
                'description': 'Technology sector including software, hardware, and semiconductors',
                'key_stocks': ['AAPL', 'MSFT', 'NVDA', 'GOOGL', 'META', 'TSLA', 'AVGO', 'ORCL', 'CRM', 'ADBE'],
                'subsectors': ['software', 'semiconductors', 'hardware', 'cloud', 'ai', 'cybersecurity']
            },
            'healthcare': {
                'etf': 'XLV',
                'description': 'Healthcare sector including pharmaceuticals, biotech, and medical devices',
                'key_stocks': ['JNJ', 'UNH', 'PFE', 'ABBV', 'MRK', 'TMO', 'ABT', 'DHR', 'BMY', 'AMGN'],
                'subsectors': ['pharma', 'biotech', 'medical_devices', 'healthcare_services', 'diagnostics']
            },
            'financials': {
                'etf': 'XLF',
                'description': 'Financial services including banks, insurance, and investment firms',
                'key_stocks': ['BRK-B', 'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'BLK', 'AXP', 'SPGI'],
                'subsectors': ['banks', 'insurance', 'investment_banking', 'fintech', 'real_estate_finance']
            },
            'consumer_discretionary': {
                'etf': 'XLY',
                'description': 'Consumer discretionary including retail, automotive, and leisure',
                'key_stocks': ['AMZN', 'HD', 'MCD', 'NKE', 'SBUX', 'LOW', 'TJX', 'BKNG', 'CMG', 'ABNB'],
                'subsectors': ['retail', 'automotive', 'restaurants', 'leisure', 'ecommerce', 'luxury_goods']
            },
            'consumer_staples': {
                'etf': 'XLP',
                'description': 'Consumer staples including food, beverages, and household products',
                'key_stocks': ['PG', 'KO', 'PEP', 'WMT', 'COST', 'CL', 'KMB', 'GIS', 'K', 'HSY'],
                'subsectors': ['food_beverages', 'household_products', 'personal_care', 'tobacco', 'retail_staples']
            },
            'industrials': {
                'etf': 'XLI',
                'description': 'Industrial sector including manufacturing, aerospace, and transportation',
                'key_stocks': ['BA', 'CAT', 'HON', 'UPS', 'RTX', 'LMT', 'GE', 'MMM', 'DE', 'EMR'],
                'subsectors': ['aerospace', 'machinery', 'transportation', 'defense', 'construction', 'logistics']
            },
            'energy': {
                'etf': 'XLE',
                'description': 'Energy sector including oil, gas, and renewable energy',
                'key_stocks': ['XOM', 'CVX', 'COP', 'EOG', 'SLB', 'PXD', 'MPC', 'VLO', 'KMI', 'PSX'],
                'subsectors': ['oil_gas', 'renewable_energy', 'utilities', 'midstream', 'refining', 'exploration']
            },
            'utilities': {
                'etf': 'XLU',
                'description': 'Utilities sector including electric, gas, and water utilities',
                'key_stocks': ['NEE', 'DUK', 'SO', 'D', 'AEP', 'EXC', 'SRE', 'XEL', 'PEG', 'WEC'],
                'subsectors': ['electric_utilities', 'gas_utilities', 'water_utilities', 'renewable_utilities']
            },
            'real_estate': {
                'etf': 'XLRE',
                'description': 'Real estate sector including REITs and real estate services',
                'key_stocks': ['AMT', 'PLD', 'CCI', 'EQIX', 'PSA', 'O', 'SPG', 'WELL', 'EXR', 'AVB'],
                'subsectors': ['reits', 'real_estate_services', 'residential', 'commercial', 'industrial_re']
            },
            'materials': {
                'etf': 'XLB',
                'description': 'Materials sector including chemicals, metals, and mining',
                'key_stocks': ['LIN', 'APD', 'SHW', 'FCX', 'NEM', 'ECL', 'DOW', 'PPG', 'DD', 'IFF'],
                'subsectors': ['chemicals', 'metals_mining', 'construction_materials', 'packaging', 'specialty_chemicals']
            },
            'communication_services': {
                'etf': 'XLC',
                'description': 'Communication services including telecom, media, and entertainment',
                'key_stocks': ['GOOGL', 'META', 'NFLX', 'DIS', 'CMCSA', 'VZ', 'T', 'CHTR', 'TMUS', 'EA'],
                'subsectors': ['telecom', 'media', 'entertainment', 'social_media', 'streaming', 'gaming']
            }
        }
    
    def parse_question(self, question: str) -> Dict[str, Any]:
        """Parse research question to extract sectors, stocks, and analysis type"""
        
        question_lower = question.lower()
        
        # Extract sectors
        sectors = []
        for sector_name, sector_info in self.sector_definitions.items():
            if sector_name in question_lower:
                sectors.append(sector_name)
            # Check for subsector mentions
            for subsector in sector_info['subsectors']:
                if subsector.replace('_', ' ') in question_lower:
                    sectors.append(sector_name)
            # Check for ETF mentions
            if sector_info['etf'].lower() in question_lower:
                sectors.append(sector_name)
        
        # Extract specific stocks
        stocks = []
        all_stocks = []
        for sector_info in self.sector_definitions.values():
            all_stocks.extend(sector_info['key_stocks'])
        
        for stock in all_stocks:
            if stock.lower() in question_lower:
                stocks.append(stock)
        
        # Extract analysis type
        analysis_type = self._classify_analysis_type(question_lower)
        
        # Extract time horizon
        time_horizon = self._extract_time_horizon(question_lower)
        
        # Extract specific metrics or focus areas
        focus_areas = self._extract_focus_areas(question_lower)
        
        return {
            'original_question': question,
            'sectors': sectors,
            'stocks': stocks,
            'analysis_type': analysis_type,
            'time_horizon': time_horizon,
            'focus_areas': focus_areas,
            'question_type': self._classify_question_type(question_lower)
        }
    
    def _classify_analysis_type(self, question_lower: str) -> str:
        """Classify the type of analysis requested"""
        if any(word in question_lower for word in ['outlook', 'forecast', 'prediction', 'expect']):
            return 'outlook'
        elif any(word in question_lower for word in ['compare', 'comparison', 'vs', 'versus']):
            return 'comparison'
        elif any(word in question_lower for word in ['performance', 'returns', 'gains', 'losses']):
            return 'performance'
        elif any(word in question_lower for word in ['risk', 'volatility', 'stability']):
            return 'risk_analysis'
        elif any(word in question_lower for word in ['valuation', 'value', 'cheap', 'expensive']):
            return 'valuation'
        elif any(word in question_lower for word in ['momentum', 'trend', 'technical']):
            return 'momentum'
        else:
            return 'general'
    
    def _extract_time_horizon(self, question_lower: str) -> str:
        """Extract time horizon from question"""
        if any(phrase in question_lower for phrase in ['1 week', 'weekly', '7 days']):
            return '1 week'
        elif any(phrase in question_lower for phrase in ['1 month', 'monthly', '30 days']):
            return '1 month'
        elif any(phrase in question_lower for phrase in ['3 months', 'quarterly', '90 days']):
            return '3 months'
        elif any(phrase in question_lower for phrase in ['6 months', 'half year', '180 days']):
            return '6 months'
        elif any(phrase in question_lower for phrase in ['1 year', 'yearly', 'annual', '12 months']):
            return '1 year'
        else:
            return '3 months'  # Default
    
    def _extract_focus_areas(self, question_lower: str) -> List[str]:
        """Extract specific focus areas from the question"""
        focus_areas = []
        
        if any(word in question_lower for word in ['earnings', 'revenue', 'profit']):
            focus_areas.append('fundamentals')
        if any(word in question_lower for word in ['technical', 'chart', 'pattern']):
            focus_areas.append('technical')
        if any(word in question_lower for word in ['news', 'sentiment', 'headlines']):
            focus_areas.append('sentiment')
        if any(word in question_lower for word in ['options', 'volatility', 'greeks']):
            focus_areas.append('options')
        if any(word in question_lower for word in ['dividend', 'yield', 'income']):
            focus_areas.append('dividends')
        
        return focus_areas
    
    def _classify_question_type(self, question_lower: str) -> str:
        """Classify the overall question type"""
        if any(word in question_lower for word in ['research', 'analyze', 'analysis']):
            return 'research'
        elif any(word in question_lower for word in ['recommend', 'suggest', 'advice']):
            return 'recommendation'
        elif any(word in question_lower for word in ['explain', 'what', 'how', 'why']):
            return 'explanation'
        else:
            return 'general'
    
    async def fetch_sector_data(self, sector: str, days: int = 30) -> Dict[str, Any]:
        """Fetch comprehensive data for a sector"""
        
        cache_key = f"sector_{sector}_{days}"
        if cache_key in self.data_cache:
            logger.info(f"📊 Using cached sector data for {sector}")
            return self.data_cache[cache_key]
        
        logger.info(f"🔍 Fetching sector data for {sector}...")
        
        try:
            sector_info = self.sector_definitions.get(sector, {})
            if not sector_info:
                return None
            
            # Fetch ETF data
            etf_ticker = sector_info['etf']
            etf_data = await self._fetch_asset_data(etf_ticker, days)
            
            # Fetch key stocks data
            key_stocks = sector_info['key_stocks'][:10]  # Limit to top 10
            stocks_data = {}
            
            for stock in key_stocks:
                stock_data = await self._fetch_asset_data(stock, days)
                if stock_data:
                    stocks_data[stock] = stock_data
            
            # Calculate sector metrics
            sector_metrics = self._calculate_sector_metrics(etf_data, stocks_data)
            
            # Get sector news and sentiment
            news_sentiment = await self._get_sector_sentiment(sector)
            
            sector_data = {
                'sector': sector,
                'description': sector_info['description'],
                'etf_data': etf_data,
                'stocks_data': stocks_data,
                'metrics': sector_metrics,
                'news_sentiment': news_sentiment,
                'timestamp': datetime.now().isoformat()
            }
            
            # Cache the data
            self.data_cache[cache_key] = sector_data
            logger.info(f"✅ Successfully fetched sector data for {sector}")
            return sector_data
            
        except Exception as e:
            logger.error(f"❌ Error fetching sector data for {sector}: {e}")
            return None
    
    async def _fetch_asset_data(self, symbol: str, days: int) -> Dict[str, Any]:
        """Fetch data for a single asset"""
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=f"{days}d")
            
            if hist.empty:
                return None
            
            info = ticker.info
            
            # Calculate technical indicators
            returns = hist['Close'].pct_change().dropna()
            volatility = returns.std() * (252 ** 0.5)  # Annualized
            
            # Calculate RSI
            rsi = self._calculate_rsi(hist['Close'], 14)
            
            # Calculate moving averages
            sma_20 = hist['Close'].rolling(20).mean().iloc[-1] if len(hist) >= 20 else hist['Close'].iloc[-1]
            sma_50 = hist['Close'].rolling(50).mean().iloc[-1] if len(hist) >= 50 else hist['Close'].iloc[-1]
            
            # Calculate support and resistance
            support = hist['Low'].tail(20).min() if len(hist) >= 20 else hist['Low'].min()
            resistance = hist['High'].tail(20).max() if len(hist) >= 20 else hist['High'].max()
            
            return {
                'symbol': symbol,
                'current_price': hist['Close'].iloc[-1],
                'price_change': returns.iloc[-1] if len(returns) > 0 else 0,
                'volume': hist['Volume'].iloc[-1],
                'volatility': volatility,
                'rsi': rsi,
                'sma_20': sma_20,
                'sma_50': sma_50,
                'support': support,
                'resistance': resistance,
                'high_52w': info.get('fiftyTwoWeekHigh', hist['High'].max()),
                'low_52w': info.get('fiftyTwoWeekLow', hist['Low'].min()),
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('trailingPE', 0),
                'beta': info.get('beta', 1.0),
                'returns': returns.tolist(),
                'prices': hist['Close'].tolist(),
                'dates': hist.index.strftime('%Y-%m-%d').tolist()
            }
            
        except Exception as e:
            logger.error(f"❌ Error fetching data for {symbol}: {e}")
            return None
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI indicator"""
        if len(prices) < period + 1:
            return 50
        
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1] if not rsi.empty else 50
    
    def _calculate_sector_metrics(self, etf_data: Dict[str, Any], stocks_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate comprehensive sector metrics"""
        if not etf_data or not stocks_data:
            return {}
        
        # Calculate average metrics across stocks
        stock_returns = [data['price_change'] for data in stocks_data.values() if data]
        stock_volatilities = [data['volatility'] for data in stocks_data.values() if data]
        stock_rsis = [data['rsi'] for data in stocks_data.values() if data]
        
        # Sector performance metrics
        avg_return = np.mean(stock_returns) if stock_returns else 0
        avg_volatility = np.mean(stock_volatilities) if stock_volatilities else 0
        avg_rsi = np.mean(stock_rsis) if stock_rsis else 50
        
        # Calculate sector strength score
        strength_score = self._calculate_sector_strength(etf_data, stocks_data)
        
        # Calculate correlation with market
        market_correlation = self._calculate_market_correlation(etf_data)
        
        return {
            'avg_return': avg_return,
            'avg_volatility': avg_volatility,
            'avg_rsi': avg_rsi,
            'strength_score': strength_score,
            'market_correlation': market_correlation,
            'stocks_analyzed': len(stocks_data),
            'bullish_stocks': len([r for r in stock_returns if r > 0.01]),
            'bearish_stocks': len([r for r in stock_returns if r < -0.01]),
            'overbought_stocks': len([rsi for rsi in stock_rsis if rsi > 70]),
            'oversold_stocks': len([rsi for rsi in stock_rsis if rsi < 30])
        }
    
    def _calculate_sector_strength(self, etf_data: Dict[str, Any], stocks_data: Dict[str, Any]) -> float:
        """Calculate overall sector strength score (0-1)"""
        if not etf_data or not stocks_data:
            return 0.5
        
        # ETF momentum
        etf_momentum = (etf_data['current_price'] - etf_data['sma_20']) / etf_data['sma_20']
        
        # Stock performance distribution
        stock_returns = [data['price_change'] for data in stocks_data.values() if data]
        positive_ratio = len([r for r in stock_returns if r > 0]) / len(stock_returns) if stock_returns else 0.5
        
        # RSI distribution
        stock_rsis = [data['rsi'] for data in stocks_data.values() if data]
        avg_rsi = np.mean(stock_rsis) if stock_rsis else 50
        rsi_score = 1 - abs(avg_rsi - 50) / 50  # Closer to 50 = better
        
        # Combine factors
        strength = (etf_momentum * 0.4 + positive_ratio * 0.3 + rsi_score * 0.3)
        return max(0, min(1, strength))
    
    def _calculate_market_correlation(self, etf_data: Dict[str, Any]) -> float:
        """Calculate correlation with market (SPY)"""
        # This is a simplified version - in practice, you'd calculate actual correlation
        return 0.7  # Placeholder
    
    async def _get_sector_sentiment(self, sector: str) -> Dict[str, Any]:
        """Get news sentiment for a sector"""
        # This would integrate with news APIs in practice
        return {
            'sentiment_score': 0.2,  # Placeholder
            'news_count': 25,
            'positive_news': 12,
            'negative_news': 8,
            'neutral_news': 5
        }
    
    async def generate_sector_analysis(self, parsed_question: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive sector analysis using LLM"""
        
        if not self.llm_client:
            return {"error": "LLM not available"}
        
        # Fetch sector data
        sector_data = {}
        for sector in parsed_question['sectors']:
            data = await self.fetch_sector_data(sector, 30)
            if data:
                sector_data[sector] = data
        
        if not sector_data:
            return {"error": "No sector data available"}
        
        # Prepare context for LLM
        context = self._prepare_sector_context(sector_data, parsed_question)
        
        # Generate LLM analysis
        prompt = self._create_sector_analysis_prompt(parsed_question, context)
        
        try:
            response = self.llm_client.generate(
                prompt=prompt,
                max_tokens=self.config['llm']['max_tokens'],
                temperature=self.config['llm']['temperature']
            )
            
            # Parse LLM response
            analysis = self._parse_llm_response(response, parsed_question)
            
            return {
                'question': parsed_question['original_question'],
                'sectors_analyzed': list(sector_data.keys()),
                'sector_data': sector_data,
                'llm_analysis': analysis,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Error generating sector analysis: {e}")
            return {"error": str(e)}
    
    def _prepare_sector_context(self, sector_data: Dict[str, Any], parsed_question: Dict[str, Any]) -> str:
        """Prepare context for LLM analysis"""
        context_parts = []
        
        for sector, data in sector_data.items():
            context_parts.append(f"\n=== {sector.upper()} SECTOR ===")
            context_parts.append(f"Description: {data['description']}")
            
            # ETF data
            etf = data['etf_data']
            context_parts.append(f"ETF ({etf['symbol']}): ${etf['current_price']:.2f} ({etf['price_change']:+.2%})")
            context_parts.append(f"Volatility: {etf['volatility']:.1%}, RSI: {etf['rsi']:.1f}")
            
            # Key metrics
            metrics = data['metrics']
            context_parts.append(f"Average Return: {metrics['avg_return']:+.2%}")
            context_parts.append(f"Average Volatility: {metrics['avg_volatility']:.1%}")
            context_parts.append(f"Strength Score: {metrics['strength_score']:.2f}")
            context_parts.append(f"Bullish Stocks: {metrics['bullish_stocks']}/{metrics['stocks_analyzed']}")
            
            # Top performers
            top_stocks = sorted(data['stocks_data'].items(), 
                             key=lambda x: x[1]['price_change'], reverse=True)[:3]
            context_parts.append("Top Performers:")
            for stock, stock_data in top_stocks:
                context_parts.append(f"  {stock}: ${stock_data['current_price']:.2f} ({stock_data['price_change']:+.2%})")
        
        return "\n".join(context_parts)
    
    def _create_sector_analysis_prompt(self, parsed_question: Dict[str, Any], context: str) -> str:
        """Create prompt for LLM analysis"""
        
        question = parsed_question['original_question']
        analysis_type = parsed_question['analysis_type']
        time_horizon = parsed_question['time_horizon']
        focus_areas = parsed_question['focus_areas']
        
        prompt = f"""
You are an expert quantitative analyst specializing in sector research. Analyze the following sector data and provide comprehensive insights.

QUESTION: {question}

ANALYSIS TYPE: {analysis_type}
TIME HORIZON: {time_horizon}
FOCUS AREAS: {', '.join(focus_areas) if focus_areas else 'General analysis'}

SECTOR DATA:
{context}

Please provide:

1. SECTOR OVERVIEW:
   - Overall sector performance and trends
   - Key strengths and weaknesses
   - Market positioning and outlook

2. TECHNICAL ANALYSIS:
   - Sector momentum and trend analysis
   - Support and resistance levels
   - Volatility assessment

3. FUNDAMENTAL ANALYSIS:
   - Sector valuation metrics
   - Key drivers and catalysts
   - Risk factors and concerns

4. STOCK-SPECIFIC INSIGHTS:
   - Top performers and underperformers
   - Individual stock analysis highlights
   - Correlation and diversification insights

5. TRADING RECOMMENDATIONS:
   - Specific sector strategies
   - Risk management considerations
   - Entry and exit points

6. RISK ASSESSMENT:
   - Sector-specific risks
   - Market correlation analysis
   - Volatility considerations

Be specific, data-driven, and actionable. Focus on providing clear insights that can guide investment decisions.
"""
        
        return prompt
    
    def _parse_llm_response(self, response: str, parsed_question: Dict[str, Any]) -> Dict[str, Any]:
        """Parse LLM response into structured format"""
        # This is a simplified parser - in practice, you'd use more sophisticated parsing
        return {
            'raw_response': response,
            'sector_overview': self._extract_section(response, 'SECTOR OVERVIEW'),
            'technical_analysis': self._extract_section(response, 'TECHNICAL ANALYSIS'),
            'fundamental_analysis': self._extract_section(response, 'FUNDAMENTAL ANALYSIS'),
            'stock_insights': self._extract_section(response, 'STOCK-SPECIFIC INSIGHTS'),
            'trading_recommendations': self._extract_section(response, 'TRADING RECOMMENDATIONS'),
            'risk_assessment': self._extract_section(response, 'RISK ASSESSMENT')
        }
    
    def _extract_section(self, text: str, section_name: str) -> str:
        """Extract a specific section from LLM response"""
        lines = text.split('\n')
        section_lines = []
        in_section = False
        
        for line in lines:
            if section_name in line.upper():
                in_section = True
                continue
            elif in_section and line.strip() and not line.startswith(' '):
                # Check if we've hit the next section
                if any(keyword in line.upper() for keyword in ['SECTOR OVERVIEW', 'TECHNICAL ANALYSIS', 'FUNDAMENTAL ANALYSIS', 'STOCK-SPECIFIC INSIGHTS', 'TRADING RECOMMENDATIONS', 'RISK ASSESSMENT']):
                    break
            elif in_section:
                section_lines.append(line)
        
        return '\n'.join(section_lines).strip()
    
    async def ask_question(self, question: str) -> Dict[str, Any]:
        """Main interface for asking sector research questions"""
        logger.info(f"🔍 Processing sector research question: {question}")
        
        # Parse the question
        parsed_question = self.parse_question(question)
        
        # Generate analysis
        analysis = await self.generate_sector_analysis(parsed_question)
        
        # Store in history
        self.research_history.append({
            'question': question,
            'parsed': parsed_question,
            'analysis': analysis,
            'timestamp': datetime.now().isoformat()
        })
        
        return analysis
    
    def format_response(self, analysis: Dict[str, Any]) -> str:
        """Format sector analysis response for display"""
        if 'error' in analysis:
            return f"❌ Error: {analysis['error']}"
        
        output = []
        output.append(f"# Sector Research Analysis")
        output.append(f"**Question:** {analysis['question']}")
        output.append(f"**Sectors Analyzed:** {', '.join(analysis['sectors_analyzed'])}")
        output.append(f"**Analysis Time:** {analysis['timestamp']}")
        output.append("")
        
        # LLM Analysis sections
        llm_analysis = analysis.get('llm_analysis', {})
        
        sections = [
            ('Sector Overview', 'sector_overview'),
            ('Technical Analysis', 'technical_analysis'),
            ('Fundamental Analysis', 'fundamental_analysis'),
            ('Stock Insights', 'stock_insights'),
            ('Trading Recommendations', 'trading_recommendations'),
            ('Risk Assessment', 'risk_assessment')
        ]
        
        for title, key in sections:
            content = llm_analysis.get(key, '')
            if content:
                output.append(f"## {title}")
                output.append(content)
                output.append("")
        
        # Raw response if available
        if 'raw_response' in llm_analysis:
            output.append("## Full Analysis")
            output.append(llm_analysis['raw_response'])
        
        return "\n".join(output)

# Example usage and testing
async def main():
    """Example usage of the Sector Research Chat interface"""
    
    # Initialize the chat interface
    chat = SectorResearchChat()
    
    # Example questions
    questions = [
        "Research the technology sector outlook for the next 3 months",
        "Compare healthcare and financial sectors performance",
        "What are the best performing stocks in the energy sector?",
        "Analyze the consumer discretionary sector for trading opportunities",
        "What's the risk profile of the real estate sector?"
    ]
    
    print("🤖 Sector Research QuantEngine Chat")
    print("=" * 60)
    
    for question in questions:
        print(f"\n🔍 Question: {question}")
        print("-" * 60)
        
        # Get analysis
        analysis = await chat.ask_question(question)
        
        # Format and display
        formatted_response = chat.format_response(analysis)
        print(formatted_response)
        print("\n" + "=" * 60)

if __name__ == "__main__":
    asyncio.run(main())
