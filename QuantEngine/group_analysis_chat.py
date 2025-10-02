#!/usr/bin/env python3
"""
Group Analysis Chat Interface for QuantEngine

A conversational AI interface for analyzing groups of stocks, portfolios, and custom stock lists.
Provides comprehensive analysis of stock groups with correlation analysis, risk assessment, and recommendations.

Features:
- Custom stock group analysis
- Portfolio correlation analysis
- Risk assessment and diversification
- Group performance comparison
- LLM-powered insights
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
from scipy.stats import pearsonr
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

class GroupAnalysisChat:
    """
    Advanced chat interface for analyzing groups of stocks and portfolios
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
        
        # Predefined stock groups
        self.predefined_groups = self._get_predefined_groups()
        
        logger.info("🤖 Group Analysis Chat initialized")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            "llm": {
                "enabled": True,
                "model": "qwen2.5:72b",
                "temperature": 0.3,
                "max_tokens": 2000
            },
            "data_sources": {
                "yahoo_finance": True,
                "quantbot": True,
                "cache_duration": 300
            },
            "analysis": {
                "default_lookback_days": 30,
                "correlation_threshold": 0.7,
                "max_stocks_per_group": 20
            }
        }
    
    def _get_predefined_groups(self) -> Dict[str, Dict[str, Any]]:
        """Get predefined stock groups for analysis"""
        return {
            'mega_cap_tech': {
                'name': 'Mega Cap Technology',
                'description': 'Largest technology companies by market cap',
                'stocks': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'AVGO', 'ORCL', 'CRM'],
                'category': 'technology'
            },
            'dividend_aristocrats': {
                'name': 'Dividend Aristocrats',
                'description': 'S&P 500 companies with 25+ years of dividend increases',
                'stocks': ['JNJ', 'PG', 'KO', 'PEP', 'WMT', 'CL', 'KMB', 'GIS', 'K', 'HSY', 'MCD', 'TGT'],
                'category': 'dividend'
            },
            'growth_stocks': {
                'name': 'High Growth Stocks',
                'description': 'Companies with high growth potential',
                'stocks': ['NVDA', 'AMD', 'TSLA', 'NFLX', 'SQ', 'PYPL', 'ZM', 'ROKU', 'CRWD', 'SNOW'],
                'category': 'growth'
            },
            'value_stocks': {
                'name': 'Value Stocks',
                'description': 'Undervalued companies with strong fundamentals',
                'stocks': ['BRK-B', 'JPM', 'BAC', 'WFC', 'XOM', 'CVX', 'JNJ', 'PG', 'KO', 'WMT'],
                'category': 'value'
            },
            'reits': {
                'name': 'Real Estate Investment Trusts',
                'description': 'Diversified REIT portfolio',
                'stocks': ['AMT', 'PLD', 'CCI', 'EQIX', 'PSA', 'O', 'SPG', 'WELL', 'EXR', 'AVB'],
                'category': 'real_estate'
            },
            'biotech': {
                'name': 'Biotechnology',
                'description': 'Leading biotechnology companies',
                'stocks': ['GILD', 'AMGN', 'BIIB', 'REGN', 'VRTX', 'ILMN', 'MRNA', 'BNTX', 'CRISPR', 'EDIT'],
                'category': 'healthcare'
            },
            'fintech': {
                'name': 'Financial Technology',
                'description': 'Technology companies in financial services',
                'stocks': ['SQ', 'PYPL', 'V', 'MA', 'AXP', 'COIN', 'SOFI', 'UPST', 'AFRM', 'HOOD'],
                'category': 'financial_technology'
            },
            'energy_clean': {
                'name': 'Clean Energy',
                'description': 'Renewable and clean energy companies',
                'stocks': ['TSLA', 'NEE', 'ENPH', 'SEDG', 'FSLR', 'SPWR', 'RUN', 'PLUG', 'BLDP', 'BE'],
                'category': 'energy'
            }
        }
    
    def parse_question(self, question: str) -> Dict[str, Any]:
        """Parse research question to extract groups, stocks, and analysis type"""
        
        question_lower = question.lower()
        
        # Extract predefined groups
        groups = []
        for group_name, group_info in self.predefined_groups.items():
            if group_name in question_lower or group_info['name'].lower() in question_lower:
                groups.append(group_name)
            # Check for category mentions
            if group_info['category'] in question_lower:
                groups.append(group_name)
        
        # Extract individual stocks
        stocks = []
        all_stocks = []
        for group_info in self.predefined_groups.values():
            all_stocks.extend(group_info['stocks'])
        
        # Remove duplicates and check for mentions
        all_stocks = list(set(all_stocks))
        for stock in all_stocks:
            if stock.lower() in question_lower:
                stocks.append(stock)
        
        # Extract custom stock lists (comma-separated)
        custom_stocks = self._extract_custom_stocks(question)
        if custom_stocks:
            stocks.extend(custom_stocks)
        
        # Extract analysis type
        analysis_type = self._classify_analysis_type(question_lower)
        
        # Extract comparison groups
        comparison_groups = self._extract_comparison_groups(question_lower)
        
        # Extract focus areas
        focus_areas = self._extract_focus_areas(question_lower)
        
        return {
            'original_question': question,
            'groups': groups,
            'stocks': stocks,
            'custom_stocks': custom_stocks,
            'analysis_type': analysis_type,
            'comparison_groups': comparison_groups,
            'focus_areas': focus_areas,
            'question_type': self._classify_question_type(question_lower)
        }
    
    def _extract_custom_stocks(self, question: str) -> List[str]:
        """Extract custom stock symbols from question"""
        # Look for patterns like "AAPL, MSFT, GOOGL" or "stocks: AAPL MSFT GOOGL"
        import re
        
        # Pattern 1: Comma-separated list
        comma_pattern = r'\b([A-Z]{1,5}(?:,|\s+[A-Z]{1,5}){2,})\b'
        matches = re.findall(comma_pattern, question.upper())
        
        stocks = []
        for match in matches:
            # Split by comma or space
            symbols = re.split(r'[,\s]+', match)
            stocks.extend([s.strip() for s in symbols if s.strip()])
        
        # Pattern 2: After keywords like "stocks:", "analyze:", etc.
        keyword_pattern = r'(?:stocks?|analyze|research|compare):\s*([A-Z\s,]+)'
        keyword_matches = re.findall(keyword_pattern, question.upper())
        
        for match in keyword_matches:
            symbols = re.split(r'[,\s]+', match)
            stocks.extend([s.strip() for s in symbols if s.strip()])
        
        return list(set(stocks))  # Remove duplicates
    
    def _classify_analysis_type(self, question_lower: str) -> str:
        """Classify the type of analysis requested"""
        if any(word in question_lower for word in ['compare', 'comparison', 'vs', 'versus', 'against']):
            return 'comparison'
        elif any(word in question_lower for word in ['correlation', 'correlated', 'relationship']):
            return 'correlation'
        elif any(word in question_lower for word in ['diversification', 'diversified', 'concentration']):
            return 'diversification'
        elif any(word in question_lower for word in ['risk', 'volatility', 'stability']):
            return 'risk_analysis'
        elif any(word in question_lower for word in ['performance', 'returns', 'gains', 'losses']):
            return 'performance'
        elif any(word in question_lower for word in ['portfolio', 'allocation', 'weight']):
            return 'portfolio'
        else:
            return 'general'
    
    def _extract_comparison_groups(self, question_lower: str) -> List[str]:
        """Extract groups mentioned for comparison"""
        comparison_groups = []
        
        # Look for comparison keywords
        comparison_keywords = ['vs', 'versus', 'against', 'compare', 'comparison']
        
        for keyword in comparison_keywords:
            if keyword in question_lower:
                # Find groups mentioned around the keyword
                for group_name, group_info in self.predefined_groups.items():
                    if group_info['name'].lower() in question_lower:
                        comparison_groups.append(group_name)
        
        return comparison_groups
    
    def _extract_focus_areas(self, question_lower: str) -> List[str]:
        """Extract specific focus areas from the question"""
        focus_areas = []
        
        if any(word in question_lower for word in ['correlation', 'correlated', 'relationship']):
            focus_areas.append('correlation')
        if any(word in question_lower for word in ['diversification', 'diversified', 'concentration']):
            focus_areas.append('diversification')
        if any(word in question_lower for word in ['risk', 'volatility', 'stability']):
            focus_areas.append('risk')
        if any(word in question_lower for word in ['performance', 'returns', 'gains']):
            focus_areas.append('performance')
        if any(word in question_lower for word in ['momentum', 'trend', 'technical']):
            focus_areas.append('momentum')
        if any(word in question_lower for word in ['valuation', 'value', 'cheap', 'expensive']):
            focus_areas.append('valuation')
        if any(word in question_lower for word in ['earnings', 'revenue', 'profit']):
            focus_areas.append('fundamentals')
        
        return focus_areas
    
    def _classify_question_type(self, question_lower: str) -> str:
        """Classify the overall question type"""
        if any(word in question_lower for word in ['research', 'analyze', 'analysis']):
            return 'research'
        elif any(word in question_lower for word in ['compare', 'comparison']):
            return 'comparison'
        elif any(word in question_lower for word in ['recommend', 'suggest', 'advice']):
            return 'recommendation'
        elif any(word in question_lower for word in ['explain', 'what', 'how', 'why']):
            return 'explanation'
        else:
            return 'general'
    
    async def fetch_group_data(self, group_name: str = None, stocks: List[str] = None, days: int = 30) -> Dict[str, Any]:
        """Fetch comprehensive data for a group of stocks"""
        
        if group_name and group_name in self.predefined_groups:
            group_info = self.predefined_groups[group_name]
            stocks = group_info['stocks']
            group_description = group_info['description']
        elif stocks:
            group_description = f"Custom group of {len(stocks)} stocks"
        else:
            return None
        
        # Limit stocks to avoid overwhelming the system
        stocks = stocks[:self.config['analysis']['max_stocks_per_group']]
        
        cache_key = f"group_{group_name or 'custom'}_{len(stocks)}_{days}"
        if cache_key in self.data_cache:
            logger.info(f"📊 Using cached group data for {group_name or 'custom group'}")
            return self.data_cache[cache_key]
        
        logger.info(f"🔍 Fetching group data for {len(stocks)} stocks...")
        
        try:
            # Fetch individual stock data
            stocks_data = {}
            for stock in stocks:
                stock_data = await self._fetch_asset_data(stock, days)
                if stock_data:
                    stocks_data[stock] = stock_data
            
            if not stocks_data:
                return None
            
            # Calculate group metrics
            group_metrics = self._calculate_group_metrics(stocks_data)
            
            # Calculate correlations
            correlation_matrix = self._calculate_correlation_matrix(stocks_data)
            
            # Calculate diversification metrics
            diversification_metrics = self._calculate_diversification_metrics(stocks_data, correlation_matrix)
            
            # Calculate risk metrics
            risk_metrics = self._calculate_risk_metrics(stocks_data)
            
            group_data = {
                'group_name': group_name or 'custom',
                'description': group_description,
                'stocks': stocks,
                'stocks_data': stocks_data,
                'metrics': group_metrics,
                'correlation_matrix': correlation_matrix,
                'diversification_metrics': diversification_metrics,
                'risk_metrics': risk_metrics,
                'timestamp': datetime.now().isoformat()
            }
            
            # Cache the data
            self.data_cache[cache_key] = group_data
            logger.info(f"✅ Successfully fetched group data for {len(stocks_data)} stocks")
            return group_data
            
        except Exception as e:
            logger.error(f"❌ Error fetching group data: {e}")
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
            
            # Calculate beta (simplified)
            beta = info.get('beta', 1.0)
            
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
                'beta': beta,
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
    
    def _calculate_group_metrics(self, stocks_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate comprehensive group metrics"""
        if not stocks_data:
            return {}
        
        # Extract metrics from all stocks
        returns = [data['price_change'] for data in stocks_data.values() if data]
        volatilities = [data['volatility'] for data in stocks_data.values() if data]
        rsis = [data['rsi'] for data in stocks_data.values() if data]
        betas = [data['beta'] for data in stocks_data.values() if data]
        
        # Calculate averages
        avg_return = np.mean(returns) if returns else 0
        avg_volatility = np.mean(volatilities) if volatilities else 0
        avg_rsi = np.mean(rsis) if rsis else 50
        avg_beta = np.mean(betas) if betas else 1.0
        
        # Calculate performance distribution
        positive_stocks = len([r for r in returns if r > 0])
        negative_stocks = len([r for r in returns if r < 0])
        
        # Calculate volatility distribution
        high_vol_stocks = len([v for v in volatilities if v > 0.3])
        low_vol_stocks = len([v for v in volatilities if v < 0.15])
        
        # Calculate RSI distribution
        overbought_stocks = len([rsi for rsi in rsis if rsi > 70])
        oversold_stocks = len([rsi for rsi in rsis if rsi < 30])
        
        return {
            'avg_return': avg_return,
            'avg_volatility': avg_volatility,
            'avg_rsi': avg_rsi,
            'avg_beta': avg_beta,
            'total_stocks': len(stocks_data),
            'positive_stocks': positive_stocks,
            'negative_stocks': negative_stocks,
            'high_vol_stocks': high_vol_stocks,
            'low_vol_stocks': low_vol_stocks,
            'overbought_stocks': overbought_stocks,
            'oversold_stocks': oversold_stocks,
            'performance_ratio': positive_stocks / len(stocks_data) if stocks_data else 0
        }
    
    def _calculate_correlation_matrix(self, stocks_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate correlation matrix for the group"""
        if len(stocks_data) < 2:
            return {}
        
        # Extract returns for correlation calculation
        returns_data = {}
        for symbol, data in stocks_data.items():
            if data and 'returns' in data:
                returns_data[symbol] = data['returns']
        
        if len(returns_data) < 2:
            return {}
        
        # Create DataFrame for correlation calculation
        df = pd.DataFrame(returns_data)
        correlation_matrix = df.corr()
        
        # Calculate average correlation
        # Get upper triangle of correlation matrix (excluding diagonal)
        upper_triangle = correlation_matrix.where(
            np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
        )
        
        # Calculate average correlation
        avg_correlation = upper_triangle.stack().mean()
        
        # Find highest correlations
        high_correlations = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_value = correlation_matrix.iloc[i, j]
                if abs(corr_value) > 0.7:  # High correlation threshold
                    high_correlations.append({
                        'stock1': correlation_matrix.columns[i],
                        'stock2': correlation_matrix.columns[j],
                        'correlation': corr_value
                    })
        
        return {
            'matrix': correlation_matrix.to_dict(),
            'avg_correlation': avg_correlation,
            'high_correlations': high_correlations
        }
    
    def _calculate_diversification_metrics(self, stocks_data: Dict[str, Any], correlation_matrix: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate diversification metrics for the group"""
        if not stocks_data or not correlation_matrix:
            return {}
        
        # Calculate Herfindahl-Hirschman Index (HHI) for concentration
        # This is a simplified version - in practice, you'd use market cap weights
        num_stocks = len(stocks_data)
        hhi = 1 / num_stocks  # Equal weight assumption
        
        # Calculate diversification ratio
        avg_correlation = correlation_matrix.get('avg_correlation', 0)
        diversification_ratio = 1 - abs(avg_correlation)
        
        # Calculate effective number of stocks (diversification measure)
        effective_stocks = 1 / hhi if hhi > 0 else num_stocks
        
        # Assess diversification quality
        if avg_correlation < 0.3:
            diversification_quality = 'Excellent'
        elif avg_correlation < 0.5:
            diversification_quality = 'Good'
        elif avg_correlation < 0.7:
            diversification_quality = 'Fair'
        else:
            diversification_quality = 'Poor'
        
        return {
            'hhi': hhi,
            'diversification_ratio': diversification_ratio,
            'effective_stocks': effective_stocks,
            'diversification_quality': diversification_quality,
            'avg_correlation': avg_correlation
        }
    
    def _calculate_risk_metrics(self, stocks_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate risk metrics for the group"""
        if not stocks_data:
            return {}
        
        # Extract returns for risk calculation
        all_returns = []
        for data in stocks_data.values():
            if data and 'returns' in data:
                all_returns.extend(data['returns'])
        
        if not all_returns:
            return {}
        
        # Calculate portfolio-level risk metrics
        portfolio_returns = np.array(all_returns)
        
        # Portfolio volatility (simplified - equal weights)
        portfolio_volatility = np.std(portfolio_returns) * (252 ** 0.5)
        
        # Portfolio VaR (Value at Risk) - 5% VaR
        var_95 = np.percentile(portfolio_returns, 5)
        
        # Portfolio CVaR (Conditional Value at Risk)
        cvar_95 = portfolio_returns[portfolio_returns <= var_95].mean()
        
        # Calculate individual stock risks
        individual_volatilities = [data['volatility'] for data in stocks_data.values() if data]
        max_volatility = max(individual_volatilities) if individual_volatilities else 0
        min_volatility = min(individual_volatilities) if individual_volatilities else 0
        
        # Risk concentration
        high_risk_stocks = len([v for v in individual_volatilities if v > 0.3])
        risk_concentration = high_risk_stocks / len(individual_volatilities) if individual_volatilities else 0
        
        return {
            'portfolio_volatility': portfolio_volatility,
            'var_95': var_95,
            'cvar_95': cvar_95,
            'max_individual_volatility': max_volatility,
            'min_individual_volatility': min_volatility,
            'high_risk_stocks': high_risk_stocks,
            'risk_concentration': risk_concentration,
            'risk_level': self._assess_risk_level(portfolio_volatility, risk_concentration)
        }
    
    def _assess_risk_level(self, portfolio_volatility: float, risk_concentration: float) -> str:
        """Assess overall risk level of the group"""
        if portfolio_volatility < 0.15 and risk_concentration < 0.3:
            return 'Low'
        elif portfolio_volatility < 0.25 and risk_concentration < 0.5:
            return 'Medium'
        elif portfolio_volatility < 0.35 and risk_concentration < 0.7:
            return 'High'
        else:
            return 'Very High'
    
    async def generate_group_analysis(self, parsed_question: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive group analysis using LLM"""
        
        if not self.llm_client:
            return {"error": "LLM not available"}
        
        # Fetch group data
        group_data = {}
        
        # Process predefined groups
        for group in parsed_question['groups']:
            data = await self.fetch_group_data(group_name=group, days=30)
            if data:
                group_data[group] = data
        
        # Process custom stocks
        if parsed_question['custom_stocks']:
            custom_data = await self.fetch_group_data(stocks=parsed_question['custom_stocks'], days=30)
            if custom_data:
                group_data['custom'] = custom_data
        
        if not group_data:
            return {"error": "No group data available"}
        
        # Prepare context for LLM
        context = self._prepare_group_context(group_data, parsed_question)
        
        # Generate LLM analysis
        prompt = self._create_group_analysis_prompt(parsed_question, context)
        
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
                'groups_analyzed': list(group_data.keys()),
                'group_data': group_data,
                'llm_analysis': analysis,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Error generating group analysis: {e}")
            return {"error": str(e)}
    
    def _prepare_group_context(self, group_data: Dict[str, Any], parsed_question: Dict[str, Any]) -> str:
        """Prepare context for LLM analysis"""
        context_parts = []
        
        for group_name, data in group_data.items():
            context_parts.append(f"\n=== {group_name.upper()} GROUP ===")
            context_parts.append(f"Description: {data['description']}")
            context_parts.append(f"Number of stocks: {data['metrics']['total_stocks']}")
            
            # Group metrics
            metrics = data['metrics']
            context_parts.append(f"Average Return: {metrics['avg_return']:+.2%}")
            context_parts.append(f"Average Volatility: {metrics['avg_volatility']:.1%}")
            context_parts.append(f"Performance Ratio: {metrics['performance_ratio']:.1%}")
            context_parts.append(f"Positive Stocks: {metrics['positive_stocks']}/{metrics['total_stocks']}")
            
            # Diversification metrics
            div_metrics = data['diversification_metrics']
            context_parts.append(f"Diversification Quality: {div_metrics['diversification_quality']}")
            context_parts.append(f"Average Correlation: {div_metrics['avg_correlation']:.3f}")
            
            # Risk metrics
            risk_metrics = data['risk_metrics']
            context_parts.append(f"Risk Level: {risk_metrics['risk_level']}")
            context_parts.append(f"Portfolio Volatility: {risk_metrics['portfolio_volatility']:.1%}")
            
            # Top performers
            top_stocks = sorted(data['stocks_data'].items(), 
                             key=lambda x: x[1]['price_change'], reverse=True)[:5]
            context_parts.append("Top Performers:")
            for stock, stock_data in top_stocks:
                context_parts.append(f"  {stock}: ${stock_data['current_price']:.2f} ({stock_data['price_change']:+.2%})")
            
            # High correlations
            high_corrs = data['correlation_matrix'].get('high_correlations', [])
            if high_corrs:
                context_parts.append("High Correlations:")
                for corr in high_corrs[:3]:  # Show top 3
                    context_parts.append(f"  {corr['stock1']} - {corr['stock2']}: {corr['correlation']:.3f}")
        
        return "\n".join(context_parts)
    
    def _create_group_analysis_prompt(self, parsed_question: Dict[str, Any], context: str) -> str:
        """Create prompt for LLM analysis"""
        
        question = parsed_question['original_question']
        analysis_type = parsed_question['analysis_type']
        focus_areas = parsed_question['focus_areas']
        
        prompt = f"""
You are an expert quantitative analyst specializing in group and portfolio analysis. Analyze the following stock group data and provide comprehensive insights.

QUESTION: {question}

ANALYSIS TYPE: {analysis_type}
FOCUS AREAS: {', '.join(focus_areas) if focus_areas else 'General analysis'}

GROUP DATA:
{context}

Please provide:

1. GROUP OVERVIEW:
   - Overall group performance and characteristics
   - Key strengths and weaknesses
   - Market positioning and outlook

2. CORRELATION ANALYSIS:
   - Inter-stock correlations and relationships
   - Diversification effectiveness
   - Concentration risks and opportunities

3. RISK ASSESSMENT:
   - Portfolio risk profile
   - Volatility analysis
   - Risk concentration and diversification

4. PERFORMANCE ANALYSIS:
   - Individual stock performance highlights
   - Group performance trends
   - Relative performance insights

5. PORTFOLIO RECOMMENDATIONS:
   - Optimal allocation strategies
   - Risk management considerations
   - Rebalancing recommendations

6. COMPARATIVE INSIGHTS:
   - Group comparison (if multiple groups)
   - Benchmark performance
   - Relative value analysis

Be specific, data-driven, and actionable. Focus on providing clear insights that can guide investment decisions and portfolio management.
"""
        
        return prompt
    
    def _parse_llm_response(self, response: str, parsed_question: Dict[str, Any]) -> Dict[str, Any]:
        """Parse LLM response into structured format"""
        return {
            'raw_response': response,
            'group_overview': self._extract_section(response, 'GROUP OVERVIEW'),
            'correlation_analysis': self._extract_section(response, 'CORRELATION ANALYSIS'),
            'risk_assessment': self._extract_section(response, 'RISK ASSESSMENT'),
            'performance_analysis': self._extract_section(response, 'PERFORMANCE ANALYSIS'),
            'portfolio_recommendations': self._extract_section(response, 'PORTFOLIO RECOMMENDATIONS'),
            'comparative_insights': self._extract_section(response, 'COMPARATIVE INSIGHTS')
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
                if any(keyword in line.upper() for keyword in ['GROUP OVERVIEW', 'CORRELATION ANALYSIS', 'RISK ASSESSMENT', 'PERFORMANCE ANALYSIS', 'PORTFOLIO RECOMMENDATIONS', 'COMPARATIVE INSIGHTS']):
                    break
            elif in_section:
                section_lines.append(line)
        
        return '\n'.join(section_lines).strip()
    
    async def ask_question(self, question: str) -> Dict[str, Any]:
        """Main interface for asking group analysis questions"""
        logger.info(f"🔍 Processing group analysis question: {question}")
        
        # Parse the question
        parsed_question = self.parse_question(question)
        
        # Generate analysis
        analysis = await self.generate_group_analysis(parsed_question)
        
        # Store in history
        self.research_history.append({
            'question': question,
            'parsed': parsed_question,
            'analysis': analysis,
            'timestamp': datetime.now().isoformat()
        })
        
        return analysis
    
    def format_response(self, analysis: Dict[str, Any]) -> str:
        """Format group analysis response for display"""
        if 'error' in analysis:
            return f"❌ Error: {analysis['error']}"
        
        output = []
        output.append(f"# Group Analysis")
        output.append(f"**Question:** {analysis['question']}")
        output.append(f"**Groups Analyzed:** {', '.join(analysis['groups_analyzed'])}")
        output.append(f"**Analysis Time:** {analysis['timestamp']}")
        output.append("")
        
        # LLM Analysis sections
        llm_analysis = analysis.get('llm_analysis', {})
        
        sections = [
            ('Group Overview', 'group_overview'),
            ('Correlation Analysis', 'correlation_analysis'),
            ('Risk Assessment', 'risk_assessment'),
            ('Performance Analysis', 'performance_analysis'),
            ('Portfolio Recommendations', 'portfolio_recommendations'),
            ('Comparative Insights', 'comparative_insights')
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
    """Example usage of the Group Analysis Chat interface"""
    
    # Initialize the chat interface
    chat = GroupAnalysisChat()
    
    # Example questions
    questions = [
        "Analyze the mega cap tech group performance",
        "Compare dividend aristocrats vs growth stocks",
        "Research AAPL, MSFT, GOOGL, AMZN, META for correlation analysis",
        "What's the risk profile of the fintech group?",
        "Analyze diversification in the biotech group"
    ]
    
    print("🤖 Group Analysis QuantEngine Chat")
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
