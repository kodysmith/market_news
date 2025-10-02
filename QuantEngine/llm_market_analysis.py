#!/usr/bin/env python3
"""
LLM-Powered Market Analysis
Uses local Ollama models for sophisticated market analysis
"""

import os
import sys
import json
import logging
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import yfinance as yf
import requests
from pathlib import Path

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    print("⚠️ Ollama not available")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/llm_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class LLMMarketAnalysis:
    """LLM-powered market analysis using local Ollama models"""
    
    def __init__(self, model: str = "mistral:7b-v2", host: str = "http://localhost:11434"):
        self.model = model
        self.host = host
        self.ollama_client = self._setup_ollama()
        
        # Analysis cache
        self.analysis_cache = {}
        
        logger.info(f"🤖 LLM Market Analysis initialized with model: {model}")
    
    def _setup_ollama(self) -> Optional[Any]:
        """Setup Ollama client"""
        if not OLLAMA_AVAILABLE:
            logger.warning("Ollama not available")
            return None
        
        try:
            # Test connection
            models = ollama.list()
            logger.info(f"✅ Ollama connected, models: {[m['name'] for m in models['models']]}")
            return ollama
        except Exception as e:
            logger.error(f"Failed to connect to Ollama: {e}")
            return None
    
    async def analyze_market_sentiment(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze market sentiment using LLM"""
        if not self.ollama_client:
            return {"error": "Ollama not available"}
        
        try:
            # Prepare market context
            context = self._prepare_market_context(market_data)
            
            prompt = f"""
You are an expert quantitative analyst. Analyze the current market sentiment based on the following data:

MARKET DATA:
{context}

Provide a comprehensive market sentiment analysis including:
1. Overall market sentiment (BULLISH/BEARISH/NEUTRAL) with confidence level (0-100%)
2. Key factors driving the sentiment
3. Risk factors to watch
4. Short-term (1-3 days) outlook
5. Medium-term (1-2 weeks) outlook
6. Key levels to watch (support/resistance)
7. Sector rotation insights
8. Volatility expectations

Be specific, data-driven, and actionable. Focus on high-probability scenarios.
"""
            
            response = self.ollama_client.generate(
                model=self.model,
                prompt=prompt,
                options={
                    'temperature': 0.3,
                    'num_predict': 1024,
                    'top_p': 0.9,
                    'top_k': 40
                }
            )
            
            analysis = response['response']
            sentiment_data = self._parse_sentiment_analysis(analysis)
            
            return {
                "analysis": analysis,
                "sentiment": sentiment_data,
                "timestamp": datetime.now().isoformat(),
                "model": self.model
            }
            
        except Exception as e:
            logger.error(f"Market sentiment analysis failed: {e}")
            return {"error": str(e)}
    
    async def analyze_stock_opportunity(self, ticker: str, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze individual stock opportunity using LLM"""
        if not self.ollama_client:
            return {"error": "Ollama not available"}
        
        try:
            # Get detailed stock data
            stock_data = await self._get_detailed_stock_data(ticker)
            
            prompt = f"""
You are an expert stock analyst. Analyze {ticker} for trading opportunities:

STOCK DATA:
{stock_data}

Provide a comprehensive analysis including:
1. Trading signal (STRONG_BUY/BUY/HOLD/SELL/STRONG_SELL) with confidence (0-100%)
2. Entry price and reasoning
3. Target price and stop-loss levels
4. Risk/reward ratio
5. Time horizon for the trade
6. Key technical levels to watch
7. Fundamental factors affecting the stock
8. Market conditions that could impact the trade
9. Alternative scenarios and probabilities
10. Position sizing recommendations

Be specific about price levels, timeframes, and risk management.
"""
            
            response = self.ollama_client.generate(
                model=self.model,
                prompt=prompt,
                options={
                    'temperature': 0.2,
                    'num_predict': 1024,
                    'top_p': 0.8,
                    'top_k': 30
                }
            )
            
            analysis = response['response']
            opportunity_data = self._parse_opportunity_analysis(analysis, ticker)
            
            return {
                "ticker": ticker,
                "analysis": analysis,
                "opportunity": opportunity_data,
                "timestamp": datetime.now().isoformat(),
                "model": self.model
            }
            
        except Exception as e:
            logger.error(f"Stock opportunity analysis failed for {ticker}: {e}")
            return {"error": str(e)}
    
    async def analyze_sector_rotation(self, sectors: List[str]) -> Dict[str, Any]:
        """Analyze sector rotation using LLM"""
        if not self.ollama_client:
            return {"error": "Ollama not available"}
        
        try:
            # Get sector data
            sector_data = await self._get_sector_data(sectors)
            
            prompt = f"""
You are an expert sector analyst. Analyze the current sector rotation:

SECTOR DATA:
{sector_data}

Provide a comprehensive sector analysis including:
1. Leading sectors (outperforming)
2. Lagging sectors (underperforming)
3. Sector rotation trends
4. Risk-on vs risk-off sentiment
5. Economic cycle positioning
6. Sector-specific catalysts
7. Relative strength analysis
8. Sector allocation recommendations
9. Risk factors by sector
10. Time horizon for sector trends

Focus on actionable insights for portfolio allocation.
"""
            
            response = self.ollama_client.generate(
                model=self.model,
                prompt=prompt,
                options={
                    'temperature': 0.4,
                    'num_predict': 1024,
                    'top_p': 0.9,
                    'top_k': 40
                }
            )
            
            analysis = response['response']
            rotation_data = self._parse_sector_rotation(analysis)
            
            return {
                "analysis": analysis,
                "rotation": rotation_data,
                "timestamp": datetime.now().isoformat(),
                "model": self.model
            }
            
        except Exception as e:
            logger.error(f"Sector rotation analysis failed: {e}")
            return {"error": str(e)}
    
    def _prepare_market_context(self, market_data: Dict[str, Any]) -> str:
        """Prepare market context for LLM analysis"""
        context = []
        
        # Add major indices
        for ticker, data in market_data.items():
            if isinstance(data, dict) and 'price' in data:
                context.append(f"{ticker}: ${data['price']:.2f} ({data.get('change_pct', 0):+.2f}%)")
        
        # Add technical indicators
        if 'technical_indicators' in market_data:
            tech = market_data['technical_indicators']
            context.append(f"RSI: {tech.get('rsi', 'N/A')}")
            context.append(f"MACD: {tech.get('macd', 'N/A')}")
            context.append(f"Volume: {tech.get('volume_ratio', 'N/A')}")
        
        # Add market sentiment
        if 'sentiment' in market_data:
            context.append(f"Market Sentiment: {market_data['sentiment']}")
        
        return "\n".join(context)
    
    async def _get_detailed_stock_data(self, ticker: str) -> str:
        """Get detailed stock data for analysis"""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            hist = stock.history(period="3mo")
            
            if hist.empty:
                return f"No data available for {ticker}"
            
            # Calculate technical indicators
            rsi = self._calculate_rsi(hist['Close'])
            sma_20 = hist['Close'].rolling(20).mean().iloc[-1]
            sma_50 = hist['Close'].rolling(50).mean().iloc[-1]
            macd = self._calculate_macd(hist['Close'])
            
            # Get recent price action
            current_price = hist['Close'].iloc[-1]
            prev_close = hist['Close'].iloc[-2]
            change = current_price - prev_close
            change_pct = (change / prev_close) * 100
            
            # Get volume analysis
            avg_volume = hist['Volume'].rolling(20).mean().iloc[-1]
            current_volume = hist['Volume'].iloc[-1]
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
            
            data = f"""
TICKER: {ticker}
CURRENT PRICE: ${current_price:.2f}
CHANGE: {change:+.2f} ({change_pct:+.2f}%)
VOLUME: {current_volume:,} (avg: {avg_volume:,.0f}, ratio: {volume_ratio:.2f})
MARKET CAP: ${info.get('marketCap', 0):,}
PE RATIO: {info.get('trailingPE', 'N/A')}
RSI (14): {rsi:.1f}
SMA 20: ${sma_20:.2f}
SMA 50: ${sma_50:.2f}
MACD: {macd:.4f}
52W HIGH: ${info.get('fiftyTwoWeekHigh', 'N/A')}
52W LOW: ${info.get('fiftyTwoWeekLow', 'N/A')}
SECTOR: {info.get('sector', 'N/A')}
INDUSTRY: {info.get('industry', 'N/A')}
"""
            
            return data
            
        except Exception as e:
            logger.error(f"Failed to get detailed data for {ticker}: {e}")
            return f"Error fetching data for {ticker}: {e}"
    
    async def _get_sector_data(self, sectors: List[str]) -> str:
        """Get sector data for analysis"""
        sector_data = []
        
        for sector in sectors:
            try:
                # Use sector ETFs as proxies
                etf_map = {
                    'technology': 'XLK',
                    'healthcare': 'XLV',
                    'financials': 'XLF',
                    'consumer_discretionary': 'XLY',
                    'consumer_staples': 'XLP',
                    'energy': 'XLE',
                    'industrials': 'XLI',
                    'materials': 'XLB',
                    'utilities': 'XLU',
                    'real_estate': 'XLRE',
                    'communication': 'XLC'
                }
                
                etf = etf_map.get(sector.lower(), sector)
                stock = yf.Ticker(etf)
                info = stock.info
                hist = stock.history(period="1mo")
                
                if not hist.empty:
                    current_price = hist['Close'].iloc[-1]
                    change_pct = ((current_price - hist['Close'].iloc[-2]) / hist['Close'].iloc[-2]) * 100
                    
                    sector_data.append(f"{sector.upper()}: ${current_price:.2f} ({change_pct:+.2f}%)")
                
            except Exception as e:
                logger.warning(f"Failed to get data for {sector}: {e}")
        
        return "\n".join(sector_data)
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI"""
        deltas = prices.diff()
        gains = deltas.where(deltas > 0, 0)
        losses = -deltas.where(deltas < 0, 0)
        
        avg_gains = gains.rolling(period).mean().iloc[-1]
        avg_losses = losses.rolling(period).mean().iloc[-1]
        
        if avg_losses == 0:
            return 100.0
        
        rs = avg_gains / avg_losses
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, prices: pd.Series) -> float:
        """Calculate MACD"""
        ema_12 = prices.ewm(span=12).mean()
        ema_26 = prices.ewm(span=26).mean()
        macd = ema_12 - ema_26
        return macd.iloc[-1]
    
    def _parse_sentiment_analysis(self, analysis: str) -> Dict[str, Any]:
        """Parse sentiment analysis from LLM response"""
        import re
        
        # Extract sentiment
        sentiment_match = re.search(r'(BULLISH|BEARISH|NEUTRAL)', analysis, re.IGNORECASE)
        sentiment = sentiment_match.group(1).upper() if sentiment_match else "NEUTRAL"
        
        # Extract confidence
        confidence_match = re.search(r'confidence[:\s]+(\d+)%', analysis, re.IGNORECASE)
        confidence = float(confidence_match.group(1)) / 100.0 if confidence_match else 0.5
        
        return {
            "sentiment": sentiment,
            "confidence": confidence,
            "analysis_length": len(analysis)
        }
    
    def _parse_opportunity_analysis(self, analysis: str, ticker: str) -> Dict[str, Any]:
        """Parse opportunity analysis from LLM response"""
        import re
        
        # Extract signal
        signal_match = re.search(r'(STRONG_BUY|BUY|HOLD|SELL|STRONG_SELL)', analysis, re.IGNORECASE)
        signal = signal_match.group(1).upper() if signal_match else "HOLD"
        
        # Extract confidence
        confidence_match = re.search(r'confidence[:\s]+(\d+)%', analysis, re.IGNORECASE)
        confidence = float(confidence_match.group(1)) / 100.0 if confidence_match else 0.5
        
        # Extract price levels
        price_matches = re.findall(r'\$(\d+\.?\d*)', analysis)
        prices = [float(p) for p in price_matches if float(p) > 0]
        
        return {
            "ticker": ticker,
            "signal": signal,
            "confidence": confidence,
            "price_levels": prices,
            "analysis_length": len(analysis)
        }
    
    def _parse_sector_rotation(self, analysis: str) -> Dict[str, Any]:
        """Parse sector rotation analysis from LLM response"""
        import re
        
        # Extract leading sectors
        leading_match = re.search(r'leading sectors[:\s]+([^.]+)', analysis, re.IGNORECASE)
        leading = leading_match.group(1).strip() if leading_match else "Not specified"
        
        # Extract lagging sectors
        lagging_match = re.search(r'lagging sectors[:\s]+([^.]+)', analysis, re.IGNORECASE)
        lagging = lagging_match.group(1).strip() if lagging_match else "Not specified"
        
        return {
            "leading_sectors": leading,
            "lagging_sectors": lagging,
            "analysis_length": len(analysis)
        }
    
    async def run_comprehensive_analysis(self, tickers: List[str] = None, sectors: List[str] = None):
        """Run comprehensive market analysis"""
        if tickers is None:
            tickers = ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'TSLA']
        
        if sectors is None:
            sectors = ['technology', 'healthcare', 'financials', 'energy', 'consumer_discretionary']
        
        logger.info(f"🤖 Starting comprehensive LLM market analysis")
        
        try:
            # Get market data
            market_data = await self._get_market_data(tickers)
            
            # Run analyses
            sentiment_analysis = await self.analyze_market_sentiment(market_data)
            sector_analysis = await self.analyze_sector_rotation(sectors)
            
            # Analyze individual stocks
            stock_analyses = []
            for ticker in tickers:
                analysis = await self.analyze_stock_opportunity(ticker, market_data)
                stock_analyses.append(analysis)
            
            # Compile results
            results = {
                "timestamp": datetime.now().isoformat(),
                "model": self.model,
                "market_sentiment": sentiment_analysis,
                "sector_rotation": sector_analysis,
                "stock_opportunities": stock_analyses,
                "tickers_analyzed": tickers,
                "sectors_analyzed": sectors
            }
            
            # Save results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = f"reports/llm_analysis_{timestamp}.json"
            
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            logger.info(f"📊 LLM analysis complete - results saved to {results_file}")
            
            # Print summary
            print("\n" + "="*60)
            print("🤖 LLM MARKET ANALYSIS RESULTS")
            print("="*60)
            print(f"📅 Timestamp: {results['timestamp']}")
            print(f"🤖 Model: {results['model']}")
            print(f"📊 Tickers Analyzed: {len(tickers)}")
            print(f"🏢 Sectors Analyzed: {len(sectors)}")
            
            if sentiment_analysis and 'sentiment' in sentiment_analysis:
                sentiment = sentiment_analysis['sentiment']
                print(f"📈 Market Sentiment: {sentiment['sentiment']} ({sentiment['confidence']:.1%})")
            
            print("\n🎯 STOCK OPPORTUNITIES:")
            for analysis in stock_analyses:
                if 'opportunity' in analysis:
                    opp = analysis['opportunity']
                    print(f"  {opp['ticker']}: {opp['signal']} ({opp['confidence']:.1%})")
            
            return results
            
        except Exception as e:
            logger.error(f"Comprehensive analysis failed: {e}")
            raise
    
    async def _get_market_data(self, tickers: List[str]) -> Dict[str, Any]:
        """Get market data for analysis"""
        market_data = {}
        
        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                info = stock.info
                hist = stock.history(period="1mo")
                
                if not hist.empty:
                    current_price = hist['Close'].iloc[-1]
                    change_pct = ((current_price - hist['Close'].iloc[-2]) / hist['Close'].iloc[-2]) * 100
                    
                    market_data[ticker] = {
                        'price': current_price,
                        'change_pct': change_pct,
                        'volume': info.get('volume', 0),
                        'market_cap': info.get('marketCap', 0)
                    }
                    
            except Exception as e:
                logger.warning(f"Failed to get data for {ticker}: {e}")
        
        return market_data

async def main():
    """Main function"""
    # Create LLM analyzer
    analyzer = LLMMarketAnalysis()
    
    # Define analysis parameters
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'TSLA', 'AMZN', 'META', 'NFLX']
    sectors = ['technology', 'healthcare', 'financials', 'energy', 'consumer_discretionary']
    
    # Run comprehensive analysis
    await analyzer.run_comprehensive_analysis(tickers, sectors)

if __name__ == "__main__":
    asyncio.run(main())
