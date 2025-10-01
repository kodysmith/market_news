#!/usr/bin/env python3
"""
LLM Integration for QuantEngine using Ollama
"""

import requests
import json
import logging
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class OllamaLLM:
    """Local LLM integration using Ollama"""
    
    def __init__(self, model: str = "llama3.2:latest", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url
        self.api_url = f"{base_url}/api/generate"
        
    def generate(self, prompt: str, max_tokens: int = 1000, temperature: float = 0.7) -> str:
        """Generate text using the local LLM"""
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens
                }
            }
            
            response = requests.post(self.api_url, json=payload, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            return result.get('response', '').strip()
            
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return ""
    
    def _get_real_time_price(self, ticker: str) -> Optional[float]:
        """Get real-time price for context"""
        try:
            import yfinance as yf
            
            # Get real-time quote
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # Try to get current price
            current_price = info.get('regularMarketPrice') or info.get('previousClose')
            
            if current_price:
                return float(current_price)
            else:
                # Fallback to previous close
                return info.get('previousClose')
                
        except Exception as e:
            logger.debug(f"Could not get real-time price for {ticker}: {e}")
            return None
    
    def analyze_market_data(self, ticker: str, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze market data using LLM"""
        
        # Prepare data summary
        current_price = float(data['close'].iloc[-1])
        price_change = float(data['close'].pct_change().iloc[-1]) * 100
        volatility = float(data['close'].pct_change().rolling(20).std().iloc[-1]) * np.sqrt(252) * 100
        
        # Calculate RSI
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        rsi_value = rsi.iloc[-1]
        if hasattr(rsi_value, 'iloc'):
            rsi_value = rsi_value.iloc[0]
        current_rsi = float(rsi_value) if not pd.isna(rsi_value) else 50
        
        # Calculate moving averages
        sma_20 = float(data['close'].rolling(20).mean().iloc[-1])
        sma_50 = float(data['close'].rolling(50).mean().iloc[-1])
        
        # Get real-time price for context
        real_time_price = self._get_real_time_price(ticker)
        price_context = ""
        if real_time_price and abs(real_time_price - current_price) > 0.01:
            price_context = f"\n⚠️ REAL-TIME UPDATE: {ticker} is currently trading at ${real_time_price:.2f} (vs historical ${current_price:.2f})"
            if real_time_price > current_price:
                price_context += f" - UP ${real_time_price - current_price:.2f} from historical data"
            else:
                price_context += f" - DOWN ${current_price - real_time_price:.2f} from historical data"

        prompt = f"""
You are analyzing {ticker} stock specifically. Provide a detailed trading analysis for {ticker}:

{ticker} Current Data:
- Current Price: ${current_price:.2f}
- Daily Change: {price_change:+.2f}%
- Volatility: {volatility:.1f}%
- RSI: {current_rsi:.1f}
- 20-day SMA: ${sma_20:.2f}
- 50-day SMA: ${sma_50:.2f}{price_context}

IMPORTANT CONTEXT:
- This data is from the most recent market close
- Consider if {ticker} has broken through key support/resistance levels
- Check if {ticker} is approaching or crossing the 20-day SMA (${sma_20:.2f})
- Monitor if {ticker} is near the 50-day SMA (${sma_50:.2f})
- RSI of {current_rsi:.1f} indicates {'overbought' if current_rsi > 70 else 'oversold' if current_rsi < 30 else 'neutral'} conditions

Provide a specific analysis for {ticker} including:
1. Market sentiment for {ticker} (bullish/bearish/neutral)
2. Key technical levels for {ticker} to watch (support/resistance)
3. Trading recommendation for {ticker} (BUY/SELL/HOLD) with specific reasoning
4. Risk factors specific to {ticker}
5. Current price action analysis - is {ticker} breaking through key levels?

Always refer to the stock as {ticker} in your analysis. Be specific and actionable.
"""
        
        response = self.generate(prompt, max_tokens=500, temperature=0.3)
        
        return {
            'ticker': ticker,
            'llm_analysis': response,
            'technical_data': {
                'current_price': current_price,
                'price_change': price_change,
                'volatility': volatility,
                'rsi': current_rsi,
                'sma_20': sma_20,
                'sma_50': sma_50
            }
        }
    
    def generate_strategy_insights(self, market_regime: str, opportunities: List[Dict]) -> str:
        """Generate strategy insights using LLM"""
        
        opportunities_text = "\n".join([
            f"- {opp['ticker']}: {opp['signal']} (Confidence: {opp['confidence']:.1%}, Reason: {opp['reason']})"
            for opp in opportunities
        ])
        
        prompt = f"""
As a quantitative trading expert, analyze these specific market opportunities:

Market Regime: {market_regime}
Trading Opportunities:
{opportunities_text}

Provide specific analysis including:
1. Overall market assessment for these opportunities
2. Best opportunities to focus on and why (mention specific tickers)
3. Risk management recommendations for each opportunity
4. Market conditions that could change the outlook

Be specific about which tickers to focus on and why. Keep response practical and actionable for day trading.
"""
        
        return self.generate(prompt, max_tokens=600, temperature=0.4)
    
    def enhance_daily_report(self, report_data: Dict[str, Any]) -> str:
        """Enhance daily report with LLM insights"""
        
        prompt = f"""
Enhance this trading report with expert analysis:

Market Summary: {report_data.get('market_summary', '')}
Top Opportunities: {len(report_data.get('opportunities', []))} opportunities found
Market Regime: {report_data.get('regime', 'unknown')}

Add:
1. Market commentary explaining current conditions
2. Key risks and opportunities
3. Trading strategy recommendations
4. Market outlook for the next few days

Make it professional and actionable for traders.
"""
        
        return self.generate(prompt, max_tokens=800, temperature=0.5)

def test_ollama_connection():
    """Test if Ollama is running and accessible"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get('models', [])
            logger.info(f"✅ Ollama connected. Available models: {[m['name'] for m in models]}")
            return True
        else:
            logger.error("❌ Ollama not responding")
            return False
    except Exception as e:
        logger.error(f"❌ Cannot connect to Ollama: {e}")
        return False

if __name__ == "__main__":
    # Test the LLM integration
    llm = OllamaLLM()
    
    if test_ollama_connection():
        # Test with sample data
        import pandas as pd
        import numpy as np
        
        # Create sample data
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        prices = 100 + np.cumsum(np.random.randn(100) * 0.02)
        
        sample_data = pd.DataFrame({
            'close': prices,
            'high': prices * 1.01,
            'low': prices * 0.99,
            'volume': np.random.randint(1000000, 5000000, 100)
        }, index=dates)
        
        # Test analysis
        result = llm.analyze_market_data("TEST", sample_data)
        print("LLM Analysis Result:")
        print(result['llm_analysis'])
    else:
        print("Please start Ollama first: ollama serve")
