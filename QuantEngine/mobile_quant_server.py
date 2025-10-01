#!/usr/bin/env python3
"""
Mobile QuantEngine Server

A lightweight server that can run on mobile devices to provide
QuantEngine chat functionality without external dependencies.
"""

import sys
import json
import logging
import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path

# Add QuantEngine to path
sys.path.append(str(Path(__file__).parent))

try:
    from conversational_chat import ConversationalChat
    from llm_integration import OllamaLLM
    import yfinance as yf
    import pandas as pd
    import numpy as np
except ImportError as e:
    print(f"Warning: Some dependencies not available: {e}")
    print("Running in limited mode...")

from flask import Flask, request, jsonify
from flask_cors import CORS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

class MobileQuantEngine:
    """Mobile-optimized QuantEngine with limited dependencies"""
    
    def __init__(self):
        self.chat_interface = None
        self.llm = None
        self.initialized = False
        self._initialize()
    
    def _initialize(self):
        """Initialize the mobile QuantEngine"""
        try:
            # Try to initialize full QuantEngine
            self.chat_interface = ConversationalChat()
            self.llm = OllamaLLM()
            self.initialized = True
            logger.info("✅ Full QuantEngine initialized")
        except Exception as e:
            logger.warning(f"⚠️ Full QuantEngine not available: {e}")
            # Fallback to basic functionality
            self._initialize_basic()
    
    def _initialize_basic(self):
        """Initialize basic functionality without full QuantEngine"""
        try:
            self.llm = OllamaLLM()
            self.initialized = True
            logger.info("✅ Basic QuantEngine initialized")
        except Exception as e:
            logger.warning(f"⚠️ LLM not available: {e}")
            self.initialized = False
    
    def get_stock_data(self, ticker: str) -> Dict[str, Any]:
        """Get basic stock data using yfinance"""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            hist = stock.history(period="1mo")
            
            if hist.empty:
                return {"success": False, "error": "No data available"}
            
            current_price = info.get('regularMarketPrice', hist['Close'].iloc[-1])
            prev_close = info.get('previousClose', hist['Close'].iloc[-2] if len(hist) > 1 else current_price)
            change = current_price - prev_close
            change_pct = (change / prev_close) * 100 if prev_close != 0 else 0
            
            # Calculate basic technical indicators
            sma_20 = hist['Close'].rolling(20).mean().iloc[-1] if len(hist) >= 20 else current_price
            sma_50 = hist['Close'].rolling(50).mean().iloc[-1] if len(hist) >= 50 else current_price
            
            # Simple RSI calculation
            rsi = self._calculate_rsi(hist['Close'])
            
            return {
                "success": True,
                "ticker": ticker,
                "current_price": float(current_price),
                "change": float(change),
                "change_pct": float(change_pct),
                "sma_20": float(sma_20),
                "sma_50": float(sma_50),
                "rsi": float(rsi),
                "volume": int(hist['Volume'].iloc[-1]) if not hist.empty else 0,
                "high_52w": float(info.get('fiftyTwoWeekHigh', current_price)),
                "low_52w": float(info.get('fiftyTwoWeekLow', current_price)),
                "market_cap": info.get('marketCap', 0),
                "pe_ratio": info.get('trailingPE', 0),
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI"""
        try:
            if len(prices) < period + 1:
                return 50.0
            
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50.0
        except:
            return 50.0
    
    def analyze_stock(self, ticker: str) -> Dict[str, Any]:
        """Analyze a stock and provide trading insights"""
        data = self.get_stock_data(ticker)
        
        if not data["success"]:
            return data
        
        # Generate analysis
        analysis = self._generate_analysis(ticker, data)
        
        return {
            "success": True,
            "ticker": ticker,
            "analysis": analysis,
            "data": data
        }
    
    def _generate_analysis(self, ticker: str, data: Dict[str, Any]) -> str:
        """Generate trading analysis"""
        current_price = data["current_price"]
        change_pct = data["change_pct"]
        sma_20 = data["sma_20"]
        sma_50 = data["sma_50"]
        rsi = data["rsi"]
        
        # Determine trend
        if current_price > sma_20 > sma_50:
            trend = "BULLISH"
            trend_strength = "Strong uptrend"
        elif current_price < sma_20 < sma_50:
            trend = "BEARISH"
            trend_strength = "Strong downtrend"
        elif current_price > sma_20:
            trend = "MIXED"
            trend_strength = "Above 20-day SMA, below 50-day SMA"
        else:
            trend = "MIXED"
            trend_strength = "Below 20-day SMA, above 50-day SMA"
        
        # RSI analysis
        if rsi > 70:
            rsi_signal = "OVERBOUGHT - Consider selling"
        elif rsi < 30:
            rsi_signal = "OVERSOLD - Consider buying"
        else:
            rsi_signal = "NEUTRAL - RSI in normal range"
        
        # Generate signal
        if trend == "BULLISH" and rsi < 70:
            signal = "BUY"
            confidence = 75
        elif trend == "BEARISH" and rsi > 30:
            signal = "SELL"
            confidence = 75
        elif rsi < 30:
            signal = "BUY"
            confidence = 60
        elif rsi > 70:
            signal = "SELL"
            confidence = 60
        else:
            signal = "HOLD"
            confidence = 50
        
        analysis = f"""
📊 {ticker} ANALYSIS - {datetime.now().strftime('%Y-%m-%d %H:%M')}

💰 PRICE ACTION:
• Current Price: ${current_price:.2f}
• Daily Change: {change_pct:+.2f}%
• 20-day SMA: {sma_20:.2f}
• 50-day SMA: {sma_50:.2f}

📈 TECHNICAL INDICATORS:
• RSI: {rsi:.1f} ({rsi_signal})
• Trend: {trend} - {trend_strength}

🎯 TRADING SIGNAL:
• Recommendation: {signal}
• Confidence: {confidence}%

💡 KEY LEVELS:
• Support: {data.get('low_52w', current_price):.2f}
• Resistance: {data.get('high_52w', current_price):.2f}

⚠️ RISK FACTORS:
• Market volatility: {'High' if abs(change_pct) > 3 else 'Normal'}
• RSI divergence: {'Watch for reversal' if (trend == 'BULLISH' and rsi > 70) or (trend == 'BEARISH' and rsi < 30) else 'None'}
"""
        
        return analysis.strip()
    
    def chat(self, message: str, conversation_history: List[str] = None) -> Dict[str, Any]:
        """Handle chat messages"""
        if not self.initialized:
            return {
                "success": False,
                "error": "QuantEngine not initialized"
            }
        
        try:
            if self.chat_interface:
                # Use full QuantEngine chat
                response = self.chat_interface.chat(message)
                return {
                    "success": True,
                    "response": response,
                    "type": "full_analysis"
                }
            else:
                # Use basic LLM response
                if self.llm:
                    # Check if asking about specific stock
                    ticker = None
                    for t in ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'TSLA', 'AMZN', 'META', 'NFLX', 'AMD', 'INTC']:
                        if t.lower() in message.lower():
                            ticker = t
                            break
                    
                    if ticker:
                        # Analyze the stock
                        analysis = self.analyze_stock(ticker)
                        if analysis["success"]:
                            return {
                                "success": True,
                                "response": analysis["analysis"],
                                "type": "stock_analysis"
                            }
                    
                    # General chat response
                    prompt = f"""
You are a helpful trading assistant. The user asked: "{message}"

Provide a helpful response about trading, stocks, or market analysis. Be concise and actionable.
"""
                    response = self.llm.generate(prompt, max_tokens=200, temperature=0.7)
                    return {
                        "success": True,
                        "response": response,
                        "type": "general_chat"
                    }
                else:
                    return {
                        "success": False,
                        "error": "No AI capabilities available"
                    }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

# Initialize the mobile QuantEngine
mobile_quant = MobileQuantEngine()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "initialized": mobile_quant.initialized,
        "timestamp": datetime.now().isoformat()
    })

@app.route('/quant-chat', methods=['POST'])
def quant_chat():
    """Main chat endpoint"""
    try:
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({"error": "Message required"}), 400
        
        message = data['message']
        conversation_history = data.get('conversation_history', [])
        
        logger.info(f"Chat request: {message[:50]}...")
        
        result = mobile_quant.chat(message, conversation_history)
        
        if result["success"]:
            return jsonify(result)
        else:
            return jsonify(result), 500
            
    except Exception as e:
        logger.error(f"Chat error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/analyze-stock/<ticker>', methods=['GET'])
def analyze_stock(ticker):
    """Analyze a specific stock"""
    try:
        result = mobile_quant.analyze_stock(ticker.upper())
        return jsonify(result)
    except Exception as e:
        logger.error(f"Stock analysis error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/stock-data/<ticker>', methods=['GET'])
def get_stock_data(ticker):
    """Get basic stock data"""
    try:
        result = mobile_quant.get_stock_data(ticker.upper())
        return jsonify(result)
    except Exception as e:
        logger.error(f"Stock data error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("🚀 Starting Mobile QuantEngine Server...")
    print(f"✅ QuantEngine initialized: {mobile_quant.initialized}")
    print("📱 Mobile-optimized for self-contained operation")
    print("🌐 Server running on http://localhost:5000")
    
    app.run(host='0.0.0.0', port=5000, debug=True)

