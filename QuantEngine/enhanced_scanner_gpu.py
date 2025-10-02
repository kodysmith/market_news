#!/usr/bin/env python3
"""
Enhanced Market Scanner with GPU and LLM Support
Leverages NVIDIA GPU and Ollama for advanced market analysis
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
import requests
import yfinance as yf
from pathlib import Path

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# GPU and LLM imports
try:
    import torch
    GPU_AVAILABLE = torch.cuda.is_available()
    if GPU_AVAILABLE:
        print(f"🎮 GPU Available: {torch.cuda.get_device_name(0)}")
        print(f"🎮 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
except ImportError:
    GPU_AVAILABLE = False
    print("⚠️ PyTorch not available - using CPU")

try:
    import ollama
    OLLAMA_AVAILABLE = True
    print("🤖 Ollama available")
except ImportError:
    OLLAMA_AVAILABLE = False
    print("⚠️ Ollama not available")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/enhanced_scanner_gpu.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class EnhancedScannerGPU:
    """Enhanced Market Scanner with GPU and LLM capabilities"""
    
    def __init__(self, config_path: str = "config/gpu_config.json"):
        self.config = self._load_config(config_path)
        self.device = self._setup_device()
        self.llm_client = self._setup_llm()
        
        # Market data cache
        self.data_cache = {}
        self.analysis_cache = {}
        
        logger.info("🚀 Enhanced Scanner GPU initialized")
        logger.info(f"🎮 GPU Enabled: {self.config['gpu']['enabled']}")
        logger.info(f"🤖 LLM Enabled: {self.config['llm']['enabled']}")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration"""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found, using defaults")
            return {
                "gpu": {"enabled": GPU_AVAILABLE, "device": "cuda:0"},
                "llm": {"enabled": OLLAMA_AVAILABLE, "model": "llama3.2:latest"},
                "analysis": {"confidence_threshold": 0.7}
            }
    
    def _setup_device(self) -> str:
        """Setup GPU device"""
        if self.config['gpu']['enabled'] and GPU_AVAILABLE:
            device = torch.device("cuda:0")
            logger.info(f"🎮 Using GPU: {torch.cuda.get_device_name(0)}")
            return device
        else:
            device = torch.device("cpu")
            logger.info("💻 Using CPU")
            return device
    
    def _setup_llm(self) -> Optional[Any]:
        """Setup LLM client"""
        if self.config['llm']['enabled'] and OLLAMA_AVAILABLE:
            try:
                # Test Ollama connection
                models = ollama.list()
                logger.info(f"🤖 Ollama models available: {[m['name'] for m in models['models']]}")
                return ollama
            except Exception as e:
                logger.warning(f"Failed to connect to Ollama: {e}")
                return None
        return None
    
    async def analyze_market_with_ai(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze market data using local LLM"""
        if not self.llm_client:
            return {"ai_analysis": "LLM not available", "confidence": 0.0}
        
        try:
            # Prepare market context
            context = self._prepare_market_context(market_data)
            
            # Create analysis prompt
            prompt = f"""
You are an expert quantitative analyst with access to real-time market data. 
Analyze the following market conditions and provide actionable insights:

MARKET DATA:
{context}

Please provide:
1. Market sentiment (BULLISH/BEARISH/NEUTRAL) with confidence level
2. Key technical levels to watch
3. Risk factors and opportunities
4. Trading recommendations for the next 1-3 days
5. Macro economic implications

Be specific, data-driven, and actionable. Focus on high-probability setups.
"""
            
            # Get LLM analysis
            response = self.llm_client.generate(
                model=self.config['llm']['model'],
                prompt=prompt,
                options={
                    'temperature': self.config['llm']['temperature'],
                    'num_predict': self.config['llm']['max_tokens']
                }
            )
            
            # Parse response
            ai_analysis = response['response']
            confidence = self._extract_confidence(ai_analysis)
            
            return {
                "ai_analysis": ai_analysis,
                "confidence": confidence,
                "timestamp": datetime.now().isoformat(),
                "model": self.config['llm']['model']
            }
            
        except Exception as e:
            logger.error(f"LLM analysis failed: {e}")
            return {"ai_analysis": f"Analysis failed: {e}", "confidence": 0.0}
    
    def _prepare_market_context(self, market_data: Dict[str, Any]) -> str:
        """Prepare market context for LLM analysis"""
        context = []
        
        # Add major indices
        for ticker, data in market_data.items():
            if isinstance(data, dict) and 'price' in data:
                context.append(f"{ticker}: ${data['price']:.2f} ({data.get('change', 0):+.2f}%)")
        
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
    
    def _extract_confidence(self, analysis: str) -> float:
        """Extract confidence level from LLM analysis"""
        try:
            # Look for confidence patterns in the text
            import re
            confidence_patterns = [
                r'confidence[:\s]+(\d+)%',
                r'(\d+)%\s+confidence',
                r'confidence[:\s]+(\d+\.\d+)',
                r'(\d+\.\d+)\s+confidence'
            ]
            
            for pattern in confidence_patterns:
                match = re.search(pattern, analysis.lower())
                if match:
                    return float(match.group(1)) / 100.0
            
            # Default confidence based on analysis length and keywords
            if len(analysis) > 500:
                return 0.8
            elif len(analysis) > 200:
                return 0.6
            else:
                return 0.4
                
        except Exception:
            return 0.5
    
    async def scan_market_with_gpu(self, tickers: List[str]) -> Dict[str, Any]:
        """Scan market using GPU acceleration"""
        logger.info(f"🔍 Scanning {len(tickers)} tickers with GPU acceleration")
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "tickers_scanned": len(tickers),
            "opportunities": [],
            "market_sentiment": {},
            "ai_analysis": {},
            "gpu_used": GPU_AVAILABLE,
            "llm_used": OLLAMA_AVAILABLE
        }
        
        # Fetch market data
        market_data = await self._fetch_market_data_batch(tickers)
        
        # GPU-accelerated technical analysis
        if GPU_AVAILABLE:
            technical_analysis = await self._gpu_technical_analysis(market_data)
        else:
            technical_analysis = await self._cpu_technical_analysis(market_data)
        
        # Find trading opportunities
        opportunities = await self._find_opportunities(market_data, technical_analysis)
        results["opportunities"] = opportunities
        
        # AI-powered market analysis
        if self.llm_client:
            ai_analysis = await self.analyze_market_with_ai(market_data)
            results["ai_analysis"] = ai_analysis
        
        # Calculate market sentiment
        results["market_sentiment"] = self._calculate_market_sentiment(opportunities)
        
        return results
    
    async def _fetch_market_data_batch(self, tickers: List[str]) -> Dict[str, Any]:
        """Fetch market data for multiple tickers"""
        market_data = {}
        
        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                info = stock.info
                hist = stock.history(period="1mo")
                
                if not hist.empty:
                    current_price = info.get('regularMarketPrice', hist['Close'].iloc[-1])
                    prev_close = info.get('previousClose', hist['Close'].iloc[-2])
                    change = current_price - prev_close
                    change_pct = (change / prev_close) * 100
                    
                    market_data[ticker] = {
                        'price': current_price,
                        'change': change,
                        'change_pct': change_pct,
                        'volume': info.get('volume', 0),
                        'market_cap': info.get('marketCap', 0),
                        'pe_ratio': info.get('trailingPE', 0),
                        'history': hist
                    }
                    
            except Exception as e:
                logger.warning(f"Failed to fetch data for {ticker}: {e}")
        
        return market_data
    
    async def _gpu_technical_analysis(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """GPU-accelerated technical analysis"""
        logger.info("🎮 Running GPU-accelerated technical analysis")
        
        technical_data = {}
        
        for ticker, data in market_data.items():
            if 'history' not in data:
                continue
                
            hist = data['history']
            if len(hist) < 20:
                continue
            
            # Convert to PyTorch tensors for GPU processing
            prices = torch.tensor(hist['Close'].values, dtype=torch.float32, device=self.device)
            volumes = torch.tensor(hist['Volume'].values, dtype=torch.float32, device=self.device)
            
            # Calculate technical indicators on GPU
            rsi = self._calculate_rsi_gpu(prices)
            sma_20 = self._calculate_sma_gpu(prices, 20)
            sma_50 = self._calculate_sma_gpu(prices, 50)
            macd = self._calculate_macd_gpu(prices)
            atr = self._calculate_atr_gpu(hist, device=self.device)
            
            # Move results back to CPU for JSON serialization
            technical_data[ticker] = {
                'rsi': float(rsi.cpu()),
                'sma_20': float(sma_20.cpu()),
                'sma_50': float(sma_50.cpu()),
                'macd': float(macd.cpu()),
                'atr': float(atr.cpu()),
                'current_price': float(prices[-1].cpu()),
                'volume_ratio': float(volumes[-1].cpu() / volumes.mean().cpu())
            }
        
        return technical_data
    
    def _calculate_rsi_gpu(self, prices: torch.Tensor, period: int = 14) -> torch.Tensor:
        """Calculate RSI on GPU"""
        deltas = torch.diff(prices)
        gains = torch.where(deltas > 0, deltas, 0)
        losses = torch.where(deltas < 0, -deltas, 0)
        
        avg_gains = torch.mean(gains[-period:])
        avg_losses = torch.mean(losses[-period:])
        
        if avg_losses == 0:
            return torch.tensor(100.0, device=self.device)
        
        rs = avg_gains / avg_losses
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_sma_gpu(self, prices: torch.Tensor, period: int) -> torch.Tensor:
        """Calculate Simple Moving Average on GPU"""
        return torch.mean(prices[-period:])
    
    def _calculate_macd_gpu(self, prices: torch.Tensor) -> torch.Tensor:
        """Calculate MACD on GPU"""
        ema_12 = self._calculate_ema_gpu(prices, 12)
        ema_26 = self._calculate_ema_gpu(prices, 26)
        return ema_12 - ema_26
    
    def _calculate_ema_gpu(self, prices: torch.Tensor, period: int) -> torch.Tensor:
        """Calculate Exponential Moving Average on GPU"""
        alpha = 2.0 / (period + 1)
        ema = prices[0]
        for price in prices[1:]:
            ema = alpha * price + (1 - alpha) * ema
        return ema
    
    def _calculate_atr_gpu(self, hist: pd.DataFrame, device: torch.device) -> torch.Tensor:
        """Calculate Average True Range on GPU"""
        high = torch.tensor(hist['High'].values, dtype=torch.float32, device=device)
        low = torch.tensor(hist['Low'].values, dtype=torch.float32, device=device)
        close = torch.tensor(hist['Close'].values, dtype=torch.float32, device=device)
        
        tr1 = high - low
        tr2 = torch.abs(high - torch.roll(close, 1))
        tr3 = torch.abs(low - torch.roll(close, 1))
        
        true_range = torch.max(torch.stack([tr1, tr2, tr3]), dim=0)[0]
        atr = torch.mean(true_range[-14:])
        
        return atr
    
    async def _cpu_technical_analysis(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """CPU-based technical analysis fallback"""
        logger.info("💻 Running CPU technical analysis")
        
        technical_data = {}
        
        for ticker, data in market_data.items():
            if 'history' not in data:
                continue
                
            hist = data['history']
            if len(hist) < 20:
                continue
            
            # Calculate RSI
            rsi = self._calculate_rsi_cpu(hist['Close'])
            
            # Calculate SMAs
            sma_20 = hist['Close'].rolling(20).mean().iloc[-1]
            sma_50 = hist['Close'].rolling(50).mean().iloc[-1]
            
            # Calculate MACD
            ema_12 = hist['Close'].ewm(span=12).mean().iloc[-1]
            ema_26 = hist['Close'].ewm(span=26).mean().iloc[-1]
            macd = ema_12 - ema_26
            
            # Calculate ATR
            high_low = hist['High'] - hist['Low']
            high_close = np.abs(hist['High'] - hist['Close'].shift())
            low_close = np.abs(hist['Low'] - hist['Close'].shift())
            true_range = np.maximum(high_low, np.maximum(high_close, low_close))
            atr = true_range.rolling(14).mean().iloc[-1]
            
            technical_data[ticker] = {
                'rsi': float(rsi),
                'sma_20': float(sma_20),
                'sma_50': float(sma_50),
                'macd': float(macd),
                'atr': float(atr),
                'current_price': float(hist['Close'].iloc[-1]),
                'volume_ratio': float(hist['Volume'].iloc[-1] / hist['Volume'].rolling(20).mean().iloc[-1])
            }
        
        return technical_data
    
    def _calculate_rsi_cpu(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI on CPU"""
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
    
    async def _find_opportunities(self, market_data: Dict[str, Any], technical_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find trading opportunities based on technical analysis"""
        opportunities = []
        
        for ticker, tech in technical_data.items():
            if ticker not in market_data:
                continue
            
            current_price = tech['current_price']
            rsi = tech['rsi']
            sma_20 = tech['sma_20']
            sma_50 = tech['sma_50']
            atr = tech['atr']
            
            # Determine signal
            signal = 'HOLD'
            confidence = 0.5
            
            if rsi < 30 and current_price > sma_20:
                signal = 'BUY'
                confidence = 0.8
            elif rsi > 70 and current_price < sma_20:
                signal = 'SELL'
                confidence = 0.8
            elif rsi < 40 and current_price > sma_50:
                signal = 'WEAK_BUY'
                confidence = 0.6
            elif rsi > 60 and current_price < sma_50:
                signal = 'WEAK_SELL'
                confidence = 0.6
            
            # Calculate targets and stops
            if signal != 'HOLD':
                if 'BUY' in signal:
                    target = current_price + (atr * 2)
                    stop_loss = current_price - (atr * 1.5)
                else:  # SELL
                    target = current_price - (atr * 2)
                    stop_loss = current_price + (atr * 1.5)
                
                risk_reward = abs(target - current_price) / abs(current_price - stop_loss)
                
                opportunity = {
                    'ticker': ticker,
                    'signal': signal,
                    'confidence': confidence,
                    'current_price': current_price,
                    'target_price': target,
                    'stop_loss': stop_loss,
                    'risk_reward': risk_reward,
                    'rsi': rsi,
                    'sma_20': sma_20,
                    'sma_50': sma_50,
                    'atr': atr,
                    'timestamp': datetime.now().isoformat()
                }
                
                opportunities.append(opportunity)
        
        return opportunities
    
    def _calculate_market_sentiment(self, opportunities: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate overall market sentiment"""
        if not opportunities:
            return {"sentiment": "NEUTRAL", "confidence": 0.5}
        
        buy_signals = sum(1 for opp in opportunities if 'BUY' in opp['signal'])
        sell_signals = sum(1 for opp in opportunities if 'SELL' in opp['signal'])
        total_signals = len(opportunities)
        
        if buy_signals > sell_signals:
            sentiment = "BULLISH"
            confidence = buy_signals / total_signals
        elif sell_signals > buy_signals:
            sentiment = "BEARISH"
            confidence = sell_signals / total_signals
        else:
            sentiment = "NEUTRAL"
            confidence = 0.5
        
        return {
            "sentiment": sentiment,
            "confidence": confidence,
            "buy_signals": buy_signals,
            "sell_signals": sell_signals,
            "total_opportunities": total_signals
        }
    
    async def run_scan(self, tickers: List[str] = None):
        """Run the enhanced market scan"""
        if tickers is None:
            tickers = ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'TSLA', 'AMZN', 'META', 'NFLX', 'AMD', 'INTC']
        
        logger.info(f"🚀 Starting enhanced market scan for {len(tickers)} tickers")
        
        try:
            # Run GPU-accelerated scan
            results = await self.scan_market_with_gpu(tickers)
            
            # Save results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = f"reports/enhanced_scan_{timestamp}.json"
            
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            logger.info(f"📊 Scan complete - results saved to {results_file}")
            
            # Print summary
            print("\n" + "="*60)
            print("🎯 ENHANCED MARKET SCAN RESULTS")
            print("="*60)
            print(f"📅 Timestamp: {results['timestamp']}")
            print(f"🎮 GPU Used: {results['gpu_used']}")
            print(f"🤖 LLM Used: {results['llm_used']}")
            print(f"📊 Tickers Scanned: {results['tickers_scanned']}")
            print(f"🎯 Opportunities Found: {len(results['opportunities'])}")
            
            if results['market_sentiment']:
                sentiment = results['market_sentiment']
                print(f"📈 Market Sentiment: {sentiment['sentiment']} ({sentiment['confidence']:.1%})")
            
            if results['ai_analysis']:
                ai = results['ai_analysis']
                print(f"🤖 AI Analysis: {ai['confidence']:.1%} confidence")
                print(f"🤖 Model: {ai.get('model', 'Unknown')}")
            
            print("\n🎯 TOP OPPORTUNITIES:")
            for i, opp in enumerate(results['opportunities'][:5], 1):
                print(f"{i}. {opp['ticker']} - {opp['signal']} ({opp['confidence']:.1%})")
                print(f"   Price: ${opp['current_price']:.2f} | Target: ${opp['target_price']:.2f} | Stop: ${opp['stop_loss']:.2f}")
                print(f"   RSI: {opp['rsi']:.1f} | R/R: {opp['risk_reward']:.2f}")
            
            return results
            
        except Exception as e:
            logger.error(f"Scan failed: {e}")
            raise

async def main():
    """Main function"""
    # Create scanner
    scanner = EnhancedScannerGPU()
    
    # Define tickers to scan
    tickers = [
        'AAPL', 'MSFT', 'GOOGL', 'NVDA', 'TSLA', 'AMZN', 'META', 'NFLX', 'AMD', 'INTC',
        'SPY', 'QQQ', 'IWM', 'VIX', '^TNX', 'DX-Y.NYB'
    ]
    
    # Run scan
    await scanner.run_scan(tickers)

if __name__ == "__main__":
    asyncio.run(main())
