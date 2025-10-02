#!/usr/bin/env python3
"""
Overbought/Oversold Stock Scanner

Continuously monitors stocks for overbought/oversold conditions and maintains
an ongoing list of opportunities. Can be run as a scheduled job every few hours.
"""

import os
import sys
import logging
import json
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np

# Add QuantEngine to path
quant_engine_dir = Path(__file__).parent
sys.path.insert(0, str(quant_engine_dir))

from engine.data_ingestion.data_manager import DataManager
from llm_integration import OllamaLLM

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('overbought_oversold_scanner.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class OverboughtOversoldScanner:
    """Scanner for overbought/oversold stock conditions"""
    
    def __init__(self):
        self.data_manager = DataManager({})
        self.llm = None
        
        # Initialize LLM if available
        try:
            self.llm = OllamaLLM(model="qwen2.5:72b")
            logger.info("✅ LLM integration available")
        except Exception as e:
            logger.warning(f"⚠️ LLM not available: {e}")
        
        # Stock universe to monitor
        self.stock_universe = [
            # Tech stocks
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'NFLX', 'ADBE', 'CRM',
            # Financial stocks
            'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'AXP', 'V', 'MA', 'PYPL',
            # Healthcare stocks
            'JNJ', 'PFE', 'UNH', 'ABBV', 'MRK', 'TMO', 'DHR', 'BMY', 'AMGN', 'GILD',
            # Consumer stocks
            'KO', 'PEP', 'WMT', 'HD', 'MCD', 'NKE', 'SBUX', 'TGT', 'LOW', 'COST',
            # Energy stocks
            'XOM', 'CVX', 'COP', 'EOG', 'SLB', 'KMI', 'WMB', 'PSX', 'VLO', 'MPC',
            # Industrial stocks
            'BA', 'CAT', 'GE', 'HON', 'MMM', 'UPS', 'FDX', 'LMT', 'RTX', 'NOC',
            # ETFs for broader market
            'SPY', 'QQQ', 'IWM', 'VTI', 'XLK', 'XLF', 'XLV', 'XLE', 'XLI', 'XLY'
        ]
        
        # RSI thresholds
        self.overbought_threshold = 70
        self.oversold_threshold = 30
        self.extreme_overbought = 80
        self.extreme_oversold = 20
        
        # Data storage
        self.data_file = Path("overbought_oversold_data.json")
        self.load_existing_data()
        
        logger.info("✅ Overbought/Oversold Scanner initialized")

    def load_existing_data(self):
        """Load existing scan data"""
        if self.data_file.exists():
            try:
                with open(self.data_file, 'r') as f:
                    self.scan_data = json.load(f)
                logger.info(f"✅ Loaded existing data with {len(self.scan_data.get('stocks', {}))} stocks")
            except Exception as e:
                logger.warning(f"⚠️ Failed to load existing data: {e}")
                self.scan_data = {'stocks': {}, 'last_scan': None, 'scan_history': []}
        else:
            self.scan_data = {'stocks': {}, 'last_scan': None, 'scan_history': []}

    def save_data(self):
        """Save scan data to file"""
        try:
            with open(self.data_file, 'w') as f:
                json.dump(self.scan_data, f, indent=2, default=str)
            logger.info("✅ Data saved successfully")
        except Exception as e:
            logger.error(f"❌ Failed to save data: {e}")

    def calculate_rsi(self, prices: pd.Series, window: int = 14) -> float:
        """Calculate RSI for a price series"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        rsi_value = rsi.iloc[-1]
        if hasattr(rsi_value, 'iloc'):
            rsi_value = rsi_value.iloc[0]
        return float(rsi_value) if not pd.isna(rsi_value) else 50

    def analyze_stock(self, ticker: str) -> Optional[Dict[str, Any]]:
        """Analyze a single stock for overbought/oversold conditions"""
        try:
            # Get recent data (last 60 days)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=60)
            
            data = self.data_manager.get_market_data([ticker], start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
            
            if not data or ticker not in data:
                logger.warning(f"⚠️ No data for {ticker}")
                return None
            
            stock_data = data[ticker]
            if stock_data.empty:
                return None
            
            # Calculate technical indicators
            current_price = float(stock_data['close'].iloc[-1])
            price_change = float(stock_data['close'].pct_change().iloc[-1]) * 100
            rsi = self.calculate_rsi(stock_data['close'])
            
            # Volume analysis
            avg_volume = float(stock_data['volume'].rolling(20).mean().iloc[-1])
            current_volume = float(stock_data['volume'].iloc[-1])
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
            
            # Moving averages
            sma_20 = float(stock_data['close'].rolling(20).mean().iloc[-1])
            sma_50 = float(stock_data['close'].rolling(50).mean().iloc[-1])
            
            # Determine condition
            condition = "neutral"
            signal_strength = "medium"
            
            if rsi >= self.extreme_overbought:
                condition = "extreme_overbought"
                signal_strength = "strong"
            elif rsi >= self.overbought_threshold:
                condition = "overbought"
                signal_strength = "medium"
            elif rsi <= self.extreme_oversold:
                condition = "extreme_oversold"
                signal_strength = "strong"
            elif rsi <= self.oversold_threshold:
                condition = "oversold"
                signal_strength = "medium"
            
            # Calculate confidence based on multiple factors
            confidence = 0.5
            
            if condition in ["overbought", "extreme_overbought"]:
                if volume_ratio > 1.5:  # High volume confirms
                    confidence += 0.2
                if current_price > sma_20 > sma_50:  # Uptrend confirms
                    confidence += 0.1
                if price_change > 2:  # Strong momentum
                    confidence += 0.1
            elif condition in ["oversold", "extreme_oversold"]:
                if volume_ratio > 1.5:  # High volume confirms
                    confidence += 0.2
                if current_price < sma_20 < sma_50:  # Downtrend confirms
                    confidence += 0.1
                if price_change < -2:  # Strong momentum
                    confidence += 0.1
            
            confidence = min(0.95, confidence)
            
            # Generate trading signal
            if condition in ["extreme_overbought", "overbought"]:
                signal = "SELL"
                target_price = current_price * 0.95
                stop_loss = current_price * 1.05
            elif condition in ["extreme_oversold", "oversold"]:
                signal = "BUY"
                target_price = current_price * 1.05
                stop_loss = current_price * 0.95
            else:
                signal = "HOLD"
                target_price = current_price * 1.02
                stop_loss = current_price * 0.98
            
            return {
                'ticker': ticker,
                'current_price': current_price,
                'price_change': price_change,
                'rsi': rsi,
                'condition': condition,
                'signal': signal,
                'confidence': confidence,
                'signal_strength': signal_strength,
                'target_price': target_price,
                'stop_loss': stop_loss,
                'volume_ratio': volume_ratio,
                'sma_20': sma_20,
                'sma_50': sma_50,
                'timestamp': datetime.now().isoformat(),
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to analyze {ticker}: {e}")
            return None

    def scan_all_stocks(self) -> Dict[str, Any]:
        """Scan all stocks in the universe"""
        logger.info(f"🔍 Scanning {len(self.stock_universe)} stocks for overbought/oversold conditions...")
        
        results = {
            'overbought': [],
            'oversold': [],
            'extreme_overbought': [],
            'extreme_oversold': [],
            'neutral': [],
            'scan_timestamp': datetime.now().isoformat()
        }
        
        for i, ticker in enumerate(self.stock_universe):
            try:
                logger.info(f"📊 Analyzing {ticker} ({i+1}/{len(self.stock_universe)})")
                
                analysis = self.analyze_stock(ticker)
                if analysis:
                    # Update existing data or add new
                    self.scan_data['stocks'][ticker] = analysis
                    
                    # Categorize by condition
                    condition = analysis['condition']
                    if condition == 'extreme_overbought':
                        results['extreme_overbought'].append(analysis)
                    elif condition == 'overbought':
                        results['overbought'].append(analysis)
                    elif condition == 'extreme_oversold':
                        results['extreme_oversold'].append(analysis)
                    elif condition == 'oversold':
                        results['oversold'].append(analysis)
                    else:
                        results['neutral'].append(analysis)
                
                # Rate limiting
                if i % 10 == 0 and i > 0:
                    time.sleep(2)  # Pause every 10 stocks
                    
            except Exception as e:
                logger.error(f"❌ Error analyzing {ticker}: {e}")
                continue
        
        # Update scan history
        self.scan_data['last_scan'] = datetime.now().isoformat()
        self.scan_data['scan_history'].append({
            'timestamp': datetime.now().isoformat(),
            'overbought_count': len(results['overbought']),
            'oversold_count': len(results['oversold']),
            'extreme_overbought_count': len(results['extreme_overbought']),
            'extreme_oversold_count': len(results['extreme_oversold'])
        })
        
        # Keep only last 100 scans
        if len(self.scan_data['scan_history']) > 100:
            self.scan_data['scan_history'] = self.scan_data['scan_history'][-100:]
        
        # Save data
        self.save_data()
        
        logger.info(f"✅ Scan completed:")
        logger.info(f"   📈 Overbought: {len(results['overbought'])}")
        logger.info(f"   📉 Oversold: {len(results['oversold'])}")
        logger.info(f"   🔥 Extreme Overbought: {len(results['extreme_overbought'])}")
        logger.info(f"   ❄️ Extreme Oversold: {len(results['extreme_oversold'])}")
        
        return results

    def generate_report(self, results: Dict[str, Any]) -> str:
        """Generate a comprehensive report"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        report = f"""
# Overbought/Oversold Stock Scanner Report
**Generated:** {timestamp}
**Total Stocks Scanned:** {len(self.stock_universe)}

## Summary
- **Overbought Stocks:** {len(results['overbought'])}
- **Oversold Stocks:** {len(results['oversold'])}
- **Extreme Overbought:** {len(results['extreme_overbought'])}
- **Extreme Oversold:** {len(results['extreme_oversold'])}

## 🔥 Extreme Overbought Stocks (RSI > 80)
"""
        
        for stock in sorted(results['extreme_overbought'], key=lambda x: x['rsi'], reverse=True):
            report += f"""
### {stock['ticker']} - {stock['signal']}
- **Current Price:** ${stock['current_price']:.2f}
- **RSI:** {stock['rsi']:.1f}
- **Daily Change:** {stock['price_change']:+.2f}%
- **Confidence:** {stock['confidence']:.1%}
- **Target Price:** ${stock['target_price']:.2f}
- **Stop Loss:** ${stock['stop_loss']:.2f}
- **Volume Ratio:** {stock['volume_ratio']:.1f}x
"""
        
        report += f"""
## 📈 Overbought Stocks (RSI 70-80)
"""
        
        for stock in sorted(results['overbought'], key=lambda x: x['rsi'], reverse=True):
            report += f"""
### {stock['ticker']} - {stock['signal']}
- **Current Price:** ${stock['current_price']:.2f}
- **RSI:** {stock['rsi']:.1f}
- **Daily Change:** {stock['price_change']:+.2f}%
- **Confidence:** {stock['confidence']:.1%}
- **Target Price:** ${stock['target_price']:.2f}
- **Stop Loss:** ${stock['stop_loss']:.2f}
"""
        
        report += f"""
## 📉 Oversold Stocks (RSI 20-30)
"""
        
        for stock in sorted(results['oversold'], key=lambda x: x['rsi']):
            report += f"""
### {stock['ticker']} - {stock['signal']}
- **Current Price:** ${stock['current_price']:.2f}
- **RSI:** {stock['rsi']:.1f}
- **Daily Change:** {stock['price_change']:+.2f}%
- **Confidence:** {stock['confidence']:.1%}
- **Target Price:** ${stock['target_price']:.2f}
- **Stop Loss:** ${stock['stop_loss']:.2f}
"""
        
        report += f"""
## ❄️ Extreme Oversold Stocks (RSI < 20)
"""
        
        for stock in sorted(results['extreme_oversold'], key=lambda x: x['rsi']):
            report += f"""
### {stock['ticker']} - {stock['signal']}
- **Current Price:** ${stock['current_price']:.2f}
- **RSI:** {stock['rsi']:.1f}
- **Daily Change:** {stock['price_change']:+.2f}%
- **Confidence:** {stock['confidence']:.1%}
- **Target Price:** ${stock['target_price']:.2f}
- **Stop Loss:** ${stock['stop_loss']:.2f}
"""
        
        report += f"""
## 📊 Top Opportunities by Confidence
"""
        
        # Combine all non-neutral stocks and sort by confidence
        all_opportunities = (results['overbought'] + results['oversold'] + 
                           results['extreme_overbought'] + results['extreme_oversold'])
        top_opportunities = sorted(all_opportunities, key=lambda x: x['confidence'], reverse=True)[:10]
        
        for i, stock in enumerate(top_opportunities, 1):
            report += f"""
### {i}. {stock['ticker']} - {stock['signal']} ({stock['condition'].replace('_', ' ').title()})
- **RSI:** {stock['rsi']:.1f}
- **Confidence:** {stock['confidence']:.1%}
- **Price:** ${stock['current_price']:.2f} ({stock['price_change']:+.2f}%)
- **Target:** ${stock['target_price']:.2f}
- **Stop Loss:** ${stock['stop_loss']:.2f}
"""
        
        report += f"""
## 📈 Scan History
Recent scan activity:
"""
        
        for scan in self.scan_data['scan_history'][-5:]:  # Last 5 scans
            scan_time = datetime.fromisoformat(scan['timestamp']).strftime('%Y-%m-%d %H:%M')
            report += f"""
- **{scan_time}**: {scan['overbought_count']} overbought, {scan['oversold_count']} oversold, {scan['extreme_overbought_count']} extreme overbought, {scan['extreme_oversold_count']} extreme oversold
"""
        
        report += f"""
---
*Generated by QuantEngine Overbought/Oversold Scanner*
*Next scan recommended in 2-4 hours*
"""
        
        return report

    def run_scan(self, save_report: bool = True) -> str:
        """Run a complete scan and generate report"""
        logger.info("🚀 Starting overbought/oversold scan...")
        
        # Run the scan
        results = self.scan_all_stocks()
        
        # Generate report
        report = self.generate_report(results)
        
        # Save report if requested
        if save_report:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            report_file = f"reports/overbought_oversold_scan_{timestamp}.md"
            
            # Create reports directory if it doesn't exist
            reports_dir = Path("reports")
            reports_dir.mkdir(exist_ok=True)
            
            with open(report_file, 'w') as f:
                f.write(report)
            
            logger.info(f"📄 Report saved: {report_file}")
        
        return report

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Overbought/Oversold Stock Scanner')
    parser.add_argument('--scan', action='store_true', help='Run a scan')
    parser.add_argument('--continuous', action='store_true', help='Run continuous scanning')
    parser.add_argument('--interval', type=int, default=4, help='Scan interval in hours (default: 4)')
    parser.add_argument('--stocks', nargs='+', help='Specific stocks to scan')
    
    args = parser.parse_args()
    
    scanner = OverboughtOversoldScanner()
    
    if args.stocks:
        # Scan specific stocks
        scanner.stock_universe = args.stocks
        logger.info(f"🎯 Scanning specific stocks: {args.stocks}")
    
    if args.continuous:
        logger.info(f"🔄 Starting continuous scanning every {args.interval} hours...")
        while True:
            try:
                report = scanner.run_scan()
                print("\n" + "="*80)
                print(report)
                print("="*80)
                
                # Wait for next scan
                sleep_seconds = args.interval * 3600
                logger.info(f"⏰ Waiting {args.interval} hours until next scan...")
                time.sleep(sleep_seconds)
                
            except KeyboardInterrupt:
                logger.info("🛑 Continuous scanning stopped by user")
                break
            except Exception as e:
                logger.error(f"❌ Error in continuous scanning: {e}")
                time.sleep(300)  # Wait 5 minutes before retry
    else:
        # Single scan
        report = scanner.run_scan()
        print("\n" + "="*80)
        print(report)
        print("="*80)

if __name__ == "__main__":
    main()
