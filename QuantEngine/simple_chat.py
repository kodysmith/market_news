#!/usr/bin/env python3
"""
Simple QuantEngine Chat Interface
"""

import sys
import json
from pathlib import Path
from datetime import datetime

# Add QuantEngine to path
quant_engine_dir = Path(__file__).parent
sys.path.insert(0, str(quant_engine_dir))

from daily_strategy_advisor import DailyStrategyAdvisor
from overbought_oversold_scanner import OverboughtOversoldScanner
from real_time_price import get_real_time_analysis

class SimpleChat:
    def __init__(self):
        self.advisor = DailyStrategyAdvisor()
        self.scanner = OverboughtOversoldScanner()
        print("✅ QuantEngine Chat initialized")

    def get_trade_signals(self, asset=None):
        """Get trade signals for an asset"""
        print(f"🔍 Getting trade signals for {asset or 'market'}...")
        
        try:
            if asset:
                report = self.advisor.get_daily_recommendations(asset=asset)
            else:
                report = self.advisor.get_daily_recommendations()
            
            # Extract key info from report
            lines = report.split('\n')
            opportunities = []
            strategies = []
            
            in_opportunities = False
            in_strategies = False
            
            for line in lines:
                if 'Top Trading Opportunities' in line:
                    in_opportunities = True
                    continue
                elif 'Recommended Strategies' in line:
                    in_opportunities = False
                    in_strategies = True
                    continue
                elif in_opportunities and line.startswith('###'):
                    parts = line.split(' - ')
                    if len(parts) >= 2:
                        ticker = parts[0].replace('### ', '').strip()
                        signal = parts[1].strip()
                        opportunities.append({'ticker': ticker, 'signal': signal})
                elif in_strategies and line.startswith('###'):
                    strategy_name = line.replace('### ', '').strip()
                    strategies.append({'name': strategy_name})
            
            return {
                'success': True,
                'asset': asset or 'market',
                'opportunities': opportunities,
                'strategies': strategies,
                'report': report
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def check_overbought_oversold(self, category=None):
        """Check overbought/oversold stocks"""
        print(f"�� Checking overbought/oversold stocks...")
        
        try:
            results = self.scanner.scan_all_stocks()
            
            if category == 'overbought':
                stocks = results.get('overbought', []) + results.get('extreme_overbought', [])
            elif category == 'oversold':
                stocks = results.get('oversold', []) + results.get('extreme_oversold', [])
            else:
                stocks = (results.get('overbought', []) + results.get('oversold', []) + 
                         results.get('extreme_overbought', []) + results.get('extreme_oversold', []))
            
            return {
                'success': True,
                'category': category or 'all',
                'stocks': stocks,
                'count': len(stocks)
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def get_real_time_analysis(self, ticker):
        """Get real-time analysis for a ticker"""
        print(f"🔍 Getting real-time analysis for {ticker}...")
        
        try:
            import io
            import contextlib
            
            f = io.StringIO()
            with contextlib.redirect_stdout(f):
                get_real_time_analysis(ticker)
            
            output = f.getvalue()
            
            return {
                'success': True,
                'ticker': ticker,
                'analysis': output
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def chat(self, message):
        """Main chat interface"""
        message_lower = message.lower()
        
        # Check for trade signals
        if any(phrase in message_lower for phrase in ['trade signals', 'trading signals', 'signals for', 'check signals']):
            asset = None
            for ticker in ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'TSLA', 'AMZN', 'META', 'NFLX', 'AMD', 'INTC']:
                if ticker.lower() in message_lower:
                    asset = ticker
                    break
            
            result = self.get_trade_signals(asset)
            
            if result['success']:
                response = f"Hey, I got that report back! Here's what I found:\n\n"
                
                if result['opportunities']:
                    response += "**Trading Opportunities:**\n"
                    for opp in result['opportunities']:
                        response += f"- {opp['ticker']}: {opp['signal']}\n"
                else:
                    response += "No clear trading opportunities right now.\n"
                
                if result['strategies']:
                    response += "\n**Recommended Strategies:**\n"
                    for strategy in result['strategies']:
                        response += f"- {strategy['name']}\n"
                
                return response
            else:
                return f"Sorry, I had trouble getting that information. Error: {result['error']}"
        
        # Check for overbought/oversold
        elif any(phrase in message_lower for phrase in ['overbought', 'oversold', 'overbought/oversold']):
            category = None
            if 'overbought' in message_lower and 'oversold' not in message_lower:
                category = 'overbought'
            elif 'oversold' in message_lower and 'overbought' not in message_lower:
                category = 'oversold'
            
            result = self.check_overbought_oversold(category)
            
            if result['success']:
                stocks = result['stocks']
                count = result['count']
                
                response = f"Found {count} overbought/oversold stocks:\n\n"
                
                for stock in stocks[:5]:  # Show top 5
                    response += f"- {stock['ticker']}: {stock['signal']} (RSI: {stock['rsi']:.1f})\n"
                
                if count > 5:
                    response += f"... and {count - 5} more\n"
                
                return response
            else:
                return f"Sorry, I had trouble getting that information. Error: {result['error']}"
        
        # Check for real-time analysis
        elif any(phrase in message_lower for phrase in ['real time', 'current price', 'right now', 'live']):
            for ticker in ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'TSLA', 'AMZN', 'META', 'NFLX', 'AMD', 'INTC']:
                if ticker.lower() in message_lower:
                    result = self.get_real_time_analysis(ticker)
                    
                    if result['success']:
                        return f"Here's the real-time analysis for {ticker}:\n\n{result['analysis']}"
                    else:
                        return f"Sorry, I had trouble getting that information. Error: {result['error']}"
        
        # Default response
        return """Hey! I can help you with:

🔍 **Trade Signals**: "Check NVDA trade signals for the day"
📊 **Overbought/Oversold**: "What stocks are overbought right now?"
💰 **Real-time Analysis**: "What's NVDA doing right now?"

Just ask me naturally and I'll run the analysis for you!"""

def main():
    """Main entry point"""
    print("🤖 QuantEngine Chat Interface")
    print("=" * 50)
    print("Type 'quit' to exit")
    print()
    
    chat = SimpleChat()
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("👋 Goodbye!")
                break
            
            if not user_input:
                continue
            
            response = chat.chat(user_input)
            print(f"\nBot: {response}\n")
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
