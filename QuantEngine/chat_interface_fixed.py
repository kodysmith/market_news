#!/usr/bin/env python3
"""
Fixed QuantEngine Chat Interface

Simplified version that avoids database conflicts.
"""

import sys
import json
import subprocess
from pathlib import Path
from datetime import datetime

def run_command(cmd):
    """Run a command and return the output"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60)
        return result.stdout, result.stderr, result.returncode
    except subprocess.TimeoutExpired:
        return "", "Command timed out", 1
    except Exception as e:
        return "", str(e), 1

class FixedChat:
    def __init__(self):
        print("✅ QuantEngine Chat Interface (Fixed)")
        print("=" * 50)

    def get_trade_signals(self, asset=None):
        """Get trade signals for an asset"""
        print(f"🔍 Getting trade signals for {asset or 'market'}...")
        
        if asset:
            cmd = f"python3 daily_strategy_advisor.py --asset {asset}"
        else:
            cmd = "python3 daily_strategy_advisor.py"
        
        stdout, stderr, returncode = run_command(cmd)
        
        if returncode == 0:
            # Extract key info from report
            lines = stdout.split('\n')
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
                'strategies': strategies
            }
        else:
            return {'success': False, 'error': stderr}

    def check_overbought_oversold(self, category=None):
        """Check overbought/oversold stocks"""
        print(f"🔍 Checking overbought/oversold stocks...")
        
        cmd = "python3 overbought_oversold_scanner.py --scan --stocks AAPL MSFT GOOGL NVDA TSLA"
        stdout, stderr, returncode = run_command(cmd)
        
        if returncode == 0:
            # Parse the output to extract overbought/oversold stocks
            lines = stdout.split('\n')
            stocks = []
            
            in_overbought = False
            in_oversold = False
            
            for line in lines:
                if 'Overbought Stocks' in line:
                    in_overbought = True
                    continue
                elif 'Oversold Stocks' in line:
                    in_overbought = False
                    in_oversold = True
                    continue
                elif in_overbought and line.startswith('###'):
                    parts = line.split(' - ')
                    if len(parts) >= 2:
                        ticker = parts[0].replace('### ', '').strip()
                        signal = parts[1].strip()
                        stocks.append({'ticker': ticker, 'signal': signal, 'category': 'overbought'})
                elif in_oversold and line.startswith('###'):
                    parts = line.split(' - ')
                    if len(parts) >= 2:
                        ticker = parts[0].replace('### ', '').strip()
                        signal = parts[1].strip()
                        stocks.append({'ticker': ticker, 'signal': signal, 'category': 'oversold'})
            
            return {
                'success': True,
                'stocks': stocks,
                'count': len(stocks)
            }
        else:
            return {'success': False, 'error': stderr}

    def get_real_time_analysis(self, ticker):
        """Get real-time analysis for a ticker"""
        print(f"🔍 Getting real-time analysis for {ticker}...")
        
        cmd = f"python3 real_time_price.py {ticker}"
        stdout, stderr, returncode = run_command(cmd)
        
        if returncode == 0:
            return {
                'success': True,
                'ticker': ticker,
                'analysis': stdout
            }
        else:
            return {'success': False, 'error': stderr}

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
            result = self.check_overbought_oversold()
            
            if result['success']:
                stocks = result['stocks']
                count = result['count']
                
                response = f"Found {count} overbought/oversold stocks:\n\n"
                
                for stock in stocks[:5]:  # Show top 5
                    response += f"- {stock['ticker']}: {stock['signal']} ({stock['category']})\n"
                
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
    print("🤖 QuantEngine Chat Interface (Fixed)")
    print("=" * 50)
    print("Type 'quit' to exit")
    print()
    
    chat = FixedChat()
    
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

