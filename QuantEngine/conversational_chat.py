#!/usr/bin/env python3
"""
Conversational QuantEngine Chat Interface

A truly conversational interface that can handle follow-up questions and dynamic responses.
"""

import sys
import json
import subprocess
import requests
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

def run_command(cmd):
    """Run a command and return the output"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60)
        return result.stdout, result.stderr, result.returncode
    except subprocess.TimeoutExpired:
        return "", "Command timed out", 1
    except Exception as e:
        return "", str(e), 1

class OllamaLLM:
    """Ollama LLM integration for conversational responses"""
    
    def __init__(self, model: str = "qwen2.5:72b", host: str = "http://localhost:11434"):
        self.model = model
        self.host = host
        self.api_url = f"{self.host}/api/generate"
        self._check_ollama_status()
    
    def _check_ollama_status(self):
        """Check if Ollama is running"""
        try:
            response = requests.get(f"{self.host}/api/tags", timeout=5)
            if response.status_code == 200:
                print("✅ Ollama is running")
            else:
                raise Exception("Ollama not responding")
        except Exception as e:
            raise Exception(f"Ollama not available: {e}")
    
    def generate(self, prompt: str, max_tokens: int = 300, temperature: float = 0.7) -> str:
        """Generate response using Ollama"""
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
            raise Exception(f"LLM generation failed: {e}")
    
    def generate_conversational_response(self, user_message: str, context: Dict[str, Any]) -> str:
        """Generate a conversational response based on context"""
        
        # Create a more conversational prompt based on context type
        if context.get('analysis_type') in ['general_conversation', 'greeting', 'acknowledgment']:
            prompt = f"""
You are a friendly trading assistant. The user said: "{user_message}"

Respond naturally and conversationally. Be brief and human-like. If they're just chatting, be friendly. If they want trading analysis, offer to help. Keep it under 50 words.
"""
        else:
            prompt = f"""
You are a helpful trading assistant. The user asked: "{user_message}"

Context from QuantEngine analysis:
{json.dumps(context, indent=2)}

Provide a natural, conversational response that:
1. Acknowledges what the user asked
2. Summarizes the key findings from the analysis
3. Offers specific insights or recommendations
4. Suggests follow-up questions they might want to ask

Be helpful, specific, and conversational. Keep it under 200 words.
"""
        
        return self.generate(prompt, max_tokens=200, temperature=0.7)

class ConversationalChat:
    def __init__(self):
        self.conversation_history = []
        self.last_analysis = None
        self.last_stocks = None
        self.llm_available = False
        
        # Initialize Ollama LLM
        try:
            self.llm = OllamaLLM()
            self.llm_available = True
            print("✅ Conversational QuantEngine Chat Interface (with LLM)")
        except Exception as e:
            print(f"⚠️ LLM not available: {e}")
            print("✅ Conversational QuantEngine Chat Interface (basic mode)")
        print("=" * 60)

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
            
            self.last_analysis = {
                'type': 'trade_signals',
                'asset': asset or 'market',
                'opportunities': opportunities,
                'strategies': strategies,
                'full_report': stdout
            }
            
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
        
        cmd = "python3 overbought_oversold_scanner.py --scan --stocks AAPL MSFT GOOGL NVDA TSLA AMZN META NFLX AMD INTC"
        stdout, stderr, returncode = run_command(cmd)
        
        if returncode == 0:
            # Parse the output to extract overbought/oversold stocks
            lines = stdout.split('\n')
            overbought_stocks = []
            oversold_stocks = []
            
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
                        overbought_stocks.append({'ticker': ticker, 'signal': signal, 'category': 'overbought'})
                elif in_oversold and line.startswith('###'):
                    parts = line.split(' - ')
                    if len(parts) >= 2:
                        ticker = parts[0].replace('### ', '').strip()
                        signal = parts[1].strip()
                        oversold_stocks.append({'ticker': ticker, 'signal': signal, 'category': 'oversold'})
            
            all_stocks = overbought_stocks + oversold_stocks
            self.last_stocks = {
                'overbought': overbought_stocks,
                'oversold': oversold_stocks,
                'all': all_stocks
            }
            
            return {
                'success': True,
                'overbought': overbought_stocks,
                'oversold': oversold_stocks,
                'all': all_stocks,
                'count': len(all_stocks)
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

    def get_fundamental_analysis(self, ticker):
        """Get fundamental analysis for a ticker"""
        print(f"📊 Getting fundamental analysis for {ticker}...")
        
        cmd = f"python3 fundamental_analysis.py {ticker}"
        stdout, stderr, returncode = run_command(cmd)
        
        if returncode == 0:
            return {
                'success': True,
                'ticker': ticker,
                'analysis': stdout
            }
        else:
            return {'success': False, 'error': stderr}

    def handle_follow_up_question(self, message):
        """Handle follow-up questions about previous analysis"""
        message_lower = message.lower()
        
        # Check if asking about overbought stocks
        if any(phrase in message_lower for phrase in ['overbought', 'over bought', 'sell signals']):
            if self.last_stocks and self.last_stocks.get('overbought'):
                stocks = self.last_stocks['overbought']
                
                # Use LLM for conversational response if available
                if self.llm_available:
                    try:
                        context = {
                            'analysis_type': 'follow_up_overbought',
                            'stocks': stocks[:10],
                            'total_count': len(stocks)
                        }
                        response = self.llm.generate_conversational_response(message, context)
                        return response
                    except Exception as e:
                        print(f"⚠️ LLM response failed: {e}")
                
                # Fallback to basic response
                response = f"Here are the overbought stocks I found:\n\n"
                for stock in stocks[:10]:  # Show top 10
                    response += f"- {stock['ticker']}: {stock['signal']}\n"
                if len(stocks) > 10:
                    response += f"... and {len(stocks) - 10} more overbought stocks\n"
                return response
            else:
                return "I don't have overbought stock data. Let me check for you..."
        
        # Check if asking about oversold stocks
        elif any(phrase in message_lower for phrase in ['oversold', 'over sold', 'buy signals']):
            if self.last_stocks and self.last_stocks.get('oversold'):
                stocks = self.last_stocks['oversold']
                
                # Use LLM for conversational response if available
                if self.llm_available:
                    try:
                        context = {
                            'analysis_type': 'follow_up_oversold',
                            'stocks': stocks[:10],
                            'total_count': len(stocks)
                        }
                        response = self.llm.generate_conversational_response(message, context)
                        return response
                    except Exception as e:
                        print(f"⚠️ LLM response failed: {e}")
                
                # Fallback to basic response
                response = f"Here are the oversold stocks I found:\n\n"
                for stock in stocks[:10]:  # Show top 10
                    response += f"- {stock['ticker']}: {stock['signal']}\n"
                if len(stocks) > 10:
                    response += f"... and {len(stocks) - 10} more oversold stocks\n"
                return response
            else:
                return "I don't have oversold stock data. Let me check for you..."
        
        # Check if asking about specific stocks
        elif any(phrase in message_lower for phrase in ['which stocks', 'what stocks', 'show me']):
            if self.last_stocks:
                all_stocks = self.last_stocks.get('all', [])
                if all_stocks:
                    # Use LLM for conversational response if available
                    if self.llm_available:
                        try:
                            context = {
                                'analysis_type': 'follow_up_all_stocks',
                                'stocks': all_stocks[:15],
                                'total_count': len(all_stocks)
                            }
                            response = self.llm.generate_conversational_response(message, context)
                            return response
                        except Exception as e:
                            print(f"⚠️ LLM response failed: {e}")
                    
                    # Fallback to basic response
                    response = f"I found {len(all_stocks)} stocks with signals:\n\n"
                    for stock in all_stocks[:15]:  # Show top 15
                        response += f"- {stock['ticker']}: {stock['signal']} ({stock['category']})\n"
                    if len(all_stocks) > 15:
                        response += f"... and {len(all_stocks) - 15} more stocks\n"
                    return response
                else:
                    return "I don't have stock data. Let me scan for you..."
            else:
                return "I don't have stock data. Let me scan for you..."
        
        # Check if asking about specific ticker
        for ticker in ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'TSLA', 'AMZN', 'META', 'NFLX', 'AMD', 'INTC']:
            if ticker.lower() in message_lower:
                # Actually execute the real-time analysis
                result = self.get_real_time_analysis(ticker)
                if result['success']:
                    if self.llm_available:
                        try:
                            context = {
                                'analysis_type': 'real_time_analysis',
                                'ticker': ticker,
                                'analysis': result['analysis']
                            }
                            response = self.llm.generate_conversational_response(message, context)
                            return response
                        except Exception as e:
                            print(f"⚠️ LLM response failed: {e}")
                    return f"Here's the real-time analysis for {ticker}:\n\n{result['analysis']}"
                else:
                    return f"Sorry, I had trouble getting real-time data for {ticker}. Error: {result['error']}"
        
        return None

    def chat(self, message):
        """Main chat interface with conversational responses"""
        message_lower = message.lower()
        
        # Add to conversation history
        self.conversation_history.append({'user': message, 'timestamp': datetime.now().isoformat()})
        
        # Handle greetings and casual conversation
        if any(phrase in message_lower for phrase in ['hi', 'hello', 'hey', 'how are you', 'what\'s up', 'good morning', 'good afternoon', 'good evening']):
            if self.llm_available:
                try:
                    context = {
                        'analysis_type': 'greeting',
                        'user_message': message,
                        'conversation_history': self.conversation_history[-2:]
                    }
                    response = self.llm.generate_conversational_response(message, context)
                    return response
                except Exception as e:
                    print(f"⚠️ LLM response failed: {e}")
            return "Hey there! I'm your QuantEngine trading assistant. I can help you analyze stocks, find trading opportunities, and answer questions about the market. What would you like to know?"
        
        # Handle thanks and acknowledgments
        if any(phrase in message_lower for phrase in ['thanks', 'thank you', 'cool', 'awesome', 'great', 'nice', 'good']):
            if self.llm_available:
                try:
                    context = {
                        'analysis_type': 'acknowledgment',
                        'user_message': message,
                        'conversation_history': self.conversation_history[-2:]
                    }
                    response = self.llm.generate_conversational_response(message, context)
                    return response
                except Exception as e:
                    print(f"⚠️ LLM response failed: {e}")
            return "You're welcome! Is there anything else you'd like me to analyze or any other questions you have?"
        
        # First, try to handle follow-up questions
        follow_up_response = self.handle_follow_up_question(message)
        if follow_up_response:
            return follow_up_response
        
        # Check for trade signals
        if any(phrase in message_lower for phrase in ['trade signals', 'trading signals', 'signals for', 'check signals', 'opportunities', 'good trade']):
            asset = None
            for ticker in ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'TSLA', 'AMZN', 'META', 'NFLX', 'AMD', 'INTC']:
                if ticker.lower() in message_lower:
                    asset = ticker
                    break
            
            result = self.get_trade_signals(asset)
            
            if result['success']:
                # Use LLM for conversational response if available
                if self.llm_available:
                    try:
                        context = {
                            'analysis_type': 'trade_signals',
                            'asset': result['asset'],
                            'opportunities': result['opportunities'],
                            'strategies': result['strategies']
                        }
                        response = self.llm.generate_conversational_response(message, context)
                        return response
                    except Exception as e:
                        print(f"⚠️ LLM response failed: {e}")
                
                # Fallback to basic response
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
                
                response += "\n💡 You can ask me follow-up questions like 'which stocks are overbought?' or 'show me the oversold ones'"
                
                return response
            else:
                return f"Sorry, I had trouble getting that information. Error: {result['error']}"
        
        # Check for overbought/oversold
        elif any(phrase in message_lower for phrase in ['overbought', 'oversold', 'overbought/oversold', 'scan stocks']):
            result = self.check_overbought_oversold()
            
            if result['success']:
                overbought = result['overbought']
                oversold = result['oversold']
                total = result['count']
                
                # Use LLM for conversational response if available
                if self.llm_available:
                    try:
                        context = {
                            'analysis_type': 'overbought_oversold',
                            'total_stocks': total,
                            'overbought_count': len(overbought),
                            'oversold_count': len(oversold),
                            'overbought_stocks': overbought[:10],  # Top 10
                            'oversold_stocks': oversold[:10]  # Top 10
                        }
                        response = self.llm.generate_conversational_response(message, context)
                        return response
                    except Exception as e:
                        print(f"⚠️ LLM response failed: {e}")
                
                # Fallback to basic response
                response = f"Found {total} stocks with signals:\n\n"
                response += f"📈 **Overbought ({len(overbought)} stocks):**\n"
                for stock in overbought[:5]:
                    response += f"- {stock['ticker']}: {stock['signal']}\n"
                if len(overbought) > 5:
                    response += f"... and {len(overbought) - 5} more overbought stocks\n"
                
                response += f"\n📉 **Oversold ({len(oversold)} stocks):**\n"
                for stock in oversold[:5]:
                    response += f"- {stock['ticker']}: {stock['signal']}\n"
                if len(oversold) > 5:
                    response += f"... and {len(oversold) - 5} more oversold stocks\n"
                
                response += "\n💡 You can ask me follow-up questions like 'show me the overbought ones' or 'which stocks are oversold?'"
                
                return response
            else:
                return f"Sorry, I had trouble getting that information. Error: {result['error']}"
        
        # Check for real-time analysis
        elif any(phrase in message_lower for phrase in ['real time', 'current price', 'right now', 'live', 'what is', 'doing']):
            for ticker in ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'TSLA', 'AMZN', 'META', 'NFLX', 'AMD', 'INTC']:
                if ticker.lower() in message_lower:
                    result = self.get_real_time_analysis(ticker)
                    
                    if result['success']:
                        if self.llm_available:
                            try:
                                context = {
                                    'analysis_type': 'real_time_analysis',
                                    'ticker': ticker,
                                    'analysis': result['analysis']
                                }
                                response = self.llm.generate_conversational_response(message, context)
                                return response
                            except Exception as e:
                                print(f"⚠️ LLM response failed: {e}")
                        return f"Here's the real-time analysis for {ticker}:\n\n{result['analysis']}"
                    else:
                        return f"Sorry, I had trouble getting that information. Error: {result['error']}"
        
        # Check for fundamental analysis
        elif any(phrase in message_lower for phrase in ['fundamentals', 'fundamental analysis', 'financials', 'earnings', 'revenue', 'profit', 'debt', 'valuation', 'pe ratio', 'balance sheet', 'income statement']):
            for ticker in ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'TSLA', 'AMZN', 'META', 'NFLX', 'AMD', 'INTC']:
                if ticker.lower() in message_lower:
                    result = self.get_fundamental_analysis(ticker)
                    
                    if result['success']:
                        if self.llm_available:
                            try:
                                context = {
                                    'analysis_type': 'fundamental_analysis',
                                    'ticker': ticker,
                                    'analysis': result['analysis']
                                }
                                response = self.llm.generate_conversational_response(message, context)
                                return response
                            except Exception as e:
                                print(f"⚠️ LLM response failed: {e}")
                        return f"Here's the fundamental analysis for {ticker}:\n\n{result['analysis']}"
                    else:
                        return f"Sorry, I had trouble getting that information. Error: {result['error']}"
        
        # Use LLM for general conversation if available
        if self.llm_available:
            try:
                # Create context for general conversation
                context = {
                    'analysis_type': 'general_conversation',
                    'user_message': message,
                    'conversation_history': self.conversation_history[-3:] if len(self.conversation_history) > 3 else self.conversation_history,
                    'last_analysis': self.last_analysis,
                    'last_stocks': self.last_stocks
                }
                
                # Generate conversational response
                response = self.llm.generate_conversational_response(message, context)
                return response
            except Exception as e:
                print(f"⚠️ LLM response failed: {e}")
        
        # Fallback to basic response
        return """Hey! I can help you with:

🔍 **Trade Signals**: "Check NVDA trade signals for the day"
📊 **Overbought/Oversold**: "What stocks are overbought right now?"
💰 **Real-time Analysis**: "What's NVDA doing right now?"

Just ask me naturally and I'll run the analysis for you! I can also answer follow-up questions about what I find."""

def main():
    """Main entry point"""
    print("🤖 Conversational QuantEngine Chat Interface")
    print("=" * 60)
    print("Type 'quit' to exit")
    print("💡 Try asking follow-up questions like 'show me the overbought ones'")
    print()
    
    chat = ConversationalChat()
    
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
