#!/usr/bin/env python3
"""
Chat Interface Demo

Demonstrates the conversational QuantEngine interface.
"""

import sys
from pathlib import Path

# Add QuantEngine to path
quant_engine_dir = Path(__file__).parent
sys.path.insert(0, str(quant_engine_dir))

from simple_chat import SimpleChat

def demo_chat():
    """Demo the chat interface"""
    print("🤖 QuantEngine Chat Interface Demo")
    print("=" * 50)
    print()
    
    chat = SimpleChat()
    
    # Demo conversations
    demos = [
        "hey, could you check nvda trade signals for the day?",
        "what stocks are overbought right now?",
        "what's AAPL doing right now?",
        "check signals for TSLA"
    ]
    
    for i, message in enumerate(demos, 1):
        print(f"Demo {i}:")
        print(f"👤 You: {message}")
        print(f"🤖 Bot: {chat.chat(message)}")
        print("-" * 50)
        print()

if __name__ == "__main__":
    demo_chat()

