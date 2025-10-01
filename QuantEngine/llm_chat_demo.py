#!/usr/bin/env python3
"""
LLM Chat Demo

Demonstrates the conversational QuantEngine interface with Ollama LLM integration.
"""

import sys
from pathlib import Path

# Add QuantEngine to path
quant_engine_dir = Path(__file__).parent
sys.path.insert(0, str(quant_engine_dir))

from conversational_chat import ConversationalChat

def demo_llm_chat():
    """Demo the LLM-enhanced chat interface"""
    print("🤖 QuantEngine Conversational Chat with LLM Demo")
    print("=" * 60)
    print()
    
    chat = ConversationalChat()
    
    # Demo conversations
    demos = [
        "hey, could you check nvda trade signals for the day?",
        "what stocks are overbought right now?",
        "ok what is the over sold ones?",
        "show me the overbought stocks"
    ]
    
    for i, message in enumerate(demos, 1):
        print(f"Demo {i}:")
        print(f"👤 You: {message}")
        response = chat.chat(message)
        print(f"🤖 Bot: {response}")
        print("-" * 60)
        print()

if __name__ == "__main__":
    demo_llm_chat()

