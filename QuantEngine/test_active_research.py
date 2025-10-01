#!/usr/bin/env python3
"""
Test script for the active research chat interface
"""

import asyncio
from active_research_chat import ActiveResearchChat

async def test_active_research():
    """Test the active research chat interface"""
    
    chat = ActiveResearchChat()
    await chat.initialize()
    
    # Test questions that require active research
    test_questions = [
        "evaluate nvda for price movement expectations over the next 30 days... will it go up, down, flat? i think it may range bound... but help me come up with a trading strategy for it",
        "what will happen with spy price over 30 days?",
        "analyze tqqq and give me a trading strategy",
        "research aapl stock outlook for 3 months"
    ]
    
    print("🧪 Testing Active Research QuantEngine Chat")
    print("=" * 60)
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n🔍 Test {i}: {question}")
        print("-" * 60)
        
        # Process question
        response = await chat.ask_question(question)
        formatted_response = chat.format_response(response)
        print(formatted_response)
        print("\n" + "=" * 60)
    
    await chat.close()

if __name__ == "__main__":
    asyncio.run(test_active_research())

