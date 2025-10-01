#!/usr/bin/env python3
"""
Test script for the enhanced research chat interface
"""

from enhanced_research_chat import EnhancedResearchChat

def test_enhanced_research():
    """Test the enhanced research chat interface"""
    
    chat = EnhancedResearchChat()
    
    # Test questions that will benefit from enhanced analysis
    test_questions = [
        "evaluate nvda for price movement expectations over the next 30 days... will it go up, down, flat? i think it may range bound... but help me come up with a trading strategy for it",
        "analyze spy and give me a comprehensive trading strategy with risk management",
        "research aapl stock outlook for 3 months with detailed technical analysis"
    ]
    
    print("🧪 Testing Enhanced Research QuantEngine Chat")
    print("=" * 60)
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n🔍 Test {i}: {question}")
        print("-" * 60)
        
        # Process question
        response = chat.ask_question(question)
        formatted_response = chat.format_response(response)
        print(formatted_response)
        print("\n" + "=" * 60)

if __name__ == "__main__":
    test_enhanced_research()

