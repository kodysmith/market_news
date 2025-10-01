#!/usr/bin/env python3
"""
Test script for the simple chat interface
"""

from simple_chat import SimpleQuantChat

def test_chat():
    """Test the chat interface with example questions"""
    
    chat = SimpleQuantChat()
    
    # Test questions
    test_questions = [
        "How will Fed rates impact housing prices in 6 months?",
        "Research NFLX company for stock price outlook 3 months from now",
        "What's the outlook for tech stocks given current market conditions?"
    ]
    
    print("🧪 Testing QuantEngine Chat Interface")
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
    test_chat()

