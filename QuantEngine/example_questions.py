#!/usr/bin/env python3
"""
Example Questions for QuantEngine Chat Interface

This file contains example research questions that demonstrate the capabilities
of the QuantEngine conversational interface.
"""

EXAMPLE_QUESTIONS = {
    "market_analysis": [
        "How will the Fed interest rate decision impact housing prices in 6 months?",
        "What's the outlook for tech stocks given current market conditions?",
        "How will inflation affect different sectors over the next year?",
        "What are the key risks for the market in the next 3 months?",
        "How will the upcoming election impact market volatility?",
        "What's the correlation between oil prices and energy stocks?",
        "How will trade tensions affect specific sectors?",
        "What's the impact of a potential recession on different assets?",
        "How will the next jobs report impact the market?",
        "What's the outlook for small cap vs large cap stocks?"
    ],
    
    "stock_research": [
        "Research NFLX company for stock price outlook 3 months from now",
        "Analyze AAPL stock performance under different scenarios",
        "What are the key risks for TSLA stock?",
        "How will GOOGL earnings impact the stock price?",
        "What's the fair value range for MSFT stock?",
        "Analyze AMZN stock in different market regimes",
        "What are the growth prospects for NVDA?",
        "How will META's metaverse strategy affect the stock?",
        "What's the impact of regulatory changes on big tech stocks?",
        "Analyze the competitive position of CRM vs competitors"
    ],
    
    "sector_analysis": [
        "How will rising interest rates impact the financial sector?",
        "What's the outlook for energy stocks in 2024?",
        "How will AI developments affect the technology sector?",
        "What's the impact of healthcare reform on pharma stocks?",
        "How will climate change policies affect utilities?",
        "What's the outlook for real estate REITs?",
        "How will consumer spending trends affect retail stocks?",
        "What's the impact of supply chain issues on industrials?",
        "How will demographic trends affect healthcare stocks?",
        "What's the outlook for materials and commodities?"
    ],
    
    "economic_events": [
        "How will the next jobs report impact the market?",
        "What's the impact of a potential recession on different assets?",
        "How will trade tensions affect specific sectors?",
        "What's the impact of infrastructure spending on different sectors?",
        "How will climate change policies affect the economy?",
        "What's the impact of demographic shifts on different sectors?",
        "How will geopolitical tensions affect markets?",
        "What's the impact of currency movements on different assets?",
        "How will regulatory changes affect specific industries?",
        "What's the impact of technological disruption on traditional sectors?"
    ],
    
    "risk_analysis": [
        "What are the tail risks for the current market?",
        "How would a 2008-style crisis affect different sectors?",
        "What's the impact of a sudden interest rate spike?",
        "How would a major cyber attack affect tech stocks?",
        "What's the impact of a major natural disaster?",
        "How would a trade war escalation affect markets?",
        "What's the impact of a major corporate scandal?",
        "How would a pandemic resurgence affect different sectors?",
        "What's the impact of a major geopolitical event?",
        "How would a major technology disruption affect markets?"
    ],
    
    "portfolio_construction": [
        "How should I allocate between growth and value stocks?",
        "What's the optimal sector allocation for the current environment?",
        "How should I hedge against market downturns?",
        "What's the best way to gain exposure to AI trends?",
        "How should I position for rising interest rates?",
        "What's the optimal allocation between US and international stocks?",
        "How should I balance risk and return in the current market?",
        "What's the best way to generate income in a low-rate environment?",
        "How should I position for inflation protection?",
        "What's the optimal allocation between stocks and bonds?"
    ]
}

def get_random_question(category: str = None) -> str:
    """Get a random question from a specific category or all categories"""
    import random
    
    if category and category in EXAMPLE_QUESTIONS:
        return random.choice(EXAMPLE_QUESTIONS[category])
    else:
        all_questions = []
        for questions in EXAMPLE_QUESTIONS.values():
            all_questions.extend(questions)
        return random.choice(all_questions)

def get_questions_by_category(category: str) -> list:
    """Get all questions from a specific category"""
    return EXAMPLE_QUESTIONS.get(category, [])

def get_all_categories() -> list:
    """Get all available question categories"""
    return list(EXAMPLE_QUESTIONS.keys())

def print_question_categories():
    """Print all available question categories with examples"""
    print("📚 QuantEngine Research Question Categories")
    print("=" * 60)
    
    for category, questions in EXAMPLE_QUESTIONS.items():
        print(f"\n🔍 {category.replace('_', ' ').title()}")
        print("-" * 40)
        for i, question in enumerate(questions[:3], 1):  # Show first 3 examples
            print(f"{i}. {question}")
        if len(questions) > 3:
            print(f"... and {len(questions) - 3} more questions")

if __name__ == "__main__":
    print_question_categories()
    
    print(f"\n🎲 Random Question:")
    print(f"'{get_random_question()}'")
    
    print(f"\n🎯 Random Market Analysis Question:")
    print(f"'{get_random_question('market_analysis')}'")

