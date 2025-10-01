# QuantEngine Chat Interface

A conversational AI interface that leverages the QuantEngine to answer research questions and generate comprehensive reports with scenario analysis and probabilities.

## Features

- **Natural Language Research Questions**: Ask questions in plain English about markets, stocks, sectors, and economic events
- **Scenario Analysis**: Get multiple scenarios with probabilities for different outcomes
- **Monte Carlo Simulations**: Advanced statistical modeling for price predictions
- **Market Context Integration**: Leverages current market regime and sentiment data
- **Comprehensive Reports**: Detailed analysis with recommendations

## Quick Start

### Basic Usage

```bash
# Interactive mode
python run_chat.py

# Single question
python run_chat.py --question "How will Fed rates impact housing prices in 6 months?"

# Advanced mode with Monte Carlo analysis
python run_chat.py --advanced --question "Research NFLX stock outlook for 3 months"
```

### Example Questions

**Market Analysis:**
- "How will the Fed interest rate decision impact housing prices in 6 months?"
- "What's the outlook for tech stocks given current market conditions?"
- "How will inflation affect different sectors over the next year?"

**Stock Research:**
- "Research NFLX company for stock price outlook 3 months from now"
- "Analyze AAPL stock performance under different scenarios"
- "What are the key risks for TSLA stock?"

**Sector Analysis:**
- "How will rising interest rates impact the financial sector?"
- "What's the outlook for energy stocks in 2024?"
- "How will AI developments affect the technology sector?"

## Architecture

### Core Components

1. **QuantResearchChat**: Main conversational interface
2. **AdvancedResearchAgent**: Sophisticated modeling with Monte Carlo simulations
3. **Market Context Integration**: Real-time regime detection and sentiment analysis
4. **Scenario Generator**: Multiple outcome scenarios with probabilities

### Data Sources

- **Live Market Data**: Yahoo Finance, Alpha Vantage, FMP News
- **Economic Indicators**: FRED data, economic calendar
- **Sentiment Analysis**: News sentiment, social media sentiment
- **Regime Detection**: Bull/bear markets, volatility regimes

## Example Output

```
# Research Analysis: How will Fed rates impact housing prices in 6 months?

**Analysis Type:** Impact Analysis
**Time Horizon:** 6 months
**Confidence Score:** 78%

## Market Context
- **Current Regime:** bull_market (confidence: 75%)
- **Market Sentiment:** 0.23
- **Volatility Level:** 18.5

## Scenario Analysis

### Scenario 1: Rate Hike Scenario
**Probability:** 35%
**Description:** Fed raises rates by 0.25-0.5%
**Impact:**
- Housing Prices: Decline 2-5% over 6 months
- Mortgage Rates: Increase 0.3-0.6%
- Housing Demand: Decrease due to higher borrowing costs
- REIT Performance: Negative impact on REITs

**Key Drivers:**
- Higher borrowing costs reduce affordability
- Reduced demand from first-time buyers
- Potential slowdown in construction activity

**Confidence:** 75%

### Scenario 2: Rate Hold Scenario
**Probability:** 45%
**Description:** Fed maintains current rates
**Impact:**
- Housing Prices: Stable to slight increase 1-3%
- Mortgage Rates: Remain relatively stable
- Housing Demand: Maintain current levels
- REIT Performance: Neutral to slightly positive

**Key Drivers:**
- Continued demand from demographic trends
- Limited supply supporting prices
- Stable financing environment

**Confidence:** 80%

### Scenario 3: Rate Cut Scenario
**Probability:** 20%
**Description:** Fed cuts rates by 0.25-0.5%
**Impact:**
- Housing Prices: Increase 3-7% over 6 months
- Mortgage Rates: Decrease 0.2-0.4%
- Housing Demand: Increase due to lower borrowing costs
- REIT Performance: Positive impact on REITs

**Key Drivers:**
- Lower borrowing costs increase affordability
- Increased demand from buyers
- Potential boost to construction activity

**Confidence:** 70%

## Recommendations
1. Monitor Fed communications closely for rate guidance
2. Consider defensive positioning in housing-related investments
3. Focus on REITs with strong balance sheets and diversified portfolios
4. Watch for signs of economic slowdown that could trigger rate cuts
```

## Advanced Features

### Monte Carlo Analysis

The advanced mode includes Monte Carlo simulations for more sophisticated price modeling:

```python
# Generate 10,000 Monte Carlo simulations
mc_results = await agent.generate_monte_carlo_scenarios(
    'NFLX', '3 months', market_context, n_simulations=10000
)

# Results include:
# - Price distribution percentiles
# - Probability of different scenarios
# - Statistical measures (skewness, kurtosis)
# - Regime-adjusted modeling
```

### Sector Impact Analysis

Analyze how specific events affect different sectors:

```python
# Analyze Fed decision impact on tech sector
sector_analysis = await agent.analyze_sector_impact(
    'tech', 'fed_decision', '6 months', market_context
)
```

## Configuration

The chat interface uses the same configuration as the main QuantEngine:

```json
{
  "data_sources": {
    "yahoo_finance": {"enabled": true, "update_interval": 60},
    "news_feeds": {"enabled": true, "update_interval": 300},
    "economic_data": {"enabled": true, "update_interval": 3600}
  },
  "universe": {
    "equities": ["SPY", "QQQ", "IWM", "VTI", "BND"],
    "sectors": ["XLE", "XLF", "XLK", "XLV", "XLY", "XLI", "XLC", "XLU", "XLB", "XLRE"],
    "leveraged": ["TQQQ", "SQQQ", "UVXY", "SVXY"]
  }
}
```

## Integration with QuantEngine

The chat interface seamlessly integrates with existing QuantEngine components:

- **LiveDataManager**: Real-time market data and news
- **MarketRegimeDetector**: Current market regime analysis
- **OpportunityScanner**: Trading opportunity discovery
- **FeatureBuilder**: Technical indicators and signals
- **SentimentSignalGenerator**: News and sentiment analysis

## Usage Examples

### Interactive Mode

```bash
python run_chat.py --interactive
```

```
🤖 QuantEngine Research Chat
==================================================
Ask research questions about markets, stocks, sectors, and economic events.
Type 'help' for examples, 'quit' to exit.

💬 Your question: How will AI affect tech stocks?
```

### Single Question Mode

```bash
python run_chat.py --question "Research NFLX for 3 months" --advanced
```

### Batch Processing

```python
import asyncio
from chat_interface import QuantResearchChat

async def batch_analysis():
    chat = QuantResearchChat()
    await chat.initialize()
    
    questions = [
        "How will Fed rates impact housing?",
        "What's the outlook for tech stocks?",
        "Analyze NFLX stock performance"
    ]
    
    for question in questions:
        response = await chat.ask_question(question)
        print(chat.format_response(response))
        print("\n" + "="*80 + "\n")

asyncio.run(batch_analysis())
```

## Dependencies

- Python 3.8+
- pandas
- numpy
- scipy
- asyncio
- aiohttp
- QuantEngine components

## Future Enhancements

- **GPT Integration**: Connect with OpenAI GPT or similar for more sophisticated natural language processing
- **Voice Interface**: Add voice input/output capabilities
- **Real-time Updates**: Live market data streaming during conversations
- **Portfolio Integration**: Connect with actual portfolio data for personalized analysis
- **Alert System**: Set up alerts for specific market conditions or price targets

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all QuantEngine components are properly installed
2. **Data Connection**: Check API keys for Yahoo Finance, Alpha Vantage, FMP
3. **Memory Issues**: Reduce Monte Carlo simulation count for large analyses
4. **Async Issues**: Ensure proper async/await usage in custom code

### Debug Mode

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
python run_chat.py --question "Your question here"
```

## Contributing

To add new question types or analysis capabilities:

1. Extend the `parse_question` method in `QuantResearchChat`
2. Add new scenario generators in `AdvancedResearchAgent`
3. Update the example questions in `example_questions.py`
4. Add tests for new functionality

## License

Same as QuantEngine project license.

