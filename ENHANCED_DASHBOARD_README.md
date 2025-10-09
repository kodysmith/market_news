# 🤖 Enhanced Market Scanner Dashboard with AI Chat

A comprehensive Streamlit dashboard that combines real-time market analysis with AI-powered conversational research capabilities.

## 🚀 Features

### 📊 Market Analysis
- **Expanded Symbol Coverage**: NVDA, NFLX, GOOG, MSFT, FDX, JNJ, DV, SPY, TQQQ, QQQ, TSLA, HD, KO
- **Real-time Market Data**: Live prices, changes, and performance metrics
- **Trading Opportunities**: AI-identified trading setups with confidence scores
- **Technical Analysis**: RSI, support/resistance, risk/reward ratios
- **Performance Metrics**: Portfolio-level analytics and insights

### 🤖 AI Chat Assistant
- **Sector Research**: Analyze specific sectors (Technology, Healthcare, Financials, etc.)
- **Group Analysis**: Compare stock groups and portfolios
- **Correlation Analysis**: Understand relationships between stocks
- **Risk Assessment**: Comprehensive risk profiling
- **Trading Recommendations**: AI-powered strategy suggestions

### 💬 Conversational Interface
- **Natural Language Queries**: Ask questions in plain English
- **Context-Aware Responses**: AI understands market context
- **Interactive Examples**: Pre-built question templates
- **Chat History**: Persistent conversation memory
- **Real-time Analysis**: Live data integration

## 🏗️ Architecture

```
Enhanced Dashboard
├── Market Overview Tab
│   ├── Real-time Data Display
│   ├── Trading Opportunities Table
│   ├── Performance Metrics
│   └── Expanded Symbol Coverage
├── AI Chat Assistant Tab
│   ├── Sector Research Interface
│   ├── Group Analysis Interface
│   ├── Chat History
│   └── Example Questions
└── Real-Time Data Tab
    ├── Technology & Growth Stocks
    ├── Market Indices & ETFs
    └── Diverse Sector Stocks
```

## 🚀 Quick Start

### Prerequisites
```bash
# Install required packages
pip install streamlit plotly yfinance pandas numpy requests

# Ensure QuantEngine dependencies are installed
cd QuantEngine
pip install -r requirements_enhanced.txt
```

### Launch Dashboard
```bash
# Option 1: Use the launcher script
python run_enhanced_dashboard.py

# Option 2: Run directly with Streamlit
streamlit run QuantEngine/enhanced_dashboard_with_chat.py
```

### Access Dashboard
- **URL**: http://localhost:8501
- **Port**: 8501 (configurable)
- **Network Access**: Available on all interfaces (0.0.0.0)

## 📊 Symbol Coverage

### Technology & Growth
- **NVDA**: NVIDIA Corporation
- **NFLX**: Netflix Inc.
- **GOOG**: Alphabet Inc. (Google)
- **MSFT**: Microsoft Corporation
- **TSLA**: Tesla Inc.

### Market Indices & ETFs
- **SPY**: SPDR S&P 500 ETF Trust
- **TQQQ**: ProShares UltraPro QQQ
- **QQQ**: Invesco QQQ Trust

### Diverse Sectors
- **FDX**: FedEx Corporation (Logistics)
- **JNJ**: Johnson & Johnson (Healthcare)
- **DV**: DoubleVerify Holdings Inc. (Technology)
- **HD**: The Home Depot Inc. (Retail)
- **KO**: The Coca-Cola Company (Consumer Staples)

## 🤖 AI Chat Capabilities

### Sector Research Examples
```
"Research the technology sector outlook for the next 3 months"
"Compare healthcare and financial sectors performance"
"What's the risk profile of the energy sector?"
"Analyze semiconductor stocks in the technology sector"
```

### Group Analysis Examples
```
"Analyze NVDA, MSFT, GOOGL for correlation analysis"
"Compare mega cap tech vs dividend aristocrats"
"What's the risk profile of the fintech group?"
"Research TSLA, NVDA, AMD for momentum analysis"
```

### Advanced Queries
```
"How will the Fed decision impact the financial sector?"
"Compare the performance of growth vs value stocks"
"Analyze the correlation between tech stocks and market volatility"
"What are the best performing stocks in the energy sector?"
```

## 🔧 Configuration

### Symbol Configuration
Update `data/config.json` to modify the symbol list:
```json
{
    "GAMMA_SCALPING_TICKERS": [
        "NVDA", "NFLX", "GOOG", "MSFT", "FDX", "JNJ", "DV", 
        "SPY", "TQQQ", "QQQ", "TSLA", "HD", "KO", "AAPL", 
        "AMZN", "META", "IWM"
    ]
}
```

### Chat Configuration
The AI chat interfaces use the following models:
- **LLM Model**: qwen2.5:72b (via Ollama)
- **Temperature**: 0.3 (focused responses)
- **Max Tokens**: 1500-2000 (comprehensive analysis)

## 📈 Dashboard Tabs

### 1. Market Overview
- **Key Metrics**: Tickers scanned, opportunities found, system status
- **Market Sentiment**: Bullish/Bearish/Neutral with confidence scores
- **Trading Opportunities**: AI-identified setups with risk/reward ratios
- **Performance Metrics**: Portfolio-level analytics

### 2. AI Chat Assistant
- **Sector Research**: Analyze specific market sectors
- **Group Analysis**: Compare stock groups and portfolios
- **Interactive Chat**: Natural language queries
- **Example Questions**: Pre-built templates for common queries

### 3. Real-Time Data
- **Live Prices**: Current market data for all symbols
- **Price Changes**: Real-time percentage changes
- **Organized Display**: Grouped by sector and type
- **Auto-refresh**: Updates every time the tab is accessed

## 🛠️ Technical Details

### Dependencies
```python
streamlit>=1.28.0
plotly>=5.15.0
yfinance>=0.2.18
pandas>=2.0.0
numpy>=1.24.0
requests>=2.31.0
```

### Data Sources
- **Yahoo Finance**: Real-time market data
- **QuantEngine**: AI analysis and insights
- **Ollama**: Local LLM for chat responses

### Performance
- **Real-time Updates**: Live data refresh
- **Caching**: Efficient data storage and retrieval
- **Async Processing**: Non-blocking AI analysis
- **Responsive UI**: Optimized for different screen sizes

## 🔍 Troubleshooting

### Common Issues

1. **Chat Not Available**
   ```
   Error: Chat interfaces not available
   Solution: Ensure QuantEngine dependencies are installed
   ```

2. **No Scanner Data**
   ```
   Error: No scanner data available
   Solution: Run the enhanced scanner first to generate data
   ```

3. **Real-time Data Failed**
   ```
   Error: Failed to load real-time data
   Solution: Check internet connection and Yahoo Finance API access
   ```

### Debug Mode
```bash
# Run with debug information
streamlit run QuantEngine/enhanced_dashboard_with_chat.py --logger.level debug
```

## 📚 API Integration

The dashboard integrates with the following APIs:
- **Chat API**: `http://localhost:5001` (if running separately)
- **Yahoo Finance**: Real-time market data
- **QuantEngine**: AI analysis and insights

## 🎯 Use Cases

### For Traders
- **Real-time Analysis**: Live market data and opportunities
- **AI Insights**: Intelligent market analysis and recommendations
- **Risk Management**: Comprehensive risk assessment tools

### For Researchers
- **Sector Analysis**: Deep dive into specific market sectors
- **Correlation Studies**: Understand relationships between stocks
- **Portfolio Analysis**: Comprehensive group and portfolio insights

### For Developers
- **API Integration**: Easy integration with existing systems
- **Customizable**: Configurable symbols and analysis parameters
- **Extensible**: Modular design for easy feature additions

## 🚀 Future Enhancements

- **Voice Input**: Speech-to-text for chat interface
- **Mobile Optimization**: Enhanced mobile experience
- **Advanced Charts**: More sophisticated visualization options
- **Portfolio Tracking**: Real-time portfolio monitoring
- **Alert System**: Customizable market alerts

## 📞 Support

For issues or questions:
1. Check the troubleshooting section
2. Review the configuration settings
3. Ensure all dependencies are installed
4. Check the QuantEngine documentation

---

**Happy Trading! 📈🤖**


