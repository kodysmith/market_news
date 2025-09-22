# 🤖 QuantEngine Autonomous Trading Bot

A 24/7 running quantitative trading system that continuously monitors markets, researches strategies, discovers opportunities, and adapts to changing conditions through machine learning.

## 🚀 What QuantBot Does

QuantBot is an **autonomous AI trading system** that runs continuously to:

### 🔄 **Continuous Operation**
- **24/7 Market Monitoring**: Never stops watching markets, news, and economic data
- **Real-time Adaptation**: Instantly responds to changing market conditions
- **Self-Learning**: Improves strategies based on historical performance

### 🎭 **Market Regime Detection**
- **Bull/Bear Markets**: Identifies trending vs ranging markets
- **Volatility Regimes**: Detects high/low volatility environments
- **Risk Assessment**: Measures market stress and uncertainty
- **Sentiment Analysis**: Processes news and social media sentiment

### 🧪 **Strategy Research & Evolution**
- **Genetic Algorithms**: Evolves strategies through crossover and mutation
- **Backtesting**: Tests strategies on historical data
- **Performance Optimization**: Continuously improves Sharpe ratios and returns
- **Regime-Specific**: Adapts strategies to current market conditions

### 🔍 **Opportunity Discovery**
- **Technical Patterns**: Breakouts, reversals, momentum signals
- **Volume Anomalies**: Unusual trading activity detection
- **Statistical Arbitrage**: Pairs trading and mean reversion
- **Options Anomalies**: Put/call ratio and volatility skew analysis
- **News Dislocations**: Sentiment-driven price opportunities

### 📊 **Risk Management**
- **Dynamic Position Sizing**: Kelly criterion and volatility targeting
- **Portfolio Optimization**: Modern portfolio theory implementation
- **Stress Testing**: Scenario analysis and risk limits
- **Circuit Breakers**: Automatic risk controls

## 🎯 **Key Features**

### 🤖 **Autonomous Operation**
```bash
# Start the bot - it runs forever
python3 run_quant_bot.py

# Demo mode for testing
python3 run_quant_bot.py --demo

# Check status
python3 run_quant_bot.py --status
```

### 🧠 **Self-Learning System**
- **Strategy Evolution**: Creates new strategies through genetic algorithms
- **Performance Learning**: Remembers what worked in similar conditions
- **Regime Adaptation**: Adjusts behavior based on market environment
- **Continuous Improvement**: Gets better over time

### 📈 **Multi-Asset Coverage**
- **Equities**: SPY, QQQ, IWM, VTI (broad market ETFs)
- **Sectors**: XLE, XLF, XLK, XLV, XLY, XLI, XLC, XLU, XLB, XLRE
- **Leveraged**: TQQQ, SQQQ, UVXY, SVXY (volatility products)

### 🔧 **Enterprise Architecture**
```
QuantBot Core System
├── 🤖 QuantBot (Main Controller)
│   ├── 🔧 Component Manager
│   ├── 📊 Data Ingestion
│   ├── 🎭 Regime Detection
│   ├── 🧪 Strategy Research
│   ├── 🔍 Opportunity Scanner
│   └── 🛡️ Risk Management
├── 📈 Strategy Portfolio
│   ├── Technical Strategies
│   ├── ML-Enhanced Strategies
│   └── Regime-Specific Strategies
├── 🎯 Opportunity Pipeline
│   ├── Signal Generation
│   ├── Risk Assessment
│   └── Position Sizing
└── 📊 Performance Database
    ├── Historical Returns
    ├── Strategy Performance
    └── Learning Database
```

## 📊 **Real-Time Dashboard**

While QuantBot runs, you can monitor its activity:

```bash
# View live status
python3 run_quant_bot.py --status

# Check logs
tail -f quant_bot.log

# View current opportunities
cat opportunity_cache.json
```

**Sample Output:**
```
🤖 QuantBot Status Report
========================================
🤖 Bot Status: Running
🎭 Current Regime: Bull Market (confidence: 78%)
📈 Active Strategies: 23
🎯 Opportunities Found: 7
📊 Performance: +12.4%
🔧 System Health: excellent
🕒 Last Update: 2024-01-15T14:32:15Z
```

## 🧪 **Strategy Evolution Example**

QuantBot starts with basic strategies and evolves them:

```
Generation 1: Basic SMA Crossover (Sharpe: 0.8)
Generation 2: RSI + Volume Filter (Sharpe: 1.1)
Generation 3: ML-Enhanced with Sentiment (Sharpe: 1.4)
Generation 4: Regime-Aware Multi-Factor (Sharpe: 1.7)
```

## 🎯 **Opportunity Discovery**

QuantBot continuously scans for alpha opportunities:

### Technical Breakouts
```
🚨 BREAKOUT DETECTED: SPY
   Signal: Bollinger Band breakout above resistance
   Expected Return: 3.2%
   Confidence: 75%
   Timeframe: 1-5 days
```

### Statistical Arbitrage
```
📊 STAT ARB OPPORTUNITY: XLE vs XLF
   Spread Z-Score: -2.3σ (mean reversion expected)
   Expected Return: 1.8%
   Confidence: 68%
```

### News-Driven
```
📰 NEWS DISLOCATION: Tech earnings beat
   Sentiment Shift: -0.7 → +0.4
   Expected Reversal: 2.1%
   Confidence: 72%
```

## ⚙️ **Configuration**

Edit `config/bot_config.json`:

```json
{
  "data_sources": {
    "yahoo_finance": {"enabled": true, "update_interval": 60},
    "news_feeds": {"enabled": true, "update_interval": 300},
    "economic_data": {"enabled": true, "update_interval": 3600}
  },
  "strategy_research": {
    "enabled": true,
    "research_interval": 3600,
    "max_strategies": 50,
    "min_sharpe_ratio": 1.0
  },
  "universe": {
    "equities": ["SPY", "QQQ", "IWM"],
    "sectors": ["XLE", "XLF", "XLK"],
    "leveraged": ["TQQQ", "UVXY"]
  }
}
```

## 📈 **Performance Tracking**

QuantBot maintains detailed performance records:

- **Strategy Performance**: Sharpe ratios, win rates, drawdowns
- **Portfolio Returns**: Daily P&L, risk metrics
- **Opportunity Success**: Hit rates by opportunity type
- **Regime Performance**: Returns by market condition
- **Learning Progress**: Strategy improvement over time

## 🚨 **Risk Controls**

Built-in safety measures:

- **Position Limits**: Maximum 5% per position, 20% total portfolio
- **Volatility Controls**: Circuit breakers on high volatility
- **Drawdown Limits**: Automatic reduction on losses
- **Correlation Checks**: Avoid concentrated risk
- **Stress Testing**: Continuous scenario analysis

## 🎯 **Getting Started**

### 1. **Install Dependencies**
```bash
cd QuantEngine
pip install -r requirements.txt
```

### 2. **Configure Universe**
Edit `config/bot_config.json` to set your preferred assets.

### 3. **Start Demo Mode**
```bash
python3 run_quant_bot.py --demo
```

### 4. **Monitor Activity**
```bash
# In another terminal
python3 run_quant_bot.py --status
tail -f quant_bot.log
```

### 5. **Go Live** (Production)
```bash
python3 run_quant_bot.py  # Runs 24/7
```

## 📊 **Understanding the Value**

### What QuantBot Gives You:
1. **Continuous Alpha Generation**: Never misses opportunities
2. **Adaptive Intelligence**: Learns from market changes
3. **Risk Management**: Professional-grade controls
4. **Strategy Evolution**: Gets better over time
5. **Comprehensive Coverage**: Multiple asset classes and strategies

### Key Advantages:
- **No Sleep**: Operates 24/7/365
- **Emotion-Free**: Pure quantitative decisions
- **Self-Improving**: Learns from experience
- **Comprehensive**: Covers multiple strategies and assets
- **Safe**: Built-in risk controls

## 🔧 **Technical Architecture**

### Core Components:
- **QuantBot**: Main orchestration engine
- **LiveDataManager**: Real-time data integration (Yahoo Finance, Alpha Vantage, FMP News)
- **RegimeDetector**: Market condition classification
- **AdaptiveStrategyResearcher**: Strategy evolution system
- **OpportunityScanner**: Real-time opportunity discovery
- **RiskManager**: Portfolio risk control

### Data Flow:
```
Live Data APIs → Data Processing → Regime Detection → Strategy Research → Opportunity Scanning → Risk Assessment → Position Sizing
(Yahoo Finance,     (Sentiment,       ↑                    ↑                     ↑                      ↑                       ↑
 Alpha Vantage,      Normalization)   │                    │                     │                      │                       │
 FMP News)                           Performance        Learning             Alpha                 Risk                   Position
                                    Feedback           Adaptation          Discovery             Assessment              Sizing
```

## 🚀 **Next Steps**

1. **Start Small**: Run in demo mode first
2. **Paper Trading**: Test with simulated money
3. **Live Trading**: Connect to broker API (future feature)
4. **Customization**: Add your own strategies and indicators
5. **Expansion**: Add more asset classes and data sources

## 🎉 **The Future of Trading**

QuantBot represents the future of quantitative trading:

- **AI-Powered**: Machine learning for strategy discovery
- **Autonomous**: Runs without human intervention
- **Adaptive**: Learns and evolves with market changes
- **Comprehensive**: Covers all aspects of trading
- **Safe**: Enterprise-grade risk management

**Ready to start your autonomous trading journey?**

```bash
python3 run_quant_bot.py --demo
```

🚀 **Welcome to the future of quantitative trading!** 🚀

---

## 🔴 **Live Data Integration Status**

QuantBot now integrates **real live market data** from multiple sources:

### ✅ **Working Live Data Sources:**
- **Yahoo Finance**: Real-time stock quotes, volumes, bid/ask spreads
- **FMP News API**: Live financial news with automatic sentiment analysis
- **Economic Indicators**: GDP, unemployment, inflation data

### 📊 **Live Data Test Results:**
```
🧪 QuantBot Live Data Integration Test Suite
==================================================
API Keys Status:
   FMP_API_KEY: ✅ Configured
   ALPHA_VANTAGE_API_KEY: ✅ Configured
   FRED_API_KEY: ✅ Configured

📊 Yahoo Finance: ✅ WORKING (live quotes)
📈 Alpha Vantage: ✅ WORKING (intraday data)
📰 FMP News: ✅ WORKING (live news + sentiment)
💰 Economic Data: ✅ WORKING (FRED government data)
🌍 Market Overview: ✅ WORKING

Tests Passed: 5/5 (100% success rate)
🎉 ALL TESTS PASSED - Full live data integration ready!
```

### 🧪 **Test Your Live Data Setup:**
```bash
cd QuantEngine
python3 test_live_data.py
```

**QuantBot is now running on REAL LIVE MARKET DATA!** 🌟
