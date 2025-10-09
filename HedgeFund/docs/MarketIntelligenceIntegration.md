# Market Intelligence Service Integration

## Overview

Integrates QuantEngine's news analysis capabilities into the Hedge Fund trading system to provide **adaptive hedging based on real-time market sentiment and news events**.

---

## Architecture

### New Module: Market Intelligence Service

**File**: `src/intelligence/market_intelligence_service.py`

**Purpose**: Monitor news 24/7, analyze sentiment, and generate risk alerts

**Data Flow**:
```
FMP News API → News Fetcher → LLM Analysis → Risk Signals → Hedge Position Adjustments
      ↓              ↓              ↓              ↓
   Database    Sentiment DB   Risk Events    Trading Engine
```

---

## Integration Points

### 1. News Data Sources

**Primary**: Financial Modeling Prep (FMP) API
- General market news
- Company-specific news
- Economic data releases
- Earnings announcements

**Backup**: Yahoo Finance News (via yfinance)

### 2. LLM Analysis (Ollama)

**Models**:
- `llama3:latest` - General analysis
- `mixtral:8x7b` - Deep reasoning (if available)

**Analysis Types**:
1. **Market Sentiment** - Overall bullish/bearish/neutral
2. **Risk Events** - Identify potential hedging triggers
3. **Volatility Forecast** - Expected vol changes
4. **Sector Rotation** - Which sectors to hedge
5. **Event Impact** - Earnings, Fed, geopolitical

### 3. Risk Signal Generation

**Output**: Risk signals that influence hedging decisions

```python
@dataclass
class RiskSignal:
    timestamp: datetime
    risk_level: str  # "LOW", "MEDIUM", "HIGH", "CRITICAL"
    risk_type: str  # "volatility", "drawdown", "sector", "market"
    description: str
    recommended_action: str  # "increase_hedge", "decrease_hedge", "hold"
    confidence: float  # 0.0 to 1.0
    affected_assets: List[str]  # ["SPY", "QQQ", etc.]
    hedge_adjustment_pct: float  # -50% to +100%
```

---

## Implementation Plan

### Phase 1: Basic News Monitoring (2-3 hours)

1. **Create Market Intelligence Service** (`src/intelligence/market_intelligence_service.py`)
   - Fetch news from FMP API
   - Store in PostgreSQL (new `news_feed` table)
   - Basic sentiment scoring

2. **Database Schema Extension**
   ```sql
   CREATE TABLE news_feed (
       id TEXT PRIMARY KEY,
       headline TEXT NOT NULL,
       source TEXT,
       url TEXT,
       content TEXT,
       sentiment TEXT,  -- bullish/bearish/neutral
       sentiment_score FLOAT,
       tickers TEXT[],
       impact_level TEXT,  -- low/medium/high/critical
       published_date TIMESTAMP,
       created_at TIMESTAMP DEFAULT NOW()
   );
   
   CREATE TABLE risk_events (
       id TEXT PRIMARY KEY,
       timestamp TIMESTAMP NOT NULL,
       risk_level TEXT NOT NULL,
       risk_type TEXT NOT NULL,
       description TEXT,
       recommended_action TEXT,
       confidence FLOAT,
       affected_assets TEXT[],
       hedge_adjustment_pct FLOAT,
       created_at TIMESTAMP DEFAULT NOW()
   );
   ```

3. **Configuration Updates** (`config/config.yaml`)
   ```yaml
   intelligence:
     enabled: true
     update_interval: 300  # 5 minutes
     
     news:
       fmp_api_key: ${FMP_API_KEY}
       fetch_limit: 50
       lookback_hours: 24
       
     llm:
       enabled: true
       model: "llama3:latest"
       temperature: 0.3
       
     risk_detection:
       high_impact_keywords:
         - "fed"
         - "fomc"
         - "recession"
         - "inflation"
         - "rate cut"
         - "rate hike"
         - "earnings miss"
         - "guidance cut"
       
       volatility_triggers:
         - "vix spike"
         - "volatility"
         - "market crash"
         - "sell-off"
   ```

### Phase 2: LLM Integration (2-3 hours)

1. **Integrate Ollama LLM**
   - Adapt QuantEngine's `llm_integration.py`
   - Create news summarization
   - Extract market sentiment
   - Identify risk events

2. **Risk Signal Generator**
   - Parse LLM output for risk indicators
   - Generate `RiskSignal` objects
   - Store in database

### Phase 3: Trading Engine Integration (3-4 hours)

1. **Modify Signal Generator** (`src/strategies/signal_generator.py`)
   - Add `generate_adaptive_hedges()` method
   - Query recent risk signals
   - Adjust hedge sizing based on sentiment

2. **Hedge Adjustment Logic**
   ```python
   def calculate_hedge_size(self, base_hedge_size: int, risk_signals: List[RiskSignal]) -> int:
       """
       Adjust hedge size based on market intelligence
       
       Base: 1-2 DTE weeknight hedge
       Adjustments:
         - HIGH risk: +50% hedge size
         - CRITICAL risk: +100% hedge size
         - LOW risk: -25% hedge size (reduce cost)
       """
       if not risk_signals:
           return base_hedge_size
       
       # Get highest risk level from recent signals
       max_risk = max(signal.risk_level for signal in risk_signals)
       
       adjustments = {
           "CRITICAL": 2.0,  # Double hedge
           "HIGH": 1.5,
           "MEDIUM": 1.0,
           "LOW": 0.75
       }
       
       return int(base_hedge_size * adjustments.get(max_risk, 1.0))
   ```

### Phase 4: Monitoring & Alerts (2 hours)

1. **Daily Digest Report**
   - Summary of top news stories
   - Sentiment analysis
   - Recommended hedge adjustments
   - Email/Slack notifications

2. **Real-time Alerts**
   - CRITICAL risk events trigger immediate notification
   - Integration with audit logger

---

## Example: Adaptive Hedging Scenarios

### Scenario 1: FOMC Meeting Week

**News Detected**:
- "Fed Chair Powell to speak on monetary policy Wednesday"
- Market expecting 25bps rate decision

**LLM Analysis**:
- Risk Level: HIGH
- Volatility Expectation: INCREASED (+30%)
- Recommended Action: INCREASE_HEDGE

**Trading Response**:
- Monday put sales: Normal (4 contracts SPY)
- Tuesday: Increase hedge from 1 to 2 contracts (7 DTE puts)
- Wednesday (Fed day): Add protective 1 DTE put spreads
- Post-announcement: Reduce hedge back to normal

### Scenario 2: Earnings Season (NVDA, MSFT, AAPL)

**News Detected**:
- "NVDA earnings after close Thursday"
- "Tech sector guidance under scrutiny"

**LLM Analysis**:
- Risk Level: MEDIUM-HIGH
- Affected Assets: QQQ (30% NVDA weight)
- Recommended Action: SECTOR_HEDGE

**Trading Response**:
- Increase QQQ hedge specifically
- Keep SPY/DIA/IWM at normal levels
- Add 7 DTE QQQ put spread Thursday morning

### Scenario 3: Calm Market

**News Detected**:
- "Markets range-bound, low volume"
- "VIX at 3-month lows"

**LLM Analysis**:
- Risk Level: LOW
- Volatility Expectation: DECREASED
- Recommended Action: REDUCE_HEDGE

**Trading Response**:
- Reduce weekend hedge from 2 to 1 contract
- Save premium costs
- More aggressive put sales (higher delta)

---

## Code Structure

```
HedgeFund/
├── src/
│   ├── intelligence/
│   │   ├── __init__.py
│   │   ├── market_intelligence_service.py  # Main service
│   │   ├── news_fetcher.py                 # FMP API integration
│   │   ├── llm_analyzer.py                 # Ollama integration
│   │   └── risk_detector.py                # Risk signal generation
│   ├── strategies/
│   │   └── signal_generator.py             # UPDATED: Add adaptive hedging
│   └── main/
│       └── trading_engine.py               # UPDATED: Query intelligence service
└── config/
    └── config.yaml                         # UPDATED: Add intelligence section
```

---

## API Specification

### Market Intelligence Service

```python
class MarketIntelligenceService:
    """24/7 news monitoring and risk analysis"""
    
    async def fetch_latest_news(self) -> List[NewsItem]:
        """Fetch news from FMP API"""
        
    async def analyze_news_sentiment(self, news_items: List[NewsItem]) -> Dict[str, Any]:
        """Use LLM to analyze sentiment and risks"""
        
    async def generate_risk_signals(self) -> List[RiskSignal]:
        """Generate actionable risk signals"""
        
    async def get_recent_risk_signals(self, lookback_hours: int = 24) -> List[RiskSignal]:
        """Query recent risk signals from database"""
        
    def calculate_hedge_adjustment(self, risk_signals: List[RiskSignal]) -> float:
        """Calculate hedge size multiplier (0.5 to 2.0)"""
```

---

## Testing Strategy

### Unit Tests
- News fetcher with mock FMP responses
- LLM analyzer with sample news articles
- Risk signal generation logic

### Integration Tests
- Full pipeline: News → LLM → Risk Signals → Hedge Adjustment
- Verify database storage
- Test edge cases (API failures, LLM timeouts)

### Backtesting
- Run backtest with historical news events
- Verify hedge adjustments during:
  - 2020 COVID crash
  - 2022 Fed rate hikes
  - NVDA earnings beats

---

## Performance Considerations

**News Fetching**:
- Cache news for 5 minutes
- Rate limit: 1 request per 5 minutes to FMP
- Async fetching to avoid blocking

**LLM Analysis**:
- Run in background thread
- Timeout after 30 seconds
- Fallback to simple sentiment if LLM unavailable

**Database**:
- Index on `published_date`, `sentiment_score`, `impact_level`
- Auto-archive news older than 30 days
- Keep risk signals for 1 year (audit trail)

---

## Cost Estimates

**FMP API**: ~$20/month (Starter plan)
**Ollama**: Free (runs locally)
**Database**: Minimal (few MB per month)

**Total**: ~$20/month for institutional-grade market intelligence

---

## Next Steps

1. ✅ Review this plan
2. ⬜ Set up FMP API account (get API key)
3. ⬜ Install Ollama locally (`curl https://ollama.ai/install.sh | sh`)
4. ⬜ Pull LLM model (`ollama pull llama3`)
5. ⬜ Implement Phase 1 (basic news monitoring)
6. ⬜ Test with live news feed
7. ⬜ Implement Phase 2 (LLM integration)
8. ⬜ Backtest with historical events
9. ⬜ Deploy to production

---

## Success Metrics

- **Coverage**: Capture 95%+ of major market-moving news within 15 minutes
- **Accuracy**: 80%+ correct risk level classification (vs manual review)
- **Performance**: Hedge adjustments improve Sharpe ratio by 10-20%
- **Cost Savings**: Reduce unnecessary hedging costs during calm periods by 30%

