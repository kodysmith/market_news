# 24/7 Continuous Operation & ML Data Pipeline

## Overview

Transform the hedge fund system into a **continuously running, self-monitoring trading system** that:
- Runs 24/7 with scheduled tasks
- Monitors market news continuously
- Logs ALL data for ML model training
- Provides real-time dashboards
- Generates daily reports

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   24/7 TRADING DAEMON                        │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │   Market     │  │   News       │  │   Trading    │     │
│  │   Hours      │  │   Monitor    │  │   Engine     │     │
│  │   Monitor    │  │   (24/7)     │  │   (Daily)    │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
│         │                 │                  │              │
│         └─────────────────┴──────────────────┘              │
│                           │                                 │
│                    ┌──────▼──────┐                         │
│                    │  Data Lake   │                         │
│                    │  PostgreSQL  │                         │
│                    └──────────────┘                         │
└─────────────────────────────────────────────────────────────┘
         │                  │                  │
         ▼                  ▼                  ▼
   ┌─────────┐      ┌─────────┐      ┌─────────┐
   │Dashboard│      │ Reports │      │ML Models│
   │(Grafana)│      │ (Daily) │      │Training │
   └─────────┘      └─────────┘      └─────────┘
```

---

## Component 1: 24/7 Daemon Scheduler

**File**: `src/daemon/trading_daemon.py`

### Responsibilities

1. **Market Hours Detection**
   - Detect market open/close
   - Handle holidays
   - Adjust for daylight saving time

2. **Scheduled Tasks**
   - Pre-market (8:00 AM): News digest, risk assessment
   - Market open (9:30 AM): Generate signals
   - Market close (4:00 PM): Calculate NAV, update positions
   - Evening (6:00 PM): Generate hedges, adjust for overnight
   - Continuous: News monitoring every 5 minutes

3. **Health Monitoring**
   - Check all services every minute
   - Auto-restart failed components
   - Alert on persistent failures

### Implementation

```python
# src/daemon/trading_daemon.py
import asyncio
import schedule
from datetime import datetime, time
from typing import Dict, Any
import pytz

class TradingDaemon:
    """24/7 trading system daemon"""
    
    def __init__(self, config):
        self.config = config
        self.timezone = pytz.timezone('US/Eastern')
        self.running = False
        
        # Initialize all services
        self.market_intelligence = MarketIntelligenceService(config)
        self.trading_engine = TradingEngine(config)
        self.data_warehouse = DataWarehouse(config)
        self.health_monitor = HealthMonitor(config)
        
    async def start(self):
        """Start the daemon"""
        self.running = True
        logger.info("🚀 Trading Daemon starting...")
        
        # Schedule tasks
        self._schedule_tasks()
        
        # Start background tasks
        tasks = [
            asyncio.create_task(self._news_monitor_loop()),
            asyncio.create_task(self._health_check_loop()),
            asyncio.create_task(self._scheduled_tasks_loop()),
        ]
        
        await asyncio.gather(*tasks)
    
    def _schedule_tasks(self):
        """Schedule all trading tasks"""
        # Pre-market analysis
        schedule.every().day.at("08:00").do(self.pre_market_analysis)
        
        # Market open tasks
        schedule.every().day.at("09:30").do(self.market_open_routine)
        
        # Intraday checks (every 30 minutes during market hours)
        for hour in range(9, 16):
            for minute in [0, 30]:
                schedule.every().day.at(f"{hour:02d}:{minute:02d}").do(
                    self.intraday_check
                )
        
        # Market close
        schedule.every().day.at("16:00").do(self.market_close_routine)
        
        # Evening hedge generation
        schedule.every().day.at("18:00").do(self.evening_hedge_routine)
        
        # Weekend hedges (Friday evening)
        schedule.every().friday.at("17:00").do(self.weekend_hedge_routine)
        
        # Daily report
        schedule.every().day.at("20:00").do(self.generate_daily_report)
    
    async def pre_market_analysis(self):
        """Pre-market routine (8:00 AM)"""
        logger.info("📊 Pre-market analysis starting...")
        
        # Get overnight news
        news = await self.market_intelligence.get_overnight_news()
        
        # Analyze sentiment
        sentiment = await self.market_intelligence.analyze_sentiment(news)
        
        # Generate risk signals
        risk_signals = await self.market_intelligence.generate_risk_signals(
            sentiment
        )
        
        # Log to data warehouse
        await self.data_warehouse.log_pre_market_analysis({
            'timestamp': datetime.now(),
            'news_count': len(news),
            'sentiment': sentiment,
            'risk_signals': risk_signals
        })
        
        # Send morning briefing
        await self._send_morning_briefing(news, sentiment, risk_signals)
    
    async def market_open_routine(self):
        """Market open routine (9:30 AM)"""
        if not self._is_market_day():
            logger.info("Market closed today (holiday)")
            return
        
        logger.info("🔔 Market open - executing trading routine...")
        
        try:
            # Run daily trading cycle
            result = await self.trading_engine.run_daily_cycle(datetime.now())
            
            # Log results
            await self.data_warehouse.log_trading_cycle(result)
            
        except Exception as e:
            logger.error(f"Market open routine failed: {e}")
            await self._alert_failure("market_open", str(e))
    
    async def _news_monitor_loop(self):
        """Continuous news monitoring (every 5 minutes)"""
        while self.running:
            try:
                # Fetch latest news
                news = await self.market_intelligence.fetch_latest_news()
                
                if news:
                    # Analyze and generate signals
                    analysis = await self.market_intelligence.analyze_news(news)
                    
                    # Check for critical events
                    if analysis['risk_level'] == 'CRITICAL':
                        await self._handle_critical_event(analysis)
                    
                    # Log all news
                    await self.data_warehouse.log_news_batch(news, analysis)
                
            except Exception as e:
                logger.error(f"News monitoring error: {e}")
            
            # Wait 5 minutes
            await asyncio.sleep(300)
    
    async def _health_check_loop(self):
        """Monitor health of all services"""
        while self.running:
            try:
                health_status = await self.health_monitor.check_all_services()
                
                # Log health metrics
                await self.data_warehouse.log_health_metrics(health_status)
                
                # Alert on failures
                for service, status in health_status.items():
                    if not status['healthy']:
                        await self._alert_service_failure(service, status)
                
            except Exception as e:
                logger.error(f"Health check error: {e}")
            
            # Check every minute
            await asyncio.sleep(60)
    
    def _is_market_day(self) -> bool:
        """Check if today is a trading day"""
        today = datetime.now(self.timezone)
        
        # Check weekday
        if today.weekday() >= 5:  # Saturday/Sunday
            return False
        
        # Check holidays (simplified - use market calendar in production)
        return True
```

---

## Component 2: Comprehensive Data Warehouse

**File**: `src/data/data_warehouse.py`

### Database Schema

```sql
-- =====================================================
-- MARKET DATA WAREHOUSE
-- Comprehensive storage for ML training
-- =====================================================

-- News and Sentiment
CREATE TABLE news_feed (
    id SERIAL PRIMARY KEY,
    news_id TEXT UNIQUE NOT NULL,
    headline TEXT NOT NULL,
    source TEXT,
    url TEXT,
    content TEXT,
    summary TEXT,
    sentiment TEXT,  -- bullish/bearish/neutral
    sentiment_score FLOAT,
    sentiment_confidence FLOAT,
    llm_analysis JSONB,  -- Full LLM response
    tickers TEXT[],
    impact_level TEXT,  -- low/medium/high/critical
    risk_signals JSONB,  -- Associated risk signals
    published_date TIMESTAMP,
    ingested_at TIMESTAMP DEFAULT NOW(),
    processed_at TIMESTAMP,
    
    INDEX idx_news_published (published_date DESC),
    INDEX idx_news_sentiment (sentiment, sentiment_score),
    INDEX idx_news_impact (impact_level, published_date)
);

-- Risk Events
CREATE TABLE risk_events (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL,
    risk_level TEXT NOT NULL,  -- LOW/MEDIUM/HIGH/CRITICAL
    risk_type TEXT NOT NULL,  -- volatility/drawdown/sector/market/news
    description TEXT,
    source TEXT,  -- news/technical/fundamental
    recommended_action TEXT,
    confidence FLOAT,
    affected_assets TEXT[],
    hedge_adjustment_pct FLOAT,
    metadata JSONB,  -- Additional context
    resolved_at TIMESTAMP,
    
    INDEX idx_risk_timestamp (timestamp DESC),
    INDEX idx_risk_level (risk_level, timestamp),
    INDEX idx_risk_unresolved (resolved_at) WHERE resolved_at IS NULL
);

-- Trading Signals (All generated signals)
CREATE TABLE trading_signals (
    id SERIAL PRIMARY KEY,
    signal_id TEXT UNIQUE NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    asset TEXT NOT NULL,
    action TEXT NOT NULL,  -- SELL_PUT/BUY_CALL/CLOSE_POSITION
    quantity INT,
    strike DECIMAL(10,2),
    expiration DATE,
    option_type CHAR(1),  -- P/C
    dte INT,
    expected_price DECIMAL(10,4),
    reasoning TEXT,
    strategy_component TEXT,
    priority INT,
    expected_delta DECIMAL(12,4),
    capital_required DECIMAL(12,2),
    
    -- Risk assessment at signal time
    portfolio_nav DECIMAL(15,2),
    portfolio_cash DECIMAL(15,2),
    portfolio_delta DECIMAL(12,4),
    risk_score FLOAT,
    
    -- Execution tracking
    executed BOOLEAN DEFAULT FALSE,
    execution_timestamp TIMESTAMP,
    execution_price DECIMAL(10,4),
    execution_status TEXT,  -- pending/filled/rejected/cancelled
    rejection_reason TEXT,
    
    INDEX idx_signal_timestamp (timestamp DESC),
    INDEX idx_signal_asset (asset, timestamp),
    INDEX idx_signal_pending (executed) WHERE NOT executed
);

-- Market State Snapshots (Every 5 minutes during market hours)
CREATE TABLE market_snapshots (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL,
    
    -- Prices
    spy_price DECIMAL(10,2),
    qqq_price DECIMAL(10,2),
    dia_price DECIMAL(10,2),
    iwm_price DECIMAL(10,2),
    vix_price DECIMAL(10,2),
    
    -- Greeks
    spy_iv DECIMAL(8,4),
    qqq_iv DECIMAL(8,4),
    
    -- Technical indicators
    spy_rsi DECIMAL(8,4),
    spy_macd DECIMAL(8,4),
    spy_sma_20 DECIMAL(10,2),
    spy_sma_50 DECIMAL(10,2),
    
    -- Market regime
    regime TEXT,  -- bull/bear/neutral/volatile
    regime_confidence FLOAT,
    
    -- Sentiment
    news_sentiment FLOAT,  -- -1 to 1
    social_sentiment FLOAT,
    
    INDEX idx_snapshot_time (timestamp DESC)
);

-- Portfolio State (End of day snapshots)
CREATE TABLE portfolio_snapshots (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL,
    
    -- NAV
    nav DECIMAL(15,2) NOT NULL,
    cash DECIMAL(15,2) NOT NULL,
    positions_value DECIMAL(15,2),
    
    -- Performance
    daily_return DECIMAL(10,6),
    ytd_return DECIMAL(10,6),
    sharpe_ratio DECIMAL(8,4),
    max_drawdown DECIMAL(8,6),
    
    -- Risk metrics
    portfolio_delta DECIMAL(12,4),
    portfolio_gamma DECIMAL(12,4),
    portfolio_theta DECIMAL(12,4),
    portfolio_vega DECIMAL(12,4),
    var_95 DECIMAL(12,2),  -- Value at Risk
    
    -- Position counts
    num_short_puts INT,
    num_covered_calls INT,
    num_shares INT,
    num_hedges INT,
    
    -- Capital deployment
    capital_deployed DECIMAL(15,2),
    capital_deployed_pct DECIMAL(6,4),
    
    -- Detailed breakdown (JSON)
    positions_detail JSONB,
    greeks_by_asset JSONB,
    
    INDEX idx_portfolio_time (timestamp DESC)
);

-- Model Training Features (Pre-computed features for ML)
CREATE TABLE ml_training_features (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL,
    
    -- Target variable (what we're predicting)
    next_day_return DECIMAL(10,6),  -- T+1 return
    next_week_return DECIMAL(10,6),  -- T+5 return
    max_drawdown_7d DECIMAL(8,6),  -- Worst drawdown next 7 days
    
    -- Market features
    spy_price DECIMAL(10,2),
    spy_return_1d DECIMAL(8,6),
    spy_return_5d DECIMAL(8,6),
    spy_volatility_20d DECIMAL(8,6),
    vix DECIMAL(10,2),
    vix_change DECIMAL(8,6),
    
    -- Technical features
    rsi_14 DECIMAL(8,4),
    macd DECIMAL(8,4),
    bbands_width DECIMAL(8,4),
    atr_14 DECIMAL(8,4),
    
    -- Sentiment features
    news_sentiment_1d DECIMAL(6,4),
    news_sentiment_3d DECIMAL(6,4),
    news_sentiment_7d DECIMAL(6,4),
    high_impact_news_count_1d INT,
    
    -- Risk features
    risk_events_count_1d INT,
    highest_risk_level TEXT,
    
    -- Portfolio features
    portfolio_delta DECIMAL(12,4),
    capital_deployed_pct DECIMAL(6,4),
    num_positions INT,
    
    -- Regime features
    regime TEXT,
    regime_duration_days INT,
    
    INDEX idx_ml_features_time (timestamp DESC)
);

-- System Performance Logs
CREATE TABLE system_logs (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT NOW(),
    log_level TEXT NOT NULL,  -- INFO/WARNING/ERROR/CRITICAL
    component TEXT NOT NULL,  -- trading_engine/news_monitor/etc
    event_type TEXT NOT NULL,
    message TEXT,
    metadata JSONB,
    
    INDEX idx_logs_time (timestamp DESC),
    INDEX idx_logs_level (log_level, timestamp),
    INDEX idx_logs_component (component, timestamp)
);

-- Daily Reports (Structured summaries)
CREATE TABLE daily_reports (
    id SERIAL PRIMARY KEY,
    report_date DATE UNIQUE NOT NULL,
    
    -- Performance summary
    opening_nav DECIMAL(15,2),
    closing_nav DECIMAL(15,2),
    daily_return DECIMAL(10,6),
    
    -- Trading activity
    signals_generated INT,
    trades_executed INT,
    trades_rejected INT,
    
    -- News summary
    news_items_processed INT,
    high_impact_news INT,
    average_sentiment FLOAT,
    
    -- Risk events
    risk_events_count INT,
    critical_events INT,
    
    -- System health
    uptime_pct DECIMAL(5,2),
    errors_count INT,
    
    -- Full report (markdown)
    report_markdown TEXT,
    report_json JSONB,
    
    generated_at TIMESTAMP DEFAULT NOW(),
    
    INDEX idx_report_date (report_date DESC)
);
```

---

## Component 3: Market Intelligence Service

**File**: `src/intelligence/market_intelligence_service.py`

```python
"""
Market Intelligence Service
Continuous news monitoring with LLM analysis
"""

import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
import aiohttp
import logging

logger = logging.getLogger(__name__)

@dataclass
class NewsItem:
    id: str
    headline: str
    source: str
    url: str
    content: str
    published_date: datetime
    tickers: List[str]
    raw_data: Dict[str, Any]

@dataclass
class RiskSignal:
    timestamp: datetime
    risk_level: str  # LOW/MEDIUM/HIGH/CRITICAL
    risk_type: str
    description: str
    recommended_action: str
    confidence: float
    affected_assets: List[str]
    hedge_adjustment_pct: float
    source: str  # news/technical/fundamental

class MarketIntelligenceService:
    """24/7 market news monitoring and risk analysis"""
    
    def __init__(self, config, data_warehouse, llm_analyzer):
        self.config = config
        self.data_warehouse = data_warehouse
        self.llm = llm_analyzer
        self.fmp_api_key = config.intelligence.fmp_api_key
        self.session: Optional[aiohttp.ClientSession] = None
        
    async def initialize(self):
        """Initialize HTTP session"""
        self.session = aiohttp.ClientSession()
        logger.info("Market Intelligence Service initialized")
    
    async def shutdown(self):
        """Cleanup"""
        if self.session:
            await self.session.close()
    
    async def fetch_latest_news(self, limit: int = 50) -> List[NewsItem]:
        """Fetch latest news from FMP API"""
        try:
            url = f"https://financialmodelingprep.com/api/v4/general_news"
            params = {
                'page': 0,
                'limit': limit,
                'apikey': self.fmp_api_key
            }
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    news_items = []
                    for item in data:
                        news_item = NewsItem(
                            id=f"fmp_{item.get('publishedDate', '')}_{hash(item.get('title', '')) % 10000}",
                            headline=item.get('title', ''),
                            source=item.get('site', 'Unknown'),
                            url=item.get('url', ''),
                            content=item.get('text', ''),
                            published_date=datetime.fromisoformat(
                                item.get('publishedDate', datetime.now().isoformat()).replace('Z', '+00:00')
                            ),
                            tickers=self._extract_tickers(item.get('text', '')),
                            raw_data=item
                        )
                        news_items.append(news_item)
                    
                    logger.info(f"Fetched {len(news_items)} news items")
                    return news_items
                else:
                    logger.error(f"FMP API error: {response.status}")
                    return []
                    
        except Exception as e:
            logger.error(f"Error fetching news: {e}")
            return []
    
    async def analyze_news_batch(self, news_items: List[NewsItem]) -> Dict[str, Any]:
        """Analyze batch of news with LLM"""
        if not news_items:
            return {'sentiment': 'neutral', 'risk_signals': []}
        
        # Prepare context for LLM
        context = self._prepare_news_context(news_items)
        
        # LLM prompt
        prompt = f"""
You are a quantitative hedge fund risk analyst. Analyze the following market news and provide:

1. Overall market sentiment (BULLISH/BEARISH/NEUTRAL) with confidence (0-100%)
2. Risk level (LOW/MEDIUM/HIGH/CRITICAL)
3. Key risk factors
4. Recommended hedging actions
5. Assets most affected (SPY/QQQ/DIA/IWM)

NEWS SUMMARY:
{context}

Provide a structured analysis focusing on actionable trading insights.
"""
        
        # Get LLM analysis
        analysis = await self.llm.analyze(prompt)
        
        # Parse response
        parsed = self._parse_llm_analysis(analysis)
        
        # Generate risk signals
        risk_signals = self._generate_risk_signals(parsed, news_items)
        
        return {
            'timestamp': datetime.now(),
            'news_count': len(news_items),
            'sentiment': parsed['sentiment'],
            'sentiment_score': parsed['sentiment_score'],
            'risk_level': parsed['risk_level'],
            'risk_signals': risk_signals,
            'llm_analysis': analysis
        }
    
    def _generate_risk_signals(self, llm_analysis: Dict, news_items: List[NewsItem]) -> List[RiskSignal]:
        """Generate actionable risk signals"""
        signals = []
        
        risk_level = llm_analysis['risk_level']
        
        # Map risk level to hedge adjustments
        hedge_adjustments = {
            'CRITICAL': 1.0,  # Double hedges
            'HIGH': 0.5,      # Increase by 50%
            'MEDIUM': 0.0,    # No change
            'LOW': -0.25      # Reduce by 25%
        }
        
        if risk_level in ['HIGH', 'CRITICAL']:
            signal = RiskSignal(
                timestamp=datetime.now(),
                risk_level=risk_level,
                risk_type='market_news',
                description=llm_analysis.get('summary', 'High risk detected in market news'),
                recommended_action='increase_hedge',
                confidence=llm_analysis['confidence'],
                affected_assets=llm_analysis.get('affected_assets', ['SPY', 'QQQ', 'DIA', 'IWM']),
                hedge_adjustment_pct=hedge_adjustments[risk_level],
                source='news_llm'
            )
            signals.append(signal)
        
        return signals
    
    async def get_overnight_news(self) -> List[NewsItem]:
        """Get news from last 16 hours (overnight period)"""
        news = await self.fetch_latest_news(limit=100)
        
        cutoff = datetime.now() - timedelta(hours=16)
        overnight_news = [
            item for item in news 
            if item.published_date >= cutoff
        ]
        
        return overnight_news
    
    def _extract_tickers(self, text: str) -> List[str]:
        """Extract stock tickers from text"""
        # Simplified - use regex in production
        common_tickers = ['SPY', 'QQQ', 'DIA', 'IWM', 'NVDA', 'AAPL', 'MSFT', 'TSLA', 'AMZN', 'META']
        found = [ticker for ticker in common_tickers if ticker in text.upper()]
        return found
    
    def _prepare_news_context(self, news_items: List[NewsItem]) -> str:
        """Prepare news for LLM"""
        context = []
        for item in news_items[:10]:  # Top 10 most recent
            context.append(f"- {item.headline} ({item.source})")
        return "\n".join(context)
    
    def _parse_llm_analysis(self, analysis: str) -> Dict[str, Any]:
        """Parse LLM response"""
        # Simplified parsing - use structured output in production
        sentiment = 'NEUTRAL'
        if 'BULLISH' in analysis.upper():
            sentiment = 'BULLISH'
        elif 'BEARISH' in analysis.upper():
            sentiment = 'BEARISH'
        
        risk_level = 'MEDIUM'
        if 'CRITICAL' in analysis.upper():
            risk_level = 'CRITICAL'
        elif 'HIGH' in analysis.upper():
            risk_level = 'HIGH'
        elif 'LOW' in analysis.upper():
            risk_level = 'LOW'
        
        return {
            'sentiment': sentiment,
            'sentiment_score': 0.0,
            'risk_level': risk_level,
            'confidence': 0.7,
            'summary': analysis[:200],
            'affected_assets': ['SPY', 'QQQ', 'DIA', 'IWM']
        }
```

---

## Component 4: Data Warehouse Implementation

**File**: `src/data/data_warehouse.py`

```python
"""
Data Warehouse - Comprehensive data logging for ML training
"""

import asyncpg
from typing import List, Dict, Any
from datetime import datetime
import json
import logging

logger = logging.getLogger(__name__)

class DataWarehouse:
    """Central data warehouse for all trading data"""
    
    def __init__(self, config):
        self.config = config
        self.pool: asyncpg.Pool = None
    
    async def initialize(self):
        """Initialize database connection pool"""
        self.pool = await asyncpg.create_pool(
            host=self.config.database.host,
            port=self.config.database.port,
            user=self.config.database.user,
            password=self.config.database.password,
            database=self.config.database.database,
            min_size=5,
            max_size=20
        )
        logger.info("Data Warehouse initialized")
    
    async def log_news_batch(self, news_items: List, analysis: Dict[str, Any]):
        """Log batch of news with analysis"""
        async with self.pool.acquire() as conn:
            for item in news_items:
                await conn.execute('''
                    INSERT INTO news_feed (
                        news_id, headline, source, url, content, summary,
                        sentiment, sentiment_score, sentiment_confidence,
                        llm_analysis, tickers, impact_level, published_date,
                        processed_at
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
                    ON CONFLICT (news_id) DO NOTHING
                ''',
                    item.id,
                    item.headline,
                    item.source,
                    item.url,
                    item.content,
                    item.content[:300] if len(item.content) > 300 else item.content,
                    analysis.get('sentiment', 'neutral'),
                    analysis.get('sentiment_score', 0.0),
                    analysis.get('confidence', 0.0),
                    json.dumps(analysis.get('llm_analysis', {})),
                    item.tickers,
                    self._determine_impact_level(item, analysis),
                    item.published_date,
                    datetime.now()
                )
        
        logger.info(f"Logged {len(news_items)} news items to warehouse")
    
    async def log_risk_signals(self, risk_signals: List):
        """Log risk signals"""
        async with self.pool.acquire() as conn:
            for signal in risk_signals:
                await conn.execute('''
                    INSERT INTO risk_events (
                        timestamp, risk_level, risk_type, description,
                        source, recommended_action, confidence,
                        affected_assets, hedge_adjustment_pct, metadata
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                ''',
                    signal.timestamp,
                    signal.risk_level,
                    signal.risk_type,
                    signal.description,
                    signal.source,
                    signal.recommended_action,
                    signal.confidence,
                    signal.affected_assets,
                    signal.hedge_adjustment_pct,
                    json.dumps({})
                )
        
        logger.info(f"Logged {len(risk_signals)} risk signals")
    
    async def log_trading_signal(self, signal, portfolio_state: Dict):
        """Log generated trading signal"""
        async with self.pool.acquire() as conn:
            await conn.execute('''
                INSERT INTO trading_signals (
                    signal_id, timestamp, asset, action, quantity,
                    strike, expiration, option_type, dte, expected_price,
                    reasoning, strategy_component, priority,
                    expected_delta, capital_required,
                    portfolio_nav, portfolio_cash, portfolio_delta,
                    risk_score
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19)
            ''',
                signal.id,
                signal.timestamp,
                signal.asset,
                signal.action.value,
                signal.quantity,
                float(signal.strike),
                signal.expiration,
                signal.option_type,
                signal.dte,
                float(signal.expected_price),
                signal.reasoning,
                signal.strategy_component,
                signal.priority,
                float(signal.expected_delta),
                float(signal.capital_required),
                float(portfolio_state['nav']),
                float(portfolio_state['cash']),
                float(portfolio_state['net_delta']),
                0.0  # Risk score calculation
            )
    
    async def log_market_snapshot(self, snapshot: Dict[str, Any]):
        """Log market state snapshot"""
        async with self.pool.acquire() as conn:
            await conn.execute('''
                INSERT INTO market_snapshots (
                    timestamp, spy_price, qqq_price, dia_price, iwm_price, vix_price,
                    spy_iv, qqq_iv, regime, news_sentiment
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            ''',
                snapshot['timestamp'],
                snapshot.get('spy_price'),
                snapshot.get('qqq_price'),
                snapshot.get('dia_price'),
                snapshot.get('iwm_price'),
                snapshot.get('vix_price'),
                snapshot.get('spy_iv'),
                snapshot.get('qqq_iv'),
                snapshot.get('regime', 'unknown'),
                snapshot.get('news_sentiment', 0.0)
            )
    
    async def log_portfolio_snapshot(self, snapshot: Dict[str, Any]):
        """Log end-of-day portfolio state"""
        async with self.pool.acquire() as conn:
            await conn.execute('''
                INSERT INTO portfolio_snapshots (
                    timestamp, nav, cash, positions_value,
                    daily_return, ytd_return, sharpe_ratio, max_drawdown,
                    portfolio_delta, portfolio_gamma, portfolio_theta, portfolio_vega,
                    num_short_puts, num_covered_calls, num_shares, num_hedges,
                    capital_deployed, capital_deployed_pct,
                    positions_detail, greeks_by_asset
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20)
            ''',
                snapshot['timestamp'],
                float(snapshot['nav']),
                float(snapshot['cash']),
                float(snapshot.get('positions_value', 0)),
                snapshot.get('daily_return', 0.0),
                snapshot.get('ytd_return', 0.0),
                snapshot.get('sharpe_ratio', 0.0),
                snapshot.get('max_drawdown', 0.0),
                float(snapshot.get('portfolio_delta', 0)),
                float(snapshot.get('portfolio_gamma', 0)),
                float(snapshot.get('portfolio_theta', 0)),
                float(snapshot.get('portfolio_vega', 0)),
                snapshot.get('num_short_puts', 0),
                snapshot.get('num_covered_calls', 0),
                snapshot.get('num_shares', 0),
                snapshot.get('num_hedges', 0),
                float(snapshot.get('capital_deployed', 0)),
                snapshot.get('capital_deployed_pct', 0.0),
                json.dumps(snapshot.get('positions_detail', {})),
                json.dumps(snapshot.get('greeks_by_asset', {}))
            )
        
        logger.info(f"Logged portfolio snapshot: NAV=${snapshot['nav']:,.2f}")
    
    def _determine_impact_level(self, news_item, analysis: Dict) -> str:
        """Determine impact level of news"""
        # High impact keywords
        high_impact_keywords = [
            'fed', 'fomc', 'rate cut', 'rate hike', 'recession',
            'inflation', 'earnings miss', 'guidance cut', 'crash'
        ]
        
        content_lower = news_item.content.lower()
        
        if any(keyword in content_lower for keyword in high_impact_keywords):
            return 'high'
        
        if analysis.get('risk_level') in ['HIGH', 'CRITICAL']:
            return 'high'
        
        return 'medium'
```

---

## Deployment & Operations

### Docker Compose for Local Development

```yaml
# docker-compose.yml
version: '3.8'

services:
  postgres:
    image: postgres:15
    environment:
      POSTGRES_USER: hedgefund
      POSTGRES_PASSWORD: ${DB_PASSWORD}
      POSTGRES_DB: trading_data
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./src/database/schema.sql:/docker-entrypoint-initdb.d/01_schema.sql
      - ./src/database/warehouse_schema.sql:/docker-entrypoint-initdb.d/02_warehouse.sql
    ports:
      - "5432:5432"
  
  redis:
    image: redis:7
    ports:
      - "6379:6379"
  
  ollama:
    image: ollama/ollama:latest
    volumes:
      - ollama_models:/root/.ollama
    ports:
      - "11434:11434"
  
  trading_daemon:
    build: .
    command: python -m src.daemon.run_daemon
    depends_on:
      - postgres
      - redis
      - ollama
    environment:
      - DATABASE_URL=postgresql://hedgefund:${DB_PASSWORD}@postgres:5432/trading_data
      - REDIS_URL=redis://redis:6379
      - OLLAMA_URL=http://ollama:11434
      - FMP_API_KEY=${FMP_API_KEY}
      - ALPACA_API_KEY=${ALPACA_API_KEY}
      - ALPACA_SECRET_KEY=${ALPACA_SECRET_KEY}
    volumes:
      - ./config:/app/config
      - ./logs:/app/logs
    restart: unless-stopped
  
  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    volumes:
      - grafana_data:/var/lib/grafana
      - ./monitoring/grafana/dashboards:/etc/grafana/provisioning/dashboards
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD}

volumes:
  postgres_data:
  ollama_models:
  grafana_data:
```

### Systemd Service for Production

```ini
# /etc/systemd/system/hedgefund-daemon.service
[Unit]
Description=Hedge Fund Trading Daemon
After=network.target postgresql.service

[Service]
Type=simple
User=trading
WorkingDirectory=/opt/hedgefund
ExecStart=/opt/hedgefund/venv/bin/python -m src.daemon.run_daemon
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

---

## ML Training Pipeline

**File**: `src/ml/training_pipeline.py`

```python
"""
ML Model Training Pipeline
Uses data warehouse for training
"""

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
import asyncpg

class MLTrainingPipeline:
    """Train models from data warehouse"""
    
    async def prepare_training_data(self, start_date, end_date):
        """Extract features from warehouse"""
        conn = await asyncpg.connect(database_url)
        
        # Query pre-computed features
        query = '''
            SELECT * FROM ml_training_features
            WHERE timestamp BETWEEN $1 AND $2
            ORDER BY timestamp
        '''
        
        rows = await conn.fetch(query, start_date, end_date)
        df = pd.DataFrame(rows)
        
        return df
    
    def train_return_predictor(self, df: pd.DataFrame):
        """Train model to predict next-day returns"""
        feature_cols = [
            'spy_return_1d', 'spy_return_5d', 'spy_volatility_20d',
            'vix', 'vix_change', 'rsi_14', 'macd',
            'news_sentiment_1d', 'news_sentiment_3d',
            'risk_events_count_1d', 'portfolio_delta'
        ]
        
        X = df[feature_cols]
        y = df['next_day_return']
        
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        return model
```

---

## Monitoring Dashboard (Grafana)

**Metrics to Track**:
- NAV over time
- Daily returns
- Sharpe ratio (rolling 30d)
- News sentiment
- Risk event frequency
- System uptime
- Trade execution success rate

---

## Summary

This creates a **production-grade, 24/7 hedge fund system** with:

✅ Continuous news monitoring
✅ LLM-powered risk analysis  
✅ Adaptive hedging based on market intelligence
✅ Comprehensive data logging for ML
✅ 24/7 daemon with health monitoring
✅ Data warehouse with 10+ years retention
✅ ML training pipeline
✅ Real-time dashboards

**Total Build Time**: ~80 additional hours (2 weeks)

Ready to implement? I can start with any component you'd like!

