# Crypto Arbitrage System Architecture

## Overview

High-speed, risk-free cryptocurrency arbitrage system that monitors price discrepancies across Binance, Coinbase, and Kraken exchanges and executes simultaneous buy/sell orders to capture profits.

---

## System Architecture

```
┌────────────────────────────────────────────────────────────┐
│                   ARBITRAGE DAEMON (24/7)                   │
│                                                             │
│  ┌──────────────┐      ┌──────────────┐                   │
│  │   Exchange   │─────▶│    Price     │                   │
│  │   Manager    │      │  Aggregator  │                   │
│  │  (3 exchanges)│      │  (Websockets)│                   │
│  └──────────────┘      └──────────────┘                   │
│         │                      │                            │
│         │                      ▼                            │
│         │              ┌──────────────┐                    │
│         │              │  Arbitrage   │                    │
│         │              │   Detector   │                    │
│         │              └──────────────┘                    │
│         │                      │                            │
│         │                      ▼                            │
│         │              ┌──────────────┐                    │
│         │              │     Risk     │                    │
│         │              │   Manager    │                    │
│         │              └──────────────┘                    │
│         │                      │                            │
│         │                      ▼                            │
│         │              ┌──────────────┐                    │
│         └─────────────▶│  Arbitrage   │                    │
│                        │   Executor   │                    │
│                        │ (Both Legs)  │                    │
│                        └──────────────┘                    │
│                               │                             │
│                               ▼                             │
│                        ┌──────────────┐                    │
│                        │   SQLite DB  │                    │
│                        │   Repository │                    │
│                        └──────────────┘                    │
└────────────────────────────────────────────────────────────┘
                               │
                               ▼
                        ┌──────────────┐
                        │    Flask     │
                        │   Dashboard  │
                        │  (Port 5001) │
                        └──────────────┘
```

---

## Component Details

### 1. Exchange Manager

**File:** `src/exchanges/exchange_manager.py`

**Purpose:** Manages connections to all exchanges

**Responsibilities:**
- Initialize CCXT clients for each exchange
- Establish websocket connections
- Coordinate parallel queries
- Health monitoring

**Key Methods:**
- `connect_all()` - Connect to all exchanges
- `get_all_prices(pair)` - Get prices from all exchanges
- `get_all_balances()` - Query balances
- `health_check_all()` - System health

---

### 2. Price Aggregator

**File:** `src/data/price_aggregator.py`

**Purpose:** Real-time price collection and caching

**Responsibilities:**
- Subscribe to websocket price feeds
- Maintain in-memory price cache
- Detect stale prices
- Track update frequency

**Features:**
- Sub-second price updates via websockets
- Automatic staleness detection (>5 seconds = stale)
- High-performance in-memory cache
- Statistics tracking

---

### 3. Arbitrage Detector

**File:** `src/detection/arbitrage_detector.py`

**Purpose:** Identify profitable arbitrage opportunities

**Algorithm:**
```python
1. For each trading pair:
   - Get prices from all exchanges
   - Find cheapest (buy) and most expensive (sell)
   - Calculate gross spread
   - Subtract exchange fees
   - Calculate net spread
   
2. If net spread > 0.5%:
   - Create ArbitrageOpportunity
   - Calculate confidence score
   - Return opportunity
```

**Filters:**
- Minimum spread: 0.5% after fees
- Minimum volume: $1M daily
- Maximum price age: 5 seconds
- Must be on different exchanges

---

### 4. Arbitrage Executor

**File:** `src/execution/arbitrage_executor.py`

**Purpose:** Execute both legs simultaneously

**Execution Flow:**
```python
1. Verify balances on both exchanges
2. Calculate trade size (limited by smaller balance)
3. Submit market orders SIMULTANEOUSLY to both exchanges
4. Monitor fills with 2-second timeout
5. If both fill → SUCCESS ✅
6. If one fills, one fails → REVERSE filled leg ❌
7. If neither fills → CANCEL both ℹ️
```

**Risk-Free Guarantee:**
- Both legs execute or neither executes
- Automatic reversal if one leg fails
- No directional exposure maintained

---

### 5. Risk Manager

**File:** `src/risk/arbitrage_risk_manager.py`

**Purpose:** Pre-trade validation and circuit breakers

**Checks:**
- ✅ Sufficient balance on both exchanges
- ✅ Spread still meets minimum threshold
- ✅ Daily loss limit not exceeded
- ✅ Rate limits not breached
- ✅ Circuit breaker not active
- ✅ Price data is fresh

**Circuit Breakers:**
- 5 consecutive failures → 30 minute halt
- $500 daily loss → 24 hour halt

---

### 6. Database Repository

**File:** `src/database/repository.py`

**Purpose:** Data persistence and analytics

**Tables:**
- `opportunities` - All detected opportunities
- `executions` - Executed trades
- `exchange_balances` - Balance snapshots
- `price_snapshots` - Price history
- `daily_performance` - Aggregated metrics

**Analytics Queries:**
- Daily P&L by exchange
- Success rates by pair
- Execution time statistics
- Opportunity frequency

---

### 7. Performance Tracker

**File:** `src/monitoring/performance_tracker.py`

**Purpose:** Calculate and track performance metrics

**Metrics:**
- Total P&L
- Win rate
- Average execution time
- ROI percentage
- Opportunities per hour

---

### 8. Arbitrage Daemon

**File:** `src/daemon/arbitrage_daemon.py`

**Purpose:** 24/7 orchestrator

**Operation:**
```python
while running:
    # Detect opportunities (every 100ms)
    opportunities = detector.detect_opportunities()
    
    if opportunities:
        # Validate with risk manager
        for opp in opportunities:
            approved, reason = risk_manager.validate(opp)
            
            if approved:
                # Execute arbitrage
                execution = executor.execute(opp)
                
                # Save to database
                repository.save(execution)
                
                break  # One at a time
    
    await sleep(0.1)
```

**Features:**
- Runs continuously (crypto never sleeps)
- Detects opportunities in real-time
- Executes immediately when found
- Handles graceful shutdown

---

### 9. Dashboard

**File:** `src/dashboard/dashboard_server.py`

**Purpose:** Web-based monitoring interface

**Endpoints:**
- `/` - Main dashboard UI
- `/api/health` - System health
- `/api/opportunities/live` - Recent opportunities
- `/api/executions/recent` - Recent trades
- `/api/performance/current` - Performance metrics

**Features:**
- Auto-refresh every 5 seconds
- Real-time opportunity display
- Execution history
- P&L tracking

---

## Data Flow

### Opportunity Detection

```
Websockets → Price Cache → Detector → Opportunity
     ↓            ↓            ↓           ↓
  Binance    In-Memory    Compare     ArbitrageOpportunity
  Coinbase   <100ms       Prices      object created
  Kraken     latency      & Fees
```

### Opportunity Execution

```
Opportunity → Risk Validation → Execution → Result
      ↓              ↓               ↓          ↓
  Net spread    Balance check    Submit 2   Both filled
  > 0.5%        Volume check     orders     or neither
  Fresh data    Rate limits      async
```

---

## Performance Characteristics

**Target Metrics:**

| Metric | Target | Actual (Expected) |
|--------|--------|-------------------|
| Detection latency | <100ms | 50-100ms |
| Execution time | <1s | 500-1000ms |
| Opportunity detection | 5+/day | 5-20/day |
| Execution success rate | >70% | 70-80% |
| Net profit/trade | >0.5% | 0.5-0.7% |

**Resource Usage:**

| Resource | Expected |
|----------|----------|
| CPU | 5-15% |
| Memory | 100-200MB |
| Network | 1-5 Mbps |
| Disk | <10 MB/day |

---

## Scalability

**Current Capacity:**
- 3 exchanges
- 25 pairs = 75 price feeds
- ~5-20 opportunities/day
- ~100 trades/day max

**Scaling Options:**
1. Add more exchanges (4-5 total)
2. Add more pairs (up to 100)
3. Add DEX integration
4. Multi-threading for execution
5. Distributed deployment

---

## Security

**API Key Security:**
- Stored in .env file (not committed to git)
- Read-only from environment variables
- Trading-only permissions (no withdrawals)
- IP whitelisting on exchanges

**Operational Security:**
- Simulation mode by default
- Circuit breakers for safety
- Daily loss limits
- Rate limiting
- Audit logging

---

## Failure Modes & Recovery

**Exchange Disconnection:**
- Auto-reconnect with exponential backoff
- Switch to other exchanges
- Log downtime

**Partial Fill:**
- Reverse filled leg immediately
- Log as failed execution
- No directional risk

**Network Issues:**
- Timeout after 2 seconds
- Cancel unfilled orders
- Retry with exponential backoff

**Database Failure:**
- In-memory fallback for operations
- Queue writes for later
- Alert operator

---

## Monitoring & Observability

**Logs:**
- `logs/arbitrage.log` - Main application log
- Structured logging for analysis
- Error tracking with stack traces

**Database:**
- All opportunities logged
- All executions tracked
- Performance metrics calculated
- Queryable for analysis

**Dashboard:**
- Real-time opportunity feed
- Execution history
- P&L tracking
- System health

---

## Configuration

**Environment-Based:**
- `development` - Simulation mode, mock data
- `simulation` - Real prices, no execution
- `production` - Live trading

**Key Parameters:**
- `min_spread_pct` - Minimum profit threshold
- `max_position_size_usd` - Max per trade
- `execution_timeout_seconds` - Order timeout
- `max_daily_loss_usd` - Circuit breaker threshold

---

## Technology Stack

| Component | Technology |
|-----------|------------|
| Exchange API | CCXT / CCXT Pro |
| Async Runtime | asyncio |
| Database | SQLite 3 |
| Web Dashboard | Flask |
| Configuration | Pydantic + YAML |
| Logging | Python logging |
| Testing | pytest |

---

## Future Enhancements

**Phase 2 (Optional):**
1. Machine learning for opportunity prediction
2. DEX integration (Uniswap, PancakeSwap)
3. Triangular arbitrage within single exchange
4. Statistical arbitrage (pairs trading)
5. Options arbitrage
6. Telegram/email alerts
7. Advanced dashboard with charts

---

**Built for speed, safety, and profitability.** ₿


