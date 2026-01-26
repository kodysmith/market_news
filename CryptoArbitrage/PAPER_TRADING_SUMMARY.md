# Paper Trading System - Summary

## What Was Built

A comprehensive paper trading and opportunity analysis system for crypto arbitrage that lets you:

1. ✅ **Test the system without real money**
2. ✅ **Track exactly how many opportunities occur per day**
3. ✅ **Measure how long opportunities stay open**
4. ✅ **Analyze which pairs and exchanges are most profitable**
5. ✅ **Build confidence before going live**

---

## Quick Start (3 Steps)

### Step 1: Run Paper Trading (10 minutes)

```bash
cd /mnt/4tb/stock_scanner/market_news/CryptoArbitrage
./run_paper_trading.sh 10
```

This will:
- Connect to exchanges (read-only)
- Monitor real prices
- Detect arbitrage opportunities
- Track opportunity lifecycles
- Simulate executions
- Log everything to database

### Step 2: Analyze Results

```bash
./analyze_opportunities.sh
```

This shows:
- 📊 **Opportunities per day**: Total count, daily breakdown
- ⏱️ **Duration analysis**: How long opportunities stay valid
- 💰 **Best pairs**: Which cryptocurrencies have most opportunities
- 🔀 **Best routes**: Which exchange combinations are optimal
- 📈 **Spread distribution**: Profitability ranges

### Step 3: Review & Decide

Based on the analysis, you'll know:
- ✅ If enough opportunities exist to be profitable
- ✅ If execution windows are realistic (> 200ms)
- ✅ If spreads are large enough (> 0.7% after fees)
- ✅ Which pairs to focus on
- ✅ Whether to proceed with live trading

---

## What Was Added

### 1. Enhanced Database Schema
**File:** `src/database/schema.sql`

Added lifecycle tracking fields to opportunities table:
- `first_seen_at` - When opportunity first detected
- `last_seen_at` - Last time opportunity was still valid
- `expired_at` - When opportunity closed
- `opportunity_duration_ms` - Total time opportunity existed
- `times_seen` - How many detection cycles caught it

### 2. Opportunity Lifecycle Tracker
**File:** `src/monitoring/opportunity_lifecycle_tracker.py`

Tracks opportunities from detection through expiration:
- Detects when new opportunities appear
- Updates existing opportunities still valid
- Marks opportunities as expired when closed
- Calculates duration metrics
- Provides statistics (avg, median, percentiles)

### 3. Paper Trading Script
**File:** `scripts/paper_trading.py`

Full simulation mode with:
- Real exchange connections (read-only)
- Real-time opportunity detection
- Lifecycle tracking for all opportunities
- Realistic execution simulation:
  - 150ms execution latency
  - 0.05% slippage
  - 85% fill success rate
- Detailed logging
- Session statistics every 60 seconds
- Can run for specific duration or indefinitely

### 4. Analytics Script
**File:** `scripts/analyze_opportunities.py`

Comprehensive analysis showing:
- Daily opportunity counts
- Duration statistics (avg, median, percentiles)
- Duration distribution buckets
- Longest-lasting opportunities
- Top pairs by opportunity count
- Top exchange routes
- Spread distribution
- Recent opportunities

### 5. Convenience Scripts
**Files:** `run_paper_trading.sh`, `analyze_opportunities.sh`

Easy-to-use wrappers:
```bash
# Run for 10 minutes
./run_paper_trading.sh 10

# Run until stopped
./run_paper_trading.sh

# Analyze results
./analyze_opportunities.sh
```

### 6. Documentation
**File:** `PAPER_TRADING_GUIDE.md`

Complete guide covering:
- How paper trading works
- Understanding the output
- Key questions answered
- Recommended testing workflow
- Interpreting results for live trading
- Optimizing strategy parameters
- Troubleshooting
- Database queries

---

## Example Output

### During Paper Trading Session

```
🎯 Found 3 opportunities

🆕 New opportunity tracked: BTC/USDT:binance->coinbase - BTC/USDT @ 0.73%

✅ Simulated execution SUCCESS: BTC/USDT
   Buy 0.02315432 @ $43,215.50 on binance
   Sell 0.02315432 @ $43,547.30 on coinbase
   Profit: $7.68 (0.73%)
   Execution time: 152ms

⏱️  Opportunity expired: BTC/USDT:binance->coinbase
    Duration: 850ms (0.85s), Seen 3 times
```

### Session Statistics (Every 60 seconds)

```
📊 SESSION STATISTICS
═══════════════════════════════════════════════════
Runtime: 5.2 minutes
Opportunities detected: 47
Unique opportunities tracked: 12
Opportunities expired: 10
Active opportunities: 2
Opportunities executed (simulated): 8
Total simulated profit: $64.32

OPPORTUNITY LIFECYCLE METRICS:
  Average duration: 1.23 seconds
  Median duration (P50): 850 ms
  P90 duration: 2340 ms
  P95 duration: 3120 ms
  Min duration: 110 ms
  Max duration: 4560 ms
═══════════════════════════════════════════════════
```

### Analysis Output

```
📊 OPPORTUNITIES PER DAY (Last 30 Days)
────────────────────────────────────────────────────
Total opportunities: 142
Average per day: 14.2
Total executed: 23
Execution rate: 16.2%

Recent days:
  2025-10-09: 47 opportunities, avg spread: 0.68%, 8 executed
  2025-10-08: 31 opportunities, avg spread: 0.72%, 5 executed

⏱️  OPPORTUNITY DURATION ANALYSIS
────────────────────────────────────────────────────
Total opportunities with duration data: 38

Duration Statistics:
  Average: 1.45 seconds (1450 ms)
  Median:  0.92 seconds (920 ms)
  Min:     0.11 seconds (110 ms)
  Max:     4.56 seconds (4560 ms)

Duration Percentiles:
  P10: 0.23s
  P25: 0.48s
  P50: 0.92s
  P75: 1.87s
  P90: 2.96s
  P95: 3.52s
  P99: 4.45s

Duration Distribution:
  < 100ms        :    2 ( 5.3%) ██
  100ms - 500ms  :   12 (31.6%) ███████████████
  500ms - 1s     :    8 (21.1%) ██████████
  1s - 2s        :    9 (23.7%) ███████████
  2s - 5s        :    6 (15.8%) ███████
  5s - 10s       :    1 ( 2.6%) █
  > 10s          :    0 ( 0.0%)

Longest-lasting opportunities:
  1. BTC/USDT via binance -> coinbase
     Duration: 4.56s, Spread: 0.82%, Seen: 15 times
  2. ETH/USDT via kraken -> coinbase
     Duration: 3.87s, Spread: 0.91%, Seen: 12 times
```

---

## Key Insights You'll Get

### 1. Opportunity Frequency
**Question:** "How many opportunities per day?"

**Answer:** The analysis shows exact counts:
- Total per day
- Daily breakdown
- Trends over time
- Which pairs contribute most

**Example insight:** "14 opportunities per day, mostly BTC/USDT and ETH/USDT"

### 2. Opportunity Duration
**Question:** "How fast do I need to execute?"

**Answer:** Duration analysis shows:
- Average and median duration
- Distribution (what % are fast vs slow)
- Percentiles (P50, P90, P95)

**Example insight:** "Median 920ms window means 200ms execution is feasible"

### 3. Profitability
**Question:** "Are spreads large enough?"

**Answer:** Spread analysis shows:
- Average spread after fees
- Distribution across ranges
- Historical max spreads

**Example insight:** "Average 0.72% spread, comfortable above 0.5% threshold"

### 4. Best Targets
**Question:** "Which pairs/exchanges should I focus on?"

**Answer:** Pair and route analysis shows:
- Most frequent pairs
- Best exchange combinations
- Consistency over time

**Example insight:** "BTC/USDT Binance->Coinbase has 40% of all opportunities"

---

## Database Schema

All data is stored in SQLite for easy querying:

```sql
-- Opportunities table (with lifecycle tracking)
CREATE TABLE opportunities (
    id TEXT PRIMARY KEY,
    detected_at DATETIME,
    pair TEXT,
    buy_exchange TEXT,
    sell_exchange TEXT,
    net_spread_pct REAL,
    
    -- NEW: Lifecycle fields
    first_seen_at DATETIME,
    last_seen_at DATETIME,
    expired_at DATETIME,
    opportunity_duration_ms REAL,
    times_seen INTEGER,
    
    -- ... other fields
);
```

Query directly:
```bash
sqlite3 data/arbitrage.db
```

---

## Recommended Workflow

### Phase 1: Quick Test (10 min)
```bash
./run_paper_trading.sh 10
./analyze_opportunities.sh
```
**Goal:** Verify system works

### Phase 2: Extended Test (2-4 hours)
```bash
./run_paper_trading.sh 120
```
**Goal:** Statistical significance

### Phase 3: Full Day (24 hours)
```bash
./run_paper_trading.sh
# Let it run, Ctrl+C when done
```
**Goal:** Daily patterns

### Phase 4: Multi-Day (3-7 days)
```bash
screen -S paper_trading
./run_paper_trading.sh
# Ctrl+A, D to detach
```
**Goal:** Market condition validation

---

## Next Steps

After paper trading, you'll have data to:

1. **Decide if live trading is viable**
   - Are there enough opportunities?
   - Are spreads profitable?
   - Are execution windows realistic?

2. **Optimize strategy parameters**
   - Adjust min_spread_pct
   - Focus on best pairs
   - Configure position sizing

3. **Estimate expected returns**
   - Opportunities per day × avg spread × success rate
   - Build realistic profit projections

4. **Start small with real money**
   - $100-500 initial capital
   - Compare live vs paper results
   - Scale gradually

---

## Files Created

```
CryptoArbitrage/
├── src/
│   ├── database/
│   │   └── schema.sql (enhanced)
│   └── monitoring/
│       └── opportunity_lifecycle_tracker.py (new)
├── scripts/
│   ├── paper_trading.py (new)
│   └── analyze_opportunities.py (new)
├── run_paper_trading.sh (new)
├── analyze_opportunities.sh (new)
├── PAPER_TRADING_GUIDE.md (new)
└── PAPER_TRADING_SUMMARY.md (this file)
```

---

## Support

- **Full guide:** See `PAPER_TRADING_GUIDE.md`
- **Main README:** See `README.md`
- **Configuration:** See `config/config.yaml`
- **Logs:** Check `logs/paper_trading.log`

---

## Ready to Test?

```bash
cd /mnt/4tb/stock_scanner/market_news/CryptoArbitrage

# Quick 10-minute test
./run_paper_trading.sh 10

# View results
./analyze_opportunities.sh
```

**That's it! You'll immediately see:**
- How many opportunities exist
- How fast they close
- Which pairs are best
- If the strategy is viable

---

**Happy Testing! 📝💰🚀**


