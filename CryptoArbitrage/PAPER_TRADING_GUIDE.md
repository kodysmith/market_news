# Paper Trading Guide

Complete guide for testing the crypto arbitrage system without risking real money.

---

## What is Paper Trading?

Paper trading runs the arbitrage system with:
- ✅ **Real exchange connections** (read-only)
- ✅ **Real price data** (live websockets)
- ✅ **Real opportunity detection**
- ✅ **Simulated trade execution** (no real orders)
- ✅ **Performance tracking** (as if trades were real)

**Perfect for:**
- Testing the system before going live
- Understanding how many opportunities exist
- Measuring opportunity lifecycles
- Optimizing strategy parameters
- Building confidence in the system

---

## Quick Start

### 1. Run Paper Trading

**Run for 10 minutes:**
```bash
./run_paper_trading.sh 10
```

**Run indefinitely (until Ctrl+C):**
```bash
./run_paper_trading.sh
```

The system will:
1. Connect to exchanges (read-only)
2. Monitor prices via websockets
3. Detect arbitrage opportunities
4. Track how long each opportunity stays open
5. Simulate execution with realistic assumptions
6. Log everything to database

### 2. Analyze Results

```bash
./analyze_opportunities.sh
```

This shows:
- 📊 Opportunities per day
- ⏱️ How long opportunities stay open
- 💰 Best trading pairs
- 🔀 Best exchange routes
- 📈 Spread distributions

---

## Understanding the Output

### During Paper Trading

```
🎯 Found 3 opportunities
✅ Simulated execution SUCCESS: BTC/USDT
   Buy 0.02315432 @ $43,215.50 on binance
   Sell 0.02315432 @ $43,547.30 on coinbase
   Profit: $7.68 (0.73%)
   Execution time: 152ms

🆕 New opportunity tracked: BTC/USDT:binance->coinbase - BTC/USDT @ 0.73%
⏱️  Opportunity expired: BTC/USDT:binance->coinbase - Duration: 850ms (0.85s), Seen 3 times
```

**Key metrics:**
- **Opportunities detected**: How many times a profitable spread was found
- **Unique opportunities**: Distinct pair/exchange combinations
- **Duration**: How long the spread stayed profitable
- **Times seen**: How many detection cycles caught the same opportunity

### Session Statistics (printed every 60 seconds)

```
📊 SESSION STATISTICS
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
```

### Analysis Output

```
📊 OPPORTUNITIES PER DAY (Last 30 Days)
Total opportunities: 142
Average per day: 14.2
Total executed: 23
Execution rate: 16.2%

Recent days:
  2025-10-09: 47 opportunities, avg spread: 0.68%, 8 executed
  2025-10-08: 31 opportunities, avg spread: 0.72%, 5 executed

⏱️  OPPORTUNITY DURATION ANALYSIS
Total opportunities with duration data: 38
Average: 1.45 seconds (1450 ms)
Median:  0.92 seconds (920 ms)
```

---

## Key Questions Answered

### 1. How Many Opportunities Per Day?

The analysis shows:
- Total opportunities detected
- Average per day
- Daily breakdown with trends

**What to look for:**
- More than 5-10 opportunities per day = good market conditions
- Higher spreads (>0.7%) = more profitable opportunities
- Consistent daily counts = stable arbitrage environment

### 2. How Long Do Opportunities Stay Open?

The duration analysis shows:
- Average and median duration
- Percentile breakdown (P50, P90, P95)
- Duration distribution histogram

**What to look for:**
- **< 500ms**: Very fast, need low-latency execution
- **500ms - 2s**: Good window for execution
- **> 2s**: Comfortable execution window
- **> 5s**: Unusual, may indicate stale markets

**Key insight:** If most opportunities are < 500ms, you need:
- Faster execution infrastructure
- Lower min_spread_pct threshold
- Better network connectivity to exchanges

### 3. Which Pairs/Exchanges Are Best?

The pair and route analysis shows:
- Most frequent pairs (BTC/USDT, ETH/USDT, etc.)
- Best exchange combinations (e.g., Binance -> Coinbase)
- Average spreads and durations by pair/route

**What to look for:**
- Focus on pairs with consistent opportunities
- Note which exchange routes have best spreads
- Consider removing pairs with zero opportunities

### 4. Are Spreads Large Enough?

The spread distribution shows:
- Average and median spreads
- Distribution across ranges
- Historical maximums

**What to look for:**
- Most spreads should be > min_spread_pct setting
- If clustered near threshold, consider lowering it
- Larger spreads (>1%) offer more profit cushion

---

## Simulation Parameters

The paper trading simulator uses realistic assumptions:

| Parameter | Value | Notes |
|-----------|-------|-------|
| Execution latency | 150ms | Average time to execute both legs |
| Slippage | 0.05% | Price movement during execution |
| Fill success rate | 85% | Percentage of trades that complete |
| Position size | 90% of max | Conservative sizing |

These match real-world crypto arbitrage conditions.

---

## Recommended Testing Workflow

### Phase 1: Quick Test (10-30 minutes)
```bash
# Run for 10 minutes
./run_paper_trading.sh 10

# Analyze results
./analyze_opportunities.sh
```

**Goal:** Verify system works and opportunities exist

### Phase 2: Extended Test (2-4 hours)
```bash
# Run for 2 hours (120 minutes)
./run_paper_trading.sh 120
```

**Goal:** Get statistical significance on opportunity frequency and duration

### Phase 3: Full Day Test (24 hours)
```bash
# Run indefinitely (let it run overnight)
./run_paper_trading.sh
```

**Goal:** Understand daily patterns, measure consistency

### Phase 4: Multi-Day Test (3-7 days)
```bash
# Run for a week to capture different market conditions
# Use tmux or screen to keep it running
screen -S paper_trading
./run_paper_trading.sh
# Ctrl+A, D to detach
```

**Goal:** Validate strategy across market conditions

---

## Reading the Database Directly

You can query the SQLite database directly:

```bash
sqlite3 data/arbitrage.db
```

**Useful queries:**

```sql
-- Opportunities per day
SELECT DATE(detected_at) as date, COUNT(*) as count
FROM opportunities
GROUP BY DATE(detected_at)
ORDER BY date DESC;

-- Average duration by pair
SELECT pair, 
       COUNT(*) as opportunities,
       AVG(opportunity_duration_ms)/1000 as avg_duration_seconds,
       AVG(net_spread_pct) as avg_spread
FROM opportunities
WHERE opportunity_duration_ms IS NOT NULL
GROUP BY pair
ORDER BY opportunities DESC;

-- Fastest opportunities
SELECT pair, buy_exchange, sell_exchange, 
       opportunity_duration_ms as duration_ms,
       net_spread_pct
FROM opportunities
WHERE opportunity_duration_ms IS NOT NULL
ORDER BY opportunity_duration_ms ASC
LIMIT 10;

-- Longest-lasting opportunities
SELECT pair, buy_exchange, sell_exchange, 
       opportunity_duration_ms/1000 as duration_seconds,
       net_spread_pct,
       times_seen
FROM opportunities
WHERE opportunity_duration_ms IS NOT NULL
ORDER BY opportunity_duration_ms DESC
LIMIT 10;

-- Execution success rate (simulated)
SELECT 
    COUNT(*) as total_executions,
    SUM(CASE WHEN is_successful = 1 THEN 1 ELSE 0 END) as successful,
    AVG(CASE WHEN is_successful = 1 THEN 1.0 ELSE 0.0 END) * 100 as success_rate,
    SUM(realized_profit_usd) as total_profit
FROM executions;
```

---

## Interpreting Results for Live Trading

### Good Indicators (Ready for Live Trading)

✅ **10+ opportunities per day** across multiple pairs
✅ **Average duration > 500ms** (execution window exists)
✅ **Average spread > 0.7%** (profitable after fees + slippage)
✅ **85%+ simulated success rate** (good execution conditions)
✅ **Consistent across days** (not just lucky timing)

### Warning Signs (Need More Testing)

⚠️ **< 5 opportunities per day** (market too efficient)
⚠️ **Average duration < 200ms** (need faster execution)
⚠️ **Average spread < 0.6%** (margins too thin)
⚠️ **< 70% success rate** (execution issues likely)
⚠️ **Highly variable day-to-day** (unstable conditions)

### Red Flags (Don't Trade Yet)

❌ **Zero opportunities** (system misconfigured or markets too efficient)
❌ **All opportunities < 100ms** (impossible to execute profitably)
❌ **Spreads clustered at threshold** (need to lower min_spread_pct)
❌ **< 50% success rate** (something wrong with logic)

---

## Optimizing Strategy Parameters

Based on paper trading results, tune these settings in `config/config.yaml`:

### If opportunities are rare (< 5/day):
```yaml
strategy:
  min_spread_pct: 0.3  # Lower from 0.5
  trading_pairs:
    # Add more pairs
```

### If opportunities are too fast (< 200ms average):
```yaml
strategy:
  min_spread_pct: 0.7  # Raise to catch longer-lasting opportunities
  execution_timeout_seconds: 1  # Faster timeout
```

### If too many opportunities (overwhelming):
```yaml
strategy:
  min_spread_pct: 0.8  # Raise threshold
  # Focus on best pairs only
```

---

## Next Steps

After successful paper trading:

1. **Review all results carefully**
   - Understand opportunity patterns
   - Verify assumptions match reality

2. **Start with small capital** ($100-500)
   - Test real execution
   - Verify fees match expectations

3. **Monitor closely** (first 24-48 hours)
   - Compare live vs. paper results
   - Watch for execution issues

4. **Scale gradually**
   - Increase capital slowly
   - Monitor performance continuously

---

## Logs and Data

All data is saved to:
- **Database**: `data/arbitrage.db` (SQLite)
- **Logs**: `logs/paper_trading.log` (detailed logs)
- **Main logs**: `logs/arbitrage.log` (general system logs)

**Viewing logs:**
```bash
# Real-time logs
tail -f logs/paper_trading.log

# Search for opportunities
grep "opportunity detected" logs/paper_trading.log

# Search for executions
grep "Simulated execution" logs/paper_trading.log
```

---

## Troubleshooting

### No opportunities detected

**Possible causes:**
1. Min spread threshold too high
2. Markets too efficient (crypto arbitrage is competitive)
3. Exchange connections not working
4. Not enough trading pairs configured

**Solutions:**
- Lower `min_spread_pct` to 0.3% temporarily to see if any exist
- Check logs for exchange connection errors
- Verify websocket connections: `grep "websocket" logs/paper_trading.log`
- Add more trading pairs in config

### All opportunities expire instantly

**Possible causes:**
1. Price data arriving slowly (latency)
2. Detection cycle too slow (100ms default)
3. Spread closes quickly (competitive market)

**Solutions:**
- Check network latency to exchanges
- Monitor `price_staleness_ms` in opportunities
- Consider co-location near exchanges for live trading

### Simulated success rate too low

**Note:** Paper trading uses 85% success rate intentionally (realistic)

If your simulated rate is much lower, check:
- Risk manager rejecting trades (check logs for rejections)
- Balance validation failing

---

## Questions?

Check the main README: `README.md`

Or review:
- System architecture: See `README.md` "Architecture" section
- Configuration: See `config/config.yaml` with comments
- Database schema: See `src/database/schema.sql`

---

**Happy Paper Trading! 📝💰**


