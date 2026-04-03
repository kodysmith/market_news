# Chapter 20: Designing the Elite Small Cap Momentum Bot

## 20.1 Why a Bot Can Beat a Human at This

Ross Cameron makes $400K-$1.1M/year trading small cap momentum. He has 10+ years of experience, a 68% win rate, and trades ~2,000 times per year. He is in the top 0.1% of day traders.

But Cameron still has bad days. He still revenge trades occasionally. He still overtrads when he's up big. He still gets tired at 11 AM after a stressful morning. He still misses setups because he's watching the wrong stock.

A bot eliminates every single one of these failure modes:

| Human Weakness | Bot Advantage |
|---------------|---------------|
| Revenge trading after a loss | Impossible — bot follows rules mechanically |
| FOMO chasing a runner | Impossible — only enters at planned levels |
| Overtrading on a good day | Hard-capped at N trades/day |
| Sizing up after wins | Position size is calculated, not felt |
| Moving stops when scared | Stop is a broker order, not a mental note |
| Missing setups (looking elsewhere) | Scans ALL qualifying stocks simultaneously |
| Fatigue at 11 AM | Runs at same performance at 3:59 PM as at 9:30 AM |
| Can't process 50 stocks at once | Can evaluate hundreds per second |
| Emotional attachment to a stock | None — every stock is a data structure |

The human edge: pattern recognition in ambiguous situations, reading Level 2 order flow nuance, interpreting news context, knowing "this feels wrong." But these can be partially captured in rules, and the bot's structural advantages more than compensate for the edges it can't replicate.

**The thesis: if we encode Cameron's decision-making process into systematic rules, the bot should achieve HIGHER consistency (fewer catastrophic days) even if individual trade quality is slightly lower. The bot won't have Cameron's best days ($50K+), but it won't have his worst days (-$10K) either. The Sharpe will be higher.**

## 20.2 The Five-Layer Architecture

```
Layer 1: UNIVERSE SCANNER (pre-market, 7:00-9:25 AM)
  → Identifies 5-15 candidates from 8,000+ stocks
  
Layer 2: CATALYST CLASSIFIER (pre-market, 7:00-9:25 AM)
  → Ranks candidates by catalyst quality and news severity
  
Layer 3: ENTRY ENGINE (9:30-10:30 AM)
  → Detects patterns on 1-min bars: first pullback, ORB, VWAP reclaim
  → Executes with limit orders at calculated prices
  
Layer 4: POSITION MANAGER (9:30-11:30 AM)
  → Manages stops, scales out, trails runners
  → Enforces per-trade and daily risk limits
  
Layer 5: RISK GOVERNOR (always on)
  → Kill switch on daily P&L limits
  → Prevents correlated positions
  → Adjusts sizing based on recent performance
```

## 20.3 Layer 1: The Universe Scanner

**What it does:** Every morning at 7:00 AM, scan all US equities for gap-up candidates. Filter aggressively. Output a ranked watchlist of 5-15 stocks.

**Filter criteria:**

| Filter | Threshold | Why |
|--------|-----------|-----|
| Gap % | > 8% | Below 8%, the move is too small for momentum scalping |
| Pre-market volume | > 500K shares | Confirms institutional/retail interest |
| Relative volume | > 5x average | Extreme interest vs normal |
| Float | 2M - 30M shares | Sweet spot: enough liquidity, low enough for big moves |
| Price | $3 - $20 | Enough room to move; enough liquidity for entries/exits |
| Avg daily volume | > 300K shares | Can exit when needed |
| Spread | < $0.05 or < 0.5% | Friction must be manageable |
| Shares outstanding | Verified (not OTC) | Filter out shell companies |

**Data source:** Polygon.io grouped daily bars (1 API call) + pre-market bars for candidates. Cost: ~$200/month for the starter plan.

**Output per candidate:**
```
{
  "symbol": "ABCD",
  "gap_pct": 15.2,
  "pre_market_vol": 1_200_000,
  "rel_vol": 8.3,
  "float": 12_000_000,
  "short_interest_pct": 22.4,
  "price": 8.50,
  "spread": 0.03,
  "atr_14d": 0.85,
  "pre_market_high": 9.20,
  "pre_market_low": 7.80,
  "prev_close": 7.38,
  "catalyst": null,  // filled by Layer 2
  "scan_rank": 1
}
```

## 20.4 Layer 2: The Catalyst Classifier

**What it does:** For each scanner candidate, pull recent news and classify the catalyst. This is where our existing news bot infrastructure pays off.

**Implementation:** Use the news aggregator (`news_bot/news_aggregator.py`) to pull headlines for each candidate. Feed through the classifier (`news_bot/classifier.py`) to determine catalyst type and severity. Optionally use Claude API for nuanced headline interpretation.

**Catalyst scoring:**

| Catalyst Type | Score | Action |
|--------------|-------|--------|
| Earnings beat + guidance raise | 10 | Trade with full size |
| FDA approval / major contract | 9 | Trade with full size |
| Earnings beat (no guidance) | 7 | Trade with standard size |
| Significant partnership | 6 | Trade with standard size |
| Analyst upgrade | 4 | Trade with half size |
| Sector sympathy | 3 | Trade with half size |
| Social media hype / Reddit | 2 | Skip or paper trade |
| No identifiable catalyst | 0 | **SKIP — do not trade** |

**The no-catalyst rule is the single most important filter.** Stocks that gap with no identifiable reason are either: (a) pump-and-dumps, (b) institutional block trades that will reverse, or (c) data errors. None of these are tradeable edges. Cameron himself says "I don't trade stocks without a catalyst."

**Headline interpretation examples:**
- "ABCD reports Q1 revenue up 45%, raises FY guidance" → Score 10
- "ABCD receives FDA breakthrough therapy designation" → Score 9
- "ABCD announces partnership with Microsoft" → Score 6 (need to verify size)
- "ABCD is trending on WallStreetBets" → Score 2 (skip)
- No news found → Score 0 (skip)

## 20.5 Layer 3: The Entry Engine

This is the core trading logic. It monitors 1-minute bars for the top 3-5 candidates and detects entry patterns.

### Pattern 1: First Pullback (Primary Strategy)

This is Cameron's bread and butter. The implementation follows the logic already in `backtest_warrior_gap_pullback.py`, enhanced with additional filters.

```
ENTRY RULES:
  1. Opening range (9:30-9:44): mark OR high and OR low
  2. Wait for breakout: first bar after 9:45 that closes above OR high
  3. Wait for pullback: first red candle after the breakout
  4. Confirm pullback holds: green candle closes above pullback low
  5. ENTRY: buy at the open of the bar after the green confirmation candle
  
  FILTERS (must ALL be true):
  - Catalyst score >= 6
  - Breakout candle volume > 2x average of OR candles
  - Pullback retraces < 50% of the breakout move
  - Pullback volume < breakout volume (sellers exhausting)
  - Stock is above VWAP at entry
  - Current time is before 10:30 AM
  - No halt has occurred (halted stocks are unpredictable at re-open)
  
  STOP: Below the pullback candle low (or VWAP, whichever is tighter)
  TARGET: Sell 1/3 at 2R, 1/3 at 3R, trail 1/3 below each new higher low
```

### Pattern 2: VWAP Reclaim

```
ENTRY RULES:
  1. Stock gapped up but has sold off below VWAP in the first 15-30 minutes
  2. Stock consolidates below VWAP for at least 10 minutes
  3. Stock crosses above VWAP with a volume surge (2x+ average bar volume)
  4. Entry on the candle that closes above VWAP

  FILTERS:
  - Catalyst score >= 6
  - Volume on the reclaim bar > 2x the consolidation average
  - Below-VWAP consolidation was "orderly" (no panic selling, declining vol)
  - Stock reclaims VWAP before 10:15 AM (late reclaims are less reliable)
  
  STOP: $0.10-0.20 below VWAP
  TARGET: Pre-market high, or 3R
```

### Pattern 3: Flat Top Breakout

```
ENTRY RULES:
  1. Stock hits the same price level 2+ times (within $0.05)
  2. Each pullback from that level makes a higher low
  3. Volume builds on each test of the flat top
  4. Entry: buy when price closes above the flat top with volume

  FILTERS:
  - At least 2 tests of the flat top
  - Flat top formation took 5-20 minutes (too fast = not enough compression)
  - Stock is above VWAP
  - Volume on breakout > volume on any individual test
  
  STOP: Below the last higher low
  TARGET: Height of the pattern added to breakout, or 2R minimum
```

### Entry Prioritization

When multiple patterns trigger on multiple stocks simultaneously, the bot must choose. Priority order:

1. **Highest catalyst score** (fundamental quality)
2. **Highest relative volume** (more interest = more follow-through)
3. **Tightest stop** (better R-multiple = better expected value)
4. **First pullback > VWAP reclaim > flat top** (first pullback has best historical win rate)

Maximum concurrent positions: 2. This prevents over-diversification and ensures focus.

## 20.6 Layer 4: The Position Manager

Once a trade is entered, the position manager handles all exit logic.

### Exit Rules

```
IMMEDIATE STOP LOSS:
  - Hard stop at the pre-defined level (below pullback low or VWAP)
  - This is a bracket order placed simultaneously with the entry
  - NEVER moved further away from entry
  - Can be moved TOWARD entry (trailing) once 1R is reached

SCALE-OUT SCHEDULE:
  At +1R: Sell 1/3 of position. Move stop to breakeven on remaining shares
  At +2R: Sell 1/3 of position. Trail stop below last 1-min higher low
  Runner: Trail remaining 1/3 below each new 1-min higher low
  
  If the runner gets stopped on the trail: position is fully closed
  Total average exit (when trade wins): ~2.2R

TIME STOP:
  If trade hasn't reached +1R within 15 minutes: close at market
  Dead trades tie up capital. The momentum window is 5-15 minutes.
  If it hasn't moved by then, the thesis is wrong.

HALT HANDLING:
  If stock halts UP while in position: hold through. Re-open usually higher
  If stock halts DOWN while in position: queue a sell order for re-open
  Do not enter new positions in halted stocks

END OF DAY:
  All positions closed by 11:30 AM. The bot does NOT trade afternoons.
  This is aggressive but intentional — 80% of small cap momentum
  opportunity is in the first 2 hours. Afternoon small cap trading
  is where most retail losses happen.
```

### Trailing Stop Logic

The trailing stop is the difference between a 2R average exit and a 5R+ average exit on runners. Implementation:

```python
def update_trail(current_trail, new_bar, direction="long"):
    """Update trailing stop after each 1-min bar."""
    if direction == "long":
        # Trail below each new higher low
        new_low = new_bar["Low"]
        if new_low > current_trail:
            return new_low - 0.02  # $0.02 buffer below the low
    return current_trail  # don't move trail down
```

The key insight: only move the trail UP (for longs). Never down. The trail ratchets in one direction only, locking in progressively more profit.

## 20.7 Layer 5: The Risk Governor

The risk governor enforces absolute limits that cannot be overridden by any other layer.

### Per-Trade Limits

```
MAX_RISK_PER_TRADE = 1% of account ($2,000 on $200K)
MAX_POSITION_VALUE = 20% of account ($40,000)
MAX_SHARES = min(risk_budget / stop_distance, max_position / price)
```

### Daily Limits

```
MAX_DAILY_LOSS = 3% of account ($6,000)
MAX_TRADES_PER_DAY = 6
MAX_CONSECUTIVE_LOSSES = 3 (then pause for 30 minutes)
MAX_CONCURRENT_POSITIONS = 2
```

### Weekly/Monthly Limits

```
MAX_WEEKLY_LOSS = 5% of account ($10,000) → paper trade rest of week
MAX_MONTHLY_DRAWDOWN = 8% → reduce to 0.5% risk until 50% recovered
```

### Correlation Guard

```
Do not enter two stocks in the same sector simultaneously.
If STOCK_A is a biotech gap and STOCK_B is also biotech,
only trade the one with the higher catalyst score.
Correlated positions double your risk without doubling your edge.
```

### Performance-Adaptive Sizing

```python
def adaptive_risk_pct(recent_trades: list[float], base_risk: float = 0.01):
    """Adjust risk based on recent performance."""
    if len(recent_trades) < 20:
        return base_risk * 0.5  # learning phase: half size
    
    last_20 = recent_trades[-20:]
    win_rate = sum(1 for t in last_20 if t > 0) / len(last_20)
    
    if win_rate >= 0.60:
        return base_risk  # full size: strategy is working
    elif win_rate >= 0.45:
        return base_risk * 0.75  # reduce: below expectations
    else:
        return base_risk * 0.25  # minimal: something is wrong, preserve capital
```

## 20.8 What the Bot Can't Do (and How to Compensate)

### Level 2 / Tape Reading

Cameron reads Level 2 to see large buyers/sellers stacking at specific levels. A bot can't interpret L2 with the same nuance, but it can:
- Monitor the bid/ask size ratio (large bid = buying support)
- Detect "walls" (orders > 10x average size at a price level)
- Track aggressive buying (trades at the ask) vs selling (trades at the bid)
- Flag when a previously visible large order disappears (could be spoofing)

These are simpler than human L2 reading but capture 60-70% of the signal.

### Novel Market Conditions

Cameron adjusts his approach when something unprecedented happens (pandemic, flash crash, meme stock mania). A bot trades the same rules regardless of context.

Compensation: the risk governor's adaptive sizing automatically reduces exposure when win rate drops. If a novel condition causes losses, the bot shrinks size before catastrophic damage occurs. It won't adapt as fast as Cameron, but it won't blow up either.

### "This Feels Wrong" Intuition

Cameron sometimes exits a position before his stop because "something feels off." This is pattern recognition that's difficult to codify.

Compensation: the 15-minute time stop captures much of this. If a trade isn't working within 15 minutes, the bot exits regardless. This is cruder than intuition but systematic.

## 20.9 Expected Performance

Based on the structural analysis and assuming the bot achieves SKILLED level (not elite):

| Metric | Human (Cameron) | Bot (Expected) |
|--------|----------------|-----------------|
| Win rate | 68% | 55-60% |
| Average R (wins) | 2.5R | 2.0-2.5R |
| Trades/year | 2,000 | 600-800 |
| Gross P&L ($200K) | ~$500K+ | $300-500K |
| Max single-day loss | -$10-30K | -$6K (hard limit) |
| Worst month | -$40K+ | -$20K (8% monthly cap) |
| Sharpe | ~1.0-1.5 | ~1.5-2.5 |
| Emotional cost | Very high | Zero |
| Time required | 4-5 hrs/day | 30 min monitoring/day |

**The bot trades fewer times** (600-800 vs 2,000) because it's more selective — it only enters when all filters pass. Cameron takes lower-quality setups that he can "read" in real-time; the bot can't read them, so it skips them. Fewer trades but higher quality = similar or better risk-adjusted returns.

**The bot has tighter drawdowns** because the risk governor is absolute. Cameron has had $30K+ drawdown days. The bot can't — it shuts off at -$6K. Over a year, this preservation of capital compounds into higher terminal wealth despite lower gross P&L.

**The bot's Sharpe is higher** because the tails are trimmed. No catastrophic days means lower variance. Lower variance with similar returns = higher Sharpe.

## 20.10 Implementation Roadmap

**Phase 1 — Scanner (1 week):**
- Integrate Polygon API for grouped daily bars + pre-market data
- Build the gap/float/RVOL scanner
- Run it in shadow mode for 2 weeks — compare scanner output to actual movers
- Validate it catches 80%+ of the stocks that actually ran

**Phase 2 — Catalyst Classifier (1 week):**
- Wire the existing news bot to the scanner output
- Add catalyst scoring (0-10 scale)
- Validate by checking historical gap days against news archives
- The classifier must achieve < 10% false positive rate on "no catalyst" detection

**Phase 3 — Entry Engine + Position Manager (2 weeks):**
- Implement first pullback, VWAP reclaim, and flat top detection on 1-min bars
- Build bracket order management (entry + stop + target as linked orders)
- Implement scale-out and trailing stop logic
- Backtest on Polygon historical 1-min data (3-6 months)

**Phase 4 — Risk Governor (1 week):**
- Implement all risk limits (per-trade, daily, weekly, monthly)
- Build the adaptive sizing logic
- Stress test against worst historical weeks (March 2020, Jan 2021 meme mania)

**Phase 5 — Paper Trading (3 months minimum):**
- Run the full system on paper with real-time data
- Target metrics: > 50% win rate, > 1.5 avg R, profit factor > 1.5
- If metrics are not met after 3 months, diagnose and rebuild before risking real capital

**Phase 6 — Live (gradual scaling):**
- Start at 0.25% risk per trade ($500)
- After 50 trades with positive expectancy: increase to 0.5% ($1,000)
- After 200 trades: increase to 1% ($2,000)
- Never skip phases. The market charges tuition to impatient traders

## 20.11 The Cost Structure

| Item | Monthly Cost |
|------|-------------|
| Polygon API (starter) | $200 |
| News API feeds (FMP + NewsAPI) | $75 |
| IBKR commissions (~800 trades/yr) | $330 |
| Claude API (catalyst classification) | $50 |
| Server (cloud or local) | $50-100 |
| **Total** | **~$700-755/month** |

At $700/month ($8,400/year) in infrastructure costs, the bot needs to generate $8,400+ just to break even. At the skilled level ($300-500K/year), the infrastructure cost is 1.7-2.8% of gross revenue — trivial.

## 20.12 Why This Bot Should Outperform Our SPY Day Trading

The SPY ORB + VWAP Bounce strategy produces $136K/year (live estimate) trading a single ETF. The small cap momentum bot should outperform because:

1. **Larger moves.** SPY moves 0.5-2% on a normal day. Small cap gappers move 10-50%+. The potential R-multiple per trade is 3-5x larger
2. **More opportunities.** SPY offers 1 ORB and maybe 1 VWAP bounce per day. The small cap scanner may surface 3-5 tradeable candidates per day
3. **Less crowded.** SPY is the most traded instrument on earth — every algo and institution is watching the same ORB and VWAP levels. Small caps have less algorithmic competition
4. **The bot's emotional advantage is larger for small caps.** Trading SPY is relatively calm (it moves $1-3). Trading a small cap that moves $2 in a minute is terrifying for humans but irrelevant to a bot. The emotional advantage of automation is proportional to the volatility of the instrument

The risk is also proportionally larger. But the risk governor caps the downside, while the upside is uncapped. This asymmetry — limited downside, unlimited upside — is exactly the profile you want for a momentum strategy.

## 20.13 The Combined System

With all three strategies running from the same $200K account:

```
7:00 AM  — Small cap scanner starts
7:30 AM  — Catalyst classifier enriches candidates  
9:15 AM  — Final watchlist locked (top 3 small caps + SPY)
9:30 AM  — Small cap entry engine active + SPY ORB monitoring
9:45 AM  — SPY ORB breakout window
10:00 AM — SPY VWAP bounce window opens
10:30 AM — Small cap positions managed/closed
11:30 AM — All day trading positions closed. Day trading done
3:50 PM  — SPX options bot enters iron condors (1 minute)
4:00 PM  — Done. Total active time: ~30 min monitoring + automated execution
```

No strategy competes with another for capital:
- Small caps use buying power during morning session
- SPY ORB uses buying power (can trade simultaneously with small caps)
- SPX options use $51K margin in the afternoon (overnight hold)

**Projected combined performance (skilled bot, $200K):**

| Strategy | Pre-Tax | After Tax (CA) |
|----------|---------|---------------|
| SPX Options (news-filtered) | $424,016 | $269,250 |
| SPY ORB + VWAP Bounce | $136,187 | $74,358 |
| Small Cap Momentum Bot | $300,000-500,000 | $163,800-273,000 |
| **TOTAL** | **$860K-$1.06M** | **$507K-$616K** |

This is the full system: automated options income + automated day trading on SPY + automated small cap momentum. One account, three strategies, three time windows, minimal correlation. The total pre-tax P&L approaches $1M/year from a $200K account — but only if the small cap bot achieves skilled-level performance, which requires building, testing, and validating before deploying real capital.
