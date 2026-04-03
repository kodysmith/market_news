# Chapter 11: Building the 4-Tier Regime Engine

## 11.1 Signal Selection

The regime engine computes a daily score from six to seven market signals. Each signal captures a different dimension of market health:

**Price Trend Signals:**

1. **SPX > SMA50 (+1) / SPX < SMA50 (-1)**
   The 50-day simple moving average captures the medium-term trend. When SPX is above SMA50, the market is in an uptrend. When below, it's in a downtrend. This is the single most important signal — most of our strategies use it as a standalone filter.

2. **SPX > SMA200 (+1) / SPX < SMA200 (-1)**
   The 200-day SMA captures the long-term trend. When SPX drops below SMA200, it signals a potentially major regime change — the kind of extended bear market that can last 6-18 months.

3. **SMA200 slope positive (+1) / negative (-1)**
   Even when SPX is above SMA200, the slope of SMA200 matters. A rising SMA200 means the long-term trend is genuinely up. A flattening or declining SMA200 means the trend is weakening even if prices haven't broken below yet.

**Volatility Signals:**

4. **VIX < 22 (+1) / VIX > 28 (-1)**
   VIX below 22 indicates normal or low market fear. VIX above 28 indicates elevated fear — the kind that produces large daily moves and breaches short strikes. The 22-28 range is neutral and doesn't contribute a positive or negative point.

5. **VIX in contango (+1) / VIX in backwardation (-1)**
   VIX term structure — the relationship between near-term and longer-term VIX — is a powerful regime indicator. Contango (near-term VIX lower than longer-term VIX) is the normal state: the market expects current calm to persist. Backwardation (near-term VIX higher than longer-term VIX) indicates that traders are hedging aggressively for near-term risk — a strong danger signal.

6. **VIX making lower highs (+1)**
   When each VIX spike is lower than the previous spike, it indicates that fear is subsiding and the market is digesting volatility. This is a recovery signal that helps confirm exits from CAUTION and BEAR.

**Flow Signal:**

7. **GEX positive (+1) / GEX negative (-1)**
   Gamma exposure (GEX) measures the net gamma position of options market makers. When GEX is positive, market makers are hedging in a way that dampens market moves (buying dips, selling rallies). When GEX is negative, market makers amplify moves (selling dips, buying rallies). Negative GEX environments are significantly more dangerous for premium sellers.

## 11.2 The Scoring System

Each signal contributes +1, 0, or -1 to the daily score. The theoretical range is -7 to +7, though in practice scores beyond -4 to +6 are rare.

**Score-to-Tier mapping:**

| Score | Tier | Action |
|-------|------|--------|
| ≤ -2 | BEAR | No trades. Park capital in SGOV. |
| -1 to +1 | CAUTION | Reduced size: 2 IC3 + 1 IC6. Wider strikes. |
| +2 to +4 | NEUTRAL | Full size: 10 IC3 + 5 IC6 + 3 FLY14. |
| ≥ +5 | BULL | Full size plus butterflies: 10 IC3 + 5 IC6 + 3 FLY14. |

The thresholds were selected based on the empirical relationship between score and next-day SPX movement:

| Score Range | Avg Next-Day Move | Prob of 2%+ Move | IC3 Win Rate |
|-------------|-------------------|------------------|-------------|
| ≤ -2 | -0.25% | 25% | 78% |
| -1 to +1 | -0.05% | 11% | 86% |
| +2 to +4 | +0.04% | 5% | 91% |
| ≥ +5 | +0.07% | 2% | 94% |

The score doesn't predict market direction (the average next-day move differences are small). What it predicts remarkably well is the probability of a large move — which is what determines iron condor success. At score ≤ -2, there's a 25% chance of a 2%+ daily move, making 10-delta iron condors borderline unprofitable. At score ≥ +5, the probability drops to 2%, making iron condors extremely safe.

## 11.3 Tier Distribution

Over the 11-year backtest period, the market spent:

| Tier | % of Trading Days | Approximate Days/Year |
|------|-------------------|----------------------|
| BULL | 22% | 55 |
| NEUTRAL | 53% | 134 |
| CAUTION | 15% | 38 |
| BEAR | 9% | 23 |

The market is in a tradeable state (BULL or NEUTRAL) 75% of the time, meaning our strategies are active on approximately 189 out of 252 trading days per year. The remaining 61 days are split between reduced trading (CAUTION, 38 days) and no trading (BEAR, 23 days).

The BEAR regime is rare but consequential. Those 23 days per year contain the majority of the strategy's potential losses. By not trading during these days, we avoid the worst outcomes while only giving up a small fraction of total trading opportunities.

**Seasonal patterns in regime distribution:**
- BEAR days cluster in specific periods: February-March and September-October historically have the highest BEAR frequency
- BULL days are concentrated in November-January and April-July
- No month is exclusively one regime — even January (traditionally bullish) has occasional CAUTION days

## 11.4 Position Sizing by Tier

The tier determines not just whether to trade, but how much:

**BULL (score ≥ +5):**
- IC3: 10 contracts
- IC6: 5 contracts
- FLY14: 3 contracts
- Total margin: ~$55K
- Character: Maximum deployment. All conditions favor premium selling.

**NEUTRAL (score +2 to +4):**
- IC3: 10 contracts
- IC6: 5 contracts
- FLY14: 3 contracts
- Total margin: ~$55K
- Character: Same as BULL. The market is "normal" — not ideal but fully tradeable.

**CAUTION (score -1 to +1):**
- IC3: 2 contracts
- IC6: 1 contract
- FLY14: 0 contracts (paused)
- Total margin: ~$10K
- Character: Minimal exposure. Wider strikes considered. Goal is to stay in the market without significant risk.

**BEAR (score ≤ -2):**
- IC3: 0 contracts
- IC6: 0 contracts
- FLY14: 0 contracts
- Capital parked in SGOV
- Character: Full capital preservation. No options exposure. Wait for recovery signals.

The position sizing in CAUTION mode (2 IC3 + 1 IC6 instead of 10 + 5) represents a 75-80% reduction in exposure. This isn't a half-measure — it's deliberately aggressive risk reduction. The logic is straightforward: if there's an 11% chance of a 2%+ move (CAUTION level), having minimal exposure limits the damage while keeping a toe in the water.

## 11.5 The Daily Computation

Every trading morning at 9:00 AM ET, the regime engine runs the following computation:

1. **Fetch T-1 data.** Yesterday's SPX close, VIX close, SMA50, SMA200, VIX term structure (front month vs. second month VIX futures), GEX estimate.

2. **Compute each signal.** Seven binary comparisons produce seven values of +1, 0, or -1.

3. **Sum the signals.** The total is the raw regime score.

4. **Apply transition rules.** The raw score determines the *potential* tier. The transition rules determine the *actual* tier based on how long the current score has persisted.

5. **Output position sizing.** The actual tier determines the contract count for each strategy layer.

The critical design choice is using **T-1 data only.** Every input to the regime score was known before today's market opened. There is no look-ahead bias. The regime is determined before the first trade is placed.

This also means the regime engine makes no intraday adjustments. If the market opens flat in NEUTRAL mode and then crashes 3% during the day, the regime engine doesn't shift to BEAR until the next morning. The stop-loss on individual positions provides intraday protection; the regime engine provides inter-day protection.

## 11.6 Implementation Walkthrough

The regime engine is implemented in `bot/regime_engine.py`. The core logic follows this flow:

```
Market Data (T-1) → Signal Computation → Raw Score → Transition Rules → Active Tier → Position Sizing
```

The state is persisted in `data/regime_state.json`, which tracks:
- Current tier
- Current score
- Days in current tier
- Consecutive recovery/deterioration day count (for transition rules)
- Historical tier transitions (for analysis)

The persistence is important because the transition rules are stateful — they depend on how many consecutive days the score has been in a particular range. A single day's score of ≥ +5 doesn't immediately trigger BULL mode; it starts a counter that must reach 20 consecutive days.

On each daily run:
1. The engine loads yesterday's state from the JSON file
2. Computes today's score from fresh market data
3. Applies the transition rules against the accumulated state
4. Updates the tier if a transition is triggered
5. Saves the new state back to the JSON file
6. Returns the position sizing parameters to the trading bot

The bot then uses these parameters to determine how many contracts to trade for each strategy layer. The regime engine has no knowledge of individual trades — it operates purely at the portfolio allocation level. Individual trade management (entries, exits, stops) is handled by the trade execution layers described in Part V.
