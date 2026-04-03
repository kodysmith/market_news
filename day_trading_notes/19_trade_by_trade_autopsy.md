# Chapter 19: Trade-by-Trade Autopsy — What Went Wrong and How to Fix It

## 19.1 The Pattern in Our Losses

Before examining individual trades, the aggregate failure patterns tell a clear story:

**Observation 1: 33 of 42 losing trades (79%) were stopped out within 5-10 minutes of entry.** The entries are triggering but immediately reversing. This means the confirmation signals (VWAP cross, candle close) are firing too early — before genuine direction is established.

**Observation 2: The tighter the stop, the worse the performance.** Trades with $0.20-0.40 stops were stopped on noise. Trades with $0.80+ stops had better R-multiples even when they lost, because the stop gave room for normal oscillation.

**Observation 3: Short trades lost money across all strategies except ORB.** This is a structural market bias issue, not a strategy issue. The market bounces sharply from oversold conditions, catching shorts repeatedly.

## 19.2 VWAP Rejection Losses (16 trades, -$17,174)

This was the worst-performing strategy. Every trade was a short, and 76% hit the stop within 5-15 minutes.

### The Common Failure Mode

The strategy looks for a stock that's been below VWAP, rallies to VWAP, and gets "rejected" (closes a candle below VWAP). The problem: in a declining market, stocks don't cleanly reject VWAP — they oscillate around it for bars before deciding direction.

**Example: 2026-02-11 | SHORT at $692.82 | Stopped at $693.09 | Loss: -$1,183 | Hold: 10 min**
- Why we entered: SPY was below VWAP for 4+ bars, rallied to VWAP, red candle formed
- Why it failed: The "rejection" was just a pause in an intraday rally. SPY pushed through VWAP $0.27 higher and triggered the stop. What looked like seller rejection was actually buyer absorption — the buyers at VWAP were stronger than they appeared
- How to avoid: Require the rejection candle to close in the bottom 25% of its range (showing real seller conviction, not just a small red body near VWAP)

**Example: 2026-03-11 | SHORT at $677.35 | Stopped at $677.73 | Loss: -$1,132 | Hold: 5 min**
- Why we entered: SPY below VWAP, rejection candle formed
- Why it failed: The $0.38 stop was within normal 5-minute noise. SPY oscillated $0.50 around VWAP multiple times before eventually dropping. The stop was correct in direction but too tight in magnitude
- How to avoid: Widen the stop to at least 1x ATR(14) above VWAP, not just $0.20 fixed. On a $670 stock with $0.80 ATR, the stop should be $0.80+ above VWAP

**The meta-lesson: VWAP Rejection is trying to catch the exact top of a bounce, which requires near-perfect timing. A $0.20-0.40 stop on SPY at VWAP is fighting against normal 5-minute bar range ($0.30-0.80). The signal triggers, but the stop can't survive the noise between the signal and the actual move.**

### Fix: Kill the strategy or rebuild it completely
- Option A: Remove VWAP Reject from the playbook. Use ORB breakdowns for short exposure instead
- Option B: Rebuild with: (1) all last 6+ bars below VWAP, (2) rejection candle closes in bottom quartile, (3) volume > 1.5x average on rejection candle, (4) stop = 1.5x ATR above VWAP, (5) maximum 1 per day

## 19.3 Trend Day Losses (9 trades, -$9,607)

The Trend Day strategy won big when right ($2,941 avg winner) but triggered too often on non-trend days.

**Example: 2026-03-23 | LONG at $660.99 | Stopped at $660.25 | Loss: -$1,067 | Hold: 5 min**
- Why we entered: SPY was above VWAP for 30+ minutes. The 9 EMA pullback appeared at 10:30 AM. It looked like a trend day pullback entry
- Why it failed: SPY wasn't trending — it was in a range. The "above VWAP" condition was met, but the stock was oscillating in a $1.50 range, not stair-stepping higher. The pullback to 9 EMA wasn't a buying opportunity — it was the stock heading to the bottom of its range
- How to avoid: Require ORB breakout in same direction BEFORE looking for trend day entries. If the ORB didn't break, it's not a trend day

**Example: 2026-03-31 | LONG at $642.27 | Stopped at $641.74 | Loss: -$1,096 | Hold: 5 min**
- Why we entered: Same pattern — above VWAP, 9 EMA pullback at 10:30 AM
- Why it failed: $0.53 stop on a stock with ~$1.00 ATR per 5-min bar. The stop was just 0.5x ATR — not enough room. Normal fluctuation hit it
- How to avoid: Minimum stop of 1x ATR(14) on the 5-minute chart. If ATR is $0.80, stop must be at least $0.80 below entry

**Example: 2026-02-23 | SHORT at $681.79 | Stopped at $682.24 | Loss: -$1,111 | Hold: 5 min**
- Why we entered: SPY was below VWAP for 30+ minutes, 9 EMA pullback to the upside
- Why it failed: The "below VWAP for 30 min" on 2/23 was during a morning selloff that reversed by lunch. The trend had already exhausted before the entry triggered
- How to avoid: Add a freshness filter — the trend must have started within the last 60 minutes. If the stock has been "trending" (above/below VWAP) for 2+ hours, the trend is likely exhausted, not fresh

### Fix: Tighten the trend identification criteria
1. ORB breakout must confirm the direction first
2. 9 EMA must be above 20 EMA (long) or below (short) — stacking confirmation
3. Today's range at entry time must exceed 1.5x the average range for that time of day
4. Stop must be at least 1x ATR(14) from entry
5. Maximum 1 trend day trade per day

## 19.4 VWAP Bounce Losses (8 trades, -$8,704)

The VWAP Bounce had 50% win rate — the best of the losing strategies. But the 8 losses share a common pattern.

**Example: 2026-03-23 | LONG at $659.28 | Stopped at $659.08 | Loss: -$1,245 | Hold: 5 min**
- Why we entered: SPY was above VWAP, pulled back to VWAP, green candle formed at VWAP
- Why it failed: The $0.20 stop was absurdly tight. SPY's 5-min ATR was ~$0.60. A $0.20 stop gave less than 0.3x ATR of room — pure noise territory. The stop hit on the very next bar
- How to avoid: Minimum stop of 0.5x ATR. With $0.60 ATR, stop should be at least $0.30 below VWAP (or use $0.20 below VWAP as a minimum floor)

**Example: 2026-02-20 | LONG at $687.09 | Stopped at $684.68 | Loss: -$1,019 | Hold: 10 min**
- Why we entered: SPY above VWAP for 4+ bars, pullback to VWAP, bounce candle
- Why it failed: This was actually a trend day DOWN that started with SPY above VWAP. The "bounce" was a dead cat bounce before the real selling began. SPY dropped $2.41 through the stop. The macro direction overwhelmed the local VWAP support
- How to avoid: Check the broader market context. If VIX is spiking or futures are selling off, VWAP support is unreliable. Add a market health filter: don't take VWAP bounces when SPY's opening range has already broken to the downside

**Example: 2026-03-18 | LONG at $668.88 | Stopped at $668.55 | Loss: -$1,153 | Hold: 5 min**
- Why we entered: Standard VWAP bounce setup at 10:05 AM
- Why it failed: Too early in the session. At 10:05 AM, VWAP has only 35 minutes of data — it's not yet a reliable indicator. The "VWAP bounce" at 10:05 is really just buying a dip in the first 35 minutes, which has random outcomes
- How to avoid: Don't trade VWAP bounces before 10:15 AM. VWAP needs at least 45 minutes of volume to establish meaningful support/resistance

### Fix: Improve VWAP Bounce entry quality
1. Minimum time: no VWAP bounces before 10:15 AM
2. Minimum stop: max($0.20, 0.5x ATR(14)) below VWAP
3. Market health filter: SPY's opening range must not have broken to the downside
4. Require the bounce candle to close in the top 33% of its range (strong close, not indecisive)
5. Volume on the bounce candle should exceed the pullback candles' average

## 19.5 Gap Fade Losses (8 trades, -$6,424)

Gap Fade losses were all stop_loss exits where the gap didn't fade — it continued.

**Example: 2026-03-12 | LONG (gap down fade) at $670.29 | Stopped at $669.91 | Loss: -$1,130 | Hold: 5 min**
- Why we entered: SPY gapped down, reclaimed VWAP, so we bought expecting the gap to fill up
- Why it failed: The gap was caused by escalating tariff fears (March 2026). This wasn't a noise gap — it was fundamental. Gaps driven by macro catalysts don't fade
- How to avoid: Check the news. If the gap has a macro catalyst (Fed, tariffs, geopolitical), don't fade it. Only fade technical/noise gaps

**Example: 2026-03-09 | LONG at $665.84 | Stopped at $664.86 | Loss: -$1,051 | Hold: 5 min**
- Why we entered: SPY gapped down, crossed above VWAP at 10:00 AM
- Why it failed: The VWAP cross was a head-fake. SPY briefly crossed above VWAP and immediately reversed. One bar above VWAP is not enough confirmation
- How to avoid: Require 3 consecutive bars above VWAP (15 min) before entering the fade. This filters out head-fakes

### Fix: More selective gap fade criteria
1. Don't fade gaps with identifiable macro catalysts
2. Require 3 consecutive closes above/below VWAP before entering
3. Narrow the gap range to 0.8%-1.8% (filter noise and fundamental gaps)
4. Time window: only 10:00-10:30 AM — if it hasn't started fading by then, it won't

## 19.6 ORB Loss (1 trade, -$229)

Only one ORB loss — and it was the smallest loss of any trade.

**2026-02-05 | SHORT at $678.74 | Exit at $679.75 via time stop | Loss: -$229 | Hold: 125 min**
- Why we entered: SPY broke below the 15-min opening range with volume
- Why it failed: The breakdown was genuine initially, but the move stalled and slowly drifted back up. After 125 minutes without hitting the target or the stop, the time stop kicked in. The exit was only $1.01 above entry — a tiny loss
- What went right: The wide stop ($4.61, the full OR range) prevented a full -1R loss. The time stop exited the dead trade before it could get worse. The risk management worked perfectly even though the trade didn't

### ORB observation: No changes needed
The ORB strategy's selectivity (only 8 trades in 42 days) is its strength. When it doesn't find a clean breakout, it doesn't trade. When it does trade, the wide stop and volume confirmation produce a high win rate. The single loss was contained by the time stop at -0.23R — less than a quarter of the planned risk.

## 19.7 The 5-Minute Stop Problem

The most striking pattern across all strategies: **33 of 42 losses hit the stop within 5-10 minutes of entry.** This means either:

1. **The entries are wrong** — we're entering at the exact wrong moment
2. **The stops are too tight** — we're getting stopped by normal noise
3. **Both**

The data suggests it's primarily #2. The entries are directionally correct more often than not (the stock eventually moves in the predicted direction within 30-60 minutes of entry), but the stop gets clipped by a counter-move first.

**The noise-vs-stop relationship:**
- SPY's average 5-minute bar range: $0.40-0.80
- Average stop distance in our losses: $0.60
- Many stops were $0.20-0.40 — WITHIN one bar of noise

When your stop is tighter than one bar of noise, you're guaranteed to get stopped on normal fluctuation, regardless of whether your directional call was right.

**The fix is universal across strategies:**
- Minimum stop = 1x ATR(14) on the 5-minute chart
- This means fewer shares per trade (to maintain 1% risk)
- But dramatically fewer false stop-outs
- Expected outcome: lower win rate on individual candle moves, but higher win rate on directional calls — which is what matters

## 19.8 Summary of Proposed Changes

| Strategy | Current Issue | Proposed Fix | Expected Impact |
|----------|--------------|-------------|-----------------|
| **ORB** | None (working well) | No changes; gather more data | Validate with 100+ trades |
| **VWAP Bounce** | Stops too tight; early AM entries fail | Min stop = max($0.20, 0.5 ATR); no trades before 10:15 AM; market health filter | Win rate 50% → 55-60% |
| **VWAP Reject** | 76% stop rate; shorting into bounces | Kill or rebuild: 6+ bars below VWAP, rejection candle quartile, 1.5 ATR stop | Kill (use ORB for shorts) |
| **Trend Day** | Triggers on range days, not trend days | Add ORB confirmation + EMA stacking + range expansion | Trades 14 → ~5, win rate 36% → 55% |
| **Gap Fade** | Fades macro gaps; too few bars of confirmation | Skip macro catalyst gaps; require 3 bars of VWAP hold; narrow gap range | Win rate 38% → 45-50% |

The single biggest improvement: **increase minimum stop to 1x ATR across all strategies.** This one change would have prevented 15-20 of the 42 losses, at the cost of smaller position sizes (which is actually a feature, not a bug — reduced tail risk).
