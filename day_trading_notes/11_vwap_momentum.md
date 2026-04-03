# Chapter 11: Strategy 2 — VWAP Momentum

## 11.1 The Theory

VWAP — Volume Weighted Average Price — is the average price of a stock weighted by the volume traded at each price. It answers the question: "What is the average price that actual shares changed hands at today?"

VWAP is the institutional benchmark. Portfolio managers measure their execution quality against VWAP. If a fund bought 1 million shares at an average price below VWAP, the trade was executed well. Above VWAP, it was executed poorly. This creates a gravitational force around VWAP:

- **Institutional buy algorithms** are programmed to buy when price is at or below VWAP. This creates buying pressure at VWAP
- **Institutional sell algorithms** are programmed to sell when price is at or above VWAP. This creates selling pressure at VWAP
- The result: VWAP acts as a dynamic support/resistance level that reflects the day's actual trading consensus

For day traders, VWAP is the dividing line between bullish and bearish:
- **Above VWAP:** Net buyers are profitable. The trend is up. Look for long trades
- **Below VWAP:** Net sellers are profitable. The trend is down. Look for short trades

## 11.2 VWAP Bounce (Long)

The VWAP bounce is a pullback entry in an intraday uptrend. The stock has been above VWAP, pulls back to touch VWAP, and bounces. You enter the bounce.

**Setup conditions:**
1. Stock is above VWAP (has been above for at least 30 minutes)
2. Stock pulls back toward VWAP on declining volume (sellers are not aggressive)
3. Price touches or comes within $0.10-0.20 of VWAP
4. A bullish candle forms at VWAP (green candle, ideally with a lower wick showing rejection of lower prices)
5. Volume picks up on the bounce candle

**Entry:** Buy above the high of the bounce candle
**Stop:** $0.10-0.20 below VWAP (if VWAP breaks, the thesis is dead)
**Target:** Prior high of day, or 2R

**Why VWAP bounces work:**
- Institutional algorithms are literally programmed to buy at VWAP
- Traders who missed the initial move see the VWAP pullback as a second chance
- Shorts who entered near VWAP have tight stops — the bounce triggers their covers
- VWAP acts as a self-fulfilling support level because enough participants believe in it

**Best conditions for VWAP bounces:**
- Stock has a catalyst (not just random movement)
- The pullback to VWAP is the first or second touch (third touches are less reliable)
- The overall market (SPY) is also above its VWAP
- The pullback took 15-30 minutes (too fast = panic selling; too slow = momentum is dead)

## 11.3 VWAP Rejection (Short)

The mirror image of the VWAP bounce. A stock in an intraday downtrend rallies to VWAP and gets rejected.

**Setup conditions:**
1. Stock is below VWAP (has been below for at least 30 minutes)
2. Stock rallies toward VWAP on declining volume
3. Price touches or comes within $0.10-0.20 of VWAP
4. A bearish candle forms at VWAP (red candle, ideally with an upper wick showing rejection of higher prices)
5. Volume increases on the rejection candle

**Entry:** Short below the low of the rejection candle
**Stop:** $0.10-0.20 above VWAP
**Target:** Prior low of day, or 2R

**When VWAP rejections are strongest:**
- Stock gapped down on bad news and has been below VWAP all morning
- The rally to VWAP occurred on notably low volume (no real buying conviction)
- Multiple sellers visible at VWAP on Level 2
- The broader market is also weak (SPY below its VWAP)

## 11.4 VWAP Reclaim

The most powerful VWAP signal. A stock that has been below VWAP (bearish) crosses above VWAP with a surge in volume. This is a sentiment reversal.

**Why reclaims are powerful:**
- Everyone who sold short below VWAP is now losing money. They cover (buy), adding fuel
- Everyone who was bearish and waiting to short more now has to reassess. Some flip long
- Algorithms that were in "sell mode" (above VWAP = sell) now switch to "buy mode"
- The reclaim often marks the day's low — the point where bears gave up

**Setup:**
1. Stock opened below VWAP (or dropped below after the open)
2. Price consolidates or drifts lower for 30-60 minutes
3. Volume starts building from the lows
4. A strong green candle crosses above VWAP with 2x+ average volume
5. The candle closes above VWAP (not just a wick — a close above)

**Entry:** Buy on the close above VWAP, or on the first pullback that holds above VWAP
**Stop:** Below VWAP by $0.15-0.25 (the reclaim must hold)
**Target:** Prior day high, or morning high of day, or 3R

**The "failed reclaim" warning:** If a stock crosses above VWAP and immediately drops back below, the reclaim failed. This is bearish — it means the selling pressure is too strong. Exit immediately and potentially reverse to short. Failed reclaims are some of the most reliable short signals.

## 11.5 Multi-Day VWAP

Standard VWAP resets every day. But anchoring VWAP to significant dates creates powerful multi-day levels.

**Prior day VWAP:** Yesterday's VWAP often acts as support/resistance today, especially in the first hour. If a stock opens above yesterday's VWAP, it has a bullish starting position. If it opens below, bearish.

**Anchored VWAP from earnings:** Set VWAP to start from the stock's most recent earnings date. This "earnings VWAP" shows the average price since the fundamental re-rating. A stock trading above its earnings VWAP is in a confirmed uptrend since earnings. Below it = the earnings reaction has been fully retraced.

**Anchored VWAP from significant lows/highs:** Start VWAP from a 52-week low or the start of a major trend. This shows whether the average participant in the trend is profitable or underwater.

**Multi-day VWAP as a swing/day trading bridge:** If you identify a stock that's above its anchored VWAP (bullish on the daily time frame) AND above today's intraday VWAP, you have alignment across multiple time frames. This alignment produces the highest-quality VWAP bounce trades.

## 11.6 When VWAP Fails

VWAP is not magic. There are specific conditions where VWAP-based strategies fail:

**Trend days:** On strong trend days (stock moves 5%+ in one direction all day), VWAP never gets retested. The stock opens and immediately pulls away from VWAP, never looking back. Waiting for a VWAP bounce on a trend day means missing the entire move. On trend days, use EMA pullbacks instead (Chapter 13).

**Low volume days:** When volume is below average for the day, VWAP becomes meaningless. With few shares traded, the VWAP calculation is dominated by a small number of prints and doesn't reflect genuine consensus. Don't trade VWAP strategies when RVOL < 0.8.

**Chop around VWAP:** Some days, the stock oscillates $0.20 above and below VWAP all day, triggering both long and short signals repeatedly. Each one fails. This happens on directionless days with no catalyst. The solution: if a stock has crossed VWAP more than 4 times by 10:30 AM, it's a chop day. Stop trading VWAP strategies on this name.

**Pre-10 AM VWAP:** VWAP is unreliable in the first 15-30 minutes because there isn't enough volume to establish a meaningful average. Don't take VWAP bounces or rejections before 10:00 AM. Use ORB strategies for the first hour instead.

**The VWAP + catalyst rule:** VWAP strategies work best on stocks with a catalyst. A VWAP bounce on a stock that's up 6% on earnings is high quality. A VWAP bounce on a random stock that's drifting 0.5% above VWAP is low quality. Without a catalyst, VWAP levels are just numbers on a screen.
