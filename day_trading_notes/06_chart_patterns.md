# Chapter 6: Chart Patterns That Actually Work Intraday

## 6.1 Why Most "Patterns" Are Noise

The trading education industry sells hundreds of chart patterns: head and shoulders, cup and handle, ascending triangles, diamond tops, three black crows, morning doji star. Most of these are noise.

When rigorously backtested, the majority of traditional chart patterns fail to produce statistically significant results. They work about as well as random entries, once you account for spread, slippage, and commissions.

**Why patterns seem to work (but don't):**
- **Confirmation bias:** You remember the patterns that worked and forget the ones that didn't
- **Hindsight pattern matching:** Patterns are easy to see after the fact, impossible to identify in real-time with confidence
- **No statistical edge:** A study by Lo, Mamaysky, and Wang (2000) found that while patterns exist in market data, their predictive power is weak and inconsistent across time periods
- **Overfitting:** With enough parameters, you can fit a pattern to any price series

**What does work:** The patterns in this chapter are not classical chart patterns. They are structural setups based on supply/demand dynamics that have measurable statistical edges when combined with volume confirmation, time-of-day filters, and risk management. They work not because of the visual pattern but because of the underlying market mechanics they represent.

## 6.2 The Opening Range Breakout (ORB)

The most studied and validated intraday pattern. It works because the opening range represents the market's initial consensus about the day's value — a break beyond that range signals a directional conviction.

**The setup:**
1. Wait for the first 5, 15, or 30 minutes to complete (the "opening range")
2. Mark the high and low of that range
3. Enter long if price breaks above the range high with above-average volume
4. Enter short if price breaks below the range low with above-average volume
5. Stop loss: opposite side of the range (5-min ORB) or middle of the range (15/30-min ORB)

**Which time frame?**
- **5-minute ORB:** Fastest. Most signals. Highest false breakout rate. Best for experienced traders on volatile stocks
- **15-minute ORB:** Good balance of speed and reliability. Most popular variant. Best for small cap momentum stocks
- **30-minute ORB:** Most reliable. Fewest signals. Best for large caps and market ETFs (SPY, QQQ)

**Filters that improve the ORB:**
- Gap direction matches breakout direction (gap up + break high = better odds)
- Relative volume > 2x at the time of breakout
- Stock is above VWAP at breakout (for long) or below VWAP (for short)
- Opening range is narrow relative to ATR (< 50% of ATR = compressed energy about to release)
- Pre-market levels confirm direction (PMH break for longs, PML break for shorts)

**Why ORB works mechanically:** During the opening range, buyers and sellers are testing each other. A sustained break above the high means buyers have won — sellers who shorted within the range are now underwater and need to cover, adding fuel. Algorithms programmed to follow ORB breakouts pile in. The result is a momentum burst that often carries for 30-90 minutes.

## 6.3 VWAP Reclaim / Rejection

VWAP is the volume-weighted average price — the institutional benchmark for "fair value" throughout the day. Institutional algorithms are programmed to buy below VWAP and sell above VWAP. This creates a self-fulfilling prophecy around this level.

**VWAP Bounce (Long):**
- Stock is in an uptrend (above VWAP for most of the day)
- Price pulls back to VWAP
- A bullish candle forms at VWAP with increasing volume
- Enter long with stop $0.10-0.20 below VWAP
- Target: prior high of day, or 2R

This works because institutional buyers are programmed to buy at VWAP. When the price reaches VWAP in an uptrend, these algorithms activate, providing a floor.

**VWAP Rejection (Short):**
- Stock is in a downtrend (below VWAP for most of the day)
- Price rallies to VWAP
- A bearish candle forms at VWAP with increasing volume
- Enter short with stop $0.10-0.20 above VWAP
- Target: prior low of day, or 2R

This works because institutional sellers are programmed to sell at VWAP. In a downtrend, rallies to VWAP are selling opportunities for institutional accounts.

**VWAP Reclaim:**
- Stock opens below VWAP (bearish start)
- Price recovers and crosses above VWAP with a surge in volume
- Enter long on the cross, stop below VWAP
- This is a powerful signal because it means the bearish narrative has failed — shorts are covering and longs are entering

**Key VWAP rules:**
- VWAP only resets at the start of each trading day
- VWAP is meaningless in the first 5 minutes (not enough data)
- VWAP works best on stocks with moderate-to-high volume (>1M shares/day)
- VWAP is a moving target — it shifts throughout the day based on volume distribution
- On a strong trend day, VWAP may never be retested

## 6.4 The Bull Flag / Bear Flag

A flag is a consolidation pattern after an impulse move. The flag "flies" from a "pole" — the pole is the impulse, the flag is the consolidation.

**Bull flag setup:**
1. Strong impulse move up (the pole): at least 3-5 green candles with increasing volume
2. Consolidation on declining volume (the flag): price drifts sideways or slightly down for 3-8 candles
3. Volume picks up on a green candle that breaks above the flag's high
4. Enter long on the breakout
5. Stop below the flag low
6. Target: measured move (pole height added to breakout point)

**What makes a good flag:**
- Tight consolidation (flag retraces < 50% of the pole)
- Declining volume during the flag (sellers are exhausted, not pressing)
- Flag forms above VWAP (institutional support underneath)
- The breakout candle has 2x+ the volume of the flag candles
- The flag holds for 5-15 minutes (too short = not a real consolidation; too long = momentum dead)

**What makes a bad flag (avoid):**
- Loose, sloppy consolidation with wide candles
- Volume increasing during the flag (sellers are active, not resting)
- Flag breaks below VWAP
- The "flag" retraces 70%+ of the pole (this is a reversal, not a consolidation)
- The stock has already made 3+ flag patterns today (each successive flag is weaker)

## 6.5 The Flat Top Breakout

A flat top is a horizontal resistance level that has been tested multiple times with each test showing a higher low. The compression between the flat top and the rising lows creates pressure that eventually resolves with a breakout.

**The setup:**
1. Stock makes a high (say $10.50) and pulls back
2. Stock rallies again to $10.50 and pulls back — but to a higher low than the first pullback
3. Stock rallies to $10.50 a third time (or more)
4. Each pullback makes a higher low, but the high ($10.50) holds like a ceiling
5. Eventually, buying pressure exceeds selling pressure at $10.50, and the stock breaks out
6. Enter long above $10.50, stop below the last higher low
7. Target: height of the pattern added to breakout level

**Why flat tops work:** Each test of the ceiling exhausts the sellers at that level. Meanwhile, the higher lows show that buyers are getting more aggressive — willing to buy at $10.30, then $10.35, then $10.40. When the sellers at $10.50 are finally overwhelmed, there's no one left to sell, and the stock can move freely higher.

**Volume signature:** Volume should decline on each pullback (less selling) and increase on each rally to the flat top (more buying). The breakout candle should have the highest volume of the entire pattern.

## 6.6 The ABCD Pattern

The ABCD is a measured move pattern based on the concept that markets move in waves of similar magnitude.

**The setup:**
1. **A to B:** Impulse move up (or down). Measure the distance.
2. **B to C:** Pullback. Typically retraces 50-61.8% of the A-B move (Fibonacci retracement). The C point is your entry.
3. **C to D:** Continuation move. Typically extends to the same distance as A-B, measured from the C point.

**Example:**
- A = $10.00, B = $11.00 (impulse of $1.00)
- C = $10.50 (pulled back 50% of the $1.00 move)
- D target = $11.50 ($1.00 from the C point)

**Entry:** At point C, after the pullback completes and a bullish candle forms
**Stop:** Below point A (or below the 78.6% retracement level)
**Target:** Point D (measured move extension)

**Why ABCD works:** It captures the natural rhythm of institutional buying. Large institutions can't buy all their shares at once — they buy in waves. The A-B wave is the initial accumulation. The B-C pullback is retail selling to the institution. The C-D wave is the institution's second phase of buying. The pattern is the market's breathing rhythm made visible.

**Fibonacci alignment:** The strongest ABCDs have C at the 61.8% retracement and D at the 127.2% or 161.8% extension. These are not magic numbers — they work because enough traders watch them that they become self-fulfilling.

## 6.7 The Red-to-Green / Green-to-Red Move

This is a simple but effective setup based on the emotional significance of unchanged (yesterday's close).

**Red-to-Green (Long):**
- Stock opens below yesterday's close (in the red)
- Over the first 15-60 minutes, the stock rallies and crosses ABOVE yesterday's close (goes green)
- Enter long on the cross with volume confirmation
- Stop below the opening low or below VWAP
- Target: prior day high or 2R

**Why it works:** When a stock goes from red to green, it means everyone who sold in pre-market or at the open is now losing money. The short-term shorts start covering. The bears who were waiting to add are now hesitant. The bulls who were waiting for confirmation start buying. The sentiment shift is self-reinforcing.

**Green-to-Red (Short):**
- Stock opens above yesterday's close (in the green)
- Over the first 15-60 minutes, the stock sells off and crosses BELOW yesterday's close (goes red)
- Enter short on the cross with volume confirmation
- Stop above the opening high or above VWAP
- Target: prior day low or 2R

**Key filter:** The cross must happen on increasing volume. A red-to-green cross on declining volume is unreliable — there's no conviction behind the move.

## 6.8 What Makes a Pattern Valid

No pattern is a guaranteed trade. What separates valid patterns from noise:

**Volume confirmation:** Every pattern in this chapter requires above-average volume at the trigger point. A breakout on low volume is a trap. Volume is the lie detector of the market — it tells you whether the move is backed by real money or just noise.

**Clean structure:** The pattern should be visually obvious. If you need to squint and draw multiple trendlines to see it, it's not a real pattern. The best patterns are so clean that five different traders would all identify the same entry, stop, and target.

**Proximity to key levels:** A bull flag forming at the VWAP with the breakout targeting the prior day high is stronger than a bull flag floating in the middle of nowhere. Patterns work best when they align with levels that other traders are watching.

**Catalyst support:** A pattern backed by a catalyst is more reliable than a pattern without one. An ORB breakout on an earnings gapper has better odds than an ORB breakout on a random stock. The catalyst provides the "why" — the pattern provides the "when."

**Time of day:** Patterns that form in the first 60 minutes of the session have better follow-through than patterns that form in the midday lull. By 12 PM, most of the day's directional conviction has been expressed. Afternoon patterns can work, but they need stronger confirmation.

**The A+ setup test:** Does the pattern have all five: volume, structure, levels, catalyst, and time? If yes, it's an A+ setup — trade it with full size and confidence. If it's missing one, it's an A or B setup — trade it with reduced size. If it's missing two or more, skip it.
