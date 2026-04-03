# Chapter 12: Strategy 3 — Momentum Scalping (Small Caps)

## 12.1 The Theory

Small cap momentum scalping is the highest-risk, highest-reward day trading strategy. It targets stocks with low floats and high relative volume that are making dramatic moves — 20%, 50%, sometimes 200%+ in a single session.

The edge comes from structural supply/demand imbalance. When a stock with 5 million shares floating suddenly has 10 million shares of demand, the price has to go up. There aren't enough shares for everyone who wants to buy. Each share sold gets replaced by a buyer at a higher price. The result is a near-vertical price move that can last minutes to hours.

This is the "warrior trader" domain — the style popularized by Ross Cameron and similar small cap momentum traders. It looks spectacular on screen. It also has the highest failure rate of any strategy. The spreads are wide, the moves are violent, the reversals are sudden, and the emotional toll is extreme.

This chapter teaches the strategy with full honesty about both the potential and the danger.

## 12.2 The Small Cap Scanner

The scanner for momentum scalping is more aggressive than the standard gap scanner:

| Filter | Threshold | Why |
|--------|-----------|-----|
| Float | < 20M shares | Low supply = bigger moves |
| Gap % | > 10% pre-market | Something major happened |
| Pre-market volume | > 1M shares | Confirmed interest |
| Price | $2-$20 | Enough room to move, enough liquidity to exit |
| Catalyst | Required (Tier 1-2) | No catalyst = pump and dump risk |
| RVOL | > 5x | Extreme interest |
| Average daily volume | > 500K | Can actually exit when you need to |

**What this scan produces:** 0-3 stocks per day that meet all criteria. Most days have zero qualifying stocks. That's fine — the strategy requires patience. When a qualifying stock appears, it demands full attention.

**Red flags to filter out:**
- No identifiable catalyst (skip — likely pump and dump)
- Stock has had 3+ halts pre-market (too volatile, spreads will be extreme)
- Float < 1M shares (micro-float = manipulation risk, impossible to exit)
- Already up 100%+ pre-market (the easy money is made; you'd be chasing)

## 12.3 The Entry

The entry for small cap momentum is NOT at the top of a spike. It's on the first pullback after the spike establishes the trend.

**The "first pullback" setup:**

1. Stock gaps up 15% on earnings beat. Opens at $8.00 (was $7.00 yesterday)
2. At the open, aggressive buying pushes it to $9.00 in the first 3 minutes (the spike)
3. Profit-taking begins. The stock pulls back to $8.50 over 2-3 candles
4. Volume declines during the pullback (sellers exhausted, not pressing)
5. A green candle forms at $8.50 with increasing volume (buyers returning)
6. **ENTRY:** Buy above the high of that green candle ($8.55)
7. **STOP:** Below the pullback low ($8.40) — risk = $0.15/share
8. **TARGET:** Retest of the spike high ($9.00) — reward = $0.45/share (3R)

**Why the first pullback, not the spike:**
- The spike is unpredictable — you don't know how far it will go
- Buying the spike means chasing — you're paying the highest price with the widest spread
- The pullback gives you a defined entry and a defined stop
- If the pullback holds, it confirms the trend. If it doesn't hold, your tight stop limits the loss

**The second pullback:** Also tradeable but with reduced conviction. Each successive pullback shows waning momentum. By the third pullback, the trend is likely exhausted. Don't trade the third pullback.

## 12.4 The Exit

Small cap momentum trades require faster exits than any other strategy. These stocks move 5-10% in minutes, in both directions. Hesitation costs real money.

**The tight stop rule:** Stop loss must be 3-5% below entry, or below the pullback low, whichever is tighter. On a $10 stock, that's $0.30-0.50. If the stop would be wider than 5%, the trade is not right — either the pullback wasn't clean or you're entering too late.

**Exit signals (any one of these = sell immediately):**
- High-volume red candle (a long red candle on 2x+ volume = sellers taking control)
- Lower high (the stock fails to make a new high after your entry = momentum dying)
- Break of VWAP (the stock drops below VWAP = institutional support gone)
- Halt down (if the stock halts on a downward move, sell when it reopens)
- Your gut says something is wrong (this sounds unscientific, but experienced traders develop pattern recognition that manifests as intuition. If something feels off, reduce or exit)

**Scaling out for small cap scalps:**
- Sell 50% at 1R (lock in profit, remove risk)
- Trail the remaining 50% with a tight stop (below the most recent 1-minute candle low)
- This captures the 10R runners while protecting capital on the reversals

**Time limit:** If the trade hasn't worked within 10-15 minutes, the momentum burst is over. Exit regardless of P&L.

## 12.5 The Halt Play

When a small cap moves 5-10% within 5 minutes, it triggers a Limit Up-Limit Down (LULD) halt. Trading is paused for 5 minutes (sometimes longer). Understanding halts is critical for small cap trading.

**Halt up (bullish halt):**
- Stock is halted because it moved up too fast
- During the halt, no trading occurs. Orders queue for the re-open auction
- The re-open price is often HIGHER than the halt price (demand accumulated during the halt)
- Strategy: If you're already long, hold through the halt. If you're not in, consider buying the re-open dip (first pullback after the re-open)

**Halt down (bearish halt):**
- Stock is halted because it moved down too fast
- The re-open price is often LOWER than the halt price (panic selling accumulated)
- Strategy: If you're long, place a sell order IMMEDIATELY for the re-open. Don't wait to see what happens. The re-open of a halt-down is almost always worse than the halt price
- If you're looking to short, the re-open of a halt-down can provide an entry

**Multiple halts in one direction:**
- A stock that halts up 3+ times is in a true squeeze. Enormous moves (50-200%+) are possible. But the risk is equally enormous — when it reverses, it halts down just as fast
- A stock with 5+ halts in either direction is too volatile to trade safely. The spreads will be $0.50-2.00, and you can't control your risk. Step aside

**The "halt and fail" pattern:** Stock halts up, re-opens, briefly spikes higher, then drops back below the halt price. This means the halt created a false sense of momentum. The buying was exhausted by the time it re-opened. This is a SHORT signal — enter short below the halt price with a stop above the re-open high.

## 12.6 Risk Management for Small Caps

Small cap momentum trading requires DIFFERENT risk management than large cap trading:

**Reduce position size by 50%.** If you normally risk 1% per trade, risk 0.5% on small caps. The wider spreads and faster moves create slippage that erodes your actual R-multiple. What looks like a 2R trade on the chart may be a 1.5R trade after slippage.

**Wider stops require fewer shares.** On a large cap with a $0.25 stop, you might trade 2,000 shares. On a small cap with a $0.50 stop, you trade 500 shares. The dollar risk is the same, but the share count is lower — this is correct.

**Use limit orders for entry, market orders for exit.** On entry, you can afford to wait for your price. On exit (especially stop losses), you need to get out NOW. A limit sell during a crash may not fill. A market sell will fill, even at a worse price. The worse fill is better than not filling at all.

**Pre-calculate your max loss before entry.** "If this stock drops 5% from my entry, I lose $X." Know that number before you click buy. If the number makes you uncomfortable, reduce your size until it doesn't.

**No averaging down. Ever.** On small caps, averaging down is how people lose 20-50% of their account in a single trade. A stock dropping from $10 to $8 might go to $5. There is no floor on a small cap sell-off. Cut and move on.

## 12.7 The Danger Zone

An honest assessment of small cap momentum trading:

**The allure:** A $5,000 position in a stock that goes from $8 to $12 makes $2,500 in 20 minutes. That's a month's rent in less time than a coffee break. This is what draws people in.

**The reality:** For every $2,500 winner, there are multiple $500-1,000 losers. The spread costs $0.05-0.10/share on entry and exit. The stop gets hit on 50-60% of trades. The emotional toll of watching positions move $1,000 against you in seconds is severe. Most traders who attempt small cap momentum quit within 6 months, poorer and emotionally drained.

**Who should trade this strategy:**
- Experienced traders with 500+ trades in other strategies
- Traders with accounts > $50K (enough to absorb the variance)
- Traders who can genuinely follow a stop loss without hesitation
- Traders who can stop after 2-3 trades (overtrading is the killer)

**Who should NOT trade this strategy:**
- Beginners (start with ORB on SPY or large cap VWAP bounces)
- Traders with accounts < $25K (the variance will blow through PDT limits and capital)
- Anyone who has ever moved a stop further away "to give it more room"
- Anyone who has ever averaged down on a losing position

The strategy works. But it works for a very small percentage of traders who combine the right temperament, adequate capital, and ironclad discipline. For everyone else, the other strategies in this book offer better risk-adjusted returns with less emotional damage.
