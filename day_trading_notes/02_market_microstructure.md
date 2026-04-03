# Chapter 2: Market Microstructure — How Prices Actually Move

## 2.1 The Order Book

Every stock has two prices at any given moment: the **bid** (the highest price someone is willing to pay) and the **ask** (the lowest price someone is willing to sell at). The difference between them is the **spread**.

When you see "XYZ: $10.00 x $10.03," this means:
- The best bid is $10.00 — someone will buy up to a certain number of shares at $10.00
- The best ask is $10.03 — someone will sell up to a certain number of shares at $10.03
- The spread is $0.03

**Why the spread matters:** If you buy at the ask ($10.03) and immediately sell at the bid ($10.00), you lose $0.03/share. That's 0.3% gone before the trade even has a chance to work. On a 500-share position, that's $15 in friction — per round trip.

The depth of the order book tells you how much liquidity exists at each price level. "100 shares at $10.00" is very different from "50,000 shares at $10.00." Deep liquidity at a level means it's likely to hold as support/resistance. Thin liquidity means it can be blown through easily.

For day traders, the practical implications are:
- **Tight spreads (≤$0.02):** You can enter and exit with minimal friction. Trade these stocks
- **Wide spreads ($0.05-0.10+):** Every trade starts at a disadvantage. The stock needs to move significantly just to break even. Avoid or reduce size
- **Variable spreads:** During high-volatility moments (open, news events), spreads widen dramatically. This is when slippage is worst

## 2.2 Market Makers and Liquidity

Market makers are firms that continuously offer to buy and sell a stock, providing liquidity. They profit from the spread. When you buy at the ask, a market maker is selling to you. When you sell at the bid, a market maker is buying from you.

Why this matters for day trading:

**Market makers are not your enemy, but they're not your friend either.** They provide the liquidity that allows you to enter and exit positions. Without them, you'd have to find another trader willing to take the other side of your exact trade at your exact price.

**Market makers see order flow.** Through payment for order flow (PFOF) arrangements, market makers at firms like Citadel and Virtu see retail order flow before it hits the exchange. They can adjust their quotes accordingly. This doesn't mean they're "front-running" you — but it does mean the price you see may not be the price you get on a market order.

**Institutional market makers operate on timescales you can't compete with.** High-frequency trading firms execute in microseconds. You cannot beat them on speed. Don't try. Instead, compete on time horizon: they're optimizing for fractions of a penny over milliseconds; you're optimizing for 1-3% over minutes to hours.

## 2.3 Volume as Information

Volume is the single most important indicator for day traders. It tells you whether a move is real (backed by conviction) or fake (low participation).

**Volume leads price.** Before a stock makes a significant move, volume typically picks up. A breakout on high volume is more likely to follow through than a breakout on low volume.

**Relative volume is what matters, not absolute volume.** Apple trading 5 million shares by 10 AM is quiet (its average is 60M/day). A $5 stock trading 5 million shares by 10 AM when its average is 500K is exploding (10x relative volume). The relative volume ratio (RVOL) is: current volume / average volume at this time of day.

**What different RVOL levels mean:**
- **RVOL < 1.0:** Below average activity. This stock isn't doing anything today. Skip it
- **RVOL 1.0-2.0:** Normal to slightly elevated. Not enough to justify special attention
- **RVOL 2.0-5.0:** Significantly elevated. Something is happening. Worth investigating
- **RVOL 5.0-10.0:** Extremely active. Major catalyst. This is on the day's scan
- **RVOL > 10.0:** Massive event. Be cautious — this level of volume often means halts, extreme volatility, and wide spreads

## 2.4 Price Discovery at the Open

The market open is not just "the start of trading." It's a fundamentally different process from normal trading, and understanding it gives you an edge.

**Pre-market (4:00-9:30 AM ET):** Limited participants, wider spreads, lower volume. Price movements in pre-market set the day's opening expectations but can be misleading — a stock that's up 5% pre-market on 50K shares can open up 3% or 7% when millions of shares trade at 9:30.

**The opening auction (9:28-9:30 AM):** Orders accumulate for two minutes before the opening cross. The exchange matches the maximum number of buy and sell orders at a single price — the opening price. This is often where the largest volume of the day occurs in a single instant.

**The first 5 minutes (9:30-9:35 AM):** Extremely volatile. The pre-market price and the regular session price reconcile. Algorithms fire. Retail orders that queued overnight execute. Spreads are wide. This is the most dangerous time to trade unless you have a specific plan.

**The first 15 minutes (9:30-9:45 AM):** The opening range forms. This range (high and low of the first 15 minutes) often defines the day's key levels. A break above the opening range high on volume is a bullish signal. A break below the low is bearish.

**The first hour (9:30-10:30 AM):** Contains 40-60% of the day's total volume and the majority of its price discovery. Most day trading strategies are designed for this window.

## 2.5 Level 2 and Time & Sales

Level 2 data shows the full depth of the order book — not just the best bid and ask, but all visible orders at every price level.

**What to look for in Level 2:**
- **Large bids (walls):** A buyer with 50,000 shares at $10.00 is providing a "floor." The stock is unlikely to drop below $10.00 as long as that order exists. But walls can be fake — algorithms place and cancel large orders to manipulate perception
- **Stacked asks:** Multiple sellers at consecutive levels above the current price ($10.05: 10K, $10.06: 15K, $10.07: 20K) indicate selling pressure. The stock will need significant buying to push through
- **Thin levels:** Empty price levels in the order book (no orders between $10.05 and $10.15) create vacuums. If the price reaches that zone, it can move quickly to the next level with orders

**Time and Sales (the "tape"):** Shows every executed trade in real-time — price, size, time, and whether it hit the bid or the ask. Reading the tape gives you a sense of who's in control:
- Large prints hitting the ask (buying at market) = aggressive buyers, bullish
- Large prints hitting the bid (selling at market) = aggressive sellers, bearish
- Lots of small trades = retail flow, less informative
- Occasional large block trades = institutional flow, more informative

For day traders, Level 2 and Time & Sales are confirmation tools. They don't generate trade ideas — the scanner and chart do that. But they confirm whether a breakout has real buying behind it or whether it's a trap.

## 2.6 Dark Pools and Hidden Liquidity

Not all trading happens on visible exchanges. Approximately 40-50% of US equity volume trades in dark pools — private exchanges where orders are not visible until they execute.

**Why dark pools exist:** Institutional investors (mutual funds, pension funds, hedge funds) need to buy or sell large positions (millions of shares) without moving the market. If they placed a visible order for 5 million shares of Apple, the price would move against them before they could complete the order. Dark pools allow them to trade anonymously.

**What this means for day traders:**
- The visible order book only shows half the story. There may be significant buying or selling happening that you can't see until it appears on Time & Sales as dark pool prints
- Large dark pool prints at a specific level indicate institutional interest at that price. If you see repeated dark pool prints at $50.00, there's a large buyer at that level
- Dark pool activity is highest in large caps and lowest in small caps. This is one reason small cap momentum is more "visible" — less is hidden

Dark pool data is available through services like BlackBox Stocks, FlowAlgo, or directly from FINRA's ATS transparency data (delayed). For most day traders, awareness is enough — you don't need real-time dark pool feeds unless you're trading large cap at significant size.

## 2.7 The Intraday Volatility Smile

Intraday volatility follows a predictable U-shaped pattern:

**9:30-10:30 AM:** High volatility, high volume. The first hour. This is where most day trading opportunity exists. Stocks gap, trends establish, breakouts happen, and the opening range resolves.

**10:30 AM-12:00 PM:** Declining volatility. Trends established in the first hour often continue but at a slower pace. This is where trend-following strategies work but scalping becomes difficult.

**12:00-2:00 PM:** The dead zone. Lowest volume of the day. Spreads widen. Choppy, directionless price action. Most professional day traders stop trading during this period. Overtrading in the dead zone is one of the primary causes of unnecessary losses.

**2:00-3:30 PM:** Volume picks up again. Institutional rebalancing begins. The "power hour" (though it's more like power-90-minutes). New trends can start or morning trends can reverse.

**3:30-4:00 PM:** Final positioning. Mutual funds that track benchmarks must complete their trades. Market-on-close (MOC) orders execute. Volatility increases. Some strategies specifically target this window.

**The practical rule:** Trade the open (9:30-10:30). If the day is trending, manage positions through the morning. Stop trading by lunch unless you have a specific afternoon edge. Resume only if there's a power hour catalyst.

The statistics from our own ORB backtesting confirm this pattern: 70%+ of profitable day trades were entered in the first 60 minutes of the session. Midday entries had negative expected value on average.
