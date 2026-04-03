# Chapter 4: The Pre-Market Scanner — Finding Today's Movers

## 4.1 Why Scanning Is Everything

Day trading is not stock picking. You don't decide that Tesla is a good stock and trade it every day. You let the market tell you which stocks are in play TODAY, and you only trade those.

The scanner is the strategy. Everything downstream — entries, exits, position sizing — depends on the quality of what the scanner surfaces. A perfect entry on the wrong stock is worse than a mediocre entry on the right stock.

What makes a stock "in play":
- It's moving unusually today (gap, volume, catalyst)
- Other traders are also watching it (ensuring liquidity)
- It has a clear technical structure to trade against (levels, patterns)
- The risk/reward is defined (you know where your stop goes)

A stock that is not in play — no gap, average volume, no catalyst, no clear levels — is a stock where your edge is zero. You're just guessing. The scanner ensures you only trade stocks where the probability of a meaningful move is elevated.

## 4.2 The Core Scan Criteria

Every morning before 9:00 AM, run a scan with these filters. The specific thresholds will vary by strategy, but this is the baseline:

**Gap % (pre-market change):**
- **Long candidates:** Gap > +4%
- **Short candidates:** Gap < -4%
- **Why 4%?** Smaller gaps are too common and often fade. 4%+ gaps indicate something material happened

**Relative Volume:**
- **Minimum:** 2x average volume at this time of day
- **Ideal:** 5x+ relative volume
- **Why?** High RVOL confirms that the gap has attracted attention and that liquidity will be available for entries and exits

**Float (shares available to trade):**
- **Small cap momentum:** Float < 50M shares (ideally < 20M for squeezes)
- **Large cap ORB:** Float > 100M shares (deep liquidity)
- **Why?** Float determines how far a stock can move on a given amount of volume. Low float + high demand = big moves

**Price range:**
- **Small cap momentum:** $2-$20 (enough room to move, enough liquidity for entries)
- **Large cap ORB:** $20-$500 (institutional stocks, tight spreads)
- **Why?** Sub-$2 stocks have wide spreads and are prone to manipulation. $500+ stocks require large capital for position sizing

**Average daily volume:**
- **Minimum:** 500,000 shares/day (ensures you can exit when you need to)
- **Ideal:** 1M+ shares/day
- **Why?** Thinly traded stocks trap you. When you need to exit, there's no one to buy your shares

## 4.3 Gap Classification

Not all gaps are created equal. The type of gap determines your strategy:

**Earnings gap (most reliable):**
- Company reports earnings before/after hours
- Gap reflects the market's repricing of the stock based on new fundamental data
- These gaps tend to continue (70%+ of the time for strong beats/misses)
- Trade: gap-and-go with trend, or wait for pullback entry

**News gap (variable reliability):**
- FDA approval, contract win, product launch, legal settlement
- Quality varies enormously — "partnership with major company" vs "updated website"
- Evaluate the news quality before trading; headline reading is a skill
- Trade: gap-and-go if catalyst is genuine; cautious if catalyst is ambiguous

**Momentum gap (lower reliability):**
- Stock ran 30% yesterday and is gapping up another 10% today on no new news
- Often driven by social media, retail FOMO, or short squeeze mechanics
- Higher risk — these are the gaps that can reverse violently
- Trade: very tight stops, take profits quickly, reduce size vs earnings gaps

**Technical gap (variable):**
- Breaking out above a multi-month resistance level
- Gap on sector rotation (sector ETF breaking out, dragging individual stocks)
- Usually lower percentage gaps (2-5%) but can start sustained moves
- Trade: only if confirmed by volume; many technical breakouts fail

**Gap down opportunities:**
- Earnings miss, guidance cut, product failure
- Gap downs on bad news tend to bounce initially (dead cat bounce), then resume falling
- Short the dead cat bounce, not the initial drop
- Or: fade the gap if the selloff is overdone relative to the news

## 4.4 The Catalyst Hierarchy

Before trading any scanned stock, verify the catalyst. Open the news feed, check the earnings calendar, read the headline. This takes 60 seconds and prevents you from trading stocks moving for no reason.

**Tier 1 — Trade with confidence (A+ setups):**
- Earnings beat/miss with revenue and guidance confirmation
- FDA approval or rejection
- M&A announcement (tender offer, acquisition)
- Index inclusion/exclusion (S&P 500 add/remove)

**Tier 2 — Trade with confirmation (A setups):**
- Major contracts or partnerships (quantified dollar value)
- Significant product launches
- Analyst initiations with price targets >20% away
- Short squeeze setups with identifiable catalyst

**Tier 3 — Trade cautiously (B setups):**
- Analyst upgrades/downgrades
- Sector sympathy moves (trading AMD because NVDA beat earnings)
- Technical breakouts above multi-month resistance
- Moderate earnings beats without guidance change

**Tier 4 — Avoid or paper trade only (C setups):**
- Social media hype without fundamental basis
- "Runner" from yesterday with no new news today
- Stocks halted pending news (unknown catalyst = unknown risk)
- Sub-$2 stocks with "biotech breakthrough" press releases

## 4.5 Float and Share Structure

Float is the number of shares available for public trading. It excludes insider shares, restricted shares, and institutional holdings that don't actively trade.

**Why float is the most important structural variable:**

A stock with 5 million float and 1 million shares of buying demand will move 10x more than a stock with 500 million float and the same buying demand. Float determines the "leverage" of demand on price.

**Float categories and their characteristics:**

| Float | Behavior | Risk Level | Best Strategy |
|-------|----------|------------|---------------|
| < 5M (micro float) | Explosive moves (50-200%+), halts, extreme spreads | Very high | Experienced only, tiny size |
| 5-20M (low float) | Large moves (10-50%), good liquidity on high RVOL days | High | Momentum scalping, tight stops |
| 20-50M (medium float) | Moderate moves (5-15%), decent liquidity | Medium | Best risk/reward for most traders |
| 50-100M (high float) | Smaller moves (2-8%), institutional participation | Medium-low | VWAP strategies, ORB |
| > 100M (mega float) | Predictable moves (0.5-3%), deep liquidity | Low | ORB, trend following, VWAP |

**Short interest and float:** When short interest exceeds 20% of the float, a squeeze becomes structurally possible. When short interest exceeds 40%, a squeeze is increasingly likely if a positive catalyst arrives. Monitor short interest as part of the pre-market scan.

## 4.6 Pre-Market Price Action as a Filter

A stock that passes the scan criteria but has bad pre-market price action should be demoted or skipped.

**Good pre-market action (confirms the setup):**
- Stock gapped up 6% and has held or increased the gap in pre-market
- Pre-market volume is building (each 15-minute bar has more volume than the last)
- The stock is above VWAP (pre-market VWAP)
- There's a clear level to trade against (pre-market high, round number)

**Bad pre-market action (demote the setup):**
- Stock gapped up 8% but has faded to only +3% in pre-market (gap is being sold)
- Pre-market volume is declining
- The stock has made a lower high in pre-market (sellers are in control)
- Wide, erratic price action with no clear level

**The "magnet" effect:** Stocks tend to get pulled toward key levels in pre-market — yesterday's close, VWAP, round numbers. If a stock gaps to $10.80 and slowly drifts toward $10.00 in pre-market, $10.00 is acting as a magnet. This tells you that the market views $10.00 as the "right" price, and the gap may fade.

## 4.7 Building a Watchlist

From the scanner output (potentially 20-50 stocks), narrow to a focused watchlist of 3-5 names. Maximum. More than 5 and your attention is too divided to execute well.

**Selection process:**
1. Filter by catalyst quality (Tier 1 and 2 only for A+ watchlist)
2. Rank by relative volume (higher = more likely to move)
3. Check pre-market price action (holding gap? building volume?)
4. Mark key levels on each chart (pre-market high/low, yesterday's high/low, VWAP, round numbers)
5. Define entry/stop/target for each name BEFORE the open

**The primary trade:** Your #1 watchlist stock is the one where your conviction is highest. This gets the most attention and the largest position size. If only one stock from the watchlist triggers, it should be this one.

**The backup trade:** #2 and #3 are backups in case #1 doesn't trigger or gets halted. Having alternatives prevents FOMO-chasing when the primary doesn't work.

**The "too good to ignore" slot:** Leave room for a stock that appears on the scanner at 9:25 AM — sometimes the best gap shows up late. But don't let it override your planned trades unless it's clearly superior.

## 4.8 The Anti-Scan: What to Avoid

Knowing what NOT to trade is as important as knowing what to trade.

**Hard avoids:**
- **Sub-$1 stocks:** Manipulation, delisting risk, impossible to short, wide spreads eat all profit
- **Stocks with pending news halts:** You don't know what the news is; binary gambling
- **Biotech with FDA binary event today:** 50/50 bet on regulatory decision; this is a coin flip, not trading
- **Stocks with history of manipulation:** Repeated pump-and-dump patterns; check the daily chart for this
- **Anything from a "stock alerts" chat room:** By the time you see the alert, the alerter has already entered. You're their exit liquidity

**Soft avoids (reduce size or skip):**
- **ADRs and foreign stocks:** Different market hours, less predictable behavior, potential currency risk
- **ETFs with less than $50M average daily volume:** Tracking error, wide spreads
- **Stocks already up 100%+ pre-market:** The easy money is already made; risk/reward is poor for latecomers
- **Stocks involved in SEC investigation or trading halt:** Elevated risk of permanent loss
- **Any stock you've lost money on three times in a row:** You don't understand it; move on
