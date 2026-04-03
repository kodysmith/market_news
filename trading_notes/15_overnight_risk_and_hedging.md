# Chapter 15: Overnight Risk and the Case for Hedging

## 15.1 The Gap Risk Problem

Every options position held overnight is exposed to gap risk — the possibility that the market opens significantly higher or lower than the previous close. For short premium positions, gaps are the primary source of catastrophic losses.

Historical SPX overnight gap statistics (2015–2025):
- Average overnight gap: ±0.3%
- Gaps > 1%: occur roughly 12 times per year
- Gaps > 2%: occur roughly 3 times per year
- Gaps > 3%: occur roughly once per year

A 2% overnight gap on a 10-delta iron condor can transform a profitable position into a maximum loss instantly. The short strike that was 70 points away from the market at close is suddenly only 10 points away at the open — and there was no opportunity to exit during the gap.

For our IC3 strategy (1DTE, 10 contracts), a 2% gap could produce a loss of $25,000-$50,000 before any exit is possible. This is the kind of event that the stop-loss partially mitigates but cannot fully prevent, because the gap has already moved the position past the stop level.

## 15.2 ATM Put Protection for Overnight Exposure

The overnight hedge strategy was developed and tested for our QQQ wheel strategy, but the principles apply equally to SPX premium selling:

**The mechanism:**
- At 3:50 PM ET (10 minutes before close), buy ATM (50-delta) put options with 1-2 DTE
- At 9:35 AM ET the next morning, sell the puts
- If the market gapped down overnight, the puts gain value, offsetting losses on short premium positions
- If the market was flat or up overnight, the puts lose some value (theta decay + adverse delta)

**Why ATM (50-delta)?**
- Maximum delta exposure per dollar of premium
- The hedge needs to move dollar-for-dollar with the underlying to provide meaningful protection
- OTM puts (20-30 delta) are cheaper but provide far less protection per dollar spent

## 15.3 The Numbers

The overnight hedge strategy was backtested extensively on the QQQ wheel strategy (2020–2025):

**Without overnight hedging:**
- Total return: 89%
- CAGR: 13.6%
- Sharpe: 0.84
- Max drawdown: -23.8%

**With overnight hedging:**
- Total return: 148%
- CAGR: 20.0%
- Sharpe: 1.60
- Max drawdown: -16.0%

**The improvement:**
- Returns: +59.1% higher
- Sharpe: +76% better
- Max drawdown: -33% lower (from -23.8% to -16.0%)

These are not incremental improvements. The overnight hedge strategy roughly doubles risk-adjusted returns while substantially reducing the worst-case drawdown.

## 15.4 Cost-Benefit Analysis

The hedge isn't free, but it's remarkably efficient:

| Metric | Value |
|--------|-------|
| Total hedges placed | 862 (Mon-Thu nights) |
| Total hedge cost | $431,000 |
| Total hedge revenue | $490,000 |
| **Net hedge profit** | **$58,799** |
| Average profit per hedge | $68 |
| Win rate on hedges | ~55-60% |

The market's structural upward bias means that on most nights, the market opens flat or higher, and the puts lose a small amount. But the losses are small (theta decay on a 1DTE ATM put is modest overnight) while the wins are large (a 2% gap down produces a substantial gain on ATM puts).

This asymmetry is the mirror image of the premium seller's usual asymmetry. As a premium seller, you win small and lose big. As an overnight hedge buyer, you lose small and win big. The two asymmetries partially cancel, creating a combined portfolio with more symmetric returns.

## 15.5 When to Skip the Hedge

Not every night requires hedging. The cost-benefit analysis identifies several scenarios where the hedge can be skipped:

**VIX below 12.** When VIX is extremely low, the probability of an overnight gap large enough to matter is negligible, and the put premium you'd pay is still nonzero. The expected value of the hedge is negative in ultra-low VIX environments.

**Holidays and shortened weeks.** The day before a market holiday, liquidity is thin and the market rarely gaps significantly. The hedge cost is not justified.

**Minimal exposure.** If your total short premium exposure is small (e.g., only 2 contracts of IC3 in CAUTION mode), the potential loss from a gap is manageable without hedging. The hedge cost might exceed the potential savings.

**Post-crash environments.** After a significant crash (BEAR regime), the market has already gapped down substantially. The probability of an *additional* large gap is actually lower than normal because much of the selling pressure has been exhausted. Additionally, put premiums are extremely elevated post-crash, making the hedge expensive.

## 15.6 The Psychological Dimension

Beyond the mathematical benefits, overnight hedging provides psychological advantages that are difficult to quantify but very real:

**Better sleep.** Knowing that a 3% overnight gap won't devastate the portfolio allows the trader to sleep without checking futures at 3 AM. This sounds trivial. It isn't. Chronic sleep disruption leads to poor decision-making, which leads to overrides of the systematic strategy, which leads to losses. The hedge breaks this cycle.

**More confident position sizing.** Without hedging, the fear of overnight gaps leads to under-sizing — trading 5 contracts instead of 10 "just in case." With hedging, the protection allows full-size deployment with confidence. The incremental P&L from proper sizing vastly exceeds the hedge cost.

**Fewer fear-driven exits.** Without hedging, an after-hours futures drop of 0.5% can trigger panic: "Should I close everything at the open?" This leads to selling positions at the worst possible time — during the fear-driven opening that often reverses by noon. With hedging, the same futures drop is covered, and the trader can execute the systematic exit rules without emotional interference.

These psychological benefits compound over time. A trader who sleeps well, sizes correctly, and doesn't panic will significantly outperform the same trader who is anxious, under-sized, and reactive — even if they're running the exact same strategy.

## 15.7 Weekend and Event-Specific Hedging

Overnight hedging covers Monday through Thursday nights. Weekends and specific events require different approaches:

**Weekend hedging:**
- Buy 7DTE puts on Friday afternoon to cover the weekend
- More expensive than 1DTE puts but covers 2+ days of potential gap risk
- Particularly important before long weekends (3-day weekends have produced some of the largest Monday gaps)

**FOMC/CPI events:**
- Major economic announcements (FOMC rate decisions, CPI releases) are scheduled in advance
- Buy event-specific hedges (14DTE puts) 2-3 days before the event
- These events can produce 2-3% moves in either direction, often within minutes of the announcement
- The hedge cost is justified by the elevated probability of a large move

**Earnings season:**
- While SPX itself doesn't have "earnings," major index constituents reporting can move the index 1-2%
- During peak earnings season (January, April, July, October), slightly increased hedge sizing is prudent
- The cost is modest; the protection during the 15-20 highest-impact reporting days per quarter is valuable

The comprehensive hedging framework:
- **Daily (Mon-Thu):** 1DTE ATM puts, routine cost, strong evidence of positive expected value
- **Weekends:** 7DTE puts on Friday, higher cost, justified for long weekends
- **Events:** 14DTE puts before known catalysts, selective deployment
- **BEAR regime:** No hedging needed because no positions are open
