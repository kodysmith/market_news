# Chapter 8: The Supporting Cast — Weekly and Conditional Strategies

## 8.1 SWEET: The VIX Sweet Spot Strategy

Not all strategies trade every day. SWEET trades only when two conditions align: SPX is above its 50-day moving average (confirming an uptrend) AND VIX is between 16 and 24 (the "sweet spot" where premiums are rich but not dangerously so).

| Parameter | Value |
|-----------|-------|
| DTE | 14 |
| Delta | 15 |
| Width | $50 |
| VIX Filter | 16–24 |
| Trend Filter | SPX > SMA50 |
| Contracts | 8 |
| Trades/Year | ~11 |
| Win Rate | 89% |
| Sharpe | 2.95 |
| Annual P&L (8 cts) | ~$41K |

The genius of SWEET is its patience. It trades only 11 times per year — roughly once per month — waiting for the specific confluence of trend and volatility that historically produces the best risk-adjusted returns. When VIX is below 16, premiums are too thin to justify the risk. When VIX is above 24, the market is too volatile for a 15-delta position to be safe without additional hedging.

The Sharpe ratio of 2.95 is among the highest in the portfolio, driven by the selectivity of the entry filter. By only trading in optimal conditions, SWEET avoids the marginal trades that drag down the average.

**The lesson from SWEET:** Sometimes the best trade is no trade. A strategy that waits for ideal conditions and trades infrequently can produce better risk-adjusted returns than one that trades every day. The daily strategies (IC3, IC6) succeed through volume and consistency. SWEET succeeds through selectivity.

## 8.2 SD1: The Weekly Base Case

SD1 is the workhorse of the weekly strategies — a straightforward 7DTE iron condor that trades whenever SPX is above its 50-day moving average.

| Parameter | Value |
|-----------|-------|
| DTE | 7 |
| Delta | 15 |
| Width | $50 |
| Trend Filter | SPX > SMA50 |
| Contracts | 8 |
| Trades/Year | ~34 |
| Win Rate | 87% |
| Sharpe | 2.57 |
| Annual P&L (8 cts) | ~$24K |

At 34 trades per year (roughly weekly when the trend filter is active), SD1 provides steady weekly income. The 7DTE timeframe offers a good balance between theta decay and gamma risk — enough time for the position to work but not so much time that directional risk becomes dominant.

The $50 wing width (versus IC3's $30) reflects the longer DTE — with a week of exposure, wider wings provide a more comfortable margin of error. The trade-off is higher margin per contract and a larger maximum loss if the position is breached.

SD1's role in the portfolio is to provide "base load" income during trending markets. It doesn't shine in any single metric — it's not the highest win rate, not the highest Sharpe, not the highest absolute return. But it's consistently profitable and it's simple to execute.

## 8.3 AG2: Volatility Detection

AG2 is the most sophisticated of the weekly strategies, adding a volatility-specific filter to the trend filter:

| Parameter | Value |
|-----------|-------|
| DTE | 21 |
| Delta | 20 |
| Width | $75 |
| Trend Filter | SPX > SMA50 |
| Vol Filter | Realized vol ratio ≤ 1.05 |
| Contracts | 6 |
| Trades/Year | ~20 |
| Win Rate | 89% |
| Sharpe | 2.57 |
| Annual P&L (6 cts) | ~$89K |

The **rv_ratio filter** is the key innovation. The realized volatility ratio compares current implied volatility to recent realized volatility. When the ratio is at or below 1.05, it means implied volatility is not significantly overstating actual market movement — the premium is "fair" rather than inflated by fear. Paradoxically, this turns out to be a good time to sell premium, because the market tends to move *less* than implied volatility suggests during these periods.

The 21DTE / 20-delta / $75-wide specification is the most aggressive in our portfolio. Wider delta, wider wings, and longer holding period mean each trade carries more risk. But the volatility filter and trend filter together ensure that AG2 only trades when conditions strongly favor the position.

The result is $89K per year from only 6 contracts — the highest per-trade profitability in the portfolio. When AG2 trades, it trades big. When conditions aren't right, it sits in cash.

## 8.4 VD2: The Compounding Engine

VD2 introduces a concept that no other strategy in our portfolio uses: **automatic position scaling based on cumulative profit.**

| Parameter | Value |
|-----------|-------|
| DTE | 14 |
| Delta | 15 |
| Width | $50 |
| Trend Filter | SPX > SMA50 |
| Base Contracts | 6 |
| Scaling Rule | +1 contract per $30K banked profit |
| Win Rate | 90% |
| Sharpe | 2.94 |
| Annual P&L | ~$23–38K (varies with scaling) |

The compounding mechanism works as follows: start with 6 contracts. After every $30,000 in cumulative profit, add 1 contract to subsequent trades. If cumulative profit reaches $60,000, trade 8 contracts. At $90,000, trade 9. And so on.

This creates an exponential growth curve — as profits accumulate, position sizes increase, which generates more profit, which increases position sizes further. Over an 11-year backtest, VD2's contract count grows from 6 to approximately 12-14, more than doubling the effective capital deployment.

The risk is that larger positions mean larger losses when they occur. But the scaling is gradual ($30K per increment) and tied to realized profit — you're only sizing up with house money. If a losing streak hits, the contract count doesn't increase during the drawdown, providing a natural brake on risk.

**The compounding lesson:** Small, consistent profits reinvested systematically produce extraordinary long-term results. VD2 is the mathematical proof of the adage "let your winners compound."

## 8.5 BC4 and BC2: Bear Market Income

Most of our strategies require a bullish bias (SPX > SMA50). What happens when the market turns bearish? That's where the bear call strategies come in.

**BC4 — Bear Call Conservative:**

| Parameter | Value |
|-----------|-------|
| DTE | 14 |
| Delta | 10 |
| Width | $50 |
| Filter | SPX < SMA50 only |
| Contracts | 6 |
| Win Rate | 89% |
| Sharpe | 1.89 |
| Annual P&L (6 cts) | ~$6K |

**BC2 — Bear Call 7DTE:**

| Parameter | Value |
|-----------|-------|
| DTE | 7 |
| Delta | 15 |
| Width | $50 |
| Filter | SPX < SMA50 only |
| Contracts | 6 |
| Win Rate | 83% |
| Sharpe | 1.54 |
| Annual P&L (6 cts) | ~$8K |

The bear call strategies sell call spreads only — no put side. In a bearish market (SPX below SMA50), selling calls is theoretically safer because the market's downward momentum should keep prices below the call strikes. In practice, the results are... adequate.

The Sharpe ratios of 1.54–1.89 are the lowest in the portfolio. The absolute returns of $6–8K are modest. The win rates of 83–89% are acceptable but not exceptional. Why are the bear call strategies weaker?

**Bear markets are choppy.** When SPX is below its SMA50, the market doesn't drop steadily — it drops, rallies sharply, drops again, rallies again. These sharp rallies (bear market bounces) breach call spreads that were set during the decline. The directional nature of a call-only position (as opposed to the non-directional iron condor) makes it vulnerable to these reversals.

**Volume is lower.** SPX spends roughly 25-30% of the time below its SMA50. This means the bear call strategies trade far less frequently than the bullish-biased strategies, producing less total income.

The bear calls exist in the portfolio primarily for completeness — ensuring some income generation occurs even in bearish markets. But the honest assessment is that they are the weakest link. In the regime-aware framework, BEAR mode replaces them entirely by going to cash (SGOV), which has historically been a better approach than trying to generate income during adverse conditions.

## 8.6 Why Most Require the SMA50 Filter

Six of our eight strategies require SPX to be above its 50-day simple moving average before entering. Only IC3 and IC6 (the daily cash flow strategies) trade regardless of trend.

Why is the SMA50 filter so important for weekly and bi-weekly strategies?

**Longer holding periods create directional exposure.** A 14DTE iron condor has 14 days during which the market can trend in one direction. In a bull market (SPX > SMA50), the market's upward drift helps keep prices away from the short put strike. In a bear market (SPX < SMA50), downward drift pushes prices toward the short put — and the 14 days of exposure gives the trend time to work against the position.

**1DTE positions reset too quickly to be affected by trends.** The daily strategies avoid this problem because each position is held for less than 2 days on average. There isn't enough time for a trend to meaningfully affect the outcome. The daily noise in SPX (which is mostly random at the 1-day scale) dominates over the trend signal.

**The SMA50 is simple and robust.** We tested multiple trend indicators — SMA20, SMA100, SMA200, exponential moving averages, MACD crossovers. The SMA50 consistently produced the best results for our holding periods. It's fast enough to respond to regime changes (catching the 2022 bear market within 2-3 weeks) but slow enough to avoid false signals from normal volatility.

## 8.7 The Combined Tier 2 Portfolio

When all weekly/bi-weekly strategies are combined into a single portfolio with proper position management:

| Metric | Value |
|--------|-------|
| Total 11-Year P&L | $3,020,407 |
| Annual Average | $280,968 |
| CAGR | 16.8% |
| Max Drawdown | $238K (34% of $700K) |
| Sharpe Ratio | 2.77 |
| Starting Capital | $700K |
| Ending Capital | $3.72M |
| Avg Capital Deployed | $87K (12.4% of account) |

The most striking number is the average capital deployed: only $87K of a $700K account is in active positions at any time. This means 87.6% of the capital is sitting idle — available for deployment during opportunities or as a cushion against drawdowns.

This capital efficiency raises an obvious question: why not deploy more? The answer is risk management. The $238K maximum drawdown (34% of starting capital) occurs even with conservative sizing. If we doubled the deployment to $174K, the maximum drawdown would roughly double to $476K — 68% of the account, which approaches the danger zone where recovery becomes difficult.

The 12.4% deployment rate is not laziness. It's the mathematically optimal level where the compounding benefit of returns is not offset by the compounding damage of drawdowns. This principle — capital preservation enables compounding — will be explored in depth in the risk management chapters.
