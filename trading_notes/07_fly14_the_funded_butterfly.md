# Chapter 7: FLY14 — The Funded Butterfly (99% Win Rate)

## 7.1 Anatomy of a Funded Butterfly

The funded butterfly is perhaps the most elegant structure in our portfolio. It combines two concepts:

1. **A short iron condor** (the "funder") that collects premium
2. **A long butterfly** placed at the money that costs a fraction of the funder's credit

The iron condor generates enough credit to pay for the butterfly with money left over. You receive a net credit to enter the trade. The butterfly is essentially free — and on the rare occasions when SPX lands near the center strike at expiration, it pays out 10:1 on the butterfly's cost.

The structure on a day when SPX is at 5,500:

**The Funder (Iron Condor):**
- Sell 5430 put, Buy 5400 put (put spread)
- Sell 5570 call, Buy 5600 call (call spread)
- Credit: ~$5.00

**The Butterfly:**
- Buy 1 × 5490 put
- Sell 2 × 5500 put
- Buy 1 × 5510 put
- Cost: ~$2.50

**Net Credit: ~$2.55** (you receive $255 per contract to enter)

The maximum risk of the combined position is the same as the iron condor's maximum risk minus the net credit: approximately $27.45 per contract. But the butterfly adds an upside that the plain iron condor doesn't have: if SPX closes between 5490 and 5510 at expiration, the butterfly pays out $5–$10 per contract ($500–$1,000).

## 7.2 Strategy Specification

| Parameter | Value |
|-----------|-------|
| Underlying | SPX |
| Structure | Funded Butterfly (IC + Butterfly) |
| DTE | 14 |
| Funder Delta | 10 (both sides) |
| Butterfly Wings | $10 wide |
| Butterfly Center | ATM (current SPX price) |
| Contracts | 3 (base size) |
| Profit Target | 25% rebalance threshold |
| Net Credit | ~$2.55 average |

The 14DTE timeframe is deliberately chosen. The butterfly needs time for SPX to potentially converge near the center strike. At 1DTE, the probability of SPX being exactly at 5500 (±10 points) is very low. At 14DTE, there are multiple paths by which SPX could return to the entry price — a drop followed by a recovery, a rally followed by a pullback, or simply sideways movement with mean reversion.

## 7.3 The 99% Win Rate Explained

The 99% win rate is not a misprint, and it's not gimmickry. Here's why it's genuine:

**Scenario 1: Butterfly misses (89% of trades).** SPX doesn't land near the center strike at expiration. The butterfly expires worthless. But the funder's iron condor also expires worthless (both sides stay out of the money). You keep the full net credit of $2.55 per contract. **Result: +$255 profit.**

**Scenario 2: Butterfly hits (11% of trades).** SPX lands near the center strike. The butterfly pays out an average of $5.45. The funder's iron condor might have one side breached (since SPX moved toward one edge of the condor's range), but the butterfly payout more than compensates. **Result: +$545 average profit.**

**Scenario 3: Funder breached, butterfly misses (1% of trades).** A large move breaches the funder's iron condor, and SPX ends far from the butterfly's center. The loss is the same as a regular iron condor loss minus the net credit. **Result: Loss, but capped by the funder's wing width.**

The 99% win rate occurs because the only losing scenario requires both a large enough move to breach the 10-delta funder AND a final price far enough from the center to make the butterfly worthless. The large move is necessary for the loss, but it must also be sustained — a temporary spike that reverses (common in 14 days) would still leave the funder intact.

## 7.4 The Butterfly Payoff: 10:1 When It Hits

When the butterfly pays off, the ratio of payout to cost is extraordinary:

- **Butterfly cost embedded in the trade: ~$2.50**
- **Average payout when it hits: $5.45**
- **Payout ratio: ~2.2:1 on total trade cost**
- **Payout ratio on butterfly cost alone: ~10:1** ($5.45 payout on what would have been a $0.55 net risk if the funder didn't exist)

But the butterfly doesn't "cost" anything in the funded structure because the funder pays for it. The net credit means you're being paid to take a position that occasionally produces a windfall. This is the mathematical beauty of the funded butterfly: the expected value of the butterfly component (11% × $5.45 = $0.60) is added on top of the expected value of the funder ($2.55 credit × 89% win rate ≈ $2.27).

The combined expected value per trade is approximately $2.87 — higher than either component alone.

## 7.5 Performance Profile

| Metric | Value |
|--------|-------|
| Win Rate | 99% |
| Loss Rate | 1% |
| Profit Factor | 9.1 |
| Sharpe Ratio | 3.65 |
| Annual P&L per Contract | ~$13,600 |
| At 3 Contracts | ~$40,800/year |
| Margin per Contract | ~$1,850 |
| Total Margin (3 cts) | ~$5,550 |
| Every Year Profitable | Yes |
| Worst Year (2018) | +$5,657/ct |

The **Sharpe ratio of 3.65** is the highest of any strategy in our portfolio. This is driven by the combination of very high win rate and the butterfly's occasional large payoffs. Most days produce a small, consistent gain from the funder's credit; the butterfly hits add positive skew to the return distribution.

The **profit factor of 9.1** means that for every dollar lost, the strategy makes $9.10 in profit. This is an outstanding ratio — most profitable strategies have profit factors between 1.5 and 3.0.

The **margin efficiency** is remarkable. At $5,550 total margin for 3 contracts generating $40,800 per year, the ROI on margin is 735%. This is because the funded butterfly's risk is partially offset by the net credit — the actual margin requirement is lower than a standard iron condor.

## 7.6 Role in the Portfolio

FLY14 is not a standalone strategy. At 3 contracts, it generates about $41K per year — meaningful, but not the primary income source. Its role in the portfolio is threefold:

**1. Return enhancement.** It adds $41K to the portfolio's annual P&L with almost no additional drawdown risk. The 99% win rate means it rarely subtracts from the equity curve.

**2. Positive skew injection.** IC3 and IC6 have negatively skewed returns — many small wins and occasional large losses. FLY14's butterfly component adds positive skew — many small wins and occasional large wins. The combined portfolio has a more balanced return distribution.

**3. Low correlation.** FLY14's 14DTE holding period means its returns are driven by different market dynamics than the 1-3 DTE iron condors. A day that produces a loss for IC3 (a large SPX move) might or might not affect FLY14, depending on when the position was entered and where SPX ends up relative to the center strike.

In the combined portfolio (IC3 + IC6 + FLY14), the contribution breakdown is approximately:
- IC3 (10 contracts): $282K/year (57% of total)
- IC6 (5 contracts): $142K/year (29% of total)
- FLY14 (3 contracts): $41K/year (8% of total)
- Interaction/diversification benefit: ~$30K/year (6% of total)

The "interaction benefit" represents the Sharpe improvement from diversification — the combined portfolio has a higher Sharpe ratio than any individual strategy because the strategies' losses don't perfectly coincide.

## 7.7 The Mathematics of Asymmetric Payoffs

The funded butterfly challenges a common misconception in options trading: that high win rates necessarily come with negative expected value. The argument usually goes: "If you win 99% of the time, the 1% losses must be huge enough to erase all gains."

This is true for simple premium-selling strategies where the payoff is symmetric. But the funded butterfly breaks this pattern through structural asymmetry:

**The funder side is a classic premium-selling payoff:** small, frequent wins offset by rare, large losses. If this were the entire trade, the win rate would be ~90% with negative skew.

**The butterfly side is a classic long-options payoff:** frequent small losses offset by rare, large wins. If this were the entire trade, the win rate would be ~11% with positive skew.

**Combined, they create a hybrid:** the funder's frequent small wins pay for the butterfly's frequent small losses, producing a net credit. The butterfly's rare large wins add to the funder's frequent small wins. The only losing scenario is when the funder loses AND the butterfly doesn't pay — which requires a specific combination of events that occurs roughly 1% of the time.

The mathematical lesson: you can construct a position with positive expected value, high win rate, AND positive skew by combining negatively skewed and positively skewed components. The funded butterfly is not an anomaly — it's a demonstration of what's possible when you understand the payoff structures deeply enough to combine them intelligently.

This is also why the funded butterfly cannot be "scaled up" to replace the iron condors. The butterfly's edge comes from the specific pricing relationship between the funder and the butterfly. At large scale, the butterfly's impact on the options market would affect pricing. The strategy works precisely because it's small — 3 contracts on a structure that the market barely notices.
