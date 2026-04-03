# Chapter 6: IC6 — Adding Duration with 3DTE Iron Condors

## 6.1 Strategy Specification

The IC6 strategy is the second layer of our daily income portfolio. It runs alongside IC3, providing diversification through a slightly different risk profile.

| Parameter | Value |
|-----------|-------|
| Underlying | SPX |
| Structure | Iron Condor |
| DTE | 3 (enter today, expires in 3 trading days) |
| Short Strike Delta | 12 (both sides) |
| Wing Width | $30 |
| Contracts | 5 (base size, regime-adjusted) |
| Profit Target | 40% of credit received |
| Stop Loss | 2× credit received |
| Entry Time | 9:40 AM ET |
| Margin per Contract | ~$3,300 |
| Total Margin (5 cts) | ~$16,500 |

Note the differences from IC3: slightly wider delta (12 vs 10), lower profit target (40% vs 50%), and half the contract count. These adjustments reflect the different character of a 3DTE position — it has more time for theta to work but also more time for the market to move adversely.

The entry time is staggered to 9:40 AM — five minutes after IC3 — to avoid competing for fills at the same moment and to allow a few extra minutes for the opening range to establish.

## 6.2 The 3DTE Sweet Spot

Three days to expiration occupies an interesting position in the DTE spectrum. It's long enough to capture meaningful theta decay but short enough to avoid the multi-week directional exposure of 14+ DTE positions.

**Theta characteristics at 3DTE.** An option with 3 days to expiration decays roughly 30-40% of its value per day (for a 12-delta OTM option). Over the full 3-day holding period, approximately 75-85% of the initial premium will decay due to time passage alone, assuming SPX doesn't move dramatically. Compare this to a 14DTE option, where theta decay in the first 3 days is only about 20% of the total premium.

**Gamma is elevated but not extreme.** At 3DTE, gamma is higher than at 14DTE but significantly lower than at 0-1DTE. This means the position is less sensitive to intraday SPX moves than a 1DTE position. A 0.5% SPX move might cause a 20% change in a 1DTE iron condor's value but only a 10% change in a 3DTE iron condor. This reduced sensitivity means fewer stop-outs from normal market noise.

**The 12-delta choice.** We use 12-delta for IC6 instead of 10-delta for two reasons. First, the 3-day holding period gives the market more time to move, so we need slightly wider cushions to maintain a comparable win rate. At 12-delta, our strikes are placed roughly 80-90 points from the current SPX price, which means SPX needs to move approximately 1.5-1.7% in a single direction over three days to breach a strike. Second, the 12-delta options have slightly higher premium per option, which partially compensates for the longer holding period.

## 6.3 Performance Comparison with IC3

| Metric | IC3 (1DTE) | IC6 (3DTE) |
|--------|------------|------------|
| Win Rate | 90% | 88% |
| Annual P&L per Contract | $28,245 | $28,493 |
| Sharpe Ratio | 2.54 | 2.56 |
| Average Hold | 1.9 days | 2.7 days |
| Positive Days | 91% | 89% |
| Worst Single Day (per ct) | -$6,349 | -$5,890 |
| Every Year Profitable | Yes | Yes |

The similarity is striking. IC6 produces nearly identical annual returns per contract, with a nearly identical Sharpe ratio, at a slightly lower win rate. The two strategies have comparable risk profiles despite their different DTEs.

The slightly lower win rate (88% vs 90%) reflects the longer holding period — there's more time for the market to move adversely. But this is offset by the slightly higher premium collected (12-delta options at 3DTE are worth more than 10-delta options at 1DTE).

The **lower worst-day loss** (-$5,890 vs -$6,349) is counterintuitive but makes sense: a 3DTE position is less sensitive to a single-day gap because it has more time value remaining. A 1DTE position hit by a gap has almost no time value cushion — the move translates almost directly into intrinsic value loss.

## 6.4 Correlation and Diversification

The primary benefit of running IC3 and IC6 together is **diversification through time**. On any given day:

- IC3 has one position that expires tomorrow
- IC6 has up to three overlapping positions at different stages of their lifecycle

When IC3 has a bad day (a large SPX move breaches the short strike), IC6's positions may or may not be affected, depending on when they were entered and where their strikes are. A position entered three days ago at different strike prices provides a natural hedge against the specific strikes chosen today.

The **correlation between daily IC3 and IC6 returns** is approximately 0.65 — positively correlated (they both suffer on large move days) but far from perfectly correlated. This means combining them reduces portfolio volatility more than simply scaling up either one alone.

Consider two scenarios for deploying $50K of margin:
1. **15 contracts of IC3**: $282K × 1.5 = $423K annual P&L, but with concentrated 1DTE exposure
2. **10 contracts IC3 + 5 contracts IC6**: $282K + $142K = $424K annual P&L, but with diversified DTE exposure

The annual P&L is similar, but the combined portfolio has a smoother equity curve — fewer and smaller drawdowns — because the two strategies don't lose on exactly the same days.

## 6.5 Margin Overlap and Capital Efficiency

The average hold time of IC6 is 2.7 days, which means that at any given time, you might have 2-3 overlapping IC6 positions. Each position requires $16,500 in margin (5 contracts × $3,300), so the peak margin requirement could reach $49,500 if three positions are open simultaneously.

In practice, many positions exit early via the 40% profit target. The average margin utilization is closer to $33,000 — roughly two positions worth. Combined with IC3's $33,000, the total margin requirement for both layers is approximately $66,000 on average, which is 33% of the $200K risk budget.

The **40% profit target** (instead of IC3's 50%) deserves explanation. With a 3-day holding period, positions reach the 40% profit threshold faster than you'd expect because theta decay is front-loaded in the first 1-2 days. By closing at 40%, we exit most positions within 1.5-2 days — before the final day's gamma acceleration kicks in. This is slightly more conservative than IC3's 50% target, reflecting the fact that IC6 positions have more time for the market to reverse a winning position.

The 40% vs 50% profit target is one of those parameters that seems minor but compounds significantly over hundreds of trades per year. Our backtesting shows that 40% is optimal for 3DTE: it maximizes the Sharpe ratio by capturing enough profit to justify the trade while exiting before the risk-reward ratio deteriorates.
