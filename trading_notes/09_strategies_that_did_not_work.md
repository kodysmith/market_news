# Chapter 9: Strategies That Did Not Work

## 9.1 Box Spread Recycling

This was the idea that excited me the most — and disappointed me the most when the numbers came in.

**The theory:** When an iron condor reaches its 50% profit target, instead of closing it for cash, convert it to a box spread. A box spread is a guaranteed-value structure (a bull call spread combined with a bear put spread at the same strikes) that can be sold to dealers at slightly below par. The conversion locks in profit without closing the position, and the freed margin can be deployed into SGOV (short-term Treasury ETF) to earn risk-free yield while waiting for the box to settle.

On paper, this sounds like financial alchemy: you keep your profit, free your margin, earn interest on the freed margin, and let the box settle naturally. What's not to love?

**The reality:** Everything about the execution costs more than the theory predicts.

**Conversion slippage: -$170 per trade.** Converting an iron condor to a box spread requires trading two additional legs. Each leg has bid-ask slippage. The box itself often trades at a discount to par value (dealers demand compensation for carrying the position). The total cost of conversion averaged $170 per trade — significantly more than the $50-80 we initially estimated.

**SGOV yield on freed margin: +$33 per trade.** At a risk-free rate of approximately 5% annually, $3,300 of freed margin (one contract's worth) earns about $3.70 per day. With an average holding period of 9 days, that's $33 per trade. Not nothing, but a fraction of the conversion cost.

**Net result: -$137 per conversion.** Every conversion destroys $137 of value. Over 200+ trades per year, that's approximately -$5,850 in annual drag versus simply closing the iron condor at the profit target.

**Why the theory fails:** The theory assumes zero-cost conversion (it costs $170), institutional SGOV yields (retail earns less), and immediate redeployment of freed margin (opportunities don't always exist at the right time). The theory also ignores the complexity cost — box conversions require careful leg management and can go wrong if one leg fills and another doesn't.

**The only scenarios where box conversion makes sense:**
- Before a weekend with elevated risk, when holding the position would be dangerous but closing it wastes remaining theta
- When the position is at 70%+ profit and you want guaranteed zero risk of reversal
- When you need to free margin immediately for a higher-conviction trade

These are tactical uses, not systematic ones. The systematic box conversion strategy is dead.

**What I learned:** Elegant theories with small edges are easily destroyed by transaction costs. The $137 per-trade loss was hiding in three places that each seemed minor: slippage ($0.05 more per leg than estimated), box discount (2-3 cents worse than par), and SGOV yield (lower than the risk-free rate due to ETF fees). None of these individually killed the strategy. Together, they made it unviable.

## 9.2 Bear Calls in Neutral Regimes

This was a more subtle failure — not catastrophic, but definitively suboptimal.

**The theory:** In a neutral market (SPX neither trending up nor down), selling call spreads should be relatively safe because the market isn't making new highs. Bear calls collect premium from the call side without the complexity of a full iron condor.

**What actually happened:** In neutral regimes, the market moves sideways — which means it moves *both* up and down within a range. The upward moves within that range are enough to breach call spreads, especially at 15-delta with 7-14 DTE.

| Strategy | Win Rate in Neutral | Sharpe in Neutral |
|----------|-------------------|------------------|
| Bear Call (BC2, 15δ) | 83% | 1.54 |
| Iron Condor (SD1, 15δ) | 89% | 2.57 |

The iron condor wins decisively. Why? Because in a neutral market, both the put side and the call side are equally safe. The iron condor collects premium from both sides, effectively doubling the credit for the same underlying exposure. The bear call only collects from one side — less credit for similar risk.

**The lesson:** Directional strategies (bear calls, bull puts) should only be used in directional markets. In neutral markets, the non-directional iron condor is strictly superior because it captures premium from both sides of the range.

I spent three months backtesting variations of bear calls — different deltas, different widths, different DTEs, different VIX filters. None of them matched the iron condor in neutral regimes. The structural advantage of collecting premium from both sides is too large to overcome with parameter optimization.

## 9.3 Directional Bets Without Hedging

Early in the project, before the regime engine existed, I experimented with using market signals to make directional trades. The logic seemed sound: if VIX is spiking and SPX is below its SMA50, sell call spreads aggressively. If VIX is low and SPX is above SMA200, sell put spreads aggressively.

**What happened:** The directional bets worked when the trend continued. They produced spectacular losses when the trend reversed.

A concrete example: in late September 2022, all signals screamed "bearish." SPX was below SMA50, VIX was above 30, and the trend was clearly down. I would have deployed maximum bear call positions. Then, on October 13, SPX reversed and rallied 15% in six weeks. Any bear call position opened in late September would have been demolished.

**The lesson:** Markets can stay irrational longer than you can stay solvent. Directional conviction, even when supported by multiple signals, doesn't protect you from the market's ability to reverse violently and without warning.

This experience was the genesis of the regime engine's asymmetric transition rules. The engine doesn't make directional bets — it adjusts position sizing based on the current state. And the "slow up" transition rules exist specifically because of this failure: entering BULL mode too quickly after a bearish period can trap you in a trend reversal.

## 9.4 Long-Dated Spreads for Income (21+ DTE)

Conventional options wisdom suggests that 30-45 DTE is the "sweet spot" for selling premium — far enough from expiration that gamma is manageable, close enough that theta decay is meaningful. I tested this thoroughly and found that for *income* strategies specifically, longer dates are inferior.

**The problem with 21+ DTE for income:**

1. **Directional exposure accumulates.** With 21 days of holding, the market can trend 3-5% in one direction. A 10-delta position that starts 100 points away from the current price can find itself 20 points away after a sustained move. This doesn't happen with 1-3 DTE positions that reset daily.

2. **Capital is tied up longer.** $33K of margin locked for 21 days produces the same return as $33K of margin turned over 7 times in 1DTE positions. The daily strategy wins on capital efficiency.

3. **Theta decay is back-loaded.** A 21DTE option doesn't decay at a constant rate. Most of the theta occurs in the final 5-7 days. For the first 14 days, you're mostly just exposed to directional and volatility risk without collecting proportional theta.

4. **Drawdowns are longer.** A 1DTE position that loses money is over in 2 days. A 21DTE position that moves against you can be underwater for weeks before the loss is realized. This creates psychological pressure and capital lockup during exactly the conditions where you'd want flexibility.

Our data shows the relationship clearly:

| DTE | Annual P&L (10 cts) | Capital Efficiency |
|-----|---------------------|-------------------|
| 1 | $282K | $282K per $33K margin |
| 3 | $285K | $285K per $33K margin |
| 7 | $238K | $238K per $44K margin |
| 14 | $205K | $205K per $49K margin |
| 21 | $178K | $178K per $56K margin |

Both absolute returns and returns per dollar of margin decline as DTE increases. The exceptions are strategies with heavy filtering (SWEET, AG2) where the selectivity compensates for the longer DTE.

## 9.5 Parameter Sweep Overfitting

This is the most important failure in the entire book, because it's the one that every quantitative trader will encounter — and the one that causes the most financial damage.

**The scenario:** I ran a comprehensive parameter sweep across delta (5-25), width ($20-$100), DTE (1-30), and profit target (30%-80%). That's approximately 5,000 parameter combinations. The best combination produced a backtest Sharpe of 4.2 with annual returns of $450K per year.

I was briefly euphoric. Then I ran it out of sample.

**The out-of-sample Sharpe: 1.1.** The in-sample Sharpe of 4.2 collapsed to 1.1 — a 74% reduction. The strategy wasn't capturing a real pattern; it was capturing noise in the specific sequence of market events from 2015 to 2020.

**Why this happens:** With 5,000 parameter combinations, you're essentially buying 5,000 lottery tickets. Some of them will produce spectacular results by pure chance. The more parameters you test, the more likely you are to find a combination that happens to work on the historical data. This is not a real edge — it's a statistical artifact.

**How to prevent it:**

1. **Limit the number of parameters.** Our final strategies have 3-4 key parameters each (delta, width, DTE, profit target). This dramatically reduces the parameter space and the opportunity for overfitting.

2. **Use walk-forward validation.** Never evaluate a strategy on the data used to select its parameters. The OOS/IS ratio is the single most important diagnostic for overfitting.

3. **Require economic justification.** Every parameter should have a reason. We use 10-delta because it approximates a 90% probability of profit. We use $30 width because it limits maximum loss to a manageable level. We use 1DTE because theta acceleration is maximized. These aren't arbitrary — they're grounded in options math.

4. **Check parameter neighborhood stability.** If changing delta from 10 to 11 causes a large change in performance, the result at delta=10 is likely noise. Our strategies show stable performance across ±2 delta points, ±$10 width, and ±1 DTE.

## 9.6 Over-Aggressive Position Sizing

The Kelly criterion, developed by John Kelly in 1956, tells you the optimal fraction of your bankroll to bet on each opportunity. For a strategy with a 90% win rate and a 3:1 loss-to-win ratio, full Kelly suggests betting approximately 57% of your bankroll on each trade.

I tried this. It was terrifying.

**The problem with full Kelly:** It assumes you know the true win rate and payoff ratio with perfect precision. You don't. Your estimates come from historical backtests, which have estimation error. If your true win rate is 87% instead of 90%, full Kelly sizing is dramatically too large and leads to ruin.

**What happened:** At full Kelly (roughly 20 contracts on a $700K account), the first losing streak produced a 45% drawdown. The strategy was still positive in expectation, but the drawdown was psychologically unbearable and came dangerously close to the point where recovery becomes impractical.

**The solution: half Kelly.** By using 50% of the Kelly-optimal sizing, you sacrifice approximately 25% of the expected return but reduce the probability of ruin from "non-negligible" to "essentially zero." Our standard sizing of 10 contracts for IC3 is approximately half Kelly for the estimated edge.

| Sizing | Annual P&L | Max Drawdown | Prob of 50% DD |
|--------|-----------|--------------|----------------|
| Full Kelly (20 cts) | $564K | $180K (26%) | ~8% |
| Half Kelly (10 cts) | $282K | $78K (11%) | <0.1% |
| Quarter Kelly (5 cts) | $141K | $35K (5%) | ~0% |

Half Kelly is the sweet spot: still generates substantial returns, but the drawdowns are manageable and the probability of catastrophic loss is negligible.

## 9.7 The Value of Dead Ends

Every failed strategy in this chapter taught something essential:

- **Box spread recycling** taught that transaction costs destroy small edges
- **Bear calls in neutral regimes** taught that non-directional beats directional in ambiguous conditions
- **Directional bets without hedges** led directly to the regime engine
- **Long-dated spreads** confirmed that capital efficiency matters as much as per-trade edge
- **Parameter sweep overfitting** established the validation discipline that prevents self-deception
- **Over-aggressive sizing** established the half-Kelly rule that preserves capital through drawdowns

The strategies that survived — IC3, IC6, FLY14, and the regime engine — are the ones that passed through this filter. They work not because I found them first, but because I found everything else first and eliminated it. The final system is not the product of insight. It's the residue of extensive elimination.

If you take only one lesson from this chapter, let it be this: **the willingness to honestly evaluate and discard strategies that don't work is the single most valuable skill in quantitative trading.** The market rewards humility and punishes attachment to ideas. Every strategy in this book earned its place by surviving rigorous testing. The ones in this chapter earned their place by failing informatively.
