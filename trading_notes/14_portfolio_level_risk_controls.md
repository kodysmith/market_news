# Chapter 14: Portfolio-Level Risk Controls

## 14.1 The Three Hard Limits

Our system enforces three overlapping portfolio-level risk limits that cannot be overridden:

1. **SPX exposure < $150,000.** Total notional exposure across all open SPX options positions must not exceed $150K. This prevents concentration in a single underlying, even though our primary strategies are all SPX-based.

2. **Account risk < 15%.** Total maximum loss across all open positions must not exceed 15% of total account value ($105K on a $700K account). This ensures that even in the absolute worst case — every position simultaneously reaching maximum loss — the account survives with 85% of capital intact.

3. **Maximum 6 concurrent spreads.** No more than 6 iron condor/spread positions can be open simultaneously. This limits the number of positions that can be affected by a single market event.

These limits are intentionally overlapping. Any single limit might allow too much risk in certain conditions. Together, they form a mesh that catches scenarios each individual limit would miss.

For example: the $150K exposure limit might allow 6 positions of $25K each, but the 6-spread limit prevents having 15 positions of $10K each (which would have the same total exposure but far more moving parts to manage). The 15% account risk limit catches scenarios where a few positions have outsized maximum loss relative to their notional exposure.

The key principle: **defense-in-depth.** No single control should be the only thing preventing catastrophe.

## 14.2 Kelly Criterion and Why Half Kelly Is the Right Answer

The Kelly criterion provides the mathematically optimal bet size for a repeated game with known probabilities. For our IC3 strategy:

- Win probability: 90%
- Average win: $300/contract
- Average loss: $1,800/contract

Full Kelly fraction = (0.90 × $300 - 0.10 × $1,800) / $300 = 0.30

This suggests betting 30% of capital on each trade, which translates to approximately 25 contracts on a $200K risk budget.

**Why we use half Kelly (10 contracts instead of 25):**

The Kelly criterion assumes you know the true probabilities with certainty. You don't. Your win rate estimate comes from historical backtests, which have estimation error. If the true win rate is 87% instead of 90%, the optimal Kelly fraction drops significantly, and the 25-contract position becomes dangerously over-sized.

Half Kelly sacrifices approximately 25% of expected return but provides enormous protection against parameter estimation error:

- **Full Kelly:** Maximum long-term growth rate, but ~12% probability of a 50%+ drawdown
- **Half Kelly:** 75% of maximum growth rate, with <0.1% probability of a 50%+ drawdown
- **Quarter Kelly:** 50% of maximum growth rate, essentially zero risk of ruin

The practical mathematics are stark. With full Kelly sizing, a run of 3 consecutive losses (which occurs roughly twice per year at 10% loss rate) produces a drawdown of approximately 35% of the account. With half Kelly, the same run produces approximately 15%. The first is psychologically devastating and potentially career-ending. The second is uncomfortable but manageable.

Half Kelly is not timidity. It's the recognition that estimation error exists, that the future may not match the past, and that the cost of ruin is infinite while the cost of slightly suboptimal growth is finite.

## 14.3 VaR and CVaR for Options Portfolios

**Value at Risk (VaR)** answers the question: "What is the maximum loss I should expect on 95% of trading days?"

We compute VaR three ways:

**Parametric VaR:** Assumes returns are normally distributed. Quick to compute but underestimates tail risk for options portfolios because options returns are not normally distributed.

**Historical VaR:** Uses the actual empirical distribution of past returns. More accurate for non-normal distributions. Our 95% daily VaR is approximately $8,500 — meaning on 95% of days, the portfolio loses less than $8,500.

**Monte Carlo VaR:** Simulates thousands of return scenarios using the estimated return distribution. Most accurate but most computationally intensive. Produces VaR estimates similar to historical but with confidence intervals.

**Why CVaR matters more than VaR for premium sellers:**

VaR tells you the threshold of the worst 5%. CVaR (Conditional Value at Risk, also called Expected Shortfall) tells you the *average* loss in the worst 5%. For our portfolio:

- **95% VaR:** -$8,500
- **95% CVaR:** -$18,200

The CVaR is more than twice the VaR, revealing the fat-tailed nature of options returns. When losses exceed the 95% threshold, they tend to be *much* larger than the threshold. This is the fundamental risk of premium selling — the losses are not just frequent enough to worry about, they're concentrated in the tail and larger than normal-distribution models predict.

This is why our risk limits are set conservatively. A VaR-based risk limit of "2% daily VaR" might seem cautious, but the CVaR tells you that on the days when VaR is breached, the actual loss averages 4.3% — more than double. The regime engine and quant signals exist specifically to reduce the frequency and severity of these tail events.

## 14.4 Stress Testing Against Historical Crises

We run the complete portfolio through three historical stress scenarios:

**Scenario 1: COVID Crash (February 19 – March 23, 2020)**
- SPX decline: -34% over 23 trading days
- Without regime engine: Portfolio loss of approximately -$180K
- With regime engine: Portfolio loss of approximately -$12K (shifted to BEAR on Feb 24, missing most of the decline)
- Reduction: 93%

**Scenario 2: 2022 Bear Market (January – October 2022)**
- SPX decline: -25% over 9 months
- Without regime engine: Cumulative losses of approximately -$95K during BEAR/CAUTION periods
- With regime engine: Cumulative gains of approximately +$145K (traded during NEUTRAL windows, avoided BEAR periods)
- The regime engine turned a painful year into the lowest-return but still profitable year

**Scenario 3: Hypothetical Flash Crash**
- SPX drops 5% intraday with no warning (score was NEUTRAL the day before)
- Regime engine: No protection (the crash occurs during the trading day)
- Stop losses: Trigger at 2× credit, limiting loss to ~$10K per IC3 contract position
- Total portfolio loss: approximately -$65K
- Recovery: Full capital recovered within 15 trading days of normal operation

The flash crash scenario reveals the limitation of the regime engine: it operates on T-1 data and cannot protect against intraday events. This is why individual position stop-losses exist as the last line of defense. The regime engine handles multi-day risk. Stops handle intraday risk.

## 14.5 The Margin Utilization Framework

Our system deploys an average of $51,000 in margin against a $200,000 risk budget — approximately 26% utilization. The remaining 74% ($149,000) sits in liquid instruments (SGOV or money market) earning risk-free yield.

This might seem wasteful. It's not. Here's why:

**Reason 1: Drawdown buffer.** When a losing trade occurs, the margin on that position is consumed by the loss. If all $200K were deployed, a -$25K loss would force liquidation of other positions to meet margin requirements — crystallizing losses that might have recovered. With 74% reserve, the same loss is absorbed without disturbing other positions.

**Reason 2: Opportunity deployment.** After a market dislocation (a crash that triggers BEAR mode followed by recovery), options premiums are significantly higher than normal. The reserve capital allows us to deploy aggressively during these rich-premium periods. This is the strategic equivalent of keeping dry powder.

**Reason 3: Margin call prevention.** Brokers can increase margin requirements during high-volatility events. A position that requires $3,300 in margin during normal conditions might require $5,000 during a VIX spike. Without reserve, this increased requirement could force unwanted liquidation.

**Reason 4: SGOV yield.** At a 5% risk-free rate, $149K in SGOV generates approximately $7,450 per year. This is free income that partially offsets the opportunity cost of undeployed capital.

## 14.6 Compounding and Capital Preservation

The most counterintuitive lesson in risk management is that **reducing losses matters more than increasing gains** for long-term wealth accumulation.

Consider two strategies over 10 years:

**Strategy A:** Average annual return 50%, maximum drawdown 40%
**Strategy B:** Average annual return 35%, maximum drawdown 15%

Intuitively, Strategy A seems better — it makes more money. But the compounding math tells a different story:

After a 40% drawdown, you need a 67% gain just to get back to breakeven. That 67% gain consumes more than a year of Strategy A's average return. During the recovery period, Strategy B — which suffered only a 15% drawdown and needs only an 18% gain to recover — is already compounding forward from a higher base.

Over 10 years with one major drawdown:
- Strategy A terminal wealth: $700K × (1.50)^9 × (0.60) = roughly $23M
- Strategy B terminal wealth: $700K × (1.35)^9 × (0.85) = roughly $14M

Strategy A still wins in this simplified example. But add a second drawdown event:
- Strategy A: $700K × (1.50)^8 × (0.60)^2 = roughly $9.3M
- Strategy B: $700K × (1.35)^8 × (0.85)^2 = roughly $8.9M

Now they're almost equal. With three drawdown events (realistic over 10 years), Strategy B wins decisively. Each drawdown punishes Strategy A disproportionately because it draws down from a higher absolute level.

This is why our regime-filtered strategy ($397K/year, -$87K max DD) outperforms the unfiltered strategy ($466K/year, -$78K single-day but larger cumulative DD) over an 11-year horizon when compounding is included. The regime engine's 15% return reduction is more than compensated by the 68% drawdown reduction, because smaller drawdowns preserve more of the compounding base.

The practical rule: **never risk more than you can recover from in a reasonable time.** For our system, "reasonable" means less than 30 trading days to recover from the worst-case loss. At $1,121/day average daily P&L, a $87K max drawdown requires approximately 78 trading days to recover — longer than we'd like, but manageable. A $180K unfiltered drawdown would require 160 days — more than 6 months of perfect execution just to get back to even.
