# Chapter 13: Quant Signals as a Second Layer of Defense

## 13.1 GARCH Volatility Forecasting

The regime engine operates on daily signals — SMA trends, VIX levels, term structure. These are effective for identifying multi-day and multi-week regimes, but they miss intraday and overnight volatility dynamics. The quant signal layer fills this gap.

**GARCH (Generalized Autoregressive Conditional Heteroskedasticity)** is a statistical model that captures a fundamental property of financial markets: volatility clusters. High-volatility days tend to follow high-volatility days. Low-volatility days tend to follow low-volatility days. The GARCH model formalizes this by modeling today's variance as a function of yesterday's variance and yesterday's squared return.

The standard GARCH(1,1) model:

```
σ²_t = ω + α × ε²_{t-1} + β × σ²_{t-1}
```

Where:
- σ²_t is today's conditional variance (the forecast)
- ω is the long-run average variance
- α captures the impact of yesterday's return shock
- β captures the persistence of yesterday's variance
- α + β close to 1 means volatility is highly persistent

Our implementation fits a GARCH(1,1) model on a rolling window of SPX daily returns and produces two outputs:

1. **garch_vol_1d:** Tomorrow's forecasted volatility (annualized)
2. **vol_expanding:** Boolean flag — true when the GARCH forecast exceeds the 20-day realized volatility by more than 20%

When vol_expanding is true, the model is saying: "Based on recent return patterns, tomorrow is likely to be more volatile than the recent average." This is a danger signal — elevated volatility means wider daily SPX ranges, which means higher probability of breaching iron condor strikes.

We also use **EGARCH** (Exponential GARCH) which captures the asymmetric response of volatility to positive and negative returns. Markets become more volatile after drops than after rallies of the same magnitude — the "leverage effect." EGARCH captures this asymmetry, producing more accurate forecasts during declining markets.

## 13.2 Extreme Value Theory and Dynamic Stops

Standard risk measures (VaR, standard deviation) describe the center of the return distribution. But for premium sellers, the center doesn't matter — what matters is the tail. How bad can the worst days get?

**Extreme Value Theory (EVT)** addresses this directly by modeling only the extreme observations — the losses beyond the 90th percentile. The approach:

1. **Identify extreme losses.** From the historical return series, extract all daily losses that exceed the 90th percentile threshold.
2. **Fit the Generalized Pareto Distribution (GPD)** to these extreme losses. The GPD has two key parameters: scale (σ) and shape (ξ, pronounced "xi").
3. **The shape parameter ξ determines the tail behavior:**
   - ξ > 0: Fat tails (Pareto-like). Extreme losses can be arbitrarily large. This is the typical finding for financial returns.
   - ξ = 0: Exponential tails. The normal distribution assumption.
   - ξ < 0: Bounded tails. There's a maximum possible loss. Rare in financial markets.

Our analysis of SPX daily returns consistently finds ξ between 0.15 and 0.35 — confirming fat tails. This means the probability of extreme events is significantly higher than the normal distribution suggests. A "3-sigma" event under the normal distribution should occur once every 741 days (roughly every 3 years). Under our GPD fit, it occurs once every 180 days (roughly twice per year).

**Dynamic stop multiplier.** The shape parameter drives a dynamic stop-loss multiplier:

| ξ Range | Tail Character | Stop Multiplier | Interpretation |
|---------|---------------|-----------------|----------------|
| < 0.1 | Thin tails | 1.5× credit | Tight stops — tails are well-behaved |
| 0.1–0.25 | Moderate tails | 2.0× credit | Standard stops |
| > 0.25 | Fat tails | 2.5× credit | Wide stops — tight stops get whipsawed |

The logic: when tails are fat (high ξ), using tight stops causes excessive stop-outs because the market frequently makes large intraday moves that then reverse. Wider stops allow the position to survive these moves. When tails are thin (low ξ), tight stops are appropriate because large moves that don't reverse are truly directional.

## 13.3 VPIN: Detecting Toxic Order Flow

**VPIN (Volume-Synchronized Probability of Informed Trading)** was developed by Easley, López de Prado, and O'Hara in 2012 as a real-time estimate of information asymmetry in order flow.

The core idea: when informed traders (those with knowledge of upcoming price-moving events) are active in the market, the order flow becomes "toxic" — it flows predominantly in one direction because informed traders are all taking the same side. This toxicity is detectable before the price move occurs, because the orders arrive before the information becomes public.

Our implementation uses daily OHLCV data as a proxy for intraday order flow:

1. **Estimate buy/sell volume** from daily bar data using the bid-ask bounce method: if the close is above the midpoint of the day's range, classify more volume as buy-initiated; if below, classify more as sell-initiated.
2. **Calculate VPIN** as the absolute imbalance between buy and sell volume, normalized by total volume, over a 20-bar window.
3. **Detect toxic flow** when VPIN exceeds 0.7 (indicating >70% of volume is one-sided) AND total volume is above average (indicating the imbalance is not just noise from low-volume days).

When toxic flow is detected:
- **If combined with vol_expanding (from GARCH):** Strong danger signal — informed traders are active and volatility is rising
- **If combined with VIX backwardation:** Extremely strong danger signal — institutional hedging is intensifying alongside informed selling

Our VPIN proxy is less precise than true tick-by-tick VPIN (which requires intraday data we don't have), but it captures the same directional signal: periods when order flow is unusually one-sided tend to precede large price moves.

## 13.4 Bayesian Regime Switching

While our heuristic 4-tier regime engine uses fixed thresholds and rules, the Bayesian regime switching model provides a probabilistic complement — estimating the probability of being in each regime rather than making a hard classification.

The model is a **2-state Hidden Markov Model (HMM)** fit to SPX daily returns:

**State 1 (Low Volatility):** Mean return ≈ +0.05%/day, standard deviation ≈ 0.7%
**State 2 (High Volatility):** Mean return ≈ -0.10%/day, standard deviation ≈ 1.8%

The HMM estimates the transition probabilities between states and, given observed returns, computes the posterior probability of being in each state using the Hamilton filter.

**How it complements the heuristic engine:**
- The heuristic engine uses market observables (SMA, VIX, etc.) — external signals
- The HMM uses return dynamics — internal signals

When both agree (heuristic says BEAR and HMM says >80% probability of high-vol state), confidence in the regime classification is very high. When they disagree (heuristic says NEUTRAL but HMM says 60% probability of high-vol state), it's a warning sign that deserves attention.

We also implement **online Bayesian changepoint detection** (Adams and MacKay, 2007), which detects abrupt changes in the statistical properties of the return series. This method doesn't assume a fixed number of regimes — it can detect novel regimes that don't match historical patterns.

## 13.5 The Danger Score

All four quant signals are combined into a single composite **danger score** ranging from 0 to 10:

| Component | Max Points | Trigger |
|-----------|-----------|---------|
| GARCH vol expanding | 3 | Vol forecast > 1.2× realized vol |
| High rv_percentile | 2 | Realized vol in top 20% of 1-year range |
| Toxic VPIN | 2 | VPIN > 0.7 with above-average volume |
| HMM high-vol regime | 2 | >70% probability of high-vol state |
| EVT fat tails | 1 | Shape parameter ξ > 0.3 |

**Action thresholds:**

| Danger Score | Action |
|-------------|--------|
| 0–4 | FULL_SIZE — proceed with tier-allocated sizing |
| 5–6 | HALF_SIZE — reduce position size by 50% |
| ≥ 7 | SKIP — do not enter new trades today |

The danger score operates independently of the regime tier. It's possible to be in NEUTRAL regime (score +3) but with a danger score of 7 — meaning the market conditions look broadly favorable by trend and VIX metrics, but the quant signals are detecting something the heuristic engine doesn't see. In this case, the danger score overrides: no new trades.

This happens rarely — perhaps 10-15 days per year. But these are precisely the days when an unexpected event (earnings surprise, geopolitical shock, flash crash) is most likely to cause a large move. The quant signals often pick up on informed trading activity and volatility clustering that precedes these events.

## 13.6 What Quant Signals Add to Performance

The honest assessment is that quant signals provide **marginal improvement to average performance but significant improvement to worst-case scenarios.**

| Metric | Regime Engine Only | Regime + Quant Signals |
|--------|-------------------|----------------------|
| Annual P&L | $403K | $397K |
| Sharpe | 3.07 | 3.13 |
| Max Drawdown | -$95K | -$87K |
| Worst Day | -$52K | -$43K |
| Days with >$20K Loss | 8/year | 5/year |

The annual P&L actually decreases slightly (-$6K) because the danger score blocks some trades that would have been winners. This is the cost of additional caution. But the Sharpe ratio improves because the blocked trades include a disproportionate share of large losers.

The most important improvement is in the tail: **the worst day improves from -$52K to -$43K, and the frequency of large loss days drops by 37%.** For a premium-selling strategy, this tail risk reduction is more valuable than the marginal P&L difference suggests, because large losses destroy the compounding base that drives long-term wealth.

The quant signals are the fine-tuning layer. The regime engine is the heavy lifter (providing the bulk of the risk reduction). The quant signals catch the edge cases that the regime engine misses — the days when the market looks fine by trend and VIX metrics but the underlying dynamics (order flow, volatility clustering, tail risk) are deteriorating.

Together, the regime engine and quant signals form a defense-in-depth: two independent systems that must both approve a trade before capital is deployed. This redundancy is not paranoia — it's sound engineering. The failure modes of the two systems are different, so when one fails to detect danger, the other has an independent chance of catching it.
