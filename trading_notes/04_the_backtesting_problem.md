# Chapter 4: The Backtesting Problem (and How to Solve It)

## 4.1 Why Most Backtests Lie

Here is an uncomfortable truth: the majority of backtests that traders use to justify their strategies are misleading. Not because the traders are dishonest, but because backtesting options strategies correctly is genuinely hard, and the shortcuts people take introduce systematic biases that almost always make results look better than reality.

**Survivorship bias.** If you test a strategy on "the S&P 500 companies," you're testing on companies that survived. The ones that went bankrupt, were delisted, or were acquired are not in your dataset. This matters less for SPX index options (the index itself always survives), but it matters enormously for single-stock strategies.

**Look-ahead bias.** This is the most insidious form of cheating, and it's often unintentional. Look-ahead bias occurs when your strategy uses information that wouldn't have been available at the time of the trade. Examples: using the closing VIX to decide whether to enter a trade at the open. Using tomorrow's realized volatility to set today's delta. Using an SMA50 calculated with today's close to make a decision at today's open. Our system is designed to use only T-1 (yesterday's) data for all signals, precisely to eliminate this bias.

**Overfitting.** This deserves its own section, but the short version: if you test enough parameter combinations, you will find one that produces spectacular historical returns. This parameter combination almost certainly doesn't work going forward because it's fit to the noise in the historical data, not to any genuine pattern. We found this the hard way — a parameter sweep that produced a Sharpe ratio of 4.2 in-sample collapsed to 1.1 out-of-sample.

**Unrealistic execution assumptions.** Many backtests assume you can execute at the theoretical Black-Scholes price, which ignores the bid-ask spread. Others ignore commissions. Some assume you can always get a fill, even in low-liquidity conditions. Our backtester models $0.10/leg slippage and $0.65/leg commissions on every trade — not because these are exact, but because they're realistic enough to prevent the backtest from lying about profitability.

**Using close prices for intraday decisions.** SPX options can move 20-30% of their value in the last hour of trading. A backtest that uses closing prices to determine entry and exit may not reflect what would actually happen if you tried to execute at, say, 9:35 AM. Our system uses opening-period execution assumptions for entries and models intraday exit possibilities based on price movement during the day.

## 4.2 Building a Realistic Options Backtester

Building a credible options backtester for SPX required solving several interconnected problems:

**Problem 1: We don't have historical options prices.** Full historical options chain data for SPX (every strike, every expiration, every day) would cost tens of thousands of dollars and would require terabytes of storage. Instead, we use Black-Scholes pricing with a critical enhancement: **dynamic implied volatility derived from VIX.**

The VIX index represents the market's expectation of 30-day implied volatility for SPX options. By using the daily VIX level as the implied volatility input to Black-Scholes, we get option prices that reflect the actual market conditions on each day. When VIX is 15, our options are priced cheaply. When VIX is 35, our options are priced richly. This captures the most important variation in options pricing without requiring the full historical chain.

Is this perfect? No. Black-Scholes underprices far out-of-the-money options (the "volatility smile" effect), and VIX represents a 30-day average that may not match the specific DTE we're trading. But for 10-delta options at 1-3 DTE, the approximation is good enough to produce results that are directionally correct and roughly magnitude-correct.

**Problem 2: Modeling the holding period.** A 1DTE iron condor entered on Monday morning might exit in three ways: it expires at end of day Tuesday, it hits the 50% profit target during the day, or it hits the stop loss. Modeling this requires simulating the daily price path and checking at each point whether the P&L has crossed a threshold.

We use the daily open and close, with the high and low of the day, to estimate whether the P&L would have crossed the profit target or stop loss during the trading session. The high and low provide the range of prices the position would have experienced.

**Problem 3: Commissions and slippage must be applied consistently.** Every entry costs $0.65/leg × 4 legs + $0.10/leg × 4 legs = $3.00 per contract. Every exit costs the same. Round-trip: $6.00 per contract per trade. This is applied to every trade in the backtest, no exceptions.

**Problem 4: Position sizing and margin.** Each iron condor requires margin equal to the wing width minus the credit. For a $30 wide iron condor with a $5.00 credit, that's $25 × 100 = $2,500 in margin per contract. At 10 contracts, that's $25,000. The backtester tracks margin utilization and does not allow trades that would exceed the allocated risk budget.

## 4.3 Eleven Years of Data (2015–2025)

The backtest period of 2015–2025 was chosen because it encompasses a wide range of market conditions:

**2015–2017: Low volatility bull market.** VIX spent extended periods below 15. SPX rose steadily. This environment is favorable for premium sellers — wide iron condors collected premium with few challenges. It's also the environment most likely to breed overconfidence.

**February 2018: The Volmageddon event.** VIX spiked from 13 to 50 in a single day, destroying several volatility-selling funds. SPX dropped 10% in two weeks. This is the kind of tail event that any robust strategy must survive. Our IC3 strategy had its worst year in 2018, but still made $7,708 per contract.

**2019: Recovery and renewed optimism.** SPX recovered to new highs. VIX settled back to the 12-18 range. Smooth sailing for iron condors.

**March 2020: COVID crash.** SPX dropped 34% in 23 trading days. VIX hit 82 — the highest level since 2008. This is the ultimate stress test for any short premium strategy. Our regime engine, had it been running, would have shifted to BEAR mode within the first few days and parked capital in SGOV for the remainder of the crash.

**2020–2021: Post-COVID recovery and meme stock era.** Massive stimulus, zero interest rates, and unprecedented retail participation drove SPX to new highs. VIX remained elevated by historical standards (15-25 range), creating rich premium for sellers.

**2022: Bear market and rate shock.** SPX dropped 25% as the Federal Reserve aggressively raised interest rates. VIX spent much of the year above 25. This was the second major stress test — a sustained bear market rather than a sharp crash. Our regime engine would have been in CAUTION or BEAR mode for significant portions of the year.

**2023–2024: Bull market recovery.** SPX recovered to new all-time highs. VIX settled to the 12-18 range. Another period favorable for premium sellers.

**2025: Strong start with elevated volatility.** SPX continued higher but with periodic VIX spikes. Our best year by raw P&L — $901K for the unfiltered strategy — driven by richer premiums.

This 11-year period includes two bear markets, two major volatility events, three distinct bull runs, and the full spectrum of VIX environments from 9 to 82. Any strategy that survives this period profitably has been tested against genuinely adverse conditions.

## 4.4 Walk-Forward Validation

The simplest form of backtesting is "in-sample testing": optimize your parameters on all available data and report the results. This is also the most dangerous form, because it virtually guarantees overfitting.

Walk-forward validation solves this by splitting the data into sequential segments:

1. **Train on 2015–2017** (3 years). Find optimal parameters.
2. **Test on 2018** (1 year). Record out-of-sample performance.
3. **Train on 2015–2018** (4 years). Re-optimize parameters.
4. **Test on 2019** (1 year). Record out-of-sample performance.
5. Continue expanding the training window and testing on the next year.

The key principle: **you never test on data that was used for optimization.** The out-of-sample results are the only results that matter.

Our QuantEngine implements this as an expanding-window walk-forward with configurable train/test ratios. The standard configuration uses a minimum of 3 years for training and 1 year for testing, with the training window expanding as more data becomes available.

The critical metric is the **OOS/IS ratio** — the ratio of out-of-sample Sharpe to in-sample Sharpe. A ratio of 0.8 or higher indicates that the strategy is robust and not significantly overfit. A ratio below 0.5 indicates severe overfitting. Our IC3 strategy achieves an OOS/IS ratio of approximately 0.85, indicating strong out-of-sample performance.

Our Phase 1 OOS validation system enforces strict gating criteria: a strategy must achieve an OOS Sharpe above 1.0, an OOS/IS ratio above 0.7, and a positive return in every out-of-sample segment to pass validation. Strategy A (our regime-filtered approach) passed with 87.5% confidence. Strategy B (an overfit alternative) failed all criteria. The gating system works.

## 4.5 Monte Carlo Simulation for Confidence

Even with walk-forward validation, a single historical path is still just one realization of many possible outcomes. Monte Carlo simulation addresses this by generating thousands of synthetic return series that share the statistical properties of the historical returns.

The process:
1. Calculate daily returns from the backtest
2. Resample these returns (with replacement) to create 10,000 synthetic 11-year paths
3. Calculate the performance metrics for each path
4. Report the distribution of outcomes, not just the single historical outcome

For our IC3 strategy, Monte Carlo analysis shows:
- **Median annual P&L**: $275K (slightly below the historical $282K, as expected)
- **5th percentile**: $195K (the "bad luck" scenario — still highly profitable)
- **95th percentile**: $370K (the "good luck" scenario)
- **Probability of a losing year**: <1%
- **Maximum drawdown, 95th percentile**: $45K (worse than the median $28K but still manageable)

The Monte Carlo results provide confidence that the strategy's performance is not dependent on the specific sequence of market events in 2015–2025. Even under unfavorable random reorderings, the strategy remains profitable.

## 4.6 The Robustness Lab

Beyond walk-forward validation and Monte Carlo, we apply several additional tests to ensure strategy robustness:

**Parameter sensitivity analysis.** How much do results change when you shift a parameter by 10%? If changing delta from 10 to 11 causes a 30% drop in Sharpe, the strategy is brittle and probably overfit to the exact delta setting. Our IC3 strategy shows stable performance across delta values from 8 to 14 — the Sharpe ratio varies from 2.3 to 2.6 across this range, indicating genuine robustness rather than parameter sensitivity.

**Regime-specific analysis.** The strategy must be profitable (or at least not catastrophically unprofitable) in each of the four regime types independently. IC3 is profitable in all four regimes because it's non-directional. The regime engine improves risk-adjusted returns by reducing size in adverse conditions, but the base strategy doesn't depend on favorable conditions to survive.

**Autocorrelation testing (Ljung-Box).** Daily returns from the strategy should not exhibit significant autocorrelation. If today's return predicts tomorrow's return, it suggests a structural issue — either a look-ahead bias or a dependency on market momentum that might not persist. Our strategies pass the Ljung-Box test at the 5% significance level.

**Transaction cost sensitivity.** What happens if slippage doubles from $0.10 to $0.20 per leg? This simulates adverse execution conditions — maybe liquidity deteriorates, or the 0DTE market becomes more competitive. At doubled slippage, IC3 annual P&L drops from $282K to approximately $259K — a meaningful reduction, but the strategy remains highly profitable. This confirms that the edge is real and not dependent on favorable execution assumptions.

## 4.7 When to Trust a Backtest (and When to Walk Away)

After building and running thousands of backtests, here are the heuristics I use to evaluate whether results are trustworthy:

**Trust when:**
- The strategy is simple (few parameters, clear logic)
- Walk-forward OOS/IS ratio is above 0.7
- Performance is stable across different market regimes
- Results degrade gracefully when parameters shift ±10%
- Transaction cost sensitivity doesn't destroy the edge
- The strategy makes economic sense (there's a reason for the edge)
- The backtest period includes multiple adverse environments

**Walk away when:**
- The strategy has more than 5-6 tunable parameters
- Performance collapses when you shift to out-of-sample periods
- Results depend heavily on a specific parameter value
- The edge disappears with realistic transaction costs
- You can't articulate why the strategy should work
- The backtest period is entirely favorable (all bull market, for example)
- The Sharpe ratio is above 4.0 (almost certainly overfit)

The last point deserves emphasis. In my experience, any options strategy with an in-sample Sharpe ratio above 4.0 is overfit. Our best strategies have Sharpe ratios in the 2.5–3.6 range. These are excellent by any standard — the average hedge fund achieves a Sharpe of 0.5–1.0 — but they're plausible. A Sharpe of 6 or 8, which I've seen claimed in online forums, is a red flag for overfitting, look-ahead bias, or both.

The goal of backtesting is not to find the perfect strategy. It's to find a strategy that is good enough, robust enough, and well-understood enough to trade with real money. Perfection in backtesting is the enemy of profitability in live trading.
