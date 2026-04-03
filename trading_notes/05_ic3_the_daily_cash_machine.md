# Chapter 5: IC3 — The Daily Cash Machine (1DTE Iron Condors)

## 5.1 Strategy Specification

The IC3 strategy is the cornerstone of our daily income portfolio. Here is the complete specification:

| Parameter | Value |
|-----------|-------|
| Underlying | SPX |
| Structure | Iron Condor (put spread + call spread) |
| DTE | 1 (enter today, expires tomorrow) |
| Short Strike Delta | 10 (both sides) |
| Wing Width | $30 |
| Contracts | 10 (base size, regime-adjusted) |
| Profit Target | 50% of credit received |
| Stop Loss | 2× credit received |
| Entry Time | 9:35 AM ET |
| Margin per Contract | ~$3,300 |
| Total Margin (10 cts) | ~$33,000 |

The strategy enters a new iron condor every trading day at 9:35 AM — five minutes after the open, allowing the initial volatility to settle. The short puts are placed at the 10-delta level below the current SPX price, and the short calls at the 10-delta level above. Both sides have $30 wide wings for protection.

Example on a day when SPX is at 5,500 and VIX is at 18:
- **Sell 5430 put, buy 5400 put** (put spread, $30 wide)
- **Sell 5570 call, buy 5600 call** (call spread, $30 wide)
- **Credit received: ~$5.00** ($500 per contract, $5,000 for 10 contracts)
- **Maximum loss: ~$25.00** ($2,500 per contract, $25,000 for 10 contracts)

The position profits if SPX stays between 5,430 and 5,570 through expiration — a range of 140 points, or roughly ±1.3% from the entry price. Historical analysis shows that SPX stays within this range on approximately 90% of trading days.

## 5.2 Why 1DTE Works

The 1DTE iron condor is counterintuitive. Conventional options education teaches that selling very short-dated options is risky because of gamma. And it's true — gamma risk at 1DTE is extreme near the strikes. But the key insight is that **at 10-delta, you're far enough from the money that gamma works in your favor most of the time.**

Here's the dynamic at play:

**Theta acceleration at 1DTE.** An option that is 10-delta with 1 day to expiration is decaying rapidly. The theta (time decay) per day as a percentage of the option's value is at its maximum. For a 10-delta SPX put with 1DTE, the option might be worth $2.50 and decay by $1.50 overnight if SPX doesn't move. That's 60% decay in one day.

**The daily reset advantage.** When a 1DTE trade expires (or is closed at profit target), yesterday's risk is gone. Tomorrow's trade starts fresh with new strikes placed 10-delta away from the current price. If SPX dropped 1% today, tomorrow's short put is placed 1% lower — automatically adjusting to the new market level. Compare this to a 14DTE position: if SPX drops 1% on day 1, you still have 13 days of exposure at strikes that were set before the drop.

**Compounding frequency.** Ten contracts earning $500 per day (the approximate daily credit after costs) for 252 trading days is $126,000. But because many trades exit early at the 50% profit target — returning margin for redeployment — the actual capital turnover is higher. The same $33,000 in margin can support multiple overlapping 1DTE positions when one exits early and another is entered.

**No weekend exposure.** A 1DTE iron condor entered on Monday morning expires Tuesday. There's no weekend gap risk. The only multi-day holds occur when entering on Thursday (expiring Friday) or when a position entered on Friday carries to Monday — which we avoid by not entering new positions on Friday afternoons when Monday expiration is the next available.

## 5.3 Performance Deep Dive

Across 2,003 trading days from 2015 to 2025, the IC3 strategy at 10 contracts produced:

| Metric | Value |
|--------|-------|
| Total P&L | $2,824,500 |
| Annual P&L | $282,450 |
| Daily Average P&L | $1,121 |
| Win Rate | 90% |
| Positive Days | 91% (1,830 of 2,003) |
| Sharpe Ratio | 2.54 |
| ROI on $200K Risk Budget | 141% annually |
| ROI on $33K Margin | 856% annually |

Some notes on these numbers:

The **91% positive days** is slightly higher than the **90% win rate** because some "losing" trades (where the iron condor breaches one side) still produce a small net gain when the other side's credit offsets the loss. A trade where the put side is breached for a $1,500 loss but the call side expires for a $2,500 gain is a "loss" on the put side but a net winner on the trade.

The **Sharpe ratio of 2.54** is exceptional by any standard. Hedge funds consider a Sharpe of 1.0 to be good and 2.0 to be excellent. The high Sharpe comes from the combination of high win rate and moderate volatility of daily returns. Most days produce similar-sized gains; losses are infrequent but larger.

The **ROI on margin** (856%) is technically accurate but misleading as a standalone number. It reflects the leverage inherent in options — $33K of margin controls $300K of notional SPX exposure. The more meaningful number is ROI on the total risk budget ($200K), which gives 141%. This represents the return on capital actually allocated to the strategy.

## 5.4 The Exit Framework

Understanding how trades exit is as important as understanding how they enter. IC3 trades exit through one of three mechanisms:

**Expiration (55% of trades).** The majority of trades are held to expiration. The iron condor expires worthless (all four legs out of the money), and the full credit is retained. These are the cleanest, easiest trades — enter, wait, collect.

**Profit target at 50% (39% of trades).** When the position's mark-to-market value reaches 50% of the initial credit, it's closed early. If the credit was $5.00, the position is closed when the iron condor can be bought back for $2.50. This typically happens when SPX moves sideways or when implied volatility drops during the holding period.

The 50% profit target serves two purposes: it locks in profit before gamma risk increases as expiration approaches, and it frees up margin for the next trade. A position exited at 50% profit after 0.8 days effectively doubles the capital turnover rate.

**Stop loss (6% of trades).** When the position's mark-to-market loss exceeds 2× the initial credit, it's closed. If the credit was $5.00, the stop triggers when the position costs $15.00 to close (a net loss of $10.00 per contract). This limits the maximum realized loss to approximately $10 per contract ($1,000) instead of the theoretical maximum of $25 per contract ($2,500).

The stop-loss exit rate of 6% is critical to the strategy's profitability. Without stops, the average loss on losing trades would be approximately $2,200 per contract (many positions would reach maximum loss). With the 2× stop, average loss is approximately $1,800 per contract. The stop doesn't prevent all losses, but it prevents the worst ones.

Average hold time across all exits: **1.9 days.** This reflects the fact that many trades exit before expiration via profit target.

## 5.5 Year-by-Year Analysis

The IC3 strategy was profitable in every single year from 2015 to 2025:

| Year | P&L per Contract | Win Rate | Notable Events |
|------|-----------------|----------|----------------|
| 2015 | $7,708 | 88% | China devaluation, Aug flash crash |
| 2016 | $14,290 | 89% | Brexit, US election |
| 2017 | $26,805 | 93% | Historically low volatility |
| 2018 | $12,340 | 86% | Volmageddon, Oct-Dec selloff |
| 2019 | $22,150 | 91% | Rate cuts, trade war volatility |
| 2020 | $35,720 | 88% | COVID crash + recovery |
| 2021 | $31,440 | 92% | Low vol, steady bull market |
| 2022 | $18,960 | 87% | Bear market, rate hikes |
| 2023 | $28,370 | 91% | AI rally, banking crisis |
| 2024 | $33,280 | 92% | Bull market continuation |
| 2025 | $59,000+ | 91% | Rich premiums, elevated VIX |

Several patterns are visible:

**2017 and 2021 were the easiest years** — low, stable VIX meant wide cushions and frequent profit target exits. Win rates above 92%.

**2018 was the hardest year.** The February VIX spike (Volmageddon) produced several consecutive losses, and the October–December selloff triggered additional stops. Even so, the strategy made $12,340 per contract — still a solid absolute return.

**2020 is the most instructive year.** The COVID crash in March produced the single worst day (-$6,349 per contract), but the strategy recovered quickly because 1DTE positions reset daily. The recovery period (April–December) was extremely profitable due to elevated VIX providing rich premiums. The net result was one of the best years overall.

**2025 stands out** with $59K per contract — the best year by far. This reflects the combination of elevated VIX (richer premiums) and a generally rising market (fewer breaches). Whether this level of performance persists is uncertain; it may reflect a temporarily favorable environment.

## 5.6 The Worst Days

Understanding the worst-case scenarios is more important than celebrating the best-case scenarios. Here are the characteristics of the worst days:

**Single-day maximum loss: -$6,349 per contract.** This occurred during the COVID crash when SPX gapped down over 4% at the open, blowing through the short put strike before any exit was possible. At 10 contracts, this represents a single-day loss of $63,490 — roughly 9% of the total account.

**What causes maximum loss days:**
1. **Overnight gaps.** SPX opens significantly higher or lower than the previous close, immediately putting the position in or near maximum loss territory.
2. **Intraday crashes.** A rapid, large move during the trading day that moves too fast for the stop loss to execute at the intended level.
3. **VIX spikes.** A simultaneous increase in implied volatility makes the position more expensive to close, even if SPX hasn't moved much.

**Frequency of large loss days:**
- Loss > $3,000/contract: ~2% of trading days (roughly 4 per year)
- Loss > $5,000/contract: ~0.3% of trading days (roughly 1 per year)
- Loss > $6,000/contract: ~0.1% of trading days (roughly 1 every 4 years)

These tail losses are why position sizing and regime filtering are essential. A $6,349 loss at 10 contracts is painful but survivable on a $700K account (0.9% of total). The same loss at 50 contracts — which is what an over-aggressive trader might deploy with the available margin — would be $317,450 (45% of the account). This is why we limit SPX exposure to $150K and use only 26% of the risk budget.

## 5.7 Why This Strategy Trades in All Regimes

Most of our strategies require a trend filter — typically SPX above the 50-day moving average — to avoid trading in bearish conditions. IC3 is the exception.

Why? Because at 10-delta and 1DTE, the strategy is fundamentally non-directional. The short put is 70+ points below the current price, and the short call is 70+ points above. SPX can drop 1% or rally 1% and the position is still profitable. The only losing scenario is a move of approximately 1.3% or more in a single day.

Such large daily moves are slightly more common in bearish environments (CAUTION and BEAR regimes), but not dramatically so. Our data shows:

| Regime | Daily Move > 1.3% | IC3 Win Rate |
|--------|-------------------|--------------|
| BULL | 5% of days | 93% |
| NEUTRAL | 8% of days | 91% |
| CAUTION | 14% of days | 86% |
| BEAR | 22% of days | 82% |

Even in BEAR regime, the win rate remains above 80%. The strategy is profitable in all environments — but the regime engine improves risk-adjusted returns by reducing position size during CAUTION (2 contracts instead of 10) and eliminating positions during BEAR (0 contracts).

The regime engine doesn't save IC3 from losing. It reduces the *amount* lost during adverse conditions, which preserves capital for compounding during favorable conditions. Over an 11-year horizon, this capital preservation effect is the single biggest driver of risk-adjusted outperformance.
