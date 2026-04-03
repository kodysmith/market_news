# Chapter 18: Backtest Results and Strategy Tweaks

## 18.1 The First Backtest

We backtested all five strategies from this book on SPY using 5-minute bars over 42 trading days (February 3 - April 2, 2026). This period was a bearish environment — SPY fell from $695 to $636, a 9% decline. The backtest used $100K starting capital with 1% risk per trade ($1,000 max risk), realistic slippage ($0.02/share), and commissions ($0.005/share).

**The results:**

| Strategy | Trades | Win % | Avg Win | Avg Loss | PF | $/Trade | Annual Est |
|----------|--------|-------|---------|----------|-----|---------|------------|
| **ORB** | **8** | **87.5%** | **$1,387** | **-$229** | **42.4** | **$1,185** | **$56,864** |
| **VWAP Bounce** | **16** | **50.0%** | **$1,930** | **-$1,088** | **1.8** | **$421** | **$40,413** |
| Trend Day | 14 | 35.7% | $2,263 | -$1,069 | 1.2 | $121 | $10,192 |
| Gap Fade | 13 | 38.5% | $1,815 | -$1,053 | 1.1 | $50 | $3,909 |
| VWAP Reject | 21 | 23.8% | $1,901 | -$1,073 | 0.6 | -$365 | -$46,016 |
| **Combined** | **72** | **41.7%** | — | — | — | **$151** | **$65,363** |

**If we run only ORB + VWAP Bounce:** 24 trades, 62% win rate, $676/trade, $97,277 annualized. Less trading, more money.

## 18.2 The Market Context Problem

This backtest ran during a 9% SPY decline. This creates specific biases:

- **Long trades won 51% of the time** with an average of +$331/trade
- **Short trades won only 32% of the time** with -$19/trade average

This is counterintuitive — in a bearish market, shorts should do better. The explanation: our short strategies (VWAP Reject, short ORBs) were fighting counter-trend rallies within the decline. SPY doesn't drop 9% in a straight line — it drops, bounces, drops more, bounces again. The bounces trigger short entries that get stopped out before the next leg down.

**Key lesson:** Short-side strategies need different parameterization than long-side strategies, even on the same instrument. The market's structural long bias (stocks go up over time, short squeezes, buybacks) means shorting requires wider stops, fewer trades, and higher conviction.

## 18.3 What Worked: ORB

The Opening Range Breakout was the clear winner with 87.5% win rate and $1,185/trade. But honest analysis requires acknowledging the caveats:

**Why ORB worked so well:**
1. **Extreme selectivity.** Only 8 trades in 42 days — the volume filter and OR range filter rejected most setups. Only the cleanest breakouts triggered. This is the right behavior: the scanner is the strategy
2. **Bearish regime alignment.** 5 of 7 winners were short ORB breakdowns, aligned with the overall market decline. The strategy naturally adapted to the regime
3. **The single loss was tiny.** -$229 (only -0.23R) via time stop, not a full stop-loss hit. The ORB's structure (stop at opposite side of range) kept the risk defined

**Concerns:**
- **Small sample size.** 8 trades is not statistically significant. Need 100+ to validate
- **Regime dependency.** In a choppy or bullish market, the short ORB breakdowns would flip to long ORB breakouts. The strategy adapts, but the win rate may be lower
- **The profit factor of 42.4 is unrealistic long-term.** This will normalize to 1.5-2.5 over more trades

**Recommended tweaks:** None. The ORB implementation is conservative and selective. The right move is to gather more data, not to change parameters.

## 18.4 What Worked: VWAP Bounce

The VWAP Bounce produced solid results: 50% win rate with $1,930 average win vs $1,088 average loss — a 1.8:1 reward-to-risk ratio.

**Why it worked:**
- All 16 trades were long, aligned with buying dips in a market that still had intraday bounces
- The 2R target was hit cleanly on winning trades (all 8 winners hit take_profit)
- The stop just below VWAP ($0.20 buffer) was tight enough to limit losses but not so tight that normal noise triggered it

**The 50% win rate is exactly what we expected.** With a 1.8 profit factor, the math works: (0.50 × $1,930) - (0.50 × $1,088) = $421/trade expectancy. Over 96 trades/year, that's $40K. Consistent, reliable, no heroics.

**Failure pattern:** The 8 losses were all stop_loss exits where the stock bounced off VWAP briefly but then broke through. The VWAP "support" failed. This happens when the overall market is selling off and VWAP support gets overwhelmed by broader selling pressure.

**Recommended tweaks:**
- Add a market direction filter: only take VWAP bounces when SPY is above its own VWAP. This would have filtered out some of the losing trades on heavy sell days
- Consider a tighter time window: the best VWAP bounces happen between 10:00-11:30 AM. Afternoon bounces were less reliable

## 18.5 What Failed: VWAP Rejection

The VWAP Rejection was the worst performer: 23.8% win rate, -$365/trade, negative profit factor (0.6). It lost $7,669 over 42 days — enough to wipe out the gains from Gap Fade and Trend Day combined.

**Root cause analysis:**

**Problem 1: Too many triggers.** 21 trades in 42 days means it triggered every other day. Compare to ORB (8 trades) or VWAP Bounce (16). The VWAP rejection was too sensitive — it saw "rejections" where there were just normal oscillations around VWAP.

**Problem 2: Shorting into support.** In a declining market, stocks below VWAP aren't being "rejected" at VWAP — they're temporarily bouncing off an oversold condition. The bounce looks like a setup for a rejection, but it's actually the start of a mean-reversion rally. The short entry gets caught in the rally.

**Problem 3: The confirmation was too weak.** The current logic only requires that the stock has been below VWAP for 4 of the last 6 bars and that the current candle closes below VWAP after touching it. This is not enough confirmation. A single red candle at VWAP doesn't mean sellers have taken control.

**Proposed fixes:**
1. **Require 2+ consecutive closes below VWAP before looking for rejections** (not 4 of 6 — ALL of the last 6)
2. **Require the rejection candle to close in the bottom 25% of its range** (showing strong seller conviction, not just a doji near VWAP)
3. **Add volume confirmation:** the rejection candle must have above-average volume
4. **Market filter:** only take VWAP rejections when SPY is below its own 9 EMA on the 5-minute chart (confirmed downtrend, not just below VWAP)
5. **Limit to 1 rejection trade per day** (prevent overtrading)

**Alternative: kill the strategy entirely.** The long-side VWAP Bounce works. The short-side VWAP Rejection doesn't. This is consistent with the market's structural long bias. Rather than fixing a broken short strategy, it may be better to only trade VWAP from the long side and use ORB breakdowns for short exposure.

## 18.6 What Failed: Trend Day

The Trend Day strategy had potential (3 winners averaged $2,941 — nearly 3R each) but too many false positives (9 of 14 trades stopped out).

**Root cause analysis:**

**Problem 1: "All above VWAP" is not enough to confirm a trend day.** A stock can be above VWAP for 30 minutes and still be in a range. The criterion correctly identified that the stock was bullish, but it didn't verify that the stock was *trending* (sustained directional movement with no significant pullbacks).

**Problem 2: The 9 EMA pullback in a range = death.** In a ranging day, the stock oscillates around the 9 EMA. The pullback entry triggers, but the stock isn't trending — it's chopping. The "pullback" continues right through the stop.

**Problem 3: 9 of 14 trades stopped out means the stop was too tight relative to the noise.** The stop below the pullback low was often only $0.10-0.25, which is within normal 5-minute bar noise.

**Proposed fixes:**
1. **Add ORB breakout confirmation:** only look for trend day trades if the ORB already broke out in the same direction. ORB breakout + sustained VWAP position = much stronger trend confirmation
2. **Require EMA stacking:** 9 EMA > 20 EMA (or 9 EMA < 20 EMA for downtrends). This confirms the trend has been established for multiple bars
3. **Range expansion filter:** today's range so far must be > 1.5x the 14-day average range at this time of day. True trend days are volatile. If the range is normal, it's not a trend day
4. **Wider stops:** use 1.5x ATR below entry instead of just below the pullback candle low. This gives the trade room to breathe through normal pullback noise
5. **Maximum 1 trend day trade per day.** If the first pullback entry gets stopped, the day is not trending. Don't try again

## 18.7 What Was Marginal: Gap Fade

Gap Fade made $652 over 42 days — positive but barely. The wins ($1,815 avg) were large enough to offset the losses ($1,053 avg), but the 38.5% win rate means you endure many small losses between the big wins.

**Root cause analysis:**

**Problem 1: The gap threshold (0.5-2.5%) was too wide.** A 0.5% gap on SPY is $3.40. This is noise. Many 0.5% gaps are just normal overnight drift, not a meaningful gap to fade. By including these, we diluted the quality of the signal with noise.

**Problem 2: Short gap fades underperformed.** 4 short gap fades produced -$244/trade average. In a falling market, shorting a gap-up fade makes sense — but the gaps were too small to overcome slippage and the fade often didn't reach the target.

**Problem 3: The VWAP cross was the only confirmation.** A gap-up stock crossing below VWAP could mean the gap is fading, or it could mean the stock is just oscillating. Additional confirmation is needed.

**Proposed fixes:**
1. **Narrow gap range to 0.8-1.8%.** This filters out noise gaps (<0.8%) and extreme gaps (>1.8% which are more likely to be fundamentally driven and should NOT be faded)
2. **Require 3+ consecutive bars below VWAP** before entering the fade, not just 1 bar crossing below
3. **Volume decline on the fade bars:** the stock should be fading on declining volume (exhaustion), not on increasing volume (conviction selling)
4. **Tighter time window:** only look for gap fades between 10:00-10:30 AM. After 10:30, if the gap hasn't faded, it's probably not going to
5. **Skip on catalyst:** don't fade gaps with genuine Tier 1 catalysts (earnings, FDA). Only fade "noise" gaps and weak Tier 3-4 catalysts

## 18.8 The Long vs Short Asymmetry

The most important cross-strategy finding:

| Direction | Trades | Win Rate | Avg $/Trade | Total |
|-----------|--------|----------|-------------|-------|
| Long | 35 | 51% | +$331 | +$11,587 |
| Short | 37 | 32% | -$19 | -$693 |

Long trades made money. Short trades didn't. This is not just this backtest — this is a structural feature of equity markets:

1. **Stocks have a long-term upward drift.** Even in a 9% decline, individual days have rallies. Fighting the structural long bias requires higher conviction
2. **Short squeezes don't happen in reverse.** There's no "long squeeze." But short covering creates violent upward spikes that kill short positions
3. **Spreads widen on drops.** When SPY drops, spreads widen, slippage increases, and short entries fill worse. The friction tax is higher on the short side
4. **Psychology:** most retail traders are long-biased. When they sell, they create temporary dips that quickly recover. Short-side strategies compete against this buying pressure

**Recommendation for the system:**
- **Primary strategies: ORB (long and short) + VWAP Bounce (long only)**
- **Use ORB for short exposure** — it's the only strategy that was profitable on both sides
- **Kill or rebuild VWAP Reject** (short-side VWAP strategy)
- **Reduce short-side positions across all strategies to 50% of long-side size**

## 18.9 The R-Multiple Distribution

| R-Multiple | Trades | % |
|------------|--------|---|
| < -1R (full stop losses) | 41 | 57% |
| -1R to 0 (partial losses) | 1 | 1% |
| 0 to +1R (small wins) | 3 | 4% |
| +1R to +2R (target hits) | 21 | 29% |
| +2R+ (big wins) | 6 | 8% |

**The distribution is bimodal:** trades either hit the full stop (-1R) or hit the full target (+2R). Very few end up in between. This tells us:

1. **Stops are correctly sized** — when wrong, you're wrong quickly (full stop, not partial)
2. **Targets are correctly set** — winners run to the full target, not just halfway
3. **The time stops are working** — the 3 trades in the +0 to +1R bucket were time stops that caught small profits rather than forcing the trade to hit the stop or target

**The problem is the 57% full-stop-loss rate.** More than half of all trades hit the maximum loss. This means the entry signals are wrong more often than right. The fix is not wider stops (that would increase the loss per trade) — it's better entry selection (fewer trades, higher quality).

## 18.10 Proposed Portfolio: ORB + VWAP Bounce Only

Based on the backtest data, the optimal portfolio is to trade only the two profitable strategies and drop the three that are losing or marginal.

**The numbers:**
- **24 trades over 42 days** (0.57 trades/day average)
- **62% win rate** (15 wins, 9 losses)
- **$676 per trade** average
- **$16,213 total P&L** ($97,277 annualized)
- **97% ROI on $100K at 1% risk**

**Compared to all five strategies combined:**
- 72 trades, 42% win rate, $151/trade, $65,363 annualized

The two-strategy portfolio makes 49% MORE money with 67% FEWER trades. The eliminated strategies (VWAP Reject, Gap Fade, Trend Day) collectively lost -$5,318, dragging down the portfolio. Removing them doesn't just eliminate losses — it frees mental bandwidth and reduces commission/slippage costs ($922 in commissions for 72 trades vs ~$310 for 24).

This validates the book's core thesis: **trade less, trade better. Quality over quantity.**

## 18.11 Next Steps

1. **Extended backtest.** 42 days is not enough. We need 6-12 months of data across different market regimes (bullish, bearish, choppy). yfinance limits 5-min data to ~60 days. Options: use Polygon API for historical 1-min data, or build a data collection pipeline that stores daily intraday bars
2. **Regime filter.** Add a market regime overlay — VIX level, SPY vs SMA50, trend direction. The strategies may perform differently in bull vs bear markets, and the portfolio allocation should adapt
3. **VWAP Reject rebuild.** If we want short-side VWAP exposure, the rejection strategy needs complete rebuilding with stricter confirmation (consecutive closes, volume, candle structure)
4. **Trend Day reimplementation.** Add ORB confirmation + EMA stacking + range expansion as prerequisites. The current "above VWAP" check is too loose
5. **Small cap extension.** This backtest only tested SPY. The book's strategies include small cap momentum (Chapter 12) which requires different data sources (Polygon scanner, individual stock 1-min bars)
6. **Live paper trading.** Run the ORB + VWAP Bounce strategies in paper trading for 30 days to validate the backtest results with real-time fills
