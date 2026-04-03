# Chapter 10: Strategy 1 — Opening Range Breakout (ORB)

## 10.1 The Theory

The Opening Range Breakout is the oldest and most studied intraday strategy. It was popularized by Toby Crabel in the 1990s and has been validated by decades of academic and practitioner research.

The premise: the first N minutes of the trading day establish a range — the opening range. This range represents the market's initial consensus on value. A sustained break above or below this range signals that the market has decided on a direction for the day.

Why it works:
- **Information aggregation.** The opening range is where overnight information (earnings, news, pre-market trading) gets reconciled with the full liquidity of the regular session. By the time the range is established, the market has processed the most important signals
- **Commitment.** A break above the opening range means buyers are willing to pay MORE than anyone paid in the first 15-30 minutes. That's a commitment of capital that often leads to follow-through
- **Algorithmic reinforcement.** Many institutional algorithms use opening range breakouts as entry signals. When the ORB triggers, these algorithms add buying pressure, creating a self-fulfilling momentum burst
- **Short covering.** Traders who shorted within the opening range are now losing money. Their stop-loss orders (buy orders) fire, adding upward pressure

## 10.2 The Setup

**Pre-conditions:**
- Stock is on your watchlist (passed the scanner, has a catalyst, elevated RVOL)
- Key levels are marked on the chart (PMH, PML, PDH, PDL, VWAP)
- Position size is calculated based on the opening range height

**Step-by-step execution:**

1. **Wait for the opening range to form.** Do nothing for the first 5, 15, or 30 minutes (your chosen time frame). Mark the high and low of that range on your chart

2. **Assess range quality:**
   - Narrow range (< 50% of ATR): Compressed energy, likely to produce a strong breakout
   - Wide range (> 75% of ATR): Most of the daily move may already be used up. Be cautious
   - Range with clear direction (opens low, closes high): Bullish bias. Favor long breakout
   - Range with no direction (opens and closes near the middle): No bias. Wait for the break to decide

3. **Set your alerts:**
   - Alert at ORB high (long trigger)
   - Alert at ORB low (short trigger)

4. **Entry on breakout:**
   - Long: Buy when price closes above ORB high on the time frame candle (not just wicks above)
   - Short: Sell when price closes below ORB low
   - Volume must be above average on the breakout candle

5. **Stop loss:**
   - Conservative: Opposite side of the opening range (risk = full range height)
   - Aggressive: Middle of the opening range (risk = half range height)
   - Very aggressive: Just below the breakout candle low (tightest stop, highest false breakout rate)

6. **Profit target:**
   - 2x the opening range height (2R)
   - Or the next technical level (PDH, round number, prior resistance)
   - Scale out: 1/3 at 1R, 1/3 at 2R, 1/3 trailing

## 10.3 Filters That Improve Win Rate

The raw ORB has a win rate of approximately 45-55%. Adding filters improves it to 55-65%:

**Gap alignment filter:** Only take long ORB breakouts on stocks that gapped up. Only take short ORB breakouts on stocks that gapped down. When the gap and the breakout agree, the probability of follow-through increases significantly. Contra-gap breakouts (gap up but breaks low) are lower probability but can work as reversal signals.

**VWAP filter:** Only take long breakouts when the stock is above VWAP at the breakout. Only take short breakouts when below VWAP. VWAP alignment confirms institutional support for the move.

**Relative volume filter:** RVOL > 2x at the time of breakout. Higher RVOL = more conviction behind the move.

**Market direction filter:** Only take long ORB breakouts when SPY/QQQ is also above its ORB high. Trading against the market tide reduces win rate by 10-15%.

**Time filter:** The strongest ORB breakouts occur between 9:45-10:15 AM. Breakouts after 10:30 AM have lower follow-through. After 11:00 AM, the ORB strategy is largely dead for the day.

## 10.4 ORB on Large Caps (SPY, QQQ)

Trading ORB on market ETFs is the most consistent variant. SPY and QQQ have massive liquidity, tight spreads, and predictable behavior.

**SPY/QQQ ORB characteristics:**
- Use 15-minute or 30-minute opening range (5-minute is too noisy for ETFs)
- Average ORB range: 0.3-0.5% of price ($1.50-$2.50 on SPY)
- Expected follow-through: 1-2x the range (0.3-1.0% move)
- Win rate with filters: 55-60%
- This produces small but consistent daily income
- Ideal for traders who want predictability over excitement

**The SPY ORB daily routine:**
1. At 9:45 AM, mark the 15-minute ORB high and low on SPY
2. If SPY breaks above the ORB high with RVOL > 1.5, go long via shares or 0DTE calls
3. Stop at ORB midpoint
4. Target: 2x ORB range or VWAP + 1 ATR
5. Exit by 11 AM if target not hit

This strategy alone, executed daily on SPY with 1% risk and 2R targets, can produce 15-25% annual returns with a Sharpe ratio above 1.5 — modest but consistent and highly backtestable.

## 10.5 ORB on Gappers

Applying ORB to stocks on your gap scanner produces larger moves but lower win rates.

**Gapper ORB differences from ETF ORB:**
- Use 5-minute or 15-minute opening range (gappers move fast)
- Expected follow-through: 2-5x the opening range (bigger moves)
- Win rate: 40-50% (more false breakouts on volatile stocks)
- Spreads are wider, slippage is higher — factor this into sizing

**The gapper ORB checklist:**
1. Stock gapped > 4% on a Tier 1 or 2 catalyst
2. RVOL > 5x pre-market
3. Float < 50M shares
4. Opening range forms with above-average volume
5. Stock is above VWAP at ORB breakout
6. Breakout candle has 2x+ volume of range candles

When all six conditions are met, the trade is A+ grade. Take full size.

## 10.6 The Failed ORB

One of the most valuable ORB signals is when the breakout fails. A stock breaks above the ORB high, fails to hold, and drops back into the range. This "failed breakout" often leads to a move in the OPPOSITE direction.

**Why failed breakouts are powerful:** Everyone who bought the breakout is now trapped. Their long positions are underwater. As the stock falls back through the range, their stop losses trigger (selling pressure), which pushes the stock toward the ORB low. If the ORB low also breaks, it's a cascade.

**Trading the failed ORB:**
1. Stock breaks above ORB high — you enter long (standard ORB play)
2. The breakout fails — price drops back below ORB high
3. Exit your long immediately (stop was at ORB mid or ORB low)
4. Wait for price to break below ORB low on volume
5. Enter short below ORB low
6. Target: 2x the ORB range below the ORB low

Failed ORBs have a higher win rate than standard ORBs (55-65%) because they come with built-in fuel — the trapped longs who are now sellers.

## 10.7 Historical Win Rates and Expectancy

Based on backtesting across multiple datasets and time periods, here are the approximate statistics for ORB strategies:

**SPY/QQQ 15-minute ORB (2015-2025):**
- Trades per year: ~250 (one per trading day)
- Win rate: 52-58%
- Average win: 1.5-2.0R
- Average loss: 1.0R
- Profit factor: 1.3-1.7
- Annual return (1% risk): 15-25%
- Sharpe: 1.2-1.8

**Small cap gapper 5-minute ORB (filtered, 2015-2025):**
- Trades per year: ~100-150 (not every day has a quality gapper)
- Win rate: 42-50%
- Average win: 2.5-4.0R (larger moves)
- Average loss: 1.0R
- Profit factor: 1.4-2.0
- Annual return (1% risk): 20-40%
- Sharpe: 0.8-1.5 (higher variance)

**Key observation:** The large cap ORB is more consistent (higher Sharpe) while the small cap ORB has higher absolute returns but more variance. The optimal approach depends on your personality and account size. Many traders run both — SPY ORB for the daily base hit, gapper ORB for the occasional home run.

**Regime sensitivity:** ORB strategies perform best in trending, moderate-volatility markets (VIX 15-25). They struggle in extremely low volatility (VIX < 12, ranges are too tight) and extremely high volatility (VIX > 35, everything whipsaws). Monitor VIX as a regime filter for ORB.
