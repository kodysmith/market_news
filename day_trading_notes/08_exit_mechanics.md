# Chapter 8: Exit Mechanics — Getting Out Right (The Hard Part)

## 8.1 Why Exits Matter More Than Entries

There's a classic trading experiment: take random entries (flip a coin to go long or short) and apply disciplined exit rules. The result is consistently better than the reverse — perfect entries with undisciplined exits.

This counterintuitive result makes sense when you think about it:
- Entries determine where you get in. Exits determine what you keep
- A bad entry with a tight stop costs you a small loss. A good entry with no stop can cost you everything
- The market is approximately 50/50 in the short term. Your edge comes from making more on winners than you lose on losers — that's entirely an exit problem

Most traders spend 90% of their study time on entries and 10% on exits. The ratio should be reversed.

## 8.2 The Stop Loss: Non-Negotiable

A stop loss is a predetermined price at which you exit a losing trade. It is not optional. It is not a suggestion. It is the single most important order in your trading system.

**Where to place the stop:**
- **Below the trigger candle low (for longs):** The most common stop placement. If the trigger candle's low is violated, the entry thesis is invalidated
- **Below VWAP:** If your thesis is "stock above VWAP is bullish," then a break below VWAP invalidates the thesis. Stop just below VWAP ($0.10-0.20 buffer)
- **Below the pattern low:** For flag breakouts, the stop goes below the flag low. For ORB, the stop goes below the opening range low (or the midpoint for tighter risk)
- **ATR-based stop:** Stop at 1x ATR below entry. Adjusts automatically for the stock's volatility

**What a stop loss is NOT:**
- It is NOT a "suggestion that I might exit if I feel like it"
- It is NOT a number you move further away when the trade goes against you
- It is NOT something you cancel because "it'll come back"
- It IS a hard order placed with your broker IMMEDIATELY after entry
- It IS the maximum amount you've agreed to lose on this trade

**Moving stops DOWN is never acceptable.** If you entered long at $10.50 with a stop at $10.30, and the stock hits $10.35, the temptation is to move the stop to $10.20 "to give it more room." This is how a planned $0.20/share loss becomes a $0.50/share loss. And then $1.00. And then you're holding a day trade overnight, praying. Don't.

**Moving stops UP (trailing) is always acceptable** and is covered in 8.4.

## 8.3 Profit Targets

Before entering any trade, define your profit target. This serves two purposes: it tells you the risk/reward ratio (is this trade worth taking?) and it gives you a specific exit price.

**Risk/reward-based targets:**
- **Minimum acceptable R:R = 1:2.** If your stop is $0.50 below entry, your target must be at least $1.00 above entry. This means you can be wrong 60% of the time and still be profitable
- **Ideal R:R = 1:3.** $0.50 risk for $1.50 reward. Now you only need to be right 25% of the time to break even
- **If R:R < 1:1.5, skip the trade.** The math doesn't work unless you have a win rate above 60%, which is rare and unsustainable

**Technical targets:**
- **Next resistance level:** If the stock breaks $10.50, the next resistance is at $11.00 (prior day high). Target = $11.00
- **Measured move:** Flag pole was $1.00. Flag breakout at $10.50. Target = $11.50 (pole height added to breakout)
- **VWAP:** If you're short, target VWAP. If you're long from VWAP, target the high of day
- **Prior day high/low:** Strong gravitational levels that attract price

**The "2R and done" approach:** For newer traders, a simple system: always target 2R. If risk is $0.50, target is $1.00. Exit 100% of position at 2R. This is boring, consistent, and profitable. You can add sophistication later.

## 8.4 Trailing Stops

Once a trade moves in your favor, the trailing stop protects your gains while allowing the trade to continue working.

**Breakeven stop (after 1R):**
- When the trade reaches 1R profit (your target is 2R away), move your stop to breakeven (entry price)
- Now you have a "free trade" — the worst case is $0 profit/loss
- This is the most important trailing stop move. It turns a risky position into a risk-free position

**Trailing below higher lows:**
- As the stock makes higher highs and higher lows, move your stop below each new higher low
- Example: Entered at $10.50, stop at $10.30. Stock makes a high at $11.00, pulls back to $10.70, then continues up. Move stop to $10.65 (below the $10.70 higher low)
- This gives the trade room to breathe while protecting profits

**ATR trailing stop:**
- Trail the stop at 1.5x ATR below the current high
- If ATR is $0.30 and the stock is at $11.00, stop is at $10.55
- As the stock moves to $11.50, stop moves to $11.05
- This automatically adjusts for volatility

**EMA trailing stop:**
- For trend day positions, trail below the 9 EMA on the 5-minute chart
- If a candle closes below the 9 EMA, exit the trade
- Simple, mechanical, and keeps you in strong trends

**When trailing stops hurt you:** In choppy markets, trailing stops get triggered constantly. The stock moves up $0.50, pulls back $0.30 (triggering your trail), then continues up $1.00 — without you. For choppy stocks, use fixed targets (2R) instead of trailing stops.

## 8.5 Scaling Out

Scaling out means exiting a position in multiple tranches. It reduces your average win but dramatically improves consistency.

**The 1/3 method:**
- **Sell 1/3 at 1R:** Covers your risk. Even if the remaining 2/3 hits the stop, you break even
- **Sell 1/3 at 2R:** Locks in profit. Your average exit is now 1.5R, with 1/3 still running
- **Let 1/3 ride with trailing stop:** This is the "runner." It captures the 5R and 10R moves that make your month

**Why this works mathematically:**
- On a trade with $0.50 risk:
  - Sell 333 shares at +$0.50 (1R): $166.50
  - Sell 333 shares at +$1.00 (2R): $333.00
  - Runner hits trail at +$0.75: $249.75
  - Total: $749.25 on 1,000 shares ($0.75/share avg)
  - vs. all-at-2R: $1,000 — scaling out got less
  - BUT on the trades that run to 5R: runner makes $833 extra that all-at-2R missed

**The psychological benefit:** Taking partial profits at 1R and 2R gives you the psychological reward of locking in wins, which makes it easier to hold the runner. Without partial profits, many traders exit the entire position too early because they can't bear to watch unrealized profit fluctuate.

## 8.6 Time-Based Exits

If a trade hasn't moved in your direction within a defined time window, something is wrong with the thesis. Exit and move on.

**The 15-minute rule (for scalps):** If the trade hasn't reached 1R within 15 minutes, close it at market. The momentum you were trading is not there.

**The 30-minute rule (for ORB and flags):** If the breakout hasn't followed through within 30 minutes, the breakout has failed. Close the position. Don't wait for the stop to be hit if the trade is clearly dead.

**The lunch rule:** If you're holding a morning trade into lunchtime (12 PM) and it's at a small gain or breakeven, take it off. The midday dead zone kills momentum trades. A stock that's up $0.50 at 10:30 AM is often back to flat by 1:00 PM.

**Why time-based exits work:** They prevent "dead trades" — positions that aren't losing (so your stop isn't triggered) but aren't winning either. Dead trades tie up capital, mental bandwidth, and emotional energy. Closing them frees you to take the next opportunity. Opportunity cost is a real cost.

## 8.7 The Close Rip and End-of-Day Exits

Day trades should be closed before the market close unless you have a specific overnight thesis.

**The 3:30 PM deadline:** By 3:30 PM, decide what to do with open positions:
- In profit: take the profit or set a trailing stop for the final 30 minutes
- At breakeven: close. Don't risk giving back the day's work for 30 minutes of potential
- In a loss: close. Holding a losing day trade overnight turns a day trade into a bad swing trade

**The 3:50 PM forced exit:** Any position still open at 3:50 PM gets closed at market. No exceptions. This is non-negotiable.

**The "power hour" exception:** If you're experienced and the stock is in a clear trend with strong volume into the close, you can hold until 3:55 PM. But this is an earned exception, not a default.

**Why overnight holding kills day traders:** The stock can gap 5%+ against you overnight. Your day trade stop ($0.50) is meaningless when the stock opens $3.00 lower. A -$500 day trade becomes a -$3,000 overnight disaster. This is how accounts blow up.

## 8.8 Partial Profits vs All-or-Nothing

The debate between scaling out (partial profits) and all-or-nothing (exit 100% at target) is one of the most discussed in day trading.

**The math says all-or-nothing is better... in theory.** If you have a 50% win rate with 2R targets, your expectancy per trade is: (0.50 × $1,000) - (0.50 × $500) = $250. With scaling out (average exit at 1.5R), expectancy drops to: (0.50 × $750) - (0.50 × $500) = $125.

**But the math doesn't account for psychology.** In practice:
- All-or-nothing traders often exit at 1.5R because they can't watch the full 2R develop, making their actual average win less than 2R
- All-or-nothing traders often hold losers past their stop "because they need the full win to make up for it"
- Scaling out traders have better execution because partial profits provide psychological relief

**The recommendation:** Start with scaling out (1/3 at 1R, 1/3 at 2R, 1/3 trailing). This builds the discipline of holding winners and cutting losers. After 500 trades with a positive track record, experiment with holding larger portions to 2R or 3R. Let the data tell you which approach works better for YOUR psychology.

The best exit strategy is the one you actually execute consistently. A slightly suboptimal exit plan followed rigidly beats the theoretically optimal plan followed inconsistently.
