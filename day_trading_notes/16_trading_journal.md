# Chapter 16: The Trading Journal — Measuring What Matters

## 16.1 What to Track

Every trade gets logged. No exceptions. The journal is your data source for improvement.

**Per-trade fields:**

| Field | Example | Why |
|-------|---------|-----|
| Date | 2026-04-02 | Time-series analysis |
| Ticker | NVDA | Performance by stock |
| Direction | Long | Long vs short win rates |
| Strategy | ORB | Performance by strategy |
| Setup grade | A+ | Performance by quality |
| Entry time | 9:47 AM | Performance by time of day |
| Exit time | 10:23 AM | Hold time analysis |
| Entry price | $142.50 | P&L calculation |
| Exit price | $144.80 | P&L calculation |
| Stop price | $141.80 | Risk calculation |
| Target price | $144.90 | R-multiple calculation |
| Shares | 350 | Position sizing review |
| Gross P&L | $805 | Raw performance |
| Commissions | $3.50 | Friction tracking |
| Net P&L | $801.50 | True performance |
| R-multiple | +3.3R | Risk-adjusted performance |
| Catalyst | Earnings beat Q1 | Performance by catalyst type |
| RVOL at entry | 4.2x | Volume filter effectiveness |
| VIX at entry | 18.5 | Regime analysis |
| Notes | Clean flag breakout, held to target | Qualitative review |
| Mistakes | None | Discipline tracking |

**End-of-day summary fields:**
- Total P&L (net)
- Number of trades
- Win rate (today)
- Average R-multiple (today)
- Largest win
- Largest loss
- Rule violations (count and description)
- Emotional state (1-5 scale: 1 = tilted, 5 = calm and focused)

## 16.2 Key Metrics

From your journal data, compute these metrics weekly and monthly:

**Win rate:** Winning trades / total trades. A useful starting point but NOT the most important metric. A 40% win rate with 3:1 R-multiple is more profitable than a 70% win rate with 0.5:1 R-multiple.

**Average win ($):** Total profit from winning trades / number of winning trades. This should be 2-3x your average loss.

**Average loss ($):** Total loss from losing trades / number of losing trades. This should be approximately 1R (your planned risk). If it's larger than 1R, you're holding losers too long or moving stops.

**Profit factor:** Gross profit / gross loss. A profit factor of 1.0 = breakeven. Target: > 1.5. Excellent: > 2.0. If profit factor < 1.0, you have no edge.

**Expectancy:** The expected dollar value of each trade.
Formula: (Win rate × Average win) - (Loss rate × Average loss)
Example: (0.50 × $400) - (0.50 × $200) = $100 per trade
At 5 trades/day, 20 days/month: $10,000/month expected value

**Average R-multiple:** Total R earned / total trades. This normalizes for position size. Target: > 0.3R average. This means each trade, on average, earns 0.3 times your risk amount.

**Sharpe ratio (estimated):** Mean daily P&L / standard deviation of daily P&L × sqrt(252). Target: > 1.5 for day trading. Above 2.0 is excellent.

**Max drawdown:** The largest peak-to-trough decline in your equity curve. Measure in both dollars and percentage. If max drawdown exceeds 15% of your account, something is wrong with your risk management.

## 16.3 Expectancy Per Trade

Expectancy is the single most important number in your trading journal. It tells you, on average, how much you make per trade.

**Positive expectancy = you have an edge.** Keep trading.
**Zero expectancy = you're breaking even.** Fix something (cut losers faster, hold winners longer, improve setup quality).
**Negative expectancy = you have no edge.** Stop trading real money. Go back to paper trading.

**How to calculate:**
1. Take your last 100 trades (minimum — more is better)
2. Separate into wins and losses
3. Calculate average win and average loss (in dollars)
4. Calculate win rate
5. Expectancy = (win rate × avg win) - (loss rate × avg loss)

**Example:**
- 100 trades: 48 wins, 52 losses
- Average win: $380
- Average loss: $180
- Expectancy = (0.48 × $380) - (0.52 × $180) = $182.40 - $93.60 = **$88.80 per trade**
- At 4 trades/day: $355/day → $7,100/month → $85,200/year

**What $88.80/trade means in practice:** Even though you lose more often than you win (48% vs 52%), your average win is 2.1x your average loss. The asymmetry makes you profitable. This is why risk/reward matters more than win rate.

**Minimum sample size:** Don't draw conclusions from fewer than 50 trades. Ideally, wait for 100 trades before making strategy changes. With fewer trades, random variance dominates and you can't distinguish signal from noise.

## 16.4 Performance by Strategy

Break down your metrics by strategy. This reveals which strategies are making money and which are bleeding.

**Example monthly breakdown:**

| Strategy | Trades | Win Rate | Avg R | Profit Factor | Expectancy |
|----------|--------|----------|-------|---------------|------------|
| ORB (SPY) | 18 | 61% | +0.52 | 2.1 | $104/trade |
| VWAP Bounce | 12 | 58% | +0.38 | 1.7 | $76/trade |
| Small Cap Momentum | 8 | 38% | +0.65 | 1.9 | $130/trade |
| Gap Fade | 6 | 33% | -0.12 | 0.8 | -$24/trade |

**Action based on this data:**
- ORB and VWAP bounce are consistent. Continue with current parameters
- Small cap momentum has the highest per-trade expectancy despite low win rate. The asymmetric R makes up for it
- Gap fade is negative expectancy. Either fix the strategy (better filters, tighter stops) or remove it from the playbook

**The "kill" decision:** If a strategy has negative expectancy over 30+ trades, kill it. Don't rationalize. Don't tweak parameters. Remove it from your playbook and replace it with a strategy that's working.

## 16.5 Performance by Time of Day

Track the time of your entries and correlate with P&L. Most day traders discover a clear pattern:

**Typical findings:**
- 9:30-10:00 AM: Moderate (high opportunity, high chaos, mixed results)
- 10:00-10:30 AM: Best (setups have had time to form, volatility is still high)
- 10:30-11:30 AM: Good (continuation moves, flags, VWAP bounces)
- 11:30 AM-1:00 PM: Poor (dead zone, choppy, overtrading territory)
- 1:00-2:00 PM: Variable (afternoon setups emerge but inconsistently)
- 2:00-3:00 PM: Moderate (power hour can be good on trend days)
- 3:00-4:00 PM: Risky (end-of-day positioning, unpredictable)

**Once you know your best and worst time windows:**
- Concentrate your trading in the profitable windows
- Set a hard rule to stop trading during your worst window (probably 11:30-1:00)
- This alone can turn a breakeven trader into a profitable one — by eliminating the hours where you give money back

## 16.6 Performance by Ticker Characteristics

Slice your data by stock characteristics to find where your edge is strongest:

**By market cap:**
- Small cap ($100M-$2B): Win rate? Average R?
- Mid cap ($2B-$10B): Win rate? Average R?
- Large cap ($10B+): Win rate? Average R?

**By float:**
- Micro float (<10M): Better or worse than your average?
- Low float (10-50M): Your best category?
- High float (>100M): Too boring for your style?

**By direction:**
- Long trades: Win rate and expectancy?
- Short trades: Win rate and expectancy?
- Many traders discover they're dramatically better at one direction. That's fine. Trade your strength and reduce the other

**By catalyst type:**
- Earnings: Win rate?
- News (non-earnings): Win rate?
- Technical only: Win rate?
- No catalyst: Win rate? (This should be the worst — if it's not, you're getting lucky)

Let the data tell you what you're good at. Then do more of that and less of everything else.

## 16.7 The Curve of Learning

Every day trader goes through a predictable learning curve. Understanding where you are prevents premature discouragement:

**Month 1-3: The Losing Phase**
- You will lose money. This is normal
- Focus on following your plan, not on P&L
- Success metric: "Did I follow my rules?" not "Did I make money?"
- Expected P&L: -5% to -15% of account (this is your tuition)

**Month 4-6: The Breakeven Phase**
- Losses shrink. Some winning days appear
- You start recognizing your best setups in real-time
- The journal starts revealing patterns in your performance
- Expected P&L: -3% to +3% of account

**Month 7-12: The Consistency Phase**
- More winning days than losing days
- You stop trading C-grade setups
- Position sizing becomes mechanical, not emotional
- Expected P&L: +1% to +5% per month

**Year 2+: The Scaling Phase**
- Consistent positive expectancy over 200+ trades
- Slowly increase position size
- Consider automation for the most mechanical strategies
- Expected P&L: +2% to +5% per month (20-60% annually)

**The critical period is months 3-6.** This is where most traders quit. They've lost enough to be discouraged but haven't traded enough to develop real skill. The ones who push through this phase — while maintaining small position sizes and rigid risk management — are the ones who eventually become profitable.

If you're in month 3 and losing, but your journal shows improving metrics (lower average loss, fewer rule violations, better setup quality), you're on the right track. The P&L will follow the process improvement. If your journal shows no improvement, you need to change your approach, not just grind longer.
