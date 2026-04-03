# Chapter 9: Position Sizing — The Only Thing That Keeps You Alive

## 9.1 Risk Per Trade: The 1% Rule

Position sizing is not exciting. It is not what makes trading stories interesting. It is, however, the single most important factor separating traders who survive from traders who blow up.

**The 1% rule:** Never risk more than 1% of your total account on a single trade.

- Account size: $25,000
- Max risk per trade: $250 (1% of $25,000)
- If your stop is $0.50 below entry, you trade 500 shares ($250 / $0.50)
- If your stop is $1.00 below entry, you trade 250 shares ($250 / $1.00)
- If your stop is $0.25 below entry, you trade 1,000 shares ($250 / $0.25)

**Why 1%?** With 1% risk per trade:
- You can take 10 consecutive losses and still have 90% of your account
- A 10-loss streak at 1% risk costs 9.6% of your account (compounding losses)
- A 10-loss streak at 5% risk costs 40.1% of your account
- A 10-loss streak at 10% risk costs 65.1% of your account — you need a 186% return to break even

Ten consecutive losses sounds unlikely, but it's not. With a 50% win rate, a 10-loss streak has a 0.1% chance of occurring on any given set of 10 trades. Over 1,000 trades, you'll likely encounter one. The 1% rule ensures you survive it.

**For beginners: start at 0.5%.** Your first 100 trades are tuition. Minimize the cost of learning. Move to 1% after you've proven you can follow your rules.

## 9.2 Calculating Position Size from Stop Distance

This is the math you do before EVERY trade. No exceptions.

**Formula:** Shares = Max Risk / Stop Distance

Where:
- Max Risk = Account × 0.01 (1% rule)
- Stop Distance = Entry Price - Stop Price (for longs)

**Examples:**

$50,000 account, entering XYZ at $15.00, stop at $14.50:
- Max risk: $500 (1% of $50K)
- Stop distance: $0.50
- Position size: 1,000 shares ($500 / $0.50)
- Position value: $15,000 (30% of account — this is fine, it's the RISK that matters, not the position value)

$50,000 account, entering TSLA at $250.00, stop at $247.00:
- Max risk: $500
- Stop distance: $3.00
- Position size: 166 shares ($500 / $3.00)
- Position value: $41,500 (83% of account — this is fine because RISK is still only 1%)

$50,000 account, entering AAPL at $180.00, stop at $179.50:
- Max risk: $500
- Stop distance: $0.50
- Position size: 1,000 shares ($500 / $0.50)
- But 1,000 × $180 = $180,000 — you don't have enough buying power
- Solution: use a wider stop or skip the trade

**The position size calculator is mandatory.** Build it into a spreadsheet, an app, or your broker's order interface. Never calculate position size in your head during a trade.

## 9.3 Adjusting for Volatility

Not all $0.50 stops are equal. A $0.50 stop on a stock with $0.30 ATR is very wide (1.7x ATR). A $0.50 stop on a stock with $5.00 ATR is very tight (0.1x ATR).

**ATR-adjusted sizing:**

Instead of a fixed dollar stop, express your stop as a multiple of ATR:
- Conservative: 1.5x ATR
- Standard: 1.0x ATR
- Aggressive: 0.5x ATR

**Example with ATR adjustment:**
- Stock ATR (14-period, 5-minute): $0.30
- Standard stop: 1.0x ATR = $0.30
- Max risk: $500
- Shares: $500 / $0.30 = 1,667 shares

vs. a higher-volatility stock:
- Stock ATR: $1.50
- Standard stop: 1.0x ATR = $1.50
- Max risk: $500
- Shares: $500 / $1.50 = 333 shares

ATR-adjusted sizing automatically reduces your position size on volatile stocks and increases it on calm stocks. This normalizes risk across different stocks — each trade risks the same dollar amount regardless of the stock's volatility.

## 9.4 Max Daily Loss

**The 3% daily loss limit:** When you've lost 3% of your account in a single day, stop trading. Turn off the screens. Walk away.

With a $50,000 account:
- Max daily loss: $1,500 (3%)
- At 1% risk per trade, this means 3 full-size losing trades

**Why this rule saves accounts:**

Bad days happen. But CATASTROPHIC days only happen when you try to "make back" the losses. The sequence is predictable:
1. Lose trade 1 (-$500)
2. Lose trade 2 (-$500)
3. Now you're frustrated. You size up to make it back faster
4. Lose trade 3 (-$1,000 because you doubled your size)
5. Now you're in revenge mode. You size up again
6. Lose trade 4 (-$2,000)
7. Total: -$4,000 (8% of account) in a single day

With the 3% rule:
1. Lose trade 1 (-$500)
2. Lose trade 2 (-$500)
3. Lose trade 3 (-$500)
4. Stop. Total: -$1,500 (3% of account)

The difference: $1,500 loss vs $4,000 loss. Same bad day, very different outcome. The 3% rule turns a potential catastrophe into a manageable setback.

**The "three strikes" variant:** Instead of a dollar limit, stop after 3 losing trades in a row, regardless of size. This catches the days when the market is choppy and your strategy isn't working. Three strikes and you're out — for the day.

## 9.5 Scaling Size with Consistency

Position sizing should adapt to your recent performance — not to your emotions, but to your statistical results.

**The consistency ladder:**
1. **Learning phase (0.25-0.5% risk):** First 100 trades. You're learning to execute. Minimize tuition
2. **Proving phase (0.5-1% risk):** Trades 100-300. You have a plan and you're following it. Win rate is stable. Slowly increase risk
3. **Standard phase (1% risk):** Trades 300+. You have a positive track record, a tested strategy, and emotional control. Full position sizing
4. **Scaling phase (1-2% risk):** After 500+ trades with consistent profitability. Only on A+ setups. Never more than 2% on any single trade

**When to scale DOWN:**
- 3 consecutive losing days → drop to 0.5% risk for the next 5 trading days
- Max drawdown exceeds 10% → drop to 0.25% risk until you recover 50% of the drawdown
- You violated your trading plan → drop to 0.5% for 10 trading days as penalty

Scaling down is not punishment — it's risk management. When you're losing, the worst thing you can do is maintain full size. Your strategy may be in a poor regime. Your psychology may be compromised. Reducing size gives you time to diagnose without compounding the damage.

## 9.6 The Account Size Problem

**The PDT rule (Pattern Day Trader):** In the US, if you make 4+ day trades in 5 business days in a margin account with less than $25,000, you're classified as a Pattern Day Trader and restricted from day trading.

**Practical implications:**
- You need $25,000+ in a margin account to day trade freely
- With $25,000 and 1% risk, your max risk per trade is $250 — enough for small cap scalps but tight for large caps
- With $50,000 and 1% risk, $500/trade gives more flexibility
- With $100,000+, you can trade most strategies comfortably

**Alternatives to PDT:**
- Cash account (no PDT, but no margin and T+1 settlement — limits trading frequency)
- Offshore brokers (no PDT, but regulatory risk and often worse execution)
- Futures (no PDT rule; ES, NQ micro futures are popular alternatives)
- Options (day trading options has different rules; more complex)

**The honest truth about undercapitalization:** Starting with $5,000 and trying to day trade is extremely difficult. At 1% risk ($50/trade), you need very tight stops ($0.10-0.25) which get triggered by normal noise. The math doesn't work at small account sizes unless you accept much higher risk per trade — which leads to blowup.

Recommended minimum to day trade seriously: $50,000. Below that, consider paper trading while building capital through a job, or trading in a cash account with limited frequency.

## 9.7 Commission and Slippage Impact

Every trade has friction costs that reduce your profits. These costs are often ignored in strategy discussions but can determine whether a strategy is actually profitable.

**Commission:**
- Most brokers: $0 commission (Robinhood, Schwab, Fidelity) — but you pay through wider spreads
- IBKR Pro: $0.005/share ($5 per 1,000 shares) — tighter spreads, better fills
- True cost at IBKR: ~$5-10 per round trip on a 1,000-share position

**Slippage:**
- The difference between the price you expect and the price you get
- On tight-spread large caps: $0.01-0.02/share
- On wide-spread small caps: $0.03-0.10/share
- During fast-moving breakouts: $0.05-0.20/share
- Estimated cost: $10-50 per round trip on a 1,000-share position

**Total friction per trade:**
- Best case (large cap, limit orders): $15-20 per round trip
- Average case (mid cap, mixed order types): $25-50
- Worst case (small cap, market orders at breakout): $50-200

**The friction test:** If your strategy's average profit per trade is $100 and your friction cost is $50, half your profit goes to friction. You need either larger average wins (bigger moves, more shares) or lower friction (better execution, tighter spreads).

At 5 trades per day, 250 trading days per year = 1,250 trades:
- At $25 friction: $31,250/year in costs
- At $50 friction: $62,500/year in costs

This is real money. A strategy that generates $100,000/year in gross profits but costs $62,500 in friction only nets $37,500. Understanding and minimizing friction is as important as the strategy itself.

**How to reduce friction:**
- Use limit orders instead of market orders (saves $0.02-0.05/share)
- Trade stocks with tight spreads (<$0.03)
- Reduce trading frequency — fewer, higher-quality trades mean less friction
- Use IBKR Pro for better execution (narrower spreads offset the commission)
- Size appropriately — 100 shares on a $10 stock costs the same friction as 1,000 shares but gives you 10x fewer shares to profit from
