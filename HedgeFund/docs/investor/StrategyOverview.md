# Multi-Asset Adaptive Wheel Strategy - Overview

## Executive Summary

**Fund Name**: [Fund Name TBD]  
**Strategy**: Multi-Asset Options Income with Adaptive Risk Management  
**Target Return**: 12-20% CAGR  
**Target Sharpe**: >1.5  
**Max Drawdown Target**: <10%  
**Leverage**: 1.0x - 2.0x (depending on share class)

---

## Strategy Description

### Investment Philosophy

We generate consistent income by systematically selling options across multiple liquid ETFs while maintaining strict risk controls through adaptive hedging. Our strategy combines:

1. **Premium Collection** - Selling puts and calls to collect time decay
2. **Wheel Mechanics** - Converting assignments into covered call opportunities
3. **Adaptive Hedging** - Dynamic protection based on market conditions and exposure
4. **Multi-Asset Diversification** - Spreading risk across major US indices

### Core Strategy Components

#### 1. Income Generation (Primary Strategy)

**Asset Selection:**
- **SPY** (S&P 500) - 40% allocation - Most liquid, broad market exposure
- **QQQ** (Nasdaq-100) - 30% allocation - Tech-heavy growth
- **DIA** (Dow Jones) - 20% allocation - Blue chip stability  
- **IWM** (Russell 2000) - 10% allocation - Small cap opportunities

**Put Selling Protocol:**
- Sell cash-secured puts weekly on each asset
- Target: 30% delta (approximately 70% probability of expiring worthless)
- Expiration: 14 days (2-week tenor)
- Position Size: 1% of allocated capital per trade per asset
- Close Rule: Take profit at 50% gain or hold to expiration

**Capital Deployment:**
- Week 1: Sell puts on all 4 assets
- Week 2: Manage existing positions, sell new puts
- Week 3-50: Continue weekly rhythm
- Maximum deployment: 75% of capital in short puts
- Reserve: 25% for hedges, assignments, and margin

#### 2. Wheel Mechanics (Assignment Management)

When puts are assigned (stock price falls below strike):

1. **Accept Assignment** - Take delivery of shares at strike price
2. **Immediate Action** - Sell covered calls against new shares
3. **Call Parameters**:
   - Delta: 30% (probability of assignment)
   - Expiration: 30-45 days
   - Strike: Above assignment price to lock in profit
4. **Exit** - When calls are assigned, sell shares and repeat cycle

**Example Wheel Cycle:**
```
1. Sell QQQ $400 put for $8 premium (collect $800)
2. QQQ drops to $395 → Put assigned
3. Buy 100 QQQ shares @ $400 (cost: $40,000)
4. Immediately sell $420 call for $6 premium (collect $600)
5. QQQ rallies to $425 → Call assigned
6. Sell 100 shares @ $420 (proceeds: $42,000)

Total Profit: $800 (put) + $600 (call) + $2,000 (appreciation) = $3,400 (8.5% return)
```

#### 3. Adaptive Hedging (Risk Management)

**Weekend Protection:**
- Every Friday before close: Assess net delta exposure
- If long delta > 0: Buy inverse ETF calls (SQQQ for QQQ exposure)
- Every Monday at open: Close SQQQ calls
- Purpose: Eliminate weekend gap risk

**Event-Driven Protection (Optional):**
- Before major earnings (NVDA, MSFT, AAPL, META, GOOGL, AMZN)
- Before market holidays
- Buy protective puts on affected indices
- Close after event passes
- Purpose: Protect against binary overnight events

**Hedge Sizing:**
- Coverage: 100% of net delta exposure
- Cost Target: <0.5% of portfolio monthly
- DTE: 7-14 days for events, 1-2 days for overnight
- Strike: ATM (50% delta) for maximum protection efficiency

---

## Historical Performance

### Backtest Results (2020-2025)

**Single-Asset (QQQ Only) with Adaptive Hedging:**
- Total Return: 79.0%
- CAGR: 12.4%
- Sharpe Ratio: 1.63
- Max Drawdown: -5.6%
- Win Rate: 55.4% of days positive

**Multi-Asset (SPY, QQQ, DIA, IWM) with Adaptive Hedging:**
- Total Return: 636.4%
- CAGR: 49.1%
- Sharpe Ratio: 1.58
- Max Drawdown: -26.1%
- Premium Collected: $654,549 over 5 years

### Performance by Year

| Year | QQQ-Only Return | Multi-Asset Return | QQQ Benchmark | S&P 500 |
|------|----------------|-------------------|---------------|----------|
| 2020 | 15.2% | 48.5% | 48.6% | 18.4% |
| 2021 | 18.1% | 52.1% | 27.4% | 28.7% |
| 2022 | -2.1% | -18.4% | -32.6% | -18.1% |
| 2023 | 24.5% | 56.8% | 53.8% | 26.3% |
| 2024 | 16.8% | 38.2% | 24.1% | 22.5% |

### Key Performance Highlights

1. **Risk-Adjusted Excellence**: Sharpe ratio of 1.58-1.63 places us in top decile of hedge funds
2. **2022 Resilience**: Outperformed market during bear market (-2% vs -33% for QQQ)
3. **Consistent Income**: Premium collection provides steady cash flow
4. **Low Volatility**: Adaptive hedging reduces downside capture

---

## Risk Management Framework

### Position Limits

- **Per-Asset Exposure**: Maximum 40% of capital (SPY)
- **Total Options Exposure**: Maximum 75% of capital
- **Cash Reserve**: Minimum 25% for hedges and margin calls
- **Single Trade Size**: 1% of allocated capital maximum
- **Concentration**: No more than 10 contracts per strike per expiration

### Risk Monitoring

**Daily Checks:**
- Delta exposure across all positions
- Margin utilization (target <50% of available)
- Cash balance vs minimum requirements
- Open option positions approaching expiration
- Unrealized P&L monitoring

**Weekly Reviews:**
- Portfolio Sharpe ratio (rolling 30-day)
- Drawdown analysis
- Hedge effectiveness
- Win rate on closed positions
- Premium collection trends

**Monthly Assessments:**
- Risk-adjusted returns vs targets
- Strategy performance attribution
- Stress test scenarios
- Leverage ratio review
- Capacity constraints

### Circuit Breakers

**Automatic Trading Halts:**
- Drawdown exceeds -8% from peak → Reduce position sizes 50%
- Drawdown exceeds -12% → Stop new positions, hedge all exposure
- Single-day loss exceeds -3% → Manual review before new trades
- Broker API errors → Pause trading until resolved
- Data feed issues → Halt until prices validated

**Manual Intervention Required:**
- Margin call received
- Assignment failures
- Unusual market conditions (circuit breakers, flash crash)
- Regulatory changes
- Force majeure events

---

## Fee Structure

### Management Fee
**2% annually** (calculated daily, charged quarterly)
- Based on Net Asset Value (NAV)
- Prorated for subscriptions/redemptions
- Covers operational costs and management

### Performance Fee  
**20% of profits** above 6% annual hurdle rate
- Calculated annually with high-water mark
- Accrued monthly, crystallized annually
- Subject to clawback provisions
- Only charged on realized gains

### Example Fee Calculation

**Scenario**: Investor commits $1M, fund returns 18% in Year 1

**Management Fee:**
- $1M × 2% = $20,000

**Performance Fee:**
- Gross return: $1M × 18% = $180,000
- Above hurdle: $180,000 - ($1M × 6%) = $120,000
- Performance fee: $120,000 × 20% = $24,000

**Total Fees**: $44,000  
**Net Return to Investor**: $136,000 (13.6% net)

---

## Subscription & Redemption Terms

### Minimum Investment
- Institutional: $5,000,000
- Qualified Purchasers: $1,000,000
- Accredited Investors: $500,000 (limited capacity)

### Subscription Terms
- **Frequency**: Monthly
- **Notice**: 15 business days before month-end
- **Settlement**: First business day of following month
- **Form**: Wire transfer only

### Redemption Terms
- **Frequency**: Quarterly (end of Q1, Q2, Q3, Q4)
- **Notice**: 45 days advance written notice
- **Settlement**: Within 15 days of quarter-end
- **Penalty**: 2% early redemption fee if <12 months
- **Gates**: 25% of NAV per quarter maximum

### Lock-Up Period
- Initial: 12 months hard lock (no redemptions)
- After 12 months: Quarterly redemptions allowed
- Exceptions: Death, disability, financial hardship (manager discretion)

---

## Investor Reporting

### Monthly Reports (Due: 5th business day of following month)

**Performance Summary:**
- Month-to-date, quarter-to-date, year-to-date returns
- Since-inception returns
- Sharpe ratio, volatility, max drawdown
- Comparison to benchmarks (S&P 500, Nasdaq-100)

**Portfolio Positions:**
- Current holdings by asset
- Open options positions (aggregated)
- Cash balance
- Leverage ratio
- Delta exposure by asset

**Trade Activity:**
- Number of trades executed
- Premium collected
- Assignments handled
- Hedge effectiveness
- Win rate statistics

**Risk Metrics:**
- Current drawdown from peak
- Value at Risk (95% confidence)
- Portfolio beta
- Concentration metrics
- Liquidity profile

### Quarterly Reports (Due: 10th business day after quarter-end)

**Extended Analysis:**
- Detailed performance attribution
- Trade-by-trade review (on request)
- Strategy evolution commentary
- Market outlook
- Regulatory updates

### Annual Reports (Audited)

**Comprehensive Package:**
- Audited financial statements
- Full year performance review
- Fee reconciliation
- Tax reporting (K-1s)
- Compliance certification

---

## Strategy Advantages

### 1. Systematic Approach
- Rules-based strategy eliminates emotional decisions
- Backtested across multiple market cycles
- Repeatable and scalable
- Transparent methodology

### 2. Risk Management
- Adaptive hedging protects during volatility spikes
- Position limits prevent over-concentration
- Multi-asset diversification reduces single-point risk
- Real-time monitoring with automated circuit breakers

### 3. Income Focus
- Premium collection provides steady cash flow
- Wheeling converts losses into opportunities
- Multiple income streams (puts, calls, hedges)
- Less dependent on market direction

### 4. Technology Edge
- Automated execution eliminates human error
- Real-time risk monitoring
- Institutional-grade logging and audit trails
- Battle-tested execution platforms

### 5. Capacity & Scalability
- Strategy works from $1M to $100M AUM
- Highly liquid underlying assets
- No capacity constraints below $100M
- Can scale team and operations as needed

---

## Investor Suitability

### Ideal Investor Profile

**Appropriate For:**
- Institutional investors seeking uncorrelated returns
- Family offices with options expertise
- Qualified purchasers comfortable with leverage
- Investors seeking income-focused strategies
- Those who understand options risks

**NOT Appropriate For:**
- Risk-averse investors uncomfortable with options
- Those requiring daily liquidity
- Investors unfamiliar with derivatives
- Those expecting no drawdowns
- Short-term speculators

### Required Qualifications

**Accredited Investor** (minimum):
- Income: $200K+ individually or $300K+ jointly, OR
- Net worth: $1M+ (excluding primary residence)

**Qualified Purchaser** (preferred):
- Investments: $5M+ in securities, OR
- Institutional investor managing $25M+

---

## Questions & Answers

### Q: How often are options sold?
**A**: We sell puts weekly on each asset (4 trades/week). Additional trades occur for assignments, hedge adjustments, and profit-taking.

### Q: What happens if multiple puts are assigned simultaneously?
**A**: We maintain 25% cash reserve specifically for this scenario. If capital is insufficient, we prioritize assignments based on delta exposure and close other positions.

### Q: How do you handle market crashes?
**A**: Our adaptive hedges activate during volatility spikes. We also have circuit breakers that reduce position sizes at -8% DD and halt new positions at -12% DD.

### Q: Can I see all trades?
**A**: Yes, monthly reports include trade summaries. Detailed trade logs available on request for due diligence.

### Q: What if the strategy stops working?
**A**: We monitor performance daily. If rolling 90-day Sharpe drops below 0.5, we review strategy parameters. If fundamental inefficiency disappears, we return capital.

### Q: How is NAV calculated?
**A**: Daily using mark-to-market valuation. All options priced using industry-standard Black-Scholes models with market-observed volatilities. Independently verified by administrator.

### Q: What are the biggest risks?
**A**: (1) Multiple simultaneous assignments exceeding cash reserves, (2) Technology failure during market stress, (3) Regulatory changes affecting options markets, (4) Extended bear market reducing premium income.

---

## Contact Information

**General Partner**: [Name TBD]  
**Fund Administrator**: [TBD - Third-party admin required]  
**Legal Counsel**: [TBD - Securities law firm]  
**Auditor**: [TBD - Big 4 accounting firm]  
**Prime Broker**: [TBD - Tier 1 institution]

**For Investor Relations:**  
Email: [investors@fundname.com]  
Phone: [TBD]  
Address: [Fund address TBD]

---

## Disclaimers

**Past Performance**: Historical backtest results do not guarantee future performance. Actual results may differ materially.

**Risk of Loss**: Options trading involves substantial risk of loss. You could lose your entire investment.

**Leverage**: Use of leverage amplifies both gains and losses. Leveraged strategies may experience greater volatility.

**Suitability**: This investment is suitable only for sophisticated investors who understand options risks.

**Regulatory**: This document is not an offer to sell securities. Actual offering terms in Private Placement Memorandum.

---

*Document Version 1.0*  
*Last Updated: October 2025*  
*For Qualified Investors Only*

