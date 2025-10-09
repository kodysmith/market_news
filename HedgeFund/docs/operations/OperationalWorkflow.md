# Operational Workflow - Daily, Weekly, Monthly Procedures

**Standard Operating Procedures for Hedge Fund Operations**

---

## Daily Operations

### Pre-Market Routine (7:00 AM - 9:30 AM ET)

#### 7:00 AM - System Health Check

**1. Infrastructure Status**
```bash
# Check all systems are running
systemctl status trading-engine
systemctl status risk-manager
systemctl status data-collector

# Check database connectivity
psql -h localhost -U trading -c "SELECT NOW();"

# Check Redis
redis-cli ping

# Check API connectivity
curl https://api.alpaca.markets/v2/account
curl https://api.interactivebrokers.com/health
```

**✓ Verify:**
- All services running
- Database accessible
- Redis responding
- Broker APIs reachable
- No alerts from previous night

**❌ If Issues:**
- Document in incident log
- Attempt automatic restart
- Escalate if restart fails
- Consider manual trading if critical

#### 8:00 AM - Market Data Validation

**2. Price Data Check**
```python
# Verify market data is current and accurate

for asset in ['SPY', 'QQQ', 'DIA', 'IWM']:
    current_price = market_data.get_price(asset)
    yf_price = yfinance.get_price(asset)
    
    # Check prices are within 1% (sanity check)
    diff_pct = abs(current_price - yf_price) / yf_price
    
    if diff_pct > 0.01:
        alert_ops(f"{asset} price mismatch: {diff_pct:.2%}")
        # Use backup data source
```

**✓ Verify:**
- Prices updating in real-time
- No stale data (check timestamps)
- Prices reasonable vs previous close
- Implied volatility data available

#### 8:30 AM - Position Reconciliation

**3. Reconcile to Broker**
```python
# Match our records to broker's overnight statements

def reconcile_positions():
    our_positions = position_manager.get_all_positions()
    broker_positions = alpaca_client.get_positions()
    
    discrepancies = []
    
    for our_pos in our_positions:
        broker_pos = find_matching_position(broker_positions, our_pos)
        
        if broker_pos is None:
            discrepancies.append(f"Missing at broker: {our_pos}")
        elif our_pos.quantity != broker_pos.quantity:
            discrepancies.append(
                f"Quantity mismatch {our_pos.symbol}: "
                f"Ours={our_pos.quantity}, Broker={broker_pos.quantity}"
            )
    
    if discrepancies:
        alert_ops("Position reconciliation failures", discrepancies)
        # HALT trading until resolved
        return False
    
    return True
```

**✓ Verify:**
- All positions match broker
- Cash balance matches
- No unexplained changes
- Margin balance correct

**❌ If Mismatches:**
- STOP - do not trade until resolved
- Investigate discrepancy immediately
- Contact broker if needed
- Document resolution
- Update system if needed

#### 9:00 AM - Pre-Market Risk Check

**4. Calculate Current Exposure**
```python
# Calculate portfolio state before market opens

portfolio_state = {
    'nav': calculate_nav(),
    'cash': get_cash_balance(),
    'margin_used': get_margin_used(),
    'delta_exposure': calculate_total_delta(),
    'positions_expiring_today': get_expiring_positions(),
    'assignments_expected': check_likely_assignments()
}

# Check if any positions need immediate attention
if portfolio_state['positions_expiring_today']:
    review_expiring_positions()

if portfolio_state['margin_used'] > 0.5:
    alert_manager("High margin utilization")
```

**✓ Review:**
- Current drawdown from peak
- Delta exposure by asset
- Margin utilization %
- Positions expiring today
- Hedges in place

#### 9:15 AM - Daily Trading Plan

**5. Generate Today's Trades**
```python
# Strategy engine determines what to trade today

trading_plan = strategy.generate_daily_plan(
    date=today,
    market_data=current_data,
    portfolio=portfolio_state
)

# Review plan
print(f"Today's Trading Plan ({len(trading_plan)} actions):")
for trade in trading_plan:
    print(f"  - {trade.action} {trade.quantity} {trade.symbol} {trade.details}")

# Risk approval
approved = risk_manager.approve_plan(trading_plan)

if not approved:
    alert_manager("Trading plan rejected by risk manager")
```

**Typical Monday Plan:**
- Sell 4 weekly puts (one per asset)
- Close any puts at 50% profit
- Check for assignments from Friday
- Buy overnight hedges (if needed)

### Market Hours (9:30 AM - 4:00 PM ET)

#### 9:30 AM - Market Open

**1. Execute MOO (Market-On-Open) Orders**
```python
# Sell weekend hedges from Friday

for hedge in get_weekend_hedges():
    order = create_market_order(
        symbol=hedge.symbol,
        qty=hedge.contracts,
        side='sell',
        time_in_force='OPG'  # Opening auction
    )
    
    fill = execute_order(order)
    log_trade(fill)
    update_positions(fill)
```

**Monitor:**
- Fill prices vs expected
- Slippage on hedge exits
- Any execution failures

#### 10:00 AM - Weekly Put Sales (Monday Only)

**2. Execute Strategy Trades**
```python
# Monday: Sell weekly puts on all assets

if today.weekday() == 0:  # Monday
    for asset in ['SPY', 'QQQ', 'DIA', 'IWM']:
        # Calculate strike at 30% delta
        current_price = get_price(asset)
        iv = get_iv(asset)
        strike = calculate_strike(current_price, iv, delta=-0.30, dte=14)
        
        # Calculate position size
        contracts = calculate_position_size(asset, capital_allocation)
        
        # Execute (limit order slightly above fair value)
        order = create_limit_order(
            symbol=create_option_symbol(asset, strike, 'P', expiration),
            qty=contracts,
            side='sell',
            limit_price=fair_value * 0.98  # Accept 2% worse than fair
        )
        
        fill = execute_order(order)
        
        # Log premium collected
        log_premium_collected(asset, fill.premium * contracts * 100)
```

**Monitor:**
- All orders filled (or investigate)
- Fill prices acceptable
- Positions created in system
- Risk limits still met

#### Throughout Day - Monitoring

**3. Continuous Monitoring** (Automated)

**Every 5 Minutes:**
- Portfolio NAV
- Delta exposure
- P&L today
- Margin utilization
- System health

**Alerts Trigger On:**
- Drawdown >5% today
- Unusual P&L swing (>2% in 15 min)
- Margin utilization >60%
- Failed order execution
- API errors

**4. Manual Checks** (Hourly)

**11:00 AM, 1:00 PM, 3:00 PM:**
- Review open positions
- Check for any assignments
- Monitor volatility (VIX)
- Review news for major events
- Verify hedges still appropriate

#### 3:45 PM - Pre-Close Procedures

**5. Evening Hedge Preparation**
```python
# Calculate if overnight hedges needed

day_of_week = today.weekday()

if day_of_week < 4:  # Mon-Thu
    # Weeknight hedges (1-2 DTE)
    net_delta = calculate_total_net_delta()
    
    if net_delta > 0:
        for asset in ['SPY', 'QQQ', 'DIA', 'IWM']:
            asset_delta = calculate_asset_delta(asset)
            if asset_delta > 0:
                prepare_overnight_hedge(asset, dte=2)

elif day_of_week == 4:  # Friday
    # Weekend hedges (7-14 DTE or SQQQ calls)
    prepare_weekend_hedges()
```

#### 3:55 PM - Execute MOC Orders

**6. Market-On-Close Execution**
```python
# Submit all hedge orders for closing auction

for hedge_order in overnight_hedges:
    order = create_market_order(
        symbol=hedge_order.symbol,
        qty=hedge_order.contracts,
        side='buy',  # Long protective puts
        time_in_force='MOC'  # Market-on-close
    )
    
    submit_order(order)
```

**Monitor:**
- All hedges submitted before 3:59 PM deadline
- No rejected orders
- Fill confirmations received

### Post-Market Routine (4:00 PM - 6:00 PM ET)

#### 4:05 PM - End-of-Day Reconciliation

**1. Verify All Fills**
```python
# Ensure all orders from today filled or cancelled

pending_orders = get_pending_orders()

for order in pending_orders:
    if order.status == 'pending':
        # Should not happen for MOO/MOC orders
        alert_ops(f"Unfilled order: {order.id}")
        investigate_order(order)
```

#### 4:15 PM - NAV Calculation

**2. Calculate Official NAV**
```python
# Mark all positions to market using 4:00 PM closing prices

def calculate_daily_nav():
    """
    Official NAV calculation for investor statements
    Administrator verifies this independently
    """
    nav = cash_balance
    
    # Mark equity positions
    for asset, shares in equity_positions.items():
        closing_price = get_closing_price(asset)
        nav += shares * closing_price
    
    # Mark options positions
    for option in option_positions:
        if option.expiration == today:
            # Expired - use intrinsic value
            value = max(0, option.intrinsic_value())
        else:
            # Mark to model
            value = black_scholes_price(option)
        
        if option.is_short:
            nav -= value
        else:
            nav += value
    
    # Record NAV
    db.insert_nav(date=today, nav=nav, methodology='Black-Scholes')
    
    # Send to administrator for verification
    send_to_administrator(nav, positions)
    
    return nav
```

**✓ Verify:**
- NAV calculated
- Stored in database
- Sent to administrator
- Matches expected range
- No calculation errors

#### 4:30 PM - Position Snapshot

**3. Daily Position Report**
```python
# Snapshot all positions for tomorrow's reconciliation

position_snapshot = {
    'date': today,
    'nav': today_nav,
    'cash': cash_balance,
    'shares': {asset: qty for asset, qty in shares.items()},
    'short_puts': list_short_puts(),
    'covered_calls': list_covered_calls(),
    'hedges': list_hedges(),
    'delta_exposure': calculate_delta_by_asset(),
    'margin_used': get_margin_used(),
    'pending_expirations': get_positions_expiring_soon(days=7)
}

# Save to database
db.save_position_snapshot(position_snapshot)

# Generate daily report PDF
generate_daily_report(position_snapshot)
```

#### 5:00 PM - Day Review

**4. Performance Review**
```python
# Calculate and log today's performance

daily_pnl = today_nav - yesterday_nav
daily_return = daily_pnl / yesterday_nav

performance_metrics = {
    'date': today,
    'pnl': daily_pnl,
    'return_pct': daily_return,
    'sharpe_30d': calculate_rolling_sharpe(days=30),
    'drawdown': calculate_current_drawdown(),
    'trades_executed': count_todays_trades(),
    'premium_collected': sum_premium_collected_today(),
    'hedge_cost': sum_hedge_costs_today()
}

log_daily_performance(performance_metrics)

# Alert if unusual
if abs(daily_return) > 0.03:  # >3% move
    alert_manager(f"Large daily move: {daily_return:.2%}")
```

#### 5:30 PM - Administrator Sync

**5. Send Data to Administrator**

**Package Includes:**
- Today's trades (CSV export)
- Position snapshot
- NAV calculation worksheet
- Cash reconciliation
- Any corporate actions affecting positions

**Administrator Will:**
- Independently calculate NAV
- Verify our calculation (usually matches)
- Flag any discrepancies
- Publish official NAV (by 6 PM)

#### 6:00 PM - End of Day

**6. Final Checks**
```
Daily Checklist:
[ ] All trades executed as planned
[ ] All fills reconciled
[ ] NAV calculated and verified
[ ] Position snapshot saved
[ ] Administrator received data
[ ] No open alerts or issues
[ ] Tomorrow's plan ready
[ ] Logs reviewed for errors
```

**If All Clear:**
- System in good state for overnight
- Hedges in place (if needed)
- Ready for tomorrow

**If Issues:**
- Document all problems
- Create tickets for resolution
- Escalate if impacts trading
- Plan fixes for tomorrow

---

## Weekly Operations

### Monday Morning (in addition to daily routine)

**Weekly Put Sales**
- Execute primary strategy trades
- Sell puts on all 4 assets
- Typical: 4 contracts sold
- Premium: $2,000-5,000 collected

**Weekend Review**
- Analyze weekend hedge performance
- Calculate SQQQ call P&L
- Review any news/events affecting holdings
- Plan week ahead

### Mid-Week (Wednesday)

**Position Review**
```python
# Review all open positions mid-week

def weekly_position_review():
    for asset in ['SPY', 'QQQ', 'DIA', 'IWM']:
        # Check short puts
        for put in get_short_puts(asset):
            days_to_exp = (put.expiration - today).days
            current_price = get_price(asset)
            profit_pct = calculate_profit_pct(put, current_price)
            
            # Close if at 50% profit target
            if profit_pct >= 0.50:
                close_position(put, reason="PROFIT_TARGET")
            
            # Alert if getting challenged
            if current_price < put.strike * 1.02:  # Within 2%
                alert_manager(f"{asset} put may be assigned")
```

**✓ Check:**
- Any positions at profit target → close
- Any positions getting challenged → monitor closely
- Any hedges needing adjustment → update
- Cash balance adequate → no margin issues

### Friday Afternoon

**Weekend Hedge Execution**
```python
# Calculate and execute weekend protection

def execute_weekend_hedges():
    # Option 1: SQQQ calls (if using)
    for asset in ['QQQ']:
        delta = get_net_delta(asset)
        if delta > 0:
            buy_sqqq_calls(contracts=delta/300)  # SQQQ is 3x inverse
    
    # Option 2: Longer-dated protective puts
    for asset in ['SPY', 'QQQ', 'DIA', 'IWM']:
        delta = get_net_delta(asset)
        if delta > 0:
            buy_protective_put(asset, dte=14, contracts=delta/100)
```

---

## Weekly Operations

### Every Monday - Strategy Review

**Performance Review** (30 minutes)
```
Review Past Week:
- Total P&L
- Premium collected
- Hedge costs
- Assignments handled
- Closed positions
- Win rate
```

**Questions to Answer:**
- Are we meeting return targets?
- Is Sharpe ratio on track?
- Any unusual losses?
- Hedge effectiveness good?
- Any operational issues?

### Every Wednesday - Risk Review

**Risk Metrics** (20 minutes)
```
Calculate and Review:
- Current drawdown from peak
- Rolling 30-day Sharpe
- Delta exposure by asset
- Margin utilization trend
- VaR (Value at Risk)
- Stress test scenarios
```

**Action Items:**
- If drawdown >5%: Review circuit breaker status
- If Sharpe <1.0: Analyze what's wrong
- If margin >60%: Consider deleveraging
- If hedges expensive: Review effectiveness

### Every Friday - Week Ahead Planning

**Preview Next Week** (15 minutes)
```
Check Calendar For:
- Market holidays (adjust hedging)
- Major earnings releases (NVDA, MSFT, AAPL, etc.)
- Fed meetings or economic data
- Options expirations
- Month-end (if applicable)
```

**Prepare:**
- Adjust hedge schedule for events
- Plan for expiration management
- Anticipate cash flows
- Update trading parameters if needed

---

## Monthly Operations

### Month-End (Last Business Day + 5 Days)

#### Day 1: Month-End Close

**1. Final Trades**
- Execute any month-end rebalancing
- Close any positions being rolled
- Ensure cash for upcoming redemptions
- Position for month ahead

**2. Month-End Snapshot**
```python
# Comprehensive end-of-month report

month_end_package = {
    'final_nav': calculate_month_end_nav(),
    'monthly_return': calculate_monthly_return(),
    'positions': get_all_positions(),
    'trades_this_month': get_monthly_trades(),
    'premium_collected': sum_monthly_premium(),
    'hedge_pnl': sum_monthly_hedge_pnl(),
    'fees_accrued': calculate_fees()
}

save_month_end_package(month_end_package)
```

#### Day 2-3: Reporting Preparation

**3. Performance Calculation**
```python
# Calculate all metrics for investor reports

metrics = {
    'mtd_return': (final_nav - start_nav) / start_nav,
    'qtd_return': calculate_qtd_return(),
    'ytd_return': calculate_ytd_return(),
    'since_inception': calculate_since_inception(),
    'sharpe_ratio': calculate_sharpe(period='monthly'),
    'sortino_ratio': calculate_sortino(),
    'max_dd_month': calculate_month_max_dd(),
    'volatility': calculate_monthly_vol(),
    'win_rate': calculate_monthly_win_rate()
}
```

**4. Fee Calculation**
```python
# Calculate fees for the month

def calculate_monthly_fees(investor_accounts):
    fees = {}
    
    for account in investor_accounts:
        # Management fee (2% annually = ~0.167% monthly)
        mgmt_fee = account.nav * 0.02 / 12
        
        # Performance fee (accrued, paid annually)
        if account.nav > account.high_water_mark:
            profit = account.nav - account.high_water_mark
            hurdle = account.high_water_mark * 0.06 / 12
            perf_fee = max(0, profit - hurdle) * 0.20
        else:
            perf_fee = 0
        
        fees[account.id] = {
            'management': mgmt_fee,
            'performance': perf_fee,
            'total': mgmt_fee + perf_fee
        }
    
    return fees
```

#### Day 4: Report Generation

**5. Create Investor Reports**
```python
# Generate monthly reports for all investors

for investor in get_all_investors():
    report = generate_monthly_report(
        investor=investor,
        month=last_month,
        performance=metrics,
        positions=month_end_positions,
        fees=investor_fees[investor.id],
        template='monthly_report_template.html'
    )
    
    # Save PDF
    pdf_path = f"reports/{investor.id}/monthly_{last_month}.pdf"
    save_pdf(report, pdf_path)
    
    # Upload to investor portal
    upload_to_portal(investor.id, pdf_path)
    
    # Email notification
    send_email(
        to=investor.email,
        subject=f"Monthly Report - {last_month}",
        body="Your monthly report is ready in the investor portal",
        attachments=[pdf_path]
    )
```

#### Day 5: Distribution & Review

**6. Final Review**
- CCO reviews all reports
- Check for errors or typos
- Verify performance calculations
- Ensure all disclaimers present

**7. Distribution**
- Upload to investor portal (primary)
- Email notifications sent
- Hard copies mailed if requested
- File copies for records

### Mid-Month Operations

#### 15th of Month: Reconciliation

**1. Broker Statement Reconciliation**
```
Match Our Records to Broker:
[ ] All trades match (ours vs broker statement)
[ ] All fees accounted for
[ ] Cash balance matches
[ ] Position quantities match
[ ] Any discrepancies investigated and resolved
```

**2. Administrator Reconciliation**
```
Match Our NAV to Administrator NAV:
[ ] Daily NAVs reviewed
[ ] Any differences <0.01% (acceptable)
[ ] Large differences (>0.1%) investigated
[ ] Methodology consistent
[ ] Pricing sources documented
```

#### 20th of Month: Forward Planning

**3. Next Month Preview**
```
Plan Ahead:
- Forecast cash needs (redemptions?)
- Anticipate expirations
- Check for major events
- Review strategy performance
- Plan any adjustments
```

---

## Quarterly Operations

### End of Quarter (+ 10 Days)

#### Expanded Reporting

**1. Quarterly Investor Letter**

**Contents:**
- Detailed performance review
- Strategy commentary
- Market outlook
- Position highlights
- Risk review
- Q&A section

**Length**: 3-5 pages

**Tone**: Professional but accessible

**Distribution**: Email PDF, post to portal

#### 2. Regulatory Filings

**Form PF** (if required - >$150M AUM):
- Due within 60 days of quarter-end (if quarterly filer)
- Complete via SEC's EDGAR system
- Extensive portfolio detail required

**Form ADV Updates** (if material changes):
- Update AUM
- Update client count
- Disclose any disciplinary events
- Note strategy changes

#### 3. Liquidity Management

**Redemption Processing:**
```python
# Handle quarterly redemptions

def process_redemptions(redemption_requests):
    total_redemptions = sum(r.amount for r in redemption_requests)
    total_aum = get_total_aum()
    
    # Check gate (25% quarterly max)
    if total_redemptions > total_aum * 0.25:
        # Pro-rate redemptions
        scaling_factor = (total_aum * 0.25) / total_redemptions
        for request in redemption_requests:
            request.amount_approved = request.amount * scaling_factor
    
    # Plan liquidations
    liquidation_plan = create_liquidation_plan(total_redemptions)
    
    # Execute over 5-10 days to minimize market impact
    execute_liquidation(liquidation_plan)
```

#### 4. Performance Attribution

**Detailed Analysis:**
- Return by asset (SPY vs QQQ vs DIA vs IWM)
- Return by strategy component (puts, calls, hedges)
- Transaction cost analysis
- Slippage vs theoretical
- Risk-adjusted metrics

**Purpose**:
- Understand what's working
- Identify improvements
- Report to investors
- Inform strategy adjustments

---

## Annual Operations

### Year-End Close (December 31 + 120 Days)

#### January: Audit Preparation

**1. Close Books**
- Final NAV for year
- Reconcile all accounts
- Verify all trades recorded
- Close out accruals
- Calculate final fees

**2. Prepare for Auditors**
```
Audit Workpaper Package:
- General ledger
- Trade blotter (all year)
- Position schedules (month-end each month)
- Fee calculations
- Expense allocations
- Related party transactions
- Bank reconciliations
- Broker statements
```

**3. Auditor Kickoff**
- Provide opening balances
- Deliver preliminary financials
- Schedule fieldwork dates
- Assign staff for questions

#### February: Audit Fieldwork

**Auditor Activities:**
- Test internal controls
- Sample trades for validation
- Verify NAV calculations
- Review fee computations
- Examine expense allocations
- Confirm positions with broker

**Our Activities:**
- Answer questions promptly
- Provide requested documents
- Explain any unusual items
- Demonstrate controls
- Fix any issues discovered

#### March: Audit Finalization

**Deliverables:**
- Audited financial statements
- Management letter (if deficiencies)
- Tax returns (Form 1065)
- K-1s for all investors

**Timeline:**
- Draft financials: March 15
- Final financials: March 31
- K-1s to investors: March 15 (tax law deadline)

#### April: Distribution & Filing

**1. Distribute Audited Financials**
- Upload to investor portal
- Email to all LPs
- File copy in records
- Post on website (if public)

**2. Annual Form ADV Update**
- Update AUM (as of fiscal year-end)
- Update any material changes
- File within 90 days of fiscal year-end
- Deliver brochure to clients within 120 days

**3. Annual Compliance Review**
- CCO conducts comprehensive review
- Test all policies and procedures
- Interview employees
- Document findings
- Update policies for next year
- Report to management

---

## Exception Handling

### When Things Go Wrong

#### Trade Execution Failure

**Scenario**: Order submitted but not filled

**Procedure:**
1. Check order status in broker system
2. Determine reason for failure (rejected, insufficient margin, etc.)
3. Document in trade log
4. Notify risk manager
5. Execute manually if critical
6. Investigate root cause
7. Update systems to prevent recurrence

**Documentation:**
- What happened
- Why it happened
- Impact on portfolio
- How resolved
- Prevention measures

#### Assignment Without Sufficient Cash

**Scenario**: Put assigned but insufficient cash to purchase shares

**Procedure:**
1. IMMEDIATE: Contact broker
2. Options: (a) Request extension, (b) Liquidate other positions, (c) Cover with margin
3. Document exception
4. Notify compliance
5. Review cash management procedures
6. Prevent recurrence (better forecasting)

**This is SERIOUS** - could violate margin requirements

#### NAV Calculation Error

**Scenario**: Discovered error in NAV after distribution

**Procedure:**
1. Quantify error magnitude
2. Determine affected periods
3. Notify administrator immediately
4. Recalculate corrected NAV
5. Disclose to investors (if material >0.5%)
6. Adjust investor accounts if needed
7. Document cause and prevention
8. May need to notify SEC if material

**Definition of Material**: >0.5% impact on NAV

#### Cybersecurity Incident

**Scenario**: Suspicious activity or confirmed breach

**Procedure:**
1. **Immediately**: Isolate affected systems
2. **Within 1 hour**: Notify senior management
3. **Within 4 hours**: Engage cybersecurity firm
4. **Within 24 hours**: Assess scope of breach
5. **Within 48 hours**: Notify affected parties if PII compromised
6. **Within 72 hours**: Report to SEC if material
7. **Ongoing**: Remediate, document, prevent

---

## Investor Servicing

### New Investor Onboarding

**Day 1: Subscription Received**
```
Onboarding Checklist:
[ ] Subscription agreement received
[ ] AML/KYC completed
[ ] Accredited investor verification
[ ] Wire instructions provided
[ ] Investor portal access created
[ ] Welcome email sent
```

**Day 5: Capital Received**
```
Post-Funding:
[ ] Wire received and verified
[ ] Capital credited to LP account
[ ] Ownership % calculated
[ ] Added to investor list
[ ] Notify administrator
[ ] Begin reporting
```

### Ongoing Investor Communications

**Proactive Communications:**
- Monthly reports (5th business day)
- Quarterly letters (10th business day)
- Material event notifications (as needed)
- Annual meeting invitation

**Responsive Communications:**
- Answer inquiries within 24 hours
- Provide requested information
- Schedule calls as needed
- Maintain professional relationship

### Redemption Processing

**Day 1: Notice Received (45 days before quarter-end)**
```
Redemption Workflow:
[ ] Log redemption request
[ ] Verify proper notice period
[ ] Confirm share count/amount
[ ] Check for lock-up violations
[ ] Apply any penalties (early redemption fee)
[ ] Plan liquidation if needed
```

**Quarter-End: Calculate Redemption Amount**
```
Redemption Calculation:
= (Investor's Units / Total Units) × Quarter-End NAV
Less: Redemption fees (if applicable)
Less: Management fees unpaid
= Net Redemption Amount
```

**15 Days After Quarter-End: Wire Payment**
```
Payment Process:
[ ] Calculate final amount
[ ] Obtain wire instructions
[ ] Submit for approval
[ ] Execute wire
[ ] Send confirmation to investor
[ ] Update investor records
[ ] Notify administrator
```

---

## Technology Operations

### System Maintenance

#### Daily Backups

**Automated** (cron job at 2:00 AM ET):
```bash
#!/bin/bash
# Daily backup script

# Backup PostgreSQL
pg_dump -U trading trading_db | gzip > \
    /backups/daily/trading_db_$(date +%Y%m%d).sql.gz

# Backup Redis
redis-cli --rdb /backups/daily/dump_$(date +%Y%m%d).rdb

# Sync to S3
aws s3 sync /backups/daily/ s3://hedge-fund-backups/daily/ --delete

# Keep only last 7 days locally
find /backups/daily/ -mtime +7 -delete
```

#### Weekly Maintenance

**Sunday 2:00 AM** (market closed):
```
Maintenance Window:
- Apply security patches
- Update Python packages
- Database optimization (VACUUM, ANALYZE)
- Clear old logs (>90 days)
- Test disaster recovery
- Review system performance
```

#### Monthly Updates

**First Sunday of Month**:
```
Scheduled Maintenance:
- Full system backup (monthly archive)
- API key rotation
- Certificate renewal (if needed)
- Dependency updates
- Security scan
- Performance tuning
```

### Monitoring

**24/7 Monitoring** (Automated):
- System health (5-minute checks)
- API availability
- Database performance
- Error rates
- Disk space
- Memory usage
- CPU utilization

**Alert Escalation:**
- Critical: Immediate phone call + SMS
- High: SMS within 5 minutes
- Medium: Email within 30 minutes
- Low: Daily digest email

---

## Disaster Recovery Drills

### Quarterly DR Test

**Scenario 1: Database Failure**
```
Drill Procedure:
1. Simulate database crash (stop service)
2. Team must:
   - Detect failure (should be automatic)
   - Activate backup database
   - Verify data integrity
   - Resume trading operations
3. Time to recovery: Target <15 minutes
4. Document results
5. Improve procedures based on learnings
```

**Scenario 2: Complete System Failure**
```
Drill Procedure:
1. Simulate total AWS outage
2. Team must:
   - Activate DR region
   - Restore from backups
   - Verify all systems operational
   - Execute manual trade if needed
3. Time to recovery: Target <2 hours
4. Document results
```

**Scenario 3: Broker API Failure**
```
Drill Procedure:
1. Simulate Alpaca API down
2. Team must:
   - Detect failure
   - Failover to Interactive Brokers
   - Re-submit pending orders
   - Verify execution
3. Time to recovery: Target <5 minutes
4. Document results
```

---

## Documentation Standards

### Trade Documentation

**Every Trade Must Include:**
- Date and time (to the second)
- Asset and contract details
- Quantity and price
- Broker used
- Rationale (strategy component)
- Expected vs actual fill
- Slippage calculation
- P&L attribution

### Incident Documentation

**Template:**
```markdown
## Incident Report

**Incident ID**: INC-2024-001
**Date/Time**: 2024-01-15 10:32:45 ET
**Severity**: High
**Status**: Resolved

**Summary**:
[What happened in 2-3 sentences]

**Impact**:
- Trading halted for 12 minutes
- 3 orders delayed
- No financial loss
- Client impact: None

**Root Cause**:
[Why it happened]

**Resolution**:
[How it was fixed]

**Prevention**:
- [Action item 1]
- [Action item 2]

**Responsible**: [Name]
**Reviewed By**: [CCO Name]
**Date Closed**: 2024-01-15
```

### Change Documentation

**For Any System Changes:**
- Change request form
- Business justification
- Technical design
- Testing plan
- Rollback procedure
- Approval signatures
- Implementation date
- Post-implementation review

---

## Staff Responsibilities

### Portfolio Manager (You)

**Daily:**
- Review overnight performance
- Approve trading plan
- Monitor positions
- Respond to alerts

**Weekly:**
- Strategy review meeting
- Risk assessment
- Adjust parameters if needed
- Investor communications

**Monthly:**
- Performance commentary
- Investor calls
- Strategy optimization
- Compliance review

### Operations Manager

**Daily:**
- Execute operations checklist
- Reconcile positions
- Submit NAV to administrator
- Resolve operational issues

**Weekly:**
- Position review
- Cash flow forecasting
- Vendor management
- Process improvements

**Monthly:**
- Generate investor reports
- Calculate fees
- Coordinate with administrator
- Month-end closing

### Compliance Officer (CCO)

**Daily:**
- Monitor alerts for violations
- Review error logs
- Approve marketing materials

**Weekly:**
- Personal trading review
- Policy enforcement
- Training delivery

**Monthly:**
- Compliance metrics
- Policy updates
- Regulatory deadline tracking

**Quarterly:**
- Compliance testing
- Advertising review
- Employee certifications

**Annually:**
- Annual review
- Policy updates
- Regulatory filings
- Training program

---

## Emergency Procedures

### Market Crash (>5% Down Day)

**Immediate Actions:**
1. Verify all hedges executed properly
2. Calculate current exposure
3. Check margin requirements
4. Prepare for potential assignments
5. Ensure sufficient cash

**Within 1 Hour:**
1. Calculate portfolio impact
2. Assess if circuit breakers should trigger
3. Communicate with investors (if severe)
4. Document situation

**End of Day:**
1. Review all positions
2. Strengthen hedges if needed
3. Plan for next day
4. Report to compliance

### Flash Crash / Trading Halt

**During Halt:**
1. Do NOT submit new orders
2. Cancel any pending orders
3. Calculate exposure if halt permanent
4. Prepare manual execution plan

**When Trading Resumes:**
1. Verify prices reasonable
2. Reassess hedge needs
3. Execute carefully (avoid panic)
4. Document all decisions

### Regulatory Inquiry

**If SEC Calls/Emails:**
1. DO NOT panic
2. Document inquiry details
3. Notify legal counsel immediately
4. Gather requested information
5. Respond by deadline
6. Be professional and cooperative
7. Do NOT hide anything

**Never:**
- Lie to regulators
- Destroy documents
- Obstruct investigation
- Make false statements
- Hide material facts

---

## Standard Operating Times

### Daily Schedule

```
7:00 AM  - Pre-market system check
8:00 AM  - Market data validation
8:30 AM  - Position reconciliation
9:00 AM  - Pre-market risk check
9:15 AM  - Daily trading plan
9:30 AM  - Market open (execute MOO orders)
10:00 AM - Strategy trades (Monday puts)
12:00 PM - Mid-day check
3:00 PM  - Profit-taking reviews
3:45 PM  - Prepare closing hedges
3:55 PM  - Execute MOC orders
4:00 PM  - Market close
4:15 PM  - NAV calculation
4:30 PM  - Position snapshot
5:00 PM  - Performance review
5:30 PM  - Send to administrator
6:00 PM  - End of day checklist
```

### Weekly Schedule

```
Monday    - Put sales, weekend review
Tuesday   - Monitor positions
Wednesday - Mid-week risk review
Thursday  - Monitor positions, close profits
Friday    - Weekend hedge execution, week review
```

### Monthly Schedule

```
Month-End Day      - Final trades, closing
ME + 1 to 3        - Report preparation
ME + 4             - Report generation
ME + 5             - Distribution to investors
15th of Month      - Mid-month reconciliation
20th of Month      - Forward planning
```

---

## Conclusion

**Operations must be:**
- ✓ **Systematic** - Follow checklists religiously
- ✓ **Documented** - Record everything
- ✓ **Verified** - Cross-check all calculations
- ✓ **Timely** - Meet all deadlines
- ✓ **Professional** - Institutional-grade standards

**Operational excellence builds trust and enables scale.**

**Poor operations destroys even the best strategy.**

---

*Document Version 1.0*  
*Last Updated: October 2025*  
*Operations Manual - Confidential*

