# 🏦 Complete Hedge Fund System Guide

## System Overview

You now have a **fully operational, institutional-grade hedge fund trading system**:

```
┌─────────────────────────────────────────────────┐
│         24/7 PAPER TRADING SYSTEM                │
│                                                  │
│  ┌──────────────┐         ┌──────────────┐     │
│  │   Trading    │────────▶│  PostgreSQL  │     │
│  │   Daemon     │         │   Database   │     │
│  └──────────────┘         └──────────────┘     │
│         │                                        │
│         │                                        │
│  ┌──────────────┐         ┌──────────────┐     │
│  │     Web      │────────▶│     API      │     │
│  │  Dashboard   │         │  Endpoints   │     │
│  └──────────────┘         └──────────────┘     │
└─────────────────────────────────────────────────┘
```

---

## ✅ What You Have

### 1. **Core Trading Engine** ✅
- Multi-asset adaptive wheel strategy (SPY, QQQ, DIA, IWM)
- Sell puts → Wheel into shares → Covered calls
- Options pricing with Black-Scholes
- Risk management & circuit breakers
- Position tracking & NAV calculation
- Full audit trail in PostgreSQL

### 2. **24/7 Daemon** ✅
- Runs continuously, 24/7
- Executes daily trades at 9:30 AM ET
- Auto-restarts on crashes
- Survives system reboots
- Health monitoring
- Comprehensive logging

### 3. **Real-Time Dashboard** ✅
- Web-based monitoring (http://localhost:5000)
- Live NAV and performance metrics
- System health visualization
- Recent trades & positions
- Interactive charts
- Auto-refreshes every 30 seconds

---

## 🚀 Getting Started

### Complete Startup Sequence

```bash
cd /mnt/4tb/stock_scanner/market_news/HedgeFund

# 1. Start the trading daemon (supervisor mode)
./start_daemon.sh supervisor

# 2. Start the dashboard (separate terminal)
./start_dashboard.sh

# 3. Check everything is running
./check_daemon.sh

# 4. Open dashboard in browser
# Navigate to: http://localhost:5000
```

---

## 📋 Daily Operations

### Morning Routine (8:00 AM)

1. **Check Dashboard**: Open http://localhost:5000
2. **Verify System Health**: Look for green indicator
3. **Review Overnight News**: (Coming in Phase 2)
4. **Confirm Trading Plan**: 9:30 AM execution

### Market Open (9:30 AM)

**Automatic**:
- Daemon detects market open
- Generates put signals (if Monday)
- Executes trades via Alpaca
- Updates positions in database
- Logs all activity

**Your Actions**:
- Monitor dashboard for execution
- Verify trades appear in dashboard table
- Check Alpaca paper account for confirmation

### End of Day (4:00 PM)

**Automatic**:
- Calculates end-of-day NAV
- Updates portfolio snapshot
- Marks-to-market all positions
- Calculates daily return

**Your Actions**:
- Review day's performance on dashboard
- Check any errors in logs
- Verify P&L matches expectations

### Evening (6:00 PM)

**Automatic** (when implemented):
- Generates overnight hedges
- Adjusts for next day's risk

---

## 🎛️ Control Commands

### Daemon Control

```bash
# Start daemon
./start_daemon.sh dev          # Foreground (testing)
./start_daemon.sh supervisor   # Background with auto-restart
sudo ./start_daemon.sh systemd # System service

# Stop daemon
./stop_daemon.sh

# Check status
./check_daemon.sh

# View logs
tail -f logs/daemon.log
```

### Dashboard Control

```bash
# Start dashboard
./start_dashboard.sh           # Default port 5000
./start_dashboard.sh 8080      # Custom port

# Stop dashboard (if in background)
pkill -f dashboard_server

# View dashboard logs
tail -f logs/dashboard.out
```

---

## 📊 Monitoring & Observability

### Dashboard URL
```
http://localhost:5000
```

### Key Metrics to Watch

| Metric | What It Means | Target |
|--------|---------------|---------|
| **NAV** | Total portfolio value | Growing steadily |
| **Daily Return** | Today's performance | Positive bias |
| **Sharpe Ratio** | Risk-adjusted return | > 1.0 |
| **Max Drawdown** | Worst peak-to-trough | < 15% |
| **Open Positions** | Active trades | 4-12 positions |
| **Capital Deployed** | % of cash used | 60-80% |

### System Health

- **CPU**: Should be < 20% normally
- **Memory**: Should be < 50%
- **Disk**: Keep < 80%
- **Daemon**: Must show "Running"
- **Database**: Must show "Connected"

---

## 🗂️ File Structure

```
HedgeFund/
├── config/
│   ├── config.yaml               # Default config
│   ├── config.paper.yaml         # Paper trading config
│   └── config.production.yaml    # Production config
│
├── src/
│   ├── common/                   # Shared models & config
│   ├── pricing/                  # Options pricing
│   ├── data/                     # Market data service
│   ├── strategies/               # Signal generation
│   ├── risk/                     # Risk management
│   ├── execution/                # Order execution
│   ├── reporting/                # NAV & audit logging
│   ├── daemon/                   # 24/7 daemon
│   ├── dashboard/                # Web dashboard
│   └── main/                     # Trading engine
│
├── logs/                         # All system logs
│   ├── daemon.log               # Main daemon log
│   ├── daemon_stdout.log        # Supervisor output
│   └── dashboard.out            # Dashboard log
│
├── docs/                         # Documentation
│   ├── DEPLOYMENT_GUIDE.md
│   ├── DASHBOARD_GUIDE.md
│   └── [many more...]
│
├── .env                          # Environment variables
├── start_daemon.sh              # Start trading daemon
├── stop_daemon.sh               # Stop daemon
├── check_daemon.sh              # Check daemon status
├── start_dashboard.sh           # Start dashboard
└── COMPLETE_SYSTEM_GUIDE.md     # This file
```

---

## 🔧 Configuration

### Environment Variables (.env)

```bash
# Trading Mode
ENV=paper

# Alpaca (Paper Trading)
ALPACA_API_KEY=your_paper_key
ALPACA_SECRET_KEY=your_paper_secret
ALPACA_BASE_URL=https://paper-api.alpaca.markets

# Database
DATABASE_HOST=localhost
DATABASE_PORT=5432
DATABASE_USER=hedgefund
DATABASE_PASSWORD=your_secure_password
DATABASE_NAME=hedgefund_db

# Dashboard
DASHBOARD_PORT=5000

# Monitoring (optional)
SLACK_WEBHOOK=https://hooks.slack.com/...
EMAIL_ALERTS=you@example.com
```

### Strategy Parameters (config/config.paper.yaml)

```yaml
strategy:
  assets: ["SPY", "QQQ", "DIA", "IWM"]
  
  allocations:
    SPY: 0.40    # 40% of trades
    QQQ: 0.30    # 30% of trades
    DIA: 0.20    # 20% of trades
    IWM: 0.10    # 10% of trades
  
  put_delta_target: -0.30  # Sell 30-delta puts
  put_dte: 14              # 14 days to expiration
  position_size_pct: 0.10  # 10% per trade
  profit_target_pct: 0.50  # Take profit at 50%
  
  max_capital_deployed: 0.80     # Max 80% invested
  max_portfolio_delta: 100.0     # Max net delta
  
risk:
  max_position_size: 0.20        # 20% per position
  stop_loss_pct: 0.05           # 5% stop loss
  circuit_breaker_daily_loss: 0.03  # 3% daily loss triggers halt
```

---

## 🔍 Database Queries

Useful SQL queries for manual inspection:

### Check Latest NAV
```sql
SELECT timestamp, nav, daily_return, sharpe_ratio 
FROM nav_history 
ORDER BY timestamp DESC 
LIMIT 1;
```

### Today's Trades
```sql
SELECT timestamp, asset, action, quantity, strike, status 
FROM trades 
WHERE DATE(timestamp) = CURRENT_DATE 
ORDER BY timestamp DESC;
```

### Open Positions
```sql
SELECT symbol, quantity, cost_basis, current_price, unrealized_pnl 
FROM positions 
WHERE status = 'open' 
ORDER BY date_opened DESC;
```

### Performance Last 30 Days
```sql
SELECT 
  DATE(timestamp) as date,
  nav,
  daily_return,
  sharpe_ratio
FROM nav_history 
WHERE timestamp >= NOW() - INTERVAL '30 days'
ORDER BY timestamp DESC;
```

---

## 🚨 Alerts & Warnings

### Critical Alerts

**Daemon Stopped**:
```bash
./check_daemon.sh
# If stopped, restart:
./start_daemon.sh supervisor
```

**Database Connection Lost**:
```bash
sudo systemctl status postgresql
# If stopped:
sudo systemctl start postgresql
```

**Disk Space Low** (> 90%):
```bash
# Check disk usage
df -h

# Clean old logs
cd logs
find . -name "*.log.*" -mtime +30 -delete
```

**High CPU/Memory** (> 80%):
```bash
# Check process
top -p $(pgrep -f paper_trading_daemon)

# Restart if needed
./stop_daemon.sh
sleep 5
./start_daemon.sh supervisor
```

---

## 🛠️ Troubleshooting

### Issue: No Trades Executing

**Possible Causes**:

1. **Weekend/Holiday**: Check logs for "not a trading day"
2. **Outside trading window**: Trades execute 9:30-10:00 AM ET
3. **Already traded today**: Only one cycle per day
4. **Broker connection issue**: Check Alpaca API status
5. **Insufficient capital**: Check available cash

**Debug Steps**:
```bash
# Check daemon logs
tail -50 logs/daemon.log | grep "TRADING CYCLE"

# Check database
psql -U hedgefund -d hedgefund_db -c "
  SELECT COUNT(*) FROM trades 
  WHERE DATE(timestamp) = CURRENT_DATE;
"

# Verify system time
date  # Should show correct ET time

# Check Alpaca connection
curl -H "APCA-API-KEY-ID: $ALPACA_API_KEY" \
     -H "APCA-API-SECRET-KEY: $ALPACA_SECRET_KEY" \
     https://paper-api.alpaca.markets/v2/account
```

---

### Issue: Dashboard Shows No Data

**Debug Steps**:
```bash
# 1. Check if daemon is running
./check_daemon.sh

# 2. Check if database has data
psql -U hedgefund -d hedgefund_db -c "
  SELECT COUNT(*) FROM nav_history;
"

# 3. Check dashboard logs
tail -30 logs/dashboard.out

# 4. Test API endpoints
curl http://localhost:5000/api/health
curl http://localhost:5000/api/status
```

---

## 📈 Performance Tracking

### Weekly Review

Every Monday, review:

1. **Last Week's Performance**
   - Total return vs S&P 500
   - Win rate on trades
   - Sharpe ratio trend
   - Max drawdown

2. **Position Analysis**
   - Average holding period
   - Profitable vs losing trades
   - Best/worst assets

3. **System Health**
   - Uptime percentage
   - Any errors/failures
   - Execution quality

### Monthly Tasks

1. **Performance Report**
   ```bash
   python scripts/monthly_report.py  # Generate full report
   ```

2. **Database Maintenance**
   ```bash
   # Vacuum and analyze
   psql -U hedgefund -d hedgefund_db -c "VACUUM ANALYZE;"
   
   # Backup
   pg_dump -U hedgefund hedgefund_db | gzip > backup_$(date +%Y%m%d).sql.gz
   ```

3. **Log Rotation**
   ```bash
   cd logs
   gzip daemon.log.$(date -d "last month" +%Y%m)
   ```

4. **Strategy Review**
   - Adjust parameters if needed
   - Review market regime
   - Update asset allocations

---

## 🎯 Next Phase: Market Intelligence

Ready to add AI-powered news analysis?

See: `docs/IMPLEMENTATION_ROADMAP.md`

**Phase 2 Features**:
- 24/7 news monitoring
- LLM sentiment analysis
- Adaptive hedging based on risk events
- Comprehensive data warehouse for ML

**Timeline**: 2-4 weeks  
**Cost**: ~$20/month (FMP API)

---

## 📚 Documentation Index

| Document | Purpose |
|----------|---------|
| **DEPLOYMENT_GUIDE.md** | How to deploy daemon 24/7 |
| **DASHBOARD_GUIDE.md** | Dashboard usage & customization |
| **IMPLEMENTATION_ROADMAP.md** | Phase 2 market intelligence plan |
| **MarketIntelligenceIntegration.md** | News analysis design |
| **ContinuousOperationArchitecture.md** | 24/7 system architecture |
| **COMPLETE_SYSTEM_GUIDE.md** | This file - complete overview |

---

## ✅ Launch Checklist

Before going live with real money:

- [ ] Paper trading for 30+ days
- [ ] Verify all trades execute correctly
- [ ] Monitor system stability (99%+ uptime)
- [ ] Backtest shows positive Sharpe ratio
- [ ] Understand all risk parameters
- [ ] Have emergency shutdown plan
- [ ] Set up monitoring/alerts
- [ ] Review with legal/compliance
- [ ] Get proper licenses/registrations
- [ ] Have adequate capital ($100K+ minimum)

---

## 🎉 Congratulations!

You've built a **production-grade hedge fund trading system** with:

✅ Institutional architecture (12 modular components)  
✅ 24/7 autonomous operation  
✅ Real-time observability  
✅ Auto-recovery & resilience  
✅ Complete audit trail  
✅ Beautiful dashboard  
✅ Professional documentation  

**You're ready to trade!** 🚀

---

## 📞 Support

**Check System Status**:
```bash
./check_daemon.sh
```

**View All Logs**:
```bash
tail -f logs/daemon.log
```

**Access Dashboard**:
```
http://localhost:5000
```

**Database Console**:
```bash
psql -U hedgefund -d hedgefund_db
```

---

**Happy Trading!** 💰

