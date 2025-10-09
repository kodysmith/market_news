# Quick Start with SQLite

## ✅ What's Set Up

Your system is now configured to use **SQLite** instead of PostgreSQL:

- ✅ Database created: `./data/hedgefund.db`
- ✅ All tables created (trades, positions, nav_history, audit_log)
- ✅ No installation required!

## 🚀 Start Trading Now

### 1. Start the Daemon

```bash
cd /mnt/4tb/stock_scanner/market_news/HedgeFund
./start_daemon.sh supervisor
```

### 2. Check Status

```bash
./check_daemon.sh
```

### 3. View Logs

```bash
tail -f logs/daemon_stdout.log
```

## 📊 Check Your Data

Since the full dashboard needs PostgreSQL updates, use SQLite directly:

```bash
# View trades
sqlite3 data/hedgefund.db "SELECT * FROM trades ORDER BY timestamp DESC LIMIT 10;"

# View NAV history
sqlite3 data/hedgefund.db "SELECT timestamp, nav, daily_return FROM nav_history ORDER BY timestamp DESC LIMIT 10;"

# View positions
sqlite3 data/hedgefund.db "SELECT * FROM positions WHERE status='open';"

# Count total trades
sqlite3 data/hedgefund.db "SELECT COUNT(*) as total_trades FROM trades;"
```

## 🎯 What Works Now

- ✅ **Trading Daemon**: Fully functional
  - Monitors market hours
  - Executes trades at 9:30 AM ET
  - Logs to database
  - Auto-restarts on crashes

- ✅ **Data Storage**: All in SQLite
  - Trades
  - Positions  
  - NAV history
  - Audit logs

- ⚠️ **Dashboard**: Needs update for SQLite
  - Coming in next iteration
  - Use SQL queries above for now

## 💡 Daily Workflow

**Morning (9:30 AM ET)**:
1. Daemon automatically executes trades
2. Check logs: `tail -f logs/daemon_stdout.log`
3. View trades: `sqlite3 data/hedgefund.db "SELECT * FROM trades WHERE DATE(timestamp) = DATE('now');"`

**Evening**:
1. Check NAV: `sqlite3 data/hedgefund.db "SELECT nav, daily_return FROM nav_history ORDER BY timestamp DESC LIMIT 1;"`
2. Review logs for any errors
3. Verify daemon still running: `./check_daemon.sh`

## 🔄 Upgrading to PostgreSQL Later

If you want the full dashboard, install PostgreSQL:

```bash
# Install PostgreSQL
sudo apt-get update
sudo apt-get install postgresql postgresql-contrib

# Run setup
./setup_database.sh

# Update .env
nano .env  # Change USE_SQLITE=true to USE_SQLITE=false

# Restart daemon
./stop_daemon.sh
./start_daemon.sh supervisor
```

## ✅ You're Ready!

The core trading system is fully operational with SQLite. Start the daemon and let it trade!

```bash
./start_daemon.sh supervisor
./check_daemon.sh
tail -f logs/daemon_stdout.log
```

Trading will begin on the next market day at 9:30 AM ET! 🚀
