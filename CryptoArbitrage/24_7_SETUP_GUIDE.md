# 24/7 Paper Trading Service Setup

Complete guide for running crypto arbitrage paper trading as a reliable 24/7 background service.

---

## Prerequisites Checklist

### 1. ✅ Exchange API Keys (Read-Only)

You need API keys from exchanges for **read-only market data access**:

**Binance:**
- Go to: https://www.binance.com/en/my/settings/api-management
- Create API Key
- **Permissions:** Enable "Read" only (NO trading, NO withdrawals)
- Save API Key and Secret

**Coinbase (Pro):**
- Go to: https://pro.coinbase.com/profile/api
- Create API Key
- **Permissions:** Check "View" only
- Save API Key, Secret, and Passphrase

**Kraken:**
- Go to: https://www.kraken.com/u/security/api
- Create API Key
- **Permissions:** "Query Funds", "Query Open/Closed Orders" (read-only)
- Save API Key and Private Key

### 2. ✅ System Requirements

- **Linux** (Ubuntu/Debian recommended)
- **Python 3.11+**
- **Stable internet connection** (low latency preferred)
- **Disk space:** 1GB minimum (for logs and database)
- **RAM:** 512MB minimum, 1GB recommended

### 3. ✅ Network Requirements

- **Ports:** No inbound ports needed (outbound only)
- **Firewall:** Allow HTTPS (443) to:
  - api.binance.com
  - api.pro.coinbase.com
  - api.kraken.com
- **Latency:** < 100ms to exchanges recommended (use `ping` to test)

---

## Installation Steps

### Step 1: Clone and Setup Environment

```bash
cd /mnt/4tb/stock_scanner/market_news/CryptoArbitrage

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 2: Configure API Keys

```bash
# Copy environment template
cp env_template.txt .env

# Edit with your API keys
nano .env
```

**Edit `.env` file:**
```bash
CRYPTO_ENV=simulation  # Use 'simulation' for paper trading

# Binance (read-only keys)
BINANCE_API_KEY=your_binance_api_key_here
BINANCE_API_SECRET=your_binance_secret_here

# Coinbase Pro (read-only keys)
COINBASE_API_KEY=your_coinbase_key_here
COINBASE_API_SECRET=your_coinbase_secret_here
COINBASE_PASSPHRASE=your_coinbase_passphrase_here

# Kraken (read-only keys)
KRAKEN_API_KEY=your_kraken_key_here
KRAKEN_API_SECRET=your_kraken_secret_here

# Database
SQLITE_DB_PATH=./data/arbitrage.db

# Dashboard
DASHBOARD_PORT=5001
```

**Security:**
```bash
# Protect .env file
chmod 600 .env
```

### Step 3: Initialize Database

```bash
# Create directories
mkdir -p data logs

# Initialize database schema
python3 -c "from src.database.repository import ArbitrageRepository; ArbitrageRepository('./data/arbitrage.db')"
```

### Step 4: Test Configuration

```bash
# Quick 1-minute test
./run_paper_trading.sh 1
```

**Expected output:**
- ✅ Config loaded
- ✅ Database initialized  
- ✅ Connected to 3 exchanges
- 🎯 Found opportunities

If you see errors:
- Check API keys are correct
- Verify network connectivity
- Review logs: `tail -f logs/paper_trading.log`

---

## Setting Up as a Systemd Service

### Step 1: Create Service File

```bash
sudo nano /etc/systemd/system/crypto-arbitrage-paper.service
```

**Service file contents:**
```ini
[Unit]
Description=Crypto Arbitrage Paper Trading Service
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=YOUR_USERNAME
WorkingDirectory=/mnt/4tb/stock_scanner/market_news/CryptoArbitrage
Environment="CRYPTO_ENV=simulation"
Environment="PATH=/mnt/4tb/stock_scanner/market_news/CryptoArbitrage/venv/bin:/usr/bin:/bin"

# Run paper trading script
ExecStart=/mnt/4tb/stock_scanner/market_news/CryptoArbitrage/venv/bin/python3 scripts/paper_trading.py

# Auto-restart on failure
Restart=always
RestartSec=10

# Logging
StandardOutput=append:/mnt/4tb/stock_scanner/market_news/CryptoArbitrage/logs/service.log
StandardError=append:/mnt/4tb/stock_scanner/market_news/CryptoArbitrage/logs/service_error.log

# Resource limits (optional)
MemoryLimit=1G
CPUQuota=50%

[Install]
WantedBy=multi-user.target
```

**Replace `YOUR_USERNAME`** with your actual username:
```bash
whoami  # Shows your username
```

### Step 2: Enable and Start Service

```bash
# Reload systemd configuration
sudo systemctl daemon-reload

# Enable service (start on boot)
sudo systemctl enable crypto-arbitrage-paper.service

# Start service now
sudo systemctl start crypto-arbitrage-paper.service

# Check status
sudo systemctl status crypto-arbitrage-paper.service
```

**Expected output:**
```
● crypto-arbitrage-paper.service - Crypto Arbitrage Paper Trading Service
   Loaded: loaded (/etc/systemd/system/crypto-arbitrage-paper.service; enabled)
   Active: active (running) since ...
```

---

## Monitoring the Service

### Check Service Status

```bash
# Quick status check
sudo systemctl status crypto-arbitrage-paper.service

# Is it running?
sudo systemctl is-active crypto-arbitrage-paper.service
# Output: active

# Is it enabled on boot?
sudo systemctl is-enabled crypto-arbitrage-paper.service
# Output: enabled
```

### View Logs in Real-Time

```bash
# Service logs (systemd journal)
sudo journalctl -u crypto-arbitrage-paper.service -f

# Application logs
tail -f logs/paper_trading.log

# Both simultaneously (split terminal)
# Terminal 1:
sudo journalctl -u crypto-arbitrage-paper.service -f
# Terminal 2:
tail -f logs/paper_trading.log
```

### View Recent Logs

```bash
# Last 100 lines
sudo journalctl -u crypto-arbitrage-paper.service -n 100

# Last 1 hour
sudo journalctl -u crypto-arbitrage-paper.service --since "1 hour ago"

# Today's logs
sudo journalctl -u crypto-arbitrage-paper.service --since today

# Logs with errors
sudo journalctl -u crypto-arbitrage-paper.service -p err
```

### Check if Opportunities Are Being Detected

```bash
# Count opportunities in last hour
grep "opportunity detected" logs/paper_trading.log | tail -20

# Check database
sqlite3 data/arbitrage.db "SELECT COUNT(*) FROM opportunities WHERE detected_at > datetime('now', '-1 hour')"
```

---

## Service Management Commands

```bash
# Start service
sudo systemctl start crypto-arbitrage-paper.service

# Stop service
sudo systemctl stop crypto-arbitrage-paper.service

# Restart service (apply config changes)
sudo systemctl restart crypto-arbitrage-paper.service

# Reload systemd after editing service file
sudo systemctl daemon-reload
sudo systemctl restart crypto-arbitrage-paper.service

# Enable on boot
sudo systemctl enable crypto-arbitrage-paper.service

# Disable on boot
sudo systemctl disable crypto-arbitrage-paper.service

# View full service configuration
systemctl cat crypto-arbitrage-paper.service
```

---

## Log Rotation Setup

Prevent logs from filling your disk:

### Step 1: Create Logrotate Configuration

```bash
sudo nano /etc/logrotate.d/crypto-arbitrage
```

**Contents:**
```
/mnt/4tb/stock_scanner/market_news/CryptoArbitrage/logs/*.log {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    notifempty
    create 644 YOUR_USERNAME YOUR_USERNAME
}
```

Replace `YOUR_USERNAME` with your username.

### Step 2: Test Logrotate

```bash
# Dry run (test without actually rotating)
sudo logrotate -d /etc/logrotate.d/crypto-arbitrage

# Force rotation (for testing)
sudo logrotate -f /etc/logrotate.d/crypto-arbitrage
```

This will:
- Rotate logs daily
- Keep 30 days of history
- Compress old logs
- Delete empty logs

---

## Database Maintenance

### Daily Backup (Cron Job)

```bash
# Edit crontab
crontab -e

# Add daily backup at 3 AM
0 3 * * * cp /mnt/4tb/stock_scanner/market_news/CryptoArbitrage/data/arbitrage.db /mnt/4tb/stock_scanner/market_news/CryptoArbitrage/data/backups/arbitrage_$(date +\%Y\%m\%d).db

# Keep only last 7 days
0 4 * * * find /mnt/4tb/stock_scanner/market_news/CryptoArbitrage/data/backups/ -name "arbitrage_*.db" -mtime +7 -delete
```

### Create Backup Directory

```bash
mkdir -p data/backups
```

### Manual Backup

```bash
# Backup database
cp data/arbitrage.db data/backups/arbitrage_$(date +%Y%m%d_%H%M%S).db

# Check backup
ls -lh data/backups/
```

---

## Health Monitoring Script

Create a script to check system health:

```bash
nano check_health.sh
```

**Contents:**
```bash
#!/bin/bash
# Health check script for crypto arbitrage service

echo "========================================"
echo "Crypto Arbitrage Service Health Check"
echo "========================================"
echo ""

# Check service status
echo "1. Service Status:"
sudo systemctl is-active crypto-arbitrage-paper.service
if [ $? -eq 0 ]; then
    echo "   ✅ Service is running"
else
    echo "   ❌ Service is not running!"
fi
echo ""

# Check if opportunities are being detected (last 5 minutes)
echo "2. Recent Opportunities (last 5 minutes):"
COUNT=$(sqlite3 data/arbitrage.db "SELECT COUNT(*) FROM opportunities WHERE detected_at > datetime('now', '-5 minutes')" 2>/dev/null)
if [ -z "$COUNT" ]; then
    echo "   ⚠️  Cannot query database"
else
    echo "   Found: $COUNT opportunities"
    if [ $COUNT -gt 0 ]; then
        echo "   ✅ System is detecting opportunities"
    else
        echo "   ⚠️  No recent opportunities (may be normal)"
    fi
fi
echo ""

# Check log file size
echo "3. Log File Size:"
LOG_SIZE=$(du -h logs/paper_trading.log 2>/dev/null | cut -f1)
if [ -z "$LOG_SIZE" ]; then
    echo "   ⚠️  Log file not found"
else
    echo "   Current size: $LOG_SIZE"
fi
echo ""

# Check database size
echo "4. Database Size:"
DB_SIZE=$(du -h data/arbitrage.db 2>/dev/null | cut -f1)
if [ -z "$DB_SIZE" ]; then
    echo "   ⚠️  Database not found"
else
    echo "   Current size: $DB_SIZE"
fi
echo ""

# Check for recent errors
echo "5. Recent Errors (last 10 lines):"
ERRORS=$(grep -i "error\|exception\|failed" logs/paper_trading.log 2>/dev/null | tail -10)
if [ -z "$ERRORS" ]; then
    echo "   ✅ No recent errors"
else
    echo "$ERRORS"
fi
echo ""

# Memory and CPU usage
echo "6. Resource Usage:"
if pgrep -f "paper_trading.py" > /dev/null; then
    PID=$(pgrep -f "paper_trading.py")
    ps aux | head -1
    ps aux | grep $PID | grep -v grep
else
    echo "   ⚠️  Process not found"
fi
echo ""

echo "========================================"
echo "Health check complete"
echo "========================================"
```

Make it executable:
```bash
chmod +x check_health.sh
```

Run it:
```bash
./check_health.sh
```

### Automate Health Checks (Every Hour)

```bash
crontab -e

# Add health check every hour, save to log
0 * * * * /mnt/4tb/stock_scanner/market_news/CryptoArbitrage/check_health.sh >> /mnt/4tb/stock_scanner/market_news/CryptoArbitrage/logs/health_checks.log 2>&1
```

---

## Analyzing Results While Running

The service runs continuously. Analyze results anytime:

```bash
# Quick analysis
./analyze_opportunities.sh

# Check database directly
sqlite3 data/arbitrage.db

# View today's opportunities
sqlite3 data/arbitrage.db "SELECT COUNT(*) as count, AVG(net_spread_pct) as avg_spread FROM opportunities WHERE DATE(detected_at) = DATE('now')"

# Check last 24 hours
sqlite3 data/arbitrage.db "SELECT COUNT(*) FROM opportunities WHERE detected_at > datetime('now', '-24 hours')"
```

---

## Troubleshooting

### Service Won't Start

**Check logs:**
```bash
sudo journalctl -u crypto-arbitrage-paper.service -n 50
```

**Common issues:**
1. **API keys invalid** - Verify in `.env`
2. **Network connectivity** - Test: `ping api.binance.com`
3. **Permission issues** - Check file ownership: `ls -la`
4. **Python dependencies** - Reinstall: `pip install -r requirements.txt`

### No Opportunities Detected

**Possible causes:**
1. Market too efficient (rare opportunities)
2. Min spread threshold too high
3. Exchange connections not working
4. Network latency too high

**Solutions:**
```bash
# Lower threshold temporarily (edit config)
nano config/config.yaml
# Set min_spread_pct: 0.3

# Restart service
sudo systemctl restart crypto-arbitrage-paper.service

# Check if any opportunities exist at all
./run_paper_trading.sh 5
```

### High Memory Usage

**Check current usage:**
```bash
ps aux | grep paper_trading
```

**Optimize (if needed):**
```yaml
# In config/config.yaml
strategy:
  trading_pairs:
    # Reduce number of pairs
    - BTC/USDT
    - ETH/USDT
    # Comment out less important pairs
```

### Database Growing Too Large

**Check size:**
```bash
du -h data/arbitrage.db
```

**Clean old data (keep last 30 days):**
```bash
sqlite3 data/arbitrage.db "DELETE FROM opportunities WHERE detected_at < datetime('now', '-30 days')"
sqlite3 data/arbitrage.db "VACUUM"
```

**Automate cleanup (monthly cron):**
```bash
crontab -e

# Add: Clean database monthly
0 2 1 * * sqlite3 /mnt/4tb/stock_scanner/market_news/CryptoArbitrage/data/arbitrage.db "DELETE FROM opportunities WHERE detected_at < datetime('now', '-30 days'); VACUUM;"
```

---

## Network Issues & Reconnection

The system has basic reconnection logic, but for extra reliability:

### Monitor Network Connectivity

```bash
# Create network monitor script
nano monitor_network.sh
```

**Contents:**
```bash
#!/bin/bash
# Monitor network connectivity and restart service if needed

EXCHANGES=("api.binance.com" "api.pro.coinbase.com" "api.kraken.com")
FAILED=0

for EXCHANGE in "${EXCHANGES[@]}"; do
    ping -c 1 -W 2 $EXCHANGE > /dev/null 2>&1
    if [ $? -ne 0 ]; then
        echo "$(date): Cannot reach $EXCHANGE" >> logs/network_monitor.log
        FAILED=$((FAILED + 1))
    fi
done

if [ $FAILED -ge 2 ]; then
    echo "$(date): Multiple exchanges unreachable. Restarting service..." >> logs/network_monitor.log
    sudo systemctl restart crypto-arbitrage-paper.service
fi
```

Make executable and run every 5 minutes:
```bash
chmod +x monitor_network.sh

crontab -e
# Add:
*/5 * * * * /mnt/4tb/stock_scanner/market_news/CryptoArbitrage/monitor_network.sh
```

---

## Performance Metrics

### Check Opportunities Per Hour

```bash
sqlite3 data/arbitrage.db "
SELECT 
    strftime('%Y-%m-%d %H:00', detected_at) as hour,
    COUNT(*) as opportunities
FROM opportunities
WHERE detected_at > datetime('now', '-24 hours')
GROUP BY hour
ORDER BY hour DESC
"
```

### Average Duration

```bash
sqlite3 data/arbitrage.db "
SELECT 
    AVG(opportunity_duration_ms)/1000 as avg_duration_seconds,
    MIN(opportunity_duration_ms)/1000 as min_duration_seconds,
    MAX(opportunity_duration_ms)/1000 as max_duration_seconds
FROM opportunities
WHERE opportunity_duration_ms IS NOT NULL
"
```

---

## Security Best Practices

1. **API Keys:** Read-only permissions only
2. **File Permissions:** `.env` should be 600 (owner only)
3. **Firewall:** Only allow outbound HTTPS
4. **No Public Access:** Service runs locally only
5. **Regular Updates:** `pip install --upgrade -r requirements.txt`
6. **Monitor Logs:** Watch for suspicious activity

---

## Summary Checklist

Before running 24/7:

- [ ] API keys configured (read-only)
- [ ] Tested with `./run_paper_trading.sh 10`
- [ ] Systemd service created and enabled
- [ ] Service is running: `sudo systemctl status crypto-arbitrage-paper.service`
- [ ] Log rotation configured
- [ ] Database backup cron job set
- [ ] Health check script created
- [ ] Monitoring working (check logs)
- [ ] Opportunities being detected (check database)
- [ ] Network connectivity stable

---

## Quick Reference Commands

```bash
# Check status
sudo systemctl status crypto-arbitrage-paper.service

# View logs
tail -f logs/paper_trading.log

# Analyze results
./analyze_opportunities.sh

# Health check
./check_health.sh

# Restart service
sudo systemctl restart crypto-arbitrage-paper.service

# Query opportunities
sqlite3 data/arbitrage.db "SELECT COUNT(*) FROM opportunities WHERE detected_at > datetime('now', '-1 hour')"
```

---

## Next Steps

Once 24/7 service is stable:
1. Run for 3-7 days to collect data
2. Analyze results with `./analyze_opportunities.sh`
3. Optimize strategy parameters
4. Consider live trading with small capital

---

**Your system is now ready for reliable 24/7 paper trading! 🚀**


