# 24/7 Paper Trading Quick Start

Get your crypto arbitrage paper trading running 24/7 in under 15 minutes.

---

## What You Need

### 1. Exchange API Keys (Read-Only)

Get free API keys from these exchanges:

| Exchange | URL | Permissions |
|----------|-----|-------------|
| **Binance** | https://www.binance.com/en/my/settings/api-management | ✅ Read only |
| **Coinbase Pro** | https://pro.coinbase.com/profile/api | ✅ View only |
| **Kraken** | https://www.kraken.com/u/security/api | ✅ Query only |

**Important:** Enable READ/VIEW permissions ONLY. No trading, no withdrawals.

### 2. System Check

```bash
# Check Python version (need 3.11+)
python3 --version

# Check network latency to exchanges
ping -c 3 api.binance.com
ping -c 3 api.pro.coinbase.com
ping -c 3 api.kraken.com
```

**Latency should be < 200ms for best results**

---

## 5-Step Setup

### Step 1: Install Dependencies (2 minutes)

```bash
cd /mnt/4tb/stock_scanner/market_news/CryptoArbitrage

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install packages
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 2: Configure API Keys (3 minutes)

```bash
# Copy template
cp env_template.txt .env

# Edit with your keys
nano .env
```

**Paste your API keys:**
```env
CRYPTO_ENV=simulation

BINANCE_API_KEY=your_binance_key_here
BINANCE_API_SECRET=your_binance_secret_here

COINBASE_API_KEY=your_coinbase_key_here
COINBASE_API_SECRET=your_coinbase_secret_here
COINBASE_PASSPHRASE=your_coinbase_passphrase_here

KRAKEN_API_KEY=your_kraken_key_here
KRAKEN_API_SECRET=your_kraken_secret_here
```

**Save and protect:**
```bash
chmod 600 .env
```

### Step 3: Test Configuration (2 minutes)

```bash
# Quick 1-minute test
./run_paper_trading.sh 1
```

**Expected output:**
```
✅ Config loaded
✅ Database initialized
✅ Connected to 3 exchanges (read-only)
✅ Price aggregator initialized
🎯 Found opportunities
```

**If you see errors:** Check API keys in `.env`

### Step 4: Setup Service (3 minutes)

```bash
# Automated setup
./setup_service.sh
```

This will:
- Create systemd service file
- Enable auto-start on boot
- Ask if you want to start now

**Say "y" to start immediately**

### Step 5: Verify Running (1 minute)

```bash
# Check status
sudo systemctl status crypto-arbitrage-paper.service

# View live logs
tail -f logs/paper_trading.log

# Health check
./check_health.sh
```

**Done! Your paper trading is now running 24/7!** 🎉

---

## Daily Monitoring (30 seconds)

### Quick Status Check

```bash
# Health check (shows everything)
./check_health.sh
```

**Example output:**
```
======================================
Overall Health Assessment:
======================================
✅ HEALTHY (Score: 90/100)
System is operating normally.

Today: 347 opportunities
Last hour: 14 opportunities
```

### View Results

```bash
# Comprehensive analysis
./analyze_opportunities.sh
```

Shows:
- Opportunities per day
- Duration statistics
- Best pairs/exchanges
- Spread distributions

---

## Common Commands

```bash
# Start service
sudo systemctl start crypto-arbitrage-paper.service

# Stop service
sudo systemctl stop crypto-arbitrage-paper.service

# Restart service
sudo systemctl restart crypto-arbitrage-paper.service

# View real-time logs
tail -f logs/paper_trading.log

# Health check
./check_health.sh

# Analyze results
./analyze_opportunities.sh

# Check database
sqlite3 data/arbitrage.db "SELECT COUNT(*) FROM opportunities WHERE DATE(detected_at) = DATE('now')"
```

---

## What's Running?

Your paper trading service:

1. **Connects to 3 exchanges** (Binance, Coinbase, Kraken)
2. **Monitors 25+ trading pairs** (BTC/USDT, ETH/USDT, etc.)
3. **Detects arbitrage opportunities** (when price spread > 0.5%)
4. **Tracks opportunity lifecycles** (how long they stay open)
5. **Simulates executions** (with realistic slippage/fees)
6. **Logs everything to database** (SQLite)
7. **Auto-restarts on errors** (systemd handles crashes)
8. **Runs 24/7** (even after reboot)

**No real trading occurs** - this is pure data collection!

---

## Understanding Your Results

### Good Indicators

✅ **10+ opportunities per day** = Market has arbitrage potential
✅ **Avg duration > 500ms** = You have time to execute
✅ **Avg spread > 0.7%** = Profitable after fees
✅ **No errors in logs** = System running smoothly

### What to Look For

After 24 hours of data:

```bash
./analyze_opportunities.sh
```

**Key metrics:**
1. **Daily opportunities**: Need 10+ per day minimum
2. **Duration**: P50 should be > 500ms (execution window)
3. **Spread**: Average should be > 0.7% (profitable)
4. **Best pairs**: Focus on pairs with most opportunities

---

## Optimization Tips

### If Too Few Opportunities (< 5/day)

Lower threshold in `config/config.yaml`:
```yaml
strategy:
  min_spread_pct: 0.3  # From 0.5 to 0.3
```

Then restart:
```bash
sudo systemctl restart crypto-arbitrage-paper.service
```

### If Too Fast (< 200ms average)

You need lower latency. Options:
1. Better network connection
2. Co-location near exchanges
3. Focus on slower opportunities (raise threshold)

### If Database Too Large

```bash
# Clean old data (keep last 30 days)
sqlite3 data/arbitrage.db "DELETE FROM opportunities WHERE detected_at < datetime('now', '-30 days'); VACUUM;"
```

---

## Troubleshooting

### Service Not Running

```bash
# Check what went wrong
sudo journalctl -u crypto-arbitrage-paper.service -n 50

# Common issues:
# - API keys wrong: Check .env
# - Network down: Check ping to exchanges
# - Python error: Reinstall dependencies
```

### No Opportunities Detected

**This is actually normal!** Crypto arbitrage opportunities are rare because:
- Markets are efficient
- Many bots compete
- Opportunities close fast

**Try:**
1. Lower threshold to 0.3%
2. Run for 24+ hours (opportunities come in bursts)
3. Check you're on fast internet

### High Memory Usage

```bash
# Check current usage
ps aux | grep paper_trading

# If > 1GB, reduce pairs in config/config.yaml
```

---

## Maintenance Tasks

### Weekly

```bash
# Check health
./check_health.sh

# Review results
./analyze_opportunities.sh

# Check disk space
df -h
```

### Monthly

```bash
# Backup database
cp data/arbitrage.db data/backups/arbitrage_$(date +%Y%m%d).db

# Clean old logs (keep last 30 days)
find logs/ -name "*.log" -mtime +30 -delete

# Update dependencies
source venv/bin/activate
pip install --upgrade -r requirements.txt
sudo systemctl restart crypto-arbitrage-paper.service
```

---

## Automated Monitoring (Optional)

### Setup Auto-Health Checks

```bash
# Edit crontab
crontab -e

# Add: Health check every hour
0 * * * * /mnt/4tb/stock_scanner/market_news/CryptoArbitrage/check_health.sh >> /mnt/4tb/stock_scanner/market_news/CryptoArbitrage/logs/health_checks.log 2>&1

# Add: Network monitoring every 5 minutes
*/5 * * * * /mnt/4tb/stock_scanner/market_news/CryptoArbitrage/monitor_network.sh

# Add: Daily backup at 3 AM
0 3 * * * cp /mnt/4tb/stock_scanner/market_news/CryptoArbitrage/data/arbitrage.db /mnt/4tb/stock_scanner/market_news/CryptoArbitrage/data/backups/arbitrage_$(date +\%Y\%m\%d).db
```

---

## Success Criteria

After running 24/7 for **3-7 days**, you should see:

**Minimum for live trading consideration:**
- ✅ 10+ opportunities per day
- ✅ P50 duration > 500ms
- ✅ Average spread > 0.7%
- ✅ System stability (no crashes)
- ✅ No API rate limit errors

**Optimal indicators:**
- 🌟 20+ opportunities per day
- 🌟 P50 duration > 1 second
- 🌟 Average spread > 1.0%
- 🌟 Multiple profitable pairs
- 🌟 Consistent across days

---

## Next Steps

### After Collecting Data (1 week)

1. **Analyze results:**
   ```bash
   ./analyze_opportunities.sh
   ```

2. **Optimize configuration:**
   - Focus on best pairs
   - Adjust spread threshold
   - Set realistic position sizes

3. **Consider live trading:**
   - Start with $100-500
   - Test execution in real conditions
   - Monitor closely for 24-48 hours
   - Scale gradually

---

## Quick Reference

| Task | Command |
|------|---------|
| Check status | `sudo systemctl status crypto-arbitrage-paper.service` |
| View logs | `tail -f logs/paper_trading.log` |
| Health check | `./check_health.sh` |
| Analyze data | `./analyze_opportunities.sh` |
| Restart | `sudo systemctl restart crypto-arbitrage-paper.service` |
| Stop | `sudo systemctl stop crypto-arbitrage-paper.service` |

---

## Support

- **Full guide:** [24_7_SETUP_GUIDE.md](24_7_SETUP_GUIDE.md)
- **Paper trading guide:** [PAPER_TRADING_GUIDE.md](PAPER_TRADING_GUIDE.md)
- **Example session:** [EXAMPLE_SESSION.md](EXAMPLE_SESSION.md)
- **Main README:** [README.md](README.md)

---

## Summary

You now have:
- ✅ 24/7 paper trading service
- ✅ Automated opportunity detection
- ✅ Lifecycle tracking
- ✅ Performance analytics
- ✅ Auto-restart on failures
- ✅ Health monitoring

**Just let it run and collect data!**

Check back daily with:
```bash
./check_health.sh && ./analyze_opportunities.sh
```

---

**Happy Trading! 📊💰🚀**


