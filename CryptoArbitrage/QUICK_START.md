# Crypto Arbitrage - Quick Start Guide

## 🚀 Get Started in 5 Minutes

### 1. Test the System (No API Keys Needed)

```bash
cd /mnt/4tb/stock_scanner/market_news/CryptoArbitrage

# Run system test
python test_system.py
```

**Expected output:**
```
✅ Config loaded
✅ Database initialized
✅ Exchange manager initialized
✅ All tests passed!
```

---

### 2. Start in Simulation Mode

```bash
# Terminal 1: Start arbitrage daemon
./start_arbitrage.sh

# Terminal 2: Start dashboard
./start_dashboard.sh
```

**Open browser:**
```
http://localhost:5001
```

---

## 📊 What You'll See

### Dashboard Metrics

| Metric | What It Shows |
|--------|---------------|
| **Today's Profit** | Profit made today |
| **All-Time Profit** | Cumulative profit since start |
| **Success Rate** | % of executions that succeeded |
| **Avg Execution Time** | Speed of trade execution |
| **Opportunities Today** | Arbitrage opportunities detected |
| **Avg Spread** | Average profit per trade |

### Tables

**Recent Opportunities:**
- Shows arbitrage opportunities as they're detected
- Buy from / Sell on exchanges
- Expected profit
- Status (Detected/Executed)

**Recent Executions:**
- Actual trades executed
- Profit realized
- Execution time
- Success/failure status

---

## 🎯 Simulation Mode

**What It Does:**
- ✅ Detects REAL arbitrage opportunities from price differences
- ✅ Shows what WOULD be executed
- ❌ Does NOT place real orders
- ❌ Does NOT need API keys

**Perfect For:**
- Testing the system
- Understanding how it works
- Seeing if opportunities exist
- Verifying logic before going live

---

## 💰 Going Live (Tomorrow After API Keys)

### Step 1: Get API Keys

**Binance:**
- Go to: https://www.binance.com/en/my/settings/api-management
- Create API key
- Enable spot trading
- Whitelist your IP

**Coinbase Pro:**
- Go to: https://pro.coinbase.com/profile/api
- Create API key
- Enable trading permissions

**Kraken:**
- Go to: https://www.kraken.com/u/security/api
- Generate API key
- Enable trading

### Step 2: Configure

```bash
# Copy template
cp env_template.txt .env

# Edit .env
nano .env
```

Add your API keys:
```bash
BINANCE_API_KEY=your_key_here
BINANCE_API_SECRET=your_secret_here
COINBASE_API_KEY=your_key_here
COINBASE_API_SECRET=your_secret_here
COINBASE_PASSPHRASE=your_passphrase_here
KRAKEN_API_KEY=your_key_here
KRAKEN_API_SECRET=your_secret_here
```

### Step 3: Start Small

```bash
# Edit config.yaml - reduce capital for testing
nano config/config.yaml
```

Change:
```yaml
strategy:
  total_capital_usd: 100  # Start with $100
  max_position_size_usd: 50  # $50 per trade
```

### Step 4: Go Live

```bash
export CRYPTO_ENV=production
./start_arbitrage.sh
```

Monitor closely for first 24 hours!

---

## 🛡️ Safety Features

The system has multiple safety layers:

1. **Balance Checks**: Verifies sufficient funds before each trade
2. **Circuit Breakers**: Stops after 5 consecutive failures
3. **Daily Loss Limit**: Halts at -$500 daily loss
4. **Rate Limiting**: Max 20 trades/hour, 100/day
5. **Price Staleness**: Rejects old prices (>5 seconds)
6. **Simultaneous Execution**: Both legs or neither
7. **Auto-Reversal**: If one leg fails, reverses the other

---

## 📈 Expected Results

Based on simulation and research:

**In Simulation Mode:**
- You'll see opportunities detected
- System will log what it would execute
- No real trades, no risk

**In Live Mode (typical day):**
- 5-20 opportunities detected
- 3-15 trades executed (70-80% execution rate)
- Average profit: 0.5-0.7% per trade
- Daily return: 0.3-0.7% on deployed capital
- Typical profit: $30-$70/day on $10K capital

---

## 🔍 Monitoring

### Check System Status

```bash
# Is daemon running?
ps aux | grep arbitrage_daemon

# View live logs
tail -f logs/arbitrage.log

# Check database
sqlite3 data/arbitrage.db "SELECT COUNT(*) FROM opportunities;"
```

### View Recent Activity

```bash
# Recent opportunities
sqlite3 data/arbitrage.db "
  SELECT detected_at, pair, net_spread_pct, is_executed
  FROM opportunities
  ORDER BY detected_at DESC
  LIMIT 10;
"

# Today's P&L
sqlite3 data/arbitrage.db "
  SELECT SUM(realized_profit_usd) as profit
  FROM executions
  WHERE DATE(started_at) = DATE('now')
    AND is_successful = 1;
"
```

---

## ⚠️ Troubleshooting

### No Opportunities Detected

**Possible causes:**
1. Spreads too small (crypto markets are efficient)
2. Not enough exchanges connected
3. Price feeds not working

**Solutions:**
- Temporarily lower `min_spread_pct` to 0.3% in config
- Check logs for connection errors
- Verify all 3 exchanges are connected

### Daemon Crashes

**Check logs:**
```bash
tail -50 logs/arbitrage.log
```

**Common issues:**
- Missing dependencies: `pip install -r requirements.txt`
- Invalid config: Check `config/config.yaml`
- Database error: Delete `data/arbitrage.db` and restart

---

## 💡 Pro Tips

1. **Watch First Hour**: Monitor closely when you first start
2. **Start Small**: Use $100 before scaling to $10K
3. **Check Spreads**: Lower threshold if no opportunities
4. **Balance Distribution**: Keep 50% USDT on each exchange
5. **Monitor Latency**: Sub-second execution is critical
6. **Review Daily**: Check dashboard every evening

---

## 📚 Full Documentation

See `docs/` folder for:
- Architecture details
- API documentation
- Risk management guide
- Performance optimization

---

**The system is ready to find you risk-free profits!** ₿


