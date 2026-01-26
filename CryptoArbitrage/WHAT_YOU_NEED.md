# What You Need for Reliable 24/7 Paper Trading

Complete checklist of requirements for running crypto arbitrage paper trading.

---

## 1. Exchange API Keys ⚡

### Required: Read-Only Keys from 3 Exchanges

You need API keys from these exchanges for **market data access only**:

#### Binance
- **Get keys:** https://www.binance.com/en/my/settings/api-management
- **Permissions:** Enable "Read" ONLY
- **Cost:** Free
- **Time:** 2 minutes

#### Coinbase Pro
- **Get keys:** https://pro.coinbase.com/profile/api
- **Permissions:** "View" ONLY (uncheck trading/transfer)
- **Cost:** Free
- **Time:** 2 minutes
- **Note:** Also get "Passphrase"

#### Kraken
- **Get keys:** https://www.kraken.com/u/security/api
- **Permissions:** "Query Funds" + "Query Orders" (read-only)
- **Cost:** Free
- **Time:** 2 minutes

### Important Security Notes:

✅ **DO:**
- Use read-only permissions ONLY
- Enable IP whitelist (optional but recommended)
- Store keys in `.env` file with 600 permissions

❌ **DON'T:**
- Enable trading permissions (not needed for paper trading)
- Enable withdrawal permissions
- Share keys publicly
- Commit `.env` to git

---

## 2. System Requirements 💻

### Operating System
- **Linux** (Ubuntu 20.04+, Debian 10+, or similar)
- **macOS** (also works but systemd service requires Linux)
- **Windows** (works but requires WSL or different service setup)

**Recommended:** Ubuntu 22.04 LTS

### Hardware
- **CPU:** 1 core minimum, 2+ cores recommended
- **RAM:** 512MB minimum, 1GB recommended
- **Disk:** 1GB minimum (for logs and database)
- **Network:** Stable broadband connection

**Can run on:**
- Raspberry Pi 4 (4GB model)
- VPS (DigitalOcean $6/month droplet)
- Home server
- Desktop computer

### Software
- **Python:** 3.11 or newer
- **pip:** Latest version
- **SQLite:** Built-in with Python
- **systemd:** For service management (Linux only)

**Check versions:**
```bash
python3 --version  # Should be 3.11+
pip3 --version
sqlite3 --version
```

---

## 3. Network Requirements 🌐

### Internet Connection
- **Speed:** 10 Mbps minimum, 50+ Mbps recommended
- **Stability:** Consistent connection (no frequent drops)
- **Latency:** < 100ms to exchanges ideal

**Test latency:**
```bash
ping -c 10 api.binance.com
ping -c 10 api.pro.coinbase.com
ping -c 10 api.kraken.com
```

**Good:** < 50ms average
**Acceptable:** 50-150ms
**Poor:** > 150ms (may miss fast opportunities)

### Firewall/Ports
- **Inbound:** None required (service doesn't listen)
- **Outbound:** HTTPS (443) to exchange APIs

**Required access:**
- api.binance.com (port 443)
- api.pro.coinbase.com (port 443)
- api.kraken.com (port 443)
- stream.binance.com (port 443) - websockets
- ws-feed.pro.coinbase.com (port 443) - websockets

**Test connectivity:**
```bash
curl -I https://api.binance.com/api/v3/ping
curl -I https://api.pro.coinbase.com/time
curl -I https://api.kraken.com/0/public/Time
```

All should return `HTTP/2 200`

---

## 4. Python Dependencies 📦

All dependencies are in `requirements.txt`:

### Key Libraries
- **ccxt** + **ccxt.pro** - Exchange API and websockets
- **aiohttp**, **websockets** - Async networking
- **pandas**, **numpy** - Data handling
- **pyyaml**, **python-dotenv** - Configuration
- **flask** - Web dashboard (optional)

**Install:**
```bash
pip install -r requirements.txt
```

**Verify:**
```bash
python3 -c "import ccxt; import ccxt.pro; print('OK')"
```

---

## 5. Configuration Files 📝

### Required Files

1. **`.env`** - API keys and configuration
   ```bash
   cp env_template.txt .env
   nano .env
   ```

2. **`config/config.yaml`** - Strategy parameters
   - Already configured with defaults
   - Can customize pairs, thresholds, etc.

3. **`data/arbitrage.db`** - SQLite database
   - Created automatically on first run

### Directory Structure
```
CryptoArbitrage/
├── .env (your API keys)
├── config/
│   └── config.yaml
├── data/
│   ├── arbitrage.db (auto-created)
│   └── backups/
├── logs/
│   ├── paper_trading.log
│   └── service.log
├── venv/ (virtual environment)
└── ... (source code)
```

---

## 6. Optional but Recommended 🌟

### For Better Performance

**1. Low-Latency Network:**
- VPS near exchanges (AWS us-east-1, DigitalOcean NYC)
- Reduces latency to < 10ms

**2. Monitoring:**
- Health check cron jobs
- Network connectivity monitoring
- Log rotation

**3. Backups:**
- Daily database backups
- Config file backups

**4. Notification (future):**
- Telegram bot for alerts
- Email notifications

### For Production Use (Live Trading)

**1. Dedicated Server:**
- Not your daily driver
- Reliable power/network
- Linux preferred

**2. Multiple API Keys:**
- Separate keys per exchange
- Different IP whitelists

**3. Enhanced Security:**
- SSH key-only access
- Firewall configured
- Regular security updates

---

## 7. Time Requirements ⏱️

### Initial Setup
- **Get API keys:** 10 minutes (3× exchanges)
- **Install dependencies:** 5 minutes
- **Configure:** 5 minutes
- **Test:** 5 minutes
- **Setup service:** 5 minutes

**Total:** ~30 minutes first time

### Ongoing Maintenance
- **Daily check:** 1 minute (./check_health.sh)
- **Weekly analysis:** 5 minutes (./analyze_opportunities.sh)
- **Monthly maintenance:** 10 minutes (backups, updates)

---

## 8. Cost 💰

### Direct Costs
- **API keys:** FREE (read-only access)
- **Software:** FREE (all open source)
- **Exchange fees:** $0 (no trading in paper mode)

### Indirect Costs (if using VPS)
- **Small VPS:** $5-10/month
- **Home electricity:** ~$2/month (Raspberry Pi)
- **Domain (optional):** $12/year

### Total Cost for Home Setup
**$0/month** if running on existing computer!

---

## 9. Prerequisites Checklist ✅

Before starting, verify you have:

- [ ] Linux/macOS system with Python 3.11+
- [ ] Stable internet connection (< 100ms latency to exchanges)
- [ ] Binance API key (read-only)
- [ ] Coinbase Pro API key (view-only)
- [ ] Kraken API key (query-only)
- [ ] 1GB free disk space
- [ ] 30 minutes for initial setup
- [ ] Basic command-line knowledge

---

## 10. What You DON'T Need ⛔

**NOT required for paper trading:**

- ❌ Money for trading (no real trades)
- ❌ Exchange account verification (API keys work)
- ❌ Trading permissions on API keys
- ❌ Withdrawal permissions
- ❌ 2FA for API (though recommended for account security)
- ❌ Public IP address
- ❌ Open inbound ports
- ❌ Advanced programming knowledge
- ❌ Database administration skills

**Paper trading is completely FREE and RISK-FREE!**

---

## Quick Start Checklist

Ready to start? Here's your checklist:

1. [ ] Get 3 sets of read-only API keys (15 minutes)
2. [ ] Test network latency (2 minutes)
3. [ ] Install Python dependencies (5 minutes)
4. [ ] Configure .env with API keys (3 minutes)
5. [ ] Run test: `./run_paper_trading.sh 1` (1 minute)
6. [ ] Setup service: `./setup_service.sh` (5 minutes)
7. [ ] Verify: `./check_health.sh` (1 minute)

**Total: ~30 minutes**

---

## Next Steps

Once you have everything:

1. **Follow:** [24_7_QUICKSTART.md](24_7_QUICKSTART.md)
2. **Read:** [PAPER_TRADING_GUIDE.md](PAPER_TRADING_GUIDE.md)
3. **Reference:** [24_7_SETUP_GUIDE.md](24_7_SETUP_GUIDE.md)

---

## Still Have Questions?

### Common Questions

**Q: Do I need money in my exchange accounts?**
A: No! Paper trading only needs API keys for price data.

**Q: Can I run this on Windows?**
A: Yes, but systemd service setup differs. Use Windows Task Scheduler instead.

**Q: What if I only have 2 exchanges?**
A: Will work but fewer opportunities. Try to get all 3.

**Q: Is my data/keys secure?**
A: Yes - stored locally, read-only keys, no network exposure.

**Q: How much data will this use?**
A: ~100MB/day network, ~10MB/day disk

**Q: Can I run multiple instances?**
A: Yes, but same API keys = same data. Not needed.

---

## Ready?

If you have everything on the checklist, you're ready to start!

```bash
cd /mnt/4tb/stock_scanner/market_news/CryptoArbitrage
./run_paper_trading.sh 1  # Quick test
```

If successful, proceed to full setup:
[24_7_QUICKSTART.md](24_7_QUICKSTART.md)

---

**Good luck! 🚀**


