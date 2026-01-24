# 🗺️ Trading Systems Navigation Guide

## Quick Access

### Hedge Fund System
```bash
cd /mnt/4tb/stock_scanner/market_news/HedgeFund
```

**Key Files:**
- `COMPLETE_SYSTEM_GUIDE.md` - Start here
- `start_daemon.sh` - Start trading
- `start_dashboard.sh` - Start monitoring
- `check_daemon.sh` - Check status

**Dashboard:** http://localhost:5000

### Crypto Arbitrage System  
```bash
cd /mnt/4tb/stock_scanner/market_news/CryptoArbitrage
```

**Key Files:**
- `MORNING_BRIEFING.md` - **START HERE**
- `test_system.py` - Verify installation
- `start_arbitrage.sh` - Start trading
- `start_dashboard.sh` - Start monitoring
- `check_status.sh` - Check status

**Dashboard:** http://localhost:5001

---

## System Comparison

| Feature | HedgeFund | CryptoArbitrage |
|---------|-----------|-----------------|
| **Location** | `/HedgeFund/` | `/CryptoArbitrage/` |
| **Capital** | $100,000 | $10,000 |
| **Strategy** | Options wheel | Price arbitrage |
| **Market** | US Equities | Cryptocurrency |
| **Hours** | 9:30-4PM ET | 24/7 |
| **Dashboard** | :5000 | :5001 |
| **Status** | ✅ Running | ✅ Ready |

---

## Both Systems Running

### Terminal Layout

```
Terminal 1: HedgeFund daemon
Terminal 2: HedgeFund dashboard
Terminal 3: CryptoArbitrage daemon
Terminal 4: CryptoArbitrage dashboard
```

### Browser Tabs

```
Tab 1: http://localhost:5000  (HedgeFund)
Tab 2: http://localhost:5001  (CryptoArbitrage)
```

---

## Documentation Index

### Crypto Arbitrage
- `CryptoArbitrage/MORNING_BRIEFING.md` ← **START HERE**
- `CryptoArbitrage/QUICK_START.md`
- `CryptoArbitrage/docs/ARCHITECTURE.md`

### Hedge Fund
- `HedgeFund/COMPLETE_SYSTEM_GUIDE.md`
- `HedgeFund/DEPLOYMENT_GUIDE.md`
- `HedgeFund/DASHBOARD_GUIDE.md`

### Overview
- `SYSTEMS_OVERVIEW.md` - Both systems comparison
- `WAKE_UP_README.txt` - Morning summary

---

## Quick Commands

### Check Both Systems

```bash
# HedgeFund
cd /mnt/4tb/stock_scanner/market_news/HedgeFund && ./check_daemon.sh

# CryptoArbitrage  
cd /mnt/4tb/stock_scanner/market_news/CryptoArbitrage && ./check_status.sh
```

### Start Both Systems

```bash
# HedgeFund
cd /mnt/4tb/stock_scanner/market_news/HedgeFund
./start_daemon.sh supervisor
./start_dashboard.sh

# CryptoArbitrage
cd /mnt/4tb/stock_scanner/market_news/CryptoArbitrage
./start_arbitrage.sh
./start_dashboard.sh
```

### View Logs

```bash
# HedgeFund
tail -f /mnt/4tb/stock_scanner/market_news/HedgeFund/logs/daemon.log

# CryptoArbitrage
tail -f /mnt/4tb/stock_scanner/market_news/CryptoArbitrage/logs/arbitrage.log
```

---

**You now have complete institutional trading infrastructure!** 🚀
