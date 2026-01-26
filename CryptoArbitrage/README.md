# Crypto Arbitrage Trading System

## Overview

High-speed cryptocurrency arbitrage system that detects and executes risk-free arbitrage opportunities across Binance, Coinbase, and Kraken.

**Target**: >0.5% profit per trade after fees  
**Speed**: Sub-second execution via websockets  
**Capital**: $10,000 allocation (10% of hedge fund)  
**Operation**: 24/7 automated trading  

---

## Features

- ✅ Real-time price monitoring via websockets (ccxt.pro)
- ✅ Multi-exchange arbitrage detection (Binance, Coinbase, Kraken)
- ✅ Simultaneous order execution for risk-free trades
- ✅ Automatic balance rebalancing across exchanges
- ✅ Comprehensive risk management and circuit breakers
- ✅ Web dashboard for real-time monitoring
- ✅ SQLite database for trade logging and analytics
- ✅ 24/7 daemon with auto-restart

---

## Quick Start

### 1. Installation

```bash
cd /mnt/4tb/stock_scanner/market_news/CryptoArbitrage

# Create virtual environment (or use existing)
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit .env and add your exchange API keys
nano .env
```

### 3. Initialize Database

```bash
python src/database/init_db.py
```

### 4. Test Connection (Simulation Mode)

```bash
# Test without API keys (uses mock data)
python test_exchanges.py
```

### 5. Start System

```bash
# Start daemon
./start_arbitrage.sh

# Start dashboard (separate terminal)
./start_dashboard.sh

# Open browser
http://localhost:5001
```

---

## Architecture

```
┌─────────────────────────────────────────────────┐
│         24/7 ARBITRAGE DAEMON                    │
│                                                  │
│  ┌─────────────┐     ┌─────────────┐           │
│  │  Websocket  │────▶│  Arbitrage  │           │
│  │    Feeds    │     │  Detector   │           │
│  │ (3 exchanges)│     │             │           │
│  └─────────────┘     └─────────────┘           │
│                             │                    │
│                             ▼                    │
│                      ┌─────────────┐            │
│                      │  Executor   │            │
│                      │ (Both Legs) │            │
│                      └─────────────┘            │
│                             │                    │
│                             ▼                    │
│                      ┌─────────────┐            │
│                      │   SQLite    │            │
│                      │   Database  │            │
│                      └─────────────┘            │
└─────────────────────────────────────────────────┘
```

---

## How Arbitrage Works

### Example: BTC/USDT Arbitrage

1. **Detection**:
   - Binance: BTC = $43,000
   - Coinbase: BTC = $43,500
   - Spread: 1.16%

2. **Validation**:
   - Gross spread: 1.16%
   - Fees (Binance 0.1% + Coinbase 0.6%): -0.7%
   - Net spread: 0.46%
   - ❌ Below 0.5% threshold - SKIP

3. **If spread was 0.8%**:
   - Net spread: 0.8% - 0.7% = 0.1% ✅ EXECUTE

4. **Execution**:
   - Buy 0.1 BTC on Binance @ $43,000 = $4,300
   - Sell 0.1 BTC on Coinbase @ $43,500 = $4,350
   - Profit: $50 - fees ≈ $30 net

5. **Result**:
   - Risk-free $30 profit
   - Both legs executed simultaneously
   - If one fails, reverse the other

---

## Configuration

### Strategy Parameters

Edit `config/config.yaml`:

```yaml
strategy:
  total_capital_usd: 10000
  max_position_size_usd: 1000
  min_spread_pct: 0.5
  execution_timeout_seconds: 2
```

### Supported Pairs

Default: Top 50 altcoins by market cap  
Customize in `config/config.yaml` under `strategy.trading_pairs`

### Risk Limits

```yaml
risk:
  max_daily_loss_usd: 500
  max_consecutive_failures: 5
  max_trades_per_hour: 20
```

---

## Monitoring

### Web Dashboard

Access at: `http://localhost:5001`

Shows:
- Live arbitrage opportunities
- Recent executions
- P&L tracking
- Exchange balances
- System health
- Latency metrics

### Command Line

```bash
# Check daemon status
./check_status.sh

# View recent opportunities
python scripts/show_opportunities.py

# Check P&L
python scripts/show_pnl.py

# View logs
tail -f logs/arbitrage.log
```

### Database Queries

```bash
# View recent opportunities
sqlite3 data/arbitrage.db "
  SELECT * FROM opportunities 
  WHERE detected_at > datetime('now', '-1 hour')
  ORDER BY net_spread_pct DESC;
"

# Check today's P&L
sqlite3 data/arbitrage.db "
  SELECT SUM(realized_profit_usd) as total_profit
  FROM executions
  WHERE DATE(started_at) = DATE('now')
    AND is_successful = 1;
"
```

---

## Expected Performance

Based on crypto arbitrage research:

| Metric | Expected Value |
|--------|----------------|
| Opportunities/day | 5-20 |
| Avg spread | 0.7% net |
| Execution success | 70-80% |
| Avg execution time | <1 second |
| Daily return | 0.3-0.7% |
| Annualized ROI | 100%+ |

---

## Risk Management

### Built-in Protections

1. **Simultaneous Execution**: Both legs execute or neither
2. **Timeout Protection**: Cancel if not filled within 2 seconds
3. **Balance Verification**: Check balances before every trade
4. **Price Staleness**: Reject old price data (>5 seconds)
5. **Circuit Breakers**: Stop trading after 5 consecutive failures
6. **Daily Loss Limit**: Halt at -$500 daily loss

### Capital Distribution

- Keep 50% USDT on each exchange initially
- System auto-rebalances when >30% imbalanced
- Monitor balances every 5 minutes

---

## Development

### Project Structure

```
CryptoArbitrage/
├── src/
│   ├── common/          # Models, config
│   ├── exchanges/       # Exchange connectors
│   ├── detection/       # Arbitrage detection
│   ├── execution/       # Order execution
│   ├── risk/            # Risk management
│   ├── monitoring/      # Performance tracking
│   ├── daemon/          # 24/7 daemon
│   └── dashboard/       # Web dashboard
├── config/              # Configuration files
├── data/                # SQLite database
├── logs/                # System logs
├── tests/               # Unit tests
└── scripts/             # Utility scripts
```

### Testing

```bash
# Run all tests
pytest

# Test specific module
pytest tests/test_detector.py

# Test with simulation
python test_simulation.py
```

---

## Deployment

### Development Mode

```bash
# Simulation with mock data (no API keys needed)
CRYPTO_ENV=development ./start_arbitrage.sh
```

### Simulation Mode

```bash
# Uses real exchange APIs but doesn't execute trades
CRYPTO_ENV=simulation ./start_arbitrage.sh
```

### 🆕 Paper Trading Mode (Recommended First Step)

**Test the system without risking money!**

```bash
# Run paper trading for 10 minutes
./run_paper_trading.sh 10

# Analyze results
./analyze_opportunities.sh
```

**See:**
- 📊 How many opportunities per day
- ⏱️ How long opportunities stay open
- 💰 Which pairs/exchanges are best
- 📈 Spread distributions

**Full guide:** See [PAPER_TRADING_GUIDE.md](PAPER_TRADING_GUIDE.md)

### Production Mode

```bash
# Real trading with real money
CRYPTO_ENV=production ./start_arbitrage.sh
```

---

## Safety Notes

⚠️ **IMPORTANT SAFETY GUIDELINES**:

1. **Start Small**: Test with $100 before scaling to $10K
2. **Monitor Closely**: Watch first 24 hours continuously
3. **API Permissions**: Limit API keys to trading only (no withdrawals)
4. **IP Whitelist**: Restrict API access to your server IP
5. **2FA Required**: Enable on all exchange accounts
6. **Withdrawal Limits**: Set low limits on exchanges
7. **Test Mode First**: Use simulation mode for at least 1 week

---

## Troubleshooting

### No Opportunities Detected

- Check websocket connections: `./check_status.sh`
- Verify exchange prices: `python scripts/check_prices.py`
- Lower min_spread_pct temporarily to see if any exist
- Check if pairs are available on all exchanges

### Orders Not Executing

- Verify API keys are correct
- Check API permissions (trading enabled)
- Verify IP whitelist on exchanges
- Check exchange balance requirements
- Review logs: `tail -f logs/arbitrage.log`

### High Latency

- Check network connection
- Verify exchange status (no API issues)
- Consider switching to faster DNS
- Monitor system resources

---

## Support

- Documentation: `docs/`
- Logs: `logs/arbitrage.log`
- Database: `data/arbitrage.db`
- Configuration: `config/config.yaml`

---

## License

Proprietary - Internal Use Only

---

**Built with the same institutional-grade architecture as the Hedge Fund system.** 🚀

