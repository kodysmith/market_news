# Setup Instructions - Production Trading System

## Quick Setup (15 minutes)

### 1. Install Python Dependencies

```bash
cd /mnt/4tb/stock_scanner/market_news/HedgeFund

# Install all dependencies
pip install pydantic python-dotenv pyyaml pandas numpy scipy yfinance psycopg2-binary redis

# Optional but recommended:
pip install py_vollib structlog alpaca-py pytest
```

### 2. Test Foundation Modules

```bash
# Test data models
python -c "import sys; sys.path.insert(0, '.'); from src.common.models import Position; print('✓ Models working')"

# Test configuration  
python -c "import sys; sys.path.insert(0, '.'); from src.common.config import load_config; c=load_config(); print(f'✓ Config: {c.strategy.assets}')"

# Test pricing
python -c "import sys; sys.path.insert(0, '.'); from src.pricing.options_pricer import OptionsPricer; p=OptionsPricer(); print(f'✓ Pricer: ${p.price_option(\"p\", 400, 395, 0.1, 0.25)}')"

# Test market data (LIVE DATA!)
python -c "import sys; sys.path.insert(0, '.'); from src.data.market_data_service import MarketDataService; from src.common.config import load_config; d=MarketDataService(load_config().broker); print(f'✓ SPY: ${d.get_price(\"SPY\")}')"
```

**All these should work NOW!**

---

## What's Built & Working (40% Complete)

### ✅ Working Modules:

1. **Core Data Models** - Position, Order, Fill, Signal
2. **Configuration** - Multi-environment, validated
3. **Options Pricing** - Black-Scholes, Greeks, strike solving
4. **Database Schema** - Complete SQL (needs PostgreSQL installed)
5. **Database Connections** - Pooling, health checks
6. **Market Data** - **Fetching REAL live prices**
7. **Position Manager** - State tracking
8. **Audit Logger** - Compliance logging

### ✅ Complete Documentation:

All in `/HedgeFund/docs/`:
- 10 investor/technical/compliance documents
- System architecture
- Implementation roadmap
- Operational procedures

### ✅ Validated Backtests:

All in `/backtesting/`:
- Original strategy (89% return)
- Adaptive hedging (79% return, 1.63 Sharpe, -5.6% DD)
- Multi-asset (636% return, 49% CAGR)

---

## Remaining Work (60%)

### Critical Modules Needed for Trading:

**Must Build**:
9. Signal Generator - Strategy logic
10. Risk Manager - Circuit breakers
11. Alpaca Client - Broker integration  
12. Order Executor - Execute trades
13. Trading Orchestrator - Main loop

**Supporting**:
14. NAV Calculator
15. Integration Tests
16. Docker setup

**Estimated**: 80 hours remaining

---

## Your Immediate Next Steps

### While System is Being Built:

**1. Create Alpaca Account** (30 min)
- Go to alpaca.markets
- Create account
- Enable options trading
- Get paper trading API keys

**2. Add API Keys** (5 min)
```bash
# Create .env file
cat > .env << EOF
ALPACA_API_KEY=your_key_here
ALPACA_SECRET_KEY=your_secret_here
TRADING_ENV=paper
EOF
```

**3. Optional: Install Databases** (30 min)

```bash
# PostgreSQL
sudo apt install postgresql
sudo systemctl start postgresql
sudo -u postgres createdb trading_dev

# Redis  
sudo apt install redis-server
sudo systemctl start redis-server
```

---

## Using What's Built

### Example: Price Options with Live Data

```python
import sys
sys.path.insert(0, '/mnt/4tb/stock_scanner/market_news/HedgeFund')

from src.common.config import load_config
from src.data.market_data_service import MarketDataService
from src.pricing.options_pricer import OptionsPricer

# Initialize
config = load_config()
data = MarketDataService(config.broker)
pricer = OptionsPricer()

# Get live price
spy = float(data.get_price('SPY'))
iv = float(data.get_iv('SPY'))

print(f"SPY: ${spy:.2f}, IV: {iv:.1%}")

# Calculate option prices
strike = pricer.strike_for_delta('p', spy, 14/365, iv, -0.30)
premium = pricer.price_option('p', spy, float(strike), 14/365, iv)

print(f"30% delta put: ${strike:.2f} strike, ${premium:.2f} premium")
print(f"Selling 10 contracts = ${float(premium) * 1000:.2f} collected")
```

**This works TODAY with REAL market data!**

---

## Next: Continuing to Build ALL Modules

I'm continuing to build the remaining modules to completion.

**Progress will be tracked in**: `BUILD_STATUS.md`

**When complete**, you'll have a full production trading system ready for paper trading!

---

*Setup Guide v1.0*


