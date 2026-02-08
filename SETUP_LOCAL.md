# Local Setup Guide

Complete guide to set up and run the trading system locally for testing and backtesting.

## Prerequisites

- Python 3.11+
- pip
- (Optional) Supabase account for paper/live modes
- (Optional) Historical data files for backtesting

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Test the System

Run the test script to verify everything works:

```bash
python test_trading_system.py
```

This will:
- Test individual components (regime detector, portfolio engine, risk manager)
- Test backtest mode with synthetic data

### 3. Run a Backtest

**Option A: Using the backtest runner script**

```bash
python backtest/run_backtest.py --start-date 2024-01-01 --end-date 2024-01-31
```

**Option B: Using the local runner script**

```bash
python run_local.py --mode backtest --start-date 2024-01-01 --end-date 2024-01-31
```

**Option C: Using Python directly**

```python
from trading.runner import run_strategy

results = run_strategy(
    mode='backtest',
    config_path='config',
    start_date='2024-01-01',
    end_date='2024-01-31'
)
print(f"Final portfolio value: ${results['final_value']:,.2f}")
```

**Note**: The system will use synthetic data if historical CSV files aren't available. To use real data:
- Place CSV files in `data/` directory
- Files should have columns: `date`, `close`, `volume`
- Name files like `spy_historical.csv`, `qqq_historical.csv`, etc.

### 4. Run Paper Trading Locally (Optional)

**Step 1: Set up Supabase**

1. Create a Supabase project at https://supabase.com
2. Go to SQL Editor
3. Copy and run the contents of `supabase_schema.sql`
4. Get your Supabase URL and anon key from Settings > API

**Step 2: Set environment variables**

```bash
export SUPABASE_URL="https://your-project.supabase.co"
export SUPABASE_KEY="your-anon-key"
export TRADING_MODE=paper
```

**Step 3: Run the Flask app**

```bash
python main_live.py
```

The app will start on `http://localhost:8080`

**Step 4: Test the endpoints**

```bash
# Health check
curl http://localhost:8080/health

# Run a paper trade
curl -X POST http://localhost:8080/run \
  -H "Content-Type: application/json" \
  -d '{"mode": "paper"}'
```

**Alternative: Run directly without Flask**

```bash
python run_local.py --mode paper
```

### 5. Run Live Trading Locally (⚠️ Use with Caution)

**WARNING**: Live mode executes real trades with real money. Only use after thorough paper trading validation.

1. Set up broker API keys (e.g., Alpaca):

```bash
export ALPACA_API_KEY="your-api-key"
export ALPACA_API_SECRET="your-api-secret"
export ALPACA_BASE_URL="https://api.alpaca.markets"  # or paper URL
export TRADING_MODE=live
```

2. Run via Flask:

```bash
python main_live.py
```

3. Or run directly:

```bash
python run_local.py --mode live
```

## Running Modes Locally

### Backtest Mode (Recommended for Development)

Backtest mode is the safest way to test your strategy. It uses historical data and doesn't require any external services.

```bash
# Quick backtest (1 month)
python backtest/run_backtest.py --start-date 2024-01-01 --end-date 2024-01-31

# Longer backtest (1 year)
python backtest/run_backtest.py --start-date 2023-01-01 --end-date 2023-12-31

# Using local runner
python run_local.py --mode backtest --start-date 2024-01-01 --end-date 2024-01-31
```

**Output**: Results are saved to `data/results/` directory as CSV files.

### Paper Trading Mode (Recommended for Validation)

Paper trading uses real-time market data but simulates execution. Perfect for validating your strategy before going live.

```bash
# Via Flask (with HTTP endpoints)
export TRADING_MODE=paper
python main_live.py

# Direct execution (single run)
python run_local.py --mode paper
```

**Requirements**: Supabase account and credentials

### Live Trading Mode (Production Only)

⚠️ **WARNING**: Only use after thorough paper trading validation (2-4 weeks minimum).

```bash
# Via Flask
export TRADING_MODE=live
python main_live.py

# Direct execution
python run_local.py --mode live
```

**Requirements**: Broker API keys (e.g., Alpaca)

## Configuration

Edit config files in `config/` directory:
- `backtest.yaml` - Backtest settings
- `paper.yaml` - Paper trading settings
- `live.yaml` - Live trading settings (requires broker API keys)

### Key Configuration Options

**Trading Parameters**:
- `target_assets`: List of assets to trade (e.g., ['SPY', 'QQQ', 'TQQQ'])
- `base_weights`: Default portfolio weights
- `initial_capital`: Starting capital (default: $100,000)

**Risk Limits**:
- `max_position_size`: Maximum position size (default: 0.5 = 50%)
- `max_portfolio_var`: Maximum portfolio VaR (default: 0.02 = 2%)
- `max_leverage`: Maximum leverage (default: 1.0 = no leverage)

**Transaction Costs**:
- `commission_rate`: Commission per trade (default: 0.001 = 0.1%)
- `slippage_rate`: Slippage assumption (default: 0.0005 = 0.05%)

## Project Structure

```
.
├── trading/              # Main trading system
│   ├── core/            # Shared core logic
│   ├── data/            # Data loaders
│   ├── execution/       # Executors
│   ├── state/           # State stores
│   ├── mode_factory.py  # Factory for mode-specific components
│   ├── runner.py        # Unified runner
│   └── config.py        # Config loader
├── config/              # Configuration files
├── backtest/            # Backtest scripts
├── main_live.py         # Flask HTTP handler
├── Dockerfile           # Docker container
└── supabase_schema.sql  # Database schema
```

## Troubleshooting

### Import Errors

Make sure you're running from the workspace root:
```bash
cd /Users/kody/base/MarketNews
python test_trading_system.py
```

### Missing Data Files

The backtest mode will generate synthetic data if CSV files aren't found. To use real data:
1. Create `data/` directory
2. Add CSV files with columns: `date`, `close`, `volume`
3. Name files like `spy_historical.csv`, `qqq_historical.csv`, etc.

**Example data file format**:
```csv
date,close,volume
2024-01-01,450.25,50000000
2024-01-02,451.50,52000000
...
```

### Supabase Connection Issues

- Verify your Supabase URL and key are correct
- Check that RLS policies are set up correctly
- Ensure the schema has been run (run `supabase_schema.sql` in SQL Editor)
- Test connection: Check Supabase dashboard for connection status

### Paper Trading Not Working

- Make sure Supabase is configured (URL and key set)
- Check that `TRADING_MODE=paper` is set
- Verify the state store is initialized correctly
- Check logs for specific error messages

### Environment Variables Not Loading

If environment variables aren't being picked up:
```bash
# Check current values
echo $SUPABASE_URL
echo $TRADING_MODE

# Set explicitly
export SUPABASE_URL="https://your-project.supabase.co"
export SUPABASE_KEY="your-anon-key"
export TRADING_MODE=paper

# Or use a .env file (requires python-dotenv)
```

### Port Already in Use

If port 8080 is already in use:
```bash
# Use a different port
PORT=8081 python main_live.py

# Or find and kill the process using port 8080
lsof -ti:8080 | xargs kill
```

## Development Workflow

### Recommended Development Process

1. **Develop & Test Locally** (Backtest Mode)
   ```bash
   # Test your changes with backtests
   python backtest/run_backtest.py --start-date 2024-01-01 --end-date 2024-01-31
   ```

2. **Validate with Paper Trading** (Paper Mode)
   ```bash
   # Run paper trading to validate with real-time data
   python run_local.py --mode paper
   ```

3. **Deploy to Cloud** (Optional)
   - Follow `DEPLOYMENT_GUIDE.md` for Cloud Run deployment
   - Use Cloud Scheduler for automated daily runs

4. **Monitor & Validate** (2-4 weeks minimum)
   - Compare paper trading results to backtest
   - Verify strategy performance matches expectations

5. **Go Live** (Only after validation)
   - Switch to live mode only after thorough validation
   - Start with small position sizes

### Running Tests

```bash
# Run all tests
python test_trading_system.py

# Run specific component tests
python -m pytest trading/tests/  # if pytest tests exist
```

### Adding New Features

1. **Core logic** goes in `trading/core/`
   - Regime detection, portfolio engine, risk management
   - These are shared across all modes

2. **Mode-specific adapters** go in:
   - `trading/data/` - Data loaders
   - `trading/execution/` - Executors
   - `trading/state/` - State stores

3. **Update `mode_factory.py`** to wire up new components

4. **Test in all modes**:
   - Backtest mode (no external dependencies)
   - Paper mode (requires Supabase)
   - Live mode (requires broker API)

### Debugging

**Enable debug logging**:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

**Or set environment variable**:
```bash
export LOG_LEVEL=DEBUG
python run_local.py --mode backtest
```

**View detailed output**:
```bash
# Verbose backtest output
python backtest/run_backtest.py --start-date 2024-01-01 --end-date 2024-01-31 --verbose
```

## Local vs Cloud Deployment

### When to Use Local

- ✅ **Development**: Testing new features and strategies
- ✅ **Backtesting**: Running historical analysis
- ✅ **Debugging**: Troubleshooting issues
- ✅ **Learning**: Understanding how the system works

### When to Use Cloud (Cloud Run)

- ✅ **Production**: Automated daily trading
- ✅ **Monitoring**: 24/7 availability
- ✅ **Scalability**: Handle multiple strategies
- ✅ **Reliability**: Automatic restarts and monitoring

### Hybrid Approach (Recommended)

1. **Develop locally** with backtest mode
2. **Validate locally** with paper mode
3. **Deploy to cloud** for production paper trading
4. **Monitor and validate** for 2-4 weeks
5. **Switch to live** only after validation

## Next Steps

1. ✅ **Test locally** with backtest mode
2. ✅ **Set up Supabase** for paper trading
3. ✅ **Run paper trading locally** for validation
4. ✅ **Deploy to Cloud Run** (optional, see `DEPLOYMENT_GUIDE.md`)
5. ⚠️ **Only then consider live trading** (after 2-4 weeks validation)
