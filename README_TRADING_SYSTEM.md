# Live Trading System - Production Architecture

Modular trading system supporting backtest, paper, and live trading modes with the same core logic.

## Architecture

The system uses a modular architecture where:
- **Core modules** (regime detection, portfolio engine, hedging, risk management) are shared across all modes
- **Adapters** (data loaders, executors, state stores) are mode-specific
- Same trading logic runs in all modes - only data sources and execution methods differ

## Directory Structure

```
trading/
├── core/              # Shared core logic
│   ├── regime_detector.py
│   ├── portfolio_engine.py
│   ├── hedging.py
│   └── risk_manager.py
├── data/              # Mode-specific data loaders
│   ├── interface.py
│   ├── backtest.py    # Historical file loader
│   └── live.py        # Real-time API loader
├── execution/         # Mode-specific executors
│   ├── interface.py
│   ├── backtest.py    # In-memory execution
│   ├── paper.py       # Supabase-backed paper trading
│   └── real.py        # Broker API execution
├── state/             # Mode-specific state stores
│   ├── interface.py
│   ├── backtest.py    # CSV/Parquet output
│   └── supabase.py    # Database operations
├── mode_factory.py    # Creates mode-specific adapters
├── runner.py          # Unified strategy runner
└── config.py          # Configuration loader

config/
├── backtest.yaml
├── paper.yaml
└── live.yaml

backtest/
└── run_backtest.py    # Backtest runner script
```

## Three Operating Modes

### 1. Backtest Mode
- **Purpose**: Strategy development and validation on historical data
- **Data**: Historical CSV/Parquet files
- **Execution**: In-memory positions, simulated fills
- **State**: CSV/Parquet output files
- **Runner**: `python backtest/run_backtest.py --start-date 2020-01-01 --end-date 2024-12-31`

### 2. Paper Trading Mode
- **Purpose**: Validate strategy with real-time data BEFORE using real money
- **Data**: Real-time market prices (yfinance)
- **Execution**: Simulated fills, positions stored in Supabase
- **State**: Supabase database
- **Runner**: Cloud Run service with `mode='paper'` or Flask app locally

### 3. Live Trading Mode
- **Purpose**: Production trading with real money
- **Data**: Real-time market prices (broker API)
- **Execution**: Real broker API orders (Alpaca, IBKR, etc.)
- **State**: Supabase database
- **Runner**: Cloud Run service with `mode='live'`

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Supabase (for paper/live modes)

1. Create a Supabase project
2. Run the SQL schema: `supabase_schema.sql`
3. Get your Supabase URL and anon key
4. Set environment variables:
   ```bash
   export SUPABASE_URL="https://your-project.supabase.co"
   export SUPABASE_KEY="your-anon-key"
   ```

### 3. Configure Broker (for live mode)

Set environment variables:
```bash
export ALPACA_API_KEY="your-api-key"
export ALPACA_API_SECRET="your-api-secret"
export ALPACA_BASE_URL="https://api.alpaca.markets"  # or paper URL
```

## Usage

### Backtest Mode

**Option 1: Using the local runner (recommended)**
```bash
python run_local.py --mode backtest --start-date 2020-01-01 --end-date 2024-12-31
```

**Option 2: Using the backtest runner**
```bash
python backtest/run_backtest.py --start-date 2020-01-01 --end-date 2024-12-31
```

**Option 3: Using Python directly**
```python
from trading.runner import run_strategy

results = run_strategy(
    mode='backtest',
    config_path='config',
    start_date='2020-01-01',
    end_date='2024-12-31'
)
```

### Paper Trading Mode

**Option 1: Using the local runner (recommended for local testing)**
```bash
export SUPABASE_URL="https://your-project.supabase.co"
export SUPABASE_KEY="your-anon-key"
python run_local.py --mode paper
```

**Option 2: Using Python directly**
```python
from trading.runner import run_strategy

results = run_strategy(
    mode='paper',
    config_path='config'
)
```

**Option 3: Via Flask (for HTTP endpoints)**
```bash
export TRADING_MODE=paper
python main_live.py
```

Then POST to `/run`:
```bash
curl -X POST http://localhost:8080/run \
  -H "Content-Type: application/json" \
  -d '{"mode": "paper"}'
```

### Live Trading Mode

```python
from trading.runner import run_strategy

results = run_strategy(
    mode='live',
    config_path='config'
)
```

**⚠️ WARNING**: Live mode executes real trades with real money. Only use after thorough paper trading validation.

## Configuration

Configuration files are in `config/` directory:
- `backtest.yaml` - Backtest configuration
- `paper.yaml` - Paper trading configuration
- `live.yaml` - Live trading configuration

Secrets (Supabase keys, broker API keys) are loaded from:
1. Environment variables (preferred for local)
2. GCP Secret Manager (for Cloud Run)

## Cloud Deployment

### Docker Build

```bash
docker build -t trading-system .
```

### Cloud Run Deployment

```bash
gcloud run deploy trading-system \
  --source . \
  --platform managed \
  --region us-central1 \
  --set-env-vars TRADING_MODE=paper \
  --set-secrets SUPABASE_URL=supabase-url:latest,SUPABASE_KEY=supabase-key:latest
```

### Cloud Scheduler

Create a job to trigger daily at 3:50pm ET (20:50 UTC):
```bash
gcloud scheduler jobs create http trading-daily \
  --schedule="50 20 * * 1-5" \
  --uri="https://trading-system-xxx.run.app/run" \
  --http-method=POST \
  --headers="Content-Type=application/json" \
  --message-body='{"mode":"paper"}'
```

## Database Schema

The Supabase schema includes:
- `runs` - Trading run records
- `positions` - Current positions
- `orders` - Executed orders
- `regime_history` - Regime detection history
- `run_logs` - Execution logs

See `supabase_schema.sql` for full schema with indexes and RLS policies.

## Key Features

1. **Single Source of Truth**: Same logic for all modes
2. **Safe Validation**: Paper mode validates before risking capital
3. **Easy Testing**: Backtest locally, paper in cloud
4. **Consistent Behavior**: No discrepancies between modes
5. **Maintainability**: Fix bugs once, applies everywhere

## Development

### Running Locally

**Quick Start** (see `SETUP_LOCAL.md` for details):

```bash
# 1. Test the system
python test_trading_system.py

# 2. Run backtest (no external dependencies)
python run_local.py --mode backtest --start-date 2024-01-01 --end-date 2024-01-31

# 3. Run paper trading (requires Supabase)
export SUPABASE_URL="https://your-project.supabase.co"
export SUPABASE_KEY="your-anon-key"
python run_local.py --mode paper

# 4. Run Flask app (for HTTP endpoints)
export TRADING_MODE=paper
python main_live.py
curl http://localhost:8080/health
```

### Testing

Test each mode:
```bash
# Backtest (no dependencies)
python run_local.py --mode backtest --start-date 2024-01-01 --end-date 2024-01-31

# Paper (requires Supabase)
export SUPABASE_URL="..." SUPABASE_KEY="..."
python run_local.py --mode paper

# Or via Flask
export TRADING_MODE=paper
python main_live.py
curl -X POST http://localhost:8080/run -H "Content-Type: application/json" -d '{"mode":"paper"}'
```

## Next Steps

1. Set up Supabase database
2. Configure GCP Cloud Run (optional)
3. Set up Cloud Scheduler (optional)
4. Create dashboard frontend (optional)
5. Run paper trading for 2-4 weeks
6. Validate results vs backtest
7. Switch to live mode (after validation)
