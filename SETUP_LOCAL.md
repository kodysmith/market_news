# Local Setup Guide

Quick guide to set up and run the trading system locally.

## Prerequisites

- Python 3.11+
- pip
- (Optional) Supabase account for paper/live modes

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

```bash
python backtest/run_backtest.py --start-date 2024-01-01 --end-date 2024-01-31
```

Note: This uses synthetic data if historical CSV files aren't available.

### 4. Set Up Paper Trading (Optional)

1. Create a Supabase project at https://supabase.com
2. Run the SQL schema from `supabase_schema.sql` in the Supabase SQL editor
3. Get your Supabase URL and anon key
4. Set environment variables:

```bash
export SUPABASE_URL="https://your-project.supabase.co"
export SUPABASE_KEY="your-anon-key"
export TRADING_MODE=paper
```

5. Run the Flask app:

```bash
python main_live.py
```

6. Test the endpoint:

```bash
curl http://localhost:8080/health
curl -X POST http://localhost:8080/run \
  -H "Content-Type: application/json" \
  -d '{"mode": "paper"}'
```

## Configuration

Edit config files in `config/` directory:
- `backtest.yaml` - Backtest settings
- `paper.yaml` - Paper trading settings
- `live.yaml` - Live trading settings (requires broker API keys)

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
cd /workspace
python test_trading_system.py
```

### Missing Data Files

The backtest mode will generate synthetic data if CSV files aren't found. To use real data:
1. Create `data/` directory
2. Add CSV files with columns: `date`, `close`, `volume`
3. Name files like `spy_historical.csv`, `qqq_historical.csv`, etc.

### Supabase Connection Issues

- Verify your Supabase URL and key are correct
- Check that RLS policies are set up correctly
- Ensure the schema has been run

### Paper Trading Not Working

- Make sure Supabase is configured
- Check that `TRADING_MODE=paper` is set
- Verify the state store is initialized correctly

## Next Steps

1. ✅ Test locally with backtest mode
2. ✅ Set up Supabase for paper trading
3. ✅ Run paper trading for validation
4. ⚠️ Only then consider live trading

## Development

### Running Tests

```bash
python test_trading_system.py
```

### Adding New Features

1. Core logic goes in `trading/core/`
2. Mode-specific adapters go in `trading/data/`, `trading/execution/`, or `trading/state/`
3. Update `mode_factory.py` to wire up new components

### Debugging

Enable debug logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```
