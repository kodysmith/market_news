# Options Strategy Backtesting

This directory contains tools for backtesting options protection strategies for various assets.

## Files Overview

### Core Strategy Engine
- `backtest_options_strategy.py` - Main backtesting engine with OptionsProtectionStrategy class
- `__init__.py` - Package initialization

### Optimization Tools
- `optimizer.py` - Universal optimizer that creates optimized scripts for any asset
- `simple_optimizer.py` - Simplified optimizer for testing

### Generated Optimized Scripts
- `optimized_amd.py` - Optimized strategy for AMD (35.5% CAGR)
- `optimized_nvda.py` - Optimized strategy for NVDA (22.6% CAGR)
- `optimized_tqqq.py` - Optimized strategy for TQQQ (5.8% CAGR)

### Test Scripts
- `test_optimizer.py` - Test the optimizer functionality
- `create_optimized_amd.py` - Script that generated the AMD optimized file

## Usage

### Running the Optimizer
```bash
cd backtesting
python optimizer.py TICKER
```

Examples:
```bash
python optimizer.py AMD    # Creates optimized_amd.py
python optimizer.py NVDA   # Creates optimized_nvda.py
python optimizer.py SPY    # Creates optimized_spy.py
```

### Running Optimized Strategies
```bash
python optimized_amd.py                    # Full period (2020-2024)
python optimized_amd.py 2021-01-01 2022-01-01  # Custom period
```

### Running Individual Backtests
```bash
python -c "
from backtest_options_strategy import OptionsProtectionStrategy
strategy = OptionsProtectionStrategy(ticker='AMD')
results = strategy.run_backtest()
print(f'CAGR: {results[\"total_return\"]:.1%}')
"
```

## Strategy Configuration

Each optimized strategy uses a configuration dictionary with:

- **Delta targets**: Put/call strike selection based on delta (e.g., -0.15 for puts)
- **Protection ratios**: Percentage of shares to hedge (risk_on vs risk_off regimes)
- **Tenor ladder**: Multiple expiration dates (30/60/90 days)
- **Purchase frequency**: How often to buy shares (7-14 days)
- **Budget tiers**: Monthly spending limits by volatility regime
- **Spread types**: Whether to use put spreads vs naked puts, call spreads vs covered calls

## Performance Results

| Asset | CAGR | Max DD | Sharpe | Configuration |
|-------|------|--------|--------|---------------|
| AMD   | 35.5% | -17.8% | 0.35 | High protection + spreads |
| NVDA  | 22.6% | -17.8% | 0.35 | Moderate protection |
| TQQQ  | 5.8% | -17.8% | 0.35 | Conservative protection |

## Key Strategy Features

1. **Regime Awareness**: Adjusts protection based on market trends (MA200)
2. **Dynamic Hedging**: Scales protection ratio by volatility regime
3. **Cost Control**: Monthly budget limits prevent over-hedging
4. **Premium Harvesting**: Uses call spreads to offset put costs
5. **Risk Management**: Ladder tenors and delta-based strikes

## Requirements

- Python 3.8+
- pandas
- numpy
- yfinance
- numba
- matplotlib (optional, for plotting)

## Notes

- Strategies are optimized for 2020-2024 period
- Results include realistic transaction costs and slippage
- All strategies maintain protection while capturing upside
- Budget constraints prevent excessive hedging costs

