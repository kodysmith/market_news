# Quant Backtesting System v2

A clean, Python-first quant stack that eats data, cranks signals, and pushes trades—repeatable and auditable. Built for serious alpha generation with real risk controls.

## 🏗️ Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Sources  │───▶│   ETL Flows     │───▶│  Feature Factory │
│                 │    │  (Prefect)      │    │   (DuckDB)       │
│ • Yahoo Finance │    │ • ingest_equity │    │ • Regime flags  │
│ • Tradier API   │    │ • ingest_options│    │ • Options Greeks │
│ • FRED API      │    │ • ingest_macro  │    │ • Macro factors │
│ • CFTC COT      │    │                 │    │ • Positioning    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                         │
┌─────────────────┐    ┌─────────────────┐              │
│ Backtest Engines│    │ Risk Management│◀─────────────┘
│                 │    │                 │
│ • VectorBT      │    │ • Kelly sizing  │
│   (Equities)    │    │ • Vol targeting │
│ • Backtrader    │    │ • Drawdown ctrl │
│   (Options)     │    │ • Stress tests  │
└─────────────────┘    └─────────────────┘
                                                         │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ MLflow Tracking │───▶│   Execution     │    │   Production   │
│                 │    │   Gateways      │    │   Models       │
│ • Experiments   │    │                 │    │               │
│ • Metrics       │    │ • Alpaca API    │    │ • Model Reg    │
│ • Artifacts     │    │ • IBKR API      │    │ • Staging      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🚀 Quick Start

### 1. Installation

```bash
cd market_news_app/backtesting_v2
pip install -r requirements.txt
```

### 2. Data Ingestion

```python
from flows.ingest_equity import ingest_equity_flow
from flows.ingest_options import ingest_options_flow
from flows.ingest_macro import ingest_macro_flow

# Ingest equity data
ingest_equity_flow(
    symbols=['SPY', 'QQQ', 'AAPL'],
    start_date='2020-01-01',
    end_date='2024-01-01'
)

# Ingest options data (requires API key)
ingest_options_flow(
    symbols=['SPY', 'QQQ'],
    data_provider='tradier',
    api_key='your_tradier_key'
)

# Ingest macroeconomic data
ingest_macro_flow(fred_api_key='your_fred_key')
```

### 3. Feature Engineering

```python
from features.build import build_features_flow

# Build features for all symbols
results = build_features_flow(symbols=['SPY', 'QQQ', 'AAPL'])
```

### 4. Backtesting

```python
from backtests.equity_vectorbt import equity_backtest_flow
from backtests.options_bt import options_backtest_flow

# Define equity strategies
equity_strategies = [
    {
        'name': 'momentum_sma',
        'momentum': {'sma_crossover': {'fast': 20, 'slow': 50}},
        'regime_aware': {'trend_filter': True}
    }
]

# Run equity backtests
equity_results = equity_backtest_flow(
    symbols=['SPY', 'QQQ'],
    strategy_configs=equity_strategies,
    experiment_name='equity_strategies'
)

# Define options strategies
options_strategies = [
    {
        'name': 'CoveredCallsStrategy',
        'params': {'call_delta_target': 0.15}
    }
]

# Run options backtests
options_results = options_backtest_flow(
    symbols=['SPY'],
    strategies=options_strategies,
    experiment_name='options_strategies'
)
```

### 5. Experiment Tracking

```python
from mlflow.tracking import create_experiment_comparator

# Compare experiment results
comparator = create_experiment_comparator('equity_strategies')
comparison_df = comparator.compare_strategies()

# Generate performance report
comparator.generate_performance_report('performance_report.html')

# Find best hyperparameters
best_params = comparator.find_best_hyperparameters('momentum_sma')
```

### 6. Live Execution (Paper Trading)

```python
from execution.alpaca_client import create_alpaca_client

# Create execution client (paper trading)
client = create_alpaca_client(paper=True)

# Get account info
account = client.get_account_info()

# Place a test order
order = client.place_market_order('SPY', 10, 'buy')
```

## 📊 Data Sources

### Free Tier
- **Yahoo Finance**: Equity prices (yfinance)
- **CBOE**: VIX data (public API)
- **FRED**: Macro data (limited API)
- **CFTC**: COT reports (public data)

### Pro Tier
- **Tradier**: Options chains + Greeks ($10/month)
- **Polygon**: Options data ($20/month)
- **Alpaca**: Commission-free execution (paper/live)

### Enterprise Tier
- **OPRA**: Full options data (Databento/dxFeed)
- **IBKR**: Institutional execution
- **RavenPack**: News sentiment
- **ORATS**: Advanced options analytics

## 🎯 Strategy Examples

### Equity Strategies

```python
# Momentum + Mean Reversion Combo
momentum_mr_strategy = {
    'name': 'combo_strategy',
    'momentum': {
        'sma_crossover': {'fast': 20, 'slow': 50},
        'rsi': {'period': 14, 'overbought': 70, 'oversold': 30}
    },
    'mean_reversion': {
        'bollinger_bands': {'period': 20, 'std': 2}
    },
    'regime_aware': {
        'risk_regime_filter': {'max_level': 2},
        'trend_filter': True
    }
}
```

### Options Strategies

```python
# Covered Calls with Dynamic Hedging
covered_calls_strategy = {
    'name': 'CoveredCallsStrategy',
    'params': {
        'call_delta_target': 0.15,
        'call_dte_min': 30,
        'call_dte_max': 60,
        'cover_ratio_max': 0.6,
        'risk_regime_max': 2
    }
}

# Protective Puts with Risk Management
protective_puts_strategy = {
    'name': 'ProtectivePutsStrategy',
    'params': {
        'put_delta_target': -0.20,
        'put_dte_target': 90,
        'protection_ratio_min': 0.4,
        'protection_ratio_max': 0.8
    }
}
```

## 📈 Risk Management

### Portfolio-Level Controls

```python
from risk.pm import create_risk_manager, create_portfolio_manager

# Create risk management components
risk_mgr = create_risk_manager(max_drawdown=0.15, max_volatility=0.20)
port_mgr = create_portfolio_manager(risk_mgr)

# Calculate position size
capital = 100000
volatility = 0.25
position_size = risk_mgr.calculate_position_size(capital, volatility)

# Optimize portfolio
returns_data = # Your returns DataFrame
weights = port_mgr.optimize_portfolio(returns_data, target_return=0.15)
```

### Execution Risk Controls

```python
from execution.alpaca_client import create_alpaca_client, create_alpaca_risk_manager

client = create_alpaca_client(paper=True)
risk_mgr = create_alpaca_risk_manager(client)

# Validate order before execution
valid, reason = risk_mgr.validate_order('SPY', 100, 'buy', 'market')
if valid:
    order = client.place_market_order('SPY', 100, 'buy')
```

## 🔬 Experiment Tracking

### MLflow Integration

```python
from mlflow.tracking import create_backtest_tracker

tracker = create_backtest_tracker('quant_strategies')

with tracker.start_run('momentum_v1'):
    # Log parameters
    tracker.log_backtest_params({
        'fast_period': 20,
        'slow_period': 50,
        'stop_loss': 0.10
    })

    # Run backtest and log results
    portfolio, metrics = run_backtest(strategy_config)
    tracker.log_portfolio_results(portfolio, 'vectorbt')
    tracker.log_backtest_metrics(metrics)
```

### Performance Analysis

```python
from mlflow.tracking import create_experiment_comparator

comparator = create_experiment_comparator('quant_strategies')

# Generate comprehensive report
comparator.generate_performance_report('strategy_comparison.html')

# Statistical significance testing
best_strategy = comparator.compare_strategies('sharpe_ratio')
```

## 🚦 Production Deployment

### Model Registry

```python
from mlflow.tracking import create_model_registry

registry = create_model_registry()

# Register best performing model
model_version = registry.register_best_model(
    experiment_name='quant_strategies',
    strategy='momentum_sma'
)

# Promote to production
registry.transition_model_stage(
    model_name='momentum_sma_production',
    version=model_version.version,
    stage='Production'
)
```

### Live Execution

```python
# Alpaca live execution
from execution.alpaca_client import create_alpaca_client

live_client = create_alpaca_client(paper=False)  # Live trading!

# IBKR institutional execution
from execution.ib_client import create_ibkr_client

ib_client = create_ibkr_client(port=7496)  # Live IBKR port

# Execute portfolio rebalancing
target_weights = {'SPY': 0.4, 'QQQ': 0.3, 'TLT': 0.3}
orders = live_client.execute_portfolio_trades(target_weights)
```

## 📋 Configuration Files

### Environment Variables

```bash
# Data APIs
export FRED_API_KEY="your_fred_key"
export TRADIER_API_KEY="your_tradier_key"
export POLYGON_API_KEY="your_polygon_key"

# Execution APIs
export ALPACA_API_KEY="your_alpaca_key"
export ALPACA_API_SECRET="your_alpaca_secret"

# MLflow
export MLFLOW_TRACKING_URI="file:./mlruns"
```

### Config File Example

```json
{
  "data": {
    "equity_symbols": ["SPY", "QQQ", "DIA", "IWM"],
    "options_symbols": ["SPY", "QQQ"],
    "start_date": "2010-01-01",
    "end_date": "2024-01-01"
  },
  "risk": {
    "max_drawdown": 0.15,
    "max_volatility": 0.20,
    "max_leverage": 2.0,
    "max_concentration": 0.10
  },
  "execution": {
    "paper_trading": true,
    "max_order_size": 100000,
    "min_order_size": 100
  }
}
```

## 🔧 Development Workflow

1. **Data Pipeline**: Run ETL flows to ingest latest data
2. **Feature Engineering**: Build and test new features
3. **Strategy Development**: Create and backtest strategies
4. **Risk Testing**: Validate with stress tests and walk-forward analysis
5. **Paper Trading**: Deploy to paper account for live testing
6. **Production**: Promote successful strategies to live execution

## 📚 Key Components

### Data Ingestion (`flows/`)
- `ingest_equity.py`: Yahoo Finance equity data
- `ingest_options.py`: Options chains and Greeks
- `ingest_macro.py`: FRED + COT macroeconomic data

### Feature Factory (`features/`)
- `build.py`: DuckDB-powered feature engineering
- Regime detection, options ladders, macro factors

### Backtest Engines (`backtests/`)
- `equity_vectorbt.py`: Vectorized equity backtesting
- `options_bt.py`: Event-driven options strategies

### Risk Management (`risk/`)
- `pm.py`: Kelly sizing, portfolio optimization, drawdown control

### Experiment Tracking (`mlflow_tracking/`)
- `tracking.py`: MLflow integration, experiment comparison

### Execution (`execution/`)
- `alpaca_client.py`: Alpaca API execution
- `ib_client.py`: IBKR institutional execution

## 🎯 Performance Targets

- **10%+ Annual Returns**: With controlled drawdowns
- **Sharpe Ratio > 1.5**: Risk-adjusted performance
- **Max Drawdown < 15%**: Capital preservation
- **99% Uptime**: Reliable execution
- **Sub-millisecond Latency**: High-frequency capable

## 🚀 Scaling Considerations

### Data Layer
- S3/Parquet for scalable storage
- DuckDB for fast queries
- Prefect for orchestration

### Compute Layer
- VectorBT for parallel backtesting
- GPU acceleration for ML features
- Distributed execution with Prefect

### Execution Layer
- Alpaca for retail execution
- IBKR for institutional volumes
- Smart order routing

---

**Built for quants who demand performance with precision. Eat data. Crank signals. Push trades.**
