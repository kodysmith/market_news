# AI Quant Trading System v1

A comprehensive, research-driven quantitative trading system that generates, tests, and validates trading strategies using modern data science and machine learning techniques.

## 🚀 Quick Start

```bash
# Check system status
python main.py status

# Run data ingestion pipeline
python main.py data

# Test the TQQQ reference strategy
python main.py tqqq

# Run a research cycle
python main.py research --objectives trend_following volatility_targeting

# Run full system cycle
python main.py cycle
```

## 🏗️ Architecture

```
[Data Ingestion] -> [Feature/Label Builder] -> [Hypothesis Generator]
                                             -> [Backtest Engine]
                                             -> [Robustness Lab]
                                             -> [Risk & Portfolio]
                                             -> [Live Paper Trader]
                                             -> [Reporting & Notes]
```

## 📁 Project Structure

```
QuantEngine/
├── config/                 # Configuration files
├── data/                   # Data storage (raw, processed, cache)
├── engine/                 # Core trading engine modules
│   ├── data_ingestion/     # Market data acquisition
│   ├── feature_builder/    # Technical indicators & signals
│   ├── backtest_engine/    # Vectorized backtesting
│   ├── robustness_lab/     # Validation & testing
│   ├── risk_portfolio/     # Risk management (TODO)
│   ├── live_paper_trader/  # Live trading interface (TODO)
│   └── reporting_notes/    # Research report generation
├── research/               # Research agent & hypothesis generation
├── live/                   # Live trading deployment
├── reports/                # Generated research reports
├── tests/                  # Unit tests
└── utils/                  # Shared utilities & DSL
```

## 🎯 Key Features

### ✅ Implemented
- **Strategy DSL**: JSON-based strategy specification language
- **Data Ingestion**: Yahoo Finance & Alpha Vantage integration
- **Feature Builder**: Technical indicators & microstructure features
- **Backtest Engine**: Vectorized pandas/NumPy backtesting
- **Robustness Lab**: Walk-forward CV, regime analysis, statistical tests
- **Research Agent**: Automated hypothesis generation & testing
- **Report Generator**: Automated research notes with charts

### 🚧 In Progress
- **Risk & Portfolio**: Advanced risk management & portfolio optimization
- **Live Paper Trader**: Live trading execution & monitoring

## 📊 Strategy DSL Example

```json
{
  "name": "tqqq_regime_puts_v1",
  "universe": ["TQQQ"],
  "signals": [
    {
      "type": "MA_cross",
      "name": "trend_filter",
      "params": {"fast": 20, "slow": 200},
      "rule": "fast>slow"
    }
  ],
  "entry": {"all": ["trend_filter.rule"]},
  "sizing": {
    "vol_target_ann": 0.15,
    "max_weight": 1.0
  },
  "overlays": {
    "puts": {
      "target_delta": -0.2,
      "ratio": 0.5,
      "budget_pct_month": 0.01
    }
  },
  "risk": {"max_dd_pct": 0.25}
}
```

## 🔧 Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Optional: Install development dependencies
pip install -r requirements-dev.txt
```

## 🧪 Testing

```bash
# Run unit tests
python -m pytest tests/

# Run specific test
python -m pytest tests/test_strategy_dsl.py
```

## 📈 Usage Examples

### Backtest a Strategy

```python
from utils.strategy_dsl import StrategyValidator, EXAMPLE_TQQQ_STRATEGY
from engine.backtest_engine.backtester import VectorizedBacktester

# Validate strategy
spec = StrategyValidator.validate_spec(EXAMPLE_TQQQ_STRATEGY)

# Run backtest
backtester = VectorizedBacktester(config)
result = backtester.run_backtest(spec, market_data)

print(f"Sharpe: {result.metrics['sharpe']:.2f}")
```

### Generate Research Note

```python
from engine.reporting_notes.report_generator import ReportGenerator

generator = ReportGenerator(config)
report_path = generator.generate_research_note(spec, result, robustness_report, market_data)
```

## 🎯 Research Agent

The research agent automatically:
1. Analyzes market data and features
2. Generates strategy hypotheses
3. Backtests and validates strategies
4. Produces research reports for approved strategies

```python
from research.research_agent import ResearchAgent

agent = ResearchAgent(config)
results = agent.run_research_cycle(['trend_following', 'mean_reversion'])
```

## 📊 Performance Metrics

Strategies are evaluated on:
- **Sharpe Ratio**: Risk-adjusted returns
- **Maximum Drawdown**: Peak-to-trough decline
- **Win Rate**: Percentage of winning trades
- **OOS Performance**: Out-of-sample validation
- **Regime Robustness**: Performance across market conditions

## 🚨 Risk Management

- Volatility targeting (10-15% annualized)
- Maximum drawdown limits (20-25%)
- Position size constraints
- Options overlays for hedging
- Circuit breakers on extreme moves

## 🤖 AI Integration

The system uses AI for:
- **Hypothesis Generation**: Creating new strategy ideas
- **Parameter Optimization**: Bayesian optimization of strategy parameters
- **Research Notes**: Automated report writing
- **Signal Enhancement**: ML-based signal processing

## 📋 Roadmap

### Phase 1 ✅ (Current)
- Core backtesting infrastructure
- Strategy DSL and validation
- Basic research agent
- Automated reporting

### Phase 2 🚧 (Next)
- Advanced risk management
- Live paper trading
- Options chain integration
- Intraday capabilities

### Phase 3 📅 (Future)
- Multi-asset portfolios
- Alternative data integration
- Real-time execution
- Performance attribution

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ⚠️ Disclaimer

This software is for educational and research purposes only. Not intended for actual trading without thorough testing and risk assessment. Past performance does not guarantee future results.

