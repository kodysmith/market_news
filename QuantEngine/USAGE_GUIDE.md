# QuantEngine Usage Guide 🚀

## Quick Start

```bash
# Activate the virtual environment
cd /Users/kody/base/MarketNews
source venv/bin/activate

# Go to QuantEngine directory
cd QuantEngine

# Run the interactive interface
python3 run_quantengine.py

# Or run specific commands directly
python3 run_quantengine.py status    # Check system status
python3 run_quantengine.py data      # Download market data
python3 run_quantengine.py tqqq      # Test TQQQ strategy
python3 run_quantengine.py cycle     # Run full system
```

## Interactive Menu

When you run `python3 run_quantengine.py` without arguments, you'll get an interactive menu:

```
🤖 AI QUANT TRADING SYSTEM v1.0
==================================================

Available Commands:
1. status     - Check system status
2. data       - Download market data
3. tqqq       - Test TQQQ reference strategy
4. research   - Run strategy research cycle
5. backtest   - Backtest a custom strategy
6. cycle      - Run full data -> research -> backtest cycle
7. test       - Run system tests
8. help       - Show this help
9. exit       - Exit

Choose command (1-9):
```

## Step-by-Step Usage

### 1. First Time Setup

```bash
# Navigate to the project
cd /Users/kody/base/MarketNews
source venv/bin/activate
cd QuantEngine

# Check everything is working
python3 run_quantengine.py status
```

You should see:
```
✅ Strategy DSL: OK
✅ Core modules: Available
```

### 2. Download Market Data

```bash
python3 run_quantengine.py data
```

This will download data for SPY, QQQ, TQQQ, and sector ETFs.

### 3. Test the Reference Strategy

```bash
python3 run_quantengine.py tqqq
```

This runs the TQQQ MA crossover strategy with options hedging that we validated in Phase 1.

**Expected Output:**
- ✅ Strategy validation successful
- ✅ Backtest completed
- 📊 Sharpe: ~1.76
- 📊 Annualized Return: ~60%
- 📊 Report saved to reports/

### 4. Run Full System Cycle

```bash
python3 run_quantengine.py cycle
```

This does everything:
1. Downloads fresh market data
2. Runs strategy research
3. Backtests top strategies
4. Generates reports

## Understanding the Output

### Performance Metrics

When strategies are tested, you'll see metrics like:

```
✅ Backtest completed
📊 Sharpe: 1.76 (excellent >1.0)
📊 Annualized Return: 60.1% (very strong)
📊 Max Drawdown: -27.8% (acceptable risk)
```

### Reports Generated

All results are saved to the `reports/` directory:

- `tqqq_regime_puts_v1_phase0_report.md` - TQQQ strategy report
- `phase1_parameter_optimization.md` - Optimization results
- `phase1_oos_validation_report.md` - Validation report

## Key Components Explained

### Strategy DSL

Strategies are defined in JSON format. Example:

```json
{
  "name": "tqqq_regime_puts_v1",
  "universe": ["TQQQ"],
  "signals": [
    {
      "type": "MA_cross",
      "params": {"fast": 20, "slow": 200}
    }
  ],
  "entry": {"all": ["signals.0.rule"]},
  "sizing": {"vol_target_ann": 0.15},
  "risk": {"max_dd_pct": 0.25}
}
```

### Data Sources

- **Yahoo Finance**: Real-time equity data
- **Mock News**: Simulated news sentiment (Phase 1)
- **Mock Filings**: Simulated SEC filing sentiment (Phase 1)

### Validation Pipeline

1. **Walk-forward CV**: Tests strategy on rolling windows
2. **OOS Gating**: 7 statistical criteria for approval
3. **Paper Trading**: Simulated execution for approved strategies

## Advanced Usage

### Custom Strategy Testing

You can modify the TQQQ strategy or create new ones by editing the strategy DSL in `utils/strategy_dsl.py`.

### Research Agent

The research agent can generate new strategies:

```python
from research.research_agent import ResearchAgent
agent = ResearchAgent(config)
results = agent.run_research_cycle(['trend_following'])
```

### Paper Trading

For approved strategies:

```python
from engine.live_paper_trader.paper_trader import PaperTradingEngine
engine = PaperTradingEngine(config)
engine.register_strategy(strategy_spec, approval_status)
```

## Troubleshooting

### Import Errors

If you see import errors, make sure you're in the QuantEngine directory and the virtual environment is activated.

### Data Download Issues

If Yahoo Finance download fails:
- Check your internet connection
- Try again later (Yahoo rate limits)
- The system will use cached data if available

### Memory Issues

For large backtests:
- Reduce the date range
- Use fewer tickers
- The system is optimized for daily data

## File Structure

```
QuantEngine/
├── config/                 # Configuration files
├── data/                   # Downloaded market data
├── engine/                 # Core system modules
├── reports/                # Generated reports & charts
├── research/               # Strategy research tools
├── tests/                  # Test scripts
├── utils/                  # Strategy DSL & utilities
├── run_quantengine.py      # Main interface (USE THIS!)
├── main.py                 # Alternative interface
└── README.md               # Documentation
```

## Performance Expectations

### TQQQ Strategy (Reference)
- **Sharpe Ratio**: 1.76 (excellent)
- **Annualized Return**: 60.1% (very strong)
- **Max Drawdown**: -27.8% (acceptable for leveraged ETF)
- **Win Rate**: 54.4%
- **OOS Validation**: ✅ Approved

### System Performance
- **Data Download**: ~30 seconds for 4 years of data
- **Backtest**: ~2 seconds for single strategy
- **Research Cycle**: ~10 seconds simulation
- **Report Generation**: ~1 second

## Next Steps

### Phase 2 Features (Coming Soon)
- Real news API integration
- Live broker execution
- Advanced ML signals
- Multi-asset portfolios

### Customization
- Add new technical indicators
- Implement custom signal types
- Create new strategy templates
- Add risk management overlays

## Support

If you encounter issues:

1. Check `python3 run_quantengine.py status`
2. Run `python3 run_quantengine.py test`
3. Check the `reports/` directory for error logs
4. Review the README.md for detailed documentation

## Example Session

```bash
# Start
cd /Users/kody/base/MarketNews
source venv/bin/activate
cd QuantEngine

# Check status
python3 run_quantengine.py status
# ✅ Strategy DSL: OK

# Download data
python3 run_quantengine.py data
# ✅ Downloaded data for 6 tickers

# Test strategy
python3 run_quantengine.py tqqq
# ✅ Backtest completed
# 📊 Sharpe: 1.76
# 📊 Annualized Return: 60.1%

# View results
ls reports/
# tqqq_regime_puts_v1_phase0_report.md
```

---

**Happy trading with QuantEngine! 🤖📈**
