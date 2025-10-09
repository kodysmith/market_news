# SQQQ-QQQ Wheel Strategy Backtest

## 🎯 Strategy Overview

A delta-hedged short put wheel strategy on QQQ with SQQQ call protection, designed to generate consistent income while managing downside risk.

### Strategy Components

1. **QQQ Put Selling (Income Generation)**
   - Sell puts weekly at 30% delta (~70% probability OTM)
   - Target 14 DTE (days to expiration)
   - Deploy 1% of capital per trade
   - Close at 50% profit target
   - Max allocation: 75% of capital

2. **Wheel on Assignment**
   - When assigned QQQ shares at strike
   - Immediately sell covered calls at 30% delta
   - Continue until shares are called away
   - Generates additional premium income

3. **SQQQ Call Hedge (Risk Management)**
   - Buy ATM calls on SQQQ (inverse 3x QQQ) every Friday
   - Sell Monday morning
   - Position sized to delta-hedge QQQ exposure
   - Uses 25% capital reserve

## 📊 Backtest Results (2020-2025)

### Performance Summary

| Metric | Value |
|--------|-------|
| **Total Return** | **89.2%** |
| **CAGR** | **13.6%** |
| **Sharpe Ratio** | **0.84** |
| **Sortino Ratio** | **1.29** |
| **Max Drawdown** | **-23.8%** |
| **Annualized Volatility** | **17.0%** |
| **Win Rate (Days)** | **55.4%** |

### Trading Activity

- **Total Trades**: 263
- **Put Assignments**: 18 (27.7% of puts sold)
- **Call Assignments**: 15 (83.3% of covered calls)
- **Successful Put Closes**: 47 (72.3% closed for profit)

### Premium & P&L

- **Total Premium Collected**: $84,919
  - Put Premium: $60,011
  - Call Premium: $24,908
- **SQQQ Hedge P&L**: $5,235 (net positive)

### Final Position

- **Portfolio Value**: $189,235
- **Cash**: $36,444
- **QQQ Shares**: 300 shares

## 🔧 Implementation

### Files Created

1. **`sqqq_qqq_wheel_strategy.py`** (31KB)
   - Main backtester with Black-Scholes pricing
   - Position tracking and management
   - Assignment detection and wheel logic
   - Delta hedging calculations

2. **`run_sqqq_qqq_backtest.py`** (Runner script)
   - Executes backtest
   - Generates visualizations
   - Exports detailed reports

### Generated Reports

1. **`sqqq_qqq_summary.txt`** - Full performance summary
2. **`sqqq_qqq_trades.csv`** - Detailed trade log (263 trades)
3. **`sqqq_qqq_portfolio.csv`** - Daily portfolio values (1,258 days)
4. **`sqqq_qqq_backtest_results.png`** - Comprehensive charts

## 🚀 Usage

### Run Complete Backtest

```bash
cd /mnt/4tb/stock_scanner/market_news/backtesting
python run_sqqq_qqq_backtest.py
```

### Customize Parameters

```python
from sqqq_qqq_wheel_strategy import SQQQQQQWheelStrategy

strategy = SQQQQQQWheelStrategy(
    initial_capital=100000,        # Starting capital
    max_qqq_allocation=0.75,       # 75% max for QQQ
    weekly_capital_pct=0.01,       # 1% per trade
    put_delta_target=-0.30,        # 30% delta puts
    put_dte=14,                    # 14 days to expiration
    call_delta_target=0.30,        # 30% delta calls
    call_dte=30,                   # 30 DTE for covered calls
    sqqq_call_dte=30,              # 30 DTE for SQQQ hedges
    start_date='2020-01-01',
    end_date='2025-01-01'
)

results = strategy.run_backtest()
strategy.print_summary(results)
```

## 📈 Key Features

### Risk Management
- **Capital Limits**: Never exceed 75% allocation to QQQ
- **Delta Hedging**: SQQQ calls offset QQQ exposure on weekends
- **Profit Taking**: Close puts at 50% profit (prevents greedy losses)
- **Position Sizing**: 1% capital per trade (consistent risk)

### Income Generation
- **Put Premium**: Primary income from selling puts
- **Call Premium**: Additional income from covered calls after assignment
- **Hedge Efficiency**: SQQQ hedges added $5,235 net profit

### Automation Ready
- Black-Scholes pricing for option valuation
- Delta calculations for strike selection
- Assignment detection
- Position tracking and management

## 📊 Performance Analysis

### Monthly Returns
- **Average**: 1.18% per month
- **Positive Months**: 36 (60%)
- **Negative Months**: 24 (40%)

### Daily Statistics
- **Best Day**: +15.4% (Feb 24, 2020 - hedge payoff)
- **Worst Day**: -3.9% (Sep 13, 2022)
- **Positive Days**: 691 (55.4%)

### Risk-Adjusted Performance
- **Sharpe Ratio**: 0.84 (good risk-adjusted returns)
- **Sortino Ratio**: 1.29 (excellent downside risk management)
- **Calmar Ratio**: 0.57 (CAGR / Max DD)

## 🎓 Strategy Insights

### What Worked Well
1. **High win rate** on put sales (72.3% closed profitably)
2. **Consistent premium collection** across market conditions
3. **SQQQ hedges** prevented larger drawdowns (especially 2020, 2022)
4. **Wheel mechanics** efficiently managed assignments

### Areas of Note
1. **Max drawdown** of -23.8% occurred during 2022 bear market
2. **Assignment rate** of 27.7% is reasonable for 30% delta
3. **Hedge costs** were offset by occasional big payoffs
4. **Capital efficiency** maintained with proper allocation limits

## 🔍 Next Steps

### Potential Optimizations
1. **Dynamic delta targeting** based on VIX levels
2. **Variable position sizing** based on market regime
3. **Roll management** for challenged positions
4. **Tax-loss harvesting** strategies

### Live Trading Considerations
1. Use real-time options data (not Black-Scholes estimates)
2. Account for bid-ask spreads and slippage
3. Monitor margin requirements closely
4. Consider early assignment risk

## 📁 File Structure

```
backtesting/
├── sqqq_qqq_wheel_strategy.py    # Main backtester
├── run_sqqq_qqq_backtest.py      # Runner script
├── sqqq_qqq_summary.txt          # Performance report
├── sqqq_qqq_trades.csv           # Trade log
├── sqqq_qqq_portfolio.csv        # Daily values
├── sqqq_qqq_backtest_results.png # Visualizations
└── SQQQ_QQQ_STRATEGY_README.md   # This file
```

## ⚠️ Disclaimers

1. **Past performance does not guarantee future results**
2. **Backtests use estimated pricing** (real spreads may differ)
3. **Transaction costs** are approximated
4. **Slippage and market impact** not fully modeled
5. **Options carry significant risk** - use proper risk management
6. **Test with paper trading** before live deployment

## 📝 License & Usage

This backtest is for educational and research purposes. Use at your own risk.

---

**Created**: October 2025  
**Version**: 1.0  
**Strategy Type**: Delta-hedged wheel with inverse leverage hedge

