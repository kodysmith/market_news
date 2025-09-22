# Phase 1 - Parameter Optimization Report

**Generated:** 2025-09-21 12:04:36

## Optimization Summary

- **Strategy:** tqqq_ma_cross_optimized
- **Metric:** sharpe
- **Combinations Tested:** 27

## Best Parameters Found

```json
{
  "signals.0.params.fast": 10,
  "signals.0.params.slow": 150,
  "sizing.vol_target_ann": 0.1
}
```

**Best Sharpe:** 1.364

## Best Strategy Performance

| Metric | Value |
|--------|-------|
| Sharpe Ratio | 1.364 |
| Total Return | 2.1289 |
| Annualized Return | 36.7% |
| Annualized Volatility | 26.9% |
| Max Drawdown | -33.6% |
| Win Rate | 38.8% |

## Parameter Sensitivity Analysis

### signals.0.params.fast

| Value | Sharpe |
|-------|--------|
| 10 | 1.262 |
| 10 | 1.262 |
| 10 | 1.262 |
| 10 | 1.364 |
| 10 | 1.364 |
| 10 | 1.364 |
| 10 | 1.161 |
| 10 | 1.161 |
| 10 | 1.161 |
| 20 | 1.240 |
| 20 | 1.240 |
| 20 | 1.240 |
| 20 | 1.347 |
| 20 | 1.347 |
| 20 | 1.347 |
| 20 | 1.347 |
| 20 | 1.347 |
| 20 | 1.347 |
| 30 | 1.204 |
| 30 | 1.204 |
| 30 | 1.204 |
| 30 | 1.338 |
| 30 | 1.338 |
| 30 | 1.338 |
| 30 | 1.255 |
| 30 | 1.255 |
| 30 | 1.255 |

### signals.0.params.slow

| Value | Sharpe |
|-------|--------|
| 100 | 1.262 |
| 100 | 1.262 |
| 100 | 1.262 |
| 100 | 1.240 |
| 100 | 1.240 |
| 100 | 1.240 |
| 100 | 1.204 |
| 100 | 1.204 |
| 100 | 1.204 |
| 150 | 1.364 |
| 150 | 1.364 |
| 150 | 1.364 |
| 150 | 1.347 |
| 150 | 1.347 |
| 150 | 1.347 |
| 150 | 1.338 |
| 150 | 1.338 |
| 150 | 1.338 |
| 200 | 1.161 |
| 200 | 1.161 |
| 200 | 1.161 |
| 200 | 1.347 |
| 200 | 1.347 |
| 200 | 1.347 |
| 200 | 1.255 |
| 200 | 1.255 |
| 200 | 1.255 |

### sizing.vol_target_ann

| Value | Sharpe |
|-------|--------|
| 0.1 | 1.262 |
| 0.1 | 1.364 |
| 0.1 | 1.161 |
| 0.1 | 1.240 |
| 0.1 | 1.347 |
| 0.1 | 1.347 |
| 0.1 | 1.204 |
| 0.1 | 1.338 |
| 0.1 | 1.255 |
| 0.15 | 1.262 |
| 0.15 | 1.364 |
| 0.15 | 1.161 |
| 0.15 | 1.240 |
| 0.15 | 1.347 |
| 0.15 | 1.347 |
| 0.15 | 1.204 |
| 0.15 | 1.338 |
| 0.15 | 1.255 |
| 0.2 | 1.262 |
| 0.2 | 1.364 |
| 0.2 | 1.161 |
| 0.2 | 1.240 |
| 0.2 | 1.347 |
| 0.2 | 1.347 |
| 0.2 | 1.204 |
| 0.2 | 1.338 |
| 0.2 | 1.255 |

## Conclusion

✅ **Strong optimization results** - strategy shows robust performance across parameter ranges.

