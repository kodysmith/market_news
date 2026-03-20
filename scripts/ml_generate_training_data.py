#!/usr/bin/env python3
"""Generate ML training data by simulating IC trades on each day and pairing with features."""
from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.optimize_spx_verticals_historical import (
    bs_call_price,
    bs_put_price,
    friday_expirations,
    get_trading_days,
    strike_for_delta,
    MULT,
)

SLIPPAGE = 0.10  # per leg
COMMISSION = 0.65  # per leg
LOOKBACK = 20  # feature window


def simulate_ic_trade(
    spx_close: pd.Series,
    entry_date: date,
    trading_days: list[date],
    vix_at_entry: float,
    dte_target: int,
    short_delta: float,
    width: float,
    tp_pct: float,
    sl_mult: float,
) -> dict | None:
    """Simulate a single iron condor trade. Returns trade dict or None."""
    ts_entry = pd.Timestamp(entry_date)
    if ts_entry not in spx_close.index:
        return None

    S = float(spx_close[ts_entry])
    if S <= 0:
        return None

    sigma = vix_at_entry / 100.0 if vix_at_entry > 3 else vix_at_entry
    if sigma <= 0:
        return None

    # Find entry index in trading days
    try:
        entry_idx = trading_days.index(entry_date)
    except ValueError:
        return None

    exp_idx = entry_idx + dte_target
    if exp_idx >= len(trading_days):
        return None
    exp_date = trading_days[exp_idx]

    T = dte_target / 252.0
    if T <= 0:
        return None

    # Strikes
    K_ps = strike_for_delta(S, T, short_delta, sigma, is_put=True)
    K_cs = strike_for_delta(S, T, short_delta, sigma, is_put=False)
    if K_ps is None or K_cs is None:
        return None
    K_pl = K_ps - width
    K_cl = K_cs + width
    if K_pl <= 0:
        return None

    # Entry credit
    put_credit = bs_put_price(S, K_ps, T, sigma) - bs_put_price(S, K_pl, T, sigma)
    call_credit = bs_call_price(S, K_cs, T, sigma) - bs_call_price(S, K_cl, T, sigma)
    total_credit = put_credit + call_credit
    entry_slip = 4 * SLIPPAGE
    credit_adj = total_credit - entry_slip
    if credit_adj <= 0:
        return None

    close_threshold = (1.0 - tp_pct) * credit_adj
    stop_threshold = sl_mult * credit_adj

    exit_date = exp_date
    exit_reason = "expiry"
    exit_debit = None

    for i in range(entry_idx + 1, exp_idx):
        if i >= len(trading_days):
            break
        d = trading_days[i]
        ts_d = pd.Timestamp(d)
        if ts_d not in spx_close.index:
            continue
        S_d = float(spx_close[ts_d])
        dte_left = exp_idx - i
        T_d = dte_left / 252.0
        if T_d <= 0:
            break
        pv = bs_put_price(S_d, K_ps, T_d, sigma) - bs_put_price(S_d, K_pl, T_d, sigma)
        cv = bs_call_price(S_d, K_cs, T_d, sigma) - bs_call_price(S_d, K_cl, T_d, sigma)
        spread_val = pv + cv
        if spread_val <= close_threshold:
            exit_date = d
            exit_reason = "tp"
            exit_debit = spread_val
            break
        if spread_val >= stop_threshold:
            exit_date = d
            exit_reason = "sl"
            exit_debit = spread_val
            break

    if exit_reason == "expiry":
        ts_exp = pd.Timestamp(exp_date)
        S_exp = float(spx_close[ts_exp]) if ts_exp in spx_close.index else S
        put_set = max(0, K_ps - S_exp) - max(0, K_pl - S_exp)
        call_set = max(0, S_exp - K_cs) - max(0, S_exp - K_cl)
        settlement = put_set + call_set
        comm = 4 * COMMISSION
        pnl = (credit_adj - settlement) * MULT - comm
    else:
        exit_slip = 4 * SLIPPAGE
        debit_adj = (exit_debit or 0) + exit_slip
        comm = 8 * COMMISSION
        pnl = (credit_adj - debit_adj) * MULT - comm

    hold_days = (exit_date - entry_date).days

    max_profit = credit_adj * MULT - 4 * COMMISSION

    return {
        "entry_date": entry_date,
        "exit_date": exit_date,
        "exit_reason": exit_reason,
        "pnl": pnl,
        "credit": credit_adj,
        "max_profit": max_profit,
        "hold_days": hold_days,
    }


def build_training_data(
    features: pd.DataFrame,
    spx_close: pd.Series,
    trading_days: list[date],
    strategy_name: str,
    dte_target: int,
    short_delta: float,
    width: float,
    tp_pct: float,
    sl_mult: float,
) -> pd.DataFrame:
    """Build training samples: each row = 1 trade with flattened lookback features + targets."""
    feature_cols = [c for c in features.columns if c not in [
        "spx_close", "spx_open", "spx_high", "spx_low", "vix_close"
    ]]

    records = []
    dates = sorted(features.index)

    for idx, ts in enumerate(dates):
        if idx < LOOKBACK:
            continue
        entry_date = ts.date() if hasattr(ts, "date") else ts
        vix_val = features.loc[ts, "vix_close"] if "vix_close" in features.columns else 20.0

        result = simulate_ic_trade(
            spx_close, entry_date, trading_days, vix_val,
            dte_target, short_delta, width, tp_pct, sl_mult,
        )
        if result is None:
            continue

        # Build feature sequence (lookback window)
        window = features.iloc[idx - LOOKBACK + 1: idx + 1][feature_cols]
        if len(window) < LOOKBACK:
            continue

        # Flatten: each row in window becomes features with suffix _t0.._t19
        flat = {}
        for t, (_, row) in enumerate(window.iterrows()):
            for col in feature_cols:
                flat[f"{col}_t{t}"] = row[col]

        # Targets
        pnl = result["pnl"]
        credit = result["credit"]
        max_profit = result["max_profit"]

        flat["entry_target"] = 1 if pnl > 0 else 0
        if pnl <= 0:
            flat["entry_target_3class"] = 0  # loss
        elif max_profit > 0 and pnl < 0.5 * max_profit:
            flat["entry_target_3class"] = 1  # small win
        else:
            flat["entry_target_3class"] = 2  # full win

        # Normalized PnL: clamp to [-2, 1]
        pnl_norm = pnl / (credit * MULT) if credit > 0 else 0
        flat["pnl_normalized"] = max(-2.0, min(1.0, pnl_norm))

        flat["strategy"] = strategy_name
        flat["entry_date"] = result["entry_date"]
        flat["exit_date"] = result["exit_date"]
        flat["exit_reason"] = result["exit_reason"]
        flat["pnl"] = pnl
        flat["hold_days"] = result["hold_days"]

        records.append(flat)

    return pd.DataFrame(records)


def main():
    features_path = ROOT / "data" / "ml_features.parquet"
    if not features_path.exists():
        print("Features not found. Run ml_feature_extraction.py first.")
        return

    print("Loading features...")
    features = pd.read_parquet(features_path)
    print(f"  {features.shape[0]} days, {features.shape[1]} columns")

    # Build spx_close series from features
    spx_close = features["spx_close"].copy()
    spx_close.index = pd.to_datetime(spx_close.index)

    start = features.index.min().date() if hasattr(features.index.min(), "date") else features.index.min()
    end = features.index.max().date() if hasattr(features.index.max(), "date") else features.index.max()
    trading_days = get_trading_days(start, end)

    # IC3: 1DTE, 10δ, $30 width, TP50%, SL 2x
    print("\nSimulating IC3 (1DTE, 10δ, $30w, TP50%, SL2x)...")
    ic3 = build_training_data(
        features, spx_close, trading_days,
        "IC3", dte_target=1, short_delta=0.10, width=30,
        tp_pct=0.50, sl_mult=2.0,
    )
    print(f"  IC3 trades: {len(ic3)}")

    # IC6: 3DTE, 12δ, $30 width, TP40%, SL 2x
    print("Simulating IC6 (3DTE, 12δ, $30w, TP40%, SL2x)...")
    ic6 = build_training_data(
        features, spx_close, trading_days,
        "IC6", dte_target=3, short_delta=0.12, width=30,
        tp_pct=0.40, sl_mult=2.0,
    )
    print(f"  IC6 trades: {len(ic6)}")

    combined = pd.concat([ic3, ic6], ignore_index=True)
    print(f"\nTotal trades: {len(combined)}")

    # Split entry vs exit data
    entry_path = ROOT / "data" / "ml_training_entry.parquet"
    exit_path = ROOT / "data" / "ml_training_exit.parquet"

    # Entry data: all feature columns + entry targets
    meta_cols = ["strategy", "entry_date", "exit_date", "exit_reason", "pnl", "hold_days"]
    target_cols = ["entry_target", "entry_target_3class", "pnl_normalized"]
    feature_seq_cols = [c for c in combined.columns if c not in meta_cols + target_cols]

    combined.to_parquet(entry_path)
    print(f"  Saved entry data to {entry_path}")

    # Exit data: subset with hold_days > 0 (multi-day trades)
    exit_data = combined[combined["hold_days"] > 0].copy()
    exit_data.to_parquet(exit_path)
    print(f"  Saved exit data ({len(exit_data)} rows) to {exit_path}")

    # Summary stats
    print("\n--- Summary ---")
    for strat in ["IC3", "IC6"]:
        sub = combined[combined["strategy"] == strat]
        if len(sub) == 0:
            continue
        wins = sub["entry_target"].sum()
        print(f"\n{strat}:")
        print(f"  Trades: {len(sub)}")
        print(f"  Win rate: {wins/len(sub)*100:.1f}%")
        print(f"  Avg PnL: ${sub['pnl'].mean():.2f}")
        print(f"  Total PnL: ${sub['pnl'].sum():.2f}")
        print(f"  3-class dist: {sub['entry_target_3class'].value_counts().sort_index().to_dict()}")
        print(f"  Exit reasons: {sub['exit_reason'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
