#!/usr/bin/env python3
"""Backtest: Daily consistent cash flow strategies.

Optimizes for:
  1. Highest win rate (daily gains > losses)
  2. Consistent daily P&L (low variance)
  3. Maximum annual return
  4. Break-even better than loss — tight stops, take profits fast

Strategies tested:
  - 0DTE/1DTE iron condors (daily income, tiny exposure)
  - 3DTE credit spreads (quick cycle)
  - 7DTE with early profit-taking (50%, 30%, 20% targets)
  - Funded butterflies (free lottery + funder income)
  - Layered: multiple small positions across DTEs
  - All direction-neutral (no SMA filter — trades in any regime)
"""
from __future__ import annotations

import math
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.optimize_spx_verticals_historical import (
    load_spx_vix, get_trading_days, friday_expirations,
    bs_put_price, bs_call_price, strike_for_delta, MULT,
)

SLIPPAGE_PER_LEG = 0.10
COMMISSION_PER_LEG = 0.65


def simulate_spread(
    config_id: str,
    strategy: str,          # "iron_condor" | "call_vertical" | "put_vertical"
    dte: int,
    short_delta: float,
    width: float,
    profit_target_pct: float,
    stop_mult: float,
    time_stop_dte: int | None,
    spx_close: pd.Series,
    vix_close: pd.Series,
    trading_days: list,
    trading_idx: dict,
    expirations: list,
    start: date,
    end: date,
    max_hold_days: int | None = None,  # force close after N days
) -> list[dict]:
    """Simulate a single spread config across all expirations."""
    trades = []

    for exp_date in expirations:
        if exp_date < start or exp_date > end:
            continue
        exp_i = trading_idx.get(exp_date)
        if exp_i is None:
            continue

        # Find entry date
        entry_date = None
        for check_i in range(max(0, exp_i - dte - 5), exp_i):
            d = trading_days[check_i]
            if d < start:
                continue
            cal_dte = (exp_date - d).days
            if abs(cal_dte - dte) <= 2:
                entry_date = d
                break

        if entry_date is None or entry_date not in spx_close.index:
            continue

        entry_idx = trading_idx[entry_date]
        dte_trading = exp_i - entry_idx
        if dte_trading <= 0:
            continue

        S_entry = float(spx_close[entry_date])
        vix_val = float(vix_close.get(entry_date, 20.0))
        sigma = vix_val / 100.0 if vix_val > 3 else vix_val
        T_entry = dte_trading / 252.0

        # Compute strikes and credit
        credit = 0.0
        K_put_short = K_put_long = K_call_short = K_call_long = None
        n_legs = 4

        if strategy in ("iron_condor", "put_vertical"):
            K_put_short = strike_for_delta(S_entry, T_entry, short_delta, sigma, is_put=True)
            if K_put_short is None:
                continue
            K_put_short = round(K_put_short / 5) * 5
            K_put_long = K_put_short - width
            if K_put_long <= 0:
                continue
            credit += bs_put_price(S_entry, K_put_short, T_entry, sigma) - \
                      bs_put_price(S_entry, K_put_long, T_entry, sigma)

        if strategy in ("iron_condor", "call_vertical"):
            K_call_short = strike_for_delta(S_entry, T_entry, short_delta, sigma, is_put=False)
            if K_call_short is None:
                continue
            K_call_short = round(K_call_short / 5) * 5
            K_call_long = K_call_short + width
            credit += bs_call_price(S_entry, K_call_short, T_entry, sigma) - \
                      bs_call_price(S_entry, K_call_long, T_entry, sigma)

        if strategy == "iron_condor":
            n_legs = 8

        entry_slippage = n_legs * SLIPPAGE_PER_LEG / 2.0
        credit_adj = credit - entry_slippage
        if credit_adj <= 0:
            continue

        # Simulate daily
        close_threshold = (1.0 - profit_target_pct) * credit_adj
        stop_threshold = stop_mult * credit_adj

        exit_date = exp_date
        exit_reason = "expiry"
        exit_debit = None

        for i in range(entry_idx + 1, exp_i):
            if i >= len(trading_days):
                break
            d = trading_days[i]
            if d not in spx_close.index:
                continue

            # Max hold days
            if max_hold_days is not None and (i - entry_idx) >= max_hold_days:
                S_d = float(spx_close[d])
                dte_left = exp_i - i
                T_d = dte_left / 252.0
                sigma_d = float(vix_close.get(d, vix_val))
                sigma_d = sigma_d / 100.0 if sigma_d > 3 else sigma_d
                spread_val = 0.0
                if K_put_short is not None and T_d > 0:
                    spread_val += bs_put_price(S_d, K_put_short, T_d, sigma_d) - \
                                  bs_put_price(S_d, K_put_long, T_d, sigma_d)
                if K_call_short is not None and T_d > 0:
                    spread_val += bs_call_price(S_d, K_call_short, T_d, sigma_d) - \
                                  bs_call_price(S_d, K_call_long, T_d, sigma_d)
                exit_date = d
                exit_reason = "max_hold"
                exit_debit = spread_val
                break

            S_d = float(spx_close[d])
            dte_left = exp_i - i
            T_d = dte_left / 252.0
            if T_d <= 0:
                break

            sigma_d = float(vix_close.get(d, vix_val))
            sigma_d = sigma_d / 100.0 if sigma_d > 3 else sigma_d

            spread_val = 0.0
            if K_put_short is not None:
                spread_val += bs_put_price(S_d, K_put_short, T_d, sigma_d) - \
                              bs_put_price(S_d, K_put_long, T_d, sigma_d)
            if K_call_short is not None:
                spread_val += bs_call_price(S_d, K_call_short, T_d, sigma_d) - \
                              bs_call_price(S_d, K_call_long, T_d, sigma_d)

            if spread_val <= close_threshold:
                exit_date = d
                exit_reason = "profit_target"
                exit_debit = spread_val
                break
            if spread_val >= stop_threshold:
                exit_date = d
                exit_reason = "stop_loss"
                exit_debit = spread_val
                break
            if time_stop_dte is not None and dte_left <= time_stop_dte:
                exit_date = d
                exit_reason = "time_stop"
                exit_debit = spread_val
                break

        # P&L
        if exit_reason == "expiry":
            S_exp = float(spx_close.get(exp_date, S_entry))
            settlement = 0.0
            if K_put_short is not None:
                settlement += max(0, K_put_short - S_exp) - max(0, K_put_long - S_exp)
            if K_call_short is not None:
                settlement += max(0, S_exp - K_call_short) - max(0, S_exp - K_call_long)
            commission = (n_legs / 2) * COMMISSION_PER_LEG
            pnl = (credit_adj - settlement) * MULT - commission
        else:
            exit_slippage = n_legs * SLIPPAGE_PER_LEG / 2.0
            debit_adj = (exit_debit or 0) + exit_slippage
            commission = n_legs * COMMISSION_PER_LEG
            pnl = (credit_adj - debit_adj) * MULT - commission

        hold_days = (exit_date - entry_date).days

        trades.append({
            "config": config_id,
            "entry_date": entry_date,
            "exit_date": exit_date,
            "exit_reason": exit_reason,
            "pnl": round(pnl, 2),
            "hold_days": hold_days,
            "credit": round(credit_adj * MULT, 2),
            "vix": vix_val,
        })

    return trades


def run_backtest(start: date, end: date):
    print("Loading SPX + VIX...", file=sys.stderr)
    spx_df, vix_df = load_spx_vix(start, end)
    if spx_df.empty:
        return pd.DataFrame()

    spx_close = spx_df["Close"].copy()
    spx_close.index = spx_close.index.date
    vix_close = vix_df["Close"].copy()
    if not vix_close.empty:
        vix_close.index = vix_close.index.date

    trading_days = get_trading_days(start, end)
    trading_idx = {d: i for i, d in enumerate(trading_days)}

    # Generate all Friday expirations
    fri_exps = friday_expirations(start, end)

    # Also generate daily expirations (Mon-Fri) for 0DTE/1DTE strategies
    daily_exps = [d for d in trading_days if d >= start and d <= end]

    all_trades = []

    # === CONFIGS: Optimized for daily consistent gains ===
    configs = [
        # (id, strategy, dte, delta, width, tp%, stop_mult, time_stop, exps, max_hold)

        # --- Ultra-short: 1-3 DTE, fast in/out ---
        ("IC1_1DTE_D8_TP30", "iron_condor", 1, 0.08, 25, 0.30, 2.0, None, daily_exps, None),
        ("IC2_1DTE_D10_TP30", "iron_condor", 1, 0.10, 25, 0.30, 2.0, None, daily_exps, None),
        ("IC3_1DTE_D10_TP50", "iron_condor", 1, 0.10, 30, 0.50, 2.0, None, daily_exps, None),
        ("IC4_2DTE_D10_TP30", "iron_condor", 2, 0.10, 25, 0.30, 2.0, None, daily_exps, None),
        ("IC5_3DTE_D10_TP30", "iron_condor", 3, 0.10, 30, 0.30, 1.5, None, daily_exps, None),
        ("IC6_3DTE_D12_TP40", "iron_condor", 3, 0.12, 30, 0.40, 2.0, None, daily_exps, None),

        # --- Short: 5-7 DTE, close early ---
        ("IC7_5DTE_D12_TP30", "iron_condor", 5, 0.12, 50, 0.30, 2.0, 1, fri_exps, None),
        ("IC8_7DTE_D10_TP20", "iron_condor", 7, 0.10, 50, 0.20, 1.5, 1, fri_exps, None),
        ("IC9_7DTE_D12_TP30", "iron_condor", 7, 0.12, 50, 0.30, 2.0, 1, fri_exps, None),
        ("IC10_7DTE_D15_TP30", "iron_condor", 7, 0.15, 50, 0.30, 2.0, 1, fri_exps, None),
        # Max hold = close after 3 days regardless
        ("IC11_7DTE_D15_MH3", "iron_condor", 7, 0.15, 50, 0.30, 2.0, None, fri_exps, 3),

        # --- 14 DTE with aggressive profit taking ---
        ("IC12_14DTE_D10_TP20", "iron_condor", 14, 0.10, 50, 0.20, 1.5, 3, fri_exps, None),
        ("IC13_14DTE_D15_TP30", "iron_condor", 14, 0.15, 50, 0.30, 2.0, 2, fri_exps, None),
        ("IC14_14DTE_D10_TP30_S15", "iron_condor", 14, 0.10, 50, 0.30, 1.5, 3, fri_exps, None),

        # --- Ultra-tight: 5-delta, very far OTM, high win rate ---
        ("IC15_7DTE_D05_TP50", "iron_condor", 7, 0.05, 25, 0.50, 3.0, 1, fri_exps, None),
        ("IC16_14DTE_D05_TP50", "iron_condor", 14, 0.05, 25, 0.50, 3.0, 2, fri_exps, None),

        # --- Call verticals only (bearish, no trend filter) ---
        ("CV1_7DTE_D10_TP30", "call_vertical", 7, 0.10, 50, 0.30, 2.0, 1, fri_exps, None),
        ("CV2_7DTE_D15_TP50", "call_vertical", 7, 0.15, 50, 0.50, 2.0, 1, fri_exps, None),

        # --- Put verticals only (bullish) ---
        ("PV1_7DTE_D10_TP30", "put_vertical", 7, 0.10, 50, 0.30, 2.0, 1, fri_exps, None),
    ]

    for cfg in configs:
        cid, strat, dte, delta, w, tp, sl, ts, exps, mh = cfg
        result = simulate_spread(
            cid, strat, dte, delta, w, tp, sl, ts,
            spx_close, vix_close, trading_days, trading_idx,
            exps, start, end, max_hold_days=mh,
        )
        all_trades.extend(result)
        print(f"  {cid}: {len(result)} trades", file=sys.stderr)

    return pd.DataFrame(all_trades)


def print_report(trades: pd.DataFrame):
    if trades.empty:
        print("No trades.")
        return

    print("\n" + "=" * 120)
    print("DAILY CONSISTENT CASH FLOW BACKTEST — ALL REGIMES")
    print("=" * 120)
    print(f"Period: {trades['entry_date'].min()} to {trades['exit_date'].max()}")
    print()

    # Rank by: win rate first, then sharpe, then total P&L
    print(f"{'Config':<25} {'Trades':>6} {'Total P&L':>11} {'Ann P&L':>10} {'Avg':>7} "
          f"{'Win%':>6} {'PF':>5} {'Sharpe':>7} {'AvgHold':>7} {'AvgCred':>8} {'Loss%':>6}")
    print("-" * 120)

    summaries = []
    for config in sorted(trades['config'].unique()):
        ct = trades[trades['config'] == config]
        n = len(ct)
        if n < 20:
            continue
        total = ct['pnl'].sum()
        avg = ct['pnl'].mean()
        wins = (ct['pnl'] > 0).sum()
        losses = (ct['pnl'] < 0).sum()
        breakeven = (ct['pnl'] == 0).sum()
        wr = 100 * wins / n
        loss_pct = 100 * losses / n
        avg_hold = ct['hold_days'].mean()
        avg_credit = ct['credit'].mean()

        gross_win = ct.loc[ct['pnl'] > 0, 'pnl'].sum()
        gross_loss = abs(ct.loc[ct['pnl'] < 0, 'pnl'].sum())
        pf = gross_win / gross_loss if gross_loss > 0 else 999

        years = (pd.to_datetime(ct['exit_date'].max()) - pd.to_datetime(ct['entry_date'].min())).days / 365.25
        ann_pnl = total / years if years > 0 else 0

        ct_monthly = ct.set_index(pd.to_datetime(ct['exit_date'])).resample('ME')['pnl'].sum()
        sharpe = (ct_monthly.mean() / ct_monthly.std() * math.sqrt(12)) if ct_monthly.std() > 0 else 0

        print(f"{config:<25} {n:>6} ${total:>10,.0f} ${ann_pnl:>9,.0f} ${avg:>6,.0f} "
              f"{wr:>5.0f}% {pf:>4.1f} {sharpe:>7.2f} {avg_hold:>6.1f}d ${avg_credit:>7,.0f} {loss_pct:>5.1f}%")

        summaries.append({
            'config': config, 'trades': n, 'total': total, 'ann_pnl': ann_pnl,
            'avg': avg, 'wr': wr, 'pf': pf, 'sharpe': sharpe, 'loss_pct': loss_pct,
        })

    sdf = pd.DataFrame(summaries)

    # Best by composite score: high win rate + high sharpe + high annual P&L
    sdf['score'] = sdf['wr'] / 100 * sdf['sharpe'] * np.log1p(sdf['ann_pnl'].clip(0))
    sdf = sdf.sort_values('score', ascending=False)

    print()
    print("=" * 80)
    print("TOP 5 COMPOSITE (Win% × Sharpe × log(Annual P&L))")
    print("=" * 80)
    for _, row in sdf.head(5).iterrows():
        print(f"  {row['config']:<25} Win%={row['wr']:.0f}%  Sharpe={row['sharpe']:.2f}  "
              f"Ann=${row['ann_pnl']:,.0f}  Loss%={row['loss_pct']:.1f}%  PF={row['pf']:.1f}")

    # Deep dive on #1
    best = sdf.iloc[0]['config']
    print()
    print("=" * 80)
    print(f"DEEP DIVE: {best}")
    print("=" * 80)
    ct = trades[trades['config'] == best].copy()
    ct['year'] = pd.to_datetime(ct['exit_date']).dt.year
    annual = ct.groupby('year').agg(
        trades=('pnl', 'count'),
        pnl=('pnl', 'sum'),
        avg=('pnl', 'mean'),
        wins=('pnl', lambda x: (x > 0).sum()),
    )
    annual['wr'] = 100 * annual['wins'] / annual['trades']
    print(f"\n{'Year':<6} {'Trades':>7} {'P&L':>10} {'Avg':>8} {'Win%':>6}")
    for year, row in annual.iterrows():
        print(f"{year:<6} {int(row['trades']):>7} ${row['pnl']:>9,.0f} ${row['avg']:>7,.0f} {row['wr']:>5.0f}%")

    # Exit reason breakdown
    print(f"\nExit reasons:")
    for reason in ct['exit_reason'].unique():
        rt = ct[ct['exit_reason'] == reason]
        print(f"  {reason:<20} {len(rt):>5} trades ({100*len(rt)/len(ct):.0f}%)  "
              f"avg P&L ${rt['pnl'].mean():,.0f}")

    # Daily P&L consistency
    daily_pnl = ct.groupby('exit_date')['pnl'].sum()
    print(f"\nDaily P&L consistency:")
    print(f"  Trading days with exits: {len(daily_pnl)}")
    print(f"  Positive days: {(daily_pnl > 0).sum()} ({(daily_pnl > 0).mean()*100:.0f}%)")
    print(f"  Negative days: {(daily_pnl < 0).sum()} ({(daily_pnl < 0).mean()*100:.0f}%)")
    print(f"  Avg daily P&L: ${daily_pnl.mean():,.0f}")
    print(f"  Worst day: ${daily_pnl.min():,.0f}")
    print(f"  Best day: ${daily_pnl.max():,.0f}")

    print("=" * 80)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end", default="2025-12-31")
    args = parser.parse_args()

    start = datetime.strptime(args.start, "%Y-%m-%d").date()
    end = datetime.strptime(args.end, "%Y-%m-%d").date()

    trades = run_backtest(start, end)
    print_report(trades)
