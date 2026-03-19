#!/usr/bin/env python3
"""Backtest: GEX-pinned funded butterfly strategy.

Structure per expiration:
  FUNDER: Bear call spread (sell 15δ call, buy 15δ+width call) → collects credit
  LOTTERY: Butterfly centered at pin target → costs debit
  Net: credit - debit (target ≥ 0, i.e. free or paid to take the trade)

Pin target proxy (no historical GEX data):
  - Nearest $25 round strike to current SPX price
  - This approximates GEX pin behavior (dealer hedging at round strikes)

Direction-neutral: we run both bullish AND bearish funded butterflies.
"""
from __future__ import annotations

import math
import sys
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.optimize_spx_verticals_historical import (
    load_spx_vix, get_trading_days, friday_expirations,
    bs_put_price, bs_call_price, strike_for_delta, R, MULT,
)

SLIPPAGE_PER_LEG = 0.10
COMMISSION_PER_LEG = 0.65


def nearest_round_strike(spot: float, rounding: float = 25.0) -> float:
    """Nearest round strike to spot (e.g., nearest $25)."""
    return round(spot / rounding) * rounding


def price_butterfly_put(S, K_lower, K_mid, K_upper, T, sigma):
    """Price a put butterfly: buy K_lower put, sell 2x K_mid put, buy K_upper put."""
    if T <= 0 or sigma <= 0:
        # At expiry, intrinsic value
        return (max(K_lower - S, 0) - 2 * max(K_mid - S, 0) + max(K_upper - S, 0))
    p_lower = bs_put_price(S, K_lower, T, sigma)
    p_mid = bs_put_price(S, K_mid, T, sigma)
    p_upper = bs_put_price(S, K_upper, T, sigma)
    return p_lower - 2 * p_mid + p_upper


def price_butterfly_call(S, K_lower, K_mid, K_upper, T, sigma):
    """Price a call butterfly: buy K_lower call, sell 2x K_mid call, buy K_upper call."""
    if T <= 0 or sigma <= 0:
        return (max(S - K_lower, 0) - 2 * max(S - K_mid, 0) + max(S - K_upper, 0))
    c_lower = bs_call_price(S, K_lower, T, sigma)
    c_mid = bs_call_price(S, K_mid, T, sigma)
    c_upper = bs_call_price(S, K_upper, T, sigma)
    return c_lower - 2 * c_mid + c_upper


def butterfly_settlement(S_exp, K_lower, K_mid, K_upper):
    """Butterfly payout at expiration (same for put or call butterfly)."""
    # Max payout = K_mid - K_lower (= wing width) when S_exp = K_mid
    width = K_mid - K_lower
    if S_exp <= K_lower or S_exp >= K_upper:
        return 0.0
    elif S_exp <= K_mid:
        return S_exp - K_lower
    else:
        return K_upper - S_exp


def run_backtest(start: date, end: date):
    print("Loading SPX + VIX...", file=sys.stderr)
    spx_df, vix_df = load_spx_vix(start, end)
    if spx_df.empty:
        print("No data.", file=sys.stderr)
        return pd.DataFrame()

    spx_close = spx_df["Close"].copy()
    spx_close.index = spx_close.index.date
    vix_close = vix_df["Close"].copy()
    if not vix_close.empty:
        vix_close.index = vix_close.index.date

    trading_days = get_trading_days(start, end)
    trading_idx = {d: i for i, d in enumerate(trading_days)}
    expirations = friday_expirations(start, end)

    trades = []

    # Test multiple configurations
    configs = [
        # (name, dte, funder_delta, fly_wing, fly_rounding)
        ("FLY7_D15_W10_R25", 7, 0.15, 10, 25),
        ("FLY7_D15_W15_R25", 7, 0.15, 15, 25),
        ("FLY7_D15_W10_R50", 7, 0.15, 10, 50),
        ("FLY14_D15_W10_R25", 14, 0.15, 10, 25),
        ("FLY14_D15_W15_R25", 14, 0.15, 15, 25),
        ("FLY14_D15_W25_R25", 14, 0.15, 25, 25),
        ("FLY14_D15_W10_R50", 14, 0.15, 10, 50),
        ("FLY14_D10_W10_R25", 14, 0.10, 10, 25),
        ("FLY21_D15_W15_R25", 21, 0.15, 15, 25),
        ("FLY21_D15_W25_R50", 21, 0.15, 25, 50),
        # Wider wings = cheaper butterfly but needs tighter pin
        ("FLY7_D15_W5_R25", 7, 0.15, 5, 25),
        ("FLY14_D15_W5_R25", 14, 0.15, 5, 25),
        # Multiple butterflies funded by one spread
        ("FLY7_D20_W10_R25", 7, 0.20, 10, 25),
        ("FLY14_D20_W15_R25", 14, 0.20, 15, 25),
    ]

    for config_name, target_dte, funder_delta, fly_wing, fly_rounding in configs:
        for exp_date in expirations:
            if exp_date < start or exp_date > end:
                continue
            exp_i = trading_idx.get(exp_date)
            if exp_i is None:
                continue

            # Find entry date
            entry_date = None
            for check_i in range(max(0, exp_i - target_dte - 5), exp_i):
                d = trading_days[check_i]
                if d < start:
                    continue
                cal_dte = (exp_date - d).days
                if abs(cal_dte - target_dte) <= 3:
                    entry_date = d
                    break

            if entry_date is None or entry_date not in spx_close.index:
                continue

            entry_idx = trading_idx[entry_date]
            dte_trading = exp_i - entry_idx
            if dte_trading <= 0:
                continue

            S_entry = float(spx_close[entry_date])
            vix_entry = float(vix_close.get(entry_date, 20.0))
            sigma = vix_entry / 100.0 if vix_entry > 3 else vix_entry
            T_entry = dte_trading / 252.0

            # === FUNDER: Bear call spread ===
            K_call_short = strike_for_delta(S_entry, T_entry, funder_delta, sigma, is_put=False)
            if K_call_short is None:
                continue
            K_call_short = round(K_call_short / 5) * 5
            K_call_long = K_call_short + 50  # $50 wide funder
            funder_credit = (
                bs_call_price(S_entry, K_call_short, T_entry, sigma) -
                bs_call_price(S_entry, K_call_long, T_entry, sigma)
            )
            funder_slippage = 4 * SLIPPAGE_PER_LEG / 2.0  # 2 legs
            funder_credit_adj = funder_credit - funder_slippage
            if funder_credit_adj <= 0:
                continue

            # === LOTTERY: Put butterfly at pin target ===
            K_mid = nearest_round_strike(S_entry, fly_rounding)
            K_lower = K_mid - fly_wing
            K_upper = K_mid + fly_wing

            fly_debit = price_butterfly_put(S_entry, K_lower, K_mid, K_upper, T_entry, sigma)
            fly_slippage = 4 * SLIPPAGE_PER_LEG / 2.0  # 3 legs but counted as 4 for safety
            fly_cost = fly_debit + fly_slippage

            # Net cost
            net_credit = funder_credit_adj - fly_cost
            # Allow small debit (up to $1.00) — still mostly funded
            if net_credit < -1.0:
                continue

            # === SETTLEMENT AT EXPIRY ===
            if exp_date not in spx_close.index:
                continue
            S_exp = float(spx_close[exp_date])

            # Funder P&L (bear call spread)
            funder_settlement = (
                max(0, S_exp - K_call_short) - max(0, S_exp - K_call_long)
            )
            funder_pnl = (funder_credit_adj - funder_settlement) * MULT

            # Butterfly P&L
            fly_payout = butterfly_settlement(S_exp, K_lower, K_mid, K_upper)
            fly_pnl = (fly_payout - fly_cost) * MULT

            # Total
            total_pnl = funder_pnl + fly_pnl
            commission = 7 * COMMISSION_PER_LEG  # 2 (funder) + 3 (fly) legs, each side

            total_pnl -= commission

            # Distance from pin
            pin_distance = abs(S_exp - K_mid)
            pin_distance_pct = pin_distance / S_entry * 100

            trades.append({
                "config": config_name,
                "entry_date": entry_date,
                "exp_date": exp_date,
                "spot_entry": S_entry,
                "spot_expiry": S_exp,
                "vix_entry": vix_entry,
                "funder_short": K_call_short,
                "fly_center": K_mid,
                "fly_wing": fly_wing,
                "funder_credit": round(funder_credit_adj * MULT, 2),
                "fly_cost": round(fly_cost * MULT, 2),
                "net_credit": round(net_credit * MULT, 2),
                "fly_payout": round(fly_payout * MULT, 2),
                "funder_pnl": round(funder_pnl, 2),
                "fly_pnl": round(fly_pnl, 2),
                "total_pnl": round(total_pnl, 2),
                "pin_distance": round(pin_distance, 1),
                "pin_distance_pct": round(pin_distance_pct, 2),
            })

    return pd.DataFrame(trades)


def print_report(trades: pd.DataFrame):
    if trades.empty:
        print("No trades.")
        return

    print("\n" + "=" * 110)
    print("FUNDED BUTTERFLY BACKTEST — DIRECTION NEUTRAL")
    print("=" * 110)
    print(f"Period: {trades['entry_date'].min()} to {trades['exp_date'].max()}")
    print(f"Total trades: {len(trades)}")
    print()

    # Pin behavior analysis first
    print("-" * 80)
    print("SPX PIN BEHAVIOR AT EXPIRATION")
    print("-" * 80)
    for rnd in [25, 50]:
        subset = trades[trades['config'].str.contains(f'R{rnd}')]
        if subset.empty:
            continue
        within_5 = (subset['pin_distance'] <= 5).mean() * 100
        within_10 = (subset['pin_distance'] <= 10).mean() * 100
        within_15 = (subset['pin_distance'] <= 15).mean() * 100
        within_25 = (subset['pin_distance'] <= 25).mean() * 100
        print(f"  Nearest ${rnd} strike: within $5={within_5:.0f}%  $10={within_10:.0f}%  "
              f"$15={within_15:.0f}%  $25={within_25:.0f}%")
    print()

    # Results by config
    print("-" * 110)
    print(f"{'Config':<24} {'Trades':>6} {'Total P&L':>10} {'Avg P&L':>8} {'Win%':>6} "
          f"{'Avg Net$':>8} {'Fly Hit%':>8} {'Avg Fly$':>9} {'PF':>6} {'Sharpe':>7}")
    print("-" * 110)

    configs_summary = []
    for config in sorted(trades['config'].unique()):
        ct = trades[trades['config'] == config]
        n = len(ct)
        total = ct['total_pnl'].sum()
        avg = ct['total_pnl'].mean()
        wins = (ct['total_pnl'] > 0).sum()
        wr = 100 * wins / n
        avg_net = ct['net_credit'].mean()
        fly_hits = (ct['fly_payout'] > 0).sum()
        fly_hit_pct = 100 * fly_hits / n
        avg_fly = ct['fly_pnl'].mean()

        gross_win = ct.loc[ct['total_pnl'] > 0, 'total_pnl'].sum()
        gross_loss = abs(ct.loc[ct['total_pnl'] < 0, 'total_pnl'].sum())
        pf = gross_win / gross_loss if gross_loss > 0 else 999

        monthly = ct.set_index(pd.to_datetime(ct['exp_date'])).resample('M')['total_pnl'].sum()
        sharpe = (monthly.mean() / monthly.std() * math.sqrt(12)) if monthly.std() > 0 else 0

        print(f"{config:<24} {n:>6} ${total:>9,.0f} ${avg:>7,.0f} {wr:>5.0f}% "
              f"${avg_net:>7,.0f} {fly_hit_pct:>7.0f}% ${avg_fly:>8,.0f} {pf:>5.1f} {sharpe:>7.2f}")
        configs_summary.append((config, total, avg, wr, pf, sharpe))

    # Top configs
    print()
    print("-" * 80)
    print("TOP 5 BY TOTAL P&L")
    print("-" * 80)
    configs_summary.sort(key=lambda x: x[1], reverse=True)
    for name, total, avg, wr, pf, sharpe in configs_summary[:5]:
        print(f"  {name:<24} Total=${total:>10,.0f}  Avg=${avg:>6,.0f}  Win%={wr:.0f}%  Sharpe={sharpe:.2f}")

    print()
    print("-" * 80)
    print("TOP 5 BY SHARPE")
    print("-" * 80)
    configs_summary.sort(key=lambda x: x[5], reverse=True)
    for name, total, avg, wr, pf, sharpe in configs_summary[:5]:
        print(f"  {name:<24} Sharpe={sharpe:.2f}  Total=${total:>10,.0f}  Win%={wr:.0f}%  PF={pf:.1f}")

    # Best config deep dive
    best = configs_summary[0][0]
    print()
    print("=" * 80)
    print(f"DEEP DIVE: {best}")
    print("=" * 80)
    ct = trades[trades['config'] == best]

    # Annual breakdown
    ct_copy = ct.copy()
    ct_copy['year'] = pd.to_datetime(ct_copy['exp_date']).dt.year
    annual = ct_copy.groupby('year')['total_pnl'].agg(['sum', 'count', 'mean'])
    annual['win_rate'] = ct_copy.groupby('year').apply(
        lambda x: (x['total_pnl'] > 0).mean() * 100
    )
    print(f"\n{'Year':<6} {'Trades':>7} {'P&L':>10} {'Avg':>8} {'Win%':>6}")
    for year, row in annual.iterrows():
        print(f"{year:<6} {int(row['count']):>7} ${row['sum']:>9,.0f} ${row['mean']:>7,.0f} "
              f"{row['win_rate']:>5.0f}%")

    # Fly hit analysis
    print(f"\nButterfly analysis:")
    print(f"  Fly hits (payout > 0): {(ct['fly_payout'] > 0).sum()}/{len(ct)} "
          f"({(ct['fly_payout'] > 0).mean()*100:.0f}%)")
    print(f"  Avg fly payout when hit: ${ct.loc[ct['fly_payout'] > 0, 'fly_payout'].mean():,.0f}")
    print(f"  Max fly payout: ${ct['fly_payout'].max():,.0f}")
    print(f"  Avg funder credit: ${ct['funder_credit'].mean():,.0f}")
    print(f"  Avg fly cost: ${ct['fly_cost'].mean():,.0f}")
    print(f"  Avg net credit: ${ct['net_credit'].mean():,.0f}")

    # Funder alone analysis
    print(f"\nFunder (bear call spread) standalone:")
    funder_wins = (ct['funder_pnl'] > 0).sum()
    print(f"  Win rate: {100*funder_wins/len(ct):.0f}%")
    print(f"  Avg P&L: ${ct['funder_pnl'].mean():,.0f}")

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
