#!/usr/bin/env python3
"""Portfolio-level backtest: runs all bot strategies together with actual
contract sizes, shared risk budget, drawdown pause, and compounding.

Outputs monthly/annual P&L, equity curve, and risk metrics for the full
portfolio — not individual strategies in isolation.
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

from scripts.evaluate_enhanced_cashflow import (
    StrategyConfig, EntryFilters, ExitRules,
    load_spx_vix, friday_expirations, get_trading_days,
    bs_put_price, bs_call_price, strike_for_delta, R, MULT,
    SLIPPAGE_PER_LEG, COMMISSION_PER_LEG,
)

# ---------------------------------------------------------------------------
# Portfolio config — mirrors bot/config.py
# ---------------------------------------------------------------------------
TOTAL_ACCOUNT = 700_000
RISK_BUDGET = 200_000
DRAWDOWN_PAUSE_PCT = 0.08
MAX_OPEN_TOTAL = 8
VIX_ABSOLUTE_CAP = 35.0

PORTFOLIO = [
    # (config, fixed_contracts)
    (StrategyConfig("15_VIX_SWEET_SPOT", "iron_condor", 14, 0.15, 50.0,
                    ExitRules(0.50, 2.0, time_stop_dte=2),
                    EntryFilters(vix_min=16, vix_max=24, trend_sma_period=50)),
     8),
    (StrategyConfig("SD1_7DTE_BASE", "iron_condor", 7, 0.15, 50.0,
                    ExitRules(0.50, 2.0),
                    EntryFilters(vix_max=28, trend_sma_period=50)),
     8),
    (StrategyConfig("AG2_VOLDET", "iron_condor", 21, 0.20, 75.0,
                    ExitRules(0.50, 2.0),
                    EntryFilters(vix_max=28, trend_sma_period=50, rv_expansion_max=1.05)),
     6),
    (StrategyConfig("VD2_ENTRY_105", "iron_condor", 14, 0.15, 50.0,
                    ExitRules(0.50, 2.0),
                    EntryFilters(vix_max=28, trend_sma_period=50, rv_expansion_max=1.05)),
     6),
    (StrategyConfig("BC4_BEAR_CALL_CONSERVATIVE", "call_vertical", 14, 0.10, 50.0,
                    ExitRules(0.50, 2.0, time_stop_dte=2),
                    EntryFilters(vix_max=28, trend_sma_period=50, trend_sma_invert=True)),
     6),
    (StrategyConfig("BC2_BEAR_CALL_7DTE", "call_vertical", 7, 0.15, 50.0,
                    ExitRules(0.50, 2.0),
                    EntryFilters(vix_min=16, vix_max=28, trend_sma_period=50, trend_sma_invert=True)),
     6),
]

# VD2 compounding
VD2_START = 6
VD2_MAX = 10
VD2_COMPOUND_THRESHOLD = 30_000


def run_portfolio_backtest(start: date, end: date) -> pd.DataFrame:
    """Simulate all strategies together, respecting shared position limits."""
    print("Loading SPX + VIX...", file=sys.stderr)
    spx_df, vix_df = load_spx_vix(start, end)
    if spx_df.empty:
        print("No SPX data.", file=sys.stderr)
        return pd.DataFrame()

    spx_close = spx_df["Close"].copy()
    spx_close.index = spx_close.index.date
    vix_close = vix_df["Close"].copy() if not vix_df.empty else pd.Series(dtype=float)
    if not vix_close.empty:
        vix_close.index = vix_close.index.date

    trading_days = get_trading_days(start, end)
    trading_idx = {d: i for i, d in enumerate(trading_days)}
    expirations = friday_expirations(start, end)

    # SMA cache
    sma50 = spx_df["Close"].rolling(50).mean()
    sma50.index = sma50.index.date
    sma50_dict = sma50.to_dict()

    # RV ratio cache
    log_ret = np.log(spx_df["Close"] / spx_df["Close"].shift(1))
    rv5 = log_ret.rolling(5).std() * np.sqrt(252)
    rv10 = log_ret.rolling(10).std() * np.sqrt(252)
    rv_ratio_series = rv5 / rv10
    rv_ratio_series.index = rv_ratio_series.index.date
    rv_ratio_dict = rv_ratio_series.to_dict()

    # Pre-compute: for each (config, expiration), find the ideal entry date
    exp_set = set(expirations)
    entry_schedule = {}  # entry_date -> list of (cfg, base_contracts, exp_date, entry_idx, exp_idx)
    for cfg, base_contracts in PORTFOLIO:
        for exp_date in expirations:
            if exp_date < start or exp_date > end:
                continue
            exp_i = trading_idx.get(exp_date)
            if exp_i is None:
                continue
            best_date = None
            best_diff = 999
            for check_i in range(max(0, exp_i - cfg.dte - 5), exp_i):
                d = trading_days[check_i]
                if d < start:
                    continue
                cal_dte = (exp_date - d).days
                diff = abs(cal_dte - cfg.dte)
                if diff <= 3 and diff < best_diff:
                    best_diff = diff
                    best_date = d
            if best_date is not None and best_date in spx_close.index:
                entry_schedule.setdefault(best_date, []).append(
                    (cfg, base_contracts, exp_date, trading_idx[best_date], exp_i)
                )

    # State
    open_positions = []  # list of dicts
    closed_trades = []
    cumulative_pnl = 0.0
    ath_equity = TOTAL_ACCOUNT
    vd2_contracts = VD2_START
    dd_pause_until = None  # date when pause expires (20 trading day cooldown)

    # Day-by-day simulation
    for day_i, today in enumerate(trading_days):
        if today < start or today > end:
            continue
        if today not in spx_close.index:
            continue

        S = float(spx_close[today])
        vix_val = float(vix_close.get(today, 20.0))
        sigma_today = vix_val / 100.0 if vix_val > 3 else vix_val

        # --- 1. Check exits on all open positions ---
        to_close = []
        for pos in open_positions:
            er = pos["exit_rules"]
            exp_idx = pos["exp_idx"]
            dte_left = exp_idx - day_i

            # Expiry
            if today >= pos["exp_date"]:
                S_exp = float(spx_close.get(pos["exp_date"], S))
                settlement = 0.0
                if pos["K_put_short"] is not None:
                    settlement += max(0, pos["K_put_short"] - S_exp) - max(0, pos["K_put_long"] - S_exp)
                if pos["K_call_short"] is not None:
                    settlement += max(0, S_exp - pos["K_call_short"]) - max(0, S_exp - pos["K_call_long"])
                commission = (pos["n_legs"] / 2) * COMMISSION_PER_LEG
                pnl = (pos["credit_adj"] - settlement) * MULT - commission
                pnl *= pos["n_contracts"]
                to_close.append((pos, "expiry", pnl))
                continue

            if day_i <= pos["entry_idx"]:
                continue  # not yet past entry day

            T_d = dte_left / 252.0
            if T_d <= 0:
                continue

            spread_val = 0.0
            if pos["K_put_short"] is not None:
                spread_val += bs_put_price(S, pos["K_put_short"], T_d, sigma_today) - \
                              bs_put_price(S, pos["K_put_long"], T_d, sigma_today)
            if pos["K_call_short"] is not None:
                spread_val += bs_call_price(S, pos["K_call_short"], T_d, sigma_today) - \
                              bs_call_price(S, pos["K_call_long"], T_d, sigma_today)

            close_threshold = (1.0 - er.profit_target_pct) * pos["credit_adj"]
            stop_threshold = er.stop_mult * pos["credit_adj"]

            exit_reason = None
            if spread_val <= close_threshold:
                exit_reason = "profit_target"
            elif spread_val >= stop_threshold:
                exit_reason = "stop_loss"
            elif er.time_stop_dte is not None and dte_left <= er.time_stop_dte:
                exit_reason = "time_stop"

            if exit_reason:
                exit_slippage = pos["n_legs"] * SLIPPAGE_PER_LEG / 2.0
                debit_adj = spread_val + exit_slippage
                commission = pos["n_legs"] * COMMISSION_PER_LEG
                pnl = (pos["credit_adj"] - debit_adj) * MULT - commission
                pnl *= pos["n_contracts"]
                to_close.append((pos, exit_reason, pnl))

        for pos, reason, pnl in to_close:
            cumulative_pnl += pnl
            current_equity = TOTAL_ACCOUNT + cumulative_pnl
            if current_equity > ath_equity:
                ath_equity = current_equity
            if cumulative_pnl > 0:
                vd2_contracts = min(VD2_START + int(cumulative_pnl // VD2_COMPOUND_THRESHOLD), VD2_MAX)

            closed_trades.append({
                "config_id": pos["config_id"],
                "strategy": pos["strategy"],
                "entry_date": pos["entry_date"],
                "exit_date": today,
                "exit_reason": reason,
                "n_contracts": pos["n_contracts"],
                "pnl": round(pnl, 2),
                "cumulative_pnl": round(cumulative_pnl, 2),
                "equity": round(TOTAL_ACCOUNT + cumulative_pnl, 2),
            })
            open_positions.remove(pos)

        # --- 2. Check entries scheduled for today ---
        if today not in entry_schedule:
            continue

        sma_val = sma50_dict.get(today)
        rv_val = rv_ratio_dict.get(today, 1.0)
        if isinstance(rv_val, float) and np.isnan(rv_val):
            rv_val = 1.0

        current_equity = TOTAL_ACCOUNT + cumulative_pnl
        in_drawdown = current_equity < ath_equity * (1 - DRAWDOWN_PAUSE_PCT)
        if in_drawdown and dd_pause_until is None:
            # Start cooldown: pause for 20 trading days then resume
            pause_idx = min(day_i + 20, len(trading_days) - 1)
            dd_pause_until = trading_days[pause_idx]
        if dd_pause_until is not None and today >= dd_pause_until:
            # Cooldown expired — reset ATH to current equity and resume
            ath_equity = current_equity
            dd_pause_until = None
        dd_paused = dd_pause_until is not None

        for cfg, base_contracts, exp_date, entry_idx, exp_idx in entry_schedule[today]:
            if dd_paused:
                continue
            if len(open_positions) >= MAX_OPEN_TOTAL:
                continue
            if vix_val > VIX_ABSOLUTE_CAP:
                continue

            f = cfg.filters
            if f.vix_min is not None and vix_val < f.vix_min:
                continue
            if f.vix_max is not None and vix_val > f.vix_max:
                continue
            if f.trend_sma_period is not None and sma_val is not None and not np.isnan(sma_val):
                if getattr(f, 'trend_sma_invert', False):
                    if S > sma_val:
                        continue
                else:
                    if S <= sma_val:
                        continue
            if f.rv_expansion_max is not None and rv_val > f.rv_expansion_max:
                continue

            # No duplicate config+expiry
            if any(p["config_id"] == cfg.config_id and p["exp_date"] == exp_date for p in open_positions):
                continue

            dte_trading = exp_idx - entry_idx
            if dte_trading <= 0:
                continue
            T_entry = dte_trading / 252.0
            sigma = vix_val / 100.0 if vix_val > 3 else vix_val
            n_contracts = vd2_contracts if cfg.config_id == "VD2_ENTRY_105" else base_contracts

            # Compute strikes and credit
            if cfg.strategy == "call_vertical":
                K_call_short = strike_for_delta(S, T_entry, cfg.short_delta, sigma, is_put=False)
                if K_call_short is None:
                    continue
                K_call_long = K_call_short + cfg.width
                credit = bs_call_price(S, K_call_short, T_entry, sigma) - \
                         bs_call_price(S, K_call_long, T_entry, sigma)
                K_put_short = K_put_long = None
            else:
                K_put_short = strike_for_delta(S, T_entry, cfg.short_delta, sigma, is_put=True)
                if K_put_short is None:
                    continue
                K_put_long = K_put_short - cfg.width
                if K_put_long <= 0:
                    continue
                credit = bs_put_price(S, K_put_short, T_entry, sigma) - \
                         bs_put_price(S, K_put_long, T_entry, sigma)
                K_call_short = K_call_long = None
                if cfg.strategy == "iron_condor":
                    K_call_short = strike_for_delta(S, T_entry, cfg.short_delta, sigma, is_put=False)
                    if K_call_short is None:
                        continue
                    K_call_long = K_call_short + cfg.width
                    credit += bs_call_price(S, K_call_short, T_entry, sigma) - \
                              bs_call_price(S, K_call_long, T_entry, sigma)

            n_legs = 8 if cfg.strategy == "iron_condor" else 4
            credit_adj = credit - n_legs * SLIPPAGE_PER_LEG / 2.0
            if credit_adj <= 0:
                continue

            max_loss_per = cfg.width * MULT * n_contracts
            current_margin = sum(p["max_loss"] for p in open_positions)
            if current_margin + max_loss_per > RISK_BUDGET:
                continue

            open_positions.append({
                "config_id": cfg.config_id,
                "strategy": cfg.strategy,
                "entry_date": today,
                "exp_date": exp_date,
                "entry_idx": entry_idx,
                "exp_idx": exp_idx,
                "n_contracts": n_contracts,
                "n_legs": n_legs,
                "credit_adj": credit_adj,
                "K_put_short": K_put_short,
                "K_put_long": K_put_long,
                "K_call_short": K_call_short,
                "K_call_long": K_call_long,
                "width": cfg.width,
                "max_loss": max_loss_per,
                "exit_rules": cfg.exit_rules,
            })

    return pd.DataFrame(closed_trades)


def print_portfolio_report(trades: pd.DataFrame) -> None:
    """Print comprehensive portfolio report."""
    if trades.empty:
        print("No trades.", file=sys.stderr)
        return

    n = len(trades)
    total_pnl = trades["pnl"].sum()
    wins = (trades["pnl"] > 0).sum()
    losses = (trades["pnl"] < 0).sum()
    win_rate = 100.0 * wins / n

    # Monthly
    trades["exit_dt"] = pd.to_datetime(trades["exit_date"])
    trades["year_month"] = trades["exit_dt"].dt.to_period("M").astype(str)
    monthly = trades.groupby("year_month")["pnl"].sum()
    n_months = len(monthly)
    avg_monthly = monthly.mean()
    worst_month = monthly.min()
    best_month = monthly.max()
    profitable_months = (monthly > 0).sum()
    monthly_std = monthly.std()
    ann_sharpe = (avg_monthly / monthly_std) * math.sqrt(12) if monthly_std > 0 else 0
    neg = monthly[monthly < 0]
    downside_std = neg.std() if len(neg) > 1 else (neg.std() if len(neg) == 1 else 0)
    sortino = (avg_monthly / downside_std) * math.sqrt(12) if downside_std > 0 else 0

    # Annual
    trades["year"] = trades["exit_dt"].dt.year
    annual = trades.groupby("year")["pnl"].sum()

    # Drawdown
    cum = trades["pnl"].cumsum()
    peak = cum.cummax()
    dd = peak - cum
    max_dd = dd.max()
    ann_return = total_pnl / (n_months / 12) if n_months else 0
    calmar = ann_return / max_dd if max_dd > 0 else 0

    # Gross win/loss
    gross_win = trades.loc[trades["pnl"] > 0, "pnl"].sum()
    gross_loss = abs(trades.loc[trades["pnl"] < 0, "pnl"].sum())
    pf = gross_win / gross_loss if gross_loss > 0 else 0

    print("\n" + "=" * 80)
    print("PORTFOLIO BACKTEST RESULTS (ALL STRATEGIES COMBINED)")
    print("=" * 80)
    print(f"Period:              {trades['entry_date'].min()} to {trades['exit_date'].max()}")
    print(f"Total trades:        {n}")
    print(f"Win rate:            {win_rate:.1f}%  ({wins}W / {losses}L)")
    print(f"Profit factor:       {pf:.2f}")
    print()
    print(f"Total P&L:           ${total_pnl:>12,.0f}")
    print(f"Annual P&L (avg):    ${ann_return:>12,.0f}")
    print(f"Avg monthly P&L:     ${avg_monthly:>12,.0f}")
    print(f"Best month:          ${best_month:>12,.0f}")
    print(f"Worst month:         ${worst_month:>12,.0f}")
    print(f"Profitable months:   {profitable_months}/{n_months} ({100*profitable_months/n_months:.0f}%)")
    print()
    print(f"Max drawdown:        ${max_dd:>12,.0f}  ({max_dd/TOTAL_ACCOUNT*100:.1f}% of account)")
    print(f"Sharpe (ann):        {ann_sharpe:>12.2f}")
    print(f"Sortino (ann):       {sortino:>12.2f}")
    print(f"Calmar:              {calmar:>12.2f}")
    print()
    print(f"Starting equity:     ${TOTAL_ACCOUNT:>12,}")
    print(f"Final equity:        ${TOTAL_ACCOUNT + total_pnl:>12,.0f}")
    print(f"Total return:        {total_pnl/TOTAL_ACCOUNT*100:>11.1f}%")
    print(f"CAGR:                {((1 + total_pnl/TOTAL_ACCOUNT)**(12/n_months) - 1)*100 if n_months > 0 else 0:>11.1f}%")

    print("\n" + "-" * 80)
    print("ANNUAL BREAKDOWN")
    print("-" * 80)
    print(f"{'Year':<8} {'P&L':>12} {'Trades':>8} {'Win%':>8} {'Cum P&L':>14} {'Equity':>14} {'ROI':>8}")
    cum_pnl = 0
    for year in sorted(annual.index):
        yr_trades = trades[trades["year"] == year]
        yr_pnl = annual[year]
        yr_wins = (yr_trades["pnl"] > 0).sum()
        yr_n = len(yr_trades)
        yr_wr = 100 * yr_wins / yr_n if yr_n else 0
        cum_pnl += yr_pnl
        equity = TOTAL_ACCOUNT + cum_pnl
        roi = yr_pnl / TOTAL_ACCOUNT * 100
        print(f"{year:<8} ${yr_pnl:>11,.0f} {yr_n:>8} {yr_wr:>7.0f}% ${cum_pnl:>13,.0f} ${equity:>13,.0f} {roi:>7.1f}%")

    print("\n" + "-" * 80)
    print("BY STRATEGY")
    print("-" * 80)
    print(f"{'Strategy':<30} {'Trades':>8} {'P&L':>12} {'Win%':>8} {'Avg P&L':>10}")
    for cid in trades["config_id"].unique():
        ct = trades[trades["config_id"] == cid]
        ct_pnl = ct["pnl"].sum()
        ct_wins = (ct["pnl"] > 0).sum()
        ct_wr = 100 * ct_wins / len(ct) if len(ct) else 0
        ct_avg = ct["pnl"].mean()
        print(f"{cid:<30} {len(ct):>8} ${ct_pnl:>11,.0f} {ct_wr:>7.0f}% ${ct_avg:>9,.0f}")

    print("\n" + "-" * 80)
    print("BY EXIT REASON")
    print("-" * 80)
    for reason in trades["exit_reason"].unique():
        rt = trades[trades["exit_reason"] == reason]
        print(f"  {reason:<20} {len(rt):>6} trades  avg P&L ${rt['pnl'].mean():>8,.0f}")

    print("=" * 80)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Portfolio backtest")
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end", default="2025-12-31")
    args = parser.parse_args()

    start = datetime.strptime(args.start, "%Y-%m-%d").date()
    end = datetime.strptime(args.end, "%Y-%m-%d").date()

    trades = run_portfolio_backtest(start, end)
    print_portfolio_report(trades)
