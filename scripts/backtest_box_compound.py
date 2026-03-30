#!/usr/bin/env python3
"""Backtest: Box Spread Compounding Strategy

The cycle:
1. Sell bear call spread (negative GEX) or iron condor (positive GEX)
2. When profit hits 40%+, convert to box spread (lock profit, free margin)
3. Deploy freed margin into BOXX (risk-free yield) + new spread
4. Repeat — compounding winners into new positions

Key insight: box conversion turns unrealized profit into guaranteed cash,
freeing margin to open the next trade without adding new capital.
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
BOXX_ANNUAL_RATE = 0.045  # 4.5% risk-free yield on parked cash
BOX_CONVERSION_THRESHOLD = 0.40  # convert to box when 40% of credit captured
MAX_CONCURRENT_POSITIONS = 4
MARGIN_PER_POSITION = 30_000  # approximate margin per spread position
STARTING_CAPITAL = 200_000


def run_backtest(start: date, end: date):
    print("Loading data...", flush=True)
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

    # SMA for regime
    sma50 = spx_df["Close"].rolling(50).mean()
    sma50.index = sma50.index.date

    # RV ratio for vol regime
    log_ret = np.log(spx_df["Close"] / spx_df["Close"].shift(1))
    rv5 = log_ret.rolling(5).std() * np.sqrt(252)
    rv10 = log_ret.rolling(10).std() * np.sqrt(252)
    rv_ratio = rv5 / rv10
    rv_ratio.index = rv_ratio.index.date

    # State
    open_positions = []  # list of dicts
    boxed_positions = []  # positions converted to box (locked profit)
    closed_trades = []
    cash = STARTING_CAPITAL
    boxx_balance = 0.0  # cash parked in BOXX
    total_boxx_yield = 0.0
    cumulative_pnl = 0.0
    equity_curve = []

    for day_i, today in enumerate(trading_days):
        if today < start or today > end:
            continue
        if today not in spx_close.index:
            continue

        S = float(spx_close[today])
        vix_val = float(vix_close.get(today, 20.0))
        sigma = vix_val / 100.0 if vix_val > 3 else vix_val
        sma50_val = sma50.get(today)
        rv_val = rv_ratio.get(today, 1.0)
        if isinstance(rv_val, float) and np.isnan(rv_val):
            rv_val = 1.0

        # Determine regime
        above_sma50 = S > sma50_val if sma50_val and not np.isnan(sma50_val) else True
        # Proxy for GEX: use RV/IV ratio — high RV = negative gamma proxy
        negative_gex_proxy = rv_val > 1.05 or not above_sma50

        # --- 1. Check existing positions for box conversion ---
        to_convert = []
        for pos in open_positions:
            exp_idx = pos["exp_idx"]
            dte_left = exp_idx - day_i
            if dte_left <= 0:
                # Expired — settle
                S_exp = float(spx_close.get(pos["exp_date"], S))
                settlement = 0.0
                if pos["K_put_short"]:
                    settlement += max(0, pos["K_put_short"] - S_exp) - max(0, pos["K_put_long"] - S_exp)
                if pos["K_call_short"]:
                    settlement += max(0, S_exp - pos["K_call_short"]) - max(0, S_exp - pos["K_call_long"])
                pnl = (pos["credit_adj"] - settlement) * MULT * pos["n_contracts"]
                pnl -= pos["n_legs"] * COMMISSION_PER_LEG
                cumulative_pnl += pnl
                cash += pnl + pos["margin_held"]  # return margin
                closed_trades.append({
                    "entry_date": pos["entry_date"], "exit_date": today,
                    "exit_reason": "expiry", "pnl": round(pnl, 2),
                    "strategy": pos["strategy"],
                })
                to_convert.append(pos)
                continue

            T_d = dte_left / 252.0
            if T_d <= 0:
                continue

            # Price current spread
            sigma_d = float(vix_close.get(today, vix_val))
            sigma_d = sigma_d / 100.0 if sigma_d > 3 else sigma_d

            spread_val = 0.0
            if pos["K_put_short"]:
                spread_val += bs_put_price(S, pos["K_put_short"], T_d, sigma_d) - \
                              bs_put_price(S, pos["K_put_long"], T_d, sigma_d)
            if pos["K_call_short"]:
                spread_val += bs_call_price(S, pos["K_call_short"], T_d, sigma_d) - \
                              bs_call_price(S, pos["K_call_long"], T_d, sigma_d)

            pnl_pct = (pos["credit_adj"] - spread_val) / pos["credit_adj"]

            # Stop loss: 2x
            if spread_val >= pos["credit_adj"] * 2.0:
                pnl = (pos["credit_adj"] - spread_val) * MULT * pos["n_contracts"]
                pnl -= pos["n_legs"] * COMMISSION_PER_LEG * 2
                cumulative_pnl += pnl
                cash += pnl + pos["margin_held"]
                closed_trades.append({
                    "entry_date": pos["entry_date"], "exit_date": today,
                    "exit_reason": "stop_loss", "pnl": round(pnl, 2),
                    "strategy": pos["strategy"],
                })
                to_convert.append(pos)
                continue

            # Box conversion: when profit >= 40%
            if pnl_pct >= BOX_CONVERSION_THRESHOLD:
                # Lock in profit via box conversion
                locked_profit = (pos["credit_adj"] - spread_val) * MULT * pos["n_contracts"]
                box_value = pos["width"] * MULT * pos["n_contracts"]

                # Put spread credit to complete the box (approximate)
                put_credit = spread_val  # roughly symmetric
                total_box_credit = (pos["credit_adj"] + put_credit) * MULT * pos["n_contracts"]

                # Slippage for the put spread legs
                put_slippage = 4 * SLIPPAGE_PER_LEG / 2.0
                total_box_credit -= put_slippage * MULT * pos["n_contracts"]

                # Cash freed = total credits - what we owe at expiry is handled at settlement
                # For now: profit is locked, margin is freed, cash goes to BOXX
                commission = pos["n_legs"] * COMMISSION_PER_LEG  # put spread commission
                net_locked = locked_profit - commission

                cumulative_pnl += net_locked
                cash += pos["margin_held"]  # margin returned
                boxx_balance += pos["margin_held"]  # deploy to BOXX

                closed_trades.append({
                    "entry_date": pos["entry_date"], "exit_date": today,
                    "exit_reason": "box_conversion", "pnl": round(net_locked, 2),
                    "strategy": pos["strategy"],
                    "days_held": (today - pos["entry_date"]).days,
                })

                boxed_positions.append({
                    "convert_date": today,
                    "exp_date": pos["exp_date"],
                    "locked_profit": net_locked,
                    "margin_freed": pos["margin_held"],
                })
                to_convert.append(pos)

        for pos in to_convert:
            if pos in open_positions:
                open_positions.remove(pos)

        # --- 2. Settle expired box positions, return BOXX capital ---
        expired_boxes = []
        for box in boxed_positions:
            if today >= box["exp_date"]:
                boxx_balance -= box["margin_freed"]
                expired_boxes.append(box)
        for box in expired_boxes:
            boxed_positions.remove(box)

        # --- 3. BOXX daily yield ---
        daily_boxx_yield = boxx_balance * BOXX_ANNUAL_RATE / 365
        total_boxx_yield += daily_boxx_yield
        cash += daily_boxx_yield

        # --- 4. Try to open new positions ---
        available_margin = cash - boxx_balance  # don't double-count BOXX
        n_open = len(open_positions)

        if n_open < MAX_CONCURRENT_POSITIONS and available_margin >= MARGIN_PER_POSITION:
            # Find nearest Friday expiration ~14 DTE out
            target_exp = today + timedelta(days=14)
            exp_date = None
            for d in trading_days:
                if d >= target_exp and d.weekday() == 4:  # Friday
                    exp_date = d
                    break
            if exp_date is None:
                for d in trading_days:
                    if d > today + timedelta(days=10):
                        exp_date = d
                        break

            if exp_date and exp_date in trading_idx:
                entry_idx = day_i
                exp_idx = trading_idx[exp_date]
                dte_trading = exp_idx - entry_idx
                if dte_trading > 0:
                    T_entry = dte_trading / 252.0

                    if negative_gex_proxy:
                        # Bear call spread only
                        K_call_short = strike_for_delta(S, T_entry, 0.12, sigma, is_put=False)
                        if K_call_short:
                            K_call_short = round(K_call_short / 5) * 5
                            K_call_long = K_call_short + 50
                            credit = bs_call_price(S, K_call_short, T_entry, sigma) - \
                                     bs_call_price(S, K_call_long, T_entry, sigma)
                            n_legs = 4
                            slippage = n_legs * SLIPPAGE_PER_LEG / 2.0
                            credit_adj = credit - slippage
                            n_cts = 6

                            if credit_adj > 0:
                                margin = 50 * MULT * n_cts  # max loss as margin
                                if margin <= available_margin:
                                    cash -= margin
                                    open_positions.append({
                                        "entry_date": today, "exp_date": exp_date,
                                        "entry_idx": entry_idx, "exp_idx": exp_idx,
                                        "K_put_short": None, "K_put_long": None,
                                        "K_call_short": K_call_short, "K_call_long": K_call_long,
                                        "credit_adj": credit_adj, "n_contracts": n_cts,
                                        "n_legs": n_legs, "width": 50,
                                        "margin_held": margin, "strategy": "bear_call",
                                    })
                    else:
                        # Iron condor
                        K_put_short = strike_for_delta(S, T_entry, 0.10, sigma, is_put=True)
                        K_call_short = strike_for_delta(S, T_entry, 0.10, sigma, is_put=False)
                        if K_put_short and K_call_short:
                            K_put_short = round(K_put_short / 5) * 5
                            K_call_short = round(K_call_short / 5) * 5
                            K_put_long = K_put_short - 30
                            K_call_long = K_call_short + 30
                            if K_put_long > 0:
                                put_cr = bs_put_price(S, K_put_short, T_entry, sigma) - \
                                         bs_put_price(S, K_put_long, T_entry, sigma)
                                call_cr = bs_call_price(S, K_call_short, T_entry, sigma) - \
                                          bs_call_price(S, K_call_long, T_entry, sigma)
                                credit = put_cr + call_cr
                                n_legs = 8
                                slippage = n_legs * SLIPPAGE_PER_LEG / 2.0
                                credit_adj = credit - slippage
                                n_cts = 10

                                if credit_adj > 0:
                                    margin = 30 * MULT * n_cts
                                    if margin <= available_margin:
                                        cash -= margin
                                        open_positions.append({
                                            "entry_date": today, "exp_date": exp_date,
                                            "entry_idx": entry_idx, "exp_idx": exp_idx,
                                            "K_put_short": K_put_short, "K_put_long": K_put_long,
                                            "K_call_short": K_call_short, "K_call_long": K_call_long,
                                            "credit_adj": credit_adj, "n_contracts": n_cts,
                                            "n_legs": n_legs, "width": 30,
                                            "margin_held": margin, "strategy": "iron_condor",
                                        })

        # Track equity
        total_equity = cash + boxx_balance + sum(p["margin_held"] for p in open_positions)
        equity_curve.append({
            "date": today, "equity": total_equity, "cash": cash,
            "boxx": boxx_balance, "positions": len(open_positions),
            "boxed": len(boxed_positions), "cum_pnl": cumulative_pnl,
        })

    return pd.DataFrame(closed_trades), pd.DataFrame(equity_curve), total_boxx_yield


def print_report(trades, equity, boxx_yield):
    if trades.empty:
        print("No trades.")
        return

    n = len(trades)
    total = trades["pnl"].sum()
    wins = (trades["pnl"] > 0).sum()
    losses = (trades["pnl"] < 0).sum()

    # By exit reason
    by_reason = trades.groupby("exit_reason")["pnl"].agg(["count", "sum", "mean"])

    # By strategy
    by_strat = trades.groupby("strategy")["pnl"].agg(["count", "sum", "mean"])

    # Annual
    trades["year"] = pd.to_datetime(trades["exit_date"]).dt.year
    annual = trades.groupby("year")["pnl"].sum()

    # Monthly for Sharpe
    trades["month"] = pd.to_datetime(trades["exit_date"]).dt.to_period("M")
    monthly = trades.groupby("month")["pnl"].sum()
    sharpe = monthly.mean() / monthly.std() * math.sqrt(12) if monthly.std() > 0 else 0

    years = (pd.to_datetime(trades["exit_date"].max()) - pd.to_datetime(trades["exit_date"].min())).days / 365.25
    ann_pnl = total / years if years > 0 else 0

    print()
    print("=" * 75)
    print("BOX SPREAD COMPOUNDING STRATEGY — BACKTEST RESULTS")
    print("=" * 75)
    print(f"Period:        {trades['entry_date'].min()} to {trades['exit_date'].max()}")
    print(f"Total trades:  {n} ({wins}W / {losses}L = {100*wins/n:.0f}% win rate)")
    print(f"Total P&L:     ${total:>12,.0f}")
    print(f"Annual P&L:    ${ann_pnl:>12,.0f}")
    print(f"BOXX yield:    ${boxx_yield:>12,.0f}")
    print(f"Combined:      ${total + boxx_yield:>12,.0f}")
    print(f"Sharpe:        {sharpe:.2f}")
    print(f"Starting cap:  ${STARTING_CAPITAL:>12,}")
    print(f"Final equity:  ${equity['equity'].iloc[-1]:>12,.0f}" if not equity.empty else "")
    print()

    print("BY EXIT REASON:")
    for reason, row in by_reason.iterrows():
        print(f"  {reason:<20} {int(row['count']):>5} trades  total ${row['sum']:>10,.0f}  avg ${row['mean']:>7,.0f}")

    print()
    print("BY STRATEGY:")
    for strat, row in by_strat.iterrows():
        print(f"  {strat:<20} {int(row['count']):>5} trades  total ${row['sum']:>10,.0f}  avg ${row['mean']:>7,.0f}")

    # Box conversion stats
    box_trades = trades[trades["exit_reason"] == "box_conversion"]
    if not box_trades.empty:
        print()
        print(f"BOX CONVERSIONS:")
        print(f"  Count:       {len(box_trades)}")
        print(f"  Total locked: ${box_trades['pnl'].sum():,.0f}")
        print(f"  Avg profit:   ${box_trades['pnl'].mean():,.0f}")
        if "days_held" in box_trades.columns:
            print(f"  Avg hold:     {box_trades['days_held'].mean():.0f} days")

    print()
    print("ANNUAL BREAKDOWN:")
    print(f"{'Year':<6} {'P&L':>12} {'Trades':>8} {'Equity':>14}")
    cum = STARTING_CAPITAL
    for year in sorted(annual.index):
        yr_pnl = annual[year]
        cum += yr_pnl
        yr_n = len(trades[trades["year"] == year])
        print(f"  {year:<4} ${yr_pnl:>11,.0f} {yr_n:>8} ${cum:>13,.0f}")

    print()
    print("CAPITAL EFFICIENCY:")
    if not equity.empty:
        avg_positions = equity["positions"].mean()
        avg_boxx = equity["boxx"].mean()
        max_positions = equity["positions"].max()
        print(f"  Avg open positions:  {avg_positions:.1f}")
        print(f"  Max open positions:  {max_positions}")
        print(f"  Avg BOXX balance:    ${avg_boxx:>10,.0f}")
        print(f"  Margin recycled via box: {len(box_trades)} times")
        if years > 0:
            roi = (total + boxx_yield) / STARTING_CAPITAL * 100
            ann_roi = ((1 + roi/100) ** (1/years) - 1) * 100
            print(f"  Total ROI:           {roi:.0f}%")
            print(f"  CAGR:                {ann_roi:.1f}%")

    print("=" * 75)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end", default="2025-12-31")
    args = parser.parse_args()

    start = datetime.strptime(args.start, "%Y-%m-%d").date()
    end = datetime.strptime(args.end, "%Y-%m-%d").date()

    trades, equity, boxx_yield = run_backtest(start, end)
    print_report(trades, equity, boxx_yield)
