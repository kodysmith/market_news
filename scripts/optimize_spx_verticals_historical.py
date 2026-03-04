#!/usr/bin/env python3
"""Historical SPX credit-structure optimizer (put verticals + iron condors).

Uses 10+ years of SPX + VIX daily closes from yfinance and BS-derived pricing
to backtest thousands of parameter combinations across weekly Friday expirations.

Grid dimensions:
- Strategy: put_vertical, iron_condor
- DTE: configurable (e.g. 7, 14, 21, 30, 45)
- Short delta: configurable (e.g. 0.08 .. 0.30)
- Wing width: configurable (e.g. 10 .. 50 SPX points)
- Profit target: % of max profit to close (e.g. 40, 50, 60, 70%)
- Stop loss: multiple of credit (e.g. 1.25, 1.5, 2.0, 2.5x)

Entry filters (optional):
- VIX max / min
- VIX percentile (IV rank proxy)
- SPX above 200d SMA (trend filter)
- Minimum credit-to-width ratio

Ranking metrics:
- avg_pnl (expected value)
- win_rate
- profit_factor
- sharpe_like (avg / std)
- max_drawdown
- cvar_5 (worst 5% average)
- consistency_score (composite)
"""
from __future__ import annotations

import argparse
import itertools
import math
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

R = 0.05
MULT = 100.0


def _d1(S: float, K: float, T: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return float("nan")
    return (np.log(S / K) + (R + 0.5 * sigma * sigma) * T) / (sigma * np.sqrt(T))


def bs_put_delta(S: float, K: float, T: float, sigma: float) -> float:
    d1v = _d1(S, K, T, sigma)
    return float(norm.cdf(d1v) - 1.0) if not np.isnan(d1v) else float("nan")


def bs_call_delta(S: float, K: float, T: float, sigma: float) -> float:
    d1v = _d1(S, K, T, sigma)
    return float(norm.cdf(d1v)) if not np.isnan(d1v) else float("nan")


def bs_put_price(S: float, K: float, T: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0:
        return max(K - S, 0.0)
    d1v = _d1(S, K, T, sigma)
    d2v = d1v - sigma * math.sqrt(T)
    return float(K * math.exp(-R * T) * norm.cdf(-d2v) - S * norm.cdf(-d1v))


def bs_call_price(S: float, K: float, T: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0:
        return max(S - K, 0.0)
    d1v = _d1(S, K, T, sigma)
    d2v = d1v - sigma * math.sqrt(T)
    return float(S * norm.cdf(d1v) - K * math.exp(-R * T) * norm.cdf(d2v))


def strike_for_delta(
    S: float, T: float, target_delta: float, sigma: float, is_put: bool,
) -> float | None:
    if T <= 0 or sigma <= 0 or S <= 0:
        return None
    lo, hi = S * 0.70, S * 1.30
    target = -abs(target_delta) if is_put else abs(target_delta)
    for _ in range(100):
        mid = (lo + hi) / 2.0
        d = bs_put_delta(S, mid, T, sigma) if is_put else bs_call_delta(S, mid, T, sigma)
        if np.isnan(d):
            return None
        if abs(d - target) < 1e-6:
            return mid
        if is_put:
            if d > target:
                lo = mid
            else:
                hi = mid
        else:
            if d < target:
                hi = mid
            else:
                lo = mid
    return (lo + hi) / 2.0


def load_spx_vix(start: date, end: date) -> tuple[pd.DataFrame, pd.DataFrame]:
    import yfinance as yf
    spx = yf.Ticker("^GSPC").history(start=start.isoformat(), end=(end + timedelta(days=10)).isoformat())
    vix = yf.Ticker("^VIX").history(start=start.isoformat(), end=(end + timedelta(days=10)).isoformat())
    for df in (spx, vix):
        if not df.empty:
            df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    return spx, vix


def friday_expirations(start: date, end: date) -> list[date]:
    out = []
    d = start
    while d <= end:
        if d.weekday() == 4:
            out.append(d)
        d += timedelta(days=1)
    return out


def get_trading_days(start: date, end: date) -> list[date]:
    try:
        from scripts.trading_calendar import get_trading_days_index
        idx = get_trading_days_index(start, end)
        return [pd.Timestamp(ts).date() for ts in idx]
    except Exception:
        out = []
        d = start
        while d <= end:
            if d.weekday() < 5:
                out.append(d)
            d += timedelta(days=1)
        return out


def simulate_one(
    strategy: str,
    spx_close: pd.Series,
    entry_date: date,
    exp_date: date,
    iv_entry: float,
    short_put_delta: float,
    short_call_delta: float,
    width: float,
    profit_target_pct: float,
    stop_mult: float,
    slippage_per_leg: float,
    commission_per_leg: float,
    trading_days: list[date],
) -> dict | None:
    if entry_date not in spx_close.index or exp_date not in spx_close.index:
        return None
    S_entry = float(spx_close[entry_date])
    S_exp = float(spx_close[exp_date])
    if S_entry <= 0:
        return None

    entry_idx = trading_days.index(entry_date)
    exp_idx = trading_days.index(exp_date)
    dte_trading = exp_idx - entry_idx
    if dte_trading <= 0:
        return None
    T_entry = dte_trading / 252.0

    sigma = iv_entry / 100.0 if iv_entry > 3 else iv_entry

    K_put_short = strike_for_delta(S_entry, T_entry, short_put_delta, sigma, is_put=True)
    if K_put_short is None:
        return None
    K_put_long = K_put_short - width
    if K_put_long <= 0:
        return None

    put_short_px = bs_put_price(S_entry, K_put_short, T_entry, sigma)
    put_long_px = bs_put_price(S_entry, K_put_long, T_entry, sigma)
    put_credit = put_short_px - put_long_px

    call_credit = 0.0
    K_call_short = K_call_long = None
    n_legs = 4
    if strategy == "iron_condor":
        K_call_short = strike_for_delta(S_entry, T_entry, short_call_delta, sigma, is_put=False)
        if K_call_short is None:
            return None
        K_call_long = K_call_short + width
        call_short_px = bs_call_price(S_entry, K_call_short, T_entry, sigma)
        call_long_px = bs_call_price(S_entry, K_call_long, T_entry, sigma)
        call_credit = call_short_px - call_long_px
        n_legs = 8

    total_credit = put_credit + call_credit
    entry_slippage = n_legs * slippage_per_leg / 2.0
    credit_adj = total_credit - entry_slippage
    if credit_adj <= 0:
        return None

    max_loss_put = width - put_credit
    if strategy == "iron_condor":
        max_loss = max(width - put_credit, width - call_credit)
    else:
        max_loss = max_loss_put
    if max_loss <= 0:
        return None

    close_threshold = (1.0 - profit_target_pct) * credit_adj
    stop_threshold = stop_mult * credit_adj

    exit_date = exp_date
    exit_reason = "expiry"
    exit_debit = None

    for i in range(entry_idx + 1, exp_idx):
        d = trading_days[i]
        if d not in spx_close.index:
            continue
        S_d = float(spx_close[d])
        dte_left = exp_idx - i
        T_d = dte_left / 252.0
        if T_d <= 0:
            break
        put_short_d = bs_put_price(S_d, K_put_short, T_d, sigma)
        put_long_d = bs_put_price(S_d, K_put_long, T_d, sigma)
        spread_val = (put_short_d - put_long_d)
        if strategy == "iron_condor":
            call_short_d = bs_call_price(S_d, K_call_short, T_d, sigma)
            call_long_d = bs_call_price(S_d, K_call_long, T_d, sigma)
            spread_val += (call_short_d - call_long_d)
        if spread_val <= close_threshold:
            exit_date = d
            exit_reason = f"tp_{int(profit_target_pct*100)}"
            exit_debit = spread_val
            break
        if spread_val >= stop_threshold:
            exit_date = d
            exit_reason = f"sl_{stop_mult}"
            exit_debit = spread_val
            break

    if exit_reason == "expiry":
        put_settlement = max(0.0, K_put_short - S_exp) - max(0.0, K_put_long - S_exp)
        settlement = put_settlement
        if strategy == "iron_condor":
            call_settlement = max(0.0, S_exp - K_call_short) - max(0.0, S_exp - K_call_long)
            settlement += call_settlement
        commission = (n_legs / 2) * commission_per_leg
        pnl = (credit_adj - settlement) * MULT - commission
    else:
        exit_slippage = n_legs * slippage_per_leg / 2.0
        debit_adj = (exit_debit or 0) + exit_slippage
        commission = n_legs * commission_per_leg
        pnl = (credit_adj - debit_adj) * MULT - commission

    return {
        "entry_date": entry_date,
        "exit_date": exit_date,
        "exp_date": exp_date,
        "dte": dte_trading,
        "spot_entry": S_entry,
        "spot_expiry": S_exp,
        "iv_entry": sigma,
        "K_put_short": K_put_short,
        "K_put_long": K_put_long,
        "K_call_short": K_call_short,
        "K_call_long": K_call_long,
        "entry_credit": credit_adj,
        "exit_reason": exit_reason,
        "pnl": pnl,
        "win": pnl > 0,
    }


def compute_metrics(pnls: np.ndarray) -> dict:
    n = len(pnls)
    if n == 0:
        return {
            "n_trades": 0, "win_rate": 0.0, "avg_pnl": 0.0, "total_pnl": 0.0,
            "pnl_std": 0.0, "sharpe_like": 0.0, "profit_factor": 0.0,
            "max_drawdown": 0.0, "cvar_5": 0.0, "worst_trade": 0.0,
            "best_trade": 0.0, "consistency_score": 0.0,
        }
    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]
    win_rate = len(wins) / n
    avg_pnl = float(np.mean(pnls))
    std_pnl = float(np.std(pnls))
    sharpe = avg_pnl / std_pnl if std_pnl > 0 else 0.0
    gross_win = float(np.sum(wins)) if len(wins) else 0.0
    gross_loss = abs(float(np.sum(losses))) if len(losses) else 1e-9
    profit_factor = gross_win / gross_loss if gross_loss > 0 else float("inf")
    equity = np.cumsum(pnls)
    peak = np.maximum.accumulate(equity)
    dd = peak - equity
    max_dd = float(np.max(dd)) if len(dd) else 0.0
    k = max(1, int(np.ceil(n * 0.05)))
    sorted_pnls = np.sort(pnls)
    cvar_5 = float(np.mean(sorted_pnls[:k]))
    worst = float(np.min(pnls))
    best = float(np.max(pnls))
    consistency = (avg_pnl * win_rate * profit_factor) / (max_dd + 1.0)
    return {
        "n_trades": n,
        "win_rate": round(win_rate, 4),
        "avg_pnl": round(avg_pnl, 2),
        "total_pnl": round(float(np.sum(pnls)), 2),
        "pnl_std": round(std_pnl, 2),
        "sharpe_like": round(sharpe, 4),
        "profit_factor": round(profit_factor, 4),
        "max_drawdown": round(max_dd, 2),
        "cvar_5": round(cvar_5, 2),
        "worst_trade": round(worst, 2),
        "best_trade": round(best, 2),
        "consistency_score": round(consistency, 6),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Optimize SPX put vertical / iron condor for expected return and consistency (10-year backtest)."
    )
    parser.add_argument("--start", type=str, default="2015-01-01")
    parser.add_argument("--end", type=str, default="2025-12-31")
    parser.add_argument("--strategies", type=str, nargs="+", default=["put_vertical", "iron_condor"])
    parser.add_argument("--dte-targets", type=str, default="7,14,21,30,45")
    parser.add_argument("--short-deltas", type=str, default="0.08,0.10,0.12,0.15,0.18,0.20,0.25,0.30")
    parser.add_argument("--widths", type=str, default="10,15,20,25,30,40,50")
    parser.add_argument("--profit-targets", type=str, default="0.40,0.50,0.60,0.70")
    parser.add_argument("--stop-mults", type=str, default="1.25,1.5,2.0,2.5")
    parser.add_argument("--iv-source", type=str, default="vix", choices=["vix", "realized"],
                        help="Use VIX or realized vol for BS pricing")
    parser.add_argument("--vix-max", type=float, default=None, help="Only enter when VIX <= this")
    parser.add_argument("--vix-min", type=float, default=None, help="Only enter when VIX >= this")
    parser.add_argument("--vix-pctile-max", type=float, default=None, help="Only enter when VIX percentile <= this (e.g. 80)")
    parser.add_argument("--trend-filter", action="store_true", help="Only enter when SPX > 200d SMA")
    parser.add_argument("--min-credit-width-ratio", type=float, default=0.0, help="Min credit/width to accept entry")
    parser.add_argument("--slippage-per-leg", type=float, default=0.10)
    parser.add_argument("--commission-per-leg", type=float, default=0.65)
    parser.add_argument("--min-trades", type=int, default=30)
    parser.add_argument("--top", type=int, default=15)
    parser.add_argument("--output-csv", type=Path, default=None)
    args = parser.parse_args()

    start = datetime.strptime(args.start, "%Y-%m-%d").date()
    end = datetime.strptime(args.end, "%Y-%m-%d").date()

    print(f"Loading SPX + VIX data ({args.start} to {args.end})...", file=sys.stderr)
    spx_df, vix_df = load_spx_vix(start, end)
    if spx_df.empty:
        print("No SPX data.", file=sys.stderr)
        return 1

    spx_close = spx_df["Close"].copy()
    spx_close.index = spx_close.index.date
    vix_close = vix_df["Close"].copy() if not vix_df.empty else pd.Series(dtype=float)
    if not vix_close.empty:
        vix_close.index = vix_close.index.date

    spx_ma200 = spx_df["Close"].rolling(200).mean()
    spx_ma200.index = spx_ma200.index.date

    realized_vol = spx_df["Close"].pct_change().rolling(20).std() * np.sqrt(252)
    realized_vol.index = realized_vol.index.date

    vix_rolling = None
    if not vix_close.empty and args.vix_pctile_max is not None:
        vix_rolling = vix_close.rolling(252, min_periods=60)

    trading = get_trading_days(start, end + timedelta(days=60))
    trading_set = set(trading)
    expirations = [e for e in friday_expirations(start, end) if e in trading_set]
    print(f"SPX days: {len(spx_close)}, VIX days: {len(vix_close)}, Expirations: {len(expirations)}", file=sys.stderr)

    dte_targets = [int(x) for x in args.dte_targets.split(",")]
    deltas = [float(x) for x in args.short_deltas.split(",")]
    widths = [float(x) for x in args.widths.split(",")]
    pts = [float(x) for x in args.profit_targets.split(",")]
    sls = [float(x) for x in args.stop_mults.split(",")]

    combos = list(itertools.product(args.strategies, dte_targets, deltas, widths, pts, sls))
    total_combos = len(combos)
    print(f"Parameter combinations: {total_combos}", file=sys.stderr)

    all_rows: list[dict] = []
    for ci, (strategy, dte_target, short_delta, width, pt, sl) in enumerate(combos):
        if (ci + 1) % 500 == 0:
            print(f"  {ci+1}/{total_combos}...", file=sys.stderr)

        pnls: list[float] = []
        for exp in expirations:
            try:
                exp_pos = trading.index(exp)
            except ValueError:
                continue
            if exp_pos < dte_target:
                continue
            entry_date = trading[exp_pos - dte_target]
            if entry_date not in spx_close.index or exp not in spx_close.index:
                continue

            if args.vix_max is not None and entry_date in vix_close.index:
                if float(vix_close[entry_date]) > args.vix_max:
                    continue
            if args.vix_min is not None and entry_date in vix_close.index:
                if float(vix_close[entry_date]) < args.vix_min:
                    continue
            if args.vix_pctile_max is not None and vix_rolling is not None and entry_date in vix_close.index:
                vix_val = float(vix_close[entry_date])
                vix_min_r = vix_rolling.min().get(entry_date)
                vix_max_r = vix_rolling.max().get(entry_date)
                if vix_min_r is not None and vix_max_r is not None and (vix_max_r - vix_min_r) > 0:
                    pctile = 100.0 * (vix_val - vix_min_r) / (vix_max_r - vix_min_r)
                    if pctile > args.vix_pctile_max:
                        continue
            if args.trend_filter and entry_date in spx_ma200.index:
                ma = spx_ma200.get(entry_date)
                if ma is not None and not np.isnan(ma):
                    if float(spx_close[entry_date]) < float(ma):
                        continue

            if args.iv_source == "vix" and entry_date in vix_close.index:
                iv_entry = float(vix_close[entry_date]) / 100.0
            elif entry_date in realized_vol.index and not np.isnan(realized_vol[entry_date]):
                iv_entry = float(realized_vol[entry_date])
            else:
                iv_entry = 0.20

            result = simulate_one(
                strategy=strategy,
                spx_close=spx_close,
                entry_date=entry_date,
                exp_date=exp,
                iv_entry=iv_entry,
                short_put_delta=short_delta,
                short_call_delta=short_delta,
                width=width,
                profit_target_pct=pt,
                stop_mult=sl,
                slippage_per_leg=args.slippage_per_leg,
                commission_per_leg=args.commission_per_leg,
                trading_days=trading,
            )
            if result is None:
                continue
            if args.min_credit_width_ratio > 0:
                if result["entry_credit"] / width < args.min_credit_width_ratio:
                    continue
            pnls.append(result["pnl"])

        metrics = compute_metrics(np.array(pnls, dtype=float))
        cfg = {
            "strategy": strategy,
            "dte": dte_target,
            "short_delta": short_delta,
            "width": width,
            "profit_target": pt,
            "stop_mult": sl,
        }
        all_rows.append({**cfg, **metrics})

    results = pd.DataFrame(all_rows)
    if results.empty:
        print("No results.")
        return 0

    filtered = results.loc[results["n_trades"] >= args.min_trades].copy()
    if filtered.empty:
        print(f"No configs with >= {args.min_trades} trades. Showing top by n_trades.")
        filtered = results.sort_values("n_trades", ascending=False).head(args.top * 2)

    cols = [
        "strategy", "dte", "short_delta", "width", "profit_target", "stop_mult",
        "n_trades", "win_rate", "avg_pnl", "total_pnl", "profit_factor",
        "sharpe_like", "max_drawdown", "cvar_5", "worst_trade", "consistency_score",
    ]

    by_return = filtered.sort_values("avg_pnl", ascending=False)
    by_consistency = filtered.sort_values("consistency_score", ascending=False)
    by_winrate = filtered.sort_values(["win_rate", "avg_pnl"], ascending=[False, False])
    by_sharpe = filtered.sort_values("sharpe_like", ascending=False)

    print()
    print("=" * 120)
    print("TOP BY EXPECTED RETURN (avg_pnl)")
    print("=" * 120)
    print(by_return[cols].head(args.top).to_string(index=False))

    print()
    print("=" * 120)
    print("TOP BY WIN RATE (+ avg_pnl tiebreak)")
    print("=" * 120)
    print(by_winrate[cols].head(args.top).to_string(index=False))

    print()
    print("=" * 120)
    print("TOP BY SHARPE-LIKE (avg/std)")
    print("=" * 120)
    print(by_sharpe[cols].head(args.top).to_string(index=False))

    print()
    print("=" * 120)
    print("TOP BY CONSISTENCY SCORE (return * winrate * PF / maxDD)")
    print("=" * 120)
    print(by_consistency[cols].head(args.top).to_string(index=False))

    if args.output_csv:
        out = args.output_csv.resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        results.sort_values("consistency_score", ascending=False).to_csv(out, index=False)
        print(f"\nWrote {len(results)} configs to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
