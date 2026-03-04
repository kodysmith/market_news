#!/usr/bin/env python3
"""Optimize SPX credit structures for expected return and consistency.

Strategies:
- put_vertical: short put vertical (credit spread)
- iron_condor: put vertical + call vertical

Data source: data/spx_options snapshots (same normalized schema used by other SPX scripts).
"""
from __future__ import annotations

import argparse
import itertools
import math
import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.backtest_gex_7dte import (  # noqa: E402
    _get_iv,
    _sigma_from_iv,
    bs_put_delta,
    bs_put_price,
    discover_snapshot_dates,
    get_mid_from_chain,
    load_options_for_date,
    load_spot_series,
    years_to_expiry,
)
from scripts.trading_calendar import (  # noqa: E402
    get_trading_days_in_range,
    get_trading_days_index,
    trading_days_between,
)

RISK_FREE_RATE = 0.05
MULT = 100.0


@dataclass
class SpreadLeg:
    contract_type: str  # "put" or "call"
    strike_short: float
    strike_long: float
    sigma_short: float
    sigma_long: float


@dataclass
class TradeSpec:
    strategy: str
    entry_date: date
    expiry_date: date
    dte_entry: int
    entry_credit: float
    put_leg: SpreadLeg
    call_leg: SpreadLeg | None


def bs_call_delta(S: float, K: float, T: float, r: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return float("nan")
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    return float(norm.cdf(d1))


def bs_call_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0:
        return max(S - K, 0.0)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return float(S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2))


def parse_float_list(value: str) -> list[float]:
    return [float(v.strip()) for v in value.split(",") if v.strip()]


def parse_int_list(value: str) -> list[int]:
    return [int(v.strip()) for v in value.split(",") if v.strip()]


def pick_expiry_for_target_dte(
    expirations: list[date],
    entry_date: date,
    trading_index: pd.DatetimeIndex,
    dte_target: int,
    tolerance: int,
) -> tuple[date | None, int]:
    best_exp = None
    best_diff = 10_000
    best_dte = -1
    for exp in expirations:
        dte_actual = trading_days_between(trading_index, entry_date, exp)
        diff = abs(dte_actual - dte_target)
        if diff <= tolerance and diff < best_diff:
            best_exp = exp
            best_diff = diff
            best_dte = dte_actual
    return best_exp, best_dte


def pick_strike_for_delta(
    df_side: pd.DataFrame,
    spot: float,
    T: float,
    target_abs_delta: float,
    is_put: bool,
    default_iv: float,
) -> pd.Series | None:
    best_row = None
    best_diff = float("inf")
    for _, row in df_side.iterrows():
        iv = _get_iv(row, default_iv)
        if iv is None or iv <= 0:
            continue
        sigma = _sigma_from_iv(float(iv))
        k = float(row["strike"])
        if is_put:
            d = bs_put_delta(spot, k, T, RISK_FREE_RATE, sigma)
            target = -abs(target_abs_delta)
        else:
            d = bs_call_delta(spot, k, T, RISK_FREE_RATE, sigma)
            target = abs(target_abs_delta)
        if np.isnan(d):
            continue
        diff = abs(d - target)
        if diff < best_diff:
            best_diff = diff
            best_row = row
    return best_row


def nearest_strike(candidates: np.ndarray, target: float) -> float | None:
    if candidates.size == 0:
        return None
    idx = int(np.argmin(np.abs(candidates - target)))
    return float(candidates[idx])


def leg_value_from_snapshot(
    df: pd.DataFrame | None,
    d: date,
    exp: date,
    leg: SpreadLeg,
    spot: float | None,
    entry_date: date,
) -> float | None:
    short_mid = None
    long_mid = None
    if df is not None and not df.empty:
        short_mid = get_mid_from_chain(df, leg.strike_short, exp, leg.contract_type)
        long_mid = get_mid_from_chain(df, leg.strike_long, exp, leg.contract_type)
    if short_mid is not None and long_mid is not None:
        return max(short_mid - long_mid, 0.0)
    if spot is None:
        return None
    T = years_to_expiry(d, exp)
    if T <= 0:
        return 0.0
    if leg.contract_type == "put":
        short_px = bs_put_price(spot, leg.strike_short, T, RISK_FREE_RATE, leg.sigma_short)
        long_px = bs_put_price(spot, leg.strike_long, T, RISK_FREE_RATE, leg.sigma_long)
    else:
        short_px = bs_call_price(spot, leg.strike_short, T, RISK_FREE_RATE, leg.sigma_short)
        long_px = bs_call_price(spot, leg.strike_long, T, RISK_FREE_RATE, leg.sigma_long)
    return max(short_px - long_px, 0.0)


def expiry_leg_settlement(spot: float, leg: SpreadLeg) -> float:
    if leg.contract_type == "put":
        return max(0.0, leg.strike_short - spot) - max(0.0, leg.strike_long - spot)
    return max(0.0, spot - leg.strike_short) - max(0.0, spot - leg.strike_long)


def build_trade_for_day(
    df: pd.DataFrame,
    entry_date: date,
    spot: float,
    trading_index: pd.DatetimeIndex,
    strategy: str,
    dte_target: int,
    dte_tolerance: int,
    short_put_delta: float,
    short_call_delta: float,
    width: float,
    min_credit: float,
    default_iv: float,
) -> TradeSpec | None:
    exp_raw = pd.to_datetime(df["expiration_date"], errors="coerce").dt.date.dropna().unique()
    expirations = sorted([e for e in exp_raw if e > entry_date])
    if not expirations:
        return None
    expiry, dte_entry = pick_expiry_for_target_dte(expirations, entry_date, trading_index, dte_target, dte_tolerance)
    if expiry is None or dte_entry <= 0:
        return None
    T = years_to_expiry(entry_date, expiry)
    if T <= 0:
        return None

    ct_col = "contract_type" if "contract_type" in df.columns else "option_type"
    ct = df[ct_col].astype(str).str.lower()
    ex = pd.to_datetime(df["expiration_date"], errors="coerce").dt.date
    puts = df.loc[(ex == expiry) & ((ct == "put") | (ct == "p"))].copy()
    calls = df.loc[(ex == expiry) & ((ct == "call") | (ct == "c"))].copy()
    if puts.empty:
        return None
    put_short_row = pick_strike_for_delta(puts, spot, T, short_put_delta, is_put=True, default_iv=default_iv)
    if put_short_row is None:
        return None
    k_put_short = float(put_short_row["strike"])
    put_strikes = np.array(sorted(puts["strike"].dropna().astype(float).unique()))
    k_put_long = nearest_strike(put_strikes, k_put_short - width)
    if k_put_long is None or k_put_long >= k_put_short:
        return None

    put_sigma_short = _sigma_from_iv(float(_get_iv(put_short_row, default_iv) or default_iv))
    put_long_row = puts.loc[puts["strike"] == k_put_long].iloc[0]
    put_sigma_long = _sigma_from_iv(float(_get_iv(put_long_row, default_iv) or default_iv))
    put_leg = SpreadLeg(
        contract_type="put",
        strike_short=k_put_short,
        strike_long=float(k_put_long),
        sigma_short=put_sigma_short,
        sigma_long=put_sigma_long,
    )

    put_credit = leg_value_from_snapshot(df, entry_date, expiry, put_leg, spot, entry_date)
    if put_credit is None or put_credit <= 0:
        return None

    call_leg = None
    call_credit = 0.0
    if strategy == "iron_condor":
        if calls.empty:
            return None
        call_short_row = pick_strike_for_delta(calls, spot, T, short_call_delta, is_put=False, default_iv=default_iv)
        if call_short_row is None:
            return None
        k_call_short = float(call_short_row["strike"])
        call_strikes = np.array(sorted(calls["strike"].dropna().astype(float).unique()))
        k_call_long = nearest_strike(call_strikes, k_call_short + width)
        if k_call_long is None or k_call_long <= k_call_short:
            return None
        call_sigma_short = _sigma_from_iv(float(_get_iv(call_short_row, default_iv) or default_iv))
        call_long_row = calls.loc[calls["strike"] == k_call_long].iloc[0]
        call_sigma_long = _sigma_from_iv(float(_get_iv(call_long_row, default_iv) or default_iv))
        call_leg = SpreadLeg(
            contract_type="call",
            strike_short=k_call_short,
            strike_long=float(k_call_long),
            sigma_short=call_sigma_short,
            sigma_long=call_sigma_long,
        )
        call_credit = leg_value_from_snapshot(df, entry_date, expiry, call_leg, spot, entry_date)
        if call_credit is None or call_credit <= 0:
            return None

    total_credit = put_credit + call_credit
    if total_credit < min_credit:
        return None
    return TradeSpec(
        strategy=strategy,
        entry_date=entry_date,
        expiry_date=expiry,
        dte_entry=dte_entry,
        entry_credit=float(total_credit),
        put_leg=put_leg,
        call_leg=call_leg,
    )


def close_value_for_trade(
    trade: TradeSpec,
    d: date,
    df: pd.DataFrame | None,
    spot: float | None,
) -> float | None:
    put_val = leg_value_from_snapshot(df, d, trade.expiry_date, trade.put_leg, spot, trade.entry_date)
    if put_val is None:
        return None
    total = put_val
    if trade.call_leg is not None:
        call_val = leg_value_from_snapshot(df, d, trade.expiry_date, trade.call_leg, spot, trade.entry_date)
        if call_val is None:
            return None
        total += call_val
    return float(total)


def simulate_trade(
    trade: TradeSpec,
    df_cache: dict[date, pd.DataFrame | None],
    spot_series: pd.Series,
    trading_index: pd.DatetimeIndex,
    profit_target: float,
    stop_mult: float,
) -> tuple[float, bool]:
    sessions = get_trading_days_in_range(trading_index, trade.entry_date, trade.expiry_date)
    for d in sessions:
        if d <= trade.entry_date:
            continue
        if d >= trade.expiry_date:
            break
        spot = float(spot_series.loc[pd.Timestamp(d)]) if pd.Timestamp(d) in spot_series.index else None
        val = close_value_for_trade(trade, d, df_cache.get(d), spot)
        if val is None:
            continue
        if val <= (1.0 - profit_target) * trade.entry_credit:
            pnl = (trade.entry_credit - val) * MULT
            return pnl, pnl > 0
        if val >= stop_mult * trade.entry_credit:
            pnl = (trade.entry_credit - val) * MULT
            return pnl, pnl > 0
    settle_spot = float(spot_series.loc[pd.Timestamp(trade.expiry_date)]) if pd.Timestamp(trade.expiry_date) in spot_series.index else None
    if settle_spot is None:
        return 0.0, False
    settlement = expiry_leg_settlement(settle_spot, trade.put_leg)
    if trade.call_leg is not None:
        settlement += expiry_leg_settlement(settle_spot, trade.call_leg)
    pnl = (trade.entry_credit - settlement) * MULT
    return pnl, pnl > 0


def summarize_run(
    cfg: dict,
    pnls: list[float],
    wins: list[bool],
) -> dict:
    if not pnls:
        return {
            **cfg,
            "n_trades": 0,
            "win_rate": 0.0,
            "avg_pnl": 0.0,
            "total_pnl": 0.0,
            "pnl_std": 0.0,
            "sharpe_like": 0.0,
            "consistency_score": 0.0,
            "worst_trade": 0.0,
        }
    arr = np.array(pnls, dtype=float)
    n = len(arr)
    win_rate = float(sum(wins) / n)
    avg = float(np.mean(arr))
    std = float(np.std(arr))
    sharpe_like = float(avg / std) if std > 0 else 0.0
    worst = float(np.min(arr))
    consistency_score = avg * win_rate / (abs(worst) + 1.0)
    return {
        **cfg,
        "n_trades": n,
        "win_rate": win_rate,
        "avg_pnl": avg,
        "total_pnl": float(np.sum(arr)),
        "pnl_std": std,
        "sharpe_like": sharpe_like,
        "consistency_score": consistency_score,
        "worst_trade": worst,
    }


def optimize(args: argparse.Namespace) -> pd.DataFrame:
    data_dir = args.data_dir.resolve()
    snapshot_dates = discover_snapshot_dates(data_dir, days_limit=None)
    if not snapshot_dates:
        raise RuntimeError("No SPX snapshot dates found. Collect data first.")
    if args.start:
        start = datetime.strptime(args.start, "%Y-%m-%d").date()
        snapshot_dates = [d for d in snapshot_dates if d >= start]
    if args.end:
        end = datetime.strptime(args.end, "%Y-%m-%d").date()
        snapshot_dates = [d for d in snapshot_dates if d <= end]
    if not snapshot_dates:
        raise RuntimeError("No snapshot dates after start/end filtering.")

    cal_start = min(snapshot_dates)
    cal_end = max(snapshot_dates) + timedelta(days=90)
    trading_index = get_trading_days_index(cal_start, cal_end)
    spot_series = load_spot_series(cal_start, cal_end)
    if spot_series.empty:
        raise RuntimeError("Could not load SPX spot data.")
    spot_series = spot_series.reindex(pd.DatetimeIndex([pd.Timestamp(d) for d in get_trading_days_in_range(trading_index, cal_start, cal_end)])).dropna()

    df_cache: dict[date, pd.DataFrame | None] = {}
    for d in snapshot_dates:
        df_cache[d] = load_options_for_date(data_dir, d, parquet_warn=[False])

    dte_targets = parse_int_list(args.dte_targets)
    deltas = parse_float_list(args.short_deltas)
    widths = parse_float_list(args.widths)
    ptargets = parse_float_list(args.profit_targets)
    stops = parse_float_list(args.stop_mults)

    rows: list[dict] = []
    combos = list(itertools.product(args.strategies, dte_targets, deltas, widths, ptargets, stops))
    for strategy, dte_target, short_delta, width, pt, sl in combos:
        cfg = {
            "strategy": strategy,
            "dte_target": dte_target,
            "short_delta": short_delta,
            "width": width,
            "profit_target": pt,
            "stop_mult": sl,
        }
        pnls: list[float] = []
        wins: list[bool] = []
        for entry_date in snapshot_dates:
            if pd.Timestamp(entry_date) not in spot_series.index:
                continue
            spot = float(spot_series.loc[pd.Timestamp(entry_date)])
            df = df_cache.get(entry_date)
            if df is None or df.empty:
                continue
            trade = build_trade_for_day(
                df=df,
                entry_date=entry_date,
                spot=spot,
                trading_index=trading_index,
                strategy=strategy,
                dte_target=dte_target,
                dte_tolerance=args.dte_tolerance,
                short_put_delta=short_delta,
                short_call_delta=short_delta,
                width=width,
                min_credit=args.min_credit,
                default_iv=args.default_iv,
            )
            if trade is None:
                continue
            pnl, win = simulate_trade(
                trade=trade,
                df_cache=df_cache,
                spot_series=spot_series,
                trading_index=trading_index,
                profit_target=pt,
                stop_mult=sl,
            )
            pnls.append(pnl)
            wins.append(win)
        rows.append(summarize_run(cfg, pnls, wins))
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Optimize SPX put vertical/iron condor for return and consistency.")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "spx_options")
    parser.add_argument("--start", type=str, default=None, help="YYYY-MM-DD")
    parser.add_argument("--end", type=str, default=None, help="YYYY-MM-DD")
    parser.add_argument("--strategies", type=str, nargs="+", default=["put_vertical", "iron_condor"])
    parser.add_argument("--dte-targets", type=str, default="14,21,30")
    parser.add_argument("--short-deltas", type=str, default="0.12,0.15,0.18,0.20,0.25")
    parser.add_argument("--widths", type=str, default="20,30,40,50")
    parser.add_argument("--profit-targets", type=str, default="0.5,0.6")
    parser.add_argument("--stop-mults", type=str, default="1.5,2.0")
    parser.add_argument("--dte-tolerance", type=int, default=3)
    parser.add_argument("--min-credit", type=float, default=0.20)
    parser.add_argument("--default-iv", type=float, default=0.20, help="Fallback IV when chain IV is missing")
    parser.add_argument("--min-trades", type=int, default=8)
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--output-csv", type=Path, default=None)
    args = parser.parse_args()

    try:
        results = optimize(args)
    except Exception as e:
        print(f"Optimizer error: {e}", file=sys.stderr)
        return 1
    if results.empty:
        print("No results.")
        return 0

    filtered = results.loc[results["n_trades"] >= args.min_trades].copy()
    if filtered.empty:
        print(f"No configs met min-trades={args.min_trades}. Showing all configs.")
        filtered = results.copy()

    by_return = filtered.sort_values(by=["avg_pnl", "win_rate", "consistency_score"], ascending=[False, False, False])
    by_consistency = filtered.sort_values(by=["consistency_score", "win_rate", "avg_pnl"], ascending=[False, False, False])

    cols = [
        "strategy",
        "dte_target",
        "short_delta",
        "width",
        "profit_target",
        "stop_mult",
        "n_trades",
        "win_rate",
        "avg_pnl",
        "total_pnl",
        "pnl_std",
        "sharpe_like",
        "consistency_score",
        "worst_trade",
    ]
    print("=== Top by expected return (avg_pnl) ===")
    print(by_return[cols].head(args.top).to_string(index=False))
    print()
    print("=== Top by consistency-adjusted score ===")
    print(by_consistency[cols].head(args.top).to_string(index=False))

    if args.output_csv is not None:
        out = args.output_csv.resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        results.sort_values(by=["consistency_score", "avg_pnl"], ascending=[False, False]).to_csv(out, index=False)
        print(f"\nWrote full results: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
