#!/usr/bin/env python3
"""
Scan for the best SPX credit spread / iron condor setups to open today.

This script reuses the enhanced cashflow strategy configs and their
historical performance, then:

- Loads recent SPX + VIX data.
- Recomputes the same entry filters used in the backtest for a given date.
- For configs that pass filters, computes suggested strikes and credits.
- Ranks candidate setups by historical Sharpe (and then by total P&L).

Usage:
  python scripts/run_spx_credit_scanner.py
  python scripts/run_spx_credit_scanner.py --as-of 2025-12-12 --top-n 15
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def _add_root_to_syspath() -> None:
    import sys

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))


_add_root_to_syspath()

from scripts.optimize_spx_verticals_historical import (  # noqa: E402
    load_spx_vix,
    friday_expirations,
    get_trading_days,
    bs_put_price,
    bs_call_price,
    strike_for_delta,
)
from scripts.evaluate_enhanced_cashflow import (  # noqa: E402
    EntryFilters,
    ExitRules,
    StrategyConfig,
    adaptive_delta_for_vix,
    build_strategy_configs,
    compute_enhanced_metrics,
)


ES_MULT = 50.0


@dataclass
class ScannerCandidate:
    config_id: str
    strategy: str
    dte: int
    short_delta: float
    width: float
    entry_date: date
    spot: float
    vix: float
    credit: float
    max_loss: float
    put_short: float
    put_long: float
    call_short: float | None
    call_long: float | None
    hist_sharpe: float
    hist_total_pnl: float

    @property
    def rr_ratio(self) -> float:
        if self.max_loss <= 0:
            return 0.0
        return float(self.credit * 100.0 / (self.max_loss * 100.0))

    @property
    def credit_per_width(self) -> float:
        if self.width <= 0:
            return 0.0
        return float(self.credit / self.width)


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Scan for best SPX credit spread setups to open.")
    ap.add_argument("--as-of", type=str, default=None, help="As-of date (YYYY-MM-DD). Default: today (America/New_York).")
    ap.add_argument("--top-n", type=int, default=15, help="Number of best historical configs to consider.")
    ap.add_argument("--min-sharpe", type=float, default=1.5, help="Minimum historical Sharpe for configs to consider.")
    ap.add_argument(
        "--only-configs",
        type=str,
        default=None,
        help="Comma-separated list of config_ids to consider (overrides top-n/min-sharpe).",
    )
    return ap.parse_args()


def _resolve_as_of(as_of_str: str | None) -> date:
    if as_of_str:
        return datetime.strptime(as_of_str, "%Y-%m-%d").date()
    try:
        from zoneinfo import ZoneInfo

        tz_et = ZoneInfo("America/New_York")
        now = datetime.now(tz=tz_et)
    except Exception:
        now = datetime.utcnow()
    return now.date()


def _load_hist_summary() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, Any]]]:
    trades_path = ROOT / "data" / "enhanced_cashflow_trades.csv"
    if not trades_path.is_file():
        raise SystemExit(f"Trades CSV not found at {trades_path}. Run evaluate_enhanced_cashflow.py first.")
    trades = pd.read_csv(trades_path)
    from scripts.evaluate_enhanced_cashflow import compute_enhanced_metrics as _cem  # noqa: E402

    monthly, equity_df, summary = _cem(trades)
    return monthly, equity_df, summary


def _select_configs(
    configs: list[StrategyConfig],
    summary: dict[str, dict[str, Any]],
    top_n: int,
    min_sharpe: float,
    only_configs: list[str] | None = None,
) -> list[StrategyConfig]:
    cfg_map = {c.config_id: c for c in configs}

    if only_configs:
        return [cfg_map[cid] for cid in only_configs if cid in cfg_map]

    ranked = sorted(
        summary.items(),
        key=lambda kv: (kv[1].get("ann_sharpe", 0.0), kv[1].get("total_pnl", 0.0)),
        reverse=True,
    )
    selected_ids: list[str] = []
    for cid, m in ranked:
        if cid not in cfg_map:
            continue
        if m.get("ann_sharpe", 0.0) < min_sharpe:
            continue
        selected_ids.append(cid)
        if len(selected_ids) >= top_n:
            break
    return [cfg_map[cid] for cid in selected_ids]


def _compute_entry_filters_context(
    spx_df: pd.DataFrame,
    vix_close: pd.Series,
    start: date,
    end: date,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    spx_close = spx_df["Close"].copy()
    spx_close.index = spx_close.index.date
    spx_open = spx_df["Open"].copy()
    spx_open.index = spx_open.index.date

    prev_close = spx_close.shift(1)
    gap_pct = (spx_open - prev_close) / prev_close * 100.0

    log_ret = np.log(spx_close / spx_close.shift(1))
    rv_5d = log_ret.rolling(5).std() * np.sqrt(252) * 100
    rv_10d = log_ret.rolling(10).std() * np.sqrt(252) * 100
    rv_ratio = rv_5d / rv_10d

    vix_series = vix_close.copy()
    if not vix_series.empty:
        vix_series.index = vix_series.index.date
    return spx_close, rv_ratio, gap_pct, vix_series


def _satisfies_entry_filters(
    cfg: StrategyConfig,
    entry_date: date,
    spx_close: pd.Series,
    vix_close: pd.Series,
    rv_ratio: pd.Series | None,
    gap_pct: pd.Series | None,
) -> bool:
    f: EntryFilters = cfg.filters
    vix_val = float(vix_close.get(entry_date, 20.0))

    if f.vix_max is not None and vix_val > f.vix_max:
        return False
    if f.vix_min is not None and vix_val < f.vix_min:
        return False

    if f.trend_sma_period is not None:
        sma = spx_close.rolling(f.trend_sma_period).mean()
        sma.index = spx_close.index
        if entry_date in sma.index:
            ma_val = sma.get(entry_date)
            if ma_val is not None and not math.isnan(ma_val):
                if float(spx_close[entry_date]) < float(ma_val):
                    return False

    if f.rv_expansion_max is not None and rv_ratio is not None:
        rv_val = float(rv_ratio.get(entry_date, 1.0))
        if not math.isnan(rv_val) and rv_val > f.rv_expansion_max:
            return False

    if f.gap_down_no_entry is not None and gap_pct is not None:
        gap_val = float(gap_pct.get(entry_date, 0.0))
        if not math.isnan(gap_val) and gap_val < f.gap_down_no_entry:
            return False

    return True


def _build_candidate_for_config(
    cfg: StrategyConfig,
    entry_date: date,
    spx_close: pd.Series,
    vix_close: pd.Series,
    hist_metrics: dict[str, Any],
) -> ScannerCandidate | None:
    if entry_date not in spx_close.index:
        return None
    S = float(spx_close[entry_date])
    if S <= 0:
        return None

    vix_val = float(vix_close.get(entry_date, 20.0))
    sigma = vix_val / 100.0 if vix_val > 3 else vix_val

    delta = cfg.short_delta
    if cfg.adaptive_delta:
        delta = adaptive_delta_for_vix(vix_val, cfg.short_delta)

    expirations = friday_expirations(entry_date, entry_date + timedelta(days=cfg.dte + 10))
    trading_days = get_trading_days(entry_date - timedelta(days=5), entry_date + timedelta(days=cfg.dte + 10))
    trading_days = [d for d in trading_days if d >= entry_date]
    if not expirations or not trading_days:
        return None

    target_exp = None
    for e in expirations:
        if e <= entry_date:
            continue
        dte = (e - entry_date).days
        if dte >= cfg.dte - 3 and dte <= cfg.dte + 3:
            target_exp = e
            break
    if target_exp is None:
        target_exp = expirations[0]

    entry_idx = trading_days.index(entry_date)
    exp_idx = trading_days.index(target_exp) if target_exp in trading_days else None
    if exp_idx is None or exp_idx <= entry_idx:
        return None
    dte_trading = exp_idx - entry_idx
    T = dte_trading / 252.0
    if T <= 0 or sigma <= 0:
        return None

    K_put_short = strike_for_delta(S, T, delta, sigma, is_put=True)
    if K_put_short is None:
        return None
    K_put_long = K_put_short - cfg.width
    if K_put_long <= 0:
        return None

    put_short_px = bs_put_price(S, K_put_short, T, sigma)
    put_long_px = bs_put_price(S, K_put_long, T, sigma)
    put_credit = put_short_px - put_long_px

    call_credit = 0.0
    K_call_short = K_call_long = None
    if cfg.strategy == "iron_condor":
        K_call_short = strike_for_delta(S, T, delta, sigma, is_put=False)
        if K_call_short is None:
            return None
        K_call_long = K_call_short + cfg.width
        call_short_px = bs_call_price(S, K_call_short, T, sigma)
        call_long_px = bs_call_price(S, K_call_long, T, sigma)
        call_credit = call_short_px - call_long_px

    total_credit = put_credit + call_credit
    if total_credit <= 0:
        return None

    max_loss_put = cfg.width - put_credit
    if cfg.strategy == "iron_condor":
        max_loss = max(cfg.width - put_credit, cfg.width - call_credit)
    else:
        max_loss = max_loss_put
    if max_loss <= 0:
        return None

    hist_sharpe = float(hist_metrics.get("ann_sharpe", 0.0))
    hist_total_pnl = float(hist_metrics.get("total_pnl", 0.0))

    return ScannerCandidate(
        config_id=cfg.config_id,
        strategy=cfg.strategy,
        dte=cfg.dte,
        short_delta=delta,
        width=cfg.width,
        entry_date=entry_date,
        spot=S,
        vix=vix_val,
        credit=float(total_credit),
        max_loss=float(max_loss),
        put_short=float(K_put_short),
        put_long=float(K_put_long),
        call_short=float(K_call_short) if K_call_short is not None else None,
        call_long=float(K_call_long) if K_call_long is not None else None,
        hist_sharpe=hist_sharpe,
        hist_total_pnl=hist_total_pnl,
    )


def run_scanner(as_of: date, top_n: int, min_sharpe: float, only_configs: list[str] | None = None) -> int:
    start = as_of - timedelta(days=90)
    end = as_of

    spx_df, vix_df = load_spx_vix(start, end)
    if spx_df.empty:
        raise SystemExit("No SPX data from yfinance.")

    vix_close = vix_df["Close"].copy() if not vix_df.empty else pd.Series(dtype=float)
    spx_close, rv_ratio, gap_pct, vix_series = _compute_entry_filters_context(spx_df, vix_close, start, end)

    monthly, equity_df, summary = _load_hist_summary()
    _ = monthly, equity_df

    configs = build_strategy_configs()
    selected_cfgs = _select_configs(configs, summary, top_n=top_n, min_sharpe=min_sharpe, only_configs=only_configs)

    candidates: list[ScannerCandidate] = []
    for cfg in selected_cfgs:
        if not _satisfies_entry_filters(cfg, as_of, spx_close, vix_series, rv_ratio, gap_pct):
            continue
        hist_metrics = summary.get(cfg.config_id, {})
        cand = _build_candidate_for_config(cfg, as_of, spx_close, vix_series, hist_metrics)
        if cand is not None:
            candidates.append(cand)

    if not candidates:
        print(f"No configs passed filters for {as_of}.")
        return 0

    candidates.sort(key=lambda c: (c.hist_sharpe, c.hist_total_pnl, c.credit_per_width), reverse=True)

    print(f"\nBest SPX credit setups for {as_of} (top {len(candidates)} passing filters):\n")
    header = (
        f"{'Config':<24s} {'Strat':<6s} {'Sharpe':>6s} {'TotPnL':>9s} "
        f"{'Spot':>8s} {'VIX':>6s} {'DTE':>4s} {'Δ':>4s} {'Width':>5s} "
        f"{'Credit':>8s} {'MaxLoss':>8s} {'C/W':>5s} {'Pshort':>8s} {'Plong':>8s} "
        f"{'Cshort':>8s} {'Clong':>8s}"
    )
    print(header)
    print("-" * len(header))
    for c in candidates:
        print(
            f"{c.config_id:<24s} {c.strategy:<6s} "
            f"{c.hist_sharpe:6.2f} {c.hist_total_pnl/1000:8.1f}k "
            f"{c.spot:8.1f} {c.vix:6.1f} {c.dte:4d} {c.short_delta:4.2f} {c.width:5.0f} "
            f"{c.credit:8.2f} {c.max_loss:8.2f} {c.credit_per_width:5.2f} "
            f"{c.put_short:8.1f} {c.put_long:8.1f} "
            f"{(c.call_short or 0):8.1f} {(c.call_long or 0):8.1f}"
        )
    print()
    return 0


def main() -> int:
    args = _parse_args()
    as_of = _resolve_as_of(args.as_of)
    only_cfgs = args.only_configs.split(",") if args.only_configs else None
    return run_scanner(as_of=as_of, top_n=args.top_n, min_sharpe=args.min_sharpe, only_configs=only_cfgs)


if __name__ == "__main__":
    raise SystemExit(main())

