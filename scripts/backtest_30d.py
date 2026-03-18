#!/usr/bin/env python3
"""
Multi-period backtest of all three strategies (Directional, Iron Condor, Gamma Scalp).

Uses SPY bars from yfinance as XSP proxy, VIX for regime classification.
- <=60 days: uses 5-min bars (~78 bars/day)
- >60 days: uses 1-hour bars (~7 bars/day)

GEX regime approximated from realized vol vs VIX:
  - Low realized vol (< 0.7x implied) → POSITIVE
  - High realized vol (> 0.95x implied) → NEGATIVE
  - Otherwise → UNKNOWN

Usage:
    python scripts/backtest_30d.py              # 30 days
    python scripts/backtest_30d.py --days 365   # 1 year
"""
from __future__ import annotations

import argparse
import csv
import logging
import sys
from datetime import datetime, time as dtime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.spx_scalp_bot import (
    ScalpSignal, TradeManager, IronCondorManager, GammaScalpManager,
    StrategyRouter, generate_signal, compute_atr, compute_ema, compute_rsi,
    compute_vwap,
)

ET = ZoneInfo("America/New_York")
log = logging.getLogger("backtest")
logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")


def download_data(days: int = 30) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Download SPY and VIX bars. Returns (spy_df, vix_df, interval)."""
    if days <= 60:
        interval = "5m"
        bars_per_day = 78
    else:
        interval = "1h"
        bars_per_day = 7

    print(f"Downloading {days}d of SPY {interval} bars...")
    spy = yf.download("SPY", period=f"{days}d", interval=interval, progress=False)
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.get_level_values(0)
    n_days = spy.index.normalize().nunique() if not spy.empty else 0
    print(f"  SPY: {len(spy)} bars, {n_days} trading days ({interval})")

    print(f"Downloading {days}d of VIX {interval} bars...")
    vix = yf.download("^VIX", period=f"{days}d", interval=interval, progress=False)
    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = vix.columns.get_level_values(0)
    print(f"  VIX: {len(vix)} bars")

    return spy, vix, interval


def classify_gex_regime(bars_window: pd.DataFrame, vix_level: float,
                        bars_per_day: int = 78) -> str:
    """Approximate GEX regime from realized vol vs implied vol."""
    if len(bars_window) < 5 or vix_level <= 0:
        return "UNKNOWN"

    returns = bars_window["Close"].pct_change().dropna()
    if len(returns) < 3:
        return "UNKNOWN"

    # Annualize based on bar frequency
    realized_vol = float(returns.std()) * np.sqrt(bars_per_day * 252) * 100

    ratio = realized_vol / vix_level if vix_level > 0 else 1.0

    if ratio < 0.70:
        return "POSITIVE"
    elif ratio > 0.95:
        return "NEGATIVE"
    return "UNKNOWN"


def run_backtest(days: int = 30):
    spy, vix_df, interval = download_data(days)

    if spy.empty:
        print("No SPY data available")
        return

    bars_per_day = 78 if interval == "5m" else 7
    min_bars = 20 if interval == "5m" else 3  # min bars needed for indicators
    window_size = 30 if interval == "5m" else 7  # regime classification window

    dates = sorted(spy.index.normalize().unique())
    print(f"\nBacktesting {len(dates)} trading days: "
          f"{dates[0].strftime('%Y-%m-%d')} → {dates[-1].strftime('%Y-%m-%d')}")
    print(f"Bar interval: {interval}  (~{bars_per_day} bars/day)")

    total_stats = {
        "directional": {"total": 0, "wins": 0, "losses": 0, "total_pnl": 0.0},
        "iron_condor": {"total": 0, "wins": 0, "losses": 0, "total_pnl": 0.0},
        "gamma_scalp": {"total": 0, "wins": 0, "losses": 0, "total_pnl": 0.0,
                        "rebalances": 0},
    }
    daily_results = []
    monthly_pnl = {}

    for day_idx, day in enumerate(dates):
        day_str = day.strftime("%Y-%m-%d")
        month_key = day.strftime("%Y-%m")

        day_bars = spy[spy.index.normalize() == day].copy()
        if day_bars.empty:
            continue

        if day_bars.index.tz is not None:
            day_bars.index = day_bars.index.tz_convert(ET)

        day_bars = day_bars.between_time("09:30", "16:00")
        if len(day_bars) < min_bars:
            continue

        vix_day = vix_df[vix_df.index.normalize() == day]
        if vix_day.index.tz is not None:
            vix_day = vix_day.copy()
            vix_day.index = vix_day.index.tz_convert(ET)

        xsp_bars = day_bars.copy()

        log_dir = ROOT / "logs" / "scalp_bot" / "backtest" / day_str
        log_dir.mkdir(parents=True, exist_ok=True)

        profile = {
            "ticker": "XSP", "mode": "0dte", "min_confidence": 0.60,
            "multiplier": 100, "contracts": 1, "max_premium": 300,
            "min_premium": 0.20, "option_tp_pct": 50, "option_sl_pct": 35,
            "option_trail_pct": 25, "max_hold_minutes": 30,
            "min_tte_minutes": 30, "commission_per_side": 0.65,
            "cooldown_sec": 300, "max_trades_per_day": 6,
            "scale_on_trail": False,
        }

        trade_mgr = TradeManager(log_dir, profile=profile, ibkr=None)
        ic_mgr = IronCondorManager(log_dir, ibkr=None, profile=profile)
        gs_mgr = GammaScalpManager(log_dir, ibkr=None, profile=profile)
        router = StrategyRouter(trade_mgr, ic_mgr, gs_mgr)

        last_sig = None
        regime_counts = {"POSITIVE": 0, "NEGATIVE": 0, "UNKNOWN": 0}

        start_idx = min(min_bars, len(xsp_bars) - 1)
        for i in range(start_idx, len(xsp_bars)):
            bar_time = xsp_bars.index[i]
            ts_str = bar_time.strftime("%Y-%m-%d %H:%M:%S")
            spot = float(xsp_bars["Close"].iloc[i])

            vix_level = 0.0
            if not vix_day.empty:
                vix_idx = vix_day.index.get_indexer([bar_time], method="nearest")[0]
                if 0 <= vix_idx < len(vix_day):
                    vix_level = float(vix_day["Close"].iloc[vix_idx])

            window = xsp_bars.iloc[max(0, i - window_size):i + 1]
            regime = classify_gex_regime(window, vix_level, bars_per_day)
            regime_counts[regime] += 1

            gex = {"regime": regime, "flip": 0, "net_gex": 0}
            bars_for_signal = xsp_bars.iloc[:i + 1]
            sig = generate_signal(
                bars_for_signal, gex, vix_level, spot,
                ticker="XSP", timestamp=ts_str,
            )

            sig = ScalpSignal(
                timestamp=sig.timestamp, ticker=sig.ticker,
                spx_price=sig.spx_price, vix=sig.vix,
                gex_regime=regime,
                flip_line=sig.flip_line, distance_to_flip=sig.distance_to_flip,
                net_gex=sig.net_gex, call_wall=sig.call_wall, put_wall=sig.put_wall,
                ema9=sig.ema9, ema21=sig.ema21, rsi14=sig.rsi14, atr14=sig.atr14,
                price_vs_vwap=sig.price_vs_vwap, signal=sig.signal,
                reason=sig.reason, confidence=sig.confidence,
            )

            router.process_tick(sig)
            last_sig = sig

        if last_sig:
            router.force_close_all(last_sig)
            if trade_mgr.open_trade:
                trade_mgr.MAX_HOLD_MINUTES = 0
                if trade_mgr.mode == "0dte":
                    trade_mgr._check_exit_option(last_sig)
                else:
                    trade_mgr._check_exit(last_sig)

        day_pnl = (trade_mgr.stats["total_pnl"] + ic_mgr.stats["total_pnl"]
                    + gs_mgr.stats["total_pnl"])
        day_range = float(xsp_bars["High"].max() - xsp_bars["Low"].min())

        daily_results.append({
            "date": day_str,
            "xsp_open": float(xsp_bars["Open"].iloc[0]),
            "xsp_close": float(xsp_bars["Close"].iloc[-1]),
            "range": day_range,
            "vix": vix_level,
            "regimes": dict(regime_counts),
            "dir_trades": trade_mgr.stats["total"],
            "dir_pnl": trade_mgr.stats["total_pnl"],
            "ic_trades": ic_mgr.stats["total"],
            "ic_pnl": ic_mgr.stats["total_pnl"],
            "gs_trades": gs_mgr.stats["total"],
            "gs_pnl": gs_mgr.stats["total_pnl"],
            "gs_rebalances": gs_mgr.stats.get("rebalances", 0),
            "combined_pnl": day_pnl,
        })

        for key, mgr_stats in [("directional", trade_mgr.stats),
                                ("iron_condor", ic_mgr.stats),
                                ("gamma_scalp", gs_mgr.stats)]:
            for k in ["total", "wins", "losses"]:
                total_stats[key][k] += mgr_stats[k]
            total_stats[key]["total_pnl"] += mgr_stats["total_pnl"]
            if "rebalances" in mgr_stats:
                total_stats["gamma_scalp"]["rebalances"] += mgr_stats.get("rebalances", 0)

        monthly_pnl[month_key] = monthly_pnl.get(month_key, 0.0) + day_pnl

        # Print daily line (compact for 1-year)
        regime_primary = max(regime_counts, key=regime_counts.get)
        strats = []
        if trade_mgr.stats["total"]:
            strats.append(f"D:{trade_mgr.stats['total']}t ${trade_mgr.stats['total_pnl']:+.0f}")
        if ic_mgr.stats["total"]:
            strats.append(f"IC:{ic_mgr.stats['total']}t ${ic_mgr.stats['total_pnl']:+.0f}")
        if gs_mgr.stats["total"]:
            strats.append(f"GS:{gs_mgr.stats['total']}t ${gs_mgr.stats['total_pnl']:+.0f}")
        strat_str = "  ".join(strats) if strats else "---"

        print(f"  {day_str}  XSP={float(xsp_bars['Close'].iloc[-1]):7.2f}  "
              f"VIX={vix_level:5.1f}  {regime_primary[:3]:3s}  {strat_str:40s}  "
              f"${day_pnl:+8.2f}")

    # ─── Final Summary ─────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  {'1-YEAR' if days > 60 else f'{days}-DAY'} BACKTEST SUMMARY  "
          f"({interval} bars)")
    print(f"{'='*70}")
    print(f"  Days: {len(daily_results)}  "
          f"({dates[0].strftime('%Y-%m-%d')} → {dates[-1].strftime('%Y-%m-%d')})")

    combined_pnl = sum(d["combined_pnl"] for d in daily_results)
    winning_days = sum(1 for d in daily_results if d["combined_pnl"] > 0)
    losing_days = sum(1 for d in daily_results if d["combined_pnl"] < 0)
    flat_days = sum(1 for d in daily_results if d["combined_pnl"] == 0)

    print(f"  Winning days: {winning_days}  |  Losing: {losing_days}  |  "
          f"Flat: {flat_days}  ({winning_days/len(daily_results)*100:.0f}% win rate)")

    pnls = [d["combined_pnl"] for d in daily_results]
    print(f"  Best day: ${max(pnls):+,.2f}  |  Worst: ${min(pnls):+,.2f}  |  "
          f"Avg: ${np.mean(pnls):+,.2f}/day")

    # Monthly breakdown
    if len(monthly_pnl) > 1:
        print(f"\n  Monthly PnL:")
        for month, pnl in sorted(monthly_pnl.items()):
            bar = "█" * max(1, int(abs(pnl) / 20))
            sign = "+" if pnl >= 0 else "-"
            print(f"    {month}:  ${pnl:+8.2f}  {'▓' if pnl >= 0 else '░'}{bar}")

    for name, stats in total_stats.items():
        label = name.upper().replace("_", " ")
        if stats["total"] == 0:
            print(f"\n  {label}: 0 trades")
            continue
        wr = stats["wins"] / stats["total"] * 100
        extra = ""
        if "rebalances" in stats and stats["rebalances"]:
            extra = f"  Rebalances: {stats['rebalances']}"
        print(f"\n  {label}: {stats['total']} trades, "
              f"{stats['wins']}W/{stats['losses']}L ({wr:.0f}% WR){extra}")
        print(f"    PnL: ${stats['total_pnl']:+,.2f}  "
              f"(avg ${stats['total_pnl']/stats['total']:+,.2f}/trade)")

    print(f"\n  COMBINED PnL: ${combined_pnl:+,.2f}")

    if len(pnls) > 1:
        sharpe_daily = np.mean(pnls) / np.std(pnls) * np.sqrt(252) if np.std(pnls) > 0 else 0
        cumulative = np.cumsum(pnls)
        peak = np.maximum.accumulate(cumulative)
        drawdown = cumulative - peak
        print(f"  Sharpe (daily, annualized): {sharpe_daily:.2f}")
        print(f"  Max Drawdown: ${min(drawdown):+,.2f}")
        print(f"  Profit Factor: {sum(p for p in pnls if p > 0) / abs(sum(p for p in pnls if p < 0)):.2f}"
              if sum(p for p in pnls if p < 0) != 0 else "  Profit Factor: ∞")

    print(f"{'='*70}")

    # Save results
    suffix = f"_{days}d" if days != 30 else "_30d"
    results_path = ROOT / "logs" / "scalp_bot" / f"backtest{suffix}_results.csv"
    with open(results_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=daily_results[0].keys())
        w.writeheader()
        for row in daily_results:
            row_copy = dict(row)
            row_copy["regimes"] = str(row_copy["regimes"])
            w.writerow(row_copy)
    print(f"\n  Results saved to: {results_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Strategy backtest")
    parser.add_argument("--days", type=int, default=30)
    args = parser.parse_args()
    run_backtest(args.days)
