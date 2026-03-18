#!/usr/bin/env python3
"""
Post-processing analyzer for SPX Scalp Bot logs.

Reads trade and signal CSVs and produces performance metrics.

Usage:
    python scripts/analyze_scalp_results.py [--date 20260310] [--log-dir logs/scalp_bot]
"""

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_trades(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def load_signals(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def analyze(trades: list[dict], signals: list[dict], ticker_filter: str = None):
    if ticker_filter:
        trades = [t for t in trades if t.get("ticker", "").upper() == ticker_filter.upper()]
        signals = [s for s in signals if s.get("ticker", "").upper() == ticker_filter.upper()]

    if not trades:
        print("No trades to analyze.")
        return

    pnls = [float(t["pnl_points"]) for t in trades]
    gross_pnls = [float(t.get("gross_pnl_points", t["pnl_points"])) for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    durations = [int(t["duration_sec"]) for t in trades]

    total_pnl = sum(pnls)
    total_gross = sum(gross_pnls)
    total_slippage = sum(float(t.get("slippage_cost", 0)) for t in trades)
    total_commission = sum(float(t.get("commission_cost", 0)) for t in trades)
    avg_win = sum(wins) / len(wins) if wins else 0
    avg_loss = sum(losses) / len(losses) if losses else 0
    win_rate = len(wins) / len(pnls) * 100
    profit_factor = abs(sum(wins) / sum(losses)) if losses and sum(losses) != 0 else float("inf")

    # Max drawdown (cumulative)
    cum = 0.0
    peak = 0.0
    max_dd = 0.0
    for p in pnls:
        cum += p
        peak = max(peak, cum)
        max_dd = min(max_dd, cum - peak)

    title = f"SCALP BOT — PERFORMANCE REPORT"
    if ticker_filter:
        title += f" ({ticker_filter})"
    print("=" * 55)
    print(title)
    print("=" * 55)
    print(f"  Total trades:     {len(pnls)}")
    print(f"  Wins / Losses:    {len(wins)} / {len(losses)}")
    print(f"  Win rate:         {win_rate:.1f}%")
    print(f"  Gross PnL:        {total_gross:+.2f} pts")
    print(f"  Net PnL:          {total_pnl:+.2f} pts")
    print(f"  Total slippage:   {total_slippage:.4f} pts")
    print(f"  Total commission: {total_commission:.2f} pts")
    print(f"  Avg win:          {avg_win:+.2f} pts")
    print(f"  Avg loss:         {avg_loss:+.2f} pts")
    print(f"  Profit factor:    {profit_factor:.2f}")
    print(f"  Max drawdown:     {max_dd:.2f} pts")
    print(f"  Avg hold time:    {sum(durations)/len(durations)/60:.1f} min")
    print()

    # Breakdown by regime
    print("BY GEX REGIME:")
    for regime in ("POSITIVE", "NEGATIVE"):
        rt = [t for t in trades if t.get("gex_regime") == regime]
        if rt:
            rp = [float(t["pnl_points"]) for t in rt]
            rw = [p for p in rp if p > 0]
            print(f"  {regime}: {len(rt)} trades, {len(rw)}/{len(rt)} wins, PnL={sum(rp):+.2f}")
    print()

    # Breakdown by direction
    print("BY DIRECTION:")
    for d in ("LONG", "SHORT"):
        dt = [t for t in trades if t["direction"] == d]
        if dt:
            dp = [float(t["pnl_points"]) for t in dt]
            dw = [p for p in dp if p > 0]
            print(f"  {d}: {len(dt)} trades, {len(dw)}/{len(dt)} wins, PnL={sum(dp):+.2f}")
    print()

    # Breakdown by exit reason
    print("BY EXIT REASON:")
    reasons = set(t["exit_reason"] for t in trades)
    for r in sorted(reasons):
        et = [t for t in trades if t["exit_reason"] == r]
        ep = [float(t["pnl_points"]) for t in et]
        print(f"  {r}: {len(et)} trades, PnL={sum(ep):+.2f}")
    print()

    # Signal distribution
    if signals:
        sig_counts = {}
        for s in signals:
            sig_counts[s["signal"]] = sig_counts.get(s["signal"], 0) + 1
        print("SIGNAL DISTRIBUTION:")
        for s, c in sorted(sig_counts.items()):
            print(f"  {s}: {c} ({c/len(signals)*100:.0f}%)")
    print("=" * 55)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, default=None, help="Date YYYYMMDD (default: today)")
    parser.add_argument("--log-dir", type=str, default="logs/scalp_bot")
    parser.add_argument("--ticker", type=str, default=None, help="Filter by ticker (e.g. TSLA, SPX)")
    args = parser.parse_args()

    log_dir = ROOT / args.log_dir
    date_str = args.date or datetime.now().strftime("%Y%m%d")

    trades = load_trades(log_dir / f"trades_{date_str}.csv")
    signals = load_signals(log_dir / f"signals_{date_str}.csv")
    analyze(trades, signals, ticker_filter=args.ticker)
