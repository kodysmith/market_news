#!/usr/bin/env python3
"""
Replay a scalp bot session from a signals CSV using the current TradeManager logic.

This lets you see how the new realism fixes (slippage, commission, exact TP/SL)
would have affected a past trading day.

Usage:
    python scripts/replay_scalp_session.py [--date 20260310] [--log-dir logs/scalp_bot]
    python scripts/replay_scalp_session.py --signals path/to/signals.csv

Options:
    --date       Date to replay (YYYYMMDD). Default: today.
    --log-dir    Directory containing signal CSVs. Default: logs/scalp_bot
    --signals    Explicit path to a signals CSV file (overrides --date/--log-dir)
    --ticker     Only replay signals for this ticker (if ticker column present)
    --verbose    Print every signal, not just trades
"""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Import the bot's TradeManager and ScalpSignal with new realism fixes
from scripts.spx_scalp_bot import (
    ScalpSignal,
    Trade,
    TradeManager,
    SLIPPAGE_PCT,
    COMMISSION_PER_TRADE,
)

ET = ZoneInfo("America/New_York")


def load_signals(path: Path) -> list[dict]:
    if not path.exists():
        print(f"Error: signals file not found: {path}")
        sys.exit(1)
    with open(path) as f:
        return list(csv.DictReader(f))


def row_to_signal(row: dict) -> ScalpSignal:
    """Convert a signals CSV row back into a ScalpSignal dataclass."""
    # Handle old-format CSVs that don't have a ticker column
    ticker = row.get("ticker", "SPX")

    # Normalize regime: old logs may have POSITIVE_GAMMA etc.
    regime = (row.get("gex_regime", "UNKNOWN") or "UNKNOWN").replace("_GAMMA", "")

    return ScalpSignal(
        timestamp=row["timestamp"],
        ticker=ticker,
        spx_price=float(row["spx_price"]),
        vix=float(row.get("vix", 0)),
        gex_regime=regime,
        flip_line=float(row.get("flip_line", 0)),
        distance_to_flip=float(row.get("distance_to_flip", 0)),
        net_gex=float(row.get("net_gex", 0)),
        call_wall=float(row.get("call_wall", 0)),
        put_wall=float(row.get("put_wall", 0)),
        ema9=float(row.get("ema9", 0)),
        ema21=float(row.get("ema21", 0)),
        rsi14=float(row.get("rsi14", 50)),
        atr14=float(row.get("atr14", 0)),
        price_vs_vwap=float(row.get("price_vs_vwap", 0)),
        signal=row.get("signal", "FLAT"),
        reason=row.get("reason", ""),
        confidence=float(row.get("confidence", 0)),
    )


def replay(signals_path: Path, ticker_filter: str = None, verbose: bool = False):
    rows = load_signals(signals_path)
    if not rows:
        print("No signals to replay.")
        return

    if ticker_filter:
        rows = [r for r in rows if (r.get("ticker", "SPX")).upper() == ticker_filter.upper()]
        if not rows:
            print(f"No signals for ticker '{ticker_filter}'.")
            return

    # Use a temp dir for replay output so we don't overwrite real logs
    replay_dir = ROOT / "logs" / "scalp_bot" / "replay"
    replay_dir.mkdir(parents=True, exist_ok=True)

    # Create the trade manager (writes replay CSVs)
    mgr = TradeManager(replay_dir)
    # Override file paths to avoid collisions
    date_str = rows[0]["timestamp"][:10].replace("-", "")
    suffix = f"_{ticker_filter}" if ticker_filter else ""
    mgr.trade_log_path = replay_dir / f"trades_replay_{date_str}{suffix}.csv"
    mgr.signal_log_path = replay_dir / f"signals_replay_{date_str}{suffix}.csv"
    mgr._init_trade_log()
    mgr._init_signal_log()

    print("=" * 60)
    print(f"REPLAY: {signals_path.name}  ({len(rows)} signals)")
    if ticker_filter:
        print(f"  Ticker filter: {ticker_filter}")
    print(f"  Slippage: {SLIPPAGE_PCT}%  Commission: ${COMMISSION_PER_TRADE}/side")
    print(f"  Replay trades → {mgr.trade_log_path}")
    print("=" * 60)
    print()

    for i, row in enumerate(rows):
        sig = row_to_signal(row)

        if verbose:
            print(
                f"  [{sig.timestamp}] price={sig.spx_price:.2f}  "
                f"regime={sig.gex_regime}  RSI={sig.rsi14:.0f}  "
                f"sig={sig.signal} ({sig.confidence:.2f})"
            )

        # Process through the trade manager (applies slippage, commission, exact TP/SL)
        mgr.process_tick(sig)

    # Force-close any open trade at end of replay
    if mgr.open_trade:
        last_sig = row_to_signal(rows[-1])
        is_long = mgr.open_trade.direction == "LONG"
        # Apply exit slippage
        if is_long:
            exit_px = last_sig.spx_price * (1 - SLIPPAGE_PCT / 100)
        else:
            exit_px = last_sig.spx_price * (1 + SLIPPAGE_PCT / 100)
        exit_px = round(exit_px, 2)

        gross_pnl = (exit_px - mgr.open_trade.entry_price) if is_long else (mgr.open_trade.entry_price - exit_px)
        slippage_entry = abs(mgr.open_trade.entry_price * SLIPPAGE_PCT / 100)
        slippage_exit = abs(exit_px * SLIPPAGE_PCT / 100)
        total_commission = 2 * COMMISSION_PER_TRADE
        net_pnl = gross_pnl - total_commission

        entry_dt = datetime.strptime(mgr.open_trade.entry_time, "%Y-%m-%d %H:%M:%S")
        exit_dt = datetime.strptime(last_sig.timestamp, "%Y-%m-%d %H:%M:%S")

        mgr.open_trade.exit_time = last_sig.timestamp
        mgr.open_trade.exit_price = exit_px
        mgr.open_trade.exit_reason = "END_OF_REPLAY"
        mgr.open_trade.gross_pnl_points = round(gross_pnl, 2)
        mgr.open_trade.slippage_cost = round(slippage_entry + slippage_exit, 4)
        mgr.open_trade.commission_cost = round(total_commission, 4)
        mgr.open_trade.pnl_points = round(net_pnl, 2)
        mgr.open_trade.duration_sec = int((exit_dt - entry_dt).total_seconds())
        mgr._log_trade(mgr.open_trade)
        mgr.stats["total"] += 1
        mgr.stats["total_pnl"] += net_pnl
        if net_pnl > 0:
            mgr.stats["wins"] += 1
        else:
            mgr.stats["losses"] += 1
        mgr.open_trade = None
        print(f"  [END] Force-closed open trade @ {exit_px:.2f} | Net PnL: {net_pnl:+.2f}")

    # ── Summary ──
    print()
    print("=" * 60)
    print("REPLAY RESULTS")
    print("=" * 60)
    s = mgr.stats
    print(f"  Total trades:     {s['total']}")
    print(f"  Wins / Losses:    {s['wins']} / {s['losses']}")
    if s["total"]:
        print(f"  Win rate:         {s['wins'] / s['total'] * 100:.1f}%")
    print(f"  Net PnL:          {s['total_pnl']:+.2f} pts")

    # Load replay trades for detailed breakdown
    try:
        with open(mgr.trade_log_path) as f:
            trades = list(csv.DictReader(f))
    except Exception:
        trades = []

    if trades:
        gross_total = sum(float(t.get("gross_pnl_points", t["pnl_points"])) for t in trades)
        slip_total = sum(float(t.get("slippage_cost", 0)) for t in trades)
        comm_total = sum(float(t.get("commission_cost", 0)) for t in trades)
        pnls = [float(t["pnl_points"]) for t in trades]
        wins = [p for p in pnls if p > 0]
        losses_list = [p for p in pnls if p <= 0]

        print(f"  Gross PnL:        {gross_total:+.2f} pts")
        print(f"  Total slippage:   {slip_total:.4f} pts")
        print(f"  Total commission: {comm_total:.2f} pts")
        print(f"  Cost drag:        {gross_total - sum(pnls):+.2f} pts")
        print()

        if wins:
            print(f"  Avg win:          {sum(wins)/len(wins):+.2f} pts")
        if losses_list:
            print(f"  Avg loss:         {sum(losses_list)/len(losses_list):+.2f} pts")

        # Max drawdown
        cum = 0.0
        peak = 0.0
        max_dd = 0.0
        for p in pnls:
            cum += p
            peak = max(peak, cum)
            max_dd = min(max_dd, cum - peak)
        print(f"  Max drawdown:     {max_dd:.2f} pts")

        durations = [int(t["duration_sec"]) for t in trades]
        print(f"  Avg hold time:    {sum(durations)/len(durations)/60:.1f} min")
        print()

        # Per-trade detail
        print("TRADE DETAIL:")
        print(f"  {'#':>3}  {'Time':>8}  {'Dir':>5}  {'Entry':>9}  {'Exit':>9}  "
              f"{'Gross':>7}  {'Net':>7}  {'Slip':>6}  {'Comm':>5}  {'Reason'}")
        print("  " + "-" * 90)
        for i, t in enumerate(trades, 1):
            entry_t = t["entry_time"].split(" ")[1] if " " in t["entry_time"] else t["entry_time"]
            gross = float(t.get("gross_pnl_points", t["pnl_points"]))
            net = float(t["pnl_points"])
            slip = float(t.get("slippage_cost", 0))
            comm = float(t.get("commission_cost", 0))
            print(
                f"  {i:>3}  {entry_t:>8}  {t['direction']:>5}  "
                f"${float(t['entry_price']):>8.2f}  ${float(t['exit_price']):>8.2f}  "
                f"{gross:>+7.2f}  {net:>+7.2f}  {slip:>6.3f}  {comm:>5.2f}  "
                f"{t['exit_reason']}"
            )

        # Compare with original if available
        print()
        print("=" * 60)

    print(f"\nReplay CSV saved to: {mgr.trade_log_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replay scalp bot session with realism fixes")
    parser.add_argument("--date", type=str, default=None, help="Date YYYYMMDD (default: today)")
    parser.add_argument("--log-dir", type=str, default="logs/scalp_bot")
    parser.add_argument("--signals", type=str, default=None, help="Explicit path to signals CSV")
    parser.add_argument("--ticker", type=str, default=None, help="Filter by ticker")
    parser.add_argument("--verbose", action="store_true", help="Print every signal")
    args = parser.parse_args()

    if args.signals:
        sig_path = Path(args.signals)
    else:
        log_dir = ROOT / args.log_dir
        date_str = args.date or datetime.now(ET).strftime("%Y%m%d")
        sig_path = log_dir / f"signals_{date_str}.csv"

    replay(sig_path, ticker_filter=args.ticker, verbose=args.verbose)
