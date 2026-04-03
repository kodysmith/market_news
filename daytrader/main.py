#!/usr/bin/env python3
"""Day Trading Bot — Main Orchestrator.

Usage:
  python -m daytrader.main --mode dry-run    # log signals, no orders
  python -m daytrader.main --mode paper      # paper trade on IBKR paper account
  python -m daytrader.main --mode backtest --date 2026-03-15  # backtest a single day

Architecture:
  7:00 AM  — Universe scanner + catalyst classifier → watchlist
  9:30 AM  — Entry engine loop (poll 1-min bars every 10s)
  9:30-11:30 — Position manager processes bars, manages exits
  11:30 AM — Force close all positions. Day done.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, date, timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from daytrader.config import (
    MODE, ACCOUNT_SIZE, DT_DATA_DIR, DT_LOG_DIR, LOG_LEVEL,
    SCAN_WATCHLIST_SIZE, ENTRY_MAX_CONCURRENT, EXIT_CLOSE_ALL_TIME,
)
from daytrader.scanner.universe_scanner import scan_for_candidates, _get_api_key
from daytrader.scanner.catalyst_classifier import enrich_candidates
from daytrader.engine.bar_manager import BarManager
from daytrader.engine.entry_engine import scan_for_entries
from daytrader.engine.position_manager import PositionManager
from daytrader.engine.risk_governor import RiskGovernor

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s %(name)-24s %(levelname)-5s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("daytrader.main")


def get_prev_trading_day(d: date) -> date:
    """Simple previous trading day (skip weekends)."""
    prev = d - timedelta(days=1)
    while prev.weekday() >= 5:
        prev -= timedelta(days=1)
    return prev


def run_premarket_scan(today: date) -> list:
    """Run the universe scanner and catalyst classifier."""
    api_key = _get_api_key()
    if not api_key:
        logger.error("No POLYGON_API_KEY found. Set in .env or environment.")
        return []

    today_str = today.isoformat()
    prev = get_prev_trading_day(today)
    prev_str = prev.isoformat()

    # Get lookback dates for RVOL calculation
    lookback = []
    d = prev
    for _ in range(25):
        if d.weekday() < 5:
            lookback.append(d.isoformat())
        d -= timedelta(days=1)
    lookback.reverse()

    # Layer 1: Universe scan
    candidates = scan_for_candidates(today_str, prev_str, api_key, lookback)
    if not candidates:
        logger.info("No candidates from universe scanner")
        return []

    # Layer 2: Catalyst classification
    qualified = enrich_candidates(candidates)
    if not qualified:
        logger.info("No candidates passed catalyst filter")
        return []

    # Final watchlist (top N by catalyst score + momentum)
    watchlist = qualified[:SCAN_WATCHLIST_SIZE]
    logger.info("WATCHLIST (%d stocks):", len(watchlist))
    for c in watchlist:
        logger.info("  #%d %s  gap=%.1f%%  RVOL=%.1f  catalyst=%s(%d)  '%s'",
                    c.scan_rank, c.symbol, c.gap_pct * 100, c.rel_vol,
                    c.catalyst_type, c.catalyst_score, c.catalyst_headline[:60])
    return watchlist


def run_trading_session(watchlist: list, mode: str = "dry-run"):
    """Run the entry engine and position manager for the trading session."""
    bar_mgr = BarManager()
    pos_mgr = PositionManager()
    risk_gov = RiskGovernor()
    risk_gov.reset_daily()

    symbols = [c.symbol for c in watchlist]
    catalyst_scores = {c.symbol: c.catalyst_score for c in watchlist}

    logger.info("Trading session started. Watching: %s", ", ".join(symbols))
    logger.info("Mode: %s | Risk budget: $%.0f/trade", mode, risk_gov.get_risk_budget())

    # Track which symbols have already triggered entries
    entered_symbols = set()
    poll_count = 0

    while True:
        now = datetime.now()

        # Check if we should stop
        close_time_parts = EXIT_CLOSE_ALL_TIME.split(":")
        close_dt = now.replace(hour=int(close_time_parts[0]),
                               minute=int(close_time_parts[1]),
                               second=0, microsecond=0)
        if now >= close_dt:
            logger.info("Close time reached (%s). Closing all positions.", EXIT_CLOSE_ALL_TIME)
            prices = {}
            for sym in symbols:
                bars = bar_mgr.get_bars(sym)
                if not bars.empty:
                    prices[sym] = float(bars.iloc[-1]["Close"])
            pos_mgr.force_close_all(prices, "eod_close", str(now))
            break

        # Check risk governor
        can_trade, reason = risk_gov.can_trade(pos_mgr.open_count)
        if not can_trade and pos_mgr.open_count == 0:
            logger.info("Risk governor: %s. No open positions. Done for the day.", reason)
            break

        # Poll bars for all symbols
        for sym in symbols:
            bars = bar_mgr.get_bars(sym)
            if bars.empty:
                continue

            # Process existing positions
            if sym in pos_mgr.positions:
                last_bar = bars.iloc[-1]
                fills = pos_mgr.process_bar(sym, last_bar, str(bars.index[-1]))
                for fill in fills:
                    risk_gov.record_trade(fill["pnl"])
                    logger.info("FILL: %s %s %d@$%.2f (%s) P&L $%.0f",
                               fill["action"], fill["symbol"], fill["shares"],
                               fill["price"], fill["reason"], fill["pnl"])

            # Scan for new entries (if allowed and not already entered)
            if (can_trade and
                    sym not in entered_symbols and
                    sym not in pos_mgr.positions and
                    pos_mgr.open_count < ENTRY_MAX_CONCURRENT):

                signals = scan_for_entries(sym, bars, catalyst_scores.get(sym, 0))
                if signals:
                    best = signals[0]  # highest confidence

                    if mode == "dry-run":
                        logger.info("DRY-RUN SIGNAL: %s %s %s %d@$%.2f stop=$%.2f target=$%.2f",
                                   best.pattern, best.direction, best.symbol,
                                   best.shares, best.entry_price, best.stop_price, best.target_price)
                        entered_symbols.add(sym)
                    else:
                        pos = pos_mgr.open_position(best)
                        if pos:
                            entered_symbols.add(sym)
                            risk_gov.register_sector("")  # TODO: add sector lookup

        poll_count += 1
        if poll_count % 30 == 0:  # every ~5 minutes
            status = risk_gov.status()
            logger.info("STATUS: P&L $%.0f | Trades %d | Open %d | %s",
                       status["daily_pnl"], status["trades_today"],
                       pos_mgr.open_count,
                       "TRADING" if status["can_trade"] else status["reason"])

        time.sleep(10)  # poll every 10 seconds

    # End of session summary
    logger.info("=" * 60)
    logger.info("SESSION COMPLETE")
    logger.info("=" * 60)
    logger.info("  Trades: %d", pos_mgr.get_trade_count())
    logger.info("  P&L: $%.0f", pos_mgr.get_daily_pnl())
    if pos_mgr.closed_trades:
        wins = sum(1 for t in pos_mgr.closed_trades if t["net_pnl"] > 0)
        logger.info("  Win rate: %.0f%%", wins / len(pos_mgr.closed_trades) * 100)
        logger.info("  Avg R: %.2f",
                    sum(t["r_multiple"] for t in pos_mgr.closed_trades) / len(pos_mgr.closed_trades))

    return pos_mgr.closed_trades


def run_backtest_day(target_date: date) -> list[dict]:
    """Backtest a single day using Polygon 1-min data."""
    api_key = _get_api_key()
    if not api_key:
        logger.error("No POLYGON_API_KEY for backtest")
        return []

    today_str = target_date.isoformat()
    prev = get_prev_trading_day(target_date)

    # Run scanner for this date
    watchlist = run_premarket_scan(target_date)
    if not watchlist:
        logger.info("No watchlist for %s", today_str)
        return []

    bar_mgr = BarManager()
    pos_mgr = PositionManager()
    risk_gov = RiskGovernor()
    risk_gov.reset_daily()

    catalyst_scores = {c.symbol: c.catalyst_score for c in watchlist}
    entered_symbols = set()

    for candidate in watchlist[:SCAN_WATCHLIST_SIZE]:
        sym = candidate.symbol
        bars = bar_mgr.get_bars_polygon(sym, today_str, api_key)
        if bars.empty or len(bars) < 30:
            continue

        # Scan for entries
        signals = scan_for_entries(sym, bars, catalyst_scores.get(sym, 0))
        if not signals or sym in entered_symbols:
            continue

        can_trade, reason = risk_gov.can_trade(pos_mgr.open_count)
        if not can_trade:
            break

        best = signals[0]
        pos = pos_mgr.open_position(best)
        if not pos:
            continue
        entered_symbols.add(sym)

        # Walk remaining bars
        for i in range(best.bar_index + 1, len(bars)):
            bar = bars.iloc[i]
            fills = pos_mgr.process_bar(sym, bar, str(bars.index[i]))
            for fill in fills:
                risk_gov.record_trade(fill["pnl"])

        # Force close if still open
        if sym in pos_mgr.positions:
            last_price = float(bars.iloc[-1]["Close"])
            pos_mgr.force_close_all({sym: last_price}, "eod_close", str(bars.index[-1]))

    return pos_mgr.closed_trades


def save_trades(trades: list[dict], date_str: str):
    """Save trade log to CSV."""
    if not trades:
        return
    DT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(trades)
    path = DT_DATA_DIR / f"trades_{date_str}.csv"
    df.to_csv(path, index=False)
    logger.info("Trades saved to %s", path)

    # Also append to master log
    master = DT_DATA_DIR / "all_trades.csv"
    if master.exists():
        existing = pd.read_csv(master)
        df = pd.concat([existing, df], ignore_index=True)
    df.to_csv(master, index=False)


def main():
    parser = argparse.ArgumentParser(description="Day Trading Bot")
    parser.add_argument("--mode", choices=["dry-run", "paper", "live", "backtest"],
                       default="dry-run")
    parser.add_argument("--date", help="Backtest date (YYYY-MM-DD)")
    parser.add_argument("--scan-only", action="store_true",
                       help="Only run the scanner, don't trade")
    args = parser.parse_args()

    today = date.today()
    mode = args.mode

    logger.info("=" * 60)
    logger.info("DAY TRADING BOT — %s mode", mode.upper())
    logger.info("Account: $%s | Date: %s", f"{ACCOUNT_SIZE:,.0f}", today)
    logger.info("=" * 60)

    if mode == "backtest":
        if not args.date:
            logger.error("--date required for backtest mode")
            return
        target = datetime.strptime(args.date, "%Y-%m-%d").date()
        logger.info("Backtesting %s", target)
        trades = run_backtest_day(target)
        save_trades(trades, args.date)
        return

    # Live / paper / dry-run mode
    if args.scan_only:
        watchlist = run_premarket_scan(today)
        if watchlist:
            print(json.dumps([{
                "symbol": c.symbol, "gap_pct": round(c.gap_pct * 100, 1),
                "rel_vol": c.rel_vol, "catalyst": c.catalyst_type,
                "score": c.catalyst_score, "price": c.price,
            } for c in watchlist], indent=2))
        return

    watchlist = run_premarket_scan(today)
    if not watchlist:
        logger.info("No trading opportunities today. Exiting.")
        return

    trades = run_trading_session(watchlist, mode=mode)
    save_trades(trades, today.isoformat())


if __name__ == "__main__":
    main()
