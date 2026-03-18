#!/usr/bin/env python3
"""
Backtest the scalp bot strategy on any historical day(s).

Fetches 1-min bars from yfinance, computes indicators, generates signals
using the same logic as the live bot, and runs TradeManager with full
realism (slippage, commission, exact TP/SL).

GEX data is not available historically — you can set a fixed regime
assumption via --regime (POSITIVE, NEGATIVE, or UNKNOWN).

Limitations:
  - yfinance 1-min bars only available for the last ~8 calendar days
  - For dates older than 8 days, use --interval 2m or 5m
  - GEX regime is assumed, not real

Usage:
    # Single day
    python scripts/backtest_scalp.py --ticker SPY --date 2026-03-07

    # Date range (all available days within range)
    python scripts/backtest_scalp.py --ticker TSLA --start 2026-03-04 --end 2026-03-10

    # With assumed GEX regime
    python scripts/backtest_scalp.py --ticker SPY --date 2026-03-07 --regime NEGATIVE

    # Multi-day with 5m bars (for dates >8 days old)
    python scripts/backtest_scalp.py --ticker SPY --start 2026-02-01 --end 2026-02-28 --interval 5m
"""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime, date, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.spx_scalp_bot import (
    ScalpSignal,
    TradeManager,
    compute_atr,
    compute_ema,
    compute_rsi,
    compute_vwap,
    yf_symbol,
    SLIPPAGE_PCT,
    COMMISSION_PER_TRADE,
)

ET = ZoneInfo("America/New_York")

# ─── VIX fetcher (cached for the whole backtest) ─────────────────────────────

_vix_cache: dict[str, pd.DataFrame] = {}


def get_vix_bars(start: date, end: date, interval: str = "1m") -> pd.DataFrame:
    """Fetch VIX bars for the date range. Cached."""
    key = f"{start}_{end}_{interval}"
    if key not in _vix_cache:
        try:
            df = yf.Ticker("^VIX").history(
                start=str(start), end=str(end + timedelta(days=1)),
                interval=interval, auto_adjust=True,
            )
            _vix_cache[key] = df
        except Exception:
            _vix_cache[key] = pd.DataFrame()
    return _vix_cache[key]


# ─── Fake GEX state for backtesting ─────────────────────────────────────────

def make_fake_gex(regime: str, spot: float) -> dict:
    """Build a minimal GEX state dict with assumed regime."""
    return {
        "regime": regime,
        "flip": 0.0,
        "net_gex": 0.0,
        "walls_regime": {},
    }


# ─── Core backtest for a single day ─────────────────────────────────────────

def backtest_day(
    ticker: str,
    day: date,
    regime: str,
    interval: str = "1m",
    mgr: TradeManager | None = None,
    vix_bars: pd.DataFrame | None = None,
    verbose: bool = False,
) -> TradeManager:
    """Run the strategy on one day of 1-min bars. Returns the TradeManager."""
    sym = yf_symbol(ticker)
    df = yf.Ticker(sym).history(
        start=str(day), end=str(day + timedelta(days=1)),
        interval=interval, auto_adjust=True,
    )
    if df.empty:
        print(f"  {day}: no data for {ticker} ({sym})")
        return mgr

    # Filter to market hours 9:30-16:00 ET
    df.index = df.index.tz_convert(ET)
    market_open = df.index[0].replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = df.index[0].replace(hour=16, minute=0, second=0, microsecond=0)
    df = df[(df.index >= market_open) & (df.index <= market_close)]

    if df.empty:
        print(f"  {day}: no market-hours bars")
        return mgr

    if mgr is None:
        out_dir = ROOT / "logs" / "scalp_bot" / "backtest"
        out_dir.mkdir(parents=True, exist_ok=True)
        mgr = TradeManager(out_dir)
        date_str = day.strftime("%Y%m%d")
        mgr.trade_log_path = out_dir / f"trades_bt_{ticker}_{date_str}.csv"
        mgr.signal_log_path = out_dir / f"signals_bt_{ticker}_{date_str}.csv"
        mgr._init_trade_log()
        mgr._init_signal_log()

    # Get VIX for corresponding timestamps
    vix_series = None
    if vix_bars is not None and not vix_bars.empty:
        vix_day = vix_bars[vix_bars.index.date == day]
        if not vix_day.empty:
            vix_series = vix_day["Close"]

    n_bars = len(df)
    print(f"  {day} ({day.strftime('%a')}): {n_bars} bars, regime={regime}")

    # Walk through bars building up the intraday context
    for i in range(20, n_bars):  # need at least 20 bars for EMA21
        bar_time = df.index[i]
        bars_so_far = df.iloc[:i+1]

        price = float(bars_so_far["Close"].iloc[-1])

        # Indicators from bars
        closes = bars_so_far["Close"]
        ema9 = float(compute_ema(closes, 9).iloc[-1])
        ema21 = float(compute_ema(closes, 21).iloc[-1])
        rsi = compute_rsi(closes, 14)
        atr = compute_atr(bars_so_far, 14)
        vwap = compute_vwap(bars_so_far)
        price_vs_vwap = price - vwap

        # VIX lookup
        vix = 20.0  # default assumption
        if vix_series is not None and len(vix_series) > 0:
            # Find nearest VIX bar
            idx = vix_series.index.get_indexer([bar_time], method="ffill")
            if idx[0] >= 0:
                vix = float(vix_series.iloc[idx[0]])

        # Build fake GEX
        gex = make_fake_gex(regime, price)
        gex_regime = regime
        flip_line = 0.0
        dist_to_flip = 0.0
        net_gex = 0.0
        call_wall = 0.0
        put_wall = 0.0

        # Use the same signal logic from the bot
        from scripts.spx_scalp_bot import generate_signal
        bar_ts = bar_time.strftime("%Y-%m-%d %H:%M:%S")
        sig = generate_signal(bars_so_far, gex, vix, price, ticker=ticker, timestamp=bar_ts)

        # Skip last 5 minutes for new entries
        minutes_left = (market_close - bar_time).total_seconds() / 60
        if minutes_left <= 5 and not mgr.open_trade:
            sig = ScalpSignal(
                timestamp=bar_time.strftime("%Y-%m-%d %H:%M:%S"),
                ticker=ticker, spx_price=price, vix=vix,
                gex_regime=gex_regime, flip_line=flip_line,
                distance_to_flip=dist_to_flip, net_gex=net_gex,
                call_wall=call_wall, put_wall=put_wall,
                ema9=ema9, ema21=ema21, rsi14=rsi, atr14=atr,
                price_vs_vwap=price_vs_vwap,
                signal="FLAT", reason="LAST_5MIN", confidence=0,
            )

        mgr.process_tick(sig)

        if verbose and sig.signal != "FLAT":
            print(
                f"    [{bar_time.strftime('%H:%M')}] {price:.2f} "
                f"RSI={rsi:.0f} sig={sig.signal} ({sig.confidence:.2f})"
            )

    # Force-close open trade at EOD
    if mgr.open_trade:
        last_price = float(df["Close"].iloc[-1])
        is_long = mgr.open_trade.direction == "LONG"
        if is_long:
            exit_px = last_price * (1 - SLIPPAGE_PCT / 100)
        else:
            exit_px = last_price * (1 + SLIPPAGE_PCT / 100)
        exit_px = round(exit_px, 2)
        gross = (exit_px - mgr.open_trade.entry_price) if is_long else (mgr.open_trade.entry_price - exit_px)
        total_comm = 2 * COMMISSION_PER_TRADE
        net = gross - total_comm
        slip_entry = abs(mgr.open_trade.entry_price * SLIPPAGE_PCT / 100)
        slip_exit = abs(exit_px * SLIPPAGE_PCT / 100)

        entry_dt = datetime.strptime(mgr.open_trade.entry_time, "%Y-%m-%d %H:%M:%S")
        exit_dt = df.index[-1].replace(tzinfo=None)

        mgr.open_trade.exit_time = df.index[-1].strftime("%Y-%m-%d %H:%M:%S")
        mgr.open_trade.exit_price = exit_px
        mgr.open_trade.exit_reason = "EOD_CLOSE"
        mgr.open_trade.gross_pnl_points = round(gross, 2)
        mgr.open_trade.slippage_cost = round(slip_entry + slip_exit, 4)
        mgr.open_trade.commission_cost = round(total_comm, 4)
        mgr.open_trade.pnl_points = round(net, 2)
        mgr.open_trade.duration_sec = int((exit_dt - entry_dt).total_seconds())
        mgr._log_trade(mgr.open_trade)
        mgr.stats["total"] += 1
        mgr.stats["total_pnl"] += net
        if net > 0:
            mgr.stats["wins"] += 1
        else:
            mgr.stats["losses"] += 1
        mgr.open_trade = None

    return mgr


# ─── Print results ───────────────────────────────────────────────────────────

def print_results(mgr: TradeManager):
    try:
        with open(mgr.trade_log_path) as f:
            trades = list(csv.DictReader(f))
    except Exception:
        trades = []

    s = mgr.stats
    print()
    print("=" * 65)
    print("BACKTEST RESULTS")
    print("=" * 65)
    print(f"  Total trades:     {s['total']}")
    print(f"  Wins / Losses:    {s['wins']} / {s['losses']}")
    if s["total"]:
        print(f"  Win rate:         {s['wins'] / s['total'] * 100:.1f}%")
    print(f"  Net PnL:          {s['total_pnl']:+.2f} pts")

    if trades:
        pnls = [float(t["pnl_points"]) for t in trades]
        gross_pnls = [float(t.get("gross_pnl_points", t["pnl_points"])) for t in trades]
        wins = [p for p in pnls if p > 0]
        losses_list = [p for p in pnls if p <= 0]
        total_slip = sum(float(t.get("slippage_cost", 0)) for t in trades)
        total_comm = sum(float(t.get("commission_cost", 0)) for t in trades)

        print(f"  Gross PnL:        {sum(gross_pnls):+.2f} pts")
        print(f"  Total slippage:   {total_slip:.4f} pts")
        print(f"  Total commission: {total_comm:.2f} pts")
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
        if durations:
            print(f"  Avg hold time:    {sum(durations)/len(durations)/60:.1f} min")

        # By exit reason
        print()
        print("  BY EXIT REASON:")
        reasons = sorted(set(t["exit_reason"] for t in trades))
        for r in reasons:
            rt = [t for t in trades if t["exit_reason"] == r]
            rp = [float(t["pnl_points"]) for t in rt]
            rw = len([p for p in rp if p > 0])
            print(f"    {r}: {len(rt)} trades, {rw}W, PnL={sum(rp):+.2f}")

        # Trade detail
        print()
        print("  TRADES:")
        print(f"    {'#':>3}  {'Date':>10}  {'Time':>5}  {'Dir':>5}  {'Entry':>9}  {'Exit':>9}  "
              f"{'Gross':>7}  {'Net':>7}  {'Reason'}")
        print("    " + "-" * 85)
        for i, t in enumerate(trades, 1):
            parts = t["entry_time"].split(" ")
            d_str = parts[0] if len(parts) > 1 else ""
            t_str = parts[1][:5] if len(parts) > 1 else parts[0][:5]
            gross = float(t.get("gross_pnl_points", t["pnl_points"]))
            net = float(t["pnl_points"])
            print(
                f"    {i:>3}  {d_str:>10}  {t_str:>5}  {t['direction']:>5}  "
                f"${float(t['entry_price']):>8.2f}  ${float(t['exit_price']):>8.2f}  "
                f"{gross:>+7.2f}  {net:>+7.2f}  {t['exit_reason']}"
            )

    print()
    print("=" * 65)
    print(f"Trade CSV: {mgr.trade_log_path}")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Backtest scalp bot on historical data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/backtest_scalp.py --ticker SPY --date 2026-03-07
  python scripts/backtest_scalp.py --ticker TSLA --start 2026-03-04 --end 2026-03-10
  python scripts/backtest_scalp.py --ticker SPY --date 2026-03-07 --regime NEGATIVE
  python scripts/backtest_scalp.py --ticker SPY --start 2026-02-01 --end 2026-02-28 --interval 5m
        """,
    )
    parser.add_argument("--ticker", type=str, default="SPY", help="Ticker (default: SPY)")
    parser.add_argument("--date", type=str, default=None, help="Single date YYYY-MM-DD")
    parser.add_argument("--start", type=str, default=None, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", type=str, default=None, help="End date YYYY-MM-DD")
    parser.add_argument("--regime", type=str, default="UNKNOWN",
                        choices=["POSITIVE", "NEGATIVE", "UNKNOWN"],
                        help="Assumed GEX regime (default: UNKNOWN)")
    parser.add_argument("--interval", type=str, default="1m",
                        help="Bar interval: 1m, 2m, 5m (default: 1m). Use 5m for dates >8 days old.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    ticker = args.ticker.upper()

    # Determine date range
    if args.date:
        start_date = date.fromisoformat(args.date)
        end_date = start_date
    elif args.start and args.end:
        start_date = date.fromisoformat(args.start)
        end_date = date.fromisoformat(args.end)
    elif args.start:
        start_date = date.fromisoformat(args.start)
        end_date = date.today()
    else:
        # Default: yesterday
        end_date = date.today() - timedelta(days=1)
        start_date = end_date

    # Build list of trading days (weekdays)
    days = []
    d = start_date
    while d <= end_date:
        if d.weekday() < 5:
            days.append(d)
        d += timedelta(days=1)

    if not days:
        print("No trading days in range.")
        sys.exit(1)

    print("=" * 65)
    print(f"BACKTEST: {ticker}  {start_date} → {end_date}  ({len(days)} days)")
    print(f"  Regime: {args.regime}  Interval: {args.interval}")
    print(f"  Slippage: {SLIPPAGE_PCT}%  Commission: ${COMMISSION_PER_TRADE}/side")
    print("=" * 65)

    # Pre-fetch VIX
    vix_bars = pd.DataFrame()
    try:
        vix_bars = yf.Ticker("^VIX").history(
            start=str(start_date), end=str(end_date + timedelta(days=1)),
            interval=args.interval, auto_adjust=True,
        )
        if not vix_bars.empty:
            vix_bars.index = vix_bars.index.tz_convert(ET)
            print(f"  VIX bars loaded: {len(vix_bars)}")
    except Exception as e:
        print(f"  VIX fetch failed ({e}), using default VIX=20")

    # Set up shared TradeManager for multi-day
    out_dir = ROOT / "logs" / "scalp_bot" / "backtest"
    out_dir.mkdir(parents=True, exist_ok=True)
    mgr = TradeManager(out_dir)
    date_range = f"{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}" if start_date != end_date else start_date.strftime("%Y%m%d")
    mgr.trade_log_path = out_dir / f"trades_bt_{ticker}_{date_range}.csv"
    mgr.signal_log_path = out_dir / f"signals_bt_{ticker}_{date_range}.csv"
    # Remove old files to start fresh
    mgr.trade_log_path.unlink(missing_ok=True)
    mgr.signal_log_path.unlink(missing_ok=True)
    mgr._init_trade_log()
    mgr._init_signal_log()

    print()
    for day in days:
        mgr = backtest_day(
            ticker=ticker,
            day=day,
            regime=args.regime,
            interval=args.interval,
            mgr=mgr,
            vix_bars=vix_bars,
            verbose=args.verbose,
        )

    print_results(mgr)


if __name__ == "__main__":
    main()
