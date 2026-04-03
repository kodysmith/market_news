#!/usr/bin/env python3
"""Backtest all 5 day trading strategies from the book on SPY.

Strategies:
  1. ORB (Opening Range Breakout) — 15-min variant
  2. VWAP Bounce — pullback to VWAP in an uptrend
  3. VWAP Rejection — rally to VWAP in a downtrend
  4. Trend Day — 9 EMA pullback on confirmed trend days
  5. Gap Fade — fading gaps that lose VWAP

Uses yfinance 5-min bars for SPY (max ~60 days of 5-min data available).
All strategies use 1% risk per trade on a $100K account.
"""
from __future__ import annotations

import sys
import math
from datetime import datetime, timedelta, time as dtime
from pathlib import Path
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ACCOUNT = 100_000
RISK_PCT = 0.01  # 1% risk per trade
MAX_RISK = ACCOUNT * RISK_PCT  # $1,000
COMMISSION_PER_SHARE = 0.005
SLIPPAGE_PER_SHARE = 0.02


@dataclass
class Trade:
    strategy: str
    date: str
    direction: str  # long / short
    entry_time: str
    exit_time: str
    entry_price: float
    exit_price: float
    stop_price: float
    target_price: float
    shares: int
    exit_reason: str  # take_profit, stop_loss, trail_stop, time_stop, eod
    gross_pnl: float = 0.0
    commission: float = 0.0
    slippage: float = 0.0
    net_pnl: float = 0.0
    r_multiple: float = 0.0


def fetch_spy_data(days: int = 59) -> pd.DataFrame:
    """Fetch SPY 5-min bars from yfinance."""
    import yfinance as yf
    end = datetime.now()
    start = end - timedelta(days=days)
    df = yf.Ticker("SPY").history(start=start.strftime("%Y-%m-%d"),
                                   end=end.strftime("%Y-%m-%d"),
                                   interval="5m")
    if df.empty:
        return df
    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_convert("America/New_York")
    else:
        df.index = df.index.tz_localize("America/New_York", ambiguous="NaT", nonexistent="NaT")
    # Filter to regular hours
    df = df.between_time("09:30", "15:55")
    return df


def compute_vwap(df: pd.DataFrame) -> pd.Series:
    """Cumulative VWAP from OHLCV bars."""
    typical = (df["High"] + df["Low"] + df["Close"]) / 3.0
    cum_vol = df["Volume"].cumsum()
    cum_tp_vol = (typical * df["Volume"]).cumsum()
    return cum_tp_vol / cum_vol.replace(0, 1)


def compute_ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def execute_trade(direction: str, entry_price: float, stop_price: float,
                  target_price: float, bars_after: pd.DataFrame,
                  max_hold_bars: int = 24) -> tuple[float, str, str]:
    """Simulate trade execution on subsequent bars. Returns (exit_price, exit_reason, exit_time)."""
    for i, (ts, bar) in enumerate(bars_after.iterrows()):
        if i >= max_hold_bars:
            return bar["Close"], "time_stop", str(ts)

        if direction == "long":
            # Check stop first (conservative)
            if bar["Low"] <= stop_price:
                return stop_price, "stop_loss", str(ts)
            if bar["High"] >= target_price:
                return target_price, "take_profit", str(ts)
        else:  # short
            if bar["High"] >= stop_price:
                return stop_price, "stop_loss", str(ts)
            if bar["Low"] <= target_price:
                return target_price, "take_profit", str(ts)

    # EOD exit
    if len(bars_after) > 0:
        last_bar = bars_after.iloc[-1]
        return last_bar["Close"], "eod", str(bars_after.index[-1])
    return entry_price, "no_bars", ""


def make_trade(strategy: str, date_str: str, direction: str,
               entry_time: str, entry_price: float, stop_price: float,
               target_price: float, bars_after: pd.DataFrame,
               max_hold_bars: int = 24) -> Trade:
    """Create and simulate a trade."""
    risk_per_share = abs(entry_price - stop_price)
    if risk_per_share < 0.01:
        risk_per_share = 0.01
    shares = min(int(MAX_RISK / risk_per_share), 5000)
    if shares < 1:
        shares = 1

    exit_price, exit_reason, exit_time = execute_trade(
        direction, entry_price, stop_price, target_price, bars_after, max_hold_bars
    )

    # Apply slippage
    if direction == "long":
        adj_entry = entry_price + SLIPPAGE_PER_SHARE
        adj_exit = exit_price - SLIPPAGE_PER_SHARE
        gross = (adj_exit - adj_entry) * shares
    else:
        adj_entry = entry_price - SLIPPAGE_PER_SHARE
        adj_exit = exit_price + SLIPPAGE_PER_SHARE
        gross = (adj_entry - adj_exit) * shares

    commission = shares * COMMISSION_PER_SHARE * 2  # round trip
    slippage_cost = SLIPPAGE_PER_SHARE * shares * 2
    net = gross - commission

    r_mult = net / (risk_per_share * shares) if risk_per_share * shares > 0 else 0

    return Trade(
        strategy=strategy, date=date_str, direction=direction,
        entry_time=entry_time, exit_time=exit_time,
        entry_price=entry_price, exit_price=exit_price,
        stop_price=stop_price, target_price=target_price,
        shares=shares, exit_reason=exit_reason,
        gross_pnl=round(gross, 2), commission=round(commission, 2),
        slippage=round(slippage_cost, 2), net_pnl=round(net, 2),
        r_multiple=round(r_mult, 2),
    )


# ============================================================
# STRATEGY 1: Opening Range Breakout (15-min)
# ============================================================
def strategy_orb(day_bars: pd.DataFrame, date_str: str, prev_close: float) -> list[Trade]:
    """15-minute Opening Range Breakout on SPY."""
    trades = []
    if len(day_bars) < 20:
        return trades

    # Opening range = first 3 bars (15 minutes of 5-min bars)
    or_bars = day_bars.iloc[:3]
    or_high = or_bars["High"].max()
    or_low = or_bars["Low"].min()
    or_range = or_high - or_low

    if or_range < 0.10:  # too tight, skip
        return trades

    # Look for breakout in bars 3-12 (9:45 - 10:30 AM)
    for i in range(3, min(12, len(day_bars))):
        bar = day_bars.iloc[i]
        ts = day_bars.index[i]

        # Long breakout
        if bar["Close"] > or_high and bar["Volume"] > day_bars.iloc[:i]["Volume"].mean():
            entry = or_high + 0.02
            stop = or_low  # conservative: opposite side of OR
            target = entry + 2 * (entry - stop)  # 2R
            remaining = day_bars.iloc[i+1:]
            if len(remaining) > 0:
                t = make_trade("ORB", date_str, "long", str(ts),
                              entry, stop, target, remaining, max_hold_bars=24)
                trades.append(t)
            break

        # Short breakout
        if bar["Close"] < or_low and bar["Volume"] > day_bars.iloc[:i]["Volume"].mean():
            entry = or_low - 0.02
            stop = or_high
            target = entry - 2 * (stop - entry)
            remaining = day_bars.iloc[i+1:]
            if len(remaining) > 0:
                t = make_trade("ORB", date_str, "short", str(ts),
                              entry, stop, target, remaining, max_hold_bars=24)
                trades.append(t)
            break

    return trades


# ============================================================
# STRATEGY 2: VWAP Bounce (Long) / STRATEGY 3: VWAP Rejection (Short)
# ============================================================
def strategy_vwap(day_bars: pd.DataFrame, date_str: str, prev_close: float) -> list[Trade]:
    """VWAP bounce (long) and VWAP rejection (short)."""
    trades = []
    if len(day_bars) < 12:
        return trades

    vwap = compute_vwap(day_bars)
    day_bars = day_bars.copy()
    day_bars["VWAP"] = vwap

    traded_today = False

    # Start looking after 10:00 AM (bar 6 = 30 min in)
    for i in range(6, min(36, len(day_bars))):  # up to 12:30 PM
        if traded_today:
            break

        bar = day_bars.iloc[i]
        ts = day_bars.index[i]
        cur_vwap = vwap.iloc[i]

        # Check if stock has been consistently above or below VWAP
        recent = day_bars.iloc[max(0, i-6):i]
        above_count = (recent["Close"] > recent["VWAP"]).sum()
        below_count = (recent["Close"] < recent["VWAP"]).sum()

        # VWAP BOUNCE (Long): stock above VWAP, pulls back to VWAP, bounces
        if above_count >= 4:  # mostly above VWAP
            # Check if current bar touches VWAP and bounces
            if bar["Low"] <= cur_vwap + 0.10 and bar["Close"] > cur_vwap:
                entry = bar["Close"]
                stop = cur_vwap - 0.20
                target = entry + 2 * (entry - stop)
                remaining = day_bars.iloc[i+1:]
                if len(remaining) > 0 and (entry - stop) > 0.10:
                    t = make_trade("VWAP_Bounce", date_str, "long", str(ts),
                                  entry, stop, target, remaining, max_hold_bars=18)
                    trades.append(t)
                    traded_today = True

        # VWAP REJECTION (Short): stock below VWAP, rallies to VWAP, gets rejected
        elif below_count >= 4:  # mostly below VWAP
            if bar["High"] >= cur_vwap - 0.10 and bar["Close"] < cur_vwap:
                entry = bar["Close"]
                stop = cur_vwap + 0.20
                target = entry - 2 * (stop - entry)
                remaining = day_bars.iloc[i+1:]
                if len(remaining) > 0 and (stop - entry) > 0.10:
                    t = make_trade("VWAP_Reject", date_str, "short", str(ts),
                                  entry, stop, target, remaining, max_hold_bars=18)
                    trades.append(t)
                    traded_today = True

    return trades


# ============================================================
# STRATEGY 4: Trend Day (9 EMA pullback)
# ============================================================
def strategy_trend_day(day_bars: pd.DataFrame, date_str: str, prev_close: float) -> list[Trade]:
    """Trend day: identified by 10:30 AM, enter on 9 EMA pullback."""
    trades = []
    if len(day_bars) < 18:
        return trades

    vwap = compute_vwap(day_bars)
    ema9 = compute_ema(day_bars["Close"], 9)

    # Check at bar 12 (10:30 AM): is this a trend day?
    check_idx = 12
    if check_idx >= len(day_bars):
        return trades

    bars_so_far = day_bars.iloc[:check_idx+1]
    all_above_vwap = all(bars_so_far["Close"].iloc[3:] > vwap.iloc[3:check_idx+1])
    all_below_vwap = all(bars_so_far["Close"].iloc[3:] < vwap.iloc[3:check_idx+1])

    if not all_above_vwap and not all_below_vwap:
        return trades  # not a trend day

    direction = "long" if all_above_vwap else "short"
    traded = False

    # Look for 9 EMA pullback entry from bar 12 onward
    for i in range(check_idx, min(48, len(day_bars))):  # up to 1:30 PM
        if traded:
            break
        bar = day_bars.iloc[i]
        ts = day_bars.index[i]
        cur_ema9 = ema9.iloc[i]

        if direction == "long":
            # Pullback to 9 EMA: low touches EMA, close above EMA
            if bar["Low"] <= cur_ema9 + 0.05 and bar["Close"] > cur_ema9:
                entry = bar["Close"]
                stop = bar["Low"] - 0.10
                target = entry + 3 * (entry - stop)  # 3R for trend days
                remaining = day_bars.iloc[i+1:]
                if len(remaining) > 0 and (entry - stop) > 0.05:
                    t = make_trade("Trend_Day", date_str, "long", str(ts),
                                  entry, stop, target, remaining, max_hold_bars=36)
                    trades.append(t)
                    traded = True
        else:  # short trend day
            if bar["High"] >= cur_ema9 - 0.05 and bar["Close"] < cur_ema9:
                entry = bar["Close"]
                stop = bar["High"] + 0.10
                target = entry - 3 * (stop - entry)
                remaining = day_bars.iloc[i+1:]
                if len(remaining) > 0 and (stop - entry) > 0.05:
                    t = make_trade("Trend_Day", date_str, "short", str(ts),
                                  entry, stop, target, remaining, max_hold_bars=36)
                    trades.append(t)
                    traded = True

    return trades


# ============================================================
# STRATEGY 5: Gap Fade
# ============================================================
def strategy_gap_fade(day_bars: pd.DataFrame, date_str: str, prev_close: float) -> list[Trade]:
    """Fade gaps that lose VWAP in the first 30 minutes."""
    trades = []
    if len(day_bars) < 10 or prev_close is None:
        return trades

    open_price = day_bars.iloc[0]["Open"]
    gap_pct = (open_price - prev_close) / prev_close * 100

    # Only fade gaps > 0.5% (meaningful but not too large)
    if abs(gap_pct) < 0.5 or abs(gap_pct) > 2.5:
        return trades

    vwap = compute_vwap(day_bars)

    # Check if gap is failing (gapped up but below VWAP by bar 6 = 10:00 AM)
    for i in range(6, min(12, len(day_bars))):
        bar = day_bars.iloc[i]
        ts = day_bars.index[i]
        cur_vwap = vwap.iloc[i]

        if gap_pct > 0.5 and bar["Close"] < cur_vwap:
            # Gap up but lost VWAP — fade it (short)
            entry = bar["Close"]
            stop = cur_vwap + 0.30
            target = prev_close  # target full gap fill
            if (stop - entry) > 0.10 and (entry - target) > (stop - entry):
                remaining = day_bars.iloc[i+1:]
                if len(remaining) > 0:
                    t = make_trade("Gap_Fade", date_str, "short", str(ts),
                                  entry, stop, target, remaining, max_hold_bars=36)
                    trades.append(t)
                break

        elif gap_pct < -0.5 and bar["Close"] > cur_vwap:
            # Gap down but reclaimed VWAP — fade it (long)
            entry = bar["Close"]
            stop = cur_vwap - 0.30
            target = prev_close
            if (entry - stop) > 0.10 and (target - entry) > (entry - stop):
                remaining = day_bars.iloc[i+1:]
                if len(remaining) > 0:
                    t = make_trade("Gap_Fade", date_str, "long", str(ts),
                                  entry, stop, target, remaining, max_hold_bars=36)
                    trades.append(t)
                break

    return trades


def main():
    print("Fetching SPY 5-min data...", file=sys.stderr)
    df = fetch_spy_data(days=59)
    if df.empty:
        print("ERROR: No data fetched", file=sys.stderr)
        return

    print(f"Got {len(df)} bars from {df.index[0]} to {df.index[-1]}", file=sys.stderr)

    # Group by trading day
    df["trade_date"] = df.index.date
    trading_days = sorted(df["trade_date"].unique())
    print(f"{len(trading_days)} trading days", file=sys.stderr)

    all_trades = []
    prev_close = None

    for day in trading_days:
        day_bars = df[df["trade_date"] == day].copy()
        if len(day_bars) < 10:
            if len(day_bars) > 0:
                prev_close = day_bars.iloc[-1]["Close"]
            continue

        date_str = str(day)

        # Run all strategies
        for strategy_fn in [strategy_orb, strategy_vwap, strategy_trend_day, strategy_gap_fade]:
            trades = strategy_fn(day_bars, date_str, prev_close)
            all_trades.extend(trades)

        prev_close = day_bars.iloc[-1]["Close"]

    if not all_trades:
        print("No trades generated", file=sys.stderr)
        return

    # Build results DataFrame
    results = pd.DataFrame([t.__dict__ for t in all_trades])

    # Save trades
    out_dir = ROOT / "data" / "day_trading_backtest"
    out_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_dir / "all_trades.csv", index=False)

    # Print comparison
    print(f"\n{'='*100}", file=sys.stderr)
    print(f"DAY TRADING STRATEGY COMPARISON — SPY 5-min bars", file=sys.stderr)
    print(f"Period: {trading_days[0]} to {trading_days[-1]} ({len(trading_days)} days)", file=sys.stderr)
    print(f"Account: ${ACCOUNT:,.0f} | Risk: {RISK_PCT*100:.0f}% per trade (${MAX_RISK:,.0f})", file=sys.stderr)
    print(f"{'='*100}", file=sys.stderr)

    print(f"\n{'Strategy':<16} {'Trades':>7} {'Wins':>5} {'Win%':>6} {'Avg Win':>9} {'Avg Loss':>9} "
          f"{'Avg R':>6} {'PF':>5} {'Total $':>10} {'$/Trade':>9} {'Best':>9} {'Worst':>9} "
          f"{'MaxDD':>9}", file=sys.stderr)
    print("-" * 115, file=sys.stderr)

    strategy_summaries = []
    for strat in sorted(results["strategy"].unique()):
        st = results[results["strategy"] == strat]
        n = len(st)
        wins = (st["net_pnl"] > 0).sum()
        losses = (st["net_pnl"] <= 0).sum()
        wr = wins / n * 100 if n > 0 else 0

        avg_win = st.loc[st["net_pnl"] > 0, "net_pnl"].mean() if wins > 0 else 0
        avg_loss = st.loc[st["net_pnl"] <= 0, "net_pnl"].mean() if losses > 0 else 0
        avg_r = st["r_multiple"].mean()

        gross_win = st.loc[st["net_pnl"] > 0, "net_pnl"].sum()
        gross_loss = abs(st.loc[st["net_pnl"] <= 0, "net_pnl"].sum())
        pf = gross_win / gross_loss if gross_loss > 0 else 999

        total = st["net_pnl"].sum()
        per_trade = total / n if n > 0 else 0
        best = st["net_pnl"].max()
        worst = st["net_pnl"].min()

        # Max drawdown
        cum = st["net_pnl"].cumsum()
        dd = (cum - cum.cummax()).min()

        print(f"{strat:<16} {n:>7} {wins:>5} {wr:>5.1f}% ${avg_win:>8,.0f} ${avg_loss:>8,.0f} "
              f"{avg_r:>5.2f} {pf:>4.1f} ${total:>9,.0f} ${per_trade:>8,.0f} ${best:>8,.0f} ${worst:>8,.0f} "
              f"${dd:>8,.0f}", file=sys.stderr)

        strategy_summaries.append({
            "strategy": strat, "trades": n, "win_rate": round(wr, 1),
            "avg_win": round(avg_win, 2), "avg_loss": round(avg_loss, 2),
            "avg_r": round(avg_r, 2), "profit_factor": round(pf, 2),
            "total_pnl": round(total, 2), "per_trade": round(per_trade, 2),
            "max_dd": round(dd, 2),
        })

    # Totals
    print("-" * 115, file=sys.stderr)
    total_trades = len(results)
    total_pnl = results["net_pnl"].sum()
    total_wins = (results["net_pnl"] > 0).sum()
    total_wr = total_wins / total_trades * 100 if total_trades > 0 else 0
    total_comm = results["commission"].sum()

    cum_all = results.sort_values("entry_time")["net_pnl"].cumsum()
    total_dd = (cum_all - cum_all.cummax()).min()

    print(f"{'ALL COMBINED':<16} {total_trades:>7} {total_wins:>5} {total_wr:>5.1f}% "
          f"{'':>9} {'':>9} {results['r_multiple'].mean():>5.2f} "
          f"{'':>5} ${total_pnl:>9,.0f} ${total_pnl/total_trades:>8,.0f} "
          f"{'':>9} {'':>9} ${total_dd:>8,.0f}", file=sys.stderr)

    print(f"\nTotal commission paid: ${total_comm:,.0f}", file=sys.stderr)
    print(f"Total slippage cost:  ${results['slippage'].sum():,.0f}", file=sys.stderr)

    # Exit reason breakdown
    print(f"\n{'='*60}", file=sys.stderr)
    print(f"EXIT REASON BREAKDOWN", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)
    for strat in sorted(results["strategy"].unique()):
        st = results[results["strategy"] == strat]
        print(f"\n  {strat}:", file=sys.stderr)
        for reason in st["exit_reason"].unique():
            rt = st[st["exit_reason"] == reason]
            print(f"    {reason:<15} {len(rt):>4} trades  avg P&L ${rt['net_pnl'].mean():>+8,.0f}", file=sys.stderr)

    # Daily P&L distribution
    print(f"\n{'='*60}", file=sys.stderr)
    print(f"DAILY P&L DISTRIBUTION", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)
    daily = results.groupby("date")["net_pnl"].sum()
    print(f"  Trading days with trades: {len(daily)}", file=sys.stderr)
    print(f"  Positive days: {(daily > 0).sum()} ({(daily > 0).mean()*100:.0f}%)", file=sys.stderr)
    print(f"  Negative days: {(daily < 0).sum()} ({(daily < 0).mean()*100:.0f}%)", file=sys.stderr)
    print(f"  Avg daily P&L: ${daily.mean():,.0f}", file=sys.stderr)
    print(f"  Median daily P&L: ${daily.median():,.0f}", file=sys.stderr)
    print(f"  Best day: ${daily.max():,.0f}", file=sys.stderr)
    print(f"  Worst day: ${daily.min():,.0f}", file=sys.stderr)

    # Annualized estimates
    days_in_sample = len(trading_days)
    if days_in_sample > 0:
        annual_factor = 252 / days_in_sample
        print(f"\n{'='*60}", file=sys.stderr)
        print(f"ANNUALIZED ESTIMATES (extrapolated from {days_in_sample} days)", file=sys.stderr)
        print(f"{'='*60}", file=sys.stderr)
        for s in strategy_summaries:
            ann_pnl = s["total_pnl"] * annual_factor
            ann_trades = s["trades"] * annual_factor
            print(f"  {s['strategy']:<16} ${ann_pnl:>10,.0f}/yr  ({ann_trades:.0f} trades/yr)", file=sys.stderr)
        ann_total = total_pnl * annual_factor
        print(f"  {'COMBINED':<16} ${ann_total:>10,.0f}/yr  ROI: {ann_total/ACCOUNT*100:.0f}%", file=sys.stderr)

    # Save summary
    pd.DataFrame(strategy_summaries).to_csv(out_dir / "strategy_comparison.csv", index=False)
    print(f"\nResults saved to {out_dir}/", file=sys.stderr)


if __name__ == "__main__":
    main()
