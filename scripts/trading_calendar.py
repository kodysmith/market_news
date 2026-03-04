#!/usr/bin/env python3
"""NYSE trading calendar for backtest: trading-day index and DTE in trading sessions."""
from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def _weekday_trading_days(start: date, end: date) -> pd.DatetimeIndex:
    """Fallback: weekdays only (no NYSE holidays). Use when pandas_market_calendars unavailable."""
    from datetime import timedelta
    out = []
    d = start
    while d <= end:
        if d.weekday() < 5:
            out.append(pd.Timestamp(d))
        d += timedelta(days=1)
    return pd.DatetimeIndex(out) if out else pd.DatetimeIndex([])


def get_trading_days_index(start: date, end: date) -> pd.DatetimeIndex:
    """Return DatetimeIndex of NYSE trading days in [start, end] (inclusive)."""
    try:
        import pandas_market_calendars as mcal
        nyse = mcal.get_calendar("NYSE")
        schedule = nyse.schedule(
            start_time=pd.Timestamp(start),
            end_time=pd.Timestamp(end),
        )
        if schedule.empty:
            return _weekday_trading_days(start, end)
        idx = schedule.index
        if idx.tz is not None:
            idx = idx.tz_localize(None)
        return idx.normalize()
    except Exception:
        return _weekday_trading_days(start, end)


def entry_date_for_dte(trading_index: pd.DatetimeIndex, expiry_date: date, dte_trading_days: int) -> date | None:
    """
    Return the entry date that is exactly dte_trading_days trading sessions before expiry.
    Entry date is the trading day such that there are dte_trading_days trading days from (entry+1) to expiry inclusive.
    Returns None if expiry not in calendar or not enough sessions before it.
    """
    if trading_index.empty:
        return None
    # All dates in index as date type
    day_list = [pd.Timestamp(ts).date() for ts in trading_index]
    try:
        exp_pos = day_list.index(expiry_date)
    except ValueError:
        return None
    # Entry is dte_trading_days sessions before expiry: entry at index exp_pos - dte_trading_days
    entry_pos = exp_pos - dte_trading_days
    if entry_pos < 0:
        return None
    return day_list[entry_pos]


def trading_days_between(trading_index: pd.DatetimeIndex, start: date, end: date) -> int:
    """Count trading days in (start, end] (start exclusive, end inclusive)."""
    if trading_index.empty:
        return 0
    day_list = [pd.Timestamp(ts).date() for ts in trading_index]
    return sum(1 for d in day_list if start < d <= end)


def get_trading_days_in_range(trading_index: pd.DatetimeIndex, start: date, end: date) -> list[date]:
    """Return sorted list of trading dates in [start, end]."""
    if trading_index.empty:
        return []
    day_list = [pd.Timestamp(ts).date() for ts in trading_index]
    return sorted(d for d in day_list if start <= d <= end)


def main() -> int:
    """CLI: export NYSE trading dates for a range to CSV or stdout."""
    import argparse
    parser = argparse.ArgumentParser(description="Export NYSE trading dates for a date range.")
    parser.add_argument("--start", default="2025-01-01", metavar="YYYY-MM-DD", help="Start date")
    parser.add_argument("--end", default="2025-12-31", metavar="YYYY-MM-DD", help="End date")
    parser.add_argument("--output", "-o", type=Path, default=None, help="Output CSV path (default: print to stdout)")
    args = parser.parse_args()
    start = datetime.strptime(args.start, "%Y-%m-%d").date()
    end = datetime.strptime(args.end, "%Y-%m-%d").date()
    idx = get_trading_days_index(start, end)
    dates = [pd.Timestamp(ts).date() for ts in idx]
    df = pd.DataFrame({"date": dates})
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.output, index=False)
        print(f"Wrote {len(dates)} trading days to {args.output}")
    else:
        df.to_csv(sys.stdout, index=False)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
