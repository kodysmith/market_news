#!/usr/bin/env python3
"""
Run scanner logic on historical dates to produce a short list of (date, symbol) candidates
for the Warrior backtest.

Data source: --source polygon | yfinance | ibkr
  polygon: Polygon grouped daily (requires POLYGON_API_KEY; may hit 403 past entitlements).
  yfinance: Free, no key; requires --symbols or --symbols-csv (no --scan-all).
  ibkr: TWS/Gateway must be running; requires --symbols or --symbols-csv.

Output: CSV with date, symbol, gap_pct, rel_vol, prev_day_dollar_vol, total_score, rank.
Use with backtest: --scanner-candidates this_file.csv

Usage:
  python scripts/run_historical_scanner.py --start 2024-01-01 --end 2024-12-31 --source yfinance --symbols-csv data/sec_universe.json
  python scripts/run_historical_scanner.py --start 2024-01-01 --end 2024-06-30 --source ibkr --symbols symbols.txt --top-n 25
"""
from __future__ import annotations

import json
import math
import os
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Reuse backtest cache and gap/relvol logic
from scripts.backtest_warrior_gap_pullback import (
    CACHE_DIR,
    gap_candidates_for_day,
    get_daily_bars_for_date,
    get_prev_trading_day,
    get_trading_days_index,
    get_trading_days_in_range,
    relvol_filter,
    to_datestr,
)

# Load .env
_env = ROOT / ".env"
if _env.is_file():
    with open(_env) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                k, v = k.strip(), v.strip().strip("'\"").strip()
                if k and k not in os.environ:
                    os.environ[k] = v


def _gap_score(gap_pct: float, cap_at: float = 0.20) -> float:
    if gap_pct <= 0:
        return 0.0
    return 40.0 * min(1.0, min(gap_pct, cap_at) / cap_at)


def _relvol_score(rel_vol: float, log_scale: bool = True) -> float:
    if rel_vol <= 0:
        return 0.0
    if log_scale:
        x = math.log2(rel_vol + 1)
        return min(30.0, x * 10.0)
    return min(30.0, rel_vol * 3.0)


def _liquidity_score(prev_day_dollar_vol: float | None, threshold_10m: float = 10_000_000) -> float:
    if prev_day_dollar_vol is None or prev_day_dollar_vol <= 0:
        return 0.0
    if prev_day_dollar_vol >= threshold_10m * 5:
        return 15.0
    return 15.0 * min(1.0, prev_day_dollar_vol / threshold_10m)


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Historical scanner: gap+relvol candidates per day, ranked, for backtest input")
    ap.add_argument("--start", required=True, help="YYYY-MM-DD")
    ap.add_argument("--end", required=True, help="YYYY-MM-DD")
    ap.add_argument("--gap", type=float, default=0.05, help="Min gap fraction (e.g. 0.05 = 5%%)")
    ap.add_argument("--relvol", type=float, default=3.0, help="Min relative volume (daily)")
    ap.add_argument("--relvol-lookback-days", type=int, default=20)
    ap.add_argument("--top-n", type=int, default=50, help="Max candidates per day (by total_score)")
    ap.add_argument("--min-price", type=float, default=1.0)
    ap.add_argument("--max-price", type=float, default=30.0)
    ap.add_argument("--output", "-o", type=Path, default=Path("data/scanner_candidates.csv"), help="Output CSV")
    ap.add_argument("--sleep", type=float, default=0.0)
    ap.add_argument("--source", choices=("polygon", "yfinance", "ibkr"), default="polygon",
                    help="Daily data source: polygon (API key), yfinance (free), ibkr (TWS/Gateway)")
    ap.add_argument("--scan-all", action="store_true", help="Scan all tickers (polygon only)")
    ap.add_argument("--symbols", type=str, default=None, help="Txt file, one symbol per line")
    ap.add_argument("--symbols-csv", type=str, default=None, help="CSV with column 'symbol'")
    ap.add_argument("--ibkr-host", default="127.0.0.1", help="IBKR TWS/Gateway host")
    ap.add_argument("--ibkr-port", type=int, default=7497, help="IBKR port (7497 paper, 7496 live)")
    ap.add_argument("--ibkr-client-id", type=int, default=1, help="IBKR client ID")
    args = ap.parse_args()

    api_key = os.environ.get("POLYGON_API_KEY") if args.source == "polygon" else None
    if args.source == "polygon" and not api_key:
        print("Set POLYGON_API_KEY for --source polygon", file=sys.stderr)
        return 1
    if args.source in ("yfinance", "ibkr") and not args.symbols and not args.symbols_csv:
        print("For --source yfinance or --source ibkr provide --symbols or --symbols-csv", file=sys.stderr)
        return 1
    if args.scan_all and args.source != "polygon":
        print("--scan-all is only valid with --source polygon", file=sys.stderr)
        return 1

    symbol_universe = None
    symbols_list: list[str] = []  # for yfinance/ibkr we need explicit list
    if args.symbols:
        path = Path(args.symbols)
        if not path.is_absolute():
            path = ROOT / path
        syms = []
        with open(path) as f:
            for line in f:
                s = line.strip().upper()
                if s and not s.startswith("#"):
                    syms.append(s)
        symbol_universe = sorted(set(syms))
        symbols_list = symbol_universe
    elif args.symbols_csv:
        path = Path(args.symbols_csv)
        if not path.is_absolute():
            path = ROOT / path
        if path.suffix.lower() == ".json":
            with open(path) as f:
                data = json.load(f)
            constituents = data.get("constituents", [])
            symbol_universe = sorted(set(str(c.get("ticker", "")).strip().upper() for c in constituents if c.get("ticker")))
        else:
            df = pd.read_csv(path)
            if "symbol" not in df.columns:
                raise ValueError("CSV must have 'symbol' column")
            symbol_universe = sorted(set(df["symbol"].astype(str).str.upper().tolist()))
        symbols_list = symbol_universe
    elif args.scan_all:
        symbol_universe = None  # all from Polygon response
        symbols_list = []

    # For yfinance: strip $ and skip tickers that often 404 (warrants, units, OTC)
    if args.source == "yfinance" and symbols_list:
        def _ok_for_yf(s: str) -> bool:
            s = (s or "").strip().lstrip("$").upper()
            if not s or len(s) < 2:
                return False
            su = s.upper()
            if su.endswith("-WT") or su.endswith("-W") or su.endswith("-WW") or ".OB" in su or ".PK" in su:
                return False
            return True
        symbols_list = sorted(set(s.lstrip("$").strip().upper() for s in symbols_list if _ok_for_yf(s)))
        if not symbols_list:
            print("No symbols left after filtering for yfinance (warrants/OTC skipped)", file=sys.stderr)
            return 1

    start_d = datetime.strptime(args.start, "%Y-%m-%d").date()
    end_d = datetime.strptime(args.end, "%Y-%m-%d").date()
    trading_index = get_trading_days_index(start_d, end_d)
    trading_dates = get_trading_days_in_range(trading_index, start_d, end_d)
    if not trading_dates:
        print("No trading days in range", file=sys.stderr)
        return 1

    # Extended range for relvol lookback
    extend_start = trading_dates[0]
    for _ in range(args.relvol_lookback_days + 5):
        prev = extend_start - timedelta(days=1)
        while prev.weekday() >= 5:
            prev -= timedelta(days=1)
        extend_start = prev
    extended_index = get_trading_days_index(extend_start, end_d)
    extended_dates_list = get_trading_days_in_range(extended_index, extend_start, end_d)

    daily_bars_by_date: dict[str, pd.DataFrame] = {}
    for d in extended_dates_list:
        ds = to_datestr(d)
        df = get_daily_bars_for_date(
            ds,
            (symbols_list if symbols_list else None),
            args.source,
            api_key,
            CACHE_DIR,
            args.sleep,
            getattr(args, "ibkr_host", "127.0.0.1"),
            getattr(args, "ibkr_port", 7497),
            getattr(args, "ibkr_client_id", 1),
        )
        if df is not None and not df.empty:
            daily_bars_by_date[ds] = df

    rows: list[dict] = []
    total_days = len(trading_dates)

    for day_idx, d in enumerate(trading_dates):
        date_str = to_datestr(d)
        prev_d = get_prev_trading_day(trading_dates, d)
        if prev_d is None:
            continue
        prev_str = to_datestr(prev_d)
        daily_today = daily_bars_by_date.get(date_str)
        daily_prev = daily_bars_by_date.get(prev_str)
        if daily_today is None or daily_prev is None or daily_today.empty or daily_prev.empty:
            continue

        candidates = gap_candidates_for_day(
            date_str, prev_str, daily_today, daily_prev, args.gap, symbol_universe
        )
        if candidates.empty:
            if (day_idx + 1) % 50 == 0 or day_idx == 0:
                print(f"  [{day_idx+1}/{total_days}] {date_str}: 0 gap candidates")
            continue

        # Price filter
        candidates = candidates[
            (candidates["day_open"] >= args.min_price) &
            (candidates["day_open"] <= args.max_price)
        ]
        if candidates.empty:
            continue

        candidates = relvol_filter(
            candidates,
            extended_dates_list,
            d,
            daily_bars_by_date,
            args.relvol,
            args.relvol_lookback_days,
        )
        if candidates.empty:
            continue

        # Prev day dollar vol = prev_close * prev_volume from daily_prev
        prev_vol = daily_prev[["symbol", "v"]].rename(columns={"v": "prev_volume"})
        candidates = candidates.merge(prev_vol, on="symbol", how="left")
        candidates["prev_day_dollar_vol"] = (candidates["prev_close"] * candidates["prev_volume"].fillna(0)).replace(0, float("nan"))

        # Score and rank
        def total_score(row: pd.Series) -> float:
            gs = _gap_score(row["gap_pct"])
            rvs = _relvol_score(row["rel_vol"])
            pv = row.get("prev_day_dollar_vol")
            if pv is None or (isinstance(pv, float) and math.isnan(pv)) or pv <= 0:
                pv = None
            liq = _liquidity_score(pv)
            return gs + rvs + liq

        candidates["total_score"] = candidates.apply(total_score, axis=1)
        candidates = candidates.sort_values("total_score", ascending=False).head(args.top_n)
        for rank_one, (_, row) in enumerate(candidates.iterrows(), start=1):
            rows.append({
                "date": date_str,
                "symbol": row["symbol"],
                "gap_pct": row["gap_pct"],
                "rel_vol": row["rel_vol"],
                "day_open": row["day_open"],
                "prev_close": row["prev_close"],
                "day_volume": row["day_volume"],
                "prev_day_dollar_vol": row.get("prev_day_dollar_vol"),
                "total_score": row["total_score"],
                "rank": rank_one,
            })

        if (day_idx + 1) % 20 == 0 or day_idx == 0:
            print(f"  [{day_idx+1}/{total_days}] {date_str}: {len(candidates)} candidates written")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.output, index=False)
    print(f"Wrote {len(out_df)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
