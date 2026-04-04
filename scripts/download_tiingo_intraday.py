#!/usr/bin/env python3
"""Download and cache Tiingo IEX intraday bars locally.

Stores 1-min bars as Parquet files: data/intraday/{SYMBOL}/{YYYY-MM-DD}.parquet
Never re-downloads a file that already exists (idempotent).

Usage:
  # Download specific symbols for a date range
  python scripts/download_tiingo_intraday.py --symbols SPY,AAPL,TSLA --start 2024-01-01 --end 2026-04-01

  # Download from a file of symbols (one per line)
  python scripts/download_tiingo_intraday.py --symbols-file data/watchlist.txt --start 2025-01-01

  # Download a curated list of known small cap gappers
  python scripts/download_tiingo_intraday.py --gappers --start 2025-01-01

  # Download SPY for full available history (backtesting)
  python scripts/download_tiingo_intraday.py --symbols SPY --start 2021-01-01 --freq 5min

  # Nightly mode: download today's scanner hits
  python scripts/download_tiingo_intraday.py --nightly
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATA_DIR = ROOT / "data" / "intraday"
TIINGO_BASE = "https://api.tiingo.com/iex"

# Rate limit: free tier = 500 req/hour, 50 symbols/day
# Be conservative: 1 request per 2 seconds
RATE_LIMIT_SECONDS = 2.0


def _get_api_key() -> str:
    key = os.getenv("TIINGO_API_KEY", "")
    if not key:
        env_file = ROOT / ".env"
        if env_file.is_file():
            for line in open(env_file):
                line = line.strip()
                if line.startswith("TIINGO_API_KEY="):
                    key = line.split("=", 1)[1].strip().strip('"').strip("'")
                    break
    return key


def _cache_path(symbol: str, date_str: str, freq: str = "1min") -> Path:
    """Local cache path: data/intraday/{SYMBOL}/{freq}/{YYYY-MM-DD}.parquet"""
    return DATA_DIR / symbol.upper() / freq / f"{date_str}.parquet"


def _is_cached(symbol: str, date_str: str, freq: str = "1min") -> bool:
    """Check if we already have this data locally."""
    return _cache_path(symbol, date_str, freq).is_file()


def _is_trading_day(d: date) -> bool:
    """Simple weekday check. Doesn't account for holidays."""
    return d.weekday() < 5


def _trading_days(start: date, end: date) -> list[date]:
    """Generate list of weekdays between start and end."""
    days = []
    d = start
    while d <= end:
        if _is_trading_day(d):
            days.append(d)
        d += timedelta(days=1)
    return days


def download_bars(symbol: str, date_str: str, api_key: str,
                  freq: str = "1min") -> pd.DataFrame | None:
    """
    Download intraday bars from Tiingo IEX for one symbol on one date.
    Returns DataFrame or None if no data.
    """
    # Check cache first
    if _is_cached(symbol, date_str, freq):
        return pd.read_parquet(_cache_path(symbol, date_str, freq))

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Token {api_key}",
    }
    params = {
        "startDate": date_str,
        "endDate": date_str,
        "resampleFreq": freq,
    }

    try:
        r = requests.get(f"{TIINGO_BASE}/{symbol}/prices",
                        headers=headers, params=params, timeout=30)
        if r.status_code != 200:
            if r.status_code == 404:
                return None  # symbol not found
            if r.status_code == 429:
                print(f"  Rate limited. Sleeping 60s...", file=sys.stderr)
                time.sleep(60)
                return download_bars(symbol, date_str, api_key, freq)
            print(f"  Error {r.status_code} for {symbol} {date_str}: {r.text[:200]}",
                  file=sys.stderr)
            return None

        data = r.json()
        if not data:
            return None

        df = pd.DataFrame(data)
        df["datetime"] = pd.to_datetime(df["date"])
        df = df.rename(columns={
            "open": "Open", "high": "High", "low": "Low",
            "close": "Close",
        })

        # Volume may or may not be in the response
        if "volume" not in df.columns:
            df["Volume"] = 0
        else:
            df = df.rename(columns={"volume": "Volume"})

        df = df.set_index("datetime")
        df = df[["Open", "High", "Low", "Close", "Volume"]]

        # Save to cache
        cache_path = _cache_path(symbol, date_str, freq)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_path)

        return df

    except requests.RequestException as e:
        print(f"  Request error for {symbol} {date_str}: {e}", file=sys.stderr)
        return None


def download_symbol_range(symbol: str, start: date, end: date,
                          api_key: str, freq: str = "1min") -> dict:
    """Download a date range for one symbol. Returns stats."""
    days = _trading_days(start, min(end, date.today()))
    downloaded = 0
    cached = 0
    empty = 0

    for d in days:
        d_str = d.isoformat()
        if _is_cached(symbol, d_str, freq):
            cached += 1
            continue

        df = download_bars(symbol, d_str, api_key, freq)
        if df is not None and len(df) > 0:
            downloaded += 1
        else:
            empty += 1

        time.sleep(RATE_LIMIT_SECONDS)

    return {
        "symbol": symbol,
        "total_days": len(days),
        "downloaded": downloaded,
        "cached": cached,
        "empty": empty,
    }


def get_known_gappers() -> list[str]:
    """
    List of stocks known to be frequent small cap gappers.
    These are stocks that have historically appeared on gap scanners.
    Expand this list over time as the scanner identifies new names.
    """
    return [
        # Large caps (for ORB/VWAP strategies)
        "SPY", "QQQ", "IWM",
        # Recent popular momentum stocks
        "SMCI", "MSTR", "PLTR", "RKLB", "IONQ", "RGTI", "QUBT",
        "LUNR", "ACHR", "JOBY", "LILM", "RIVN", "LCID",
        # Biotech movers
        "MRNA", "BNTX", "NVAX",
        # Frequently gapping small/mid caps
        "CVNA", "AFRM", "UPST", "SOFI", "HOOD", "COIN",
        "DKNG", "AEHR", "ASTS", "GSAT",
        # Meme / high short interest
        "GME", "AMC", "BBBY", "KOSS",
        # Tech momentum
        "NVDA", "AMD", "TSLA", "META", "AAPL", "MSFT", "AMZN", "GOOGL",
    ]


def main():
    parser = argparse.ArgumentParser(description="Download Tiingo IEX intraday bars")
    parser.add_argument("--symbols", help="Comma-separated symbols (e.g., SPY,AAPL)")
    parser.add_argument("--symbols-file", help="File with one symbol per line")
    parser.add_argument("--gappers", action="store_true",
                       help="Download curated list of known gappers")
    parser.add_argument("--nightly", action="store_true",
                       help="Download today's data for scanner hits")
    parser.add_argument("--start", default="2024-01-01",
                       help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default=None,
                       help="End date (default: today)")
    parser.add_argument("--freq", default="1min",
                       choices=["1min", "5min", "15min", "1hour"],
                       help="Bar frequency")
    args = parser.parse_args()

    api_key = _get_api_key()
    if not api_key:
        print("ERROR: No TIINGO_API_KEY found in .env or environment", file=sys.stderr)
        sys.exit(1)

    # Determine symbols
    symbols = []
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",")]
    elif args.symbols_file:
        with open(args.symbols_file) as f:
            symbols = [line.strip().upper() for line in f if line.strip()]
    elif args.gappers:
        symbols = get_known_gappers()
    elif args.nightly:
        # Try to load today's scanner output
        today_str = date.today().isoformat()
        scanner_file = ROOT / "data" / "daytrader" / f"watchlist_{today_str}.json"
        if scanner_file.is_file():
            with open(scanner_file) as f:
                data = json.load(f)
                symbols = [item["symbol"] for item in data]
        else:
            print("No scanner output found for today. Run the scanner first.", file=sys.stderr)
            print("Or use --gappers for the curated list.", file=sys.stderr)
            sys.exit(1)
    else:
        print("Specify --symbols, --symbols-file, --gappers, or --nightly", file=sys.stderr)
        sys.exit(1)

    start = datetime.strptime(args.start, "%Y-%m-%d").date()
    end = datetime.strptime(args.end, "%Y-%m-%d").date() if args.end else date.today()

    print(f"Downloading {args.freq} bars for {len(symbols)} symbols", file=sys.stderr)
    print(f"Date range: {start} to {end}", file=sys.stderr)
    print(f"Cache dir: {DATA_DIR}", file=sys.stderr)
    print(f"Rate limit: {RATE_LIMIT_SECONDS}s between requests", file=sys.stderr)
    print(file=sys.stderr)

    total_downloaded = 0
    total_cached = 0
    total_empty = 0

    for i, sym in enumerate(symbols):
        print(f"[{i+1}/{len(symbols)}] {sym}...", end="", file=sys.stderr, flush=True)
        stats = download_symbol_range(sym, start, end, api_key, args.freq)
        print(f" {stats['downloaded']} new, {stats['cached']} cached, "
              f"{stats['empty']} empty ({stats['total_days']} days)",
              file=sys.stderr)
        total_downloaded += stats["downloaded"]
        total_cached += stats["cached"]
        total_empty += stats["empty"]

    print(file=sys.stderr)
    print(f"DONE: {total_downloaded} downloaded, {total_cached} already cached, "
          f"{total_empty} empty days", file=sys.stderr)

    # Show disk usage
    total_size = sum(f.stat().st_size for f in DATA_DIR.rglob("*.parquet"))
    print(f"Total cache size: {total_size / 1024 / 1024:.1f} MB", file=sys.stderr)


if __name__ == "__main__":
    main()
