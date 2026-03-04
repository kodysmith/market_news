#!/usr/bin/env python3
"""Download Massive.com options minute-aggregates flat files for SPX and build daily Parquet snapshots.

Massive provides daily S3 flat files at us_options_opra/minute_aggs_v1 with columns:
  ticker, volume, open, close, high, low, window_start (nanosecond timestamp), transactions

This script:
- Downloads (or reads from a local directory) daily .csv.gz files for a date range.
- Filters to SPX options (ticker prefix O:SPX).
- Aggregates to end-of-day (last close per contract per day) and maps to our backtest schema.
- Writes data/spx_options/spx_options_YYYY-MM-DD.parquet for each date.

Requires: MASSIVE_API_KEY in data/config.json or env. Base URL for flat files is configurable
(default uses Massive's file browser / S3 path; you may need to obtain the exact download URL
from your plan's File Browser or API docs).
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# OPRA option ticker: O:SPX or O:SPXW + YYMMDD + C/P + 8-char strike
# Monthly = O:SPX250321P00550000, Weekly/0DTE = O:SPXW250321P00550000
OPRA_PATTERN = re.compile(r"^O:SPXW?(\d{6})([CP])(\d{8})$")


def load_config() -> dict:
    path = ROOT / "data" / "config.json"
    if not path.exists():
        return {}
    with open(path, "r") as f:
        return json.load(f)


def parse_opra_ticker(ticker: str) -> tuple[date, str, float] | None:
    """Parse O:SPXYYMMDDC/Pstrike -> (expiration_date, contract_type, strike)."""
    if not isinstance(ticker, str) or not ticker.startswith("O:SPX"):
        return None
    m = OPRA_PATTERN.match(ticker.strip().upper())
    if not m:
        return None
    yy, mm, dd = int(m.group(1)[:2]), int(m.group(1)[2:4]), int(m.group(1)[4:6])
    year = 2000 + yy if yy < 90 else 1900 + yy
    try:
        exp = date(year, mm, dd)
    except ValueError:
        return None
    cp = "call" if m.group(2) == "C" else "put"
    # OPRA strike: 8 digits = strike * 1000 (e.g. 03500000 = 3500, 06000000 = 6000)
    raw = int(m.group(3))
    strike = raw / 1000.0
    return (exp, cp, strike)


def process_minute_agg_chunk(df: pd.DataFrame, snapshot_date: date) -> pd.DataFrame:
    """From minute-agg rows (ticker, close, ...), produce EOD snapshot: one row per contract with last close."""
    if df.empty or "ticker" not in df.columns or "close" not in df.columns:
        return pd.DataFrame()
    # Keep last (by time) close per ticker for this day
    if "window_start" in df.columns:
        df = df.sort_values("window_start")
    last = df.groupby("ticker", as_index=False).last()
    rows = []
    for _, r in last.iterrows():
        parsed = parse_opra_ticker(str(r["ticker"]))
        if parsed is None:
            continue
        exp, contract_type, strike = parsed
        close = r["close"]
        if pd.isna(close) or close <= 0:
            continue
        rows.append({
            "underlying": "SPX",
            "strike": strike,
            "expiration_date": exp,
            "contract_type": contract_type,
            "open_interest": 1,  # flat files have no OI; use 1 so backtest adapter keeps row (use --default-iv)
            "implied_volatility": float("nan"),
            "snapshot_date": snapshot_date,
            "option_symbol": r["ticker"],
            "shares_per_contract": 100,
            "bid": None,
            "ask": None,
            "mid": float(close),
        })
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    out["expiration_date"] = pd.to_datetime(out["expiration_date"]).dt.date
    return out


def read_one_gz(path: Path) -> pd.DataFrame:
    """Read a single .csv.gz file into a DataFrame (tab or comma separated)."""
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
        head = f.read(4096)
    sep = "\t" if "\t" in head else ","
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
        return pd.read_csv(f, sep=sep, low_memory=False)


def fetch_one_day_url(base_url: str, api_key: str, d: date) -> pd.DataFrame | None:
    """Download one day's file from a URL pattern. base_url may contain {date} or {yyyy},{mm},{dd}."""
    url = base_url.replace("{date}", d.isoformat()).replace("{yyyy}", str(d.year)).replace("{mm}", f"{d.month:02d}").replace("{dd}", f"{d.day:02d}")
    if "?" in url:
        url += "&" if url.rstrip().endswith("?") else "&"
    else:
        url += "?"
    url += f"apiKey={api_key}"
    try:
        import requests
        r = requests.get(url, timeout=120, stream=True)
        r.raise_for_status()
        with gzip.GzipFile(fileobj=r.raw) as gz:
            return pd.read_csv(gz, low_memory=False)
    except Exception:
        return None


def date_range(start: date, end: date) -> list[date]:
    out = []
    d = start
    while d <= end:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def run_local(
    input_dir: Path,
    output_dir: Path,
    start: date,
    end: date,
    verbose: bool,
) -> list[str]:
    """Process local .csv.gz files in input_dir; write Parquet to output_dir."""
    written = []
    for d in date_range(start, end):
        # Common patterns: 2025-01-15.csv.gz or 20250115.csv.gz
        for name in (f"{d}.csv.gz", f"{d.isoformat()}.csv.gz", f"{d:%Y%m%d}.csv.gz"):
            p = input_dir / name
            if not p.exists():
                continue
            try:
                df = read_one_gz(p)
            except Exception as e:
                if verbose:
                    print(f"  {d}: skip read error {e}", file=sys.stderr)
                continue
            if df.empty:
                continue
            spx = df[df["ticker"].astype(str).str.upper().str.startswith("O:SPX")] if "ticker" in df.columns else pd.DataFrame()
            if spx.empty:
                if verbose:
                    print(f"  {d}: no SPX rows", file=sys.stderr)
                continue
            out_df = process_minute_agg_chunk(spx, d)
            if out_df.empty:
                continue
            output_dir.mkdir(parents=True, exist_ok=True)
            out_path = output_dir / f"spx_options_{d.isoformat()}.parquet"
            out_df.to_parquet(out_path, index=False)
            written.append(str(out_path))
            if verbose:
                print(f"  {d}: {len(out_df)} contracts -> {out_path}", file=sys.stderr)
            break
    return written


def run_url(
    base_url: str,
    api_key: str,
    output_dir: Path,
    start: date,
    end: date,
    verbose: bool,
) -> list[str]:
    """Download from URL pattern and process each day."""
    written = []
    for d in date_range(start, end):
        df = fetch_one_day_url(base_url, api_key, d)
        if df is None or df.empty:
            if verbose:
                print(f"  {d}: download failed or empty", file=sys.stderr)
            continue
        spx = df[df["ticker"].astype(str).str.upper().str.startswith("O:SPX")] if "ticker" in df.columns else pd.DataFrame()
        if spx.empty:
            continue
        out_df = process_minute_agg_chunk(spx, d)
        if out_df.empty:
            continue
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / f"spx_options_{d.isoformat()}.parquet"
        out_df.to_parquet(out_path, index=False)
        written.append(str(out_path))
        if verbose:
            print(f"  {d}: {len(out_df)} contracts -> {out_path}", file=sys.stderr)
    return written


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build SPX options daily Parquet from Massive minute-aggregates flat files (local or URL).",
    )
    parser.add_argument("--start", default="2025-01-01", metavar="YYYY-MM-DD", help="Start date")
    parser.add_argument("--end", default="2025-12-31", metavar="YYYY-MM-DD", help="End date")
    parser.add_argument("--input-dir", type=Path, default=None, help="Directory containing YYYY-MM-DD.csv.gz (or YYYYMMDD.csv.gz) files")
    parser.add_argument("--base-url", type=str, default=None, help="URL template for download (e.g. https://data.massive.com/.../minute_aggs_v1/{date}.csv.gz)")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "spx_options", help="Output directory for Parquet files")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()
    start = datetime.strptime(args.start, "%Y-%m-%d").date()
    end = datetime.strptime(args.end, "%Y-%m-%d").date()

    if args.input_dir is not None:
        args.input_dir = args.input_dir.resolve()
        if not args.input_dir.is_dir():
            print(f"Error: input directory not found: {args.input_dir}", file=sys.stderr)
            return 1
        written = run_local(args.input_dir, args.output_dir.resolve(), start, end, args.verbose)
    elif args.base_url:
        config = load_config()
        api_key = config.get("MASSIVE_API_KEY") or os.environ.get("MASSIVE_API_KEY")
        if not api_key:
            print("Error: MASSIVE_API_KEY required in data/config.json or env for URL download.", file=sys.stderr)
            return 1
        written = run_url(args.base_url, api_key, args.output_dir.resolve(), start, end, args.verbose)
    else:
        print(
            "Use either --input-dir (directory of .csv.gz files) or --base-url (URL template with {date}).\n"
            "Example: --input-dir /path/to/massive_minute_aggs\n"
            "Example: --base-url 'https://data.massive.com/us_options_opra/minute_aggs_v1/{date}.csv.gz'",
            file=sys.stderr,
        )
        return 1

    if not written:
        print("No files written. Check --start/--end and that input files exist and contain O:SPX tickers.", file=sys.stderr)
        return 0
    for p in written:
        print(p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
