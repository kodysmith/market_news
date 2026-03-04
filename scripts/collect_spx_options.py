#!/usr/bin/env python3
"""Collect SPX options data for the last 30–90 days.

Supports two sources:
- massive: Current snapshot from Massive API (I:SPX). Run daily to build history.
- tradier: By-expiration chains from Tradier; expirations in [today - days, today + 30].

Output: Parquet (and optional CSV) under --output-dir with normalized schema
for GEX/backtest (underlying, strike, expiration_date, contract_type, open_interest,
implied_volatility, snapshot_date, bid, ask, mid, etc.).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

# Repo root (parent of scripts/)
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Normalized column names (GEX/backtest compatible); bid/ask/mid for realistic backtest pricing
NORMALIZED_COLUMNS = [
    "underlying",
    "strike",
    "expiration_date",
    "contract_type",
    "open_interest",
    "implied_volatility",
    "snapshot_date",
    "option_symbol",
    "shares_per_contract",
    "bid",
    "ask",
    "mid",
]


def load_config() -> dict:
    """Load data/config.json for MASSIVE_API_KEY and optional TRADIER_API_KEY."""
    path = ROOT / "data" / "config.json"
    if not path.exists():
        return {}
    with open(path, "r") as f:
        return json.load(f)


def fetch_massive_snapshot(
    api_key: str,
    limit: int = 250,
    days_out: int = 65,
    max_pages: int = 200,
    page_pause_sec: float = 0.05,
) -> dict:
    """Fetch SPX options snapshot from Massive API (I:SPX), including pagination."""
    url = "https://api.massive.com/v3/snapshot/options/I:SPX"
    today = datetime.now().strftime("%Y-%m-%d")
    end_date = (datetime.now() + timedelta(days=days_out)).strftime("%Y-%m-%d")
    params = {
        "apiKey": api_key,
        "limit": limit,
        "expiration_date.gte": today,
        "expiration_date.lte": end_date,
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    payload = r.json()

    all_results = list(payload.get("results", []))
    next_url = payload.get("next_url")
    pages = 1
    while next_url and pages < max_pages:
        if next_url.startswith("/"):
            next_url = f"https://api.massive.com{next_url}"
        next_params = None if "apiKey=" in next_url else {"apiKey": api_key}
        r_next = requests.get(next_url, params=next_params, timeout=30)
        r_next.raise_for_status()
        next_payload = r_next.json()
        all_results.extend(next_payload.get("results", []))
        next_url = next_payload.get("next_url")
        pages += 1
        if page_pause_sec > 0:
            time.sleep(page_pause_sec)

    payload["results"] = all_results
    payload["pages_fetched"] = pages
    return payload


def _quote_mid(bid: float | None, ask: float | None, midpoint: float | None) -> float | None:
    if midpoint is not None and not (isinstance(midpoint, float) and (midpoint != midpoint)):
        return float(midpoint)
    if bid is not None and ask is not None:
        try:
            return (float(bid) + float(ask)) / 2.0
        except (TypeError, ValueError):
            pass
    return None


def normalize_massive_to_df(snap: dict, snapshot_date: str) -> pd.DataFrame:
    """Convert Massive API response to normalized DataFrame. Includes bid/ask/mid from last_quote when available."""
    rows = []
    for item in snap.get("results", []):
        d = item.get("details") or {}
        strike = d.get("strike_price")
        exp_str = d.get("expiration_date")
        contract_type = (d.get("contract_type") or "").lower()
        if contract_type not in ("call", "put") or strike is None or exp_str is None:
            continue
        try:
            strike = float(strike)
        except (ValueError, TypeError):
            continue
        oi = item.get("open_interest")
        oi = int(oi) if isinstance(oi, (int, float)) and oi == oi else 0
        iv = item.get("implied_volatility")
        if iv is None:
            continue
        try:
            iv = float(iv)
        except (ValueError, TypeError):
            continue
        mult = d.get("shares_per_contract", 100)
        mult = int(mult) if isinstance(mult, (int, float)) else 100
        q = item.get("last_quote") or {}
        bid = q.get("bid")
        ask = q.get("ask")
        midpoint = q.get("midpoint")
        bid_f = float(bid) if bid is not None and not (isinstance(bid, float) and (bid != bid)) else None
        ask_f = float(ask) if ask is not None and not (isinstance(ask, float) and (ask != ask)) else None
        mid_f = _quote_mid(bid_f, ask_f, float(midpoint) if midpoint is not None else None)
        rows.append({
            "underlying": "SPX",
            "strike": strike,
            "expiration_date": exp_str,
            "contract_type": contract_type,
            "open_interest": oi,
            "implied_volatility": iv,
            "snapshot_date": snapshot_date,
            "option_symbol": item.get("symbol"),
            "shares_per_contract": mult,
            "bid": bid_f,
            "ask": ask_f,
            "mid": mid_f,
        })
    if not rows:
        return pd.DataFrame(columns=NORMALIZED_COLUMNS)
    df = pd.DataFrame(rows)
    df["expiration_date"] = pd.to_datetime(df["expiration_date"]).dt.date
    return df


def get_tradier_expirations(api_key: str, symbol: str, days_back: int, days_forward: int, no_expired: bool) -> list:
    """Get SPX expiration dates from Tradier in [today - days_back, today + days_forward]."""
    url = "https://api.tradier.com/v1/markets/options/expirations"
    headers = {"Authorization": f"Bearer {api_key}", "Accept": "application/json"}
    params = {"symbol": symbol}
    r = requests.get(url, headers=headers, params=params, timeout=15)
    r.raise_for_status()
    data = r.json()
    if "expirations" not in data or "date" not in data["expirations"]:
        return []
    dates_raw = data["expirations"]["date"]
    if isinstance(dates_raw, str):
        dates_raw = [dates_raw]
    today = datetime.now().date()
    start = today - timedelta(days=days_back)
    end = today + timedelta(days=days_forward)
    out = []
    for date_str in dates_raw:
        try:
            exp_date = datetime.strptime(date_str, "%Y-%m-%d").date()
        except (ValueError, TypeError):
            continue
        if no_expired and exp_date < today:
            continue
        if start <= exp_date <= end:
            out.append(date_str)
    return sorted(out)


def fetch_tradier_chain(api_key: str, symbol: str, expiration: str) -> pd.DataFrame:
    """Fetch options chain for one expiration from Tradier."""
    url = "https://api.tradier.com/v1/markets/options/chains"
    headers = {"Authorization": f"Bearer {api_key}", "Accept": "application/json"}
    params = {"symbol": symbol, "expiration": expiration, "greeks": "true"}
    r = requests.get(url, headers=headers, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    if "options" not in data or "option" not in data["options"]:
        return pd.DataFrame()
    raw = data["options"]["option"]
    if isinstance(raw, dict):
        raw = [raw]
    rows = []
    snapshot_date = datetime.now().strftime("%Y-%m-%d")
    for item in raw:
        strike = item.get("strike")
        exp_date = item.get("expiration_date")
        option_type = (item.get("option_type") or item.get("type") or "").lower()
        if option_type in ("c", "call"):
            contract_type = "call"
        elif option_type in ("p", "put"):
            contract_type = "put"
        else:
            continue
        if strike is None or exp_date is None:
            continue
        try:
            strike = float(strike)
        except (ValueError, TypeError):
            continue
        oi = item.get("open_interest")
        oi = int(oi) if isinstance(oi, (int, float)) and oi == oi else 0
        iv = item.get("implied_volatility")
        if iv is not None:
            try:
                iv = float(iv)
            except (ValueError, TypeError):
                iv = None
        bid = item.get("bid")
        ask = item.get("ask")
        bid_f = float(bid) if bid is not None and not (isinstance(bid, float) and (bid != bid)) else None
        ask_f = float(ask) if ask is not None and not (isinstance(ask, float) and (ask != ask)) else None
        mid_f = _quote_mid(bid_f, ask_f, None)
        rows.append({
            "underlying": "SPX",
            "strike": strike,
            "expiration_date": exp_date,
            "contract_type": contract_type,
            "open_interest": oi,
            "implied_volatility": iv if iv is not None else float("nan"),
            "snapshot_date": snapshot_date,
            "option_symbol": item.get("symbol"),
            "shares_per_contract": 100,
            "bid": bid_f,
            "ask": ask_f,
            "mid": mid_f,
        })
    if not rows:
        return pd.DataFrame(columns=NORMALIZED_COLUMNS)
    df = pd.DataFrame(rows)
    df["expiration_date"] = pd.to_datetime(df["expiration_date"]).dt.date
    return df


def run_massive(api_key: str, output_dir: Path, snapshot_date: str, fmt: str, days_out: int = 65) -> list:
    """Fetch current SPX snapshot from Massive and write Parquet/CSV."""
    snap = fetch_massive_snapshot(api_key, days_out=days_out)
    df = normalize_massive_to_df(snap, snapshot_date)
    if df.empty:
        return []
    output_dir.mkdir(parents=True, exist_ok=True)
    base = output_dir / f"spx_options_{snapshot_date}"
    written = []
    if fmt in ("parquet", "both"):
        path = base.with_suffix(".parquet")
        df.to_parquet(path, index=False)
        written.append(str(path))
    if fmt in ("csv", "both"):
        path = base.with_suffix(".csv")
        df.to_csv(path, index=False)
        written.append(str(path))
    return written


def run_tradier(api_key: str, output_dir: Path, days: int, no_expired: bool, snapshot_date: str, fmt: str) -> list:
    """Fetch SPX chains by expiration from Tradier and write single combined Parquet/CSV."""
    expirations = get_tradier_expirations(api_key, "SPX", days_back=days, days_forward=30, no_expired=no_expired)
    if not expirations:
        return []
    all_dfs = []
    for exp in expirations:
        df = fetch_tradier_chain(api_key, "SPX", exp)
        if not df.empty:
            all_dfs.append(df)
        time.sleep(0.15)
    if not all_dfs:
        return []
    combined = pd.concat(all_dfs, ignore_index=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    base = output_dir / f"spx_options_expirations_{snapshot_date}"
    written = []
    if fmt in ("parquet", "both"):
        path = base.with_suffix(".parquet")
        combined.to_parquet(path, index=False)
        written.append(str(path))
    if fmt in ("csv", "both"):
        path = base.with_suffix(".csv")
        combined.to_csv(path, index=False)
        written.append(str(path))
    return written


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Collect SPX options data for the last 30–90 days (Massive snapshot or Tradier by-expiration)."
    )
    parser.add_argument(
        "--days",
        type=int,
        default=90,
        choices=[30, 90],
        help="Lookback days; for Tradier defines expiration window [today-days, today+30] (default 90)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data" / "spx_options",
        help="Output directory for Parquet/CSV (default data/spx_options)",
    )
    parser.add_argument(
        "--source",
        choices=["massive", "tradier"],
        default="massive",
        help="Data source: massive (current snapshot) or tradier (by expiration) (default massive)",
    )
    parser.add_argument(
        "--format",
        choices=["parquet", "csv", "both"],
        default="parquet",
        help="Output format (default parquet)",
    )
    parser.add_argument(
        "--no-expired",
        action="store_true",
        help="Tradier only: exclude past expirations (only future expirations in window)",
    )
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = ROOT / "data" / "spx_options"
    args.output_dir = args.output_dir.resolve()

    config = load_config()
    snapshot_date = datetime.now().strftime("%Y-%m-%d")

    if args.source == "massive":
        api_key = config.get("MASSIVE_API_KEY") or os.environ.get("MASSIVE_API_KEY")
        if not api_key:
            print("Error: MASSIVE_API_KEY required. Set in data/config.json or MASSIVE_API_KEY env.", file=sys.stderr)
            return 1
        written = run_massive(api_key, args.output_dir, snapshot_date, args.format, days_out=args.days)
    else:
        api_key = config.get("TRADIER_API_KEY") or os.environ.get("TRADIER_API_KEY")
        if not api_key:
            print("Error: TRADIER_API_KEY required. Set in data/config.json or TRADIER_API_KEY env.", file=sys.stderr)
            return 1
        written = run_tradier(api_key, args.output_dir, args.days, args.no_expired, snapshot_date, args.format)

    if not written:
        print("No data written.", file=sys.stderr)
        return 1
    for p in written:
        print(p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
