"""
Congressional (House) trades aggregation from FMP house-latest API.
Used by the compute worker and by the news search job; results are pushed to Supabase.
"""

from __future__ import annotations

import json
import logging
import os
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import requests

logger = logging.getLogger(__name__)

# Sentinel symbol for cache/queue (result is not per-symbol)
CONGRESSIONAL_SENTINEL_SYMBOL = "*"

# FMP house-latest base URL (stable API)
FMP_HOUSE_LATEST_URL = "https://financialmodelingprep.com/stable/house-latest"

# Amount range to approximate midpoint (for volume score). Order matters: check longer first.
_AMOUNT_MIDPOINTS = [
    ("$10,000,001 - $25,000,000", 17_500_000),
    ("$5,000,001 - $10,000,000", 7_500_000),
    ("$1,000,001 - $5,000,000", 3_000_000),
    ("$500,001 - $1,000,000", 750_000),
    ("$250,001 - $500,000", 375_000),
    ("$100,001 - $250,000", 175_000),
    ("$50,001 - $100,000", 75_000),
    ("$15,001 - $50,000", 32_500),
    ("$1,001 - $15,000", 8_000),
    ("$0 - $1,000", 500),
]


def _parse_amount_to_score(amount: str | None) -> float:
    """Convert FMP amount string to a numeric score for ranking. Returns 0 if unparseable."""
    if not amount or not isinstance(amount, str):
        return 0.0
    amount = amount.strip()
    for pattern, mid in _AMOUNT_MIDPOINTS:
        if pattern in amount or amount == pattern:
            return float(mid)
    # Try parsing explicit number e.g. "$360.00"
    if amount.startswith("$"):
        amount = amount[1:].replace(",", "").strip()
        try:
            return float(amount)
        except ValueError:
            pass
    return 0.0


def _load_fmp_key() -> str:
    """Load FMP_API_KEY from data/config.json (same as cockpit)."""
    root = Path(__file__).resolve().parent.parent
    config_path = root / "data" / "config.json"
    if not config_path.exists():
        return os.environ.get("FMP_API_KEY", "")
    try:
        with open(config_path) as f:
            config = json.load(f)
        return config.get("FMP_API_KEY", "") or os.environ.get("FMP_API_KEY", "")
    except Exception as e:
        logger.warning("Could not load config for FMP key: %s", e)
        return os.environ.get("FMP_API_KEY", "")


def fetch_house_latest_page(apikey: str, page: int = 0, limit: int = 100) -> list[dict[str, Any]]:
    """Fetch one page of house-latest. Returns list of disclosure objects."""
    if not apikey:
        return []
    url = FMP_HOUSE_LATEST_URL
    params = {"page": page, "limit": limit, "apikey": apikey}
    try:
        resp = requests.get(url, params=params, timeout=15)
        if resp.status_code != 200:
            logger.warning("FMP house-latest status %s: %s", resp.status_code, resp.text[:200])
            return []
        data = resp.json()
        if isinstance(data, list):
            return data
        if isinstance(data, dict) and "content" in data:
            return data.get("content", [])
        if isinstance(data, dict) and "data" in data:
            return data.get("data", [])
        return []
    except Exception as e:
        logger.exception("FMP house-latest request failed: %s", e)
        return []


def fetch_aggregated(days: int = 30, apikey: str | None = None) -> dict[str, Any]:
    """
    Fetch house-latest (paginate as needed), filter to last `days` days by transactionDate,
    aggregate by symbol: buyCount, sellCount, repCount, volumeScore, and rank.

    Returns a payload suitable for compute_result_cache and the app:
    {
      "symbols": [ { "symbol", "buyCount", "sellCount", "repCount", "volumeScore", "transactionCount" }, ... ],
      "updatedAt": "ISO8601",
      "days": 30
    }
    """
    key = apikey or _load_fmp_key()
    if not key:
        return {"error": "FMP_API_KEY not configured", "symbols": [], "days": days}

    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).date()
    all_items: list[dict[str, Any]] = []
    page = 0
    limit = 100
    while True:
        chunk = fetch_house_latest_page(key, page=page, limit=limit)
        if not chunk:
            break
        for item in chunk:
            td = item.get("transactionDate")
            if not td:
                continue
            if isinstance(td, str):
                try:
                    # "2026-01-16" or similar
                    d = datetime.strptime(td.split("T")[0], "%Y-%m-%d").date()
                except ValueError:
                    continue
            else:
                continue
            if d >= cutoff:
                all_items.append(item)
        if len(chunk) < limit:
            break
        page += 1
        if page > 20:
            break

    # Aggregate by symbol
    by_symbol: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "buyCount": 0,
            "sellCount": 0,
            "reps": set(),
            "volumeScore": 0.0,
            "transactionCount": 0,
        }
    )
    for item in all_items:
        sym = (item.get("symbol") or "").strip().upper()
        if not sym:
            continue
        rec = by_symbol[sym]
        rec["transactionCount"] += 1
        rec["volumeScore"] += _parse_amount_to_score(item.get("amount"))
        rep_key = (item.get("firstName") or "", item.get("lastName") or "", item.get("office") or "")
        rec["reps"].add(rep_key)
        t = (item.get("type") or "").lower()
        if "purchase" in t or "buy" in t:
            rec["buyCount"] += 1
        elif "sale" in t or "sell" in t:
            rec["sellCount"] += 1

    # Build sorted list: rank by rep count then volume score
    symbols_list = []
    for sym, rec in by_symbol.items():
        symbols_list.append({
            "symbol": sym,
            "buyCount": rec["buyCount"],
            "sellCount": rec["sellCount"],
            "repCount": len(rec["reps"]),
            "volumeScore": round(rec["volumeScore"], 0),
            "transactionCount": rec["transactionCount"],
        })
    symbols_list.sort(key=lambda x: (-x["repCount"], -x["volumeScore"]))

    return {
        "symbols": symbols_list,
        "updatedAt": datetime.now(timezone.utc).isoformat(),
        "days": days,
    }
