#!/usr/bin/env python3
"""
Publish trade ideas to Supabase when SPX (or core symbols) meet entry criteria.

Runs on a schedule (cron, same machine as GEX precompute). Calls the local
trade-ideas API, then upserts each idea into trade_ideas_published with
valid_date = today (ET). The app reads from this table by date (today/yesterday)
and shows "Valid: YYYY-MM-DD".

Usage:
  python scripts/publish_trade_ideas.py

Requires:
  - SUPABASE_URL and SUPABASE_SECRET_KEY (or SUPABASE_SERVICE_ROLE_KEY)
  - API_BASE_URL (default http://localhost:5000) — API must be reachable from this host
  - Optional: API_SECRET_KEY for x-api-key header
  - Optional: data/config.json GEX_CORE_SYMBOLS or default [SPX, XSP, SPY, NDX]

Run migration 002_trade_ideas_published.sql in Supabase SQL editor once.
Schedule like GEX (e.g. every 5 min during market hours):
  */5 * * * * cd /path/to/repo && . venv/bin/activate && python3 scripts/publish_trade_ideas.py
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Load .env from repo root
_env_file = ROOT / ".env"
if _env_file.exists():
    try:
        with open(_env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, _, v = line.partition("=")
                    v = v.strip()
                    if (v.startswith("'") and v.endswith("'")) or (v.startswith('"') and v.endswith('"')):
                        v = v[1:-1]
                    if k and k not in os.environ:
                        os.environ[k] = v
    except Exception:
        pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_CORE_SYMBOLS = ["SPX", "XSP", "SPY", "NDX"]
DEFAULT_API_BASE_URL = "http://localhost:5000"


def get_today_et() -> str:
    """Return today's date in ET as YYYY-MM-DD."""
    try:
        from zoneinfo import ZoneInfo
        tz = ZoneInfo("America/New_York")
    except ImportError:
        try:
            import pytz
            tz = pytz.timezone("America/New_York")
        except ImportError:
            return datetime.utcnow().strftime("%Y-%m-%d")
    return datetime.now(tz).strftime("%Y-%m-%d")


def get_now_iso_et() -> str:
    """Return current time in ET as ISO string for as_of_et."""
    try:
        from zoneinfo import ZoneInfo
        tz = ZoneInfo("America/New_York")
    except ImportError:
        try:
            import pytz
            tz = pytz.timezone("America/New_York")
        except ImportError:
            return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    return datetime.now(tz).strftime("%Y-%m-%dT%H:%M:%S")


def get_core_symbols() -> list[str]:
    """Load GEX_CORE_SYMBOLS from data/config.json or return default."""
    config_path = ROOT / "data" / "config.json"
    if config_path.exists():
        try:
            with open(config_path) as f:
                config = json.load(f)
            symbols = config.get("GEX_CORE_SYMBOLS")
            if symbols and isinstance(symbols, list):
                return [str(s).strip().upper() for s in symbols if s]
        except Exception as e:
            logger.warning("Could not load GEX_CORE_SYMBOLS: %s", e)
    return list(DEFAULT_CORE_SYMBOLS)


def get_api_base_url() -> str:
    url = os.environ.get("API_BASE_URL", "").strip().rstrip("/")
    if url:
        return url
    config_path = ROOT / "data" / "config.json"
    if config_path.exists():
        try:
            with open(config_path) as f:
                config = json.load(f)
            url = (config.get("API_BASE_URL") or "").strip().rstrip("/")
            if url:
                return url
        except Exception:
            pass
    return DEFAULT_API_BASE_URL


def get_api_key() -> str:
    key = os.environ.get("API_SECRET_KEY", "").strip()
    if key:
        return key
    config_path = ROOT / "data" / "config.json"
    if config_path.exists():
        try:
            with open(config_path) as f:
                config = json.load(f)
            return (config.get("API_SECRET_KEY") or "").strip()
        except Exception:
            pass
    return ""


def fetch_from_api(base_url: str, path: str, api_key: str) -> dict | None:
    """GET base_url + path; return JSON dict or None on error."""
    url = f"{base_url.rstrip('/')}{path}"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["x-api-key"] = api_key
    try:
        req = Request(url, headers=headers, method="GET")
        with urlopen(req, timeout=120) as resp:
            data = resp.read().decode()
            if not data:
                return None
            out = json.loads(data)
            return out if isinstance(out, dict) else None
    except (HTTPError, URLError, json.JSONDecodeError, OSError) as e:
        logger.warning("API %s: %s", url, e)
        return None


def flatten_ideas(response: dict, symbol: str) -> list[dict]:
    """Extract all trade ideas from API response (unlocked or unlockedByTimeframe)."""
    ideas = []
    if response.get("unlocked") is not None:
        u = response["unlocked"]
        if isinstance(u, list):
            ideas.extend(u)
        elif isinstance(u, dict):
            for v in u.values():
                if isinstance(v, list):
                    ideas.extend(v)
    if response.get("ideas") is not None and isinstance(response["ideas"], list):
        ideas.extend(response["ideas"])
    for idea in ideas:
        if not isinstance(idea, dict):
            continue
        if "id" not in idea and "symbol" not in idea:
            idea["symbol"] = symbol
    return ideas


def get_supabase_client():
    url = (os.environ.get("SUPABASE_URL") or "").strip().rstrip("/")
    key = (os.environ.get("SUPABASE_SECRET_KEY") or os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or "").strip()
    if not url or not key:
        return None
    try:
        from supabase import create_client
        return create_client(url, key)
    except ImportError:
        logger.warning("supabase not installed; pip install supabase")
        return None


def main() -> int:
    supabase = get_supabase_client()
    if not supabase:
        logger.error("SUPABASE_URL and SUPABASE_SECRET_KEY (or SUPABASE_SERVICE_ROLE_KEY) required")
        return 1

    base_url = get_api_base_url()
    api_key = get_api_key()
    symbols = get_core_symbols()
    valid_date = get_today_et()
    as_of_et = get_now_iso_et()

    rows = []
    for symbol in symbols:
        path = f"/trade-ideas/allowed?ticker={symbol}&max_ideas=5&timeframe=all"
        result = fetch_from_api(base_url, path, api_key)
        if result is None:
            logger.warning("Trade ideas %s: no response from API", symbol)
            continue
        if result.get("error"):
            logger.warning("Trade ideas %s: API error %s", symbol, result.get("error"))
            continue
        ideas = flatten_ideas(result, symbol)
        for idea in ideas:
            if not isinstance(idea, dict):
                continue
            idea_id = idea.get("id") or f"{symbol}_{len(rows)}"
            rows.append({
                "valid_date": valid_date,
                "as_of_et": as_of_et,
                "symbol": symbol,
                "idea_id": str(idea_id)[:128],
                "payload": idea,
            })
        if ideas:
            logger.info("Trade ideas %s: %d idea(s)", symbol, len(ideas))

    if not rows:
        logger.info("No ideas to publish (valid_date=%s)", valid_date)
        return 0

    try:
        supabase.table("trade_ideas_published").upsert(
            rows,
            on_conflict="valid_date,symbol,idea_id",
        ).execute()
        logger.info("Upserted %d row(s) to trade_ideas_published (valid_date=%s)", len(rows), valid_date)
    except Exception as e:
        logger.exception("Supabase upsert failed: %s", e)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
