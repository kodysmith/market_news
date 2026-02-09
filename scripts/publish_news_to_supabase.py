#!/usr/bin/env python3
"""
Publish news feed to Supabase so the app can read from DB (no API required) and data is available for backtesting.
Reads from data/news.json or fetches from API news.json; upserts into news_feed table.

Usage:
  python scripts/publish_news_to_supabase.py

Requires:
  - SUPABASE_URL and SUPABASE_SECRET_KEY in .env (same as precompute).
  - Run scripts/news_feed_schema.sql once in Supabase SQL editor.

Data source (first that works):
  1. data/news.json (if exists)
  2. GET {API_BASE_URL}/news.json (if API_BASE_URL set and API responds)
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Load repo root .env
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def get_supabase_client():
    """Return Supabase client if SUPABASE_URL and SUPABASE_SECRET_KEY are set."""
    url = (os.environ.get("SUPABASE_URL") or "").strip().rstrip("/")
    key = (os.environ.get("SUPABASE_SECRET_KEY") or os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or "").strip()
    if not url or not key:
        return None
    try:
        from supabase import create_client
        return create_client(url, key)
    except ImportError:
        logger.warning("supabase package not installed; pip install supabase")
        return None


def load_news_from_file() -> list[dict] | None:
    """Load news from data/news.json. Returns list of items or None."""
    for path in [ROOT / "data" / "news.json", ROOT / "news.json"]:
        if path.exists():
            try:
                with open(path) as f:
                    data = json.load(f)
                if isinstance(data, list):
                    return data
                if isinstance(data, dict) and "articles" in data:
                    return data["articles"]
                return []
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning("Could not load %s: %s", path, e)
    return None


def fetch_news_from_api(base_url: str) -> list[dict] | None:
    """GET {base_url}/news.json and return list of items."""
    url = f"{base_url.rstrip('/')}/news.json"
    try:
        req = Request(url, headers={"Accept": "application/json"}, method="GET")
        with urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())
        if isinstance(data, list):
            return data
        if isinstance(data, dict) and "articles" in data:
            return data["articles"]
        return []
    except (HTTPError, URLError, json.JSONDecodeError, OSError) as e:
        logger.warning("API %s: %s", url, e)
        return None


def news_items_to_rows(items: list[dict]) -> list[dict]:
    """Convert news.json-style items to news_feed row dicts (headline, source, url, summary, published_date, category)."""
    rows = []
    for item in items:
        if not isinstance(item, dict):
            continue
        headline = (item.get("headline") or item.get("title") or "").strip()
        if not headline:
            continue
        published = item.get("published_date") or item.get("published") or item.get("date")
        if isinstance(published, str) and len(published) >= 10:
            try:
                published = published[:10]  # YYYY-MM-DD
            except Exception:
                published = None
        else:
            published = None
        rows.append({
            "headline": headline,
            "source": (item.get("source") or "").strip() or None,
            "url": (item.get("url") or "").strip() or None,
            "summary": (item.get("summary") or "").strip() or None,
            "published_date": published,
            "category": (item.get("category") or "").strip() or None,
        })
    return rows


def main() -> int:
    # Only run during market window (8:30–17:00 ET, Mon–Fri)
    try:
        import sys
        sys.path.insert(0, str(ROOT / "scripts"))
        from market_window import should_run_scheduled_publish
        if not should_run_scheduled_publish(check_queue=False):
            logger.info("Outside market window (8:30–17:00 ET, Mon–Fri); skipping news publish.")
            return 0
    except ImportError:
        pass

    supabase = get_supabase_client()
    if not supabase:
        logger.error("Set SUPABASE_URL and SUPABASE_SECRET_KEY in .env. Run news_feed_schema.sql in Supabase SQL editor.")
        return 1

    # Load news: file first, then API
    items = load_news_from_file()
    if items is None:
        api_url = os.environ.get("API_BASE_URL", "").strip().rstrip("/") or "http://localhost:5000"
        items = fetch_news_from_api(api_url)
    if items is None:
        items = []

    rows = news_items_to_rows(items)
    if not rows:
        logger.warning("No news items to publish. Add data/news.json or run API that serves /news.json.")
        return 0

    try:
        supabase.table("news_feed").insert(rows).execute()
        logger.info("Published %s news items to Supabase news_feed.", len(rows))
    except Exception as e:
        logger.exception("Supabase publish failed: %s", e)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
