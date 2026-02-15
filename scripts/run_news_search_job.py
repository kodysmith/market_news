#!/usr/bin/env python3
"""
Daily (or on-demand) news search keyed by congressional trades symbols.
Gets symbol list from congressional_trades.fetch_aggregated(), fetches FMP stock news
for those symbols, writes data/news.json in the shape expected by publish_news_to_supabase.py,
then runs that script to push to Supabase.

Usage:
  python scripts/run_news_search_job.py [--no-publish]

Requires:
  - FMP_API_KEY in data/config.json or env
  - For publish: SUPABASE_URL, SUPABASE_SECRET_KEY in .env
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Load .env
_env_file = ROOT / ".env"
if _env_file.exists():
    try:
        with open(_env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, _, v = line.partition("=")
                    v = v.strip().strip("'\"")
                    if k and k not in os.environ:
                        os.environ[k] = v
    except Exception:
        pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

FMP_BASE = "https://financialmodelingprep.com/api/v3"
NEWS_JSON_PATH = ROOT / "data" / "news.json"


def _load_fmp_key() -> str:
    config_path = ROOT / "data" / "config.json"
    if not config_path.exists():
        return os.environ.get("FMP_API_KEY", "")
    try:
        with open(config_path) as f:
            return (json.load(f).get("FMP_API_KEY") or "") or os.environ.get("FMP_API_KEY", "")
    except Exception:
        return os.environ.get("FMP_API_KEY", "")


def _get_congressional_symbols(days: int = 30) -> list[str]:
    """Return list of symbols from congressional trades (last N days)."""
    from QuantEngine.congressional_trades import fetch_aggregated

    result = fetch_aggregated(days=days)
    if result.get("error") and not result.get("symbols"):
        logger.warning("Congressional trades unavailable: %s", result.get("error"))
        return []
    symbols = [s["symbol"] for s in (result.get("symbols") or []) if s.get("symbol")]
    return symbols[:80]  # Cap to avoid rate limits


def _fetch_fmp_stock_news(apikey: str, tickers: list[str], limit_per_batch: int = 15) -> list[dict]:
    """Fetch FMP stock_news for given tickers. Returns list of items with headline, url, source, summary, published_date."""
    import requests

    if not apikey or not tickers:
        return []
    seen_urls = set()
    out = []
    # Batch tickers (FMP accepts comma-separated)
    batch_size = 5
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i : i + batch_size]
        ticker_str = ",".join(batch)
        try:
            url = f"{FMP_BASE}/stock_news"
            params = {"tickers": ticker_str, "limit": limit_per_batch, "apikey": apikey}
            r = requests.get(url, params=params, timeout=15)
            if r.status_code != 200:
                logger.warning("FMP stock_news %s: %s", r.status_code, r.text[:200])
                time.sleep(1)
                continue
            data = r.json()
            items = data if isinstance(data, list) else []
            for a in items:
                url_val = (a.get("url") or "").strip()
                if not url_val or url_val in seen_urls:
                    continue
                seen_urls.add(url_val)
                title = (a.get("title") or "").strip()
                if not title:
                    continue
                pub = a.get("publishedDate") or a.get("date") or ""
                if isinstance(pub, str) and len(pub) >= 10:
                    pub = pub[:10]
                else:
                    pub = None
                source = (a.get("source") or "FMP").strip() or "FMP"
                text = (a.get("text") or "")[:500].strip() if a.get("text") else ""
                out.append({
                    "headline": title,
                    "url": url_val,
                    "source": source,
                    "summary": text,
                    "published_date": pub,
                    "category": "market",
                })
            time.sleep(0.4)
        except Exception as e:
            logger.warning("FMP request failed for %s: %s", ticker_str, e)
            time.sleep(1)
    return out


def run_news_search(days: int = 30, max_symbols: int = 80) -> list[dict]:
    """Get congressional symbols, fetch FMP news, return list of news items (same shape as news.json)."""
    apikey = _load_fmp_key()
    if not apikey:
        logger.error("FMP_API_KEY not set in data/config.json or env")
        return []
    symbols = _get_congressional_symbols(days=days)
    if not symbols:
        symbols = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA"]
        logger.info("No congressional symbols; using default tickers")
    symbols = symbols[:max_symbols]
    logger.info("Fetching FMP news for %d symbols", len(symbols))
    items = _fetch_fmp_stock_news(apikey, symbols)
    logger.info("Collected %d news items", len(items))
    return items


def main() -> int:
    ap = argparse.ArgumentParser(description="Run news search from congressional symbols and publish to Supabase")
    ap.add_argument("--no-publish", action="store_true", help="Do not run publish_news_to_supabase.py")
    ap.add_argument("--days", type=int, default=30, help="Congressional trades lookback days")
    args = ap.parse_args()

    items = run_news_search(days=args.days)
    NEWS_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(NEWS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(items, f, indent=2)
    logger.info("Wrote %s with %d items", NEWS_JSON_PATH, len(items))

    if not args.no_publish and items:
        script = ROOT / "scripts" / "publish_news_to_supabase.py"
        if script.exists():
            try:
                subprocess.run([sys.executable, str(script)], cwd=str(ROOT), check=True)
            except subprocess.CalledProcessError as e:
                logger.warning("Publish script failed: %s", e)
                return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
