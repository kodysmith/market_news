"""Layer 1: Universe Scanner — find gap-up small cap candidates pre-market.

Scans all US equities via Polygon grouped daily bars, filters by gap %,
float, relative volume, and price range. Outputs ranked candidates.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from daytrader.config import (
    SCAN_MIN_GAP_PCT, SCAN_MIN_RVOL, SCAN_MIN_FLOAT, SCAN_MAX_FLOAT,
    SCAN_MIN_PRICE, SCAN_MAX_PRICE, SCAN_MIN_AVG_VOL,
    SCAN_MIN_PREMARKET_VOL, SCAN_MAX_CANDIDATES, POLYGON_CACHE_DIR,
)

logger = logging.getLogger("daytrader.scanner")

POLYGON_BASE = "https://api.polygon.io"


def _get_api_key() -> str:
    key = os.getenv("POLYGON_API_KEY", "")
    if not key:
        env_file = ROOT / ".env"
        if env_file.is_file():
            for line in open(env_file):
                line = line.strip()
                if line.startswith("POLYGON_API_KEY="):
                    key = line.split("=", 1)[1].strip().strip('"').strip("'")
                    break
    return key


def polygon_get(path: str, params: dict, api_key: str, max_retries: int = 3) -> dict:
    """GET with retry on 429/5xx."""
    params = dict(params)
    params["apiKey"] = api_key
    url = f"{POLYGON_BASE}{path}"
    backoff = 1.0
    for attempt in range(max_retries):
        try:
            r = requests.get(url, params=params, timeout=30)
            if r.status_code == 200:
                return r.json()
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(backoff)
                backoff *= 2
                continue
            logger.warning("Polygon %d: %s", r.status_code, r.text[:200])
            return {}
        except requests.RequestException as e:
            logger.warning("Polygon request error: %s", e)
            time.sleep(backoff)
            backoff *= 2
    return {}


def _cache_path(kind: str, key: str) -> Path:
    safe = key.replace("/", "_")
    d = POLYGON_CACHE_DIR / kind
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{safe}.json"


def _read_cache(path: Path) -> Optional[dict]:
    if not path.is_file():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def _write_cache(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f)


@dataclass
class ScanCandidate:
    """A stock that passed the universe scanner filters."""
    symbol: str
    gap_pct: float
    rel_vol: float
    price: float
    prev_close: float
    day_open: float
    day_volume: int
    avg_volume: float
    shares_outstanding: Optional[float] = None
    float_est: Optional[float] = None
    catalyst_score: int = 0
    catalyst_type: str = ""
    catalyst_headline: str = ""
    pre_market_high: Optional[float] = None
    pre_market_low: Optional[float] = None
    atr_14d: Optional[float] = None
    short_interest_pct: Optional[float] = None
    scan_rank: int = 0


def get_grouped_daily(date_str: str, api_key: str) -> pd.DataFrame:
    """Fetch all US stock bars for a date (Polygon grouped daily)."""
    cache = _cache_path("grouped_daily", date_str)
    data = _read_cache(cache)
    if data is None:
        data = polygon_get(
            f"/v2/aggs/grouped/locale/us/market/stocks/{date_str}",
            {"adjusted": "true"},
            api_key,
        )
        if data:
            _write_cache(cache, data)

    results = data.get("results", [])
    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)
    # Polygon fields: T=ticker, o=open, h=high, l=low, c=close, v=volume, vw=vwap
    df = df.rename(columns={"T": "symbol", "o": "open", "h": "high",
                             "l": "low", "c": "close", "v": "volume", "vw": "vwap"})
    return df


def get_shares_outstanding(symbol: str, api_key: str) -> Optional[float]:
    """Get shares outstanding from Polygon ticker details."""
    cache = _cache_path("ticker", symbol)
    data = _read_cache(cache)
    if data is None:
        data = polygon_get(f"/v3/reference/tickers/{symbol}", {}, api_key)
        if data:
            _write_cache(cache, data)
    if not data:
        return None
    res = data.get("results", {})
    for key in ("share_class_shares_outstanding", "weighted_shares_outstanding"):
        val = res.get(key)
        if isinstance(val, (int, float)) and val > 0:
            return float(val)
    return None


def scan_for_candidates(
    today_str: str,
    prev_day_str: str,
    api_key: str,
    lookback_dates: list[str] | None = None,
) -> list[ScanCandidate]:
    """
    Run the full universe scan for a given date.

    Returns ranked list of ScanCandidates that pass all filters.
    """
    logger.info("Scanning universe for %s (prev: %s)", today_str, prev_day_str)

    today_bars = get_grouped_daily(today_str, api_key)
    prev_bars = get_grouped_daily(prev_day_str, api_key)

    if today_bars.empty or prev_bars.empty:
        logger.warning("No bar data for %s or %s", today_str, prev_day_str)
        return []

    # Merge today's open with prev close
    prev_close = prev_bars[["symbol", "close"]].rename(columns={"close": "prev_close"})
    today_data = today_bars[["symbol", "open", "high", "low", "close", "volume"]].rename(
        columns={"open": "day_open", "volume": "day_volume"}
    )
    merged = today_data.merge(prev_close, on="symbol", how="inner")
    merged = merged[merged["prev_close"] > 0]

    # Gap %
    merged["gap_pct"] = (merged["day_open"] - merged["prev_close"]) / merged["prev_close"]

    # Price filter
    merged = merged[
        (merged["day_open"] >= SCAN_MIN_PRICE) &
        (merged["day_open"] <= SCAN_MAX_PRICE) &
        (merged["gap_pct"] >= SCAN_MIN_GAP_PCT)
    ]
    logger.info("After gap + price filter: %d stocks", len(merged))

    if merged.empty:
        return []

    # Volume filter (today's volume as proxy for pre-market interest)
    merged = merged[merged["day_volume"] >= SCAN_MIN_PREMARKET_VOL]
    logger.info("After pre-market volume filter: %d stocks", len(merged))

    # Relative volume (need lookback data)
    if lookback_dates:
        relvol_data = {}
        for sym in merged["symbol"].unique():
            hist_vols = []
            for past_str in lookback_dates[-20:]:
                past_bars = get_grouped_daily(past_str, api_key)
                if past_bars.empty:
                    continue
                row = past_bars[past_bars["symbol"] == sym]
                if not row.empty:
                    v = float(row["volume"].iloc[0])
                    if v > 0:
                        hist_vols.append(v)
            if len(hist_vols) >= 5:
                avg_vol = sum(hist_vols) / len(hist_vols)
                relvol_data[sym] = {
                    "rel_vol": float(merged.loc[merged["symbol"] == sym, "day_volume"].iloc[0]) / avg_vol,
                    "avg_volume": avg_vol,
                }

        merged["rel_vol"] = merged["symbol"].map(lambda s: relvol_data.get(s, {}).get("rel_vol", 0))
        merged["avg_volume"] = merged["symbol"].map(lambda s: relvol_data.get(s, {}).get("avg_volume", 0))
        merged = merged[merged["rel_vol"] >= SCAN_MIN_RVOL]
        merged = merged[merged["avg_volume"] >= SCAN_MIN_AVG_VOL]
    else:
        # Without lookback, use absolute volume as proxy
        merged["rel_vol"] = 0.0
        merged["avg_volume"] = 0.0
        merged = merged[merged["day_volume"] >= SCAN_MIN_AVG_VOL * 3]

    logger.info("After RVOL + avg volume filter: %d stocks", len(merged))

    # Float filter (requires per-ticker API call — rate limit)
    candidates = []
    for _, row in merged.nlargest(SCAN_MAX_CANDIDATES * 2, "gap_pct").iterrows():
        shares = get_shares_outstanding(row["symbol"], api_key)
        if shares is None:
            continue
        if shares < SCAN_MIN_FLOAT or shares > SCAN_MAX_FLOAT:
            continue

        candidates.append(ScanCandidate(
            symbol=row["symbol"],
            gap_pct=round(float(row["gap_pct"]), 4),
            rel_vol=round(float(row.get("rel_vol", 0)), 2),
            price=round(float(row["day_open"]), 2),
            prev_close=round(float(row["prev_close"]), 2),
            day_open=round(float(row["day_open"]), 2),
            day_volume=int(row["day_volume"]),
            avg_volume=float(row.get("avg_volume", 0)),
            shares_outstanding=shares,
            float_est=shares,  # proxy — true float requires insider data
        ))

        if len(candidates) >= SCAN_MAX_CANDIDATES:
            break
        time.sleep(0.2)  # rate limit for ticker details

    # Rank by gap_pct * rel_vol (momentum score)
    candidates.sort(key=lambda c: c.gap_pct * max(c.rel_vol, 1), reverse=True)
    for i, c in enumerate(candidates):
        c.scan_rank = i + 1

    logger.info("Final candidates: %d", len(candidates))
    return candidates
