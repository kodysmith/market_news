"""
Daily market data job: Yahoo OHLCV → fisher_market_bar_daily, peer_set by sector.
"""

from __future__ import annotations

import json
import logging
import time
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any

import yfinance as yf

from fisher.config import get_sp500_constituents
from fisher.db import get_connection, ensure_company

logger = logging.getLogger(__name__)

# Default lookback for daily bars (e.g. 1 year)
DEFAULT_DAYS = 365
RATE_LIMIT_SEC = 0.2


def fetch_daily_bars(ticker: str, days: int = DEFAULT_DAYS) -> list[tuple[str, float | None, float | None, float | None, float, int | None]]:
    """
    Fetch daily OHLCV from Yahoo. Returns list of (date_str, open, high, low, close, volume).
    """
    try:
        t = yf.Ticker(ticker)
        end = datetime.utcnow()
        start = end - timedelta(days=days)
        hist = t.history(start=start, end=end, auto_adjust=True)
        if hist is None or hist.empty:
            return []
        out = []
        for dt, row in hist.iterrows():
            date_str = dt.strftime("%Y-%m-%d") if hasattr(dt, "strftime") else str(dt)[:10]
            open_ = float(row["Open"]) if "Open" in row and row["Open"] is not None else None
            high = float(row["High"]) if "High" in row and row["High"] is not None else None
            low = float(row["Low"]) if "Low" in row and row["Low"] is not None else None
            close = float(row["Close"]) if "Close" in row and row["Close"] is not None else 0.0
            vol = int(row["Volume"]) if "Volume" in row and row["Volume"] is not None else None
            out.append((date_str, open_, high, low, close, vol))
        return out
    except Exception as e:
        logger.warning("yfinance %s: %s", ticker, e)
        return []


def update_market_bars(conn: Any, days: int = DEFAULT_DAYS) -> int:
    """
    For each S&P 500 constituent: ensure company, fetch daily bars, upsert fisher_market_bar_daily.
    Returns number of bar rows written.
    """
    constituents = get_sp500_constituents()
    if not constituents:
        logger.warning("No S&P 500 constituents found")
        return 0
    written = 0
    with conn.cursor() as cur:
        for c in constituents:
            ticker = (c.get("ticker") or "").strip()
            if not ticker:
                continue
            company_id = ensure_company(
                conn,
                ticker,
                name=c.get("name"),
                cik=c.get("cik") and str(c.get("cik")).strip() or None,
                sector=c.get("sector"),
                industry=c.get("industry"),
            )
            bars = fetch_daily_bars(ticker, days=days)
            time.sleep(RATE_LIMIT_SEC)
            for date_str, open_, high, low, close, volume in bars:
                try:
                    cur.execute(
                        """
                        INSERT INTO fisher_market_bar_daily (company_id, date, open, high, low, close, volume)
                        VALUES (%s, %s::date, %s, %s, %s, %s, %s)
                        ON CONFLICT (company_id, date) DO UPDATE SET
                            open = EXCLUDED.open,
                            high = EXCLUDED.high,
                            low = EXCLUDED.low,
                            close = EXCLUDED.close,
                            volume = EXCLUDED.volume
                        """,
                        (company_id, date_str, open_, high, low, close, volume),
                    )
                    written += 1
                except Exception as e:
                    logger.debug("insert bar %s %s: %s", ticker, date_str, e)
    return written


def update_peer_sets(conn: Any) -> int:
    """
    Build peer_set per company: same GICS sector from S&P 500. Upsert fisher_peer_set.
    Returns number of peer_set rows updated.
    """
    constituents = get_sp500_constituents()
    by_sector: dict[str, list[str]] = defaultdict(list)
    for c in constituents:
        ticker = (c.get("ticker") or "").strip()
        sector = (c.get("sector") or "Unknown").strip()
        if ticker:
            by_sector[sector].append(ticker)
    count = 0
    with conn.cursor() as cur:
        for c in constituents:
            ticker = (c.get("ticker") or "").strip()
            sector = (c.get("sector") or "Unknown").strip()
            if not ticker:
                continue
            cur.execute("SELECT id FROM fisher_company WHERE ticker = %s", (ticker.upper(),))
            row = cur.fetchone()
            if not row:
                continue
            company_id = row["id"]
            peers = [p for p in by_sector.get(sector, []) if p != ticker]
            cur.execute(
                """
                INSERT INTO fisher_peer_set (company_id, peer_tickers, source, updated_at)
                VALUES (%s, %s::jsonb, 'sp500_sector', NOW())
                ON CONFLICT (company_id) DO UPDATE SET
                    peer_tickers = EXCLUDED.peer_tickers,
                    source = EXCLUDED.source,
                    updated_at = NOW()
                """,
                (company_id, json.dumps(peers)),
            )
            count += 1
    return count


def run_market_updater(days: int = DEFAULT_DAYS, update_peers: bool = True) -> dict[str, int]:
    """
    Run daily market job: update_market_bars then update_peer_sets.
    Returns {"bars": N, "peer_sets": M}.
    """
    with get_connection() as conn:
        bars = update_market_bars(conn, days=days)
        peer_sets = update_peer_sets(conn) if update_peers else 0
    return {"bars": bars, "peer_sets": peer_sets}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    result = run_market_updater()
    print("Market updater:", result)
