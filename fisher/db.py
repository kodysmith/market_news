"""
Fisher Postgres database access.

Prefers Supabase Python client (SUPABASE_URL + SUPABASE_SECRET_KEY) when set — no direct DB host needed.
Falls back to Postgres (SUPABASE_DB_URL or DATABASE_URL) when client not set.
Run supabase_schema_fisher.sql in Supabase SQL editor before use.
"""

from __future__ import annotations

import json
import logging
import os
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Generator
from uuid import UUID

logger = logging.getLogger(__name__)

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
except ImportError:
    psycopg2 = None  # type: ignore
    RealDictCursor = None  # type: ignore


def get_supabase_client() -> Any | None:
    """Return Supabase client if SUPABASE_URL and SUPABASE_SECRET_KEY are set (avoids direct DB host)."""
    url = (os.environ.get("SUPABASE_URL") or "").strip().rstrip("/")
    key = (os.environ.get("SUPABASE_SECRET_KEY") or os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or "").strip()
    if not url or not key:
        return None
    try:
        from supabase import create_client
        return create_client(url, key)
    except ImportError:
        logger.warning("supabase package not installed; pip install supabase to use client instead of DB URL.")
        return None


def _is_supabase_client(conn: Any) -> bool:
    return conn is not None and hasattr(conn, "table")


def get_connection_url() -> str | None:
    """Postgres connection URL from env. Prefers SUPABASE_DB_URL so Fisher uses Supabase when both are set."""
    return os.environ.get("SUPABASE_DB_URL") or os.environ.get("DATABASE_URL")


def _connection_urls_in_try_order() -> list[str]:
    """Primary (Supabase) first, then DATABASE_URL as fallback when set."""
    urls = []
    supabase = os.environ.get("SUPABASE_DB_URL")
    database = os.environ.get("DATABASE_URL")
    if supabase:
        urls.append(supabase)
    if database and database not in urls:
        urls.append(database)
    return urls


@contextmanager
def get_connection() -> Generator[Any, None, None]:
    """Yield Supabase client (SUPABASE_URL + SUPABASE_SECRET_KEY) when set, else Postgres connection. Prefer client to avoid DB host DNS issues."""
    client = get_supabase_client()
    if client is not None:
        logger.debug("Using Supabase Python client (SUPABASE_URL + SUPABASE_SECRET_KEY)")
        yield client
        return

    urls = _connection_urls_in_try_order()
    if not urls:
        raise RuntimeError(
            "Set SUPABASE_URL and SUPABASE_SECRET_KEY (preferred), or DATABASE_URL/SUPABASE_DB_URL for Fisher. "
            "Get URL/key from Supabase Project Settings -> API; DB URL from Database -> Connection string (use pooler if direct fails)."
        )
    if psycopg2 is None:
        raise RuntimeError("Install psycopg2-binary for Fisher database support.")
    last_error = None
    for i, url in enumerate(urls):
        try:
            conn = psycopg2.connect(url)
            conn.cursor_factory = RealDictCursor
            if i > 0 and last_error:
                logger.warning(
                    "Supabase unreachable (%s), using fallback DATABASE_URL",
                    getattr(last_error, "message", str(last_error))[:80],
                )
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()
            return
        except Exception as e:
            last_error = e
            if url == urls[-1]:
                raise
            logger.debug("Connection failed for URL (trying next): %s", str(e)[:80])
    if last_error:
        raise last_error


def company_id_by_ticker(conn: Any, ticker: str) -> UUID | None:
    """Return fisher_company.id for ticker (latest row if multiple), or None. Works with Supabase client or Postgres conn."""
    ticker_upper = ticker.upper()
    if _is_supabase_client(conn):
        r = conn.table("fisher_company").select("id").eq("ticker", ticker_upper).order("created_at", desc=True).limit(1).execute()
        rows = getattr(r, "data", None) or []
        if not rows:
            return None
        return rows[0].get("id")
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id FROM fisher_company
            WHERE ticker = %s
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (ticker_upper,),
        )
        row = cur.fetchone()
        return row["id"] if row else None


def ensure_company(conn: Any, ticker: str, name: str | None = None, cik: str | None = None,
                   sector: str | None = None, industry: str | None = None) -> UUID:
    """Insert or get fisher_company; return id. Works with Supabase client or Postgres conn."""
    ticker_upper = ticker.upper()
    if _is_supabase_client(conn):
        r = conn.table("fisher_company").select("id").eq("ticker", ticker_upper).order("created_at", desc=True).limit(1).execute()
        rows = getattr(r, "data", None) or []
        if rows:
            row_id = rows[0].get("id")
            upd = {"updated_at": datetime.utcnow().isoformat() + "Z"}
            if name is not None:
                upd["name"] = name
            if cik is not None:
                upd["cik"] = cik
            if sector is not None:
                upd["sector"] = sector
            if industry is not None:
                upd["industry"] = industry
            if len(upd) > 1:
                conn.table("fisher_company").update(upd).eq("id", row_id).execute()
            return row_id
        ins = conn.table("fisher_company").insert({
            "ticker": ticker_upper,
            "name": name or ticker_upper,
            "cik": cik,
            "sector": sector,
            "industry": industry,
        }).execute()
        data = getattr(ins, "data", None)
        if data and len(data) > 0:
            return data[0].get("id")
        raise RuntimeError("Supabase insert fisher_company returned no id")
    # Postgres path
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id FROM fisher_company
            WHERE ticker = %s
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (ticker_upper,),
        )
        row = cur.fetchone()
    if row:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE fisher_company SET
                    name = COALESCE(%s, name),
                    cik = COALESCE(%s, cik),
                    sector = COALESCE(%s, sector),
                    industry = COALESCE(%s, industry),
                    updated_at = NOW()
                WHERE id = %s
                """,
                (name or ticker_upper, cik, sector, industry, row["id"]),
            )
        return row["id"]
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO fisher_company (ticker, name, cik, sector, industry)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING id
            """,
            (ticker_upper, name or ticker_upper, cik, sector, industry),
        )
        return cur.fetchone()["id"]


# --- Supabase-only helpers for scoring and worker (used when conn is client) ---

def get_latest_facts_supabase(client: Any, company_id: Any) -> tuple[dict[str, float], dict[str, float] | None]:
    """Same contract as scoring_engine get_latest_facts: (current_facts, prior_facts)."""
    from fisher.scoring_engine import _decimal_to_float
    r = client.table("fisher_filing").select("id, period_end, form_type").eq("company_id", str(company_id)).in_("form_type", ["10-K", "10-Q"]).order("period_end", desc=True).limit(2).execute()
    rows = getattr(r, "data", None) or []
    if not rows:
        return {}, None
    current_filing_id = rows[0]["id"]
    prior_filing_id = rows[1]["id"] if len(rows) > 1 else None

    def facts_for_filing(filing_id: Any) -> dict[str, float]:
        rr = client.table("fisher_financial_fact").select("fact_name, value, unit").eq("filing_id", str(filing_id)).execute()
        out = {}
        for row in getattr(rr, "data", None) or []:
            val = _decimal_to_float(row.get("value"))
            if val is not None and row.get("fact_name") not in out:
                out[row["fact_name"]] = val
        return out

    current = facts_for_filing(current_filing_id)
    prior = facts_for_filing(prior_filing_id) if prior_filing_id else None
    return current, prior


def _json_serialize(obj: Any) -> Any:
    """Make payload JSON-serializable for Supabase (Decimal -> float, etc.)."""
    from decimal import Decimal
    if obj is None:
        return None
    if isinstance(obj, (Decimal,)):
        return float(obj)
    if isinstance(obj, dict):
        return {k: _json_serialize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_serialize(x) for x in obj]
    return obj


def write_snapshot_supabase(client: Any, company_id: Any, payload: dict, version: int = 1) -> None:
    """Insert one row into fisher_score_snapshot via Supabase client."""
    points = _json_serialize(payload.get("points") or {})
    category_scores = _json_serialize(payload.get("category_scores") or {})
    total = payload.get("total_score")
    client.table("fisher_score_snapshot").insert({
        "company_id": str(company_id),
        "snapshot_at": datetime.utcnow().isoformat() + "Z",
        "points": points,
        "category_scores": category_scores,
        "total_score": total,
        "version": version,
    }).execute()


def get_company_ids_supabase(client: Any) -> list[Any]:
    """Return list of fisher_company id for run_scoring when company_ids is None."""
    r = client.table("fisher_company").select("id").execute()
    return [row["id"] for row in (getattr(r, "data", None) or [])]


def claim_batch_supabase(client: Any, batch_size: int) -> list[tuple[Any, str]]:
    """Claim up to batch_size pending rows from fisher_scan_queue; return list of (id, ticker). No FOR UPDATE SKIP LOCKED; best-effort."""
    r = client.table("fisher_scan_queue").select("id, ticker").eq("status", "pending").order("queued_at").limit(batch_size).execute()
    rows = getattr(r, "data", None) or []
    if not rows:
        return []
    now = datetime.utcnow().isoformat() + "Z"
    for row in rows:
        client.table("fisher_scan_queue").update({"status": "processing", "started_at": now}).eq("id", row["id"]).execute()
    return [(row["id"], row["ticker"]) for row in rows]


def mark_done_supabase(client: Any, ids: list[Any], error_text: str | None = None) -> None:
    """Mark fisher_scan_queue rows done or failed."""
    now = datetime.utcnow().isoformat() + "Z"
    payload = {"status": "failed" if error_text else "done", "completed_at": now, "error_text": (error_text[:2000] if error_text else None)}
    for id_ in ids:
        client.table("fisher_scan_queue").update(payload).eq("id", id_).execute()
