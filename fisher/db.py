"""
Fisher Postgres database access.

Uses DATABASE_URL or SUPABASE_DB_URL for connection.
Run supabase_schema_fisher.sql in Supabase SQL editor before use.
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import Any, Generator
from uuid import UUID

logger = logging.getLogger(__name__)

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
except ImportError:
    psycopg2 = None  # type: ignore
    RealDictCursor = None  # type: ignore


def get_connection_url() -> str | None:
    """Postgres connection URL from env."""
    return os.environ.get("DATABASE_URL") or os.environ.get("SUPABASE_DB_URL")


@contextmanager
def get_connection() -> Generator[Any, None, None]:
    """Context manager for Postgres connection. Uses RealDictCursor."""
    url = get_connection_url()
    if not url:
        raise RuntimeError(
            "Set DATABASE_URL or SUPABASE_DB_URL for Fisher database. "
            "Get the connection string from Supabase Project Settings -> Database."
        )
    if psycopg2 is None:
        raise RuntimeError("Install psycopg2-binary for Fisher database support.")
    conn = psycopg2.connect(url)
    conn.cursor_factory = RealDictCursor
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def company_id_by_ticker(conn: Any, ticker: str) -> UUID | None:
    """Return fisher_company.id for ticker, or None."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id FROM fisher_company WHERE ticker = %s",
            (ticker.upper(),),
        )
        row = cur.fetchone()
        return row["id"] if row else None


def ensure_company(conn: Any, ticker: str, name: str | None = None, cik: str | None = None,
                   sector: str | None = None, industry: str | None = None) -> UUID:
    """Insert or get fisher_company; return id."""
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO fisher_company (ticker, name, cik, sector, industry)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (ticker) DO UPDATE SET
                name = COALESCE(EXCLUDED.name, fisher_company.name),
                cik = COALESCE(EXCLUDED.cik, fisher_company.cik),
                sector = COALESCE(EXCLUDED.sector, fisher_company.sector),
                industry = COALESCE(EXCLUDED.industry, fisher_company.industry),
                updated_at = NOW()
            RETURNING id
            """,
            (ticker.upper(), name or ticker, cik, sector, industry),
        )
        return cur.fetchone()["id"]
