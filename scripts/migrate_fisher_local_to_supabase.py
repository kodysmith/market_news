#!/usr/bin/env python3
"""
Copy Fisher data from local Postgres to Supabase.

Modes:
  1. Direct: read from DATABASE_URL, write to SUPABASE_DB_URL (run from host that can reach both).
  2. Dump:   --dump-to DIR  reads from DATABASE_URL, writes JSON files under DIR (no Supabase needed).
  3. Load:   --load-from DIR reads JSON from DIR, writes to SUPABASE_DB_URL (run from host that can reach Supabase).

Run once before shutting down local DB. .env: DATABASE_URL (local), SUPABASE_DB_URL (Supabase).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

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

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
except ImportError:
    psycopg2 = None

# Tables in FK order so inserts don't violate references
FISHER_TABLES = [
    "fisher_company",
    "fisher_filing",
    "fisher_financial_fact",
    "fisher_segment_fact",
    "fisher_transcript",
    "fisher_transcript_chunk",
    "fisher_market_bar_daily",
    "fisher_peer_set",
    "fisher_score_snapshot",
    "fisher_scan_queue",
]


def _serialize_row(row: dict) -> dict:
    """Convert row dict for JSON (dates, decimals, UUIDs -> str)."""
    out = {}
    for k, v in row.items():
        if v is None:
            out[k] = None
        elif hasattr(v, "isoformat"):
            out[k] = v.isoformat()
        elif hasattr(v, "__float__") and hasattr(v, "as_integer_ratio"):
            out[k] = float(v)
        else:
            out[k] = v
    return out


def run_dump(local_url: str, dump_dir: Path) -> int:
    """Dump local Fisher tables to JSON files under dump_dir. Returns total rows."""
    print("Connecting to local (DATABASE_URL)...")
    conn = psycopg2.connect(local_url)
    conn.cursor_factory = RealDictCursor
    dump_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    for table in FISHER_TABLES:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT 1 FROM information_schema.tables WHERE table_schema = 'public' AND table_name = %s",
                (table,),
            )
            if not cur.fetchone():
                print(f"  Skip {table} (not present on local)")
                continue
            cur.execute(f"SELECT * FROM {table}")
            rows = cur.fetchall()
        if not rows:
            print(f"  {table}: 0 rows")
            continue
        cols = list(rows[0].keys())
        data = {"columns": cols, "rows": [_serialize_row(r) for r in rows]}
        out_file = dump_dir / f"{table}.json"
        with open(out_file, "w") as f:
            json.dump(data, f, indent=0)
        total += len(rows)
        print(f"  {table}: {len(rows)} rows -> {out_file}")
    conn.close()
    return total


def run_load(supabase_url: str, dump_dir: Path) -> int:
    """Load Fisher data from JSON files under dump_dir into Supabase. Returns total rows."""
    print("Connecting to Supabase (SUPABASE_DB_URL)...")
    conn = psycopg2.connect(supabase_url)
    conn.cursor_factory = RealDictCursor
    total = 0
    for table in FISHER_TABLES:
        in_file = dump_dir / f"{table}.json"
        if not in_file.exists():
            continue
        with open(in_file) as f:
            data = json.load(f)
        cols = data["columns"]
        rows = data["rows"]
        if not rows:
            print(f"  {table}: 0 rows")
            continue
        cols_sql = ", ".join(f'"{c}"' for c in cols)
        placeholders = ", ".join("%s" for _ in cols)
        upsert_sql = (
            f"INSERT INTO {table} ({cols_sql}) VALUES ({placeholders}) "
            "ON CONFLICT (id) DO UPDATE SET "
            + ", ".join(f'"{c}" = EXCLUDED."{c}"' for c in cols if c != "id")
        )
        with conn.cursor() as cur:
            for row in rows:
                cur.execute(upsert_sql, [row.get(c) for c in cols])
        conn.commit()
        total += len(rows)
        print(f"  {table}: {len(rows)} rows")
    conn.close()
    return total


def run_direct(local_url: str, supabase_url: str) -> int:
    """Copy from local to Supabase in one run. Returns total rows."""
    print("Connecting to local (DATABASE_URL)...")
    local_conn = psycopg2.connect(local_url)
    local_conn.cursor_factory = RealDictCursor

    print("Connecting to Supabase (SUPABASE_DB_URL)...")
    supabase_conn = psycopg2.connect(supabase_url)
    supabase_conn.cursor_factory = RealDictCursor

    total_rows = 0
    for table in FISHER_TABLES:
        with local_conn.cursor() as cur:
            cur.execute(
                "SELECT 1 FROM information_schema.tables WHERE table_schema = 'public' AND table_name = %s",
                (table,),
            )
            if not cur.fetchone():
                print(f"  Skip {table} (not present on local)")
                continue
            cur.execute(f"SELECT * FROM {table}")
            rows = cur.fetchall()
        if not rows:
            print(f"  {table}: 0 rows")
            continue
        cols = list(rows[0].keys())
        cols_sql = ", ".join(f'"{c}"' for c in cols)
        placeholders = ", ".join("%s" for _ in cols)
        upsert_sql = (
            f"INSERT INTO {table} ({cols_sql}) VALUES ({placeholders}) "
            "ON CONFLICT (id) DO UPDATE SET "
            + ", ".join(f'"{c}" = EXCLUDED."{c}"' for c in cols if c != "id")
        )
        with supabase_conn.cursor() as cur:
            for row in rows:
                cur.execute(upsert_sql, [row[c] for c in cols])
        supabase_conn.commit()
        total_rows += len(rows)
        print(f"  {table}: {len(rows)} rows")

    local_conn.close()
    supabase_conn.close()
    return total_rows


def main() -> None:
    if psycopg2 is None:
        print("Install psycopg2-binary.", file=sys.stderr)
        sys.exit(1)

    ap = argparse.ArgumentParser(description="Migrate Fisher data from local Postgres to Supabase")
    ap.add_argument("--dump-to", metavar="DIR", help="Dump local data to JSON files in DIR (no Supabase needed)")
    ap.add_argument("--load-from", metavar="DIR", help="Load from JSON files in DIR into Supabase")
    args = ap.parse_args()

    if args.dump_to:
        local_url = os.environ.get("DATABASE_URL")
        if not local_url:
            print("Set DATABASE_URL in .env.", file=sys.stderr)
            sys.exit(1)
        n = run_dump(local_url, Path(args.dump_to))
        print(f"Done. {n} rows dumped. Copy DIR to a host that can reach Supabase, then run with --load-from DIR.")
        return

    if args.load_from:
        supabase_url = os.environ.get("SUPABASE_DB_URL")
        if not supabase_url:
            print("Set SUPABASE_DB_URL in .env.", file=sys.stderr)
            sys.exit(1)
        n = run_load(supabase_url, Path(args.load_from))
        print(f"Done. {n} rows loaded into Supabase.")
        return

    # Direct copy
    local_url = os.environ.get("DATABASE_URL")
    supabase_url = os.environ.get("SUPABASE_DB_URL")
    if not local_url:
        print("Set DATABASE_URL (local Postgres) in .env.", file=sys.stderr)
        sys.exit(1)
    if not supabase_url:
        print("Set SUPABASE_DB_URL (Supabase Postgres) in .env.", file=sys.stderr)
        sys.exit(1)
    n = run_direct(local_url, supabase_url)
    print(f"Done. {n} total rows copied to Supabase.")


if __name__ == "__main__":
    main()
