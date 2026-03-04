#!/usr/bin/env python3
"""
Run premarket gap scanner pipeline and upsert results to Supabase.

Pipeline: Universe -> Premarket snapshots (IBKR/fallbacks) -> Features -> Hard filters
-> Score + reasons -> Top N -> scan_runs + scanner_trade_ideas.

Requires: SUPABASE_URL, SUPABASE_SECRET_KEY in .env (or env). Tables: scan_runs, scanner_trade_ideas (run scripts/scanner_schema.sql once).

Config (data/config.json or env): SCANNER_UNIVERSE_LIMIT, SCANNER_MIN_GAP, SCANNER_TOP_N,
SCANNER_MIN_PRICE, SCANNER_MAX_PRICE, FMP_API_KEY, ALPHAVANTAGE_API_KEY, etc.

Usage:
  python scripts/run_premarket_scanner.py
  python scripts/run_premarket_scanner.py --limit 100
"""
from __future__ import annotations

import json
import logging
import os
import sys
import uuid
from datetime import datetime
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
                    v = v.strip().strip("'\"").strip()
                    if k and k not in os.environ:
                        os.environ[k] = v
    except Exception:
        pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Scanner defaults
DEFAULT_MIN_GAP = 0.05
DEFAULT_TOP_N = 25
DEFAULT_MIN_PRICE = 1.0
DEFAULT_MAX_PRICE = 30.0


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


def load_config():
    path = ROOT / "data" / "config.json"
    if not path.is_file():
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Run premarket gap scanner and write to Supabase")
    ap.add_argument("--limit", type=int, default=None, help="Max universe symbols to scan (default from config)")
    ap.add_argument("--dry-run", action="store_true", help="Do not write to DB")
    args = ap.parse_args()

    config = load_config()
    limit = args.limit or config.get("SCANNER_UNIVERSE_LIMIT") or 500
    min_gap = config.get("SCANNER_MIN_GAP", DEFAULT_MIN_GAP)
    top_n = config.get("SCANNER_TOP_N", DEFAULT_TOP_N)
    min_price = config.get("SCANNER_MIN_PRICE", DEFAULT_MIN_PRICE)
    max_price = config.get("SCANNER_MAX_PRICE", DEFAULT_MAX_PRICE)

    from scanner.universe import get_candidates
    from scanner.market_data import get_premarket_snapshot
    from scanner.feature_engine import compute_features
    from scanner.scoring_engine import score_and_build_idea, rank_ideas, PremarketTradeIdea

    candidates = get_candidates(config=config, limit=limit)
    if not candidates:
        logger.warning("No candidates from universe. Add data/scanner_universe.txt or ensure data/sec_universe.json exists.")
        return 0

    logger.info("Scanning %s symbols (min_gap=%.2f%%, price %.1f-%.1f)", len(candidates), min_gap * 100, min_price, max_price)

    # ET timestamp for as_of
    try:
        from zoneinfo import ZoneInfo
        tz_et = ZoneInfo("America/New_York")
    except ImportError:
        try:
            import pytz
            tz_et = pytz.timezone("America/New_York")
        except ImportError:
            tz_et = None
    now = datetime.now(tz_et) if tz_et else datetime.utcnow()
    as_of_et = now.isoformat()

    ideas: list[PremarketTradeIdea] = []
    for i, symbol in enumerate(candidates):
        if (i + 1) % 50 == 0:
            logger.info("Progress %s/%s", i + 1, len(candidates))
        try:
            snap = get_premarket_snapshot(symbol, config)
            if snap is None or snap.prior_close is None or snap.prior_close <= 0:
                continue
            if snap.pm_last < min_price or snap.pm_last > max_price:
                continue
            feats = compute_features(snap, avg_pm_vol_20d=None)
            if feats.gap_pct < min_gap:
                continue
            idea = score_and_build_idea(snap, feats, as_of_et, avg_pm_vol_20d=None)
            ideas.append(idea)
        except Exception as e:
            logger.debug("Skip %s: %s", symbol, e)
            continue

    ranked = rank_ideas(ideas, top_n=top_n)
    logger.info("Ranked %s ideas (top %s)", len(ranked), top_n)

    if args.dry_run:
        for r, idea in enumerate(ranked[:10], start=1):
            logger.info("#%s %s score=%.2f gap=%.1f%% %s", r, idea.symbol, idea.total_score, idea.pm_gap_pct * 100, idea.reasons[:2])
        return 0

    supabase = get_supabase_client()
    if not supabase:
        logger.error("Supabase not configured. Set SUPABASE_URL and SUPABASE_SECRET_KEY.")
        return 1

    run_id = str(uuid.uuid4())
    # Insert scan_runs (as_of_et stored as timestamptz; Supabase accepts ISO string)
    try:
        supabase.table("scan_runs").insert({
            "id": run_id,
            "as_of_et": as_of_et,
            "config_hash": None,
            "created_at": now.isoformat(),
        }).execute()
    except Exception as e:
        logger.exception("Insert scan_runs failed: %s", e)
        return 1

    rows = []
    for rank_one, idea in enumerate(ranked, start=1):
        rows.append({
            "run_id": run_id,
            "symbol": idea.symbol,
            "rank": rank_one,
            "total_score": float(idea.total_score),
            "payload": idea.to_payload(),
        })

    if rows:
        try:
            supabase.table("scanner_trade_ideas").insert(rows).execute()
            logger.info("Inserted %s rows into scanner_trade_ideas for run %s", len(rows), run_id)
        except Exception as e:
            logger.exception("Insert scanner_trade_ideas failed: %s", e)
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
