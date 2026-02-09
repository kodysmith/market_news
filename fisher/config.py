"""
Fisher pipeline configuration.

Loads from data/config.json and env (DATABASE_URL, SEC user-agent).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

# Project root (parent of fisher/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CONFIG_PATH = DATA_DIR / "config.json"
SP500_PATH = DATA_DIR / "sp500_constituents.json"
SEC_UNIVERSE_PATH = DATA_DIR / "sec_universe.json"
HIGH_GROWTH_UNIVERSE_PATH = DATA_DIR / "high_growth_universe.json"

# SEC EDGAR requires a descriptive User-Agent (see https://www.sec.gov/developer)
SEC_USER_AGENT = os.environ.get(
    "SEC_USER_AGENT",
    "MarketNews Fisher Pipeline (contact@example.com)",
)


def load_config() -> dict:
    """Load data/config.json."""
    if not CONFIG_PATH.exists():
        return {}
    with open(CONFIG_PATH, "r") as f:
        return json.load(f)


def get_database_url() -> str | None:
    """Postgres connection URL. Prefers SUPABASE_DB_URL so Fisher uses Supabase when both are set."""
    url = os.environ.get("SUPABASE_DB_URL") or os.environ.get("DATABASE_URL")
    if url:
        return url
    # Supabase project URL + key don't give direct Postgres; user must set DATABASE_URL
    return None


def get_sp500_constituents() -> list[dict]:
    """Load S&P 500 constituents from data/sp500_constituents.json."""
    if not SP500_PATH.exists():
        return []
    with open(SP500_PATH, "r") as f:
        data = json.load(f)
    return data.get("constituents", [])


def get_sec_universe() -> list[dict]:
    """Load SEC company list from data/sec_universe.json (run scripts/fetch_sec_universe.py first)."""
    if not SEC_UNIVERSE_PATH.exists():
        return []
    with open(SEC_UNIVERSE_PATH, "r") as f:
        data = json.load(f)
    return data.get("constituents", [])


def get_high_growth_universe() -> list[dict]:
    """Load high-growth list from data/high_growth_universe.json (run scripts/fetch_high_growth_universe.py first)."""
    if not HIGH_GROWTH_UNIVERSE_PATH.exists():
        return []
    with open(HIGH_GROWTH_UNIVERSE_PATH, "r") as f:
        data = json.load(f)
    return data.get("constituents", [])
