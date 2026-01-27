"""
Shared wiring/utilities for the Market News Flask API.

Keeps path manipulation, cache instance, and common constants in one place so
blueprint modules can stay focused on request handling.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# ---- import/path setup ----

# Add parent directory (market_news) to path for QuantEngine imports
# QuantEngine is at market_news/QuantEngine, so we need market_news in path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # market_news
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

# Add QuantEngine directory itself to path (for direct imports like 'from gex_calculator import ...')
quant_engine_dir = Path(__file__).parent.parent / "QuantEngine"
if str(quant_engine_dir) not in sys.path:
    sys.path.insert(0, str(quant_engine_dir))

# Add utils to path for API cache
utils_path = Path(__file__).parent.parent / "utils"
if str(utils_path) not in sys.path:
    sys.path.insert(0, str(utils_path))

from api_cache import APICache  # noqa: E402

# ---- shared instances/constants ----

# Initialize API cache with 10-second TTL
api_cache = APICache(default_ttl=10)

REPORT_PATH = "data/report.json"
REFRESH_INTERVAL = 30 * 60  # 30 minutes in seconds


def is_report_stale() -> bool:
    if not os.path.exists(REPORT_PATH):
        return True
    mtime = os.path.getmtime(REPORT_PATH)
    return (time.time() - mtime) > REFRESH_INTERVAL

