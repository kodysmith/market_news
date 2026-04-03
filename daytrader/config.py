"""Day trading bot configuration — all thresholds and risk parameters."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# ── Mode ──────────────────────────────────────────────────────────────
MODE = os.getenv("DT_MODE", "dry-run")  # dry-run | paper | live

# ── Account ───────────────────────────────────────────────────────────
ACCOUNT_SIZE = float(os.getenv("DT_ACCOUNT", "200000"))
RISK_PER_TRADE = 0.01       # 1% of account
MAX_RISK_DOLLARS = ACCOUNT_SIZE * RISK_PER_TRADE

# ── Scanner thresholds ────────────────────────────────────────────────
SCAN_MIN_GAP_PCT = 0.08     # 8% minimum gap
SCAN_MIN_RVOL = 5.0         # 5x relative volume
SCAN_MIN_FLOAT = 2_000_000  # 2M shares
SCAN_MAX_FLOAT = 30_000_000 # 30M shares
SCAN_MIN_PRICE = 3.0
SCAN_MAX_PRICE = 20.0
SCAN_MIN_AVG_VOL = 300_000  # 300K avg daily volume
SCAN_MIN_PREMARKET_VOL = 500_000
SCAN_MAX_SPREAD_PCT = 0.005 # 0.5% max spread
SCAN_MAX_CANDIDATES = 15    # max scanner output
SCAN_WATCHLIST_SIZE = 5     # final watchlist after catalyst scoring

# ── Catalyst scoring ──────────────────────────────────────────────────
CATALYST_MIN_SCORE = 6      # minimum to trade
CATALYST_SCORES = {
    "earnings_beat_guidance": 10,
    "earnings_beat": 7,
    "fda_approval": 9,
    "major_contract": 8,
    "acquisition": 8,
    "partnership": 6,
    "analyst_upgrade": 4,
    "sector_sympathy": 3,
    "social_media": 2,
    "unknown": 0,
}

# ── Entry engine ──────────────────────────────────────────────────────
ENTRY_OR_MINUTES = 15                # opening range = first 15 min
ENTRY_BREAKOUT_VOL_MULT = 2.0       # breakout bar must be 2x avg OR bar vol
ENTRY_PULLBACK_MAX_RETRACE = 0.50   # pullback can't retrace > 50% of breakout
ENTRY_MAX_ENTRY_TIME_MIN = 60       # no entries after first 60 min (10:30 AM)
ENTRY_REQUIRE_ABOVE_VWAP = True     # entry must be above VWAP
ENTRY_MAX_CONCURRENT = 2            # max 2 positions at once
ENTRY_MIN_STOP_ATR_MULT = 1.0       # minimum stop = 1x ATR(14) on 1-min

# ── Position manager ─────────────────────────────────────────────────
EXIT_SCALE_1_R = 1.0        # sell 1/3 at 1R
EXIT_SCALE_2_R = 2.0        # sell 1/3 at 2R
EXIT_TRAIL_BUFFER = 0.02    # trail $0.02 below higher lows
EXIT_TIME_STOP_MIN = 15     # close if no 1R within 15 min
EXIT_CLOSE_ALL_TIME = "11:30"  # force close all by 11:30 AM ET
EXIT_EOD_TIME = "15:50"     # absolute latest close

# ── Risk governor ─────────────────────────────────────────────────────
RISK_MAX_DAILY_LOSS_PCT = 0.03      # 3% daily loss limit
RISK_MAX_WEEKLY_LOSS_PCT = 0.05     # 5% weekly loss limit
RISK_MAX_MONTHLY_DD_PCT = 0.08      # 8% monthly drawdown limit
RISK_MAX_TRADES_PER_DAY = 6
RISK_MAX_CONSECUTIVE_LOSSES = 3     # pause after 3 losses in a row
RISK_PAUSE_MINUTES = 30             # pause duration after consecutive losses
RISK_ADAPTIVE_LOOKBACK = 20         # rolling 20 trades for adaptive sizing
RISK_ADAPTIVE_FULL_THRESHOLD = 0.55 # full size if win rate >= 55%
RISK_ADAPTIVE_REDUCE_THRESHOLD = 0.45  # 75% size if 45-55%
RISK_ADAPTIVE_MINIMAL_THRESHOLD = 0.35 # 25% size if < 45%

# ── Friction ──────────────────────────────────────────────────────────
SLIPPAGE_PER_SHARE = 0.03   # $0.03 estimated slippage for small caps
COMMISSION_PER_SHARE = 0.005 # IBKR Pro rate

# ── IBKR ──────────────────────────────────────────────────────────────
IBKR_HOST = "127.0.0.1"
IBKR_PORT_PAPER = 7497
IBKR_PORT_LIVE = 7496
IBKR_CLIENT_ID = 20         # separate from IC bot (10) and scalp bot (15)

# ── Data sources ──────────────────────────────────────────────────────
# Priority: IBKR (free, real-time) → yfinance (free, delayed) → Polygon (paid, historical)
# IBKR provides: scanner, 1-min bars, pre-market data, quotes, execution — all free
# Polygon only needed for historical backtests if IBKR historical data isn't enough
# Massive.com used for OPTIONS data (GEX) — not for equity day trading
DATA_SOURCE = os.getenv("DT_DATA_SOURCE", "ibkr")  # ibkr | yfinance | polygon
POLYGON_CACHE_DIR = ROOT / "data" / "polygon_cache"
DT_DATA_DIR = ROOT / "data" / "daytrader"
DT_LOG_DIR = ROOT / "logs" / "daytrader"

# ── Logging ───────────────────────────────────────────────────────────
LOG_LEVEL = os.getenv("DT_LOG_LEVEL", "INFO")
