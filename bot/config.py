"""Strategy configurations and compound growth rules for the SPX IC bot.

Multi-layer SPX premium selling — $700K account, $200K risk budget:

  BULLISH (SPX > SMA50):
    SWEET: 15_VIX_SWEET_SPOT × 8 cts (14 DTE, VIX 16-24)
    SD1:   SD1_7DTE_BASE × 8 cts (7 DTE, high frequency)
    AG2:   AG2_VOLDET × 6 cts (21 DTE, 20-delta, $75 wide)
    VD2:   VD2_ENTRY_105 × N cts (14 DTE, compounds every $30K)

  BEARISH (SPX < SMA50):
    BC4:   BC4_BEAR_CALL_CONSERVATIVE × 6 cts (14 DTE, 10-delta, safe)
    BC2:   BC2_BEAR_CALL_7DTE × 6 cts (7 DTE, fast cycle)

  Always generating cash flow regardless of regime.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_enhanced_cashflow import (
    EntryFilters,
    ExitRules,
    StrategyConfig,
)

# ---------------------------------------------------------------------------
# Account / risk constants
# ---------------------------------------------------------------------------
TOTAL_ACCOUNT = 700_000       # Total account size
RISK_BUDGET = 200_000         # Max capital deployed across all open ICs

# ---------------------------------------------------------------------------
# Compound growth parameters
# ---------------------------------------------------------------------------
SWEET_CONTRACTS = 8           # Fixed — selective VIX sweet spot (~$44K margin)
SD1_CONTRACTS = 8             # Fixed — 7DTE high frequency (~$44K margin)
AG2_CONTRACTS = 6             # Fixed — 21DTE wider strikes (~$49.5K margin)
BC4_CONTRACTS = 6             # Fixed — bear call conservative, below SMA50 (~$33K margin)
BC2_CONTRACTS = 6             # Fixed — bear call 7DTE, below SMA50 (~$33K margin)
VD2_CONTRACTS_START = 6       # Starting VD2 contract count (~$33K margin)
VD2_CONTRACTS_MAX = 10        # Hard ceiling (~$55K margin)
COMPOUND_THRESHOLD = 30_000   # Add 1 VD2 contract every $30K of banked profit

# ---------------------------------------------------------------------------
# Hard safety caps (enforced regardless of config)
# ---------------------------------------------------------------------------
MAX_OPEN_IC_TOTAL = 8         # Max iron condors open simultaneously (all configs)
DRAWDOWN_PAUSE_PCT = 0.08     # Pause new entries if account down >8% from ATH ($56K on $700K)
VIX_ABSOLUTE_CAP = 35.0       # Never enter above VIX 35 (crash conditions)
MIN_ENTRY_DTE = 3             # Never open a position expiring in <3 calendar days (lowered for 7DTE strategy)
MARGIN_BUFFER_MULT = 1.1      # Require 110% of max-loss as buying power before entry

# ---------------------------------------------------------------------------
# Order execution timing
# ---------------------------------------------------------------------------
FILL_WAIT_SECONDS = 60        # Wait this long before improving limit price
IMPROVE_STEP = 0.05           # Improve limit by $0.05/contract each time
FILL_TIMEOUT_SECONDS = 120    # Switch to market order after this long

# ---------------------------------------------------------------------------
# Strategy configs
# ---------------------------------------------------------------------------
SWEET_CONFIG = StrategyConfig(
    config_id="15_VIX_SWEET_SPOT",
    strategy="iron_condor",
    dte=14,
    short_delta=0.15,
    width=50.0,
    exit_rules=ExitRules(
        profit_target_pct=0.50,
        stop_mult=2.0,
        time_stop_dte=2,
    ),
    filters=EntryFilters(
        vix_min=16,
        vix_max=24,
        trend_sma_period=50,
    ),
)

SD1_CONFIG = StrategyConfig(
    config_id="SD1_7DTE_BASE",
    strategy="iron_condor",
    dte=7,
    short_delta=0.15,
    width=50.0,
    exit_rules=ExitRules(
        profit_target_pct=0.50,
        stop_mult=2.0,
        # No time_stop — 7DTE is already short duration
    ),
    filters=EntryFilters(
        vix_max=28,
        trend_sma_period=50,
    ),
)

AG2_CONFIG = StrategyConfig(
    config_id="AG2_VOLDET",
    strategy="iron_condor",
    dte=21,
    short_delta=0.20,
    width=75.0,
    exit_rules=ExitRules(
        profit_target_pct=0.50,
        stop_mult=2.0,
    ),
    filters=EntryFilters(
        vix_max=28,
        trend_sma_period=50,
        rv_expansion_max=1.05,
    ),
)

VD2_CONFIG = StrategyConfig(
    config_id="VD2_ENTRY_105",
    strategy="iron_condor",
    dte=14,
    short_delta=0.15,
    width=50.0,
    exit_rules=ExitRules(
        profit_target_pct=0.50,
        stop_mult=2.0,
        # No time_stop for VD2 — let it run to profit target or stop
    ),
    filters=EntryFilters(
        vix_max=28,
        trend_sma_period=50,
        rv_expansion_max=1.05,
    ),
)

# ---------------------------------------------------------------------------
# Bearish configs (SPX below SMA50 — bear call verticals)
# ---------------------------------------------------------------------------
BC4_CONFIG = StrategyConfig(
    config_id="BC4_BEAR_CALL_CONSERVATIVE",
    strategy="call_vertical",
    dte=14,
    short_delta=0.10,
    width=50.0,
    exit_rules=ExitRules(
        profit_target_pct=0.50,
        stop_mult=2.0,
        time_stop_dte=2,
    ),
    filters=EntryFilters(
        vix_max=28,
        trend_sma_period=50,
        trend_sma_invert=True,
    ),
)

BC2_CONFIG = StrategyConfig(
    config_id="BC2_BEAR_CALL_7DTE",
    strategy="call_vertical",
    dte=7,
    short_delta=0.15,
    width=50.0,
    exit_rules=ExitRules(
        profit_target_pct=0.50,
        stop_mult=2.0,
    ),
    filters=EntryFilters(
        vix_min=16,
        vix_max=28,
        trend_sma_period=50,
        trend_sma_invert=True,
    ),
)

ALL_CONFIGS = [SWEET_CONFIG, SD1_CONFIG, AG2_CONFIG, VD2_CONFIG, BC4_CONFIG, BC2_CONFIG]

# ---------------------------------------------------------------------------
# Supabase table names
# ---------------------------------------------------------------------------
TABLE_POSITIONS = "bot_positions"
TABLE_STATE = "bot_state"
TABLE_TRADES = "bot_trades"

# ---------------------------------------------------------------------------
# State keys (used in bot_state table)
# ---------------------------------------------------------------------------
STATE_VD2_CONTRACTS = "vd2_contracts"
STATE_CUMULATIVE_PNL = "cumulative_pnl"
STATE_ATH_EQUITY = "ath_equity"         # All-time high account equity for drawdown calc
STATE_PAUSED = "entries_paused"          # "1" if manually or circuit-breaker paused

# ---------------------------------------------------------------------------
# Firebase FCM
# ---------------------------------------------------------------------------
FCM_TOPIC = "market_alerts"
