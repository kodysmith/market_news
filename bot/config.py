"""Strategy configurations and compound growth rules for the SPX IC bot.

OPT2b compound strategy (Calmar 3.41, MaxDD 6.3% of $700K over 10 years):
  SWEET: 15_VIX_SWEET_SPOT × 4 contracts (fixed, selective)
  VD2:   VD2_ENTRY_105 × N contracts (grows every $30K banked)
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
TOTAL_ACCOUNT = 700_000       # Nominal account size for ROI calculations
RISK_BUDGET = 200_000         # Max capital deployed across all open ICs

# ---------------------------------------------------------------------------
# Compound growth parameters
# ---------------------------------------------------------------------------
SWEET_CONTRACTS = 4           # Fixed forever — never compound
VD2_CONTRACTS_START = 6       # Starting VD2 contract count
VD2_CONTRACTS_MAX = 22        # Hard ceiling
COMPOUND_THRESHOLD = 30_000   # Add 1 VD2 contract every $30K of banked profit

# ---------------------------------------------------------------------------
# Hard safety caps (enforced regardless of config)
# ---------------------------------------------------------------------------
MAX_OPEN_IC_TOTAL = 8         # Max iron condors open simultaneously (all configs)
DRAWDOWN_PAUSE_PCT = 0.08     # Pause new entries if account down >8% from ATH
VIX_ABSOLUTE_CAP = 35.0       # Never enter above VIX 35 (crash conditions)
MIN_ENTRY_DTE = 5             # Never open a position expiring in <5 calendar days
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

ALL_CONFIGS = [SWEET_CONFIG, VD2_CONFIG]

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
