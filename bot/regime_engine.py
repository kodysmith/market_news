"""Regime confidence engine — smooth transition between market modes.

Prevents whipsawing by requiring SUSTAINED confirmation before increasing
exposure. Drops to defensive mode IMMEDIATELY on danger signals.

Tiers:
  BEAR:    bear calls only, half size
  CAUTION: bear calls or small ICs, wider strikes
  NEUTRAL: full-size ICs (core strategy)
  BULL:    full-size ICs + funded butterflies + aggressive delta
"""
from __future__ import annotations

import logging
import sys
from datetime import date, timedelta
from pathlib import Path
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logger = logging.getLogger("bot.regime_engine")


@dataclass
class RegimeState:
    tier: str = "CAUTION"         # BEAR | CAUTION | NEUTRAL | BULL
    score: int = 0                # -5 to +6
    days_in_tier: int = 0         # consecutive days in current tier
    confirmation_days: int = 0    # consecutive days qualifying for next tier up
    history: list = field(default_factory=list)  # last 20 scores

    # What the bot should do
    @property
    def position_scale(self) -> float:
        """Position sizing by tier. BEAR=0 (SGOV), CAUTION=25%, NEUTRAL/BULL=100%."""
        return {"BEAR": 0.0, "CAUTION": 0.25, "NEUTRAL": 1.0, "BULL": 1.0}[self.tier]

    @property
    def allow_ics(self) -> bool:
        return self.tier in ("NEUTRAL", "BULL", "CAUTION")

    @property
    def allow_bear_calls(self) -> bool:
        return False  # bear calls not worth the drawdown per backtest

    @property
    def park_in_sgov(self) -> bool:
        return self.tier == "BEAR"

    @property
    def use_wide_strikes(self) -> bool:
        return self.tier == "CAUTION"

    @property
    def allow_butterfly(self) -> bool:
        return self.tier in ("NEUTRAL", "BULL")


# Confirmation days required to move UP one tier
CONFIRM_DAYS = {
    "BEAR": 3,      # need 3 consecutive days to leave BEAR
    "CAUTION": 5,    # need 5 consecutive days to reach NEUTRAL
    "NEUTRAL": 20,   # need 20 consecutive days to reach BULL
}


def compute_regime_score(
    spx_price: float,
    sma50: float,
    sma200: float,
    sma200_prev: float,  # SMA200 from 20 days ago (for slope)
    vix: float,
    vix_3m: float | None = None,
    vix_20d_high: float | None = None,
    vix_40d_high: float | None = None,
    gex_positive: bool | None = None,
) -> int:
    """Compute daily regime score from -5 to +6.

    All inputs should use T-1 data (no look-ahead).
    """
    score = 0

    # Price vs SMAs
    if spx_price > sma50:
        score += 1
    else:
        score -= 1

    if spx_price > sma200:
        score += 1
    else:
        score -= 1

    # SMA200 slope (is the long-term trend rising or falling?)
    if sma200_prev and sma200_prev > 0:
        if sma200 > sma200_prev:
            score += 1  # trend is up
        # Don't subtract — flat SMA200 is neutral, not bearish

    # VIX level
    if vix < 22:
        score += 1
    elif vix > 28:
        score -= 1

    # VIX term structure
    if vix_3m and vix_3m > 0:
        ratio = vix / vix_3m
        if ratio < 0.95:
            score += 1  # contango = calm
        elif ratio > 1.05:
            score -= 1  # backwardation = crisis

    # VIX making lower highs (fear fading)
    if vix_20d_high and vix_40d_high:
        if vix_20d_high < vix_40d_high:
            score += 1  # fear is fading

    # GEX
    if gex_positive is not None:
        if not gex_positive:
            score -= 1  # negative GEX = amplified moves

    return score


def update_regime(state: RegimeState, score: int) -> RegimeState:
    """Update regime state with today's score. Returns new state.

    Key asymmetry:
    - Moving DOWN (defensive): IMMEDIATE on any qualifying score
    - Moving UP (aggressive): requires N CONSECUTIVE days of confirmation
    """
    state.score = score
    state.history.append(score)
    if len(state.history) > 20:
        state.history = state.history[-20:]

    old_tier = state.tier

    # Determine target tier from score
    if score <= -2:
        target = "BEAR"
    elif score <= 1:
        target = "CAUTION"
    elif score <= 4:
        target = "NEUTRAL"
    else:
        target = "BULL"

    tier_order = ["BEAR", "CAUTION", "NEUTRAL", "BULL"]
    current_idx = tier_order.index(state.tier)
    target_idx = tier_order.index(target)

    if target_idx < current_idx:
        # Moving DOWN — IMMEDIATE (fast to protect)
        state.tier = target
        state.days_in_tier = 1
        state.confirmation_days = 0
        logger.info("REGIME DROP: %s → %s (score=%d) — immediate", old_tier, target, score)

    elif target_idx > current_idx:
        # Moving UP — requires confirmation
        state.confirmation_days += 1
        required = CONFIRM_DAYS.get(state.tier, 5)

        if state.confirmation_days >= required:
            # Only move up ONE tier at a time
            new_tier = tier_order[current_idx + 1]
            state.tier = new_tier
            state.days_in_tier = 1
            state.confirmation_days = 0
            logger.info("REGIME UP: %s → %s (score=%d, confirmed %dd)",
                       old_tier, new_tier, score, required)
        else:
            state.days_in_tier += 1
            logger.info("REGIME HOLD: %s (score=%d targeting %s, confirm %d/%d)",
                       state.tier, score, target, state.confirmation_days, required)
    else:
        # Same tier
        state.days_in_tier += 1
        state.confirmation_days = 0  # reset confirmation if not consistently qualifying

    return state


def get_regime_status(state: RegimeState) -> dict:
    """Get human-readable regime status for logging/display."""
    return {
        "tier": state.tier,
        "score": state.score,
        "days_in_tier": state.days_in_tier,
        "confirmation_days": state.confirmation_days,
        "position_scale": state.position_scale,
        "allow_ics": state.allow_ics,
        "allow_bear_calls": state.allow_bear_calls,
        "use_wide_strikes": state.use_wide_strikes,
        "avg_score_5d": np.mean(state.history[-5:]) if len(state.history) >= 5 else state.score,
        "trend": "improving" if len(state.history) >= 5 and np.mean(state.history[-5:]) > np.mean(state.history[-10:-5]) else "deteriorating" if len(state.history) >= 10 else "insufficient data",
    }
