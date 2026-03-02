"""Black-Scholes pricing wrapper for the SPX IC bot.

Thin layer over the existing BS functions in optimize_spx_verticals_historical.
All dollar values are per-contract unit (multiply by 100 for dollar P&L).

Actual signatures from optimize_spx_verticals_historical:
  bs_put_price(S, K, T, sigma)           — no R parameter (baked in)
  bs_call_price(S, K, T, sigma)
  strike_for_delta(S, T, target_delta, sigma, is_put)
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.optimize_spx_verticals_historical import (
    bs_call_price,
    bs_put_price,
    strike_for_delta,
)


def compute_strikes(
    spx: float,
    vix: float,
    dte: int,
    delta: float = 0.15,
    width: float = 50.0,
) -> tuple[float, float, float, float]:
    """Return (k_put_short, k_put_long, k_call_short, k_call_long).

    Uses target delta for the short legs and subtracts/adds width for the long legs.
    Both strikes are rounded to the nearest 5 (SPX strikes are in $5 increments).
    """
    sigma = vix / 100.0
    T = max(dte, 1) / 365.0

    k_put_short = strike_for_delta(spx, T, delta, sigma, is_put=True)
    k_call_short = strike_for_delta(spx, T, delta, sigma, is_put=False)

    if k_put_short is None or k_call_short is None:
        raise ValueError(
            f"strike_for_delta returned None for SPX={spx} VIX={vix} DTE={dte}"
        )

    # Round to nearest 5
    k_put_short = round(k_put_short / 5) * 5
    k_call_short = round(k_call_short / 5) * 5

    k_put_long = k_put_short - width
    k_call_long = k_call_short + width

    return float(k_put_short), float(k_put_long), float(k_call_short), float(k_call_long)


def price_put_spread(
    spx: float,
    sigma: float,
    T: float,
    k_short: float,
    k_long: float,
) -> float:
    """Mark value of one put credit spread (short k_short, long k_long), per unit.

    Returns positive value = debit to close (liability for the seller).
    """
    short_val = bs_put_price(spx, k_short, T, sigma)
    long_val = bs_put_price(spx, k_long, T, sigma)
    return max(short_val - long_val, 0.0)


def price_call_spread(
    spx: float,
    sigma: float,
    T: float,
    k_short: float,
    k_long: float,
) -> float:
    """Mark value of one call credit spread (short k_short, long k_long), per unit."""
    short_val = bs_call_price(spx, k_short, T, sigma)
    long_val = bs_call_price(spx, k_long, T, sigma)
    return max(short_val - long_val, 0.0)


def price_iron_condor(
    spx: float,
    vix: float,
    dte: int,
    k_put_short: float,
    k_put_long: float,
    k_call_short: float,
    k_call_long: float,
) -> float:
    """Mark value of the full IC (both spreads combined) per contract unit.

    This represents the debit required to close the position.
    """
    sigma = vix / 100.0
    T = max(dte, 0) / 365.0
    put_spread = price_put_spread(spx, sigma, T, k_put_short, k_put_long)
    call_spread = price_call_spread(spx, sigma, T, k_call_short, k_call_long)
    return put_spread + call_spread


def compute_credit(
    spx: float,
    vix: float,
    dte: int,
    k_put_short: float,
    k_put_long: float,
    k_call_short: float,
    k_call_long: float,
) -> float:
    """Entry credit per contract (same as price_iron_condor at entry)."""
    return price_iron_condor(spx, vix, dte, k_put_short, k_put_long, k_call_short, k_call_long)


def spread_pnl_pct(current_spread_val: float, entry_credit: float) -> float:
    """Fraction of premium captured (0 = no decay; 1 = full profit; >1 = loss territory).

    At entry:        pnl_pct =  0.0 (no decay yet)
    At profit target: current_spread_val = 0.5 × credit → pnl_pct = 0.5
    At stop loss:     current_spread_val = 2.0 × credit → pnl_pct = -1.0
    """
    if entry_credit < 1e-9:
        return 0.0
    return (entry_credit - current_spread_val) / entry_credit


def dte_remaining(expiration: date, today: date | None = None) -> int:
    """Calendar days remaining until expiration (clamped to 0)."""
    if today is None:
        today = date.today()
    return max((expiration - today).days, 0)
