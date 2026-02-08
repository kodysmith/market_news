"""
Probability Range API — "From now to close" cone, HPH/HPL, close distribution.

Panel A: cone percentiles (P10/P25/P50/P75/P90) per time bucket.
Panel B: HPH/HPL (modes of high/low), P50/P80 high and low.
Panel C: close mode and 50%/80% credible intervals.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict

from flask import Blueprint, jsonify, request

from . import shared as _shared  # noqa: F401
from .shared import api_cache


bp = Blueprint("probability", __name__)

# US market close: 4:00 PM ET = 16:00. Assume ET ≈ UTC-5 (no DST handling for v1).
MARKET_CLOSE_HOUR_ET = 16
MARKET_CLOSE_MINUTE_ET = 0
MARKET_OPEN_MINUTES = 9 * 60 + 30  # 9:30 AM
MARKET_CLOSE_MINUTES = 16 * 60 + 0  # 4:00 PM
TRADING_MINUTES_PER_DAY = MARKET_CLOSE_MINUTES - MARKET_OPEN_MINUTES  # 390


def _minutes_to_close_simple() -> int:
    """Minutes until 4 PM ET. Naive: assume ET = UTC-5."""
    utc_now = datetime.now(timezone.utc)
    # ET = UTC - 5 (no DST)
    et_hour = utc_now.hour - 5
    et_minute = utc_now.minute
    if et_hour < 0:
        et_hour += 24
    minutes_since_midnight = et_hour * 60 + et_minute
    if minutes_since_midnight < MARKET_OPEN_MINUTES:
        return TRADING_MINUTES_PER_DAY
    if minutes_since_midnight >= MARKET_CLOSE_MINUTES:
        return 0
    return MARKET_CLOSE_MINUTES - minutes_since_midnight


@bp.route("/probability/range")
def probability_range():
    """
    Get probability range: cone (Panel A), HPH/HPL (Panel B), close distribution (Panel C).

    Query params:
        ticker: Symbol (default SPY)
        minutes_to_close: Optional. Minutes until market close (default: computed for US session)

    Returns:
        JSON with inputs, cone, high_distribution, low_distribution, hph, hpl,
        close_distribution, close_mode, close_p25/p75, close_p10/p90, put_wall, call_wall (optional).
    """
    ticker = (request.args.get("ticker") or "SPY").strip().upper()
    if not ticker or len(ticker) > 10:
        return jsonify({"error": "Invalid ticker"}), 400

    try:
        from QuantEngine.probability_cone import (
            compute_probability_range,
            result_to_api_dict,
        )
        from QuantEngine.volatility_analyzer import analyze_volatility
        from QuantEngine.decision_cockpit import DecisionCockpit
        from QuantEngine.gex_calculator import (
            get_spot_price,
            get_option_chain_snapshot,
            compute_cockpit_state as compute_gex_state,
        )

        # Optional: use cockpit state for gamma + depin + walls
        config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "config.json")
        config = {}
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = json.load(f)
        massive_key = config.get("MASSIVE_API_KEY", "")
        alphavantage_key = config.get("ALPHAVANTAGE_API_KEY", "")

        spot = get_spot_price(ticker, massive_key, alphavantage_key)
        if not spot:
            return jsonify({"error": "Could not get spot price", "ticker": ticker}), 502

        vol = analyze_volatility(ticker)
        front_iv = vol.get("front_iv")
        if front_iv is None:
            front_iv = vol.get("vix")  # fallback to VIX
        if front_iv is None:
            front_iv = 20.0  # fallback 20% IV
        iv_annual = float(front_iv) / 100.0  # percent -> decimal

        minutes_to_close = request.args.get("minutes_to_close", type=int)
        if minutes_to_close is None or minutes_to_close < 0:
            minutes_to_close = _minutes_to_close_simple()
        minutes_to_close = max(1, min(minutes_to_close, TRADING_MINUTES_PER_DAY))

        gamma_negative = True
        depin_risk = 50.0
        put_wall = None
        call_wall = None
        try:
            snap = get_option_chain_snapshot(massive_key, ticker, days_out=65, spot_price=spot, strike_range=60)
            if snap and snap.get("results"):
                gex_state = compute_gex_state(snap, spot, ticker=ticker, num_strikes_each_side=20, strike_increment=1.0)
                cockpit = DecisionCockpit(gex_state, vol)
                state = cockpit.get_state(ticker)
                regime = state.get("regime", {})
                gamma_negative = regime.get("is_negative", True)
                depin = state.get("de_pin_risk")
                if depin is not None and depin.get("score") is not None:
                    depin_risk = float(depin.get("score", 50))
                structure = state.get("structure", {})
                primary = structure.get("primary_walls", {}) or {}
                put_wall = primary.get("put")
                call_wall = primary.get("call")
        except Exception:
            pass

        result = compute_probability_range(
            spot=spot,
            iv_annual=iv_annual,
            minutes_to_close=minutes_to_close,
            gamma_negative=gamma_negative,
            depin_risk=depin_risk,
            put_wall=put_wall,
            call_wall=call_wall,
            n_paths=5000,
            step_minutes=5,
            time_bucket_minutes=10,
        )
        out = result_to_api_dict(result)
        out["ticker"] = ticker
        out["put_wall"] = put_wall
        out["call_wall"] = call_wall
        return jsonify(out)
    except Exception as e:
        import traceback
        return jsonify({
            "error": str(e),
            "traceback": traceback.format_exc(),
        }), 500
