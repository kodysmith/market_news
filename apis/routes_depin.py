from __future__ import annotations

import os

from flask import Blueprint, jsonify, request

# Ensure shared path setup runs (QuantEngine/utils on sys.path)
from . import shared as _shared  # noqa: F401

bp = Blueprint("depin", __name__)


@bp.route("/depin/risk")
def depin_risk():
    """
    Compute de-pin risk for a ticker.

    This endpoint:
    1. Fetches latest 5-minute bar from yfinance
    2. Fetches options snapshot (DTE <= 2) from Massive API
    3. Loads rolling state from database
    4. Computes de-pin risk score (0-100)
    5. Saves updated state and result

    Query params:
        ticker: Stock/ETF symbol (default: SPY)
        bucket_dte_max: Max DTE for options bucket (default: 2)
        strike_window_pct: Strike window as % of spot (default: 0.01)
        strike_window_floor: Minimum strike window in points (default: 5.0)

    Returns:
        JSON with de-pin risk result including:
        - de_pin_risk: Score 0-100
        - band: LOW/MID/HIGH
        - guidance: Actionable text
        - All component metrics (atr5m, pin_persist, trend_strength, etc.)
    """
    ticker = request.args.get("ticker", "SPY").upper()
    bucket_dte_max = int(request.args.get("bucket_dte_max", 2))
    strike_window_pct = float(request.args.get("strike_window_pct", 0.01))
    strike_window_floor = float(request.args.get("strike_window_floor", 5.0))

    try:
        from QuantEngine.depin_risk import fetch_5m_bars, convert_options_to_contracts, compute_depin_risk
        from QuantEngine.depin_risk_database import load_state, save_state, save_risk_result
        from QuantEngine.gex_calculator import get_option_chain_snapshot, get_spot_price
        import json

        # Load config for API keys
        config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "config.json")
        config = {}
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = json.load(f)

        massive_key = config.get("MASSIVE_API_KEY", "")
        if not massive_key:
            return jsonify({"error": "MASSIVE_API_KEY not configured"}), 500

        # Step 1: Fetch latest 5m bar
        bars = fetch_5m_bars(ticker, period="5d")
        if not bars:
            return jsonify({"error": f"Failed to fetch 5m bars for {ticker}"}), 500

        latest_bar = bars[-1]  # Most recent bar

        # Step 2: Get spot price (FMP fallback when IBKR unavailable)
        alphavantage_key = config.get("ALPHAVANTAGE_API_KEY", "")
        fmp_key = config.get("FMP_API_KEY", "")
        spot = get_spot_price(ticker, massive_key, alphavantage_key, fmp_key)
        if not spot:
            spot = latest_bar.close  # Fallback to bar close

        # Step 3: Fetch options snapshot (DTE <= 2)
        strike_range = max(strike_window_pct * spot, strike_window_floor) * 2  # ±range
        snap = get_option_chain_snapshot(
            massive_key,
            ticker,
            days_out=5,  # Only need near-term expiries
            spot_price=spot,
            strike_range=strike_range,
        )

        if not snap or not snap.get("results"):
            return jsonify({"error": f"Failed to fetch options data for {ticker}"}), 500

        # Step 4: Convert options to contracts
        options = convert_options_to_contracts(snap["results"], spot)

        # Filter to DTE <= bucket_dte_max
        options = [opt for opt in options if opt.dte <= bucket_dte_max]

        if not options:
            return (
                jsonify(
                    {
                        "error": f"No options found with DTE <= {bucket_dte_max}",
                        "ticker": ticker,
                        "spot": spot,
                    }
                ),
                400,
            )

        # Step 5: Load rolling state
        state = load_state(ticker)

        # Step 6: Compute de-pin risk
        result = compute_depin_risk(
            symbol=ticker,
            bar=latest_bar,
            options_snapshot=options,
            state=state,
            bucket_dte_max=bucket_dte_max,
            strike_window_pct=strike_window_pct,
            strike_window_floor=strike_window_floor,
            vol_ref_30m=None,  # Can add time-of-day median later
        )

        # Step 7: Save updated state and result
        save_state(ticker, state)
        save_risk_result(result)

        # Step 8: Return result as JSON
        return jsonify(
            {
                "symbol": result.symbol,
                "timestamp": result.ts,
                "spot": result.spot,
                "bucket": result.bucket,
                "de_pin_risk": result.de_pin_risk,
                "band": result.band,
                "guidance": result.guidance,
                "dominant_strike": result.dominant_strike,
                "put_wall": result.put_wall,
                "call_wall": result.call_wall,
                "atr5m": result.atr5m,
                "pin_persist": result.pin_persist,
                "trend_strength": result.trend_strength,
                "move30": result.move30,
                "vol_ratio": result.vol_ratio,
                "net_gex": result.net_gex,
                "gex_change_30m": result.gex_change_30m,
                "gex_collapse": result.gex_collapse,
                "liq_fade": result.liq_fade,
                "wall_drift": result.wall_drift,
                "x_raw": result.x_raw,
                "options_count": len(options),
            }
        )

    except Exception as e:
        import traceback

        traceback.print_exc()
        return jsonify({"error": f"Failed to compute de-pin risk: {str(e)}", "ticker": ticker}), 500


@bp.route("/depin/state")
def depin_state():
    """
    Get current rolling state for a ticker.

    Query params:
        ticker: Stock/ETF symbol (default: SPY)

    Returns:
        JSON with current rolling state (EMA, ATR, history arrays)
    """
    ticker = request.args.get("ticker", "SPY").upper()

    try:
        from QuantEngine.depin_risk_database import load_state

        state = load_state(ticker)

        return jsonify(
            {
                "symbol": ticker,
                "prev_close": state.prev_close,
                "ema8": state.ema8,
                "ema21": state.ema21,
                "atr14": state.atr14,
                "tr_history": state.tr_history,
                "closes_30m": state.closes_30m,
                "netgex_30m": state.netgex_30m,
                "pin_scores_15m": state.pin_scores_15m,
                "vol_30m": state.vol_30m,
            }
        )

    except Exception as e:
        import traceback

        traceback.print_exc()
        return jsonify({"error": f"Failed to get state: {str(e)}", "ticker": ticker}), 500


@bp.route("/depin/history")
def depin_history():
    """
    Get recent de-pin risk history for a ticker.

    Query params:
        ticker: Stock/ETF symbol (default: SPY)
        limit: Maximum number of records (default: 100)

    Returns:
        JSON array of historical risk results
    """
    ticker = request.args.get("ticker", "SPY").upper()
    limit = int(request.args.get("limit", 100))

    try:
        from QuantEngine.depin_risk_database import get_risk_history

        history = get_risk_history(ticker, limit=limit)

        return jsonify({"symbol": ticker, "count": len(history), "history": history})

    except Exception as e:
        import traceback

        traceback.print_exc()
        return jsonify({"error": f"Failed to get history: {str(e)}", "ticker": ticker}), 500

