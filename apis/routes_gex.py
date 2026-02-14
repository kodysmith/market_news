from __future__ import annotations

import time

from flask import Blueprint, jsonify, request

# Ensure shared path setup runs (QuantEngine/utils on sys.path)
from . import shared as _shared  # noqa: F401

bp = Blueprint("gex", __name__)


# GEX (Gamma Exposure) Endpoints
@bp.route("/gex/calculate", methods=["POST", "GET"])
def calculate_gex():
    """Calculate GEX for a given ticker"""
    try:
        import sys
        import json
        from pathlib import Path

        # Add QuantEngine to path
        quant_engine_path = Path(__file__).parent.parent / "QuantEngine"
        if str(quant_engine_path) not in sys.path:
            sys.path.insert(0, str(quant_engine_path))

        from gex_calculator import (
            calculate_gex as calc_gex,
            get_option_chain_snapshot,
            get_spot_price,
        )

        # Get parameters
        if request.method == "POST":
            data = request.get_json()
            ticker = data.get("ticker", "SPY").upper()
            massive_api_key = data.get("massive_api_key", "")
            alphavantage_api_key = data.get("alphavantage_api_key", "")
            spot_price = data.get("spot_price")  # Optional manual override
        else:
            ticker = request.args.get("ticker", "SPY").upper()
            massive_api_key = request.args.get("massive_api_key", "")
            alphavantage_api_key = request.args.get("alphavantage_api_key", "")
            spot_price = request.args.get("spot_price", type=float)

        # Load from config if keys not provided
        config_path = Path(__file__).parent.parent / "data" / "config.json"
        if config_path.exists():
            with open(config_path, "r") as f:
                config = json.load(f)
            if not massive_api_key:
                massive_api_key = config.get("MASSIVE_API_KEY", "")
            if not alphavantage_api_key:
                alphavantage_api_key = config.get("ALPHAVANTAGE_API_KEY", "")
            fmp_api_key = config.get("FMP_API_KEY", "")
        else:
            fmp_api_key = ""

        if not massive_api_key:
            return jsonify({"error": "Massive API key is required"}), 400

        # Configuration for symmetric strike selection
        # Strategy: Fetch wide range from API, then filter locally to ±20 strikes
        API_FETCH_RANGE = 60  # Points to fetch from API (wider than needed)
        NUM_STRIKES_EACH_SIDE = 20  # ±20 strikes around ATM (41 total)
        STRIKE_INCREMENT = 1.0  # $1 strikes for SPY

        # === STEP 1: Get spot price (Alpha Vantage → FMP → yfinance) ===
        if not spot_price:
            spot_price = get_spot_price(ticker, massive_api_key, alphavantage_api_key, fmp_api_key)
            if not spot_price:
                return (
                    jsonify(
                        {
                            "error": "Could not fetch spot price. Please provide spot_price parameter.",
                            "ticker": ticker,
                        }
                    ),
                    400,
                )

        # === STEP 2: Fetch WIDE range from Massive API ===
        # Get more than we need, then filter locally for symmetric ±20
        snap = get_option_chain_snapshot(
            massive_api_key,
            ticker,
            limit=250,
            spot_price=spot_price,
            strike_range=API_FETCH_RANGE,
        )

        if not snap.get("results"):
            return jsonify({"error": f"No options data found for {ticker}"}), 404

        # === STEP 3: Calculate GEX with symmetric ±20 strike filter ===
        df, agg, metrics, skipped = calc_gex(
            snap,
            spot_price,
            massive_api_key,
            alphavantage_api_key,
            num_strikes_each_side=NUM_STRIKES_EACH_SIDE,
            strike_increment=STRIKE_INCREMENT,
        )

        if df.empty:
            return (
                jsonify(
                    {
                        "error": "No computable contracts found",
                        "ticker": ticker,
                        "skipped": skipped,
                    }
                ),
                400,
            )

        # Calculate breakdown
        call_gex = float(df[df["type"] == "call"]["gex"].sum())
        put_gex = float(df[df["type"] == "put"]["gex"].sum())

        # Prepare chart annotations for frontend visualization
        chart_annotations = {
            "spot_price": spot_price,
            "flip_line": metrics.get("flip_line"),
            "put_wall": metrics.get("put_wall"),
            "call_wall": metrics.get("call_wall"),
        }

        # Get diagnostics from metrics
        diagnostics = metrics.get("diagnostics", {})

        # Compute cumulative GEX and gamma slope
        from gex_calculator import compute_cumulative_gex, compute_gamma_slope

        # Build gex_by_strike dict from agg DataFrame
        gex_by_strike_dict = dict(zip(agg["strike"], agg["gex"]))
        cumulative_gex_list = compute_cumulative_gex(gex_by_strike_dict)

        # Format cumulative_gex for API response
        cumulative_gex_array = [
            {"strike": strike, "cumulative_gex": cum_gex}
            for strike, cum_gex in cumulative_gex_list
        ]

        # Compute gamma slope
        flip_line = metrics.get("flip_line")
        gamma_slope = compute_gamma_slope(cumulative_gex_list, spot_price, flip_line)

        # Format response with cumulative_gex included in gex_by_strike
        return jsonify(
            {
                "ticker": ticker,
                "spot_price": spot_price,
                "metrics": metrics,
                "breakdown": {
                    "call_gex": call_gex,
                    "put_gex": put_gex,
                    "total_contracts": len(df),
                    "skipped_contracts": skipped,
                    "call_contracts": len(df[df["type"] == "call"]),
                    "put_contracts": len(df[df["type"] == "put"]),
                },
                "gex_by_strike": agg.to_dict("records"),
                "cumulative_gex": cumulative_gex_array,
                "gamma_slope": gamma_slope,
                "chart_annotations": chart_annotations,
                "diagnostics": {
                    "calls_in_target": diagnostics.get("calls_in_target", 0),
                    "puts_in_target": diagnostics.get("puts_in_target", 0),
                    "is_symmetric": diagnostics.get("is_symmetric", False),
                    "target_strikes": NUM_STRIKES_EACH_SIDE * 2 + 1,
                    "num_strikes_each_side": NUM_STRIKES_EACH_SIDE,
                    "api_fetch_range": API_FETCH_RANGE,
                    "strike_range": f"±{NUM_STRIKES_EACH_SIDE} strikes around {spot_price:.0f}",
                    "missing_from_api": diagnostics.get("missing_from_api", 0),
                    "returned_but_unusable": diagnostics.get("returned_but_unusable", 0),
                },
                "timestamp": time.time(),
            }
        )

    except Exception as e:
        import traceback

        return (
            jsonify(
                {"error": f"Failed to calculate GEX: {str(e)}", "traceback": traceback.format_exc()}
            ),
            500,
        )


@bp.route("/gex/max-pain")
def get_max_pain():
    """
    Get max pain strike for a ticker and expiration.
    Query params: ticker (required), dte (optional int, default 0 = front expiry),
    expiration (optional YYYY-MM-DD; if set, overrides dte).
    """
    try:
        import sys
        import json
        from pathlib import Path

        quant_engine_path = Path(__file__).parent.parent / "QuantEngine"
        if str(quant_engine_path) not in sys.path:
            sys.path.insert(0, str(quant_engine_path))

        from gex_calculator import (
            get_option_chain_snapshot,
            get_spot_price,
            compute_max_pain as calc_max_pain,
            get_expiration_dates_from_snap,
        )

        ticker = request.args.get("ticker", "SPY").upper()
        dte = request.args.get("dte", type=int, default=0)
        expiration = request.args.get("expiration", "").strip() or None

        config_path = Path(__file__).parent.parent / "data" / "config.json"
        massive_api_key = ""
        alphavantage_api_key = ""
        fmp_api_key = ""
        if config_path.exists():
            with open(config_path, "r") as f:
                config = json.load(f)
            massive_api_key = config.get("MASSIVE_API_KEY", "")
            alphavantage_api_key = config.get("ALPHAVANTAGE_API_KEY", "")
            fmp_api_key = config.get("FMP_API_KEY", "")

        if not massive_api_key:
            return jsonify({"error": "Massive API key is required"}), 400

        spot_price = get_spot_price(ticker, massive_api_key, alphavantage_api_key, fmp_api_key)
        if not spot_price:
            return (
                jsonify(
                    {
                        "error": "Could not fetch spot price.",
                        "ticker": ticker,
                    }
                ),
                400,
            )

        snap = get_option_chain_snapshot(
            massive_api_key,
            ticker,
            limit=250,
            spot_price=spot_price,
            strike_range=60,
        )

        if not snap.get("results"):
            return jsonify({"error": f"No options data found for {ticker}"}), 404

        all_expirations = get_expiration_dates_from_snap(snap)
        if not all_expirations:
            return jsonify({"error": "No expiration dates in chain", "ticker": ticker}), 404

        # Resolve target expiration: explicit param > dte index
        if expiration and expiration in all_expirations:
            target_expiration = expiration
        else:
            idx = max(0, min(dte, len(all_expirations) - 1))
            target_expiration = all_expirations[idx]

        max_pain_strike, chosen_exp, expirations_list = calc_max_pain(
            snap, spot_price, expiration_str=target_expiration
        )

        if max_pain_strike is None:
            return (
                jsonify(
                    {
                        "error": "Could not compute max pain for selected expiration",
                        "ticker": ticker,
                        "expiration": target_expiration,
                    }
                ),
                400,
            )

        return jsonify(
            {
                "ticker": ticker,
                "expiration": chosen_exp,
                "max_pain_strike": max_pain_strike,
                "spot_price": spot_price,
                "expirations": expirations_list,
            }
        )

    except Exception as e:
        import traceback

        return (
            jsonify(
                {"error": f"Failed to get max pain: {str(e)}", "traceback": traceback.format_exc()}
            ),
            500,
        )


@bp.route("/gex/tickers")
def get_gex_tickers():
    """Get list of supported GEX tickers from config"""
    try:
        import json
        from pathlib import Path

        config_path = Path(__file__).parent.parent / "data" / "config.json"
        if config_path.exists():
            with open(config_path, "r") as f:
                config = json.load(f)
            tickers = config.get("GEX_TICKERS", ["SPY", "SPX", "QQQ"])
        else:
            tickers = ["SPY", "SPX", "QQQ", "IWM", "NVDA", "TSLA", "AAPL", "MSFT"]

        return jsonify({"tickers": tickers, "default": "SPY"})

    except Exception as e:
        return jsonify({"error": f"Failed to get tickers: {e}"}), 500


@bp.route("/gex/summary")
def get_gex_summary():
    """
    Get batch GEX summary for all configured tickers.
    Returns quick overview with flip line and regime for each ticker.
    """
    try:
        import sys
        import json
        from pathlib import Path

        # Add QuantEngine to path
        quant_engine_path = Path(__file__).parent.parent / "QuantEngine"
        if str(quant_engine_path) not in sys.path:
            sys.path.insert(0, str(quant_engine_path))

        from gex_calculator import (
            calculate_gex as calc_gex,
            get_option_chain_snapshot,
            get_spot_price,
        )

        # Load config
        config_path = Path(__file__).parent.parent / "data" / "config.json"
        config = {}
        if config_path.exists():
            with open(config_path, "r") as f:
                config = json.load(f)

        massive_api_key = config.get("MASSIVE_API_KEY", "")
        alphavantage_api_key = config.get("ALPHAVANTAGE_API_KEY", "")
        fmp_api_key = config.get("FMP_API_KEY", "")
        tickers = config.get("GEX_TICKERS", ["SPY", "SPX", "QQQ"])

        if not massive_api_key:
            return jsonify({"error": "Massive API key not configured"}), 400

        # Get optional ticker filter from query params
        ticker_filter = request.args.get("tickers")
        if ticker_filter:
            tickers = [t.strip().upper() for t in ticker_filter.split(",")]

        summaries = []
        errors = []

        for ticker in tickers:
            try:
                # Get options snapshot
                snap = get_option_chain_snapshot(massive_api_key, ticker, limit=250)

                if not snap.get("results"):
                    errors.append({"ticker": ticker, "error": "No options data"})
                    continue

                # Get spot price (FMP fallback)
                spot_price = get_spot_price(ticker, massive_api_key, alphavantage_api_key, fmp_api_key)
                if not spot_price:
                    errors.append({"ticker": ticker, "error": "Could not fetch spot price"})
                    continue

                # Calculate GEX
                df, agg, metrics, skipped = calc_gex(snap, spot_price, massive_api_key, alphavantage_api_key)

                if df.empty:
                    errors.append({"ticker": ticker, "error": "No computable contracts"})
                    continue

                summaries.append(
                    {
                        "ticker": ticker,
                        "spot": spot_price,
                        "flip_line": metrics.get("flip_line"),
                        "put_wall": metrics.get("put_wall"),
                        "call_wall": metrics.get("call_wall"),
                        "total_gex": metrics.get("total_gex"),
                        "regime": metrics.get("regime", "unknown"),
                    }
                )

            except Exception as e:
                errors.append({"ticker": ticker, "error": str(e)})

        return jsonify({"tickers": summaries, "errors": errors if errors else None, "timestamp": time.time()})

    except Exception as e:
        import traceback

        return (
            jsonify({"error": f"Failed to get GEX summary: {str(e)}", "traceback": traceback.format_exc()}),
            500,
        )


@bp.route("/gex/price-comparison")
def get_gex_price_comparison():
    """
    Compare spot prices from different data sources for debugging discrepancies.
    Useful for understanding why GEX calculations might differ from other providers.
    """
    try:
        import sys
        import json
        from pathlib import Path

        # Add QuantEngine to path
        quant_engine_path = Path(__file__).parent.parent / "QuantEngine"
        if str(quant_engine_path) not in sys.path:
            sys.path.insert(0, str(quant_engine_path))

        from gex_calculator import get_spot_price_comparison

        # Get ticker from query params
        ticker = request.args.get("ticker", "SPY").upper()

        # Load config
        config_path = Path(__file__).parent.parent / "data" / "config.json"
        config = {}
        if config_path.exists():
            with open(config_path, "r") as f:
                config = json.load(f)

        massive_api_key = config.get("MASSIVE_API_KEY", "")
        alphavantage_api_key = config.get("ALPHAVANTAGE_API_KEY", "")
        fmp_api_key = config.get("FMP_API_KEY", "")

        # Get prices from all sources (including FMP)
        prices = get_spot_price_comparison(ticker, massive_api_key, alphavantage_api_key, fmp_api_key)

        # Calculate max discrepancy
        valid_prices = [p for p in prices.values() if p is not None]
        if len(valid_prices) >= 2:
            max_price = max(valid_prices)
            min_price = min(valid_prices)
            max_discrepancy_percent = ((max_price - min_price) / min_price) * 100 if min_price > 0 else 0
        else:
            max_discrepancy_percent = 0

        return jsonify(
            {
                "ticker": ticker,
                "prices": prices,
                "max_discrepancy_percent": round(max_discrepancy_percent, 4),
                "timestamp": time.time(),
            }
        )

    except Exception as e:
        import traceback

        return (
            jsonify(
                {"error": f"Failed to get price comparison: {str(e)}", "traceback": traceback.format_exc()}
            ),
            500,
        )

