from __future__ import annotations

import os

from flask import Blueprint, jsonify, request

# Ensure shared path setup runs (QuantEngine/utils on sys.path)
from . import shared as _shared  # noqa: F401
from .shared import api_cache

from QuantEngine.decision_cockpit import DecisionCockpit
from QuantEngine.volatility_analyzer import analyze_volatility

bp = Blueprint("cockpit", __name__)


# ===============================
# Decision Cockpit Endpoints
# ===============================

@bp.route("/cockpit/state")
def cockpit_state():
    """
    Get the complete Decision Cockpit state.

    This is a single-screen trading state view with:
    - REGIME: GEX state (positive/negative gamma) + transition flag
    - VOLATILITY: IV direction and state
    - STRUCTURE: Multi-lens walls (Today/Tactical/Regime) and no-trade zones
    - ACTION FILTER: Allowed/forbidden actions based on regime + transition

    Enhanced with:
    - OPEX-aware expiry weighting
    - Dealer-centric GEX calculation
    - Multi-lens analysis (0-2, 0-14, 0-60 DTE)
    - Transition detection (near flip, flip moving)

    Query params:
        ticker: Stock ticker (any valid ticker symbol) - default: SPY

    Returns:
        JSON with regime, volatility, structure, action_filter, and net_series
    """
    ticker = request.args.get("ticker", "SPY").upper()

    # Basic ticker validation (format only, not restricted list)
    if not ticker or len(ticker) > 5 or not ticker.isalnum():
        return (
            jsonify({"error": "Invalid ticker format. Ticker must be 1-5 alphanumeric characters."}),
            400,
        )

    try:
        # Import GEX calculator with new functions
        from QuantEngine.gex_calculator import (
            get_option_chain_snapshot,
            get_spot_price,
            get_quote_yfinance,
            compute_cockpit_state as compute_gex_state,
            compute_max_pain as calc_max_pain,
            get_expiration_dates_from_snap,
        )
        import json

        # Load config for API keys
        config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "config.json")
        config = {}
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = json.load(f)

        massive_key = config.get("MASSIVE_API_KEY", "")
        alphavantage_key = config.get("ALPHAVANTAGE_API_KEY", "")

        # Get GEX state using new multi-lens computation
        # Strategy: Fetch wide range from API, then filter locally to ±20 strikes
        gex_state = {}
        snap = None  # Initialize snap for de-pin risk computation
        spot = None  # Initialize spot for de-pin risk computation
        try:
            # === STEP 1: Get spot price first (Yahoo/Alpha Vantage) ===
            spot = get_spot_price(ticker, massive_key, alphavantage_key)
            if spot:
                # === STEP 2: Fetch WIDE range from Massive API ===
                # Fetch ±60 points to ensure we have enough data
                # Then we'll filter locally to exactly ±20 strikes
                api_fetch_range = 60  # Points to request from API
                num_strikes_each_side = 20  # Strikes to keep each side of ATM
                strike_increment = 1.0  # $1 strikes for SPY

                snap = get_option_chain_snapshot(
                    massive_key,
                    ticker,
                    days_out=65,
                    spot_price=spot,
                    strike_range=api_fetch_range,
                )

                if snap and snap.get("results"):
                    # === STEP 3: Process with symmetric ±20 strike filter ===
                    # compute_cockpit_state will use process_chain_symmetric
                    # which filters to ONLY the ±20 strikes around ATM
                    gex_state = compute_gex_state(
                        snap,
                        spot,
                        ticker=ticker,
                        num_strikes_each_side=num_strikes_each_side,
                        strike_increment=strike_increment,
                    )

                    # Add metadata about the query
                    gex_state["strike_range"] = {
                        "api_fetch_range": api_fetch_range,
                        "processed_strikes_each_side": num_strikes_each_side,
                        "total_target_strikes": num_strikes_each_side * 2 + 1,
                        "center": spot,
                    }
        except Exception as e:
            print(f"[cockpit] GEX calculation failed: {e}")
            import traceback

            traceback.print_exc()

        # Get volatility data
        volatility_data = {}
        try:
            volatility_data = analyze_volatility(ticker)
        except Exception as e:
            print(f"[cockpit] Volatility analysis failed: {e}")

        # Build cockpit state with new multi-lens data
        cockpit = DecisionCockpit(gex_state, volatility_data)
        state = cockpit.get_state(ticker)

        # Compute de-pin risk and add to state
        depin_risk_data = None
        print(f"[depin] Starting de-pin risk computation for {ticker}, spot={spot}, snap={snap is not None}")
        try:
            from QuantEngine.depin_risk import fetch_5m_bars, convert_options_to_contracts, compute_depin_risk
            from QuantEngine.depin_risk_database import (
                load_state,
                save_state,
                save_risk_result,
                get_risk_30m_ago,
            )

            # Fetch latest 5m bar
            bars = fetch_5m_bars(ticker, period="5d")
            print(f"[depin] Fetched {len(bars) if bars else 0} bars, spot={spot}")
            if bars and spot:
                latest_bar = bars[-1]
                print(f"[depin] Latest bar: {latest_bar.ts}, close: {latest_bar.close}")

                # Fetch options snapshot (DTE <= 2) - reuse the same snapshot if available
                if snap and snap.get("results"):
                    print(f"[depin] Options snapshot has {len(snap['results'])} results")
                    options = convert_options_to_contracts(snap["results"], spot)
                    print(f"[depin] Converted to {len(options)} contracts")
                    options = [opt for opt in options if opt.dte <= 2]
                    print(f"[depin] After DTE filter: {len(options)} contracts")

                    if options:
                        # Load rolling state
                        depin_state = load_state(ticker)

                        # Compute de-pin risk
                        result = compute_depin_risk(
                            symbol=ticker,
                            bar=latest_bar,
                            options_snapshot=options,
                            state=depin_state,
                            bucket_dte_max=2,
                            strike_window_pct=0.01,
                            strike_window_floor=5.0,
                            vol_ref_30m=None,
                        )

                        # Save updated state and result
                        save_state(ticker, depin_state)
                        save_risk_result(result)

                        # Get previous risk for delta calculation
                        prev_risk = get_risk_30m_ago(ticker)
                        delta_30m = None
                        delta_direction = None
                        if prev_risk is not None:
                            delta_30m = result.de_pin_risk - prev_risk
                            delta_direction = "up" if delta_30m > 0 else "down" if delta_30m < 0 else "stable"

                        # Extract top 3 drivers (sorted by absolute contribution to x_raw)
                        # The drivers are: gex_collapse, trend_strength, move30, (1-pin_persist), liq_fade, wall_drift
                        drivers = [
                            {"name": "GEX Collapse", "contribution": result.gex_collapse * 1.40},
                            {"name": "Trend Strength", "contribution": result.trend_strength * 0.90},
                            {"name": "Move30", "contribution": result.move30 * 0.70},
                            {"name": "Pin Persist (inv)", "contribution": (1.0 - result.pin_persist) * 0.80},
                            {"name": "Liquidity Fade", "contribution": result.liq_fade * 0.50},
                            {"name": "Wall Drift", "contribution": result.wall_drift * 0.50},
                        ]
                        # Sort by absolute contribution and take top 3
                        drivers.sort(key=lambda x: abs(x["contribution"]), reverse=True)
                        top_3_drivers = drivers[:3]

                        depin_risk_data = {
                            "score": result.de_pin_risk,
                            "band": result.band,
                            "delta_30m": delta_30m,
                            "delta_direction": delta_direction,
                            "guidance": result.guidance,
                            "drivers": top_3_drivers,
                        }
        except Exception as e:
            print(f"[cockpit] De-pin risk calculation failed: {e}")
            import traceback

            traceback.print_exc()
            # Add error to response for debugging
            state["depin_risk_error"] = str(e)

        # Add de-pin risk to state if computed
        if depin_risk_data:
            state["de_pin_risk"] = depin_risk_data

        # Add quote (current, previous_close, open) for Block A
        try:
            quote = get_quote_yfinance(ticker)
            state["quote"] = {
                "current": quote.get("current"),
                "previous_close": quote.get("previous_close"),
                "open": quote.get("open"),
                "change": quote.get("change"),
                "change_pct": quote.get("change_pct"),
            }
        except Exception as e:
            print(f"[cockpit] Quote fetch failed: {e}")
            state["quote"] = None

        # Add max_pain for nearest expiry (Block D)
        try:
            if snap and snap.get("results") and spot is not None:
                all_exp = get_expiration_dates_from_snap(snap)
                target_exp = all_exp[0] if all_exp else None
                if target_exp:
                    max_pain_strike, chosen_exp, _ = calc_max_pain(
                        snap, spot, expiration_str=target_exp
                    )
                    if max_pain_strike is not None:
                        state["max_pain"] = {
                            "strike": round(max_pain_strike, 2),
                            "expiration": chosen_exp,
                        }
                    else:
                        state["max_pain"] = None
                else:
                    state["max_pain"] = None
            else:
                state["max_pain"] = None
        except Exception as e:
            print(f"[cockpit] Max pain failed: {e}")
            state["max_pain"] = None

        return jsonify(state)

    except Exception as e:
        import traceback

        traceback.print_exc()
        return jsonify({"error": f"Failed to get cockpit state: {str(e)}", "ticker": ticker}), 500


@bp.route("/cockpit/quote")
def cockpit_quote():
    """
    Get quote (current, previous_close, open) for dashboard price strip.
    Query params: ticker (default SPY)
    """
    ticker = request.args.get("ticker", "SPY").upper()
    if not ticker or len(ticker) > 5 or not ticker.isalnum():
        return jsonify({"error": "Invalid ticker format"}), 400
    try:
        from QuantEngine.gex_calculator import get_quote_yfinance

        quote = get_quote_yfinance(ticker)
        return jsonify({
            "ticker": ticker,
            "current": quote.get("current"),
            "previous_close": quote.get("previous_close"),
            "open": quote.get("open"),
            "change": quote.get("change"),
            "change_pct": quote.get("change_pct"),
        })
    except Exception as e:
        return jsonify({"error": str(e), "ticker": ticker}), 500


@bp.route("/cockpit/tickers")
def cockpit_tickers():
    """Get list of supported tickers for the cockpit"""
    return jsonify({"tickers": ["SPY", "QQQ", "IWM"], "default": "SPY"})


@bp.route("/trade-ideas/allowed")
def trade_ideas_allowed():
    """
    Get allowed trade ideas based on current regime.

    Only returns ideas that are permitted by the current market regime,
    ranked by risk-adjusted ROI using GEX walls and options chain.

    Query params:
        ticker: Stock ticker (any valid ticker symbol) - default: SPY
        max_ideas: Maximum number of ideas to return per timeframe (default: 3)
        timeframe: Timeframe selection - 'thisWeek', 'thisMonth', 'thisYear', or 'all' (default: 'all')
        min_dte: Optional override for minimum DTE (overrides timeframe)
        max_dte: Optional override for maximum DTE (overrides timeframe)

    Returns:
        JSON object with ideas grouped by timeframe (or single array if timeframe specified)
    """
    ticker = request.args.get("ticker", "SPY").upper().strip()
    max_ideas = int(request.args.get("max_ideas", 3))
    timeframe = request.args.get("timeframe", "all")  # 'all', 'thisWeek', 'thisMonth', 'thisYear'
    min_dte_override = request.args.get("min_dte", type=int)
    max_dte_override = request.args.get("max_dte", type=int)

    # Basic ticker validation (format only, not restricted list)
    if not ticker or len(ticker) > 5 or not ticker.isalnum():
        return (
            jsonify({"error": "Invalid ticker format. Ticker must be 1-5 alphanumeric characters."}),
            400,
        )

    try:
        # Import required modules
        from QuantEngine.trade_ideas_engine import (
            MarketContext,
            enhance_contract_with_pricing,
            generate_trade_ideas,
            generate_trade_ideas_with_preview,
            load_trade_ideas_config,
        )
        from QuantEngine.gex_calculator import get_option_chain_snapshot, get_spot_price, parse_option_chain
        from QuantEngine.decision_cockpit import DecisionCockpit
        from QuantEngine.volatility_analyzer import analyze_volatility
        import json

        # Load config for API keys
        config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "config.json")
        config = {}
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = json.load(f)

        massive_key = config.get("MASSIVE_API_KEY", "")
        alphavantage_key = config.get("ALPHAVANTAGE_API_KEY", "")

        # Get spot price
        spot = get_spot_price(ticker, massive_key, alphavantage_key)
        if not spot:
            return jsonify({"error": "Failed to get spot price"}), 500

        # Fetch options chain (wide range for trade ideas)
        snap = get_option_chain_snapshot(
            massive_key,
            ticker,
            days_out=65,
            spot_price=spot,
            strike_range=60,  # Wide range for trade ideas
        )

        if not snap or not snap.get("results"):
            return jsonify({"error": "Failed to fetch options chain"}), 500

        # Parse contracts
        contracts = parse_option_chain(snap, spot)

        # Enhance contracts with pricing
        quotes = []
        api_results = snap.get("results", [])
        api_index = {}  # Build index by (expiry, strike, type)

        for item in api_results:
            details = item.get("details", {})
            exp_str = details.get("expiration_date")
            strike = details.get("strike_price")
            contract_type = details.get("contract_type")
            if exp_str and strike and contract_type:
                key = (exp_str, float(strike), contract_type)
                api_index[key] = item

        for contract in contracts:
            key = (contract["expiry_str"], contract["strike"], contract["type"])
            api_item = api_index.get(key)
            if api_item:
                quote = enhance_contract_with_pricing(contract, spot, api_item)
                if quote:
                    quotes.append(quote)

        if not quotes:
            return jsonify({"error": "No valid option quotes found"}), 500

        # Get cockpit state for context (reuse the /cockpit/state endpoint logic to get de-pin risk)
        from QuantEngine.gex_calculator import compute_cockpit_state as compute_gex_state

        gex_state = compute_gex_state(snap, spot, ticker=ticker, num_strikes_each_side=20, strike_increment=1.0)
        volatility_data = analyze_volatility(ticker)
        cockpit = DecisionCockpit(gex_state, volatility_data)
        cockpit_state = cockpit.get_state(ticker)

        # Compute de-pin risk (same logic as /cockpit/state endpoint)
        depin_risk_data = None
        try:
            from QuantEngine.depin_risk import fetch_5m_bars, convert_options_to_contracts, compute_depin_risk
            from QuantEngine.depin_risk_database import (
                load_state,
                save_state,
                save_risk_result,
                get_risk_30m_ago,
            )

            # Fetch latest 5m bar
            bars = fetch_5m_bars(ticker, period="5d")
            if bars and spot and snap and snap.get("results"):
                latest_bar = bars[-1]

                # Convert options snapshot (DTE <= 2)
                options = convert_options_to_contracts(snap["results"], spot)
                options = [opt for opt in options if opt.dte <= 2]

                if options:
                    # Load rolling state
                    depin_state = load_state(ticker)

                    # Compute de-pin risk
                    result = compute_depin_risk(
                        symbol=ticker,
                        bar=latest_bar,
                        options_snapshot=options,
                        state=depin_state,
                        bucket_dte_max=2,
                        strike_window_pct=0.01,
                        strike_window_floor=5.0,
                        vol_ref_30m=None,
                    )

                    # Save updated state and result
                    save_state(ticker, depin_state)
                    save_risk_result(result)

                    # Get previous risk for delta calculation
                    prev_risk = get_risk_30m_ago(ticker)
                    delta_30m = None
                    delta_direction = None
                    if prev_risk is not None:
                        delta_30m = result.de_pin_risk - prev_risk
                        delta_direction = "up" if delta_30m > 0 else "down" if delta_30m < 0 else "stable"

                    depin_risk_data = {
                        "score": result.de_pin_risk,
                        "band": result.band,
                        "delta_30m": delta_30m,
                        "delta_direction": delta_direction,
                        "guidance": result.guidance,
                    }
        except Exception as e:
            print(f"[trade-ideas] De-pin risk calculation failed: {e}")
            # Don't fail the whole request, just log the error

        # Add de-pin risk to cockpit state if computed
        if depin_risk_data:
            cockpit_state["de_pin_risk"] = depin_risk_data

        # Extract events for earnings/macro detection
        events_data = {}
        try:
            from datetime import datetime, timedelta, date

            today = datetime.now().date()
            end_date = today + timedelta(days=1)

            # Check for earnings (reuse cockpit events logic)
            spy_top_holdings = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "GOOG", "META", "TSLA"]
            earnings_count = 0
            mega_cap_earnings = False

            # Try yfinance for earnings
            try:
                import yfinance as yf

                for holding in spy_top_holdings[:10]:
                    try:
                        stock = yf.Ticker(holding)
                        calendar = stock.calendar
                        if calendar and isinstance(calendar, dict) and "Earnings Date" in calendar:
                            earnings_dates = calendar["Earnings Date"]
                            if isinstance(earnings_dates, list) and len(earnings_dates) > 0:
                                next_earnings = earnings_dates[0]
                                if isinstance(next_earnings, datetime):
                                    report_date = next_earnings.date()
                                elif isinstance(next_earnings, date):
                                    report_date = next_earnings
                                elif isinstance(next_earnings, str):
                                    report_date = datetime.strptime(next_earnings.split()[0], "%Y-%m-%d").date()
                                else:
                                    continue

                                if today <= report_date <= end_date:
                                    earnings_count += 1
                                    if holding in ["AAPL", "MSFT", "GOOGL", "META", "AMZN"]:
                                        mega_cap_earnings = True
                    except Exception:
                        continue
            except Exception:
                pass

            events_data = {
                "earningsClusterScore": earnings_count,
                "megaCapEarningsNext24h": mega_cap_earnings,
                "macroEventNext24h": False,  # Could enhance with economic calendar
            }
        except Exception as e:
            print(f"[trade-ideas] Error fetching events: {e}")
            events_data = {
                "earningsClusterScore": 0,
                "megaCapEarningsNext24h": False,
                "macroEventNext24h": False,
            }

        # Build MarketContext from cockpit state
        regime_data = cockpit_state.get("regime", {})
        structure_data = cockpit_state.get("structure", {})
        volatility_data_state = cockpit_state.get("volatility", {})
        action_filter_data = cockpit_state.get("action_filter", {})
        depin_risk_data = cockpit_state.get("de_pin_risk")

        # Map volatility state to regime
        vol_state = volatility_data_state.get("state", "NORMAL")
        volatility_regime_map = {
            "COMPRESSED": "LOW",
            "NORMAL": "NORMAL",
            "ELEVATED": "ELEVATED",
            "EXPANDING": "EXTREME",
        }
        volatility_regime = volatility_regime_map.get(vol_state, "NORMAL")

        # Map trend regime from spot vs flip
        # Flip line is in regime section, not structure
        flip_line = regime_data.get("flip_line")
        if flip_line and spot:
            if spot > flip_line * 1.02:
                trend_regime = "UP"
            elif spot < flip_line * 0.98:
                trend_regime = "DOWN"
            else:
                trend_regime = "RANGE"
        else:
            trend_regime = "RANGE"

        # Get pin confidence from depin risk
        pin_confidence = "LOW"
        if depin_risk_data:
            depin_band = depin_risk_data.get("band", "MID")
            if depin_band == "LOW":
                pin_confidence = "HIGH"
            elif depin_band == "MID":
                pin_confidence = "MEDIUM"

        # Get action filter
        action_filter = action_filter_data.get("status", "REAL_OK")
        if action_filter == "WAIT":
            action_filter = "WAIT_FOR_CLARITY"
        elif action_filter == "BLOCKED":
            action_filter = "BLOCKED"
        elif action_filter == "PAPER_ONLY":
            action_filter = "PAPER_ONLY"
        else:
            action_filter = "REAL_OK"

        # Map GEX state from regime data
        # Use is_positive if available, otherwise fall back to label
        is_positive = regime_data.get("is_positive")
        if is_positive is not None:
            gex_state = "POSITIVE" if is_positive else "NEGATIVE"
        else:
            # Fall back to label parsing
            regime_label = regime_data.get("label", "UNKNOWN")
            if "POSITIVE" in regime_label.upper():
                gex_state = "POSITIVE"
            elif "NEGATIVE" in regime_label.upper():
                gex_state = "NEGATIVE"
            else:
                gex_state = "NEUTRAL"

        context = MarketContext(
            symbol=ticker,
            asOf=__import__("datetime").datetime.now().isoformat(),
            spot=spot,
            volatilityRegime=volatility_regime,
            trendRegime=trend_regime,
            liquidityState="NORMAL",
            putWall=structure_data.get("put_wall"),
            callWall=structure_data.get("call_wall"),
            flipLine=flip_line,
            rangePts=structure_data.get("range_pts"),
            depinRisk=depin_risk_data.get("score") if depin_risk_data else None,
            pinConfidence=pin_confidence,
            gexState=gex_state,
            earningsClusterScore=events_data.get("earningsClusterScore", 0),
            macroEventNext24h=events_data.get("macroEventNext24h", False),
            megaCapEarningsNext24h=events_data.get("megaCapEarningsNext24h", False),
            actionFilter=action_filter,
        )

        # Load trade ideas config
        trade_config = load_trade_ideas_config(ticker)
        trade_config["maxIdeasToShow"] = max_ideas

        # Handle timeframe selection and DTE overrides
        if min_dte_override is not None or max_dte_override is not None:
            # User-specified DTE range overrides everything
            trade_config["expiryCandidates"] = [
                {"minDTE": min_dte_override if min_dte_override is not None else 0, "maxDTE": max_dte_override if max_dte_override is not None else 365}
            ]
            trade_config["earningsWeekExpiryCandidates"] = trade_config["expiryCandidates"]
            timeframes_to_generate = ["custom"]
        elif timeframe == "all":
            # Generate for all timeframes
            timeframes_to_generate = ["thisWeek", "thisMonth", "thisYear"]
        else:
            # Single timeframe
            timeframes_to_generate = [timeframe]

        # Check for blocking reasons BEFORE generating ideas (use base config)
        from QuantEngine.trade_ideas_engine import check_global_blocks, determine_mode, filter_expiry_candidates

        base_config = load_trade_ideas_config(ticker)  # Use base config for blocking check
        is_blocked, block_reason = check_global_blocks(context, base_config)

        # Import helper functions for diagnostics
        from QuantEngine.trade_ideas_engine import (
            is_spot_above_flip_line,
            is_depin_risk_confirmed,
            is_depin_risk_elevated,
            get_allowed_strategies,
        )

        # Calculate flip line position and DePin status
        spot_above_flip = is_spot_above_flip_line(context, base_config)
        depin_elevated = is_depin_risk_elevated(context, base_config)
        depin_confirmed = is_depin_risk_confirmed(context, base_config)
        allowed_strategies_list = get_allowed_strategies(context, base_config)

        # Determine strategy filter reason if strategies were filtered
        strategy_filter_reason = None
        if not allowed_strategies_list:
            if spot_above_flip is False:
                strategy_filter_reason = "Spot below flip line - only hedged structures allowed (not yet implemented)"
            elif context.gexState == "NEGATIVE":
                strategy_filter_reason = "GEX negative - time-decay strategies blocked"
            elif depin_confirmed:
                strategy_filter_reason = f"DePin risk confirmed ({context.depinRisk}) - all strategies blocked"
            elif spot_above_flip is None:
                strategy_filter_reason = "Flip line unavailable - conservative blocking"

        # Collect diagnostic information
        diagnostics = {
            "context": {
                "symbol": context.symbol,
                "spot": context.spot,
                "gexState": context.gexState,
                "actionFilter": context.actionFilter,
                "depinRisk": context.depinRisk,
                "pinConfidence": context.pinConfidence,
                "putWall": context.putWall,
                "callWall": context.callWall,
                "flipLine": context.flipLine,
                "volatilityRegime": context.volatilityRegime,
                "trendRegime": context.trendRegime,
                "earningsClusterScore": context.earningsClusterScore,
                "macroEventNext24h": context.macroEventNext24h,
                "megaCapEarningsNext24h": context.megaCapEarningsNext24h,
            },
            "flipLineAnalysis": {"spotAboveFlip": spot_above_flip, "flipLine": context.flipLine, "spot": context.spot},
            "depinRiskAnalysis": {
                "depinRisk": context.depinRisk,
                "depinRiskElevated": depin_elevated,
                "depinRiskConfirmed": depin_confirmed,
            },
            "strategyFiltering": {
                "allowedStrategies": allowed_strategies_list,
                "strategyFilterReason": strategy_filter_reason,
            },
            "blocked": is_blocked,
            "blockReason": block_reason,
            "quotesCount": len(quotes),
            "config": {
                "strategies": trade_config.get("baseStrategySet", []),
                "minCredit": trade_config.get("riskEnvelope", {}).get("minCredit", 0.20),
                "maxLoss": trade_config.get("riskEnvelope", {}).get("maxLossPerIdea", 1000.0),
                "minOI": trade_config.get("minOI", 100),
                "minVolume": trade_config.get("minVolume", 10),
            },
        }

        # Generate trade ideas for each timeframe
        all_ideas_by_timeframe = {}

        if is_blocked:
            # If blocked, return empty for all timeframes
            for tf in timeframes_to_generate:
                all_ideas_by_timeframe[tf] = []
        else:
            # Generate ideas for each timeframe
            original_config = trade_config.copy()

            for tf in timeframes_to_generate:
                # Set expiry candidates for this timeframe
                if tf == "custom":
                    # Already set above with DTE overrides
                    pass
                elif tf in ["thisWeek", "thisMonth", "thisYear"]:
                    tf_config = original_config.get("timeframes", {}).get(tf, {})
                    if tf_config:
                        trade_config["expiryCandidates"] = tf_config.get("expiryCandidates", [{"minDTE": 7, "maxDTE": 10}])
                        trade_config["earningsWeekExpiryCandidates"] = tf_config.get(
                            "earningsWeekExpiryCandidates", trade_config["expiryCandidates"]
                        )
                    else:
                        # Fallback to default if timeframe config not found
                        trade_config["expiryCandidates"] = [{"minDTE": 7, "maxDTE": 10}]
                        trade_config["earningsWeekExpiryCandidates"] = [{"minDTE": 5, "maxDTE": 7}]

                # Generate ideas for this timeframe
                ideas = generate_trade_ideas(context, quotes, trade_config)
                all_ideas_by_timeframe[tf] = ideas

                # Restore original config for next iteration
                trade_config = original_config.copy()

        # Collect diagnostic info (use first timeframe's diagnostics)
        ideas_for_diagnostics = all_ideas_by_timeframe.get(timeframes_to_generate[0] if timeframes_to_generate else "thisMonth", [])

        # If blocked or no ideas, add diagnostic info
        if is_blocked or len(ideas_for_diagnostics) == 0:
            # Check additional reasons why no ideas might be generated
            mode = determine_mode(context, base_config)
            # Use first timeframe's config for diagnostics
            diag_config = trade_config.copy()
            if timeframes_to_generate and timeframes_to_generate[0] in ["thisWeek", "thisMonth", "thisYear"]:
                tf_config = diag_config.get("timeframes", {}).get(timeframes_to_generate[0], {})
                if tf_config:
                    diag_config["expiryCandidates"] = tf_config.get("expiryCandidates", [{"minDTE": 7, "maxDTE": 10}])
            filtered_quotes = filter_expiry_candidates(quotes, context, diag_config)

            diagnostics["mode"] = mode
            diagnostics["filteredQuotesCount"] = len(filtered_quotes)
            diagnostics["totalQuotesCount"] = len(quotes)

            # Get available DTE values for diagnostics
            if quotes:
                available_dtes = sorted(set(q.dte for q in quotes if q.dte >= 0))
                diagnostics["availableDTEs"] = available_dtes[:10]  # First 10 for display

            # Check if we have valid quotes after filtering
            if len(filtered_quotes) == 0:
                from datetime import datetime

                day_of_week = datetime.now().weekday()
                day_name = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][day_of_week]

                reasons = [f"No options quotes match expiry criteria (Today: {day_name})"]

                if day_of_week >= 4:
                    if day_of_week == 4:
                        reasons.append("Friday: Looking for 0DTE or next week (1-5 DTE)")
                    else:
                        reasons.append("Weekend: Looking for next week options (1-5 DTE)")
                else:
                    base_ranges = diag_config.get("expiryCandidates", [{"minDTE": 7, "maxDTE": 10}])
                    reasons.append(f"Required expiries: {base_ranges}")

                if quotes:
                    reasons.append(f"Available DTE values: {available_dtes[:10]}")

                diagnostics["additionalReasons"] = reasons
            elif not is_blocked:
                # Not blocked but no ideas - check candidate generation
                diagnostics["additionalReasons"] = [
                    "No valid candidates found after applying filters",
                    "Check: delta targets, wall buffers, liquidity requirements",
                ]

        # Generate preview candidates (once, not per timeframe)
        preview_result = generate_trade_ideas_with_preview(context, quotes, trade_config)
        preview_ideas = preview_result.get("preview", [])
        next_window = preview_result.get("nextWindow", {})
        current_phase = preview_result.get("currentPhase", "UNKNOWN")

        # Convert to JSON-serializable format
        from QuantEngine.trade_ideas_engine import TradeIdea

        def idea_to_dict(idea: TradeIdea) -> dict:
            return {
                "id": idea.id,
                "symbol": idea.symbol,
                "asOf": idea.asOf,
                "strategy": idea.strategy,
                "label": idea.label,
                "allowed": idea.allowed,
                "mode": idea.mode,
                "confidence": idea.confidence,
                "reasons": idea.reasons,
                "status": idea.status,
                "blockReason": idea.blockReason,
                "nextWindow": idea.nextWindow,
                "preMarket": idea.preMarket,
                "marketStatusNote": idea.marketStatusNote,
                "executions": [
                    {
                        "expiry": exec.expiry,
                        "dte": exec.dte,
                        "legs": [
                            {
                                "action": leg.action,
                                "type": leg.type,
                                "strike": leg.strike,
                                "priceMid": leg.priceMid,
                                "bid": leg.bid,
                                "ask": leg.ask,
                                "delta": leg.delta,
                                "oi": leg.oi,
                                "volume": leg.volume,
                            }
                            for leg in exec.legs
                        ],
                        "metrics": {
                            "netCredit": exec.netCredit,
                            "width": exec.width,
                            "maxProfit": exec.maxProfit,
                            "maxLoss": exec.maxLoss,
                            "roc": exec.roc,
                            "breakeven": exec.breakeven,
                            "pop": exec.pop,
                            "ev": exec.ev,
                            "wallBufferPct": exec.wallBufferPct,
                            "liquidityScore": exec.liquidityScore,
                            "score": exec.score,
                        },
                    }
                    for exec in idea.executions
                ],
                "best": {
                    "expiry": idea.best.expiry,
                    "dte": idea.best.dte,
                    "netCredit": idea.best.netCredit,
                    "maxProfit": idea.best.maxProfit,
                    "maxLoss": idea.best.maxLoss,
                    "roc": idea.best.roc,
                    "pop": idea.best.pop,
                    "ev": idea.best.ev,
                    "wallBufferPct": idea.best.wallBufferPct,
                    "liquidityGrade": idea.best.liquidityGrade,
                    "score": idea.best.score,
                }
                if idea.best
                else None,
                "contextSnapshot": {
                    "spot": idea.contextSnapshot.spot,
                    "putWall": idea.contextSnapshot.putWall,
                    "callWall": idea.contextSnapshot.callWall,
                    "flipLine": idea.contextSnapshot.flipLine,
                    "depinRisk": idea.contextSnapshot.depinRisk,
                    "gexState": idea.contextSnapshot.gexState,
                    "pinConfidence": idea.contextSnapshot.pinConfidence,
                    "earningsClusterScore": idea.contextSnapshot.earningsClusterScore,
                    "macroEventNext24h": idea.contextSnapshot.macroEventNext24h,
                }
                if idea.contextSnapshot
                else None,
            }

        # Return ideas with diagnostics
        if timeframe == "all":
            # Return grouped by timeframe
            result = {
                "ideasByTimeframe": {
                    tf: [idea_to_dict(idea) for idea in ideas_list] for tf, ideas_list in all_ideas_by_timeframe.items()
                },
                "diagnostics": diagnostics,
            }
        else:
            # Return single array for specific timeframe
            tf_ideas = all_ideas_by_timeframe.get(timeframe, [])
            result = {"ideas": [idea_to_dict(idea) for idea in tf_ideas], "diagnostics": diagnostics}

        return jsonify(result)

    except Exception as e:
        import traceback

        traceback.print_exc()
        return (
            jsonify(
                {
                    "error": f"Failed to generate trade ideas: {str(e)}",
                    "ideas": [],
                    "diagnostics": {"blocked": True, "blockReason": f"Error: {str(e)}"},
                }
            ),
            500,
        )


@bp.route("/cockpit/events")
def cockpit_events():
    """
    Get upcoming high-impact market events for cockpit display.

    Uses Alpha Vantage for earnings calendar and generates OPEX dates.

    Query params:
        days: Number of days ahead to look (default: 7)

    Returns:
        JSON with badges (compact for header) and full_calendar (expandable section)
    """
    from datetime import datetime, timedelta, date
    import requests

    days_ahead = int(request.args.get("days", 7))
    symbol = (request.args.get("symbol") or "").strip().upper() or None

    try:
        events = []
        today = datetime.now().date()
        end_date = today + timedelta(days=days_ahead)

        # SPY top holdings for earnings filter
        spy_top_holdings = [
            "AAPL",
            "MSFT",
            "NVDA",
            "AMZN",
            "GOOGL",
            "GOOG",
            "META",
            "TSLA",
            "BRK.B",
            "JPM",
            "V",
            "MA",
            "UNH",
            "HD",
            "PG",
            "JNJ",
            "XOM",
            "CVX",
            "ABBV",
            "MRK",
            "PFE",
            "KO",
            "PEP",
            "COST",
            "WMT",
            "BAC",
            "WFC",
            "NFLX",
            "ADBE",
            "CRM",
            "ORCL",
            "INTC",
            "AMD",
            "QCOM",
        ]

        # 1. Load config
        config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "config.json")
        config = {}
        if os.path.exists(config_path):
            import json

            with open(config_path, "r") as f:
                config = json.load(f)

        # Try FMP first (more reliable for earnings)
        fmp_key = config.get("FMP_API_KEY", "")
        if fmp_key:
            try:
                earnings_url = "https://financialmodelingprep.com/api/v3/earning_calendar"
                params = {"from": today, "to": end_date, "apikey": fmp_key}

                # Generate cache key and check cache
                cache_key = api_cache.generate_key(earnings_url, params, fmp_key)
                cached_earnings = api_cache.get(cache_key)

                if cached_earnings is not None:
                    earnings_data = cached_earnings
                else:
                    # Make API call
                    full_url = f"{earnings_url}?from={today}&to={end_date}&apikey={fmp_key}"
                    resp = requests.get(full_url, timeout=10)
                    if resp.status_code == 200:
                        earnings_data = resp.json()
                        # Cache the result
                        api_cache.set(cache_key, earnings_data, ttl=10)
                    else:
                        earnings_data = []

                if earnings_data:
                    for earning in earnings_data:
                        ticker = earning.get("symbol", "")
                        if ticker in spy_top_holdings:
                            event_date = earning.get("date", "")
                            event_time = earning.get("time", "")
                            time_label = "BMO" if event_time == "bmo" else ("AH" if event_time == "amc" else "")
                            events.append(
                                {
                                    "type": "earnings",
                                    "title": f"{ticker} Earnings",
                                    "date": event_date,
                                    "time": time_label,
                                    "impact": "high",
                                    "ticker": ticker,
                                    "source": "fmp",
                                }
                            )
                    print(f"[cockpit/events] Loaded {len([e for e in events if e['type']=='earnings'])} earnings from FMP")
            except Exception as e:
                print(f"[cockpit/events] FMP earnings error: {e}")

        # Fallback to yfinance for earnings if no FMP key or no earnings found
        if not [e for e in events if e["type"] == "earnings"]:
            try:
                import yfinance as yf

                print("[cockpit/events] Trying yfinance for earnings...")
                earnings_count = 0
                for ticker in spy_top_holdings[:15]:  # Limit to top 15 to avoid rate limits
                    try:
                        stock = yf.Ticker(ticker)
                        calendar = stock.calendar
                        if calendar and isinstance(calendar, dict) and "Earnings Date" in calendar:
                            earnings_dates = calendar["Earnings Date"]
                            if isinstance(earnings_dates, list) and len(earnings_dates) > 0:
                                # Get next earnings date (first in list)
                                next_earnings = earnings_dates[0]
                                if isinstance(next_earnings, datetime):
                                    report_date = next_earnings.date()
                                elif isinstance(next_earnings, date):
                                    report_date = next_earnings
                                elif isinstance(next_earnings, str):
                                    report_date = datetime.strptime(next_earnings.split()[0], "%Y-%m-%d").date()
                                else:
                                    continue

                                if today <= report_date <= end_date:
                                    # Check if we already have this earnings event
                                    if not any(e.get("ticker") == ticker and e.get("date") == report_date.isoformat() for e in events):
                                        events.append(
                                            {
                                                "type": "earnings",
                                                "title": f"{ticker} Earnings",
                                                "date": report_date.isoformat(),
                                                "time": "",
                                                "impact": "high",
                                                "ticker": ticker,
                                                "source": "yfinance",
                                            }
                                        )
                                        earnings_count += 1
                    except Exception:
                        continue
                print(
                    f"[cockpit/events] Loaded {earnings_count} earnings from yfinance (total earnings events: {len([e for e in events if e['type']=='earnings'])})"
                )
            except Exception as e:
                print(f"[cockpit/events] yfinance earnings error: {e}")
                import traceback

                traceback.print_exc()

        # 2. Add OPEX dates (3rd Friday of each month)
        def get_third_friday(year: int, month: int) -> int:
            """Get the third Friday of a given month"""
            from calendar import monthcalendar

            cal = monthcalendar(year, month)
            fridays = [week[4] for week in cal if week[4] != 0]
            return fridays[2] if len(fridays) >= 3 else fridays[-1]

        # Check current and next month for OPEX
        for month_offset in range(2):
            check_date = today + timedelta(days=30 * month_offset)
            opex_day = get_third_friday(check_date.year, check_date.month)
            opex_date = datetime(check_date.year, check_date.month, opex_day).date()

            if today <= opex_date <= end_date:
                events.append(
                    {
                        "type": "opex",
                        "title": "Monthly OPEX",
                        "date": opex_date.isoformat(),
                        "time": "Close",
                        "impact": "high",
                        "ticker": None,
                        "source": "calculated",
                    }
                )

        # 3. Add known recurring high-impact events (FOMC dates for 2026)
        fomc_dates_2026 = [
            "2026-01-28",
            "2026-01-29",  # Jan meeting
            "2026-03-17",
            "2026-03-18",  # Mar meeting
            "2026-05-05",
            "2026-05-06",  # May meeting
            "2026-06-16",
            "2026-06-17",  # Jun meeting
            "2026-07-28",
            "2026-07-29",  # Jul meeting
            "2026-09-15",
            "2026-09-16",  # Sep meeting
            "2026-11-03",
            "2026-11-04",  # Nov meeting
            "2026-12-15",
            "2026-12-16",  # Dec meeting
        ]

        for fomc_date_str in fomc_dates_2026:
            try:
                fomc_date = datetime.strptime(fomc_date_str, "%Y-%m-%d").date()
                if today <= fomc_date <= end_date:
                    # Only add the second day (decision day)
                    if fomc_date_str.endswith(("29", "18", "06", "17", "16", "04")):
                        events.append(
                            {
                                "type": "fomc",
                                "title": "FOMC Decision",
                                "date": fomc_date.isoformat(),
                                "time": "2:00 PM",
                                "impact": "high",
                                "ticker": None,
                                "source": "scheduled",
                            }
                        )
            except (ValueError, TypeError):
                pass

        # Sort events by date
        events.sort(key=lambda x: x["date"])

        # Create compact badges for header (top 3 upcoming)
        badges = []
        for event in events[:3]:
            event_date = datetime.strptime(event["date"], "%Y-%m-%d").date()

            # Format relative date
            days_until = (event_date - today).days
            if days_until == 0:
                date_label = "Today"
            elif days_until == 1:
                date_label = "Tomorrow"
            else:
                date_label = event_date.strftime("%a")  # Mon, Tue, etc.

            # Create compact badge text
            if event["type"] == "earnings":
                badge_text = f"{event['ticker']}"
            elif event["type"] == "opex":
                badge_text = "OPEX"
            elif event["type"] == "fomc":
                badge_text = "FOMC"
            else:
                title = event["title"]
                short_names = ["CPI", "PPI", "NFP", "GDP"]
                badge_text = next((name for name in short_names if name in title), title[:10])

            badges.append(
                {
                    "text": badge_text,
                    "type": event["type"],
                    "date_label": date_label,
                    "full_title": event["title"],
                    "impact": event["impact"],
                }
            )

        # Add impact_on_symbol for each event when symbol is provided
        index_tickers = {"SPY", "SPX", "QQQ", "NDX"}
        if symbol:
            for event in events:
                ev_type = event.get("type", "")
                ev_ticker = event.get("ticker")
                if ev_type == "earnings" and ev_ticker:
                    if ev_ticker == symbol or (symbol in index_tickers and ev_ticker in spy_top_holdings):
                        event["impact_on_symbol"] = "high"
                    else:
                        event["impact_on_symbol"] = "low"
                elif ev_type in ("fomc", "cpi", "ppi", "nfp", "gdp") and symbol in index_tickers:
                    event["impact_on_symbol"] = "high"
                elif ev_type == "opex" and symbol in index_tickers:
                    event["impact_on_symbol"] = "medium"
                else:
                    event["impact_on_symbol"] = "low"

        return jsonify({"badges": badges, "events": events, "days_ahead": days_ahead, "count": len(events)})

    except Exception as e:
        import traceback

        traceback.print_exc()
        return jsonify({"error": f"Failed to get cockpit events: {str(e)}", "badges": [], "events": []}), 500

