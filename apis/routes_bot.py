"""Bot status REST endpoints for the mobile app and monitoring.

Blueprint: /bot/...

Endpoints:
  GET  /bot/status      — Overall bot state (mobile app)
  GET  /bot/positions   — Open positions with current mark P&L
  GET  /bot/trades      — Recent completed trades
  GET  /bot/performance — Monthly P&L summary
  GET  /bot/health      — Health check for uptime monitors / load balancers
  GET  /bot/market-regime — SPX vs SMA50/200, VIX zone, guidance
  GET  /bot/trading-view — Complete morning trading dashboard (regime, GEX, setups, model)
  POST /bot/pause       — Manually pause new entries
  POST /bot/resume      — Resume new entries
"""
from __future__ import annotations

import logging
import time
from datetime import date, datetime

from flask import Blueprint, jsonify, request

logger = logging.getLogger(__name__)

bp = Blueprint("bot", __name__, url_prefix="/bot")

# Track when the API itself last served a request (liveness proxy)
_last_api_call_ts: float = time.time()


def _state() -> object:
    """Lazy-instantiate StateManager (avoids Supabase init at import time)."""
    from bot.state_manager import StateManager
    return StateManager()


# ---------------------------------------------------------------------------
# GET /bot/status
# ---------------------------------------------------------------------------

@bp.route("/health")
def health():
    """Health check endpoint for uptime monitors.

    Returns 200 OK with basic status. Suitable for:
      - Cloud Run health check
      - UptimeRobot / Better Uptime HTTP checks
      - Load balancer health probes
    """
    global _last_api_call_ts
    _last_api_call_ts = time.time()
    try:
        sm = _state()
        paused = sm.is_paused()
        return jsonify({
            "status": "ok",
            "bot_paused": paused,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        })
    except Exception as e:
        logger.error("GET /bot/health error: %s", e)
        return jsonify({"status": "error", "detail": str(e)}), 503


@bp.route("/status")
def status():
    """Overall bot status summary."""
    try:
        from bot.config import TOTAL_ACCOUNT, VD2_CONTRACTS_START
        sm = _state()
        open_positions = sm.get_open_positions()
        cumulative_pnl = sm.get_cumulative_pnl()
        vd2_contracts = sm.get_vd2_contract_count()
        paused = sm.is_paused()
        ytd_roi = cumulative_pnl / TOTAL_ACCOUNT * 100

        return jsonify({
            "status": "paused" if paused else "running",
            "open_positions": len(open_positions),
            "cumulative_pnl": round(cumulative_pnl, 2),
            "vd2_contracts": vd2_contracts,
            "ytd_roi_pct": round(ytd_roi, 2),
            "last_check": datetime.utcnow().isoformat() + "Z",
        })
    except Exception as e:
        logger.error("GET /bot/status error: %s", e)
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
# GET /bot/positions
# ---------------------------------------------------------------------------

@bp.route("/positions")
def positions():
    """List of open positions with current model P&L."""
    try:
        from bot.state_manager import StateManager
        from bot.pricer import price_iron_condor, dte_remaining

        # Get a fresh VIX/SPX snapshot for mark-to-market
        try:
            from bot.market_data import get_market_snapshot
            snap = get_market_snapshot()
            spx = snap.spx_price
            vix = snap.vix_level
        except Exception:
            spx = vix = None

        sm = StateManager()
        open_positions = sm.get_open_positions()
        result = []

        for pos in open_positions:
            dte = dte_remaining(pos.expiration)
            mark = None
            pnl_pct = None
            if spx and vix:
                try:
                    mark = price_iron_condor(
                        spx, vix, dte,
                        pos.k_put_short, pos.k_put_long,
                        pos.k_call_short, pos.k_call_long,
                    )
                    pnl_pct = round(
                        (pos.credit_collected - mark) / pos.credit_collected * 100, 1
                    ) if pos.credit_collected > 0 else None
                except Exception:
                    pass

            result.append({
                "id": pos.id,
                "config_id": pos.config_id,
                "entry_date": pos.entry_date.isoformat(),
                "expiration": pos.expiration.isoformat(),
                "dte_remaining": dte,
                "k_put_short": pos.k_put_short,
                "k_put_long": pos.k_put_long,
                "k_call_short": pos.k_call_short,
                "k_call_long": pos.k_call_long,
                "credit_collected": pos.credit_collected,
                "n_contracts": pos.n_contracts,
                "mark": round(mark, 4) if mark is not None else None,
                "pnl_pct": pnl_pct,
                "max_loss": round(pos.total_max_loss, 2),
            })

        return jsonify(result)
    except Exception as e:
        logger.error("GET /bot/positions error: %s", e)
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
# GET /bot/trades
# ---------------------------------------------------------------------------

@bp.route("/trades")
def trades():
    """Recent completed trades (last 90 days)."""
    try:
        days = int(request.args.get("days", 90))
        sm = _state()
        rows = sm.get_recent_trades(days)
        return jsonify(rows)
    except Exception as e:
        logger.error("GET /bot/trades error: %s", e)
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
# GET /bot/performance
# ---------------------------------------------------------------------------

@bp.route("/performance")
def performance():
    """Monthly P&L summary."""
    try:
        sm = _state()
        monthly = sm.get_monthly_summary()
        cumulative_pnl = sm.get_cumulative_pnl()

        from bot.config import TOTAL_ACCOUNT
        ytd_roi = cumulative_pnl / TOTAL_ACCOUNT * 100

        return jsonify({
            "monthly": monthly,
            "cumulative_pnl": round(cumulative_pnl, 2),
            "ytd_roi_pct": round(ytd_roi, 2),
            "total_account": TOTAL_ACCOUNT,
        })
    except Exception as e:
        logger.error("GET /bot/performance error: %s", e)
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
# GET /bot/trading-view — Complete morning trading dashboard
# ---------------------------------------------------------------------------

@bp.route("/trading-view")
def trading_view():
    """One-stop morning trading dashboard.

    Returns everything needed for manual SPX trading decisions:
    - Regime tier (BEAR/CAUTION/NEUTRAL/BULL) with score
    - GEX state (positive/negative, net GEX, walls)
    - Best trade setups (from BS pricing)
    - Quant signals (danger score, GARCH, flow)
    - Open positions with P&L
    """
    try:
        import json as _json
        from bot.market_data import get_market_snapshot
        from scripts.optimize_spx_verticals_historical import (
            bs_put_price, bs_call_price, strike_for_delta,
        )

        snap = get_market_snapshot()
        spx = snap.spx_price
        vix = snap.vix_level
        sma50 = snap.sma50

        result = {
            "spx": round(spx, 2),
            "vix": round(vix, 2),
            "sma50": round(sma50, 2),
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }

        # --- Regime tier ---
        try:
            from pathlib import Path
            state_path = Path(__file__).resolve().parent.parent / "data" / "regime_state.json"
            if state_path.exists():
                saved = _json.load(open(state_path))
                result["regime"] = {
                    "tier": saved.get("tier", "UNKNOWN"),
                    "score": saved.get("score", 0),
                    "days_in_tier": saved.get("days_in_tier", 0),
                    "history": saved.get("history", [])[-5:],
                }
                tier = saved.get("tier", "NEUTRAL")
                if tier == "BEAR":
                    result["regime"]["action"] = "SGOV — no trades, protect capital"
                elif tier == "CAUTION":
                    result["regime"]["action"] = "25% size ICs — testing the water"
                elif tier == "BULL":
                    result["regime"]["action"] = "Full size ICs + butterflies — max income"
                else:
                    result["regime"]["action"] = "Full size ICs — normal conditions"
            else:
                result["regime"] = {"tier": "UNKNOWN", "action": "Regime state not computed yet"}
        except Exception as e:
            result["regime"] = {"tier": "ERROR", "error": str(e)}

        # --- GEX ---
        try:
            from bot.daily_cashflow import check_gex_regime
            gex_regime, net_gex = check_gex_regime()
            result["gex"] = {
                "regime": gex_regime,
                "net_gex_billions": round(net_gex / 1e9, 1) if net_gex else 0,
                "is_positive": gex_regime == "POSITIVE_GAMMA",
            }
        except Exception as e:
            result["gex"] = {"regime": "UNKNOWN", "error": str(e)}

        # --- Quant signals ---
        try:
            from bot.quant_signals import get_all_quant_signals
            from scripts.optimize_spx_verticals_historical import load_spx_vix
            from datetime import timedelta
            spx_df, _ = load_spx_vix(
                (datetime.utcnow() - timedelta(days=400)).date(),
                datetime.utcnow().date(),
            )
            if not spx_df.empty:
                quant = get_all_quant_signals(spx_df, vix)
                result["quant"] = {
                    "danger_score": quant["danger_score"],
                    "action": quant["action"],
                    "position_scale": quant["position_scale"],
                    "garch_vol_ratio": round(quant["garch"]["vol_ratio"], 2),
                    "vol_expanding": quant["garch"]["vol_expanding"],
                    "vpin": round(quant["flow"]["vpin_proxy"], 2),
                    "toxic_flow": quant["flow"]["toxic_flow"],
                    "dynamic_stop": quant["dynamic_stop_mult"],
                }
        except Exception as e:
            result["quant"] = {"danger_score": -1, "error": str(e)}

        # --- Best trade setups ---
        try:
            sigma = vix / 100.0
            setups = []
            for name, dte, delta, width in [
                ("IC 1DTE", 1, 0.10, 30),
                ("IC 3DTE", 3, 0.12, 30),
                ("IC 14DTE", 14, 0.10, 50),
                ("Bear Call 14DTE", 14, 0.10, 50),
            ]:
                T = dte / 252.0
                kps = strike_for_delta(spx, T, delta, sigma, is_put=True)
                kcs = strike_for_delta(spx, T, delta, sigma, is_put=False)
                if not kps or not kcs:
                    continue
                kps = round(kps / 5) * 5
                kcs = round(kcs / 5) * 5

                if "Bear Call" in name:
                    cr = bs_call_price(spx, kcs, T, sigma) - bs_call_price(spx, kcs + width, T, sigma)
                    cr -= 4 * 0.10 / 2
                    setups.append({
                        "name": name, "type": "bear_call",
                        "strikes": f"C {kcs}/{kcs + width}",
                        "credit": round(cr, 2),
                        "cushion_pts": round(kcs - spx),
                        "cushion_pct": round((kcs - spx) / spx * 100, 1),
                    })
                else:
                    put_cr = bs_put_price(spx, kps, T, sigma) - bs_put_price(spx, kps - width, T, sigma)
                    call_cr = bs_call_price(spx, kcs, T, sigma) - bs_call_price(spx, kcs + width, T, sigma)
                    total = put_cr + call_cr - 8 * 0.10 / 2
                    setups.append({
                        "name": name, "type": "iron_condor",
                        "strikes": f"P{kps}/{kps - width} C{kcs}/{kcs + width}",
                        "credit": round(total, 2),
                        "put_cushion": round(spx - kps),
                        "call_cushion": round(kcs - spx),
                        "range_pct": round((kcs - kps) / spx * 100, 1),
                    })
            result["setups"] = setups
        except Exception as e:
            result["setups"] = []

        # --- Open positions ---
        try:
            from bot.state_manager import StateManager
            from bot.pricer import price_iron_condor, dte_remaining
            sm = StateManager()
            positions = sm.get_open_positions()
            pos_list = []
            for pos in positions:
                dte = dte_remaining(pos.expiration)
                pos_list.append({
                    "config_id": pos.config_id,
                    "expiration": pos.expiration.isoformat(),
                    "dte": dte,
                    "credit": pos.credit_collected,
                    "n_contracts": pos.n_contracts,
                })
            result["positions"] = pos_list
        except Exception:
            result["positions"] = []

        return jsonify(result)
    except Exception as e:
        logger.error("GET /bot/trading-view error: %s", e)
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
# POST /bot/pause
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# GET /bot/market-regime
# ---------------------------------------------------------------------------

@bp.route("/market-regime")
def market_regime():
    """Market regime snapshot: SPX vs SMA50, VIX, trend status.

    Used by the mobile app to show a prominent trend banner and
    guide equity/options positioning decisions.
    """
    try:
        from bot.market_data import get_market_snapshot
        snap = get_market_snapshot()
        above_sma = snap.spx_price > snap.sma50
        distance_pct = ((snap.spx_price - snap.sma50) / snap.sma50 * 100) if snap.sma50 else 0

        # VIX zone classification
        vix = snap.vix_level
        if vix <= 15:
            vix_zone = "LOW"
            vix_label = "Low vol — thin premium"
        elif vix <= 24:
            vix_zone = "SWEET"
            vix_label = "Sweet spot — good premium, controlled risk"
        elif vix <= 35:
            vix_zone = "ELEVATED"
            vix_label = "Elevated — be selective, hedges on"
        else:
            vix_zone = "CRISIS"
            vix_label = "Crisis — no new entries"

        # Actionable guidance
        if above_sma:
            stance = "BULLISH"
            guidance = "Trend is UP. OK to sell premium (ICs), hold equities, buy quality dips."
        else:
            stance = "DEFENSIVE"
            guidance = (
                "SPX below SMA50 — trend is DOWN. "
                "Avoid new IC entries. Hedge or reduce equity exposure. "
                "Short weak names, stay in cash, or buy puts for protection."
            )

        return jsonify({
            "spx_price": round(snap.spx_price, 2),
            "sma50": round(snap.sma50, 2),
            "above_sma50": above_sma,
            "distance_pct": round(distance_pct, 2),
            "vix": round(vix, 2),
            "vix_zone": vix_zone,
            "vix_label": vix_label,
            "rv_ratio": round(snap.rv_ratio, 3) if snap.rv_ratio else None,
            "stance": stance,
            "guidance": guidance,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        })
    except Exception as e:
        logger.error("GET /bot/market-regime error: %s", e)
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
# POST /bot/pause
# ---------------------------------------------------------------------------

@bp.route("/pause", methods=["POST"])
def pause():
    """Manually pause new entries."""
    try:
        sm = _state()
        sm.set_paused(True)
        from bot import notifier
        notifier.bot_paused("Manual pause via API")
        return jsonify({"status": "paused"})
    except Exception as e:
        logger.error("POST /bot/pause error: %s", e)
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
# POST /bot/resume
# ---------------------------------------------------------------------------

@bp.route("/resume", methods=["POST"])
def resume():
    """Resume new entries."""
    try:
        sm = _state()
        sm.set_paused(False)
        from bot import notifier
        notifier.bot_resumed()
        return jsonify({"status": "running"})
    except Exception as e:
        logger.error("POST /bot/resume error: %s", e)
        return jsonify({"error": str(e)}), 500
