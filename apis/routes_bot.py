"""Bot status REST endpoints for the mobile app and monitoring.

Blueprint: /bot/...

Endpoints:
  GET  /bot/status      — Overall bot state (mobile app)
  GET  /bot/positions   — Open positions with current mark P&L
  GET  /bot/trades      — Recent completed trades
  GET  /bot/performance — Monthly P&L summary
  GET  /bot/health      — Health check for uptime monitors / load balancers
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
