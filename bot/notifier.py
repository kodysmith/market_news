"""Firebase FCM push notifications for the SPX IC bot.

Reuses the send_fcm_to_topic pattern from scripts/run_gex_cockpit_precompute.py.
All sends are non-blocking; failures are logged but never crash the bot.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Import FCM helpers from the existing precompute script
from scripts.run_gex_cockpit_precompute import (
    send_fcm_to_topic,
    FCM_TOPIC_MARKET_ALERTS as _TOPIC,
)

FCM_TOPIC = _TOPIC


def _send(title: str, body: str, data: dict | None = None) -> None:
    """Fire-and-forget FCM push. Never raises."""
    try:
        ok = send_fcm_to_topic(_TOPIC, title=title, body=body, data=data or {})
        if not ok:
            logger.debug("FCM send skipped (not configured)")
    except Exception as e:
        logger.warning("notifier._send failed: %s", e)


# ---------------------------------------------------------------------------
# Trade lifecycle alerts
# ---------------------------------------------------------------------------

def trade_opened(
    config_id: str,
    expiration: str,
    k_put_short: float,
    k_put_long: float,
    k_call_short: float,
    k_call_long: float,
    credit: float,
    n_contracts: int,
    fill_price: float,
) -> None:
    title = f"IC Opened: {config_id}"
    body = (
        f"Exp {expiration} | "
        f"Put {k_put_short:.0f}/{k_put_long:.0f} "
        f"Call {k_call_short:.0f}/{k_call_long:.0f} | "
        f"Credit ${fill_price * 100:.0f} × {n_contracts}c"
    )
    _send(title, body, {
        "type": "trade_opened",
        "config_id": config_id,
        "n_contracts": str(n_contracts),
        "fill_price": str(round(fill_price, 4)),
    })


def trade_closed(
    config_id: str,
    reason: str,
    pnl: float,
    cumulative_pnl: float,
    n_contracts: int,
) -> None:
    sign = "+" if pnl >= 0 else ""
    title = f"IC Closed ({reason}): {config_id}"
    body = (
        f"P&L: {sign}${pnl:,.0f} × {n_contracts}c | "
        f"Cumulative: ${cumulative_pnl:,.0f}"
    )
    _send(title, body, {
        "type": "trade_closed",
        "config_id": config_id,
        "reason": reason,
        "pnl": str(round(pnl, 2)),
        "cumulative_pnl": str(round(cumulative_pnl, 2)),
    })


def compound_milestone(new_vd2_count: int, cumulative_pnl: float) -> None:
    title = "Compound Milestone Reached"
    body = (
        f"VD2 contracts increased to {new_vd2_count} | "
        f"Cumulative P&L: ${cumulative_pnl:,.0f}"
    )
    _send(title, body, {
        "type": "compound_milestone",
        "vd2_contracts": str(new_vd2_count),
        "cumulative_pnl": str(round(cumulative_pnl, 2)),
    })


def daily_summary(
    open_positions: int,
    day_pnl: float,
    cumulative_pnl: float,
    ytd_roi_pct: float,
    vd2_contracts: int,
) -> None:
    title = "Bot Daily Summary"
    body = (
        f"Open ICs: {open_positions} | "
        f"Day P&L: ${day_pnl:+,.0f} | "
        f"Cumulative: ${cumulative_pnl:,.0f} | "
        f"YTD ROI: {ytd_roi_pct:.1f}%"
    )
    _send(title, body, {
        "type": "daily_summary",
        "open_positions": str(open_positions),
        "day_pnl": str(round(day_pnl, 2)),
        "cumulative_pnl": str(round(cumulative_pnl, 2)),
        "ytd_roi_pct": str(round(ytd_roi_pct, 2)),
        "vd2_contracts": str(vd2_contracts),
    })


def risk_alert(message: str, drawdown_pct: float) -> None:
    title = "RISK ALERT: Drawdown Threshold"
    body = f"Drawdown: {drawdown_pct:.1f}% | {message}"
    _send(title, body, {
        "type": "risk_alert",
        "drawdown_pct": str(round(drawdown_pct, 2)),
    })


def trend_break(spx_price: float, sma50: float, crossed_above: bool) -> None:
    """Alert when SPX crosses SMA50 — major regime change."""
    if crossed_above:
        title = "TREND: SPX Back Above SMA50"
        body = (
            f"SPX {spx_price:,.0f} > SMA50 {sma50:,.0f}. "
            f"Trend turning bullish — IC entries re-enabled, equities OK."
        )
    else:
        title = "TREND: SPX Broke Below SMA50"
        body = (
            f"SPX {spx_price:,.0f} < SMA50 {sma50:,.0f}. "
            f"Go defensive — no new ICs, hedge equities, short weak names."
        )
    _send(title, body, {
        "type": "trend_break",
        "spx_price": str(round(spx_price, 2)),
        "sma50": str(round(sma50, 2)),
        "crossed_above": str(crossed_above),
    })


def bot_paused(reason: str) -> None:
    _send("Bot Entries PAUSED", reason, {"type": "bot_paused"})


def bot_resumed() -> None:
    _send("Bot Entries RESUMED", "New entries will be evaluated at next scheduled check.",
          {"type": "bot_resumed"})
