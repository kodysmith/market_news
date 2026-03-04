"""Publish bot entry decisions to Supabase compute_result_cache as TradeIdeas.

Called after run_entry_checks() so the market news app's Trade Ideas screen
shows live SPX Iron Condor bot signals alongside the GEX-based ideas.

The result dict matches the shape _parseTradeIdeasResponse() in the Flutter app
expects: {"unlocked": [...], "preview": [...], "diagnostics": {...}}.
"""
from __future__ import annotations

import logging
import os
import uuid
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

_SYMBOL = "SPX"
_TASK_TYPE = "trade_ideas"

_CONFIG_LABELS = {
    "15_VIX_SWEET_SPOT": "Sweet Spot Iron Condor",
    "VD2_ENTRY_105": "VD2 Iron Condor",
}


def _get_client():
    from supabase import create_client
    url = os.environ.get("SUPABASE_URL", "")
    key = os.environ.get("SUPABASE_SECRET_KEY") or os.environ.get("SUPABASE_KEY", "")
    if not url or not key:
        return None
    return create_client(url, key)


def _build_trade_idea(
    config_id: str,
    ok: bool,
    reason: str,
    order,   # EntryOrder | None
    snapshot,  # MarketSnapshot
) -> dict:
    """Format one bot decision as a TradeIdea dict the Flutter app understands."""
    now = datetime.now(timezone.utc).isoformat()
    label = _CONFIG_LABELS.get(config_id, config_id)

    executions: list = []
    best = None

    if ok and order is not None:
        width = order.k_put_short - order.k_put_long
        net_credit = order.credit          # BS credit per unit
        max_profit = round(net_credit * 100, 2)
        max_loss = round((width - net_credit) * 100, 2)
        roc = round(max_profit / max_loss, 4) if max_loss > 0 else 0.0
        pop = 0.68                         # rough estimate for ~15-delta short strikes
        score = round(roc * pop, 4)
        dte = (order.expiration - datetime.now(timezone.utc).date()).days
        expiry_str = order.expiration.isoformat()

        legs = [
            {"action": "SELL", "type": "PUT",  "strike": order.k_put_short,
             "priceMid": 0.0, "bid": 0.0, "ask": 0.0, "oi": 0, "volume": 0},
            {"action": "BUY",  "type": "PUT",  "strike": order.k_put_long,
             "priceMid": 0.0, "bid": 0.0, "ask": 0.0, "oi": 0, "volume": 0},
            {"action": "SELL", "type": "CALL", "strike": order.k_call_short,
             "priceMid": 0.0, "bid": 0.0, "ask": 0.0, "oi": 0, "volume": 0},
            {"action": "BUY",  "type": "CALL", "strike": order.k_call_long,
             "priceMid": 0.0, "bid": 0.0, "ask": 0.0, "oi": 0, "volume": 0},
        ]
        metrics = {
            "netCredit": round(net_credit, 4),
            "width": width,
            "maxProfit": max_profit,
            "maxLoss": max_loss,
            "roc": roc,
            "breakeven": round(order.k_put_short - net_credit, 2),
            "pop": pop,
            "ev": 0.0,
            "wallBufferPct": 0.0,
            "liquidityScore": 0.0,
            "score": score,
        }
        executions = [{"expiry": expiry_str, "dte": dte, "legs": legs, "metrics": metrics}]
        best = {
            "expiry": expiry_str,
            "dte": dte,
            "netCredit": max_profit,      # dollars (not per-unit)
            "maxProfit": max_profit,
            "maxLoss": max_loss,
            "roc": roc,
            "pop": pop,
            "ev": 0.0,
            "wallBufferPct": 0.0,
            "liquidityGrade": "B",
            "score": score,
        }

    confidence = (
        "HIGH" if ok and 18 <= snapshot.vix_level <= 22
        else "MEDIUM" if ok
        else "LOW"
    )

    return {
        "id": str(uuid.uuid4()),
        "symbol": _SYMBOL,
        "asOf": now,
        "strategy": "iron_condor",
        "label": label,
        "allowed": ok,
        "mode": "INTRADAY_ONLY",
        "confidence": confidence,
        "reasons": [reason],
        "status": "UNLOCKED" if ok else "PREVIEW",
        "blockReason": None if ok else "REGIME_BLOCKED",
        "preMarket": False,
        "marketStatusNote": None,
        "executions": executions,
        "best": best,
        "contextSnapshot": {
            "spot": snapshot.spx_price,
            "putWall": None,
            "callWall": None,
            "flipLine": None,
            "depinRisk": None,
            "gexState": "NEUTRAL",
            "pinConfidence": "LOW",
            "earningsClusterScore": 1 if snapshot.is_news_day else 0,
            "macroEventNext24h": snapshot.is_news_day,
        },
    }


def publish_bot_decisions(decisions: list, snapshot) -> None:
    """Upsert all bot entry decisions into compute_result_cache as TradeIdeas.

    Args:
        decisions: list of (config_id, ok, reason, order_or_None) from run_entry_checks()
        snapshot:  MarketSnapshot used for context
    """
    client = _get_client()
    if client is None:
        logger.debug("Supabase not configured — skipping trade idea publish")
        return

    unlocked = []
    preview = []
    for config_id, ok, reason, order in decisions:
        idea = _build_trade_idea(config_id, ok, reason, order, snapshot)
        (unlocked if ok else preview).append(idea)

    result = {
        "unlocked": unlocked,
        "preview": preview,
        "diagnostics": {
            "source": "spx_ic_bot",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "vix": snapshot.vix_level,
            "spx": snapshot.spx_price,
        },
    }

    try:
        client.table("compute_result_cache").upsert(
            {
                "symbol": _SYMBOL,
                "task_type": _TASK_TYPE,
                "result": result,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
            on_conflict="symbol,task_type",
        ).execute()
        logger.info(
            "Published bot trade ideas → %d unlocked, %d preview (SPX)",
            len(unlocked), len(preview),
        )
    except Exception as e:
        logger.warning("Failed to publish bot trade ideas: %s", e)
