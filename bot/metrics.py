"""Prometheus metrics for the SPX IC bot.

Exposes a /metrics endpoint (port 9101 by default) for Prometheus scraping.
Also provides structured JSON logging helpers for all critical events.

Metrics exposed:
  bot_heartbeat_timestamp_seconds    — Unix timestamp of last scheduler tick
  bot_entries_paused                 — 1 if entries are paused, 0 if running
  bot_open_positions_total           — Current open iron condor count
  bot_cumulative_pnl_dollars         — Cumulative P&L in dollars
  bot_vd2_contracts                  — Current VD2 contract count
  bot_trades_total{result}           — Counter of closed trades by result
  bot_job_last_run_seconds{job}      — Unix timestamp of last run per job
  bot_job_errors_total{job}          — Error counter per job
  bot_order_fills_total{config_id}   — Filled entry/exit orders
  bot_order_errors_total{config_id}  — Failed order placements
"""
from __future__ import annotations

import logging
import os
import threading
import time
from datetime import datetime

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prometheus metrics (lazy-init so import doesn't fail if library missing)
# ---------------------------------------------------------------------------

_prom_available = False
_metrics: dict = {}


def _init_prometheus():
    global _prom_available, _metrics
    if _prom_available or _metrics:
        return
    try:
        from prometheus_client import Gauge, Counter, start_http_server
        _metrics = {
            "heartbeat": Gauge(
                "bot_heartbeat_timestamp_seconds",
                "Unix timestamp of last scheduler heartbeat",
            ),
            "paused": Gauge(
                "bot_entries_paused",
                "1 if bot entries are paused, 0 if running",
            ),
            "open_positions": Gauge(
                "bot_open_positions_total",
                "Number of currently open iron condor positions",
            ),
            "cumulative_pnl": Gauge(
                "bot_cumulative_pnl_dollars",
                "Cumulative P&L in dollars since inception",
            ),
            "vd2_contracts": Gauge(
                "bot_vd2_contracts",
                "Current VD2 contract count",
            ),
            "trades_total": Counter(
                "bot_trades_total",
                "Total closed trades by result",
                ["result"],  # profit_target | stop_loss | time_stop | spx_drop | vix_spike
            ),
            "job_last_run": Gauge(
                "bot_job_last_run_seconds",
                "Unix timestamp of last run for each scheduled job",
                ["job"],
            ),
            "job_errors": Counter(
                "bot_job_errors_total",
                "Error count per scheduled job",
                ["job"],
            ),
            "order_fills": Counter(
                "bot_order_fills_total",
                "Successful order fills by config",
                ["config_id", "side"],  # side: entry | exit
            ),
            "order_errors": Counter(
                "bot_order_errors_total",
                "Failed order placements by config",
                ["config_id"],
            ),
        }
        _prom_available = True
    except ImportError:
        logger.debug("prometheus_client not installed — metrics disabled")


def start_metrics_server(port: int | None = None) -> None:
    """Start the Prometheus HTTP metrics server in a background thread."""
    _init_prometheus()
    if not _prom_available:
        return
    from prometheus_client import start_http_server
    metrics_port = port or int(os.environ.get("BOT_METRICS_PORT", "9101"))
    try:
        start_http_server(metrics_port)
        logger.info("Prometheus metrics server started on port %d", metrics_port)
    except OSError as e:
        if "Address already in use" in str(e):
            logger.warning("Metrics port %d already in use — skipping (another bot instance has it)", metrics_port)
        else:
            logger.warning("Metrics server failed: %s", e)


def _m(name: str):
    """Return a metric by name, or None if prometheus not available."""
    _init_prometheus()
    return _metrics.get(name)


# ---------------------------------------------------------------------------
# Metric update helpers (called from bot jobs)
# ---------------------------------------------------------------------------

def record_heartbeat() -> None:
    m = _m("heartbeat")
    if m:
        m.set(time.time())


def record_state(open_positions: int, cumulative_pnl: float,
                 vd2_contracts: int, paused: bool) -> None:
    for name, val in [
        ("open_positions", open_positions),
        ("cumulative_pnl", cumulative_pnl),
        ("vd2_contracts", vd2_contracts),
        ("paused", 1 if paused else 0),
    ]:
        m = _m(name)
        if m:
            m.set(val)


def record_job_run(job_name: str) -> None:
    m = _m("job_last_run")
    if m:
        m.labels(job=job_name).set(time.time())


def record_job_error(job_name: str) -> None:
    m = _m("job_errors")
    if m:
        m.labels(job=job_name).inc()


def record_trade_closed(reason: str) -> None:
    m = _m("trades_total")
    if m:
        m.labels(result=reason).inc()


def record_order_fill(config_id: str, side: str) -> None:
    m = _m("order_fills")
    if m:
        m.labels(config_id=config_id, side=side).inc()


def record_order_error(config_id: str) -> None:
    m = _m("order_errors")
    if m:
        m.labels(config_id=config_id).inc()


# ---------------------------------------------------------------------------
# Structured JSON logging
# ---------------------------------------------------------------------------

import json


def _log_event(event_type: str, **kwargs) -> None:
    """Emit a structured JSON log line for critical events."""
    record = {
        "ts": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z",
        "event": event_type,
        **{k: v for k, v in kwargs.items() if v is not None},
    }
    logger.info("EVENT %s", json.dumps(record))


def log_entry(config_id: str, expiration: str, n_contracts: int,
              credit: float, spx: float, vix: float) -> None:
    _log_event("trade_entry", config_id=config_id, expiration=expiration,
               n_contracts=n_contracts, credit=credit, spx=spx, vix=vix)


def log_exit(config_id: str, reason: str, pnl: float, cumulative_pnl: float) -> None:
    _log_event("trade_exit", config_id=config_id, reason=reason,
               pnl=pnl, cumulative_pnl=cumulative_pnl)


def log_circuit_breaker(drawdown_pct: float) -> None:
    _log_event("circuit_breaker_triggered", drawdown_pct=drawdown_pct)


def log_compound(new_vd2: int, cumulative_pnl: float) -> None:
    _log_event("compound_milestone", new_vd2_contracts=new_vd2, cumulative_pnl=cumulative_pnl)
