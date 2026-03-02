"""Tests for bot/metrics.py — verify metrics helpers don't crash."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from bot import metrics


class TestMetricsNoOp:
    """All metrics calls must be safe no-ops if prometheus_client is absent."""

    def test_record_heartbeat_no_crash(self):
        metrics.record_heartbeat()

    def test_record_state_no_crash(self):
        metrics.record_state(open_positions=3, cumulative_pnl=12000.0,
                             vd2_contracts=8, paused=False)

    def test_record_job_run_no_crash(self):
        metrics.record_job_run("entry_engine")

    def test_record_job_error_no_crash(self):
        metrics.record_job_error("entry_engine")

    def test_record_trade_closed_no_crash(self):
        metrics.record_trade_closed("profit_target")

    def test_record_order_fill_no_crash(self):
        metrics.record_order_fill("15_VIX_SWEET_SPOT", "entry")

    def test_record_order_error_no_crash(self):
        metrics.record_order_error("VD2_ENTRY_105")


class TestStructuredLogs:
    def test_log_entry_no_crash(self):
        metrics.log_entry("VD2_ENTRY_105", "2026-03-15", 8, 0.92, 5050.0, 19.5)

    def test_log_exit_no_crash(self):
        metrics.log_exit("VD2_ENTRY_105", "profit_target", 5520.0, 47320.0)

    def test_log_circuit_breaker_no_crash(self):
        metrics.log_circuit_breaker(8.5)

    def test_log_compound_no_crash(self):
        metrics.log_compound(9, 30100.0)
