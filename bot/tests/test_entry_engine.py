"""Tests for bot/entry_engine.py — verify entry filter pass/fail at edge cases."""
import sys
from datetime import date
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from bot.entry_engine import should_enter, build_entry
from bot.market_data import MarketSnapshot
from bot.config import SWEET_CONFIG, VD2_CONFIG, MAX_OPEN_IC_TOTAL


def _snap(
    spx=5000.0,
    vix=20.0,
    sma50=4800.0,
    rv_ratio=1.0,
    gap_pct=0.0,
    is_news_day=False,
) -> MarketSnapshot:
    return MarketSnapshot(
        spx_price=spx,
        vix_level=vix,
        spx_open=spx,
        prev_close=spx,
        sma50=sma50,
        rv_ratio=rv_ratio,
        gap_pct=gap_pct,
        is_news_day=is_news_day,
    )


class TestSweetEntryFilters:
    """SWEET: VIX 16-24, SPX > SMA50."""

    def test_vix_exactly_at_min_passes(self):
        ok, _ = should_enter(SWEET_CONFIG, _snap(vix=16.0), [])
        assert ok

    def test_vix_below_min_fails(self):
        ok, reason = should_enter(SWEET_CONFIG, _snap(vix=15.9), [])
        assert not ok
        assert "VIX" in reason

    def test_vix_exactly_at_max_passes(self):
        ok, _ = should_enter(SWEET_CONFIG, _snap(vix=24.0), [])
        assert ok

    def test_vix_above_max_fails(self):
        ok, reason = should_enter(SWEET_CONFIG, _snap(vix=24.1), [])
        assert not ok
        assert "VIX" in reason

    def test_spx_above_sma_passes(self):
        ok, _ = should_enter(SWEET_CONFIG, _snap(spx=5000, sma50=4999), [])
        assert ok

    def test_spx_below_sma_fails(self):
        ok, reason = should_enter(SWEET_CONFIG, _snap(spx=4999, sma50=5000), [])
        assert not ok
        assert "SMA50" in reason

    def test_spx_equal_sma_fails(self):
        """SPX must be strictly > SMA50 (equal does not qualify)."""
        ok, reason = should_enter(SWEET_CONFIG, _snap(spx=5000, sma50=5000), [])
        assert not ok
        assert "SMA50" in reason


class TestVd2EntryFilters:
    """VD2: VIX <= 28, SPX > SMA50, rv_ratio <= 1.05."""

    def test_vix_28_passes(self):
        ok, _ = should_enter(VD2_CONFIG, _snap(vix=28.0), [])
        assert ok

    def test_vix_28_1_fails(self):
        ok, reason = should_enter(VD2_CONFIG, _snap(vix=28.1), [])
        assert not ok
        assert "VIX" in reason

    def test_rv_ratio_1_04_passes(self):
        ok, _ = should_enter(VD2_CONFIG, _snap(rv_ratio=1.04), [])
        assert ok

    def test_rv_ratio_1_05_passes(self):
        ok, _ = should_enter(VD2_CONFIG, _snap(rv_ratio=1.05), [])
        assert ok

    def test_rv_ratio_1_06_fails(self):
        ok, reason = should_enter(VD2_CONFIG, _snap(rv_ratio=1.06), [])
        assert not ok
        assert "rv_ratio" in reason

    def test_vd2_no_vix_min(self):
        """VD2 has no minimum VIX — should allow low VIX."""
        ok, _ = should_enter(VD2_CONFIG, _snap(vix=10.0), [])
        assert ok


class TestHardSafetyRules:
    def test_vix_hard_cap_35(self):
        from bot.config import VIX_ABSOLUTE_CAP
        ok, reason = should_enter(VD2_CONFIG, _snap(vix=VIX_ABSOLUTE_CAP), [])
        assert not ok
        assert "cap" in reason.lower()

    def test_max_open_ic_cap(self):
        """Should refuse entry when max open ICs reached."""
        class FakePos:
            config_id = "VD2_ENTRY_105"

        fake_positions = [FakePos() for _ in range(MAX_OPEN_IC_TOTAL)]
        ok, reason = should_enter(VD2_CONFIG, _snap(), fake_positions)
        assert not ok
        assert "cap" in reason.lower() or "Max" in reason


class TestBuildEntry:
    def test_returns_entry_order(self):
        snap = _snap()
        with patch("bot.entry_engine._find_target_expiration",
                   return_value=date(2026, 3, 14)):
            order = build_entry(SWEET_CONFIG, snap, n_contracts=4, today=date(2026, 3, 1))
        assert order is not None
        assert order.config_id == "15_VIX_SWEET_SPOT"
        assert order.n_contracts == 4
        assert order.k_put_short < 5000
        assert order.k_call_short > 5000
        assert order.credit > 0

    def test_returns_none_when_expiry_too_close(self):
        snap = _snap()
        with patch("bot.entry_engine._find_target_expiration",
                   return_value=date(2026, 3, 3)):  # only 2 days away
            order = build_entry(SWEET_CONFIG, snap, n_contracts=4, today=date(2026, 3, 1))
        assert order is None

    def test_vd2_entry_with_10_contracts(self):
        snap = _snap(vix=22.0)
        with patch("bot.entry_engine._find_target_expiration",
                   return_value=date(2026, 3, 14)):
            order = build_entry(VD2_CONFIG, snap, n_contracts=10, today=date(2026, 3, 1))
        assert order is not None
        assert order.n_contracts == 10
