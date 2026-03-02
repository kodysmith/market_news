"""Tests for bot/exit_engine.py — verify exit signals fire at correct thresholds."""
import sys
from datetime import date
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from bot.exit_engine import check_exit, scan_all_exits
from bot.market_data import MarketSnapshot
from bot.state_manager import Position


def _snap(spx=5000.0, vix=18.0) -> MarketSnapshot:
    return MarketSnapshot(
        spx_price=spx,
        vix_level=vix,
        spx_open=spx,
        prev_close=spx,
        sma50=4800.0,
        rv_ratio=1.0,
        gap_pct=0.0,
        is_news_day=False,
    )


def _pos(
    config_id="15_VIX_SWEET_SPOT",
    credit=1.0,           # per-contract credit ($/unit)
    expiration=None,
    spx_entry=5000.0,
    vix_entry=18.0,
    k_put_short=4900.0,
    k_put_long=4850.0,
    k_call_short=5100.0,
    k_call_long=5150.0,
) -> Position:
    if expiration is None:
        expiration = date(2026, 3, 15)
    return Position(
        id="test-uuid",
        config_id=config_id,
        entry_date=date(2026, 3, 1),
        expiration=expiration,
        k_put_short=k_put_short,
        k_put_long=k_put_long,
        k_call_short=k_call_short,
        k_call_long=k_call_long,
        credit_collected=credit,
        n_contracts=4,
        spx_entry=spx_entry,
        vix_entry=vix_entry,
    )


class TestProfitTarget:
    def test_profit_target_fires_when_spread_decayed_50pct(self):
        """Spread worth 0.5 × credit (50% decay) → profit_target."""
        pos = _pos(credit=1.0)
        # Mock price_iron_condor to return 50% of credit
        import bot.exit_engine as ee
        from unittest.mock import patch

        with patch("bot.exit_engine.price_iron_condor", return_value=0.5):
            signal = check_exit(pos, _snap(), today=date(2026, 3, 8))
        assert signal is not None
        assert signal.reason == "profit_target"

    def test_hold_when_spread_not_yet_decayed(self):
        pos = _pos(credit=1.0)
        from unittest.mock import patch

        with patch("bot.exit_engine.price_iron_condor", return_value=0.8):
            signal = check_exit(pos, _snap(), today=date(2026, 3, 8))
        assert signal is None


class TestStopLoss:
    def test_stop_fires_at_2x_credit(self):
        pos = _pos(credit=1.0)
        from unittest.mock import patch

        with patch("bot.exit_engine.price_iron_condor", return_value=2.0):
            signal = check_exit(pos, _snap(), today=date(2026, 3, 8))
        assert signal is not None
        assert signal.reason == "stop_loss"

    def test_hold_just_below_stop(self):
        pos = _pos(credit=1.0)
        from unittest.mock import patch

        with patch("bot.exit_engine.price_iron_condor", return_value=1.99):
            signal = check_exit(pos, _snap(), today=date(2026, 3, 8))
        # Should NOT trigger stop (1.99 < 2.0)
        assert signal is None or signal.reason != "stop_loss"


class TestTimeStop:
    """SWEET config has time_stop_dte=2."""

    def test_time_stop_fires_at_2_dte(self):
        pos = _pos(config_id="15_VIX_SWEET_SPOT", credit=1.0,
                   expiration=date(2026, 3, 3))
        from unittest.mock import patch

        with patch("bot.exit_engine.price_iron_condor", return_value=0.9):
            signal = check_exit(pos, _snap(), today=date(2026, 3, 1))
        # 2 days remaining; time_stop_dte=2 → should fire
        assert signal is not None
        assert signal.reason == "time_stop"

    def test_time_stop_doesnt_fire_at_3_dte(self):
        pos = _pos(config_id="15_VIX_SWEET_SPOT", credit=1.0,
                   expiration=date(2026, 3, 4))
        from unittest.mock import patch

        with patch("bot.exit_engine.price_iron_condor", return_value=0.9):
            signal = check_exit(pos, _snap(), today=date(2026, 3, 1))
        assert signal is None or signal.reason != "time_stop"

    def test_vd2_has_no_time_stop(self):
        """VD2 config has no time_stop_dte — should not fire time stop."""
        pos = _pos(config_id="VD2_ENTRY_105", credit=1.0,
                   expiration=date(2026, 3, 2))  # 1 day remaining
        from unittest.mock import patch

        with patch("bot.exit_engine.price_iron_condor", return_value=0.9):
            signal = check_exit(pos, _snap(), today=date(2026, 3, 1))
        assert signal is None or signal.reason != "time_stop"


class TestScanAllExits:
    def test_no_signals_on_empty_positions(self):
        signals = scan_all_exits([], _snap())
        assert signals == []

    def test_multiple_positions_only_triggered_returned(self):
        from unittest.mock import patch

        pos1 = _pos(credit=1.0, k_put_short=4900.0)  # will hit profit target
        # Use different strikes so the mock can distinguish by k_put_short
        pos2 = _pos(config_id="VD2_ENTRY_105", credit=1.0, k_put_short=4800.0,
                    k_put_long=4750.0)  # will hold

        def mock_pricer(spx, vix, dte, kps, kpl, kcs, kcl):
            # 0.4 triggers profit target (50% rule: 0.4 <= 0.5 * 1.0)
            # 0.7 means hold (0.7 > 0.5 × 1.0)
            if kps == pos1.k_put_short:
                return 0.4
            return 0.7

        with patch("bot.exit_engine.price_iron_condor", side_effect=mock_pricer):
            signals = scan_all_exits([pos1, pos2], _snap())

        assert len(signals) == 1
        assert signals[0].reason == "profit_target"
        assert signals[0].position_id == pos1.id
