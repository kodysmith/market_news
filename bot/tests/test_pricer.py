"""Tests for bot/pricer.py — verify BS prices match evaluate_enhanced_cashflow output."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from bot.pricer import (
    compute_strikes,
    price_iron_condor,
    compute_credit,
    spread_pnl_pct,
    dte_remaining,
)
from scripts.optimize_spx_verticals_historical import (
    bs_put_price,
    bs_call_price,
)


class TestComputeStrikes:
    def test_put_short_below_spot(self):
        k_ps, k_pl, k_cs, k_cl = compute_strikes(spx=5000, vix=18, dte=14)
        assert k_ps < 5000, "Put short strike must be below spot"

    def test_call_short_above_spot(self):
        k_ps, k_pl, k_cs, k_cl = compute_strikes(spx=5000, vix=18, dte=14)
        assert k_cs > 5000, "Call short strike must be above spot"

    def test_put_long_below_put_short(self):
        k_ps, k_pl, k_cs, k_cl = compute_strikes(spx=5000, vix=18, dte=14)
        assert k_pl < k_ps, "Put long must be below put short"

    def test_call_long_above_call_short(self):
        k_ps, k_pl, k_cs, k_cl = compute_strikes(spx=5000, vix=18, dte=14)
        assert k_cl > k_cs, "Call long must be above call short"

    def test_width_50_points(self):
        k_ps, k_pl, k_cs, k_cl = compute_strikes(spx=5000, vix=18, dte=14, width=50)
        assert abs((k_ps - k_pl) - 50) < 1, "Put spread width should be 50"
        assert abs((k_cl - k_cs) - 50) < 1, "Call spread width should be 50"

    def test_strikes_rounded_to_5(self):
        k_ps, k_pl, k_cs, k_cl = compute_strikes(spx=5123, vix=20, dte=14)
        for strike in (k_ps, k_pl, k_cs, k_cl):
            assert strike % 5 == 0, f"Strike {strike} not a multiple of 5"

    def test_symmetric_around_spot(self):
        """Put and call short strikes should be roughly equidistant from spot at 0.15 delta."""
        k_ps, k_pl, k_cs, k_cl = compute_strikes(spx=5000, vix=18, dte=14, delta=0.15)
        dist_put = 5000 - k_ps
        dist_call = k_cs - 5000
        # Typically put side is slightly wider due to skew, but both should be ~same order
        assert 0.5 < dist_put / dist_call < 2.0, (
            f"Strikes should be roughly equidistant from spot; put={dist_put} call={dist_call}"
        )


class TestPriceIronCondor:
    def test_positive_value_atm(self):
        """At-the-money-ish position should have positive value."""
        val = price_iron_condor(5000, 18, 14, 4950, 4900, 5050, 5100)
        assert val > 0

    def test_zero_dte_worth_max_or_zero(self):
        """At expiration, spread is either 0 (OTM) or at max loss (ITM)."""
        # Deep OTM for both sides — all legs expire worthless
        val = price_iron_condor(5000, 1, 0, 4000, 3950, 6000, 6050)
        assert val == pytest.approx(0.0, abs=0.01)

    def test_matches_manual_bs(self):
        """Verify against manual BS calculation."""
        spx, vix, dte = 5000.0, 20.0, 14
        k_ps, k_pl, k_cs, k_cl = 4900.0, 4850.0, 5100.0, 5150.0
        sigma = vix / 100
        T = dte / 365

        manual_put_spread = bs_put_price(spx, k_ps, T, sigma) - bs_put_price(spx, k_pl, T, sigma)
        manual_call_spread = bs_call_price(spx, k_cs, T, sigma) - bs_call_price(spx, k_cl, T, sigma)
        expected = max(manual_put_spread, 0) + max(manual_call_spread, 0)

        result = price_iron_condor(spx, vix, dte, k_ps, k_pl, k_cs, k_cl)
        assert result == pytest.approx(expected, rel=1e-6)

    def test_higher_vix_increases_value(self):
        """Higher IV should increase spread value."""
        val_low = price_iron_condor(5000, 15, 14, 4900, 4850, 5100, 5150)
        val_high = price_iron_condor(5000, 30, 14, 4900, 4850, 5100, 5150)
        assert val_high > val_low


class TestSpreadPnlPct:
    def test_full_decay(self):
        """Spread worth 0 means 100% profit."""
        assert spread_pnl_pct(0.0, 1.0) == pytest.approx(1.0)

    def test_half_decay(self):
        assert spread_pnl_pct(0.5, 1.0) == pytest.approx(0.5)

    def test_no_decay(self):
        assert spread_pnl_pct(1.0, 1.0) == pytest.approx(0.0)

    def test_stop_territory(self):
        """Spread worth 2× credit means -100% (stop territory)."""
        assert spread_pnl_pct(2.0, 1.0) == pytest.approx(-1.0)

    def test_zero_credit_no_crash(self):
        assert spread_pnl_pct(0.5, 0.0) == pytest.approx(0.0)


class TestDteRemaining:
    def test_future_expiry(self):
        from datetime import date
        exp = date(2026, 3, 15)
        today = date(2026, 3, 1)
        assert dte_remaining(exp, today) == 14

    def test_today_expiry(self):
        from datetime import date
        today = date(2026, 3, 1)
        assert dte_remaining(today, today) == 0

    def test_past_expiry_clamps_to_zero(self):
        from datetime import date
        exp = date(2026, 2, 28)
        today = date(2026, 3, 1)
        assert dte_remaining(exp, today) == 0
