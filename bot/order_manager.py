"""IBKR order execution for the SPX IC bot.

Wraps IBKRExecutionClient to place and monitor iron condor combo orders.
Supports paper (port 7497) and live (port 7496) modes.
"""
from __future__ import annotations

import logging
import os
import sys
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@dataclass
class Quote:
    """Bid/ask for one option leg."""
    bid: float
    ask: float

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2


@dataclass
class OrderResult:
    """Result of a placed order."""
    success: bool
    fill_price: float          # actual fill price per unit
    order_id: Optional[str] = None
    message: str = ""


def _port_for_mode(mode: str) -> int:
    if mode == "live":
        return int(os.environ.get("IBKR_LIVE_PORT", "7496"))
    return int(os.environ.get("IBKR_PAPER_PORT", "7497"))


class OrderManager:
    """Places and tracks IBKR combo orders for iron condors."""

    def __init__(self, mode: str = "dry-run"):
        """
        Args:
            mode: 'dry-run' | 'paper' | 'live'
        """
        self.mode = mode
        self._ib = None  # lazy connect

    def _connect(self):
        """Lazy-connect to IBKR TWS."""
        if self._ib is not None:
            return
        if self.mode == "dry-run":
            return  # No connection needed

        try:
            import ib_insync as ib_mod
            port = _port_for_mode(self.mode)
            client_id = int(os.environ.get("IBKR_CLIENT_ID", "10"))
            ib = ib_mod.IB()
            ib.connect("127.0.0.1", port, clientId=client_id, timeout=10)
            self._ib = ib
            logger.info("Connected to IBKR (%s) port=%d", self.mode, port)
        except Exception as e:
            logger.error("IBKR connect failed: %s", e)
            raise

    def _disconnect(self):
        if self._ib:
            try:
                self._ib.disconnect()
            except Exception:
                pass
            self._ib = None

    def get_option_quote(
        self,
        expiry: date,
        strike: float,
        right: str,  # 'P' or 'C'
    ) -> Optional[Quote]:
        """Fetch bid/ask for one SPX option leg from IBKR."""
        if self.mode == "dry-run":
            return None

        try:
            self._connect()
            import ib_insync as ib_mod
            contract = ib_mod.Option(
                symbol="SPX",
                lastTradeDateOrContractMonth=expiry.strftime("%Y%m%d"),
                strike=strike,
                right=right,
                exchange="CBOE",
                currency="USD",
                multiplier="100",
            )
            self._ib.qualifyContracts(contract)
            [ticker] = self._ib.reqTickers(contract)
            bid = float(ticker.bid or 0)
            ask = float(ticker.ask or 0)
            return Quote(bid=bid, ask=ask)
        except Exception as e:
            logger.warning("get_option_quote failed (%s %s %s): %s", expiry, strike, right, e)
            return None

    def _has_put_side(self, entry) -> bool:
        return entry.k_put_short > 0 and entry.k_put_long > 0

    def _has_call_side(self, entry) -> bool:
        return entry.k_call_short > 0 and entry.k_call_long > 0

    def _get_ic_mid(self, entry) -> float:
        """Compute midpoint credit for active legs. Falls back to BS credit if IBKR unavailable."""
        quotes = {}
        if self._has_put_side(entry):
            quotes["put_short"] = self.get_option_quote(entry.expiration, entry.k_put_short, "P")
            quotes["put_long"] = self.get_option_quote(entry.expiration, entry.k_put_long, "P")
        if self._has_call_side(entry):
            quotes["call_short"] = self.get_option_quote(entry.expiration, entry.k_call_short, "C")
            quotes["call_long"] = self.get_option_quote(entry.expiration, entry.k_call_long, "C")

        if quotes and all(q is not None for q in quotes.values()):
            credit = 0.0
            if "put_short" in quotes:
                credit += quotes["put_short"].mid - quotes["put_long"].mid
            if "call_short" in quotes:
                credit += quotes["call_short"].mid - quotes["call_long"].mid
            logger.debug("IBKR mid credit: %.4f", credit)
            return max(credit, 0)
        else:
            logger.debug("Using BS credit fallback: %.4f", entry.credit)
            return entry.credit

    def place_iron_condor(self, entry) -> OrderResult:
        """Place a 4-leg iron condor order.

        In dry-run mode: logs the order but does not submit.
        In paper/live mode: submits combo to IBKR.
        """
        from bot.config import FILL_WAIT_SECONDS, FILL_TIMEOUT_SECONDS, IMPROVE_STEP

        mid_credit = self._get_ic_mid(entry)
        # Ensure credit is never NaN — fall back to BS model credit
        if mid_credit != mid_credit or mid_credit <= 0:  # NaN check
            mid_credit = entry.credit
        if mid_credit != mid_credit or mid_credit <= 0:
            return OrderResult(success=False, fill_price=0, message="Credit is zero or NaN")

        log_msg = (
            f"[{self.mode.upper()}] IC order | {entry.config_id} | "
            f"exp={entry.expiration} | "
            f"put {entry.k_put_short:.0f}/{entry.k_put_long:.0f} | "
            f"call {entry.k_call_short:.0f}/{entry.k_call_long:.0f} | "
            f"credit={mid_credit:.4f} | n={entry.n_contracts}"
        )
        logger.info(log_msg)

        if self.mode == "dry-run":
            return OrderResult(
                success=True,
                fill_price=mid_credit,
                message=f"DRY-RUN: {log_msg}",
            )

        try:
            self._connect()
            import ib_insync as ib_mod

            # Build combo legs — only include active sides
            expiry_str = entry.expiration.strftime("%Y%m%d")
            legs = []
            if self._has_put_side(entry):
                legs.extend([
                    ib_mod.ComboLeg(action="SELL", ratio=1,
                                    conId=self._get_conid("SPX", expiry_str, entry.k_put_short, "P"),
                                    exchange="CBOE"),
                    ib_mod.ComboLeg(action="BUY", ratio=1,
                                    conId=self._get_conid("SPX", expiry_str, entry.k_put_long, "P"),
                                    exchange="CBOE"),
                ])
            if self._has_call_side(entry):
                legs.extend([
                    ib_mod.ComboLeg(action="SELL", ratio=1,
                                    conId=self._get_conid("SPX", expiry_str, entry.k_call_short, "C"),
                                    exchange="CBOE"),
                    ib_mod.ComboLeg(action="BUY", ratio=1,
                                    conId=self._get_conid("SPX", expiry_str, entry.k_call_long, "C"),
                                    exchange="CBOE"),
                ])
            combo = ib_mod.Contract(
                symbol="SPX", secType="BAG", currency="USD",
                exchange="CBOE", comboLegs=legs,
            )

            # Limit order at midpoint (net credit)
            limit_price = round(mid_credit, 2)
            order = ib_mod.LimitOrder(
                action="SELL", totalQuantity=entry.n_contracts,
                lmtPrice=limit_price, tif="DAY",
            )
            trade = self._ib.placeOrder(combo, order)

            # Wait for fill with price improvement
            fill_price = self._wait_for_fill(trade, mid_credit, FILL_WAIT_SECONDS,
                                             FILL_TIMEOUT_SECONDS, IMPROVE_STEP)

            if fill_price is not None:
                return OrderResult(
                    success=True,
                    fill_price=fill_price,
                    order_id=str(trade.order.orderId),
                    message="Filled",
                )
            else:
                return OrderResult(
                    success=False,
                    fill_price=0.0,
                    message="Order not filled within timeout",
                )
        except Exception as e:
            logger.error("place_iron_condor failed: %s", e)
            return OrderResult(success=False, fill_price=0.0, message=str(e))

    def close_iron_condor(self, position, reason: str) -> OrderResult:
        """Place a closing (buy-back) order for an existing IC position."""
        from bot.config import FILL_WAIT_SECONDS, FILL_TIMEOUT_SECONDS, IMPROVE_STEP
        from bot.pricer import price_iron_condor, dte_remaining

        # Compute model debit as starting point
        dte = dte_remaining(position.expiration)
        import yfinance as yf
        try:
            vix = float(yf.Ticker("^VIX").fast_info["lastPrice"])
            spx = float(yf.Ticker("^GSPC").fast_info["lastPrice"])
        except Exception:
            vix = position.vix_entry
            spx = position.spx_entry

        model_debit = price_iron_condor(
            spx, vix, dte,
            position.k_put_short, position.k_put_long,
            position.k_call_short, position.k_call_long,
        )

        log_msg = (
            f"[{self.mode.upper()}] CLOSE IC | {position.config_id} | "
            f"reason={reason} | debit={model_debit:.4f} | n={position.n_contracts}"
        )
        logger.info(log_msg)

        if self.mode == "dry-run":
            return OrderResult(
                success=True,
                fill_price=model_debit,
                message=f"DRY-RUN: {log_msg}",
            )

        try:
            self._connect()
            import ib_insync as ib_mod

            expiry_str = position.expiration.strftime("%Y%m%d")
            legs = [
                ib_mod.ComboLeg(action="BUY", ratio=1,
                                conId=self._get_conid("SPX", expiry_str, position.k_put_short, "P"),
                                exchange="CBOE"),
                ib_mod.ComboLeg(action="SELL", ratio=1,
                                conId=self._get_conid("SPX", expiry_str, position.k_put_long, "P"),
                                exchange="CBOE"),
                ib_mod.ComboLeg(action="BUY", ratio=1,
                                conId=self._get_conid("SPX", expiry_str, position.k_call_short, "C"),
                                exchange="CBOE"),
                ib_mod.ComboLeg(action="SELL", ratio=1,
                                conId=self._get_conid("SPX", expiry_str, position.k_call_long, "C"),
                                exchange="CBOE"),
            ]
            combo = ib_mod.Contract(
                symbol="SPX", secType="BAG", currency="USD",
                exchange="CBOE", comboLegs=legs,
            )
            limit_price = round(model_debit, 2)
            order = ib_mod.LimitOrder(
                action="BUY", totalQuantity=position.n_contracts,
                lmtPrice=limit_price, tif="DAY",
            )
            trade = self._ib.placeOrder(combo, order)
            fill_price = self._wait_for_fill(trade, model_debit, FILL_WAIT_SECONDS,
                                             FILL_TIMEOUT_SECONDS, IMPROVE_STEP)

            if fill_price is not None:
                return OrderResult(
                    success=True,
                    fill_price=fill_price,
                    order_id=str(trade.order.orderId),
                    message=f"Closed: {reason}",
                )
            else:
                return OrderResult(
                    success=False,
                    fill_price=model_debit,
                    message="Close order not filled — using model price",
                )
        except Exception as e:
            logger.error("close_iron_condor failed: %s", e)
            return OrderResult(success=False, fill_price=model_debit, message=str(e))

    def get_account_buying_power(self) -> float:
        """Return available options buying power from IBKR."""
        if self.mode == "dry-run":
            return 1_000_000.0  # Assume unlimited in dry-run
        try:
            self._connect()
            summary = self._ib.accountSummary()
            for item in summary:
                if item.tag == "OptionBuyingPower":
                    return float(item.value)
        except Exception as e:
            logger.warning("get_account_buying_power failed: %s", e)
        return 0.0

    def _get_conid(self, symbol: str, expiry_str: str, strike: float, right: str) -> int:
        """Resolve contract ID for an option leg."""
        import ib_insync as ib_mod
        contract = ib_mod.Option(
            symbol=symbol,
            lastTradeDateOrContractMonth=expiry_str,
            strike=strike,
            right=right,
            exchange="CBOE",
            currency="USD",
            multiplier="100",
        )
        qualified = self._ib.qualifyContracts(contract)
        if qualified:
            return qualified[0].conId
        raise ValueError(f"Could not qualify option: {symbol} {expiry_str} {strike} {right}")

    def _wait_for_fill(
        self,
        trade,
        initial_limit: float,
        wait_secs: float,
        timeout_secs: float,
        improve_step: float,
    ) -> Optional[float]:
        """Wait for fill; improve price every wait_secs; market order after timeout_secs."""
        import ib_insync as ib_mod

        start = time.monotonic()
        current_limit = initial_limit

        while time.monotonic() - start < timeout_secs:
            self._ib.sleep(1)  # Non-blocking event loop tick

            if trade.isDone():
                if trade.orderStatus.status == "Filled":
                    return float(trade.orderStatus.avgFillPrice)
                return None  # Cancelled or error

            elapsed = time.monotonic() - start

            # After wait_secs, improve price by one step
            if elapsed > wait_secs:
                action = trade.order.action
                improve = improve_step if action == "BUY" else -improve_step
                new_limit = round(current_limit + improve, 2)
                if new_limit != current_limit:
                    current_limit = new_limit
                    trade.order.lmtPrice = current_limit
                    self._ib.placeOrder(trade.contract, trade.order)
                    logger.debug("Improved limit price to %.2f", current_limit)

        # Timeout: switch to market order
        logger.warning("Fill timeout — switching to market order")
        trade.order.orderType = "MKT"
        trade.order.lmtPrice = 0.0
        self._ib.placeOrder(trade.contract, trade.order)
        self._ib.sleep(5)

        if trade.isDone() and trade.orderStatus.status == "Filled":
            return float(trade.orderStatus.avgFillPrice)
        return None
