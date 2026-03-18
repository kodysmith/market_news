"""IBKR execution layer for the 0DTE scalp bot.

Handles:
  - Connecting to TWS (paper port 7497, live port 7496)
  - Real-time option quotes (bid/ask/mid)
  - Buying single-leg 0DTE options (calls & puts)
  - Selling to close open positions
  - Position tracking & contract ID caching

Usage:
    from scripts.scalp_ibkr import ScalpOrderManager
    mgr = ScalpOrderManager(mode="paper", ticker="SPX")
    result = mgr.buy_option(expiry="20260311", strike=6830.0, right="P", qty=1)
    result = mgr.sell_option(result.con_id, qty=1)
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from datetime import date, datetime
from typing import Optional
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")
log = logging.getLogger("scalp_ibkr")


@dataclass
class OptionQuote:
    bid: float
    ask: float
    last: float
    mid: float


@dataclass
class FillResult:
    success: bool
    fill_price: float = 0.0       # per-contract premium
    avg_fill_price: float = 0.0   # same (single leg)
    con_id: int = 0               # IBKR contract ID
    order_id: int = 0
    qty: int = 0
    message: str = ""


def _port_for_mode(mode: str) -> int:
    if mode == "live":
        return int(os.environ.get("IBKR_LIVE_PORT", "7496"))
    return int(os.environ.get("IBKR_PAPER_PORT", "7497"))


class ScalpOrderManager:
    """Places and tracks single-leg 0DTE option orders via IBKR."""

    # Fill management
    FILL_WAIT_SECS = 10       # wait before improving
    FILL_TIMEOUT_SECS = 30    # switch to market after this
    IMPROVE_STEP = 0.05       # improve limit by $0.05

    def __init__(self, mode: str = "paper", ticker: str = "SPX"):
        """
        Args:
            mode: 'dry-run' | 'paper' | 'live'
            ticker: 'SPX' or 'XSP'
        """
        self.mode = mode
        self.ticker = ticker
        self._ib = None
        self._contract_cache: dict[str, object] = {}  # "SPX_20260311_6830_P" -> Contract
        self._conid_cache: dict[str, int] = {}
        self._position_conid: Optional[int] = None  # active position contract ID

    def connect(self):
        """Connect to IBKR TWS/Gateway."""
        if self._ib is not None:
            return
        if self.mode == "dry-run":
            log.info("[DRY-RUN] Skipping IBKR connection")
            return

        import ib_insync as ib_mod
        port = _port_for_mode(self.mode)
        # Use client ID 15 for scalp bot (separate from IC bot at 10)
        client_id = int(os.environ.get("IBKR_SCALP_CLIENT_ID", "15"))
        ib = ib_mod.IB()
        ib.connect("127.0.0.1", port, clientId=client_id, timeout=10)
        self._ib = ib
        log.info("Connected to IBKR (%s) port=%d clientId=%d ticker=%s",
                 self.mode, port, client_id, self.ticker)

    def disconnect(self):
        if self._ib:
            try:
                self._ib.disconnect()
            except Exception:
                pass
            self._ib = None
            log.info("Disconnected from IBKR")

    def is_connected(self) -> bool:
        return self._ib is not None and self._ib.isConnected()

    def reconnect(self):
        """Force disconnect and reconnect to IBKR."""
        log.warning("Forcing IBKR reconnect...")
        if self._ib:
            try:
                self._ib.disconnect()
            except Exception:
                pass
            self._ib = None
        # Clear contract cache — contracts may be stale after reconnect
        self._contract_cache.clear()
        self._conid_cache.clear()
        time.sleep(2)
        self.connect()

    def _ensure_connected(self):
        if not self.is_connected():
            if self._ib is not None:
                # Was connected but lost connection — force reconnect
                self.reconnect()
            else:
                self.connect()

    # ─── Contract helpers ────────────────────────────────────────────────────

    def _make_contract(self, expiry: str, strike: float, right: str):
        """Create and qualify an option contract."""
        import ib_insync as ib_mod

        key = f"{self.ticker}_{expiry}_{strike}_{right}"
        if key in self._contract_cache:
            return self._contract_cache[key]

        contract = ib_mod.Option(
            symbol=self.ticker,
            lastTradeDateOrContractMonth=expiry,
            strike=strike,
            right=right,
            exchange="SMART" if self.ticker == "XSP" else "CBOE",
            currency="USD",
            multiplier="100",
        )
        qualified = self._ib.qualifyContracts(contract)
        if not qualified:
            raise ValueError(f"Could not qualify: {key}")

        self._contract_cache[key] = qualified[0]
        self._conid_cache[key] = qualified[0].conId
        return qualified[0]

    def _today_expiry(self) -> str:
        """Return today's expiry in YYYYMMDD format."""
        return datetime.now(ET).strftime("%Y%m%d")

    # ─── Quotes ──────────────────────────────────────────────────────────────

    def get_quote(self, expiry: str, strike: float, right: str) -> Optional[OptionQuote]:
        """Get real-time bid/ask for an option."""
        if self.mode == "dry-run":
            return None
        try:
            self._ensure_connected()
            contract = self._make_contract(expiry, strike, right)
            [ticker] = self._ib.reqTickers(contract)
            self._ib.sleep(0.5)  # let data arrive
            [ticker] = self._ib.reqTickers(contract)

            bid = float(ticker.bid) if ticker.bid and ticker.bid > 0 else 0.0
            ask = float(ticker.ask) if ticker.ask and ticker.ask > 0 else 0.0
            last = float(ticker.last) if ticker.last and ticker.last > 0 else 0.0
            mid = (bid + ask) / 2 if bid > 0 and ask > 0 else last

            return OptionQuote(bid=bid, ask=ask, last=last, mid=mid)
        except Exception as e:
            log.warning("get_quote failed (%s %s %s): %s", expiry, strike, right, e)
            return None

    def get_spot_price(self) -> Optional[float]:
        """Get real-time underlying price from IBKR."""
        if self.mode == "dry-run":
            return None
        try:
            self._ensure_connected()
            import ib_insync as ib_mod
            if self.ticker == "SPX":
                contract = ib_mod.Index("SPX", "CBOE", "USD")
            elif self.ticker == "XSP":
                contract = ib_mod.Index("XSP", "CBOE", "USD")
            else:
                return None
            self._ib.qualifyContracts(contract)
            [ticker] = self._ib.reqTickers(contract)
            self._ib.sleep(0.3)
            [ticker] = self._ib.reqTickers(contract)
            return float(ticker.last) if ticker.last and ticker.last > 0 else None
        except Exception as e:
            log.warning("get_spot_price failed: %s", e)
            return None

    # ─── Order Execution ─────────────────────────────────────────────────────

    def buy_option(
        self,
        expiry: str,
        strike: float,
        right: str,      # "C" or "P"
        qty: int,
        limit_price: Optional[float] = None,
    ) -> FillResult:
        """Buy to open a 0DTE option.

        If limit_price is None, uses mid-price from quote.
        """
        action_str = f"BUY {qty} {self.ticker} {expiry} {strike} {right}"
        log.info(">>> ORDER: %s (limit=$%.2f)", action_str,
                 limit_price if limit_price else 0)

        if self.mode == "dry-run":
            price = limit_price or 10.0
            log.info("[DRY-RUN] Simulated fill @ $%.2f", price)
            return FillResult(
                success=True, fill_price=price, avg_fill_price=price,
                qty=qty, message=f"DRY-RUN: {action_str}",
            )

        try:
            self._ensure_connected()
            import ib_insync as ib_mod

            contract = self._make_contract(expiry, strike, right)

            # Get market price if no limit specified
            if limit_price is None:
                quote = self.get_quote(expiry, strike, right)
                if quote and quote.ask > 0:
                    # Buy at ask (or slightly below for paper)
                    limit_price = round(quote.ask, 2)
                else:
                    log.error("No quote available, cannot determine limit price")
                    return FillResult(success=False, message="No quote available")

            order = ib_mod.LimitOrder(
                action="BUY",
                totalQuantity=qty,
                lmtPrice=round(limit_price, 2),
                tif="DAY",
            )
            trade = self._ib.placeOrder(contract, order)
            log.info("  Order placed, orderId=%s, waiting for fill...", trade.order.orderId)

            fill_price = self._wait_for_fill(trade, limit_price)

            if fill_price is not None:
                self._position_conid = contract.conId
                result = FillResult(
                    success=True,
                    fill_price=fill_price,
                    avg_fill_price=fill_price,
                    con_id=contract.conId,
                    order_id=trade.order.orderId,
                    qty=qty,
                    message="Filled",
                )
                log.info("  FILLED @ $%.2f (orderId=%d)", fill_price, trade.order.orderId)
                return result
            else:
                # Cancel unfilled order
                self._ib.cancelOrder(order)
                self._ib.sleep(1)
                log.warning("  Order NOT filled within timeout")
                return FillResult(success=False, message="Fill timeout")

        except Exception as e:
            log.error("buy_option failed: %s", e, exc_info=True)
            return FillResult(success=False, message=str(e))

    def sell_option(
        self,
        expiry: str,
        strike: float,
        right: str,
        qty: int,
        limit_price: Optional[float] = None,
    ) -> FillResult:
        """Sell to close an open option position."""
        action_str = f"SELL {qty} {self.ticker} {expiry} {strike} {right}"
        log.info("<<< ORDER: %s (limit=$%.2f)", action_str,
                 limit_price if limit_price else 0)

        if self.mode == "dry-run":
            price = limit_price or 10.0
            log.info("[DRY-RUN] Simulated sell fill @ $%.2f", price)
            return FillResult(
                success=True, fill_price=price, avg_fill_price=price,
                qty=qty, message=f"DRY-RUN: {action_str}",
            )

        try:
            self._ensure_connected()
            import ib_insync as ib_mod

            contract = self._make_contract(expiry, strike, right)

            if limit_price is None:
                quote = self.get_quote(expiry, strike, right)
                if quote and quote.bid > 0:
                    limit_price = round(quote.bid, 2)
                else:
                    log.warning("No quote for exit, using market order")
                    order = ib_mod.MarketOrder(action="SELL", totalQuantity=qty)
                    trade = self._ib.placeOrder(contract, order)
                    self._ib.sleep(5)
                    if trade.isDone() and trade.orderStatus.status == "Filled":
                        fp = float(trade.orderStatus.avgFillPrice)
                        self._position_conid = None
                        return FillResult(
                            success=True, fill_price=fp, avg_fill_price=fp,
                            con_id=contract.conId, order_id=trade.order.orderId,
                            qty=qty, message="Filled (MKT)",
                        )
                    return FillResult(success=False, message="Market order not filled")

            order = ib_mod.LimitOrder(
                action="SELL",
                totalQuantity=qty,
                lmtPrice=round(limit_price, 2),
                tif="DAY",
            )
            trade = self._ib.placeOrder(contract, order)
            log.info("  Sell order placed, orderId=%s", trade.order.orderId)

            fill_price = self._wait_for_fill(trade, limit_price)

            if fill_price is not None:
                self._position_conid = None
                result = FillResult(
                    success=True,
                    fill_price=fill_price,
                    avg_fill_price=fill_price,
                    con_id=contract.conId,
                    order_id=trade.order.orderId,
                    qty=qty,
                    message="Filled",
                )
                log.info("  SELL FILLED @ $%.2f", fill_price)
                return result
            else:
                # Switch to market on timeout for exits (must get out)
                log.warning("  Sell limit timeout — switching to MKT")
                trade.order.orderType = "MKT"
                trade.order.lmtPrice = 0.0
                self._ib.placeOrder(trade.contract, trade.order)
                self._ib.sleep(5)
                if trade.isDone() and trade.orderStatus.status == "Filled":
                    fp = float(trade.orderStatus.avgFillPrice)
                    self._position_conid = None
                    return FillResult(
                        success=True, fill_price=fp, avg_fill_price=fp,
                        con_id=contract.conId, order_id=trade.order.orderId,
                        qty=qty, message="Filled (MKT after timeout)",
                    )
                return FillResult(success=False, message="Sell order failed")

        except Exception as e:
            log.error("sell_option failed: %s", e, exc_info=True)
            return FillResult(success=False, message=str(e))

    # ─── Scale (add to existing position) ────────────────────────────────────

    def scale_up(
        self,
        expiry: str,
        strike: float,
        right: str,
        add_qty: int,
    ) -> FillResult:
        """Buy additional contracts for scale-on-trail."""
        log.info("  ↑↑ SCALE UP: +%d contracts %s %s %s", add_qty, self.ticker, strike, right)
        return self.buy_option(expiry, strike, right, add_qty)

    # ─── Fill Wait Logic ─────────────────────────────────────────────────────

    def _wait_for_fill(self, trade, initial_limit: float) -> Optional[float]:
        """Wait for fill; improve price; return fill price or None."""
        start = time.monotonic()
        current_limit = initial_limit

        while time.monotonic() - start < self.FILL_TIMEOUT_SECS:
            self._ib.sleep(0.5)

            if trade.isDone():
                if trade.orderStatus.status == "Filled":
                    return float(trade.orderStatus.avgFillPrice)
                return None

            elapsed = time.monotonic() - start
            if elapsed > self.FILL_WAIT_SECS:
                # Improve buy: raise price; sell: lower price
                action = trade.order.action
                if action == "BUY":
                    new_limit = round(current_limit + self.IMPROVE_STEP, 2)
                else:
                    new_limit = round(current_limit - self.IMPROVE_STEP, 2)
                    new_limit = max(new_limit, 0.01)

                if new_limit != current_limit:
                    current_limit = new_limit
                    trade.order.lmtPrice = current_limit
                    self._ib.placeOrder(trade.contract, trade.order)
                    log.debug("  Improved limit to $%.2f", current_limit)

        # Timeout — for buys return None (skip trade); for sells handled by caller
        return None

    # ─── Market Data (bars & VIX) ────────────────────────────────────────────

    def get_bars(self, ticker: str = None, duration: str = "1 D",
                 bar_size: str = "1 min") -> "pd.DataFrame":
        """Fetch historical bars from IBKR for the given ticker.

        Returns a DataFrame with columns: Open, High, Low, Close, Volume
        (capitalized to match yfinance format).
        """
        import pandas as pd
        ticker = ticker or self.ticker

        if self.mode == "dry-run":
            log.warning("[DRY-RUN] get_bars not available — returning empty DataFrame")
            return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])

        try:
            self._ensure_connected()
            import ib_insync as ib_mod

            # Cache the index contract
            cache_key = f"_index_{ticker}"
            if cache_key in self._contract_cache:
                contract = self._contract_cache[cache_key]
            else:
                exchange = "CBOE"
                contract = ib_mod.Index(ticker.upper(), exchange, "USD")
                self._ib.qualifyContracts(contract)
                self._contract_cache[cache_key] = contract

            bars = self._ib.reqHistoricalData(
                contract,
                endDateTime="",
                durationStr=duration,
                barSizeSetting=bar_size,
                whatToShow="TRADES",
                useRTH=False,
                formatDate=1,
            )

            if not bars:
                log.warning("get_bars(%s): no bars returned — reconnecting and retrying", ticker)
                self.reconnect()
                # Re-qualify contract after reconnect
                cache_key_inner = f"_index_{ticker}"
                self._contract_cache.pop(cache_key_inner, None)
                contract = ib_mod.Index(ticker.upper(), "CBOE", "USD")
                self._ib.qualifyContracts(contract)
                self._contract_cache[cache_key_inner] = contract
                bars = self._ib.reqHistoricalData(
                    contract,
                    endDateTime="",
                    durationStr=duration,
                    barSizeSetting=bar_size,
                    whatToShow="TRADES",
                    useRTH=False,
                    formatDate=1,
                )
                if not bars:
                    log.error("get_bars(%s): still no bars after reconnect", ticker)
                    return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])

            df = pd.DataFrame(
                [{
                    "Open": float(b.open),
                    "High": float(b.high),
                    "Low": float(b.low),
                    "Close": float(b.close),
                    "Volume": int(b.volume) if b.volume >= 0 else 0,
                } for b in bars],
                index=pd.DatetimeIndex([b.date for b in bars]),
            )
            log.debug("get_bars(%s): %d bars, last close=%.2f",
                       ticker, len(df), df["Close"].iloc[-1])
            return df

        except Exception as e:
            log.warning("get_bars(%s) failed: %s — attempting reconnect", ticker, e)
            try:
                self.reconnect()
            except Exception as re:
                log.error("Reconnect failed: %s", re)
            return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])

    def get_vix(self) -> float:
        """Get current VIX level from IBKR."""
        if self.mode == "dry-run":
            log.warning("[DRY-RUN] get_vix not available — returning 0.0")
            return 0.0

        try:
            self._ensure_connected()
            import ib_insync as ib_mod

            cache_key = "_index_VIX"
            if cache_key in self._contract_cache:
                contract = self._contract_cache[cache_key]
            else:
                contract = ib_mod.Index("VIX", "CBOE", "USD")
                self._ib.qualifyContracts(contract)
                self._contract_cache[cache_key] = contract

            [ticker] = self._ib.reqTickers(contract)
            self._ib.sleep(0.3)
            [ticker] = self._ib.reqTickers(contract)

            import math
            last = ticker.last
            price = float(last) if last is not None and not math.isnan(last) and last > 0 else 0.0
            if price <= 0:
                close = ticker.close
                if close is not None and not math.isnan(close) and close > 0:
                    price = float(close)
            if price <= 0:
                # Try marketPrice as last resort
                mp = getattr(ticker, 'marketPrice', None)
                if mp is not None and callable(mp):
                    mp = mp()
                if mp is not None and not math.isnan(mp) and mp > 0:
                    price = float(mp)
            log.debug("get_vix: %.2f", price)
            return price

        except Exception as e:
            log.warning("get_vix failed: %s", e)
            return 0.0

    # ─── Iron Condor Orders ─────────────────────────────────────────────────

    def sell_iron_condor(
        self,
        expiry: str,
        put_short: float, put_long: float,
        call_short: float, call_long: float,
        qty: int,
        limit_credit: Optional[float] = None,
    ) -> FillResult:
        """Sell an iron condor (4-leg combo order).

        Returns FillResult with fill_price = net credit per contract.
        """
        desc = (f"IC {self.ticker} {expiry}: "
                f"sell P{put_short}/P{put_long} C{call_short}/C{call_long} x{qty}")
        log.info(">>> %s (credit=$%.2f)", desc, limit_credit or 0)

        if self.mode == "dry-run":
            credit = limit_credit or 0.50
            log.info("[DRY-RUN] Simulated IC fill @ $%.2f credit", credit)
            return FillResult(success=True, fill_price=credit, avg_fill_price=credit,
                              qty=qty, message=f"DRY-RUN: {desc}")

        try:
            self._ensure_connected()
            import ib_insync as ib_mod

            # Qualify all 4 legs
            ps = self._make_contract(expiry, put_short, "P")
            pl = self._make_contract(expiry, put_long, "P")
            cs = self._make_contract(expiry, call_short, "C")
            cl = self._make_contract(expiry, call_long, "C")

            # Build combo legs: sell short strikes, buy long strikes
            combo = ib_mod.Contract(
                symbol=self.ticker,
                secType="BAG",
                exchange="SMART" if self.ticker == "XSP" else "CBOE",
                currency="USD",
            )
            combo.comboLegs = [
                ib_mod.ComboLeg(conId=ps.conId, ratio=1, action="SELL", exchange=ps.exchange),
                ib_mod.ComboLeg(conId=pl.conId, ratio=1, action="BUY", exchange=pl.exchange),
                ib_mod.ComboLeg(conId=cs.conId, ratio=1, action="SELL", exchange=cs.exchange),
                ib_mod.ComboLeg(conId=cl.conId, ratio=1, action="BUY", exchange=cl.exchange),
            ]

            if limit_credit is None:
                # Get mid-price: sum (short bids - long asks)
                ps_q = self.get_quote(expiry, put_short, "P")
                pl_q = self.get_quote(expiry, put_long, "P")
                cs_q = self.get_quote(expiry, call_short, "C")
                cl_q = self.get_quote(expiry, call_long, "C")
                if all(q and q.mid > 0 for q in [ps_q, pl_q, cs_q, cl_q]):
                    limit_credit = round(
                        (ps_q.mid - pl_q.mid) + (cs_q.mid - cl_q.mid), 2
                    )
                else:
                    log.error("Cannot get quotes for all IC legs")
                    return FillResult(success=False, message="No quotes for IC legs")

            # Negative limit = credit order in IBKR combo
            order = ib_mod.LimitOrder(
                action="SELL",
                totalQuantity=qty,
                lmtPrice=-abs(round(limit_credit, 2)),
                tif="DAY",
            )
            trade = self._ib.placeOrder(combo, order)
            log.info("  IC order placed, orderId=%s, credit=$%.2f", trade.order.orderId, limit_credit)

            fill = self._wait_for_fill(trade, -abs(limit_credit))
            if fill is not None:
                credit = abs(fill)
                log.info("  IC FILLED @ $%.2f credit (orderId=%d)", credit, trade.order.orderId)
                return FillResult(
                    success=True, fill_price=credit, avg_fill_price=credit,
                    con_id=0, order_id=trade.order.orderId, qty=qty, message="Filled",
                )
            else:
                self._ib.cancelOrder(order)
                self._ib.sleep(1)
                log.warning("  IC order NOT filled within timeout")
                return FillResult(success=False, message="Fill timeout")

        except Exception as e:
            log.error("sell_iron_condor failed: %s", e, exc_info=True)
            return FillResult(success=False, message=str(e))

    def close_iron_condor(
        self,
        expiry: str,
        put_short: float, put_long: float,
        call_short: float, call_long: float,
        qty: int,
        limit_debit: Optional[float] = None,
    ) -> FillResult:
        """Buy back an iron condor to close. Returns FillResult with fill_price = debit paid."""
        desc = (f"CLOSE IC {self.ticker} {expiry}: "
                f"P{put_short}/P{put_long} C{call_short}/C{call_long} x{qty}")
        log.info("<<< %s (debit=$%.2f)", desc, limit_debit or 0)

        if self.mode == "dry-run":
            debit = limit_debit or 0.25
            log.info("[DRY-RUN] Simulated IC close @ $%.2f debit", debit)
            return FillResult(success=True, fill_price=debit, avg_fill_price=debit,
                              qty=qty, message=f"DRY-RUN: {desc}")

        try:
            self._ensure_connected()
            import ib_insync as ib_mod

            ps = self._make_contract(expiry, put_short, "P")
            pl = self._make_contract(expiry, put_long, "P")
            cs = self._make_contract(expiry, call_short, "C")
            cl = self._make_contract(expiry, call_long, "C")

            combo = ib_mod.Contract(
                symbol=self.ticker,
                secType="BAG",
                exchange="SMART" if self.ticker == "XSP" else "CBOE",
                currency="USD",
            )
            combo.comboLegs = [
                ib_mod.ComboLeg(conId=ps.conId, ratio=1, action="BUY", exchange=ps.exchange),
                ib_mod.ComboLeg(conId=pl.conId, ratio=1, action="SELL", exchange=pl.exchange),
                ib_mod.ComboLeg(conId=cs.conId, ratio=1, action="BUY", exchange=cs.exchange),
                ib_mod.ComboLeg(conId=cl.conId, ratio=1, action="SELL", exchange=cl.exchange),
            ]

            if limit_debit is None:
                ps_q = self.get_quote(expiry, put_short, "P")
                pl_q = self.get_quote(expiry, put_long, "P")
                cs_q = self.get_quote(expiry, call_short, "C")
                cl_q = self.get_quote(expiry, call_long, "C")
                if all(q and q.mid > 0 for q in [ps_q, pl_q, cs_q, cl_q]):
                    limit_debit = round(
                        (ps_q.mid - pl_q.mid) + (cs_q.mid - cl_q.mid), 2
                    )
                else:
                    # Use market order as last resort for exits
                    log.warning("No quotes for IC close — using market order")
                    order = ib_mod.MarketOrder(action="BUY", totalQuantity=qty)
                    trade = self._ib.placeOrder(combo, order)
                    self._ib.sleep(5)
                    if trade.isDone() and trade.orderStatus.status == "Filled":
                        fp = abs(float(trade.orderStatus.avgFillPrice))
                        return FillResult(success=True, fill_price=fp, avg_fill_price=fp,
                                          qty=qty, message="Filled (MKT)")
                    return FillResult(success=False, message="Market order not filled")

            order = ib_mod.LimitOrder(
                action="BUY",
                totalQuantity=qty,
                lmtPrice=abs(round(limit_debit, 2)),
                tif="DAY",
            )
            trade = self._ib.placeOrder(combo, order)
            log.info("  IC close order placed, orderId=%s, debit=$%.2f",
                     trade.order.orderId, limit_debit)

            fill = self._wait_for_fill(trade, abs(limit_debit))
            if fill is not None:
                debit = abs(fill)
                log.info("  IC CLOSE FILLED @ $%.2f debit", debit)
                return FillResult(
                    success=True, fill_price=debit, avg_fill_price=debit,
                    con_id=0, order_id=trade.order.orderId, qty=qty, message="Filled",
                )
            else:
                # Must get out — switch to market
                log.warning("  IC close limit timeout — switching to MKT")
                trade.order.orderType = "MKT"
                trade.order.lmtPrice = 0.0
                self._ib.placeOrder(trade.contract, trade.order)
                self._ib.sleep(5)
                if trade.isDone() and trade.orderStatus.status == "Filled":
                    fp = abs(float(trade.orderStatus.avgFillPrice))
                    return FillResult(success=True, fill_price=fp, avg_fill_price=fp,
                                      qty=qty, message="Filled (MKT after timeout)")
                return FillResult(success=False, message="IC close failed")

        except Exception as e:
            log.error("close_iron_condor failed: %s", e, exc_info=True)
            return FillResult(success=False, message=str(e))

    # ─── Account Info ────────────────────────────────────────────────────────

    def get_buying_power(self) -> float:
        if self.mode == "dry-run":
            return 999_999.0
        try:
            self._ensure_connected()
            summary = self._ib.accountSummary()
            for item in summary:
                if item.tag == "AvailableFunds":
                    return float(item.value)
        except Exception as e:
            log.warning("get_buying_power failed: %s", e)
        return 0.0

    def get_positions(self) -> list:
        """Return current option positions."""
        if self.mode == "dry-run":
            return []
        try:
            self._ensure_connected()
            return [p for p in self._ib.positions()
                    if p.contract.symbol == self.ticker
                    and p.contract.secType == "OPT"]
        except Exception as e:
            log.warning("get_positions failed: %s", e)
            return []
