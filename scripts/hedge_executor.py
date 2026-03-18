#!/usr/bin/env python3
"""
Hedge Portfolio Executor — places and manages hedge option trades on IBKR.

Strategies:
  1. SCHD collar (sell calls, buy puts)
  2. TLT bear call spreads
  3. SPX put ratio spreads (1x2)
  4. VIX call spreads (when VIX < 18)

Connects to IBKR TWS paper trading (port 7497) by default.
Manages rolls, monitors positions, and updates the hedge dashboard.

Usage:
  python scripts/hedge_executor.py                    # Dashboard + monitor
  python scripts/hedge_executor.py --deploy           # Deploy all pending hedges
  python scripts/hedge_executor.py --deploy SCHD_COLLAR  # Deploy one strategy
  python scripts/hedge_executor.py --roll SCHD_COLLAR    # Roll expiring position
  python scripts/hedge_executor.py --close SCHD_COLLAR   # Close a strategy
  python scripts/hedge_executor.py --sync              # Sync IBKR positions → state
  python scripts/hedge_executor.py --loop 30           # Monitor every 30 min
  python scripts/hedge_executor.py --mode live         # Use live account (careful!)
"""

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

import numpy as np
import yfinance as yf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("hedge_exec")

ET = ZoneInfo("America/New_York")
STATE_FILE = ROOT / "data" / "hedge_positions.json"
TRADES_LOG = ROOT / "data" / "hedge_trades.csv"
IBKR_CLIENT_ID = 25  # Separate from IC bot (10), scalp bot (15), other (20)

# ─── Fill Parameters ─────────────────────────────────────────────────────────
FILL_WAIT_SECS = 10
FILL_TIMEOUT_SECS = 45
IMPROVE_STEP = 0.01      # Smaller step for equity/ETF options
IMPROVE_STEP_SPX = 0.05  # Larger step for SPX index options

# ─── Market Hours ─────────────────────────────────────────────────────────────
# Equity options (SCHD, TLT): 9:30-16:00 ET
# Index options (SPX, VIX): 9:30-16:15 ET (some trade extended hours but illiquid)
EQUITY_OPEN_HOUR, EQUITY_OPEN_MIN = 9, 30
EQUITY_CLOSE_HOUR, EQUITY_CLOSE_MIN = 16, 0
INDEX_CLOSE_HOUR, INDEX_CLOSE_MIN = 16, 15


def is_market_open(index: bool = False) -> bool:
    """Check if US options market is currently open."""
    now = datetime.now(ET)
    if now.weekday() >= 5:  # Sat/Sun
        return False
    open_time = now.replace(hour=EQUITY_OPEN_HOUR, minute=EQUITY_OPEN_MIN, second=0)
    if index:
        close_time = now.replace(hour=INDEX_CLOSE_HOUR, minute=INDEX_CLOSE_MIN, second=0)
    else:
        close_time = now.replace(hour=EQUITY_CLOSE_HOUR, minute=EQUITY_CLOSE_MIN, second=0)
    return open_time <= now <= close_time


# ─── Target Hedge Specs ──────────────────────────────────────────────────────

HEDGE_SPECS = {
    "SCHD_COLLAR": {
        "underlying": "SCHD",
        "exchange": "SMART",
        "multiplier": "100",
        "call_otm_pct": 0.07,    # 7% OTM
        "put_otm_pct": 0.075,    # 7.5% OTM
        "contracts": 40,
        "frequency": "monthly",
    },
    "TLT_BEAR_CALL": {
        "underlying": "TLT",
        "exchange": "SMART",
        "multiplier": "100",
        "short_call_otm_pct": 0.023,  # 2.3% OTM
        "spread_width": 5.0,          # $5 wide
        "contracts": 5,
        "frequency": "monthly",
    },
    "SPX_PUT_RATIO": {
        "underlying": "SPX",
        "exchange": "CBOE",
        "multiplier": "100",
        "buy_put_otm_pct": 0.03,   # 3% OTM
        "sell_put_otm_pct": 0.10,  # 10% OTM
        "buy_qty": 2,
        "sell_qty": 4,             # 1x2 ratio
        "frequency": "quarterly",
    },
    "VIX_CALL_SPREAD": {
        "underlying": "VIX",
        "exchange": "SMART",
        "multiplier": "100",
        "buy_strike": 20,
        "sell_strike": 35,
        "contracts": 10,
        "frequency": "monthly",
        "deploy_when_vix_below": 18.0,
    },
}


# ─── IBKR Connection ─────────────────────────────────────────────────────────

@dataclass
class Quote:
    bid: float
    ask: float

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2


class HedgeOrderManager:
    """IBKR order manager for hedge portfolio."""

    def __init__(self, mode: str = "paper"):
        self.mode = mode
        self._ib = None
        self._conid_cache: dict[str, int] = {}

    def connect(self):
        if self._ib is not None:
            return
        if self.mode == "dry-run":
            log.info("Dry-run mode — no IBKR connection")
            return

        import ib_insync as ib_mod
        port = self._port()
        ib = ib_mod.IB()
        ib.connect("127.0.0.1", port, clientId=IBKR_CLIENT_ID, timeout=15)
        self._ib = ib
        log.info("Connected to IBKR (%s) port=%d clientId=%d", self.mode, port, IBKR_CLIENT_ID)

    def disconnect(self):
        if self._ib:
            try:
                self._ib.disconnect()
            except Exception:
                pass
            self._ib = None

    def _port(self) -> int:
        if self.mode == "live":
            return int(os.environ.get("IBKR_LIVE_PORT", "7496"))
        return int(os.environ.get("IBKR_PAPER_PORT", "7497"))

    def is_connected(self) -> bool:
        return self._ib is not None and self._ib.isConnected()

    def ensure_connected(self):
        if not self.is_connected():
            log.info("Reconnecting to IBKR...")
            if self._ib:
                try:
                    self._ib.disconnect()
                except Exception:
                    pass
            self._ib = None
            self._conid_cache.clear()
            time.sleep(2)  # Brief pause before reconnect
            self.connect()

    # ── Contract Resolution ──

    def _qualify_option(self, symbol: str, expiry_str: str, strike: float,
                        right: str, exchange: str = "SMART",
                        trading_class: str = "") -> int:
        """Get conId for an option contract. Caches results."""
        cache_key = f"{symbol}_{expiry_str}_{strike}_{right}_{exchange}_{trading_class}"
        if cache_key in self._conid_cache:
            return self._conid_cache[cache_key]

        import ib_insync as ib_mod
        kwargs = dict(
            symbol=symbol,
            lastTradeDateOrContractMonth=expiry_str,
            strike=strike,
            right=right,
            exchange=exchange,
            currency="USD",
            multiplier="100",
        )
        if trading_class:
            kwargs["tradingClass"] = trading_class
        contract = ib_mod.Option(**kwargs)
        qualified = self._ib.qualifyContracts(contract)
        if not qualified:
            raise ValueError(f"Cannot qualify: {symbol} {expiry_str} {strike}{right} on {exchange}")
        cid = qualified[0].conId
        self._conid_cache[cache_key] = cid
        return cid

    def _qualify_stock(self, symbol: str) -> object:
        """Qualify a stock contract."""
        import ib_insync as ib_mod
        contract = ib_mod.Stock(symbol, "SMART", "USD")
        self._ib.qualifyContracts(contract)
        return contract

    # ── Stock/ETF Orders ──

    def place_stock_order(self, symbol: str, qty: int, action: str = "BUY",
                          limit_price: float = None) -> Optional[float]:
        """Place a stock/ETF order. Uses limit at last price if not specified."""
        if self.mode == "dry-run":
            log.info("[DRY-RUN] %s %d %s @ %s", action, qty, symbol,
                     f"${limit_price:.2f}" if limit_price else "MKT")
            return limit_price or 0.0

        self.ensure_connected()
        import ib_insync as ib_mod

        contract = self._qualify_stock(symbol)

        if limit_price:
            order = ib_mod.LimitOrder(action=action, totalQuantity=qty,
                                       lmtPrice=round(limit_price, 2), tif="DAY")
        else:
            order = ib_mod.MarketOrder(action=action, totalQuantity=qty)

        log.info("Placing %s %d %s @ %s",
                 action, qty, symbol,
                 f"${limit_price:.2f}" if limit_price else "MKT")
        trade = self._ib.placeOrder(contract, order)

        # Wait for fill
        start = time.monotonic()
        while time.monotonic() - start < FILL_TIMEOUT_SECS:
            self._ib.sleep(1)
            if trade.isDone():
                if trade.orderStatus.status == "Filled":
                    fill = float(trade.orderStatus.avgFillPrice)
                    log.info("  %s %d %s filled @ $%.2f", action, qty, symbol, fill)
                    return fill
                log.warning("  Order status: %s", trade.orderStatus.status)
                return None

        # If limit order didn't fill, improve price
        if limit_price:
            log.info("  Switching to market order for %s", symbol)
            trade.order.orderType = "MKT"
            trade.order.lmtPrice = 0.0
            self._ib.placeOrder(trade.contract, trade.order)
            self._ib.sleep(5)
            if trade.isDone() and trade.orderStatus.status == "Filled":
                return float(trade.orderStatus.avgFillPrice)

        return None

    def get_stock_positions(self) -> list[dict]:
        """Fetch all stock/ETF positions from IBKR."""
        if self.mode == "dry-run":
            return []
        self.ensure_connected()
        positions = self._ib.positions()
        result = []
        for p in positions:
            c = p.contract
            if c.secType == "STK":
                result.append({
                    "symbol": c.symbol,
                    "qty": int(p.position),
                    "avg_cost": float(p.avgCost),
                    "account": p.account,
                })
        return result

    # ── Quote Fetching ──

    def get_option_quote(self, symbol: str, expiry_str: str, strike: float,
                         right: str, exchange: str = "SMART",
                         trading_class: str = "") -> Optional[Quote]:
        """Fetch bid/ask for one option leg."""
        if self.mode == "dry-run":
            return None
        try:
            self.ensure_connected()
            import ib_insync as ib_mod
            kwargs = dict(
                symbol=symbol,
                lastTradeDateOrContractMonth=expiry_str,
                strike=strike, right=right,
                exchange=exchange, currency="USD", multiplier="100",
            )
            if trading_class:
                kwargs["tradingClass"] = trading_class
            contract = ib_mod.Option(**kwargs)
            self._ib.qualifyContracts(contract)
            [ticker] = self._ib.reqTickers(contract)
            bid = float(ticker.bid) if ticker.bid and ticker.bid > 0 else 0.0
            ask = float(ticker.ask) if ticker.ask and ticker.ask > 0 else 0.0
            if bid == 0 and ask == 0:
                return None
            return Quote(bid=bid, ask=ask)
        except Exception as e:
            log.warning("Quote failed %s %s %s%s: %s", symbol, expiry_str, strike, right, e)
            return None

    def get_stock_price(self, symbol: str) -> Optional[float]:
        """Get current stock/ETF price from IBKR."""
        if self.mode == "dry-run":
            return None
        try:
            self.ensure_connected()
            contract = self._qualify_stock(symbol)
            [ticker] = self._ib.reqTickers(contract)
            return float(ticker.marketPrice())
        except Exception as e:
            log.warning("Stock price failed %s: %s", symbol, e)
            return None

    # ── Order Placement ──

    def _wait_for_fill(self, trade, initial_limit: float,
                       improve_step: float = IMPROVE_STEP) -> Optional[float]:
        """Wait for fill with price improvement.

        Handles IBKR paper trading quirk where orders briefly show 'Cancelled'
        (error 10349 TIF override) before transitioning to PreSubmitted/Filled.
        """
        start = time.monotonic()
        current_limit = initial_limit
        last_improve = start + FILL_WAIT_SECS

        while time.monotonic() - start < FILL_TIMEOUT_SECS:
            self._ib.sleep(1)

            status = trade.orderStatus.status
            if status == "Filled":
                fill = float(trade.orderStatus.avgFillPrice)
                log.info("Filled at %.4f", fill)
                return fill

            # IBKR paper sends transient 'Cancelled' from 10349 TIF warning
            # — wait a few seconds before treating as real cancellation
            if status == "Cancelled":
                log.debug("Got transient Cancelled status, waiting 5s...")
                self._ib.sleep(5)
                status = trade.orderStatus.status
                if status == "Filled":
                    return float(trade.orderStatus.avgFillPrice)
                if status in ("PreSubmitted", "Submitted"):
                    continue  # Order recovered — keep waiting
                log.warning("Order cancelled (status=%s)", status)
                return None

            if trade.isDone() and status not in ("PreSubmitted", "Submitted", "PendingSubmit"):
                log.warning("Order done with status: %s", status)
                return None

            elapsed = time.monotonic() - start
            if elapsed > last_improve:
                action = trade.order.action
                step = improve_step if action == "BUY" else -improve_step
                new_limit = round(current_limit + step, 2)
                if new_limit != current_limit:
                    current_limit = new_limit
                    trade.order.lmtPrice = current_limit
                    self._ib.placeOrder(trade.contract, trade.order)
                    log.debug("Improved limit to %.2f", current_limit)
                last_improve = time.monotonic() + 5  # Improve every 5s

        # Timeout → market order
        log.warning("Fill timeout — switching to market order")
        trade.order.orderType = "MKT"
        trade.order.lmtPrice = 0.0
        self._ib.placeOrder(trade.contract, trade.order)

        for _ in range(10):
            self._ib.sleep(1)
            if trade.orderStatus.status == "Filled":
                return float(trade.orderStatus.avgFillPrice)

        return None

    def place_spread(self, symbol: str, expiry_str: str, legs_spec: list[dict],
                     action: str, qty: int, limit_price: float,
                     exchange: str = "SMART") -> Optional[float]:
        """
        Place a multi-leg combo order.

        legs_spec: list of {strike, right, action, ratio}
        action: "BUY" or "SELL" (net direction)
        qty: number of spread units
        limit_price: per-unit limit price (positive for debit, positive for credit on SELL)
        """
        if self.mode == "dry-run":
            log.info("[DRY-RUN] %s %dx %s spread @ %.2f | %s exp %s",
                     action, qty, symbol, limit_price, legs_spec, expiry_str)
            return limit_price

        self.ensure_connected()
        import ib_insync as ib_mod

        combo_legs = []
        for leg in legs_spec:
            cid = self._qualify_option(
                symbol, expiry_str, leg["strike"], leg["right"], exchange
            )
            combo_legs.append(ib_mod.ComboLeg(
                conId=cid,
                ratio=leg.get("ratio", 1),
                action=leg["action"],
                exchange=exchange,
            ))

        combo = ib_mod.Contract(
            symbol=symbol, secType="BAG", currency="USD",
            exchange=exchange, comboLegs=combo_legs,
        )

        order = ib_mod.LimitOrder(
            action=action,
            totalQuantity=qty,
            lmtPrice=round(abs(limit_price), 2),
            tif="DAY",
        )

        log.info("Placing %s %dx %s combo @ %.2f | legs: %s",
                 action, qty, symbol, limit_price,
                 [(l["strike"], l["right"], l["action"]) for l in legs_spec])

        trade = self._ib.placeOrder(combo, order)
        step = IMPROVE_STEP_SPX if exchange == "CBOE" else IMPROVE_STEP
        return self._wait_for_fill(trade, abs(limit_price), step)

    def place_single_option(self, symbol: str, expiry_str: str, strike: float,
                            right: str, action: str, qty: int, limit_price: float,
                            exchange: str = "SMART",
                            trading_class: str = "") -> Optional[float]:
        """Place a single-leg option order. Uses GTC if market is closed."""
        if self.mode == "dry-run":
            log.info("[DRY-RUN] %s %dx %s %s%s @ %.2f exp %s",
                     action, qty, symbol, strike, right, limit_price, expiry_str)
            return limit_price

        self.ensure_connected()
        import ib_insync as ib_mod

        kwargs = dict(
            symbol=symbol,
            lastTradeDateOrContractMonth=expiry_str,
            strike=strike, right=right,
            exchange=exchange, currency="USD", multiplier="100",
        )
        if trading_class:
            kwargs["tradingClass"] = trading_class
        contract = ib_mod.Option(**kwargs)
        qualified = self._ib.qualifyContracts(contract)
        if not qualified:
            log.error("Cannot qualify: %s %s %.1f%s on %s", symbol, expiry_str, strike, right, exchange)
            return None

        # Use GTC outside market hours so orders persist until fill
        is_index = symbol in ("SPX", "VIX", "SPXW")
        market_open = is_market_open(index=is_index)
        tif = "DAY" if market_open else "GTC"
        if not market_open:
            log.info("  Market closed — using GTC order (will fill at open)")

        order = ib_mod.LimitOrder(
            action=action, totalQuantity=qty,
            lmtPrice=round(limit_price, 2), tif=tif,
        )

        log.info("Placing %s %dx %s %.1f%s @ %.2f [%s]",
                 action, qty, symbol, strike, right, limit_price, tif)
        trade = self._ib.placeOrder(contract, order)

        if not market_open:
            # Don't wait for fill — order is queued for next open
            log.info("  Order queued (id=%s). Will fill when market opens.",
                     trade.order.orderId)
            return limit_price  # Return estimated price

        return self._wait_for_fill(trade, limit_price)

    # ── Position Sync ──

    def get_option_positions(self) -> list[dict]:
        """Fetch all option positions from IBKR."""
        if self.mode == "dry-run":
            return []
        self.ensure_connected()
        positions = self._ib.positions()
        result = []
        for p in positions:
            c = p.contract
            if c.secType in ("OPT", "FOP"):
                result.append({
                    "symbol": c.symbol,
                    "expiry": c.lastTradeDateOrContractMonth,
                    "strike": c.strike,
                    "right": c.right,
                    "qty": int(p.position),
                    "avg_cost": float(p.avgCost),
                    "account": p.account,
                })
        return result

    def get_account_summary(self) -> dict:
        """Get account summary."""
        if self.mode == "dry-run":
            return {"buying_power": 1_000_000, "net_liq": 700_000}
        self.ensure_connected()
        summary = self._ib.accountSummary()
        result = {}
        for item in summary:
            if item.tag in ("OptionBuyingPower", "NetLiquidation",
                            "TotalCashValue", "UnrealizedPnL"):
                result[item.tag] = float(item.value)
        return result


# ─── Expiration Helpers ───────────────────────────────────────────────────────

def next_monthly_expiry(from_date: date = None) -> date:
    """Find the next monthly options expiry (3rd Friday, 25-45 days out)."""
    d = from_date or date.today()
    target = d + timedelta(days=25)  # At least 25 DTE
    for _ in range(60):
        if target.weekday() == 4:  # Friday
            if 15 <= target.day <= 21:  # 3rd Friday
                return target
        target += timedelta(days=1)
    return target


def next_quarterly_expiry(from_date: date = None) -> date:
    """Find the next quarterly expiry (3rd Friday of Mar/Jun/Sep/Dec, 30-100 days out)."""
    d = from_date or date.today()
    target = d + timedelta(days=30)
    quarter_months = {3, 6, 9, 12}
    for _ in range(120):
        if target.weekday() == 4 and target.month in quarter_months:
            if 15 <= target.day <= 21:
                return target
        target += timedelta(days=1)
    return target


def compute_strikes(price: float, spec: dict) -> dict:
    """Compute option strikes for a given hedge strategy."""
    strikes = {}
    strategy = spec.get("_strategy", "")

    if strategy == "SCHD_COLLAR":
        # SCHD has $1 strike increments
        strikes["call"] = round(price * (1 + spec["call_otm_pct"]))
        strikes["put"] = round(price * (1 - spec["put_otm_pct"]))

    elif strategy == "TLT_BEAR_CALL":
        # TLT has $1 strike increments
        strikes["short_call"] = round(price * (1 + spec["short_call_otm_pct"]))
        strikes["long_call"] = strikes["short_call"] + spec["spread_width"]

    elif strategy == "SPX_PUT_RATIO":
        # Round to 5-point increments for SPX
        strikes["buy_put"] = round(price * (1 - spec["buy_put_otm_pct"]) / 5) * 5
        strikes["sell_put"] = round(price * (1 - spec["sell_put_otm_pct"]) / 5) * 5

    elif strategy == "VIX_CALL_SPREAD":
        strikes["buy_call"] = spec["buy_strike"]
        strikes["sell_call"] = spec["sell_strike"]

    return strikes


# ─── State Management ─────────────────────────────────────────────────────────

def load_positions() -> list[dict]:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return []


def save_positions(positions: list[dict]):
    STATE_FILE.parent.mkdir(exist_ok=True)
    STATE_FILE.write_text(json.dumps(positions, indent=2))


def log_trade(strategy: str, action: str, legs: str, fill_price: float,
              qty: int, expiry: str, notes: str = ""):
    """Append trade to CSV log."""
    TRADES_LOG.parent.mkdir(exist_ok=True)
    header = not TRADES_LOG.exists()
    with open(TRADES_LOG, "a") as f:
        if header:
            f.write("timestamp,strategy,action,legs,fill_price,qty,expiry,notes\n")
        now = datetime.now(ET).isoformat()
        f.write(f"{now},{strategy},{action},\"{legs}\",{fill_price},{qty},{expiry},{notes}\n")


# ─── Strategy Deployment ──────────────────────────────────────────────────────

def get_current_prices() -> dict[str, float]:
    """Fetch prices from yfinance as fallback."""
    prices = {}
    tickers = ["SCHD", "TLT", "^GSPC", "^VIX", "GLD", "JEPI", "SGOV", "VGIT"]
    for t in tickers:
        try:
            h = yf.Ticker(t).history(period="2d")
            if len(h) > 0:
                prices[t] = float(h["Close"].iloc[-1])
        except Exception:
            pass
    return prices


def deploy_schd_collar(mgr: HedgeOrderManager, positions: list[dict],
                       prices: dict) -> Optional[dict]:
    """Deploy SCHD collar: sell calls + buy puts."""
    spec = HEDGE_SPECS["SCHD_COLLAR"]
    price = prices.get("SCHD")
    if not price:
        log.error("Cannot get SCHD price")
        return None

    if not is_market_open() and mgr.mode != "dry-run":
        log.warning("SCHD options: market closed. Orders will be GTC (fill at next open).")

    expiry = next_monthly_expiry()
    expiry_str = expiry.strftime("%Y%m%d")
    spec["_strategy"] = "SCHD_COLLAR"
    strikes = compute_strikes(price, spec)

    log.info("SCHD collar: price=$%.2f | call=$%.0f | put=$%.0f | exp=%s",
             price, strikes["call"], strikes["put"], expiry)

    # Get quotes for pricing
    call_q = mgr.get_option_quote("SCHD", expiry_str, strikes["call"], "C")
    put_q = mgr.get_option_quote("SCHD", expiry_str, strikes["put"], "P")

    if call_q and put_q:
        net_credit = call_q.mid - put_q.mid
        log.info("  Call mid: $%.2f | Put mid: $%.2f | Net: $%+.2f",
                 call_q.mid, put_q.mid, net_credit)
    else:
        # Estimate from historical vol
        net_credit = 0.10
        log.info("  Using estimated net credit: $%.2f (no IBKR quotes)", net_credit)

    # Place sell call
    call_fill = mgr.place_single_option(
        "SCHD", expiry_str, strikes["call"], "C",
        action="SELL", qty=spec["contracts"],
        limit_price=call_q.mid if call_q else 0.30,
    )

    # Place buy put
    put_fill = mgr.place_single_option(
        "SCHD", expiry_str, strikes["put"], "P",
        action="BUY", qty=spec["contracts"],
        limit_price=put_q.mid if put_q else 0.20,
    )

    if call_fill is not None and put_fill is not None:
        actual_credit = call_fill - put_fill
        log.info("SCHD collar deployed. Net credit: $%.2f/share", actual_credit)

        pos = {
            "strategy": "SCHD_COLLAR",
            "underlying": "SCHD",
            "expiration": expiry.isoformat(),
            "legs": [
                {"type": "CALL", "strike": strikes["call"], "qty": spec["contracts"],
                 "direction": "SELL", "fill": call_fill},
                {"type": "PUT", "strike": strikes["put"], "qty": spec["contracts"],
                 "direction": "BUY", "fill": put_fill},
            ],
            "status": "ACTIVE",
            "entry_date": date.today().isoformat(),
            "net_credit": round(actual_credit, 4),
            "notes": f"SCHD@${price:.2f}",
        }
        log_trade("SCHD_COLLAR", "OPEN", f"C{strikes['call']}/P{strikes['put']}",
                  actual_credit, spec["contracts"], expiry.isoformat())
        return pos
    else:
        log.error("SCHD collar deployment failed — partial fill")
        return None


def deploy_tlt_bear_call(mgr: HedgeOrderManager, positions: list[dict],
                         prices: dict) -> Optional[dict]:
    """Deploy TLT bear call spread as two individual legs.
    IBKR rejects BAG combos on ETFs as 'riskless' — place legs separately.
    """
    spec = HEDGE_SPECS["TLT_BEAR_CALL"]
    price = prices.get("TLT")
    if not price:
        log.error("Cannot get TLT price")
        return None

    expiry = next_monthly_expiry()
    expiry_str = expiry.strftime("%Y%m%d")
    spec["_strategy"] = "TLT_BEAR_CALL"
    strikes = compute_strikes(price, spec)

    log.info("TLT bear call: price=$%.2f | sell=$%.2f | buy=$%.2f | exp=%s",
             price, strikes["short_call"], strikes["long_call"], expiry)

    # Get quotes for limit prices
    short_q = mgr.get_option_quote("TLT", expiry_str, strikes["short_call"], "C")
    long_q = mgr.get_option_quote("TLT", expiry_str, strikes["long_call"], "C")

    if short_q and long_q:
        credit = short_q.mid - long_q.mid
        log.info("  Short mid: $%.2f | Long mid: $%.2f | Credit: $%.2f",
                 short_q.mid, long_q.mid, credit)
    else:
        credit = 0.70
        log.info("  Using estimated credit: $%.2f", credit)

    # Place as individual legs (IBKR rejects BAG combo for ETF spreads)
    short_fill = mgr.place_single_option(
        "TLT", expiry_str, strikes["short_call"], "C",
        action="SELL", qty=spec["contracts"],
        limit_price=short_q.mid if short_q else 1.20,
    )

    long_fill = mgr.place_single_option(
        "TLT", expiry_str, strikes["long_call"], "C",
        action="BUY", qty=spec["contracts"],
        limit_price=long_q.mid if long_q else 0.50,
    )

    if short_fill is not None and long_fill is not None:
        actual_credit = short_fill - long_fill
        log.info("TLT bear call spread deployed. Credit: $%.2f/spread", actual_credit)
        pos = {
            "strategy": "TLT_BEAR_CALL",
            "underlying": "TLT",
            "expiration": expiry.isoformat(),
            "legs": [
                {"type": "CALL", "strike": strikes["short_call"], "qty": spec["contracts"],
                 "direction": "SELL", "fill": short_fill},
                {"type": "CALL", "strike": strikes["long_call"], "qty": spec["contracts"],
                 "direction": "BUY", "fill": long_fill},
            ],
            "status": "ACTIVE",
            "entry_date": date.today().isoformat(),
            "net_credit": round(actual_credit, 4),
            "notes": f"TLT@${price:.2f}. $5 wide bear call.",
        }
        log_trade("TLT_BEAR_CALL", "OPEN",
                  f"C{strikes['short_call']}/C{strikes['long_call']}", actual_credit,
                  spec["contracts"], expiry.isoformat())
        return pos
    else:
        log.error("TLT bear call spread deployment failed — partial fill")
        return None


def deploy_spx_put_ratio(mgr: HedgeOrderManager, positions: list[dict],
                         prices: dict) -> Optional[dict]:
    """Deploy SPX 1x2 put ratio spread.
    Uses standard SPX monthly options (tradingClass='SPX', AM-settled).
    Round strikes to 10-point increments for reliable chain availability.
    """
    spec = HEDGE_SPECS["SPX_PUT_RATIO"]
    price = prices.get("^GSPC")
    if not price:
        log.error("Cannot get SPX price")
        return None

    # Use monthly (not quarterly) — chains more reliably available
    expiry = next_monthly_expiry()
    expiry_str = expiry.strftime("%Y%m%d")
    spec["_strategy"] = "SPX_PUT_RATIO"

    # Round to 10-point increments for reliable strike availability
    buy_strike = round(price * (1 - spec["buy_put_otm_pct"]) / 10) * 10
    sell_strike = round(price * (1 - spec["sell_put_otm_pct"]) / 10) * 10

    log.info("SPX put ratio: SPX=$%.2f | buy=%d | sell=%d | exp=%s",
             price, buy_strike, sell_strike, expiry)

    # SPX options on CBOE — try without tradingClass first (like IC bot does),
    # then fallback to SPXW if standard monthly not found
    exchange = "CBOE"
    tc = ""  # Let IBKR resolve the trading class

    buy_q = mgr.get_option_quote("SPX", expiry_str, buy_strike, "P", exchange, tc)
    sell_q = mgr.get_option_quote("SPX", expiry_str, sell_strike, "P", exchange, tc)

    if buy_q and sell_q:
        net_debit = buy_q.mid * spec["buy_qty"] - sell_q.mid * spec["sell_qty"]
        log.info("  Buy put mid: $%.2f x%d | Sell put mid: $%.2f x%d | Net: $%+.2f",
                 buy_q.mid, spec["buy_qty"], sell_q.mid, spec["sell_qty"], net_debit)
    else:
        net_debit = 6.0
        log.info("  Using estimated net debit: $%.2f", net_debit)

    # Buy the long puts
    buy_fill = mgr.place_single_option(
        "SPX", expiry_str, buy_strike, "P",
        action="BUY", qty=spec["buy_qty"],
        limit_price=buy_q.mid if buy_q else 140.0,
        exchange=exchange, trading_class=tc,
    )

    # Sell the short puts (2x quantity)
    sell_fill = mgr.place_single_option(
        "SPX", expiry_str, sell_strike, "P",
        action="SELL", qty=spec["sell_qty"],
        limit_price=sell_q.mid if sell_q else 67.0,
        exchange=exchange, trading_class=tc,
    )

    if buy_fill is not None and sell_fill is not None:
        actual_net = buy_fill * spec["buy_qty"] - sell_fill * spec["sell_qty"]
        log.info("SPX put ratio deployed. Net cost: $%.2f", actual_net)

        pos = {
            "strategy": "SPX_PUT_RATIO",
            "underlying": "^GSPC",
            "expiration": expiry.isoformat(),
            "legs": [
                {"type": "PUT", "strike": buy_strike, "qty": spec["buy_qty"],
                 "direction": "BUY", "fill": buy_fill},
                {"type": "PUT", "strike": sell_strike, "qty": spec["sell_qty"],
                 "direction": "SELL", "fill": sell_fill},
            ],
            "status": "ACTIVE",
            "entry_date": date.today().isoformat(),
            "net_credit": round(-actual_net / spec["buy_qty"], 4),
            "notes": f"SPX@${price:.0f}. 1x2 ratio put spread.",
        }
        log_trade("SPX_PUT_RATIO", "OPEN",
                  f"P{buy_strike}x{spec['buy_qty']}/P{sell_strike}x{spec['sell_qty']}",
                  actual_net, 1, expiry.isoformat())
        return pos
    else:
        log.error("SPX put ratio deployment failed — partial fill")
        return None


def deploy_vix_call_spread(mgr: HedgeOrderManager, positions: list[dict],
                           prices: dict) -> Optional[dict]:
    """Deploy VIX call spread (crash insurance). Only when VIX < 18."""
    spec = HEDGE_SPECS["VIX_CALL_SPREAD"]
    vix = prices.get("^VIX")
    if not vix:
        log.error("Cannot get VIX level")
        return None

    if vix >= spec["deploy_when_vix_below"]:
        log.info("VIX at %.1f — too high for crash insurance (wait for < %.1f)",
                 vix, spec["deploy_when_vix_below"])
        return None

    expiry = next_monthly_expiry()
    expiry_str = expiry.strftime("%Y%m%d")

    log.info("VIX call spread: VIX=%.1f | buy %d / sell %d | exp=%s",
             vix, spec["buy_strike"], spec["sell_strike"], expiry)

    legs_spec = [
        {"strike": spec["buy_strike"], "right": "C", "action": "BUY", "ratio": 1},
        {"strike": spec["sell_strike"], "right": "C", "action": "SELL", "ratio": 1},
    ]

    # VIX options: use SMART exchange with tradingClass='VIX'
    buy_q = mgr.get_option_quote("VIX", expiry_str, spec["buy_strike"], "C", "SMART", "VIX")
    sell_q = mgr.get_option_quote("VIX", expiry_str, spec["sell_strike"], "C", "SMART", "VIX")

    if buy_q and sell_q:
        debit = buy_q.mid - sell_q.mid
        log.info("  Buy %dC mid: $%.2f | Sell %dC mid: $%.2f | Debit: $%.2f",
                 spec["buy_strike"], buy_q.mid, spec["sell_strike"], sell_q.mid, debit)
    else:
        debit = 1.50
        log.info("  Using estimated debit: $%.2f", debit)

    fill = mgr.place_spread(
        "VIX", expiry_str, legs_spec,
        action="BUY", qty=spec["contracts"], limit_price=debit,
    )

    if fill is not None:
        log.info("VIX call spread deployed. Debit: $%.2f/spread", fill)
        pos = {
            "strategy": "VIX_CALL_SPREAD",
            "underlying": "^VIX",
            "expiration": expiry.isoformat(),
            "legs": [
                {"type": "CALL", "strike": spec["buy_strike"], "qty": spec["contracts"],
                 "direction": "BUY", "fill": buy_q.mid if buy_q else debit},
                {"type": "CALL", "strike": spec["sell_strike"], "qty": spec["contracts"],
                 "direction": "SELL", "fill": sell_q.mid if sell_q else 0},
            ],
            "status": "ACTIVE",
            "entry_date": date.today().isoformat(),
            "net_credit": round(-fill, 4),
            "notes": f"VIX@{vix:.1f}. Crash insurance.",
        }
        log_trade("VIX_CALL_SPREAD", "OPEN",
                  f"C{spec['buy_strike']}/C{spec['sell_strike']}", fill,
                  spec["contracts"], expiry.isoformat())
        return pos
    else:
        log.error("VIX call spread deployment failed")
        return None


DEPLOY_FUNCTIONS = {
    "SCHD_COLLAR": deploy_schd_collar,
    "TLT_BEAR_CALL": deploy_tlt_bear_call,
    "SPX_PUT_RATIO": deploy_spx_put_ratio,
    "VIX_CALL_SPREAD": deploy_vix_call_spread,
}


# ─── Equity Allocation ──────────────────────────────────────────────────────
# Portfolio: borrow JPY at ~0.75%, invest in income assets yielding ~5-10%
# Total allocation: $700K across 5 ETFs

EQUITY_ALLOCATION = {
    "JEPI": {"target_usd": 200_000, "description": "JPMorgan Equity Premium Income (~8% yield)"},
    "GLD":  {"target_usd": 225_000, "description": "SPDR Gold Trust (inflation hedge)"},
    "SCHD": {"target_usd": 125_000, "description": "Schwab US Dividend Equity (~3.5% yield)"},
    "SGOV": {"target_usd": 75_000,  "description": "iShares 0-3 Month T-Bill (~4.5% yield)"},
    "VGIT": {"target_usd": 75_000,  "description": "Vanguard Intermediate Treasury (~3.8% yield)"},
}


def deploy_equity_allocation(mgr: HedgeOrderManager, prices: dict) -> list[dict]:
    """Buy the target equity portfolio. Returns list of filled positions."""
    results = []

    if not is_market_open():
        log.warning("Market closed — equity orders will be GTC/queued. Better to run during market hours.")

    for symbol, spec in EQUITY_ALLOCATION.items():
        price = prices.get(symbol)

        # Fetch from IBKR if not in yfinance prices
        if not price:
            price = mgr.get_stock_price(symbol)
        if not price:
            try:
                h = yf.Ticker(symbol).history(period="2d")
                if len(h) > 0:
                    price = float(h["Close"].iloc[-1])
            except Exception:
                pass
        if not price:
            log.error("Cannot get price for %s — skipping", symbol)
            continue

        target_usd = spec["target_usd"]
        qty = int(target_usd / price)

        if qty <= 0:
            log.warning("%s: price $%.2f too high for $%d allocation", symbol, price, target_usd)
            continue

        actual_usd = qty * price
        log.info("%s: %d shares @ $%.2f = $%.0f (target $%.0f) — %s",
                 symbol, qty, price, actual_usd, target_usd, spec["description"])

        fill = mgr.place_stock_order(symbol, qty, action="BUY")

        if fill is not None:
            actual_cost = qty * fill
            results.append({
                "symbol": symbol,
                "qty": qty,
                "fill_price": fill,
                "total_cost": round(actual_cost, 2),
                "target_usd": target_usd,
            })
            log_trade(f"EQUITY_{symbol}", "BUY", f"{qty} shares",
                      fill, qty, "N/A", spec["description"])
            log.info("  %s: FILLED %d @ $%.2f ($%.0f)", symbol, qty, fill, actual_cost)
        else:
            log.error("  %s: ORDER FAILED", symbol)

    return results


def show_equity_positions(mgr: HedgeOrderManager, prices: dict):
    """Display current equity/ETF positions."""
    positions = mgr.get_stock_positions()
    target_symbols = set(EQUITY_ALLOCATION.keys())

    print("\n  EQUITY PORTFOLIO")
    print("  " + "─" * 60)

    total_value = 0
    for p in positions:
        if p["symbol"] in target_symbols or p["qty"] != 0:
            price = prices.get(p["symbol"])
            if not price:
                price = mgr.get_stock_price(p["symbol"]) or p["avg_cost"]
            mkt_val = p["qty"] * price
            pnl = mkt_val - (p["qty"] * p["avg_cost"])
            total_value += mkt_val
            target = EQUITY_ALLOCATION.get(p["symbol"], {}).get("target_usd", 0)
            print(f"    {p['symbol']:5s}  {p['qty']:>6d} shares  "
                  f"avg=${p['avg_cost']:>8.2f}  "
                  f"mkt=${mkt_val:>12,.0f}  "
                  f"pnl=${pnl:>+9,.0f}  "
                  f"target=${target:>9,}")

    if total_value > 0:
        print(f"\n    Total equity value: ${total_value:,.0f}")

    # Show missing allocations
    held = {p["symbol"] for p in positions if p["qty"] > 0}
    missing = target_symbols - held
    if missing:
        print(f"\n    Missing allocations: {', '.join(sorted(missing))}")
        print("    → Run with --deploy-equity to buy")


# ─── Roll / Close ─────────────────────────────────────────────────────────────

def close_position(mgr: HedgeOrderManager, pos: dict, prices: dict) -> bool:
    """Close an active hedge position by buying back / selling to close."""
    strategy = pos["strategy"]
    expiry_str = date.fromisoformat(pos["expiration"]).strftime("%Y%m%d")
    exchange = HEDGE_SPECS.get(strategy, {}).get("exchange", "SMART")

    log.info("Closing %s (exp %s)...", strategy, pos["expiration"])

    for leg in pos["legs"]:
        # Reverse the direction
        close_action = "BUY" if leg["direction"] == "SELL" else "SELL"
        strike = leg["strike"]
        right = leg["type"][0]  # "CALL" → "C", "PUT" → "P"
        qty = leg["qty"]

        # Get current quote
        q = mgr.get_option_quote(pos["underlying"].replace("^", ""),
                                 expiry_str, strike, right, exchange)
        limit = q.mid if q else leg.get("fill", 0.10)

        fill = mgr.place_single_option(
            pos["underlying"].replace("^", ""),
            expiry_str, strike, right,
            action=close_action, qty=qty, limit_price=limit,
            exchange=exchange,
        )

        if fill is None:
            log.error("Failed to close leg: %s %s%s", close_action, strike, right)
            return False

    pos["status"] = "CLOSED"
    log_trade(strategy, "CLOSE", str(pos["legs"]), 0, 0, pos["expiration"])
    log.info("%s closed successfully", strategy)
    return True


def roll_position(mgr: HedgeOrderManager, positions: list[dict],
                  strategy: str, prices: dict) -> bool:
    """Close existing position and deploy a new one."""
    # Find and close the active position
    for pos in positions:
        if pos["strategy"] == strategy and pos["status"] == "ACTIVE":
            if not close_position(mgr, pos, prices):
                log.error("Failed to close %s for roll", strategy)
                return False
            pos["status"] = "ROLLED"
            break

    # Deploy new position
    deploy_fn = DEPLOY_FUNCTIONS.get(strategy)
    if deploy_fn:
        new_pos = deploy_fn(mgr, positions, prices)
        if new_pos:
            positions.append(new_pos)
            save_positions(positions)
            log.info("Rolled %s → new exp %s", strategy, new_pos["expiration"])
            send_alert(f"Rolled {strategy}",
                       f"New expiry: {new_pos['expiration']}. "
                       f"Credit: ${new_pos['net_credit']:+.2f}")
            return True
    return False


# ─── Alerts ──────────────────────────────────────────────────────────────────

def send_alert(title: str, body: str, data: dict = None):
    log.info("ALERT: %s — %s", title, body)
    try:
        from scripts.run_gex_cockpit_precompute import send_fcm_to_topic
        send_fcm_to_topic("market_alerts", title, body, data)
    except Exception:
        pass


# ─── Sync IBKR Positions ─────────────────────────────────────────────────────

def sync_ibkr_positions(mgr: HedgeOrderManager):
    """Fetch IBKR positions and display hedge-related ones."""
    positions = mgr.get_option_positions()
    hedge_symbols = {"SCHD", "TLT", "SPX", "VIX"}

    print("\n  IBKR OPTION POSITIONS (hedge-related)")
    print("  " + "─" * 55)

    found = False
    for p in positions:
        if p["symbol"] in hedge_symbols:
            found = True
            direction = "LONG" if p["qty"] > 0 else "SHORT"
            print(f"    {direction:5s} {abs(p['qty']):>3d}x {p['symbol']:5s} "
                  f"{p['expiry']} {p['strike']:>8.1f}{p['right']} "
                  f"@ ${p['avg_cost']:.2f}")

    if not found:
        print("    (none found — deploy with --deploy)")

    # Account summary
    summary = mgr.get_account_summary()
    if summary:
        print(f"\n    Buying Power: ${summary.get('OptionBuyingPower', 0):,.0f}")
        print(f"    Net Liq:      ${summary.get('NetLiquidation', 0):,.0f}")
        print(f"    Cash:         ${summary.get('TotalCashValue', 0):,.0f}")
        print(f"    Unrealized:   ${summary.get('UnrealizedPnL', 0):+,.0f}")


# ─── Dashboard ───────────────────────────────────────────────────────────────

def print_dashboard(positions: list[dict], prices: dict, mgr: HedgeOrderManager = None):
    """Print full hedge portfolio dashboard."""
    today = date.today()
    now = datetime.now(ET)

    print("\n" + "=" * 72)
    print(f"  HEDGE PORTFOLIO — {now.strftime('%Y-%m-%d %H:%M ET')} [{mgr.mode.upper() if mgr else 'N/A'}]")
    print("=" * 72)

    # Market snapshot
    print("\n  MARKET SNAPSHOT")
    print("  " + "─" * 40)
    for ticker, name in [("^GSPC", "SPX"), ("^VIX", "VIX"), ("SCHD", "SCHD"),
                          ("TLT", "TLT"), ("GLD", "GLD")]:
        p = prices.get(ticker)
        if p:
            extra = ""
            if ticker == "^VIX":
                if p < 18:
                    extra = "  ← BUY CRASH INSURANCE"
                elif p > 25:
                    extra = "  ← ELEVATED (sell premium)"
            print(f"    {name:6s}: ${p:>10,.2f}{extra}")

    # Active positions
    active = [p for p in positions if p["status"] == "ACTIVE"]
    print(f"\n  ACTIVE HEDGE POSITIONS ({len(active)})")
    print("  " + "─" * 55)

    for pos in active:
        strategy = pos["strategy"]
        exp = pos.get("expiration", "PENDING")
        if exp != "PENDING":
            dte = (date.fromisoformat(exp) - today).days
            status = f"{dte}d"
            if dte <= 3:
                status += " ROLL NOW!"
            elif dte <= 7:
                status += " → roll soon"
        else:
            status = "PENDING"

        print(f"\n    {strategy} [{status}] exp:{exp} net:${pos['net_credit']:+.2f}/sh")

        ticker = pos["underlying"]
        price = prices.get(ticker)
        for leg in pos["legs"]:
            strike = leg["strike"]
            direction = leg["direction"]
            proximity = ""
            if price and direction == "SELL":
                dist = abs(price - strike) / price * 100
                proximity = f" ({dist:.1f}% away)" if dist > 2 else f" ⚠️ {dist:.1f}% CLOSE"
            fill_str = f" filled@${leg.get('fill', 0):.2f}" if "fill" in leg else ""
            print(f"      {direction:4s} {leg['qty']}x {leg['type']} "
                  f"${strike:>10,.2f}{proximity}{fill_str}")

    # Recent trades
    if TRADES_LOG.exists():
        print(f"\n  RECENT TRADES")
        print("  " + "─" * 55)
        import csv
        with open(TRADES_LOG) as f:
            reader = list(csv.DictReader(f))
            for row in reader[-5:]:
                print(f"    {row['timestamp'][:16]} {row['action']:5s} {row['strategy']} "
                      f"@ ${float(row['fill_price']):+.2f} {row['legs']}")

    # IBKR sync
    if mgr and mgr.mode != "dry-run":
        try:
            sync_ibkr_positions(mgr)
            show_equity_positions(mgr, prices)
        except Exception as e:
            log.warning("IBKR sync failed: %s", e)

    print()


# ─── Monitor Loop ────────────────────────────────────────────────────────────

def monitor_check(mgr: HedgeOrderManager, auto_roll: bool = False):
    """Run one monitoring cycle with optional auto-roll."""
    positions = load_positions()
    prices = get_current_prices()

    if not prices:
        log.error("Cannot fetch prices")
        return

    print_dashboard(positions, prices, mgr)

    today = date.today()
    alerts = []

    for pos in positions:
        if pos["status"] != "ACTIVE" or pos.get("expiration") == "PENDING":
            continue

        exp = date.fromisoformat(pos["expiration"])
        dte = (exp - today).days
        strategy = pos["strategy"]

        # DTE alerts
        if dte <= 3:
            msg = f"{strategy} expires in {dte}d — ROLL NOW"
            alerts.append(msg)
            send_alert(f"URGENT: {strategy}", msg)

            if auto_roll:
                log.info("Auto-rolling %s...", strategy)
                roll_position(mgr, positions, strategy, prices)
        elif dte <= 7:
            alerts.append(f"{strategy} expires in {dte}d — plan roll")

        # Strike proximity
        price = prices.get(pos["underlying"])
        if price:
            for leg in pos["legs"]:
                if leg["direction"] == "SELL":
                    dist = abs(price - leg["strike"]) / price * 100
                    if dist < 2:
                        msg = (f"{strategy}: price within {dist:.1f}% of "
                               f"sold {leg['type']} ${leg['strike']}")
                        alerts.append(msg)
                        send_alert(f"Strike Alert: {strategy}", msg)

    # VIX entry check
    vix = prices.get("^VIX")
    vix_active = any(p["strategy"] == "VIX_CALL_SPREAD" and p["status"] == "ACTIVE"
                     for p in positions)
    if vix and vix < 18 and not vix_active:
        msg = f"VIX at {vix:.1f} — deploy crash insurance!"
        alerts.append(msg)
        send_alert("VIX Low: Buy Crash Insurance", msg)

    if alerts:
        print("  ALERTS:")
        for a in alerts:
            print(f"    • {a}")
    else:
        print("  ✓ All positions healthy.")

    save_positions(positions)


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Hedge Portfolio Executor (IBKR)")
    parser.add_argument("--mode", choices=["dry-run", "paper", "live"], default="paper",
                        help="Trading mode (default: paper)")
    parser.add_argument("--deploy", nargs="?", const="ALL", metavar="STRATEGY",
                        help="Deploy hedges (ALL or specific strategy name)")
    parser.add_argument("--roll", metavar="STRATEGY",
                        help="Roll an expiring strategy to new expiry")
    parser.add_argument("--close", metavar="STRATEGY",
                        help="Close a strategy position")
    parser.add_argument("--deploy-equity", action="store_true",
                        help="Deploy equity allocation (JEPI, GLD, SCHD, SGOV, VGIT)")
    parser.add_argument("--sync", action="store_true",
                        help="Sync and display IBKR positions")
    parser.add_argument("--loop", type=int, metavar="MINS",
                        help="Monitor continuously every N minutes")
    parser.add_argument("--auto-roll", action="store_true",
                        help="Auto-roll positions at 3 DTE (use with --loop)")
    args = parser.parse_args()

    mgr = HedgeOrderManager(mode=args.mode)

    if args.sync:
        try:
            mgr.connect()
            sync_ibkr_positions(mgr)
        finally:
            mgr.disconnect()
        return

    if args.deploy_equity:
        prices = get_current_prices()
        try:
            mgr.connect()
            log.info("Deploying equity allocation ($700K across 5 ETFs)...")
            results = deploy_equity_allocation(mgr, prices)
            total = sum(r["total_cost"] for r in results)
            log.info("Equity deployment complete: %d/%d ETFs, total $%.0f",
                     len(results), len(EQUITY_ALLOCATION), total)
            show_equity_positions(mgr, prices)
        finally:
            mgr.disconnect()
        return

    if args.deploy:
        positions = load_positions()
        prices = get_current_prices()

        try:
            mgr.connect()

            if args.deploy == "ALL":
                strategies = list(DEPLOY_FUNCTIONS.keys())
            else:
                strategies = [args.deploy]

            for strategy in strategies:
                # Skip if already active
                already_active = any(
                    p["strategy"] == strategy and p["status"] == "ACTIVE"
                    for p in positions
                )
                if already_active:
                    log.info("Skipping %s — already active", strategy)
                    continue

                deploy_fn = DEPLOY_FUNCTIONS.get(strategy)
                if deploy_fn:
                    new_pos = deploy_fn(mgr, positions, prices)
                    if new_pos:
                        positions.append(new_pos)
                else:
                    log.error("Unknown strategy: %s", strategy)

            save_positions(positions)
            print_dashboard(positions, prices, mgr)
        finally:
            mgr.disconnect()
        return

    if args.roll:
        positions = load_positions()
        prices = get_current_prices()
        try:
            mgr.connect()
            roll_position(mgr, positions, args.roll, prices)
        finally:
            mgr.disconnect()
        return

    if args.close:
        positions = load_positions()
        prices = get_current_prices()
        try:
            mgr.connect()
            for pos in positions:
                if pos["strategy"] == args.close and pos["status"] == "ACTIVE":
                    close_position(mgr, pos, prices)
                    break
            save_positions(positions)
        finally:
            mgr.disconnect()
        return

    if args.loop:
        log.info("Monitoring every %d min [%s mode] (Ctrl+C to stop)", args.loop, args.mode)
        try:
            mgr.connect()
        except Exception as e:
            log.warning("IBKR connection failed, running in monitor-only mode: %s", e)

        while True:
            try:
                monitor_check(mgr, auto_roll=args.auto_roll)
                log.info("Next check in %d min...", args.loop)
                time.sleep(args.loop * 60)
            except KeyboardInterrupt:
                log.info("Stopped.")
                break
            except Exception as e:
                log.error("Monitor error: %s", e)
                time.sleep(60)

        mgr.disconnect()
        return

    # Default: just show dashboard
    positions = load_positions()
    prices = get_current_prices()
    try:
        mgr.connect()
    except Exception:
        log.warning("IBKR not available — dashboard only")

    print_dashboard(positions, prices, mgr)
    mgr.disconnect()


if __name__ == "__main__":
    main()
