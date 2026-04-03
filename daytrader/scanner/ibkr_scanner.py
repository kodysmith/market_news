"""Layer 1 (IBKR): Universe Scanner using Interactive Brokers API.

Uses IBKR's built-in scanner to find gap-up small cap stocks pre-market.
No external API needed — everything comes from your existing IBKR account.

IBKR Scanner capabilities:
  - Gap % (TOP_PERC_GAIN)
  - Volume (MOST_ACTIVE)
  - Price range filters
  - Market cap filters
  - Can run pre-market
"""
from __future__ import annotations

import logging
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from daytrader.config import (
    SCAN_MIN_GAP_PCT, SCAN_MIN_PRICE, SCAN_MAX_PRICE,
    SCAN_MAX_CANDIDATES, IBKR_HOST, IBKR_PORT_PAPER, IBKR_PORT_LIVE,
    IBKR_CLIENT_ID, MODE,
)
from daytrader.scanner.universe_scanner import ScanCandidate

logger = logging.getLogger("daytrader.ibkr_scanner")

try:
    import ib_insync as ib
    _IB_AVAILABLE = True
except ImportError:
    ib = None
    _IB_AVAILABLE = False


class IBKRScanner:
    """Scans for gap-up small cap candidates using IBKR's built-in scanner."""

    def __init__(self, host: str = IBKR_HOST, port: int = None,
                 client_id: int = IBKR_CLIENT_ID):
        if not _IB_AVAILABLE:
            raise ImportError("ib_insync is required: pip install ib_insync")

        self.host = host
        self.port = port or (IBKR_PORT_LIVE if MODE == "live" else IBKR_PORT_PAPER)
        self.client_id = client_id
        self.ib: Optional[ib.IB] = None

    def connect(self):
        """Connect to IBKR TWS/Gateway."""
        if self.ib and self.ib.isConnected():
            return
        self.ib = ib.IB()
        self.ib.connect(self.host, self.port, clientId=self.client_id, timeout=15)
        self.ib.reqMarketDataType(3)  # delayed data if no live subscription
        logger.info("Connected to IBKR at %s:%d (client %d)", self.host, self.port, self.client_id)

    def disconnect(self):
        if self.ib and self.ib.isConnected():
            self.ib.disconnect()

    def scan_top_gainers(self, max_results: int = SCAN_MAX_CANDIDATES) -> list[ScanCandidate]:
        """
        Use IBKR scanner to find top percentage gainers.

        IBKR ScannerSubscription parameters:
          instrument: STK
          locationCode: STK.US.MAJOR (US major exchanges)
          scanCode: TOP_PERC_GAIN (top gainers by %)
          filters: price, market cap, volume
        """
        self.connect()

        # Build scanner subscription
        sub = ib.ScannerSubscription(
            instrument="STK",
            locationCode="STK.US.MAJOR",
            scanCode="TOP_PERC_GAIN",
            numberOfRows=max_results * 2,  # over-fetch, then filter
        )

        # Tag filters for price range
        # IBKR uses XML tag values for additional filters
        tag_values = [
            ib.TagValue("priceAbove", str(SCAN_MIN_PRICE)),
            ib.TagValue("priceBelow", str(SCAN_MAX_PRICE)),
            ib.TagValue("volumeAbove", "300000"),    # min avg volume
            # Market cap filter (small cap: roughly $100M-$2B)
            ib.TagValue("marketCapBelow", "2000000000"),  # $2B max
        ]

        logger.info("Running IBKR scanner: TOP_PERC_GAIN, $%.0f-$%.0f, vol>300K",
                    SCAN_MIN_PRICE, SCAN_MAX_PRICE)

        try:
            results = self.ib.reqScannerData(sub, scannerSubscriptionFilterOptions=tag_values)
            self.ib.sleep(2)
        except Exception as e:
            logger.error("IBKR scanner failed: %s", e)
            return []

        candidates = []
        for i, item in enumerate(results):
            contract = item.contractDetails.contract
            symbol = contract.symbol

            # Get current quote data
            try:
                snapshot = self._get_snapshot(contract)
                if snapshot is None:
                    continue
            except Exception as e:
                logger.debug("Failed to get snapshot for %s: %s", symbol, e)
                continue

            price = snapshot.get("last", 0)
            prev_close = snapshot.get("prev_close", 0)
            volume = snapshot.get("volume", 0)

            if prev_close <= 0 or price <= 0:
                continue

            gap_pct = (price - prev_close) / prev_close
            if gap_pct < SCAN_MIN_GAP_PCT:
                continue

            candidates.append(ScanCandidate(
                symbol=symbol,
                gap_pct=round(gap_pct, 4),
                rel_vol=0,  # will be computed separately if needed
                price=round(price, 2),
                prev_close=round(prev_close, 2),
                day_open=round(snapshot.get("open", price), 2),
                day_volume=int(volume),
                avg_volume=0,  # IBKR doesn't directly give this in scanner
                shares_outstanding=snapshot.get("shares", None),
                scan_rank=i + 1,
            ))

            if len(candidates) >= max_results:
                break

        # Sort by gap %
        candidates.sort(key=lambda c: c.gap_pct, reverse=True)
        for i, c in enumerate(candidates):
            c.scan_rank = i + 1

        logger.info("IBKR scanner returned %d candidates (filtered from %d results)",
                    len(candidates), len(results))
        return candidates

    def _get_snapshot(self, contract) -> Optional[dict]:
        """Get a market data snapshot for a contract."""
        ticker = self.ib.reqMktData(contract, "", True, False)  # snapshot=True
        self.ib.sleep(2)

        result = {}
        if ticker.last and ticker.last > 0:
            result["last"] = float(ticker.last)
        elif ticker.close and ticker.close > 0:
            result["last"] = float(ticker.close)
        else:
            self.ib.cancelMktData(contract)
            return None

        result["prev_close"] = float(ticker.close) if ticker.close else 0
        result["open"] = float(ticker.open) if ticker.open else result["last"]
        result["volume"] = int(ticker.volume) if ticker.volume else 0

        self.ib.cancelMktData(contract)
        return result

    def get_historical_bars(self, symbol: str, duration: str = "1 D",
                            bar_size: str = "1 min") -> Optional[list]:
        """Get historical bars for a symbol via IBKR."""
        self.connect()

        contract = ib.Stock(symbol, "SMART", "USD")
        qualified = self.ib.qualifyContracts(contract)
        if not qualified:
            logger.warning("Could not qualify %s", symbol)
            return None

        try:
            bars = self.ib.reqHistoricalData(
                qualified[0],
                endDateTime="",
                durationStr=duration,
                barSizeSetting=bar_size,
                whatToShow="TRADES",
                useRTH=True,
                formatDate=1,
            )
            self.ib.sleep(1)
            return bars
        except Exception as e:
            logger.warning("Failed to get bars for %s: %s", symbol, e)
            return None

    def get_1min_bars_df(self, symbol: str) -> Optional["pd.DataFrame"]:
        """Get today's 1-min bars as a DataFrame."""
        import pandas as pd

        bars = self.get_historical_bars(symbol, duration="1 D", bar_size="1 min")
        if not bars:
            return None

        data = []
        for bar in bars:
            data.append({
                "Open": bar.open,
                "High": bar.high,
                "Low": bar.low,
                "Close": bar.close,
                "Volume": bar.volume,
            })

        if not data:
            return None

        df = pd.DataFrame(data)
        # Create datetime index from bar dates
        dates = [bar.date for bar in bars]
        df.index = pd.DatetimeIndex(dates)
        if df.index.tz is None:
            try:
                df.index = df.index.tz_localize("America/New_York")
            except Exception:
                pass

        return df

    def scan_high_volume(self, max_results: int = 50) -> list[ScanCandidate]:
        """Alternative scanner: most active by volume (relative volume proxy)."""
        self.connect()

        sub = ib.ScannerSubscription(
            instrument="STK",
            locationCode="STK.US.MAJOR",
            scanCode="MOST_ACTIVE",
            numberOfRows=max_results,
        )
        tag_values = [
            ib.TagValue("priceAbove", str(SCAN_MIN_PRICE)),
            ib.TagValue("priceBelow", str(SCAN_MAX_PRICE)),
            ib.TagValue("marketCapBelow", "2000000000"),
        ]

        try:
            results = self.ib.reqScannerData(sub, scannerSubscriptionFilterOptions=tag_values)
            self.ib.sleep(2)
        except Exception as e:
            logger.error("IBKR volume scanner failed: %s", e)
            return []

        candidates = []
        for item in results:
            contract = item.contractDetails.contract
            candidates.append(ScanCandidate(
                symbol=contract.symbol,
                gap_pct=0,
                rel_vol=0,
                price=0,
                prev_close=0,
                day_open=0,
                day_volume=0,
                avg_volume=0,
            ))

        return candidates
