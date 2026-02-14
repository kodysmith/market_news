#!/usr/bin/env python3
"""
Test IBKR TWS/Gateway connection and basic API access.

Use this to verify auth/connectivity before building the full IBKR data layer.
No API key is used: a desktop app (TWS or IB Gateway) must be running and
logged into your IBKR account; this script connects to it. See docs/IBKR_SETUP.md.

Usage:
  # From repo root (install ib_insync if needed: pip install ib_insync)
  python scripts/test_ibkr_connection.py          # paper (7497)
  python scripts/test_ibkr_connection.py --live   # live (7496)

If you get "Connection refused": install and run TWS or IB Gateway, log in,
then enable API (Edit → Global Configuration → API → Settings). Full steps:
  docs/IBKR_SETUP.md

Environment (optional):
  IBKR_HOST     default 127.0.0.1
  IBKR_PORT     default 7497 (TWS paper), 7496 live, 4001/4002 Gateway
  IBKR_CLIENT_ID default 1
"""

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

def main():
    parser = argparse.ArgumentParser(description="Test IBKR TWS/Gateway connection")
    parser.add_argument("--live", action="store_true", help="Use live TWS port 7496")
    parser.add_argument("--paper", action="store_true", help="Use paper TWS port 7497 (default)")
    args = parser.parse_args()

    host = os.environ.get("IBKR_HOST", "127.0.0.1")
    if os.environ.get("IBKR_PORT"):
        port = int(os.environ.get("IBKR_PORT"))
    elif args.live:
        port = 7496
    else:
        port = 7497  # paper default
    client_id = int(os.environ.get("IBKR_CLIENT_ID", "1"))

    print("IBKR connection test")
    print("=" * 50)
    print(f"  Host:       {host}")
    print(f"  Port:       {port}  ({'live' if port == 7496 else 'paper'})")
    print(f"  Client ID:  {client_id}")
    print()
    print("Make sure TWS or IB Gateway is running and logged in,")
    print("and API is enabled (Edit → Global Configuration → API → Settings).")
    print()

    try:
        import ib_insync as ib
    except ImportError:
        print("ERROR: ib_insync not installed.")
        print("  pip install ib_insync")
        print("  Or from repo: pip install -r market_news_app/backtesting_v2/requirements.txt")
        return 1

    ib_obj = ib.IB()
    try:
        print("Connecting...")
        ib_obj.connect(host, port, clientId=client_id, timeout=15, readonly=True)
        print("  Connected.")
    except Exception as e:
        print(f"  FAILED: {e}")
        print()
        print("Common causes:")
        print("  - TWS/Gateway not running or not logged in")
        print("  - API not enabled in TWS (Edit → Global Configuration → API)")
        print("  - Wrong port (7497=paper, 7496=live, 4001/4002=Gateway)")
        print("  - Another app already using this client ID")
        return 1

    try:
        # 1) Server time (proves socket + API is alive)
        print()
        print("1. Requesting server time...")
        t = ib_obj.reqCurrentTime()
        print(f"   Server time: {t}")

        # 2) Account summary (proves session is authenticated)
        print()
        print("2. Requesting account summary (first 3 items)...")
        summary = ib_obj.accountSummary()
        for item in list(summary)[:3]:
            print(f"   {item.tag}: {item.value} ({item.currency})")
        if len(summary) > 3:
            print(f"   ... and {len(summary) - 3} more")

        # 3) Quote: try your subscribed exchanges first (IBKR-PRO = BATS, BYX, EDGX, EDGA, IEX)
        symbol = "SPY"  # or NVDA
        # Exchanges covered by "US Real-Time Non Consolidated Streaming Quotes (IBKR-PRO)"
        realtime_exchanges = ("EDGX", "BATS", "BYX", "EDGA", "IEX")
        print()
        print(f"3. Requesting quote for {symbol}...")
        price = None
        used_exchange = None
        # Try real-time on an exchange you're subscribed to
        for exchange in realtime_exchanges:
            contract = ib.Stock(symbol, exchange, "USD")
            ib_obj.reqMktData(contract, "", False, False)
            ib_obj.sleep(3)
            ticker = ib_obj.ticker(contract)
            ib_obj.cancelMktData(contract)
            if ticker and (ticker.last or ticker.close):
                p = ticker.last if ticker.last else ticker.close
                if p is not None and not (isinstance(p, float) and (p != p)):
                    price = p
                    used_exchange = exchange
                    break
        if price is None:
            # Fallback: delayed (no subscription) via SMART
            contract = ib.Stock(symbol, "SMART", "USD")
            ib_obj.reqMarketDataType(3)  # delayed
            ib_obj.sleep(1)
            ib_obj.reqMktData(contract, "", False, False)
            ib_obj.sleep(5)
            ticker = ib_obj.ticker(contract)
            ib_obj.cancelMktData(contract)
            if ticker and (ticker.last or ticker.close):
                p = ticker.last if ticker.last else ticker.close
                if p is not None and not (isinstance(p, float) and (p != p)):
                    price = p
                    used_exchange = "SMART (delayed)"
        if price is not None:
            label = f"real-time ({used_exchange})" if used_exchange and "delayed" not in str(used_exchange) else "delayed"
            print(f"   {symbol}: {price} ({label})")
        else:
            print("   (No quote — check market hours or Market Data Connections in TWS)")

        # 4) Options chain: expirations and strikes for the underlying
        print()
        print(f"4. Requesting options chain (expirations + strikes) for {symbol}...")
        try:
            stock = ib.Stock(symbol, "SMART", "USD")
            qualified = ib_obj.qualifyContracts(stock)
            if not qualified:
                print(f"   Could not qualify {symbol} (no conId)")
            else:
                con_id = qualified[0].conId
                chains = ib_obj.reqSecDefOptParams(symbol, "", "STK", con_id)
                if not chains:
                    print(f"   No option chains returned for {symbol}")
                else:
                    # Chains can be per-exchange; take first (or merge)
                    chain = chains[0]
                    expirations = sorted(chain.expirations)
                    strikes = sorted(chain.strikes)
                    exp_sample = expirations[:3] if len(expirations) > 3 else expirations
                    print(f"   Expirations: {len(expirations)} (e.g. {exp_sample} ...)")
                    if strikes:
                        print(f"   Strikes: {len(strikes)} from {strikes[0]:.0f} to {strikes[-1]:.0f}")
                    else:
                        print(f"   Strikes: 0")
                    print(f"   Multiplier: {chain.multiplier}")
        except Exception as ex:
            print(f"   Error: {ex}")

        print()
        print("=" * 50)
        print("SUCCESS: IBKR connection and auth are working.")
        return 0
    except Exception as e:
        print(f"   Error during test: {e}")
        return 1
    finally:
        ib_obj.disconnect()
        print("Disconnected.")

if __name__ == "__main__":
    sys.exit(main())
