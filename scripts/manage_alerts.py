#!/usr/bin/env python3
"""
Manage price alerts in Supabase — used by cloud-based price monitor.

Usage:
  python scripts/manage_alerts.py list
  python scripts/manage_alerts.py add --ticker "^GSPC" --condition below --threshold 6640 \
    --name "SPX_IBF_lower" --severity critical --strategy "0DTE_IBF" --expiry 2026-03-13
  python scripts/manage_alerts.py add-position   # Seed alerts for current positions
  python scripts/manage_alerts.py disable <alert_id>
  python scripts/manage_alerts.py clear-expired
  python scripts/manage_alerts.py test            # Test price fetching + FCM
"""

import argparse
import json
import os
import sys
from datetime import date, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Load env
env_file = ROOT / ".env"
if env_file.exists():
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

from supabase import create_client

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")


def get_sb():
    if not SUPABASE_URL or not SUPABASE_KEY:
        print("Error: SUPABASE_URL and SUPABASE_KEY must be set")
        sys.exit(1)
    return create_client(SUPABASE_URL, SUPABASE_KEY)


def cmd_list(args):
    sb = get_sb()
    result = sb.table("price_alerts").select("*").order("created_at", desc=True).execute()
    alerts = result.data

    if not alerts:
        print("No alerts configured.")
        return

    print(f"\n{'ID':<38} {'Name':<25} {'Ticker':<8} {'Cond':<8} {'Thresh':>10} {'Sev':<9} {'Enabled':<8} {'Triggered'}")
    print("-" * 130)

    for a in alerts:
        thresh = f"{float(a['threshold']):,.2f}"
        if a.get("threshold_upper"):
            thresh += f"-{float(a['threshold_upper']):,.2f}"
        triggered = a.get("trigger_count", 0) or 0
        last = a.get("last_triggered_at", "")
        if last:
            last = last[:16]
        enabled = "YES" if a["enabled"] else "no"
        expiry = a.get("expiry", "")
        expired = " EXPIRED" if expiry and str(expiry) < str(date.today()) else ""

        print(f"{a['id']:<38} {a['name']:<25} {a['ticker']:<8} {a['condition']:<8} "
              f"{thresh:>10} {a.get('severity','')::<9} {enabled:<8} "
              f"{triggered}x {last}{expired}")


def cmd_add(args):
    sb = get_sb()
    row = {
        "name": args.name,
        "ticker": args.ticker,
        "condition": args.condition,
        "threshold": float(args.threshold),
        "severity": args.severity or "warning",
        "enabled": True,
        "cooldown_minutes": args.cooldown or 15,
    }
    if args.threshold_upper:
        row["threshold_upper"] = float(args.threshold_upper)
    if args.message:
        row["message"] = args.message
    if args.strategy:
        row["strategy"] = args.strategy
    if args.expiry:
        row["expiry"] = args.expiry

    result = sb.table("price_alerts").insert(row).execute()
    alert = result.data[0]
    print(f"Alert created: {alert['id']}")
    print(f"  {alert['name']}: {alert['ticker']} {alert['condition']} {alert['threshold']}")


def cmd_add_position(args):
    """Seed alerts for current active positions."""
    sb = get_sb()

    # Define alerts for current portfolio
    today = date.today().isoformat()
    alerts = [
        # ─── SPX 0DTE Iron Butterfly at 6655 ───
        {
            "name": "SPX below IBF lower wing",
            "ticker": "^GSPC",
            "condition": "below",
            "threshold": 6640,
            "severity": "critical",
            "message": "SPX breached 6640 — 0DTE IBF put wing in danger",
            "strategy": "0DTE_IBF_6655",
            "expiry": today,
            "cooldown_minutes": 5,
        },
        {
            "name": "SPX above IBF upper wing",
            "ticker": "^GSPC",
            "condition": "above",
            "threshold": 6670,
            "severity": "critical",
            "message": "SPX breached 6670 — 0DTE IBF call wing in danger",
            "strategy": "0DTE_IBF_6655",
            "expiry": today,
            "cooldown_minutes": 5,
        },
        {
            "name": "SPX approaching IBF lower BE",
            "ticker": "^GSPC",
            "condition": "below",
            "threshold": 6645,
            "severity": "warning",
            "message": "SPX approaching 6645 — IBF put breakeven",
            "strategy": "0DTE_IBF_6655",
            "expiry": today,
            "cooldown_minutes": 10,
        },
        {
            "name": "SPX approaching IBF upper BE",
            "ticker": "^GSPC",
            "condition": "above",
            "threshold": 6665,
            "severity": "warning",
            "message": "SPX approaching 6665 — IBF call breakeven",
            "strategy": "0DTE_IBF_6655",
            "expiry": today,
            "cooldown_minutes": 10,
        },
        # ─── VIX spike alert ───
        {
            "name": "VIX spike above 25",
            "ticker": "^VIX",
            "condition": "above",
            "threshold": 25,
            "severity": "critical",
            "message": "VIX above 25 — elevated volatility, check all positions",
            "strategy": "PORTFOLIO",
            "cooldown_minutes": 30,
        },
        {
            "name": "VIX below 18 — buy crash insurance",
            "ticker": "^VIX",
            "condition": "below",
            "threshold": 18,
            "severity": "info",
            "message": "VIX below 18 — deploy VIX call spread crash insurance",
            "strategy": "VIX_CALL_SPREAD",
            "cooldown_minutes": 60,
        },
        # ─── Equity portfolio alerts ───
        {
            "name": "SCHD drop > 5%",
            "ticker": "SCHD",
            "condition": "below",
            "threshold": 29.36,  # ~5% below ~30.90
            "severity": "warning",
            "message": "SCHD down >5% — check collar hedge",
            "strategy": "SCHD_COLLAR",
            "cooldown_minutes": 60,
        },
        {
            "name": "GLD drop > 3%",
            "ticker": "GLD",
            "condition": "below",
            "threshold": 453.68,  # ~3% below ~467.71
            "severity": "warning",
            "message": "GLD down >3% from entry",
            "strategy": "EQUITY_GLD",
            "cooldown_minutes": 60,
        },
        {
            "name": "SPX crash alert -3%",
            "ticker": "^GSPC",
            "condition": "below",
            "threshold": 6450,
            "severity": "critical",
            "message": "SPX down >3% — major sell-off, check all hedges",
            "strategy": "PORTFOLIO",
            "cooldown_minutes": 30,
        },
    ]

    count = 0
    for alert in alerts:
        alert["enabled"] = True
        try:
            sb.table("price_alerts").insert(alert).execute()
            count += 1
            print(f"  + {alert['name']}: {alert['ticker']} {alert['condition']} {alert['threshold']}")
        except Exception as e:
            print(f"  ! Failed: {alert['name']}: {e}")

    print(f"\nCreated {count} alerts.")


def cmd_disable(args):
    sb = get_sb()
    sb.table("price_alerts").update({"enabled": False}).eq("id", args.alert_id).execute()
    print(f"Disabled alert {args.alert_id}")


def cmd_clear_expired(args):
    sb = get_sb()
    today = date.today().isoformat()
    result = sb.table("price_alerts").update({"enabled": False}).lt("expiry", today).eq("enabled", True).execute()
    count = len(result.data) if result.data else 0
    print(f"Disabled {count} expired alerts.")


def cmd_test(args):
    """Test price fetching and FCM."""
    # Test price fetching
    sys.path.insert(0, str(ROOT / "cloud_functions" / "price_monitor"))
    from main import fetch_prices, send_push

    tickers = ["^GSPC", "^VIX", "SCHD", "TLT", "GLD"]
    print("Fetching prices...")
    prices = fetch_prices(tickers)
    for t, p in prices.items():
        print(f"  {t}: ${p:,.2f}")

    if not prices:
        print("ERROR: No prices fetched!")
        return

    # Test FCM
    print("\nSending test FCM notification...")
    spx = prices.get("^GSPC", 0)
    send_push(
        "Test: Price Monitor",
        f"SPX: ${spx:,.2f} | Monitor working correctly",
        {"test": "true"},
    )
    print("Done. Check your phone for the notification.")


def main():
    parser = argparse.ArgumentParser(description="Manage price alerts")
    sub = parser.add_subparsers(dest="cmd")

    sub.add_parser("list", help="List all alerts")

    add_p = sub.add_parser("add", help="Add a new alert")
    add_p.add_argument("--name", required=True)
    add_p.add_argument("--ticker", required=True)
    add_p.add_argument("--condition", required=True, choices=["below", "above", "between", "outside"])
    add_p.add_argument("--threshold", required=True, type=float)
    add_p.add_argument("--threshold-upper", type=float)
    add_p.add_argument("--severity", default="warning", choices=["info", "warning", "critical"])
    add_p.add_argument("--message")
    add_p.add_argument("--strategy")
    add_p.add_argument("--expiry")
    add_p.add_argument("--cooldown", type=int, default=15)

    sub.add_parser("add-position", help="Seed alerts for current positions")

    dis_p = sub.add_parser("disable", help="Disable an alert")
    dis_p.add_argument("alert_id")

    sub.add_parser("clear-expired", help="Disable all expired alerts")
    sub.add_parser("test", help="Test price fetching and FCM")

    args = parser.parse_args()

    if args.cmd == "list":
        cmd_list(args)
    elif args.cmd == "add":
        cmd_add(args)
    elif args.cmd == "add-position":
        cmd_add_position(args)
    elif args.cmd == "disable":
        cmd_disable(args)
    elif args.cmd == "clear-expired":
        cmd_clear_expired(args)
    elif args.cmd == "test":
        cmd_test(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
