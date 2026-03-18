#!/usr/bin/env python3
"""
Hedge Portfolio Monitor — tracks option hedge positions, alerts on roll/entry triggers.

Monitors:
  1. SCHD collar (40 contracts) — roll alert at 5-7 DTE
  2. TLT bear call spreads (5 contracts) — roll alert at 5-7 DTE
  3. SPX put ratio spreads (2 spreads) — roll alert quarterly
  4. VIX level — alert when < 18 to buy crash insurance
  5. Strike proximity — alert when underlying approaches sold strikes

Sends FCM push notifications via existing infrastructure.
Run via cron at market open and close, or continuously with --loop.
"""

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

import yfinance as yf

# Setup
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("hedge_monitor")

ET = ZoneInfo("America/New_York")
STATE_FILE = ROOT / "data" / "hedge_monitor_state.json"
POSITIONS_FILE = ROOT / "data" / "hedge_positions.json"

# ─── Alert Thresholds ────────────────────────────────────────────────────────

ROLL_DTE_ALERT = 7          # Alert when <= 7 DTE
ROLL_DTE_URGENT = 3         # Urgent alert when <= 3 DTE
VIX_BUY_THRESHOLD = 18.0    # Buy VIX call spreads when VIX drops below this
VIX_ELEVATED = 25.0         # VIX is elevated — note for context
STRIKE_PROXIMITY_PCT = 2.0  # Alert when price within 2% of sold strike
GLD_REBALANCE_PCT = 15.0    # Alert if GLD moves >15% from cost basis


# ─── Position Definitions ────────────────────────────────────────────────────

@dataclass
class HedgePosition:
    strategy: str           # "SCHD_COLLAR", "TLT_BEAR_CALL", "SPX_PUT_RATIO", "VIX_CALL_SPREAD"
    underlying: str         # ticker symbol
    expiration: str         # YYYY-MM-DD
    legs: list              # list of dicts: {type, strike, qty, direction}
    status: str = "ACTIVE"  # ACTIVE, ROLLED, EXPIRED, PENDING
    entry_date: str = ""
    net_credit: float = 0.0
    notes: str = ""


def default_positions() -> list[dict]:
    """Generate template positions based on current specs."""
    # Next monthly expiration (~30 days out)
    today = date.today()
    # Find 3rd Friday of next month for standard monthly expiry
    if today.day > 15:
        exp_month = today.month + 1 if today.month < 12 else 1
        exp_year = today.year if today.month < 12 else today.year + 1
    else:
        exp_month = today.month
        exp_year = today.year

    # Find 3rd Friday
    first_day = date(exp_year, exp_month, 1)
    first_friday = first_day + timedelta(days=(4 - first_day.weekday()) % 7)
    third_friday = first_friday + timedelta(weeks=2)
    monthly_exp = third_friday.isoformat()

    # Quarterly expiry for SPX put ratios
    quarter_month = ((today.month - 1) // 3 + 1) * 3 + 1
    if quarter_month > 12:
        quarter_month -= 12
        q_year = today.year + 1
    else:
        q_year = today.year
    q_first = date(q_year, quarter_month, 1)
    q_friday = q_first + timedelta(days=(4 - q_first.weekday()) % 7)
    q_third_friday = q_friday + timedelta(weeks=2)
    quarterly_exp = q_third_friday.isoformat()

    return [
        asdict(HedgePosition(
            strategy="SCHD_COLLAR",
            underlying="SCHD",
            expiration=monthly_exp,
            legs=[
                {"type": "CALL", "strike": 33.00, "qty": 40, "direction": "SELL"},
                {"type": "PUT", "strike": 28.50, "qty": 40, "direction": "BUY"},
            ],
            entry_date=today.isoformat(),
            net_credit=0.10,
            notes="7% OTM call, 7.5% OTM put. Roll at 5-7 DTE.",
        )),
        asdict(HedgePosition(
            strategy="TLT_BEAR_CALL",
            underlying="TLT",
            expiration=monthly_exp,
            legs=[
                {"type": "CALL", "strike": 89.00, "qty": 5, "direction": "SELL"},
                {"type": "CALL", "strike": 94.00, "qty": 5, "direction": "BUY"},
            ],
            entry_date=today.isoformat(),
            net_credit=0.70,
            notes="Rate hedge for VGIT. $5 wide spread.",
        )),
        asdict(HedgePosition(
            strategy="SPX_PUT_RATIO",
            underlying="^GSPC",
            expiration=quarterly_exp,
            legs=[
                {"type": "PUT", "strike": 6470, "qty": 2, "direction": "BUY"},
                {"type": "PUT", "strike": 6000, "qty": 4, "direction": "SELL"},
            ],
            entry_date=today.isoformat(),
            net_credit=-0.06,
            notes="1x2 ratio. Protection 3-10% zone. Naked below 5530.",
        )),
        asdict(HedgePosition(
            strategy="VIX_CALL_SPREAD",
            underlying="^VIX",
            expiration="PENDING",
            legs=[
                {"type": "CALL", "strike": 20, "qty": 10, "direction": "BUY"},
                {"type": "CALL", "strike": 35, "qty": 10, "direction": "SELL"},
            ],
            status="PENDING",
            notes="Deploy when VIX < 18. Cost target ~$1.50/spread.",
        )),
    ]


# ─── State Management ────────────────────────────────────────────────────────

def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {"last_alerts": {}, "vix_buy_alerted": False}


def save_state(state: dict):
    STATE_FILE.parent.mkdir(exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2))


def load_positions() -> list[dict]:
    if POSITIONS_FILE.exists():
        return json.loads(POSITIONS_FILE.read_text())
    # Initialize with defaults
    positions = default_positions()
    save_positions(positions)
    log.info("Initialized default hedge positions → %s", POSITIONS_FILE)
    return positions


def save_positions(positions: list[dict]):
    POSITIONS_FILE.parent.mkdir(exist_ok=True)
    POSITIONS_FILE.write_text(json.dumps(positions, indent=2))


# ─── Price Fetching ──────────────────────────────────────────────────────────

def get_prices() -> dict[str, float]:
    """Fetch current prices for all monitored instruments."""
    tickers = ["SCHD", "TLT", "^GSPC", "^VIX", "GLD", "JEPI", "VGIT"]
    prices = {}
    for t in tickers:
        try:
            h = yf.Ticker(t).history(period="2d")
            if len(h) > 0:
                prices[t] = float(h["Close"].iloc[-1])
        except Exception as e:
            log.warning("Failed to fetch %s: %s", t, e)
    return prices


# ─── Alert Logic ─────────────────────────────────────────────────────────────

def send_alert(title: str, body: str, data: dict = None):
    """Send FCM notification and log to console."""
    log.info("🔔 %s — %s", title, body)
    try:
        from scripts.run_gex_cockpit_precompute import send_fcm_to_topic
        send_fcm_to_topic("market_alerts", title, body, data)
    except Exception as e:
        log.warning("FCM send failed (console-only): %s", e)


def check_dte_alerts(positions: list[dict], state: dict) -> list[str]:
    """Check for positions approaching expiration."""
    alerts = []
    today = date.today()

    for pos in positions:
        if pos["status"] != "ACTIVE":
            continue

        exp = date.fromisoformat(pos["expiration"])
        dte = (exp - today).days
        strategy = pos["strategy"]
        alert_key = f"dte_{strategy}_{pos['expiration']}"

        if dte <= ROLL_DTE_URGENT and state["last_alerts"].get(alert_key) != "urgent":
            msg = f"⚠️ {strategy} expires in {dte} days ({pos['expiration']}). ROLL NOW."
            send_alert(f"URGENT: Roll {strategy}", msg)
            state["last_alerts"][alert_key] = "urgent"
            alerts.append(msg)
        elif dte <= ROLL_DTE_ALERT and alert_key not in state["last_alerts"]:
            msg = f"{strategy} expires in {dte} days ({pos['expiration']}). Plan roll."
            send_alert(f"Roll Soon: {strategy}", msg)
            state["last_alerts"][alert_key] = "warning"
            alerts.append(msg)

    return alerts


def check_strike_proximity(positions: list[dict], prices: dict, state: dict) -> list[str]:
    """Alert when underlying approaches a sold strike."""
    alerts = []

    for pos in positions:
        if pos["status"] != "ACTIVE":
            continue

        ticker = pos["underlying"]
        price = prices.get(ticker)
        if price is None:
            continue

        for leg in pos["legs"]:
            if leg["direction"] != "SELL":
                continue

            strike = leg["strike"]
            distance_pct = abs(price - strike) / price * 100
            alert_key = f"strike_{pos['strategy']}_{strike}"

            if distance_pct <= STRIKE_PROXIMITY_PCT:
                if state["last_alerts"].get(alert_key) != "close":
                    msg = (f"{pos['strategy']}: {ticker} at ${price:.2f} is "
                           f"{distance_pct:.1f}% from sold {leg['type']} ${strike}. "
                           f"Consider adjusting.")
                    send_alert(f"Strike Alert: {pos['strategy']}", msg)
                    state["last_alerts"][alert_key] = "close"
                    alerts.append(msg)
            else:
                # Clear alert if price moved away
                state["last_alerts"].pop(alert_key, None)

    return alerts


def check_vix_entry(prices: dict, state: dict) -> list[str]:
    """Alert when VIX drops below threshold — time to buy crash insurance."""
    alerts = []
    vix = prices.get("^VIX")
    if vix is None:
        return alerts

    if vix < VIX_BUY_THRESHOLD and not state.get("vix_buy_alerted"):
        msg = (f"VIX at {vix:.1f} — below {VIX_BUY_THRESHOLD}. "
               f"Deploy VIX 20/35 call spreads (10x) for ~$1.50/spread. "
               f"Total cost ~$1,500 for $15K crash payoff.")
        send_alert("VIX Low: Buy Crash Insurance", msg)
        state["vix_buy_alerted"] = True
        alerts.append(msg)
    elif vix >= VIX_BUY_THRESHOLD + 2:
        # Reset alert once VIX goes back above threshold + buffer
        state["vix_buy_alerted"] = False

    return alerts


def check_portfolio_health(prices: dict) -> list[str]:
    """General portfolio health checks."""
    alerts = []
    vix = prices.get("^VIX", 20)

    # High VIX warning — collar premiums are fat, good time to roll
    if vix > 30:
        alerts.append(f"VIX at {vix:.1f} (>30). Collar premiums elevated — "
                      f"consider rolling SCHD collar for extra credit.")

    # GLD position check
    gld = prices.get("GLD")
    if gld and gld > 520:
        alerts.append(f"GLD at ${gld:.0f} — consider trimming 10% of position "
                      f"and redeploying to SCHD/JEPI.")

    return alerts


# ─── Dashboard ───────────────────────────────────────────────────────────────

def print_dashboard(positions: list[dict], prices: dict):
    """Print hedge portfolio status dashboard."""
    today = date.today()
    now = datetime.now(ET)

    print("\n" + "=" * 72)
    print(f"  HEDGE PORTFOLIO MONITOR — {now.strftime('%Y-%m-%d %H:%M ET')}")
    print("=" * 72)

    # Market snapshot
    print("\n  MARKET SNAPSHOT")
    print("  " + "─" * 40)
    for ticker, name in [("^GSPC", "SPX"), ("^VIX", "VIX"), ("SCHD", "SCHD"),
                          ("TLT", "TLT"), ("GLD", "GLD"), ("JEPI", "JEPI"), ("VGIT", "VGIT")]:
        p = prices.get(ticker)
        if p:
            extra = ""
            if ticker == "^VIX":
                if p < 18:
                    extra = " ← BUY CRASH INSURANCE"
                elif p > 25:
                    extra = " ← ELEVATED (sell premium)"
            print(f"    {name:6s}: ${p:>10,.2f}{extra}")

    # Positions
    print("\n  HEDGE POSITIONS")
    print("  " + "─" * 40)

    for pos in positions:
        strategy = pos["strategy"]
        status = pos["status"]
        exp = pos.get("expiration", "PENDING")

        if exp != "PENDING" and status == "ACTIVE":
            dte = (date.fromisoformat(exp) - today).days
            dte_str = f"{dte}d"
            if dte <= ROLL_DTE_URGENT:
                dte_str += " ⚠️ ROLL NOW"
            elif dte <= ROLL_DTE_ALERT:
                dte_str += " → roll soon"
        else:
            dte_str = status

        print(f"\n    {strategy} [{dte_str}]")
        print(f"    Exp: {exp} | Net: ${pos['net_credit']:+.2f}/share")

        ticker = pos["underlying"]
        price = prices.get(ticker)

        for leg in pos["legs"]:
            strike = leg["strike"]
            direction = leg["direction"]
            qty = leg["qty"]
            leg_type = leg["type"]

            proximity = ""
            if price and direction == "SELL":
                dist = abs(price - strike) / price * 100
                if dist < STRIKE_PROXIMITY_PCT:
                    proximity = f" ← {dist:.1f}% CLOSE!"
                else:
                    proximity = f" ({dist:.1f}% away)"

            print(f"      {direction:4s} {qty}x {leg_type} ${strike:>10,.2f}{proximity}")

    # Income summary
    print("\n  PROJECTED ANNUAL INCOME FROM HEDGES")
    print("  " + "─" * 40)
    schd_collar_annual = 0.10 * 4000 * 12  # net credit × shares × months
    tlt_spread_annual = 0.70 * 5 * 100 * 12
    spx_ratio_cost = 600 * 4  # $600/spread × 4 quarters
    vix_cost = 1500 * 4  # when deployed, quarterly

    print(f"    SCHD collar:       ${schd_collar_annual:>+10,.0f}/yr")
    print(f"    TLT bear calls:    ${tlt_spread_annual:>+10,.0f}/yr")
    print(f"    SPX put ratios:    ${-spx_ratio_cost:>+10,.0f}/yr")
    print(f"    VIX calls (est):   ${-vix_cost:>+10,.0f}/yr")
    net = schd_collar_annual + tlt_spread_annual - spx_ratio_cost - vix_cost
    print(f"    {'─' * 32}")
    print(f"    Net hedge income:  ${net:>+10,.0f}/yr")

    # Portfolio value estimate
    print("\n  PORTFOLIO ALLOCATIONS")
    print("  " + "─" * 40)
    allocations = [
        ("FX Carry Bot", 200_000, "~8.7%", 17_353),
        ("JEPI", 200_000, "8.0%", 24_000),
        ("GLD", 225_000, "~12%", 27_000),
        ("SCHD", 125_000, "~10%", 12_500),
        ("SGOV", 75_000, "4.3%", 3_225),
        ("VGIT", 75_000, "4.5%", 3_375),
    ]
    total_alloc = 0
    total_income = 0
    for name, amount, yld, income in allocations:
        print(f"    {name:15s}: ${amount:>10,}  ({yld:>5s})  → ${income:>+8,}/yr")
        total_alloc += amount
        total_income += income

    borrow_cost = 6_000
    total_income -= borrow_cost
    total_income += net  # add hedge income

    print(f"    {'─' * 50}")
    print(f"    Borrow cost:       ${-borrow_cost:>+10,}/yr (3% on $200K)")
    print(f"    Hedge income:      ${net:>+10,.0f}/yr")
    print(f"    {'─' * 50}")
    print(f"    TOTAL NET:         ${total_income:>+10,.0f}/yr  "
          f"({total_income/700_000*100:.1f}% on $700K)")

    print()


# ─── Roll Helper ─────────────────────────────────────────────────────────────

def suggest_roll(pos: dict, prices: dict):
    """Suggest new strikes for rolling a position."""
    strategy = pos["strategy"]
    ticker = pos["underlying"]
    price = prices.get(ticker)
    if not price:
        return

    print(f"\n  ROLL SUGGESTION: {strategy}")
    print(f"  Current {ticker}: ${price:.2f}")

    if strategy == "SCHD_COLLAR":
        new_call = round(price * 1.07 * 2) / 2  # Round to $0.50
        new_put = round(price * 0.925 * 2) / 2
        print(f"    New call: ${new_call:.2f} (7% OTM)")
        print(f"    New put:  ${new_put:.2f} (7.5% OTM)")

    elif strategy == "TLT_BEAR_CALL":
        new_short = round(price * 1.023 * 2) / 2
        new_long = new_short + 5
        print(f"    New short call: ${new_short:.2f} (2.3% OTM)")
        print(f"    New long call:  ${new_long:.2f} ($5 wide)")

    elif strategy == "SPX_PUT_RATIO":
        new_buy = round(price * 0.97 / 5) * 5  # Round to 5 points
        new_sell = round(price * 0.90 / 5) * 5
        print(f"    New buy put:  ${new_buy:,.0f} (3% OTM)")
        print(f"    New sell put: ${new_sell:,.0f} (10% OTM, 2x qty)")
        print(f"    Naked exposure below: ${new_sell - (new_buy - new_sell):,.0f}")


# ─── Main ────────────────────────────────────────────────────────────────────

def run_check():
    """Run one monitoring cycle."""
    state = load_state()
    positions = load_positions()
    prices = get_prices()

    if not prices:
        log.error("Could not fetch any prices — aborting check")
        return

    # Print dashboard
    print_dashboard(positions, prices)

    # Run all checks
    all_alerts = []
    all_alerts.extend(check_dte_alerts(positions, state))
    all_alerts.extend(check_strike_proximity(positions, prices, state))
    all_alerts.extend(check_vix_entry(prices, state))
    all_alerts.extend(check_portfolio_health(prices))

    # Suggest rolls for positions near expiry
    today = date.today()
    for pos in positions:
        if pos["status"] != "ACTIVE" or pos.get("expiration") == "PENDING":
            continue
        exp = date.fromisoformat(pos["expiration"])
        dte = (exp - today).days
        if dte <= ROLL_DTE_ALERT:
            suggest_roll(pos, prices)

    if all_alerts:
        print("\n  ALERTS:")
        for a in all_alerts:
            print(f"    • {a}")
    else:
        print("\n  ✓ No alerts — all positions healthy.")

    save_state(state)


def main():
    parser = argparse.ArgumentParser(description="Hedge Portfolio Monitor")
    parser.add_argument("--loop", type=int, metavar="MINS",
                        help="Run continuously, checking every N minutes")
    parser.add_argument("--roll", type=str, metavar="STRATEGY",
                        help="Mark a strategy as rolled and generate new position")
    parser.add_argument("--update-strike", nargs=3, metavar=("STRATEGY", "LEG_IDX", "NEW_STRIKE"),
                        help="Update a strike price: STRATEGY LEG_INDEX NEW_STRIKE")
    parser.add_argument("--set-expiry", nargs=2, metavar=("STRATEGY", "DATE"),
                        help="Update expiration: STRATEGY YYYY-MM-DD")
    args = parser.parse_args()

    if args.roll:
        positions = load_positions()
        prices = get_prices()
        for pos in positions:
            if pos["strategy"] == args.roll and pos["status"] == "ACTIVE":
                pos["status"] = "ROLLED"
                log.info("Marked %s as ROLLED", args.roll)
                suggest_roll(pos, prices)
                # Create new position from template
                for default in default_positions():
                    if default["strategy"] == args.roll:
                        default["entry_date"] = date.today().isoformat()
                        positions.append(default)
                        log.info("Created new %s position expiring %s",
                                 args.roll, default["expiration"])
                        break
                break
        save_positions(positions)
        return

    if args.update_strike:
        strategy, leg_idx, new_strike = args.update_strike
        positions = load_positions()
        for pos in positions:
            if pos["strategy"] == strategy and pos["status"] == "ACTIVE":
                pos["legs"][int(leg_idx)]["strike"] = float(new_strike)
                log.info("Updated %s leg %s strike to %s", strategy, leg_idx, new_strike)
                break
        save_positions(positions)
        return

    if args.set_expiry:
        strategy, new_date = args.set_expiry
        positions = load_positions()
        for pos in positions:
            if pos["strategy"] == strategy and pos["status"] == "ACTIVE":
                pos["expiration"] = new_date
                log.info("Updated %s expiration to %s", strategy, new_date)
                break
        save_positions(positions)
        return

    if args.loop:
        log.info("Running hedge monitor every %d minutes (Ctrl+C to stop)", args.loop)
        while True:
            try:
                run_check()
                log.info("Next check in %d minutes...", args.loop)
                time.sleep(args.loop * 60)
            except KeyboardInterrupt:
                log.info("Stopped.")
                break
    else:
        run_check()


if __name__ == "__main__":
    main()
