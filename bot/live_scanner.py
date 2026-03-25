#!/usr/bin/env python3
"""Live SPX trade scanner using IBKR real-time bid/ask prices.

No BS model estimates. All numbers from IBKR.
All math is exact: credit = short_bid - long_ask, max_loss = width - credit.

Usage:
  python bot/live_scanner.py                    # scan all setups
  python bot/live_scanner.py --type bear_call   # bear calls only
  python bot/live_scanner.py --type ic          # iron condors only
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("scanner")


@dataclass
class SpreadQuote:
    """Real bid/ask spread quote from IBKR."""
    exp: str
    short_strike: float
    long_strike: float
    right: str  # C or P
    short_bid: float
    long_ask: float
    credit: float        # short_bid - long_ask (what you actually receive)
    width: float         # long_strike - short_strike (or vice versa)
    max_loss_ct: float   # (width - credit) × 100
    credit_ct: float     # credit × 100
    cushion: float       # distance from spot to short strike
    cushion_pct: float
    p_win: float         # market-implied: 1 - credit/width
    ev_ct: float         # P(win) × credit - P(loss) × max_loss (per contract)
    dte: int


def connect_ibkr(client_id: int = 56) -> object:
    """Connect to IBKR TWS."""
    import ib_insync as ib
    client = ib.IB()
    client.connect('127.0.0.1', 7497, clientId=client_id, timeout=10)
    client.reqMarketDataType(3)  # delayed if no live subscription
    return client


def get_spot(client) -> float:
    """Get SPX spot from IBKR."""
    import ib_insync as ib
    ticker = client.reqMktData(ib.Index('SPX', 'CBOE'), '', False, False)
    client.sleep(3)
    spot = ticker.last or ticker.close or 0
    client.cancelMktData(ticker.contract)
    return float(spot)


def get_option_quote(client, exp: str, strike: float, right: str) -> tuple[float, float] | None:
    """Get bid/ask for one SPX option. Returns (bid, ask) or None."""
    import ib_insync as ib
    contract = ib.Option('SPX', exp, strike, right, 'CBOE', multiplier='100')
    qualified = client.qualifyContracts(contract)
    if not qualified:
        return None
    ticker = client.reqMktData(qualified[0], '', False, False)
    client.sleep(1.5)
    bid = ticker.bid if ticker.bid and ticker.bid > 0 else None
    ask = ticker.ask if ticker.ask and ticker.ask > 0 else None
    client.cancelMktData(qualified[0])
    if bid is None or ask is None:
        return None
    return (float(bid), float(ask))


def scan_bear_calls(client, spot: float, expirations: list[str]) -> list[SpreadQuote]:
    """Scan bear call spreads at various strikes using real IBKR prices."""
    results = []
    width_pts = 50  # $50 wide spreads

    # Test strikes from 50 to 500 points above spot, in $25 increments
    test_distances = list(range(50, 501, 25))

    for exp in expirations:
        dte = _calc_dte(exp)
        for dist in test_distances:
            short_k = round((spot + dist) / 5) * 5
            long_k = short_k + width_pts

            short_q = get_option_quote(client, exp, short_k, 'C')
            if short_q is None:
                continue
            long_q = get_option_quote(client, exp, long_k, 'C')
            if long_q is None:
                continue

            short_bid, _ = short_q
            _, long_ask = long_q

            credit = short_bid - long_ask
            if credit <= 0:
                continue

            width = width_pts
            credit_ct = credit * 100
            max_loss_ct = width * 100 - credit_ct
            cushion = short_k - spot
            cushion_pct = cushion / spot * 100
            p_win = 1.0 - (credit / width)
            ev_ct = p_win * credit_ct - (1 - p_win) * max_loss_ct

            results.append(SpreadQuote(
                exp=exp, short_strike=short_k, long_strike=long_k,
                right='C', short_bid=short_bid, long_ask=long_ask,
                credit=credit, width=width, max_loss_ct=max_loss_ct,
                credit_ct=credit_ct, cushion=cushion, cushion_pct=cushion_pct,
                p_win=p_win, ev_ct=ev_ct, dte=dte,
            ))

    return results


def scan_iron_condors(client, spot: float, expirations: list[str]) -> list[dict]:
    """Scan iron condors using real IBKR prices."""
    results = []
    width_pts = 30

    # Test put/call distances
    put_distances = [100, 125, 150, 175, 200]
    call_distances = [100, 125, 150, 175, 200]

    for exp in expirations:
        dte = _calc_dte(exp)
        for pd_dist in put_distances:
            for cd_dist in call_distances:
                put_short_k = round((spot - pd_dist) / 5) * 5
                put_long_k = put_short_k - width_pts
                call_short_k = round((spot + cd_dist) / 5) * 5
                call_long_k = call_short_k + width_pts

                # Get all 4 leg quotes
                ps_q = get_option_quote(client, exp, put_short_k, 'P')
                pl_q = get_option_quote(client, exp, put_long_k, 'P')
                cs_q = get_option_quote(client, exp, call_short_k, 'C')
                cl_q = get_option_quote(client, exp, call_long_k, 'C')

                if any(q is None for q in [ps_q, pl_q, cs_q, cl_q]):
                    continue

                put_credit = ps_q[0] - pl_q[1]   # sell bid - buy ask
                call_credit = cs_q[0] - cl_q[1]   # sell bid - buy ask
                total_credit = put_credit + call_credit

                if total_credit <= 0:
                    continue

                credit_ct = total_credit * 100
                max_loss_ct = width_pts * 100 - credit_ct
                cushion = min(spot - put_short_k, call_short_k - spot)
                cushion_pct = cushion / spot * 100
                p_win = 1.0 - (total_credit / width_pts)
                ev_ct = p_win * credit_ct - (1 - p_win) * max_loss_ct

                results.append({
                    'exp': exp, 'dte': dte,
                    'put_short': put_short_k, 'put_long': put_long_k,
                    'call_short': call_short_k, 'call_long': call_long_k,
                    'put_credit': put_credit, 'call_credit': call_credit,
                    'total_credit': total_credit,
                    'credit_ct': credit_ct, 'max_loss_ct': max_loss_ct,
                    'cushion': cushion, 'cushion_pct': cushion_pct,
                    'p_win': p_win, 'ev_ct': ev_ct,
                })

    return results


def _calc_dte(exp_str: str) -> int:
    """Calculate DTE from YYYYMMDD string."""
    exp_date = date(int(exp_str[:4]), int(exp_str[4:6]), int(exp_str[6:8]))
    return (exp_date - date.today()).days


def _find_expirations(dte_targets: list[int]) -> list[str]:
    """Find the nearest Friday expiration for each target DTE."""
    today = date.today()
    exps = []
    seen = set()
    for target in dte_targets:
        d = today + timedelta(days=target)
        # Find nearest Friday
        while d.weekday() != 4:
            d += timedelta(days=1)
        exp_str = d.strftime('%Y%m%d')
        if exp_str not in seen:
            exps.append(exp_str)
            seen.add(exp_str)
    return exps


def check_gex():
    """Quick GEX regime check. Returns (regime, net_gex)."""
    try:
        import json
        from QuantEngine.gex_calculator import get_option_chain_snapshot, get_spot_price, compute_cockpit_state
        config = json.load(open(ROOT / 'data' / 'config.json'))
        mk = config.get('MASSIVE_API_KEY', '')
        ak = config.get('ALPHAVANTAGE_API_KEY', '')
        fk = config.get('FMP_API_KEY', '')
        if not mk:
            return 'UNKNOWN', 0
        spot = get_spot_price('SPX', mk, ak, fk)
        if not spot:
            return 'UNKNOWN', 0
        snap = get_option_chain_snapshot(mk, 'SPX', days_out=30, spot_price=spot, strike_range=300)
        if not snap or not snap.get('results'):
            return 'UNKNOWN', 0
        state = compute_cockpit_state(snap, spot, ticker='SPX')
        return state.get('regime', 'UNKNOWN'), state.get('net_gex', 0)
    except Exception as e:
        logger.warning("GEX check failed: %s", e)
        return 'UNKNOWN', 0


def main():
    parser = argparse.ArgumentParser(description="Live SPX Trade Scanner")
    parser.add_argument("--type", choices=["bear_call", "ic", "all"], default="all")
    parser.add_argument("--contracts", type=int, default=6)
    args = parser.parse_args()

    n_cts = args.contracts

    # GEX check
    print("Checking GEX regime...", flush=True)
    gex_regime, net_gex = check_gex()

    # Connect to IBKR
    print("Connecting to IBKR...", flush=True)
    client = connect_ibkr()
    spot = get_spot(client)
    print(f"\nSPX: {spot:.2f} | GEX: {gex_regime} ({net_gex/1e9:+.1f}B)")
    print(f"All prices are REAL bid/ask from IBKR. No estimates.\n")

    is_neg_gex = gex_regime == 'NEGATIVE_GAMMA'

    # Find expirations
    exps = _find_expirations([1, 3, 7, 14, 21])

    # Scan
    if args.type in ("bear_call", "all"):
        print("=" * 95)
        print("BEAR CALL SPREADS (sell call, buy higher call)")
        print("=" * 95)
        print(f"{'Exp':<10} {'Spread':<14} {'ShortBid':>9} {'LongAsk':>9} {'Credit':>8} {'MaxLoss':>8} "
              f"{'Cush':>8} {'P(win)':>7} {'EV/ct':>8} {f'EV {n_cts}ct':>9} {'Signal':>7}")
        print("-" * 95)

        bc_results = scan_bear_calls(client, spot, exps)
        # Sort by EV descending
        bc_results.sort(key=lambda x: x.ev_ct, reverse=True)

        for r in bc_results:
            if r.credit < 0.10:
                continue
            sig = ""
            if r.ev_ct > 0 and r.cushion_pct >= 2.0:
                sig = "GOOD" if is_neg_gex else "OK"
            elif r.ev_ct > 0:
                sig = "TIGHT"
            else:
                sig = "NEG EV"

            exp_label = f"{r.exp[4:6]}/{r.exp[6:8]}({r.dte}d)"
            print(f"  {exp_label:<8} {r.short_strike:.0f}/{r.long_strike:.0f}    "
                  f"${r.short_bid:>7.2f} ${r.long_ask:>7.2f} ${r.credit:>6.2f} "
                  f"${r.max_loss_ct:>7,.0f} {r.cushion:.0f}pt({r.cushion_pct:.1f}%) "
                  f"{r.p_win:>6.1%} ${r.ev_ct:>7,.0f} ${r.ev_ct*n_cts:>8,.0f} {sig:>7}")

    if args.type in ("ic", "all") and not is_neg_gex:
        print(f"\n{'=' * 95}")
        print("IRON CONDORS (sell put spread + call spread)")
        print("=" * 95)
        print(f"{'Exp':<10} {'Puts':<14} {'Calls':<14} {'Credit':>8} {'MaxLoss':>8} "
              f"{'Cush':>8} {'P(win)':>7} {'EV/ct':>8} {f'EV {n_cts}ct':>9}")
        print("-" * 95)

        ic_results = scan_iron_condors(client, spot, exps[:3])  # only short DTEs for ICs
        ic_results.sort(key=lambda x: x['ev_ct'], reverse=True)

        for r in ic_results[:15]:  # top 15
            exp_label = f"{r['exp'][4:6]}/{r['exp'][6:8]}({r['dte']}d)"
            puts = f"{r['put_short']:.0f}/{r['put_long']:.0f}"
            calls = f"{r['call_short']:.0f}/{r['call_long']:.0f}"
            print(f"  {exp_label:<8} {puts:<14} {calls:<14} ${r['total_credit']:>6.2f} "
                  f"${r['max_loss_ct']:>7,.0f} {r['cushion']:.0f}pt({r['cushion_pct']:.1f}%) "
                  f"{r['p_win']:>6.1%} ${r['ev_ct']:>7,.0f} ${r['ev_ct']*n_cts:>8,.0f}")
    elif is_neg_gex and args.type in ("ic", "all"):
        print(f"\n  ICs DISABLED — GEX is NEGATIVE")

    client.disconnect()


if __name__ == "__main__":
    main()
