#!/usr/bin/env python3
"""Live trade performance tracker.

Records every trade with real numbers (credit, max_loss, risk:reward ratio).
Tracks cumulative P&L, win rate, and whether we're beating the breakeven
probability for our actual risk:reward ratios.

Stores data in a local CSV for simplicity — no Supabase dependency.
"""
from __future__ import annotations

import csv
import logging
import os
from dataclasses import dataclass, asdict
from datetime import date, datetime
from pathlib import Path

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
TRADES_FILE = ROOT / "data" / "live_trades.csv"

FIELDS = [
    "trade_id", "entry_date", "exit_date", "strategy", "expiration",
    "short_strike", "long_strike", "right",  # C or P
    "n_contracts", "width",
    "credit_per_ct",     # actual credit received per contract ($)
    "max_loss_per_ct",   # width × 100 - credit ($)
    "risk_reward",       # max_loss / credit
    "breakeven_winrate", # 1 / (1 + credit/max_loss)
    "exit_reason",       # profit_target, stop_loss, expiry, manual
    "pnl_per_ct",        # actual P&L per contract ($)
    "pnl_total",         # total P&L (pnl_per_ct × n_contracts)
    "is_win",            # 1 or 0
    "gex_regime",        # POSITIVE_GAMMA, NEGATIVE_GAMMA, UNKNOWN
    "ssan_p_loss",       # model's loss probability at entry
    "spx_at_entry",
    "vix_at_entry",
    "source",            # "paper" or "live"
]


@dataclass
class TradeRecord:
    trade_id: str = ""
    entry_date: str = ""
    exit_date: str = ""
    strategy: str = ""
    expiration: str = ""
    short_strike: float = 0
    long_strike: float = 0
    right: str = ""
    n_contracts: int = 0
    width: float = 0
    credit_per_ct: float = 0
    max_loss_per_ct: float = 0
    risk_reward: float = 0
    breakeven_winrate: float = 0
    exit_reason: str = ""
    pnl_per_ct: float = 0
    pnl_total: float = 0
    is_win: int = 0
    gex_regime: str = ""
    ssan_p_loss: float = 0
    spx_at_entry: float = 0
    vix_at_entry: float = 0
    source: str = "paper"


def record_entry(
    strategy: str,
    expiration: str,
    short_strike: float,
    long_strike: float,
    right: str,
    n_contracts: int,
    credit_per_ct: float,
    spx: float,
    vix: float,
    gex_regime: str = "",
    ssan_p_loss: float = 0,
    source: str = "paper",
) -> TradeRecord:
    """Record a new trade entry. Returns the record for later update on exit."""
    width = abs(long_strike - short_strike)
    max_loss = width * 100 - credit_per_ct
    rr = max_loss / credit_per_ct if credit_per_ct > 0 else 999
    be_wr = max_loss / (max_loss + credit_per_ct) if (max_loss + credit_per_ct) > 0 else 1.0

    trade = TradeRecord(
        trade_id=f"{strategy}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        entry_date=date.today().isoformat(),
        strategy=strategy,
        expiration=expiration,
        short_strike=short_strike,
        long_strike=long_strike,
        right=right,
        n_contracts=n_contracts,
        width=width,
        credit_per_ct=round(credit_per_ct, 2),
        max_loss_per_ct=round(max_loss, 2),
        risk_reward=round(rr, 1),
        breakeven_winrate=round(be_wr * 100, 1),
        spx_at_entry=round(spx, 2),
        vix_at_entry=round(vix, 2),
        gex_regime=gex_regime,
        ssan_p_loss=round(ssan_p_loss, 3),
        source=source,
    )

    logger.info(
        "ENTRY: %s | %s %.0f/%.0f | credit=$%.0f | max_loss=$%.0f | R:R=%.1f:1 | need %.0f%% win rate",
        strategy, right, short_strike, long_strike,
        credit_per_ct, max_loss, rr, be_wr * 100,
    )

    _append_row(trade)
    return trade


def record_exit(trade: TradeRecord, exit_reason: str, pnl_per_ct: float):
    """Update a trade with exit information."""
    trade.exit_date = date.today().isoformat()
    trade.exit_reason = exit_reason
    trade.pnl_per_ct = round(pnl_per_ct, 2)
    trade.pnl_total = round(pnl_per_ct * trade.n_contracts, 2)
    trade.is_win = 1 if pnl_per_ct > 0 else 0

    logger.info(
        "EXIT: %s | %s | P&L=$%.0f/ct ($%.0f total) | %s",
        trade.strategy, exit_reason,
        pnl_per_ct, trade.pnl_total,
        "WIN" if trade.is_win else "LOSS",
    )

    _update_row(trade)


def get_performance_summary() -> dict:
    """Calculate running performance metrics from all recorded trades."""
    trades = _load_all()
    closed = [t for t in trades if t.exit_date]

    if not closed:
        return {"total_trades": 0, "message": "No closed trades yet"}

    wins = [t for t in closed if t.is_win]
    losses = [t for t in closed if not t.is_win]

    total_pnl = sum(t.pnl_total for t in closed)
    win_rate = len(wins) / len(closed) * 100

    # Average risk:reward
    avg_rr = sum(t.risk_reward for t in closed) / len(closed)
    # Breakeven win rate for average R:R
    avg_be = sum(t.breakeven_winrate for t in closed) / len(closed)

    # Is our actual win rate beating the breakeven?
    edge = win_rate - avg_be

    avg_win = sum(t.pnl_total for t in wins) / len(wins) if wins else 0
    avg_loss = sum(t.pnl_total for t in losses) / len(losses) if losses else 0

    return {
        "total_trades": len(closed),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": round(win_rate, 1),
        "avg_risk_reward": round(avg_rr, 1),
        "breakeven_winrate": round(avg_be, 1),
        "edge": round(edge, 1),  # positive = profitable, negative = losing
        "profitable": edge > 0,
        "total_pnl": round(total_pnl, 2),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "profit_factor": round(abs(sum(t.pnl_total for t in wins)) / abs(sum(t.pnl_total for t in losses)), 2) if losses else 999,
    }


def print_summary():
    """Print a formatted performance summary."""
    s = get_performance_summary()
    if s["total_trades"] == 0:
        print("No closed trades yet.")
        return

    print(f"\n{'='*60}")
    print(f"LIVE TRADING PERFORMANCE")
    print(f"{'='*60}")
    print(f"  Trades: {s['total_trades']} ({s['wins']}W / {s['losses']}L)")
    print(f"  Win rate: {s['win_rate']:.1f}%")
    print(f"  Avg R:R: {s['avg_risk_reward']:.1f}:1")
    print(f"  Breakeven needed: {s['breakeven_winrate']:.1f}%")
    print(f"  Edge: {s['edge']:+.1f}pp {'(PROFITABLE)' if s['profitable'] else '(LOSING)'}")
    print(f"  Total P&L: ${s['total_pnl']:,.2f}")
    print(f"  Avg win: ${s['avg_win']:,.2f}")
    print(f"  Avg loss: ${s['avg_loss']:,.2f}")
    print(f"  Profit factor: {s['profit_factor']:.2f}")
    print(f"{'='*60}")


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------
def _append_row(trade: TradeRecord):
    """Append a trade to the CSV file."""
    file_exists = TRADES_FILE.exists()
    TRADES_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(TRADES_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(asdict(trade))


def _update_row(trade: TradeRecord):
    """Update an existing trade row (rewrite file)."""
    if not TRADES_FILE.exists():
        _append_row(trade)
        return

    rows = []
    with open(TRADES_FILE, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["trade_id"] == trade.trade_id:
                rows.append(asdict(trade))
            else:
                rows.append(row)

    with open(TRADES_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _load_all() -> list[TradeRecord]:
    """Load all trades from CSV."""
    if not TRADES_FILE.exists():
        return []
    trades = []
    with open(TRADES_FILE, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            t = TradeRecord()
            for k, v in row.items():
                if hasattr(t, k):
                    field_type = type(getattr(t, k))
                    try:
                        if field_type == float:
                            setattr(t, k, float(v) if v else 0.0)
                        elif field_type == int:
                            setattr(t, k, int(float(v)) if v else 0)
                        else:
                            setattr(t, k, v)
                    except (ValueError, TypeError):
                        setattr(t, k, v)
            trades.append(t)
    return trades


if __name__ == "__main__":
    print_summary()
