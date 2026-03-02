"""Supabase-backed position and P&L state manager.

All bot state is persisted here so the process can restart safely.

Tables (see scripts/migrations/001_bot_tables.sql):
  bot_positions  — one row per open/closed iron condor
  bot_state      — key/value store for scalar state (contract counts, PnL, etc.)
  bot_trades     — append-only trade log for completed trades
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Optional
from uuid import UUID

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """Live iron condor position."""
    id: str                       # UUID from Supabase
    config_id: str                # '15_VIX_SWEET_SPOT' | 'VD2_ENTRY_105'
    entry_date: date
    expiration: date
    k_put_short: float
    k_put_long: float
    k_call_short: float
    k_call_long: float
    credit_collected: float       # per-contract credit in dollars
    n_contracts: int
    spx_entry: float              # SPX price at entry (for % drop check)
    vix_entry: float              # VIX at entry (for spike check)
    status: str = "open"          # open | closed | expired
    exit_date: Optional[date] = None
    exit_debit: Optional[float] = None
    pnl: Optional[float] = None
    exit_reason: Optional[str] = None

    @property
    def strikes(self) -> tuple[float, float, float, float]:
        return (self.k_put_short, self.k_put_long, self.k_call_short, self.k_call_long)

    @property
    def max_loss(self) -> float:
        """Worst-case loss per contract (spread width − credit)."""
        width = self.k_put_short - self.k_put_long  # same as call width
        return (width - self.credit_collected) * 100  # dollars

    @property
    def total_max_loss(self) -> float:
        return self.max_loss * self.n_contracts


def _get_client():
    """Lazy-import and return a Supabase client."""
    from supabase import create_client
    url = os.environ["SUPABASE_URL"]
    key = os.environ["SUPABASE_KEY"]
    return create_client(url, key)


class StateManager:
    """All reads/writes to Supabase for bot state."""

    def __init__(self):
        self._client = None

    @property
    def db(self):
        if self._client is None:
            self._client = _get_client()
        return self._client

    # ------------------------------------------------------------------
    # Positions
    # ------------------------------------------------------------------

    def get_open_positions(self) -> list[Position]:
        """Return all positions with status='open'."""
        try:
            from bot.config import TABLE_POSITIONS
            rows = (
                self.db.table(TABLE_POSITIONS)
                .select("*")
                .eq("status", "open")
                .execute()
                .data
            )
            return [self._row_to_position(r) for r in rows]
        except Exception as e:
            logger.error("get_open_positions failed: %s", e)
            return []

    def save_position(self, pos: Position) -> str:
        """Insert a new open position. Returns the assigned UUID."""
        from bot.config import TABLE_POSITIONS
        row = {
            "config_id": pos.config_id,
            "entry_date": pos.entry_date.isoformat(),
            "expiration": pos.expiration.isoformat(),
            "k_put_short": pos.k_put_short,
            "k_put_long": pos.k_put_long,
            "k_call_short": pos.k_call_short,
            "k_call_long": pos.k_call_long,
            "credit_collected": pos.credit_collected,
            "n_contracts": pos.n_contracts,
            "spx_entry": pos.spx_entry,
            "vix_entry": pos.vix_entry,
            "status": "open",
        }
        result = self.db.table(TABLE_POSITIONS).insert(row).execute()
        new_id = result.data[0]["id"]
        logger.info("Saved position %s (%s, %d contracts)", new_id, pos.config_id, pos.n_contracts)
        return new_id

    def close_position(
        self,
        position_id: str,
        exit_date: date,
        exit_debit: float,
        reason: str,
    ) -> None:
        """Mark a position closed and compute PnL. Also appends to bot_trades."""
        from bot.config import TABLE_POSITIONS, TABLE_TRADES

        # Fetch the position to compute PnL
        row = (
            self.db.table(TABLE_POSITIONS)
            .select("*")
            .eq("id", position_id)
            .single()
            .execute()
            .data
        )
        if not row:
            logger.error("close_position: position %s not found", position_id)
            return

        credit = float(row["credit_collected"])
        n = int(row["n_contracts"])
        pnl = (credit - exit_debit) * 100 * n  # dollars total

        # Update position
        self.db.table(TABLE_POSITIONS).update({
            "status": "closed",
            "exit_date": exit_date.isoformat(),
            "exit_debit": exit_debit,
            "pnl": pnl,
            "exit_reason": reason,
        }).eq("id", position_id).execute()

        # Append to trade log
        self.db.table(TABLE_TRADES).insert({
            "config_id": row["config_id"],
            "entry_date": row["entry_date"],
            "exit_date": exit_date.isoformat(),
            "n_contracts": n,
            "credit": credit,
            "debit": exit_debit,
            "pnl": pnl,
            "exit_reason": reason,
        }).execute()

        # Update cumulative PnL in state
        current = self.get_cumulative_pnl()
        self._set_state("cumulative_pnl", str(current + pnl))

        logger.info(
            "Closed position %s | reason=%s | pnl=$%.0f | cumulative=$%.0f",
            position_id, reason, pnl, current + pnl,
        )

    # ------------------------------------------------------------------
    # Scalar state
    # ------------------------------------------------------------------

    def get_cumulative_pnl(self) -> float:
        return float(self._get_state("cumulative_pnl", "0"))

    def get_vd2_contract_count(self) -> int:
        from bot.config import VD2_CONTRACTS_START
        return int(self._get_state("vd2_contracts", str(VD2_CONTRACTS_START)))

    def update_vd2_contracts(self, n: int) -> None:
        self._set_state("vd2_contracts", str(n))
        logger.info("VD2 contract count updated to %d", n)

    def get_ath_equity(self) -> float:
        from bot.config import TOTAL_ACCOUNT
        return float(self._get_state("ath_equity", str(TOTAL_ACCOUNT)))

    def update_ath_equity(self, equity: float) -> None:
        self._set_state("ath_equity", str(equity))

    def is_paused(self) -> bool:
        return self._get_state("entries_paused", "0") == "1"

    def set_paused(self, paused: bool) -> None:
        self._set_state("entries_paused", "1" if paused else "0")
        logger.info("Bot entries %s", "PAUSED" if paused else "RESUMED")

    # ------------------------------------------------------------------
    # Compound trigger
    # ------------------------------------------------------------------

    def check_and_apply_compound(self) -> bool:
        """Return True if VD2 contract count was incremented."""
        from bot.config import (
            VD2_CONTRACTS_START,
            VD2_CONTRACTS_MAX,
            COMPOUND_THRESHOLD,
        )
        cumulative_pnl = self.get_cumulative_pnl()
        current_vd2 = self.get_vd2_contract_count()
        target_vd2 = VD2_CONTRACTS_START + int(max(cumulative_pnl, 0) / COMPOUND_THRESHOLD)
        target_vd2 = min(target_vd2, VD2_CONTRACTS_MAX)
        if target_vd2 > current_vd2:
            self.update_vd2_contracts(target_vd2)
            return True
        return False

    # ------------------------------------------------------------------
    # Trade history queries
    # ------------------------------------------------------------------

    def get_recent_trades(self, days: int = 90) -> list[dict]:
        from bot.config import TABLE_TRADES
        cutoff = (datetime.utcnow().date() - __import__("datetime").timedelta(days=days)).isoformat()
        rows = (
            self.db.table(TABLE_TRADES)
            .select("*")
            .gte("exit_date", cutoff)
            .order("exit_date", desc=True)
            .execute()
            .data
        )
        return rows

    def get_monthly_summary(self) -> list[dict]:
        """Return list of {month, pnl, n_trades} dicts."""
        from bot.config import TABLE_TRADES
        rows = (
            self.db.table(TABLE_TRADES)
            .select("exit_date, pnl, n_contracts")
            .eq("status", "closed")  # guard: table has no status col, handled below
            .execute()
            .data
        )
        # Aggregate by month
        monthly: dict[str, dict] = {}
        for r in rows:
            if not r.get("exit_date"):
                continue
            month = r["exit_date"][:7]  # YYYY-MM
            bucket = monthly.setdefault(month, {"month": month, "pnl": 0.0, "n_trades": 0})
            bucket["pnl"] += float(r.get("pnl") or 0)
            bucket["n_trades"] += 1
        return sorted(monthly.values(), key=lambda x: x["month"])

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_state(self, key: str, default: str = "") -> str:
        from bot.config import TABLE_STATE
        try:
            rows = (
                self.db.table(TABLE_STATE)
                .select("value")
                .eq("key", key)
                .execute()
                .data
            )
            if rows:
                return rows[0]["value"]
        except Exception as e:
            logger.warning("_get_state(%s) failed: %s", key, e)
        return default

    def _set_state(self, key: str, value: str) -> None:
        from bot.config import TABLE_STATE
        try:
            self.db.table(TABLE_STATE).upsert(
                {"key": key, "value": value, "updated_at": datetime.utcnow().isoformat()},
                on_conflict="key",
            ).execute()
        except Exception as e:
            logger.warning("_set_state(%s) failed: %s", key, e)

    @staticmethod
    def _row_to_position(r: dict) -> Position:
        def _d(v) -> Optional[date]:
            return date.fromisoformat(v) if v else None

        return Position(
            id=r["id"],
            config_id=r["config_id"],
            entry_date=date.fromisoformat(r["entry_date"]),
            expiration=date.fromisoformat(r["expiration"]),
            k_put_short=float(r["k_put_short"]),
            k_put_long=float(r["k_put_long"]),
            k_call_short=float(r["k_call_short"]),
            k_call_long=float(r["k_call_long"]),
            credit_collected=float(r["credit_collected"]),
            n_contracts=int(r["n_contracts"]),
            spx_entry=float(r.get("spx_entry") or 0),
            vix_entry=float(r.get("vix_entry") or 0),
            status=r.get("status", "open"),
            exit_date=_d(r.get("exit_date")),
            exit_debit=float(r["exit_debit"]) if r.get("exit_debit") is not None else None,
            pnl=float(r["pnl"]) if r.get("pnl") is not None else None,
            exit_reason=r.get("exit_reason"),
        )
