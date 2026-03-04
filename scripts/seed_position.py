#!/usr/bin/env python3
"""One-time script to seed a manually-entered iron condor into bot_positions.

Usage:
    python scripts/seed_position.py [--dry-run]

Loads .env from repo root, then inserts the position into Supabase so the bot
picks it up for exit monitoring immediately.
"""
from __future__ import annotations

import os
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Load .env
env_path = ROOT / ".env"
if env_path.exists():
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip())

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dry-run", action="store_true", help="Print what would be inserted, don't write")
args = parser.parse_args()

# ── Position details ───────────────────────────────────────────────────────
# SPX Iron Condor — Mar 26, 2026 expiration
# Entered ~Mar 1, 2026 | SPX ~6878.88

POSITION = dict(
    config_id     = "MANUAL_IC",          # tag so we know it was hand-entered
    entry_date    = date(2026, 3, 1),      # adjust if entered on a different day
    expiration    = date(2026, 3, 26),
    # Put spread: Short 6575 / Long 6525  (width 50)
    k_put_short   = 6575.0,
    k_put_long    = 6525.0,
    # Call spread: Short 7050 / Long 7060 (width 10)
    k_call_short  = 7050.0,
    k_call_long   = 7060.0,
    # Total credit collected (put 14.30 + call 1.95)
    credit_collected = 16.25,
    n_contracts   = 1,
    spx_entry     = 6878.88,
    vix_entry     = 17.0,                 # approximate at time of entry
    status        = "open",
)

# ── Max loss check (informational) ────────────────────────────────────────
put_width  = POSITION["k_put_short"] - POSITION["k_put_long"]    # 50
call_width = POSITION["k_call_long"] - POSITION["k_call_short"]  # 10
total_credit = POSITION["credit_collected"]
max_loss_put  = (put_width  - total_credit) * 100
max_loss_call = (call_width - total_credit) * 100
# True IC max loss: worst single side net of total credit
max_loss = max(put_width, call_width) * 100 - total_credit * 100

print("=" * 55)
print("  Seeding Manual Iron Condor Position")
print("=" * 55)
print(f"  Expiration : {POSITION['expiration']}")
print(f"  Put spread : {POSITION['k_put_short']:.0f} / {POSITION['k_put_long']:.0f}  (width {put_width:.0f})")
print(f"  Call spread: {POSITION['k_call_short']:.0f} / {POSITION['k_call_long']:.0f} (width {call_width:.0f})")
print(f"  Credit     : {total_credit:.2f} × 100 = ${total_credit*100:,.0f}")
print(f"  Max loss   : ${max_loss:,.0f}  (put side dominates)")
print(f"  SPX entry  : {POSITION['spx_entry']}")
print(f"  Contracts  : {POSITION['n_contracts']}")
print()

if args.dry_run:
    print("[DRY RUN] Would insert:", POSITION)
    sys.exit(0)

# ── Write to Supabase ─────────────────────────────────────────────────────
from bot.state_manager import StateManager, Position

state = StateManager()
pos = Position(
    id="",  # assigned by Supabase
    **POSITION,
)

new_id = state.save_position(pos)
print(f"  Inserted → id: {new_id}")
print()
print("  Bot will now monitor this position for exits at the next scan.")
print("=" * 55)
