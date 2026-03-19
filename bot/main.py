"""SPX Iron Condor Bot — main entry point.

Usage:
    python -m bot.main --mode dry-run    # No orders, just log decisions
    python -m bot.main --mode paper      # IBKR paper account (port 7497)
    python -m bot.main --mode live       # IBKR live account (port 7496)
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Configure logging before any imports that might log
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("bot.main")


class BotRunner:
    """Orchestrates all scheduled jobs.

    One instance is shared across all APScheduler callbacks.
    """

    def __init__(self, mode: str):
        self.mode = mode
        self._snapshot = None   # cached until next morning_data_refresh
        self._prev_above_sma = None  # track SMA50 crossovers

        from bot.state_manager import StateManager
        from bot.order_manager import OrderManager

        self.state = StateManager()
        self.orders = OrderManager(mode=mode)

    # ------------------------------------------------------------------
    # Scheduled jobs
    # ------------------------------------------------------------------

    def morning_data_refresh(self) -> None:
        """9:25 AM — fetch and cache market snapshot."""
        from bot.market_data import get_market_snapshot
        from bot import metrics

        logger.info("=== Morning data refresh ===")
        try:
            self._snapshot = get_market_snapshot()
            metrics.record_heartbeat()
            metrics.record_job_run("morning_data_refresh")
        except Exception:
            metrics.record_job_error("morning_data_refresh")
            raise
        logger.info(
            "Snapshot: SPX=%.1f  VIX=%.1f  SMA50=%.1f  rv_ratio=%.3f  "
            "gap=%.2f%%  news=%s",
            self._snapshot.spx_price,
            self._snapshot.vix_level,
            self._snapshot.sma50,
            self._snapshot.rv_ratio,
            self._snapshot.gap_pct,
            self._snapshot.is_news_day,
        )

        # Detect SMA50 crossover and send FCM alert
        above_sma = self._snapshot.spx_price > self._snapshot.sma50
        if self._prev_above_sma is not None and above_sma != self._prev_above_sma:
            from bot import notifier
            notifier.trend_break(
                self._snapshot.spx_price,
                self._snapshot.sma50,
                crossed_above=above_sma,
            )
            logger.info("SMA50 crossover detected: %s", "above" if above_sma else "below")
        self._prev_above_sma = above_sma

    def run_entry(self) -> None:
        """9:30 AM — evaluate and place new IC entries."""
        from bot.config import DRAWDOWN_PAUSE_PCT, TOTAL_ACCOUNT
        from bot.entry_engine import run_entry_checks
        from bot import notifier

        logger.info("=== Entry engine ===")

        if self.state.is_paused():
            logger.info("Entries paused — skipping")
            return

        snapshot = self._get_snapshot()
        if snapshot is None:
            return

        # Drawdown circuit breaker
        open_positions = self.state.get_open_positions()
        cumulative_pnl = self.state.get_cumulative_pnl()
        equity = TOTAL_ACCOUNT + cumulative_pnl
        ath = self.state.get_ath_equity()
        if ath > 0 and (ath - equity) / ath >= DRAWDOWN_PAUSE_PCT:
            dd_pct = (ath - equity) / ath * 100
            msg = f"Drawdown {dd_pct:.1f}% exceeds {DRAWDOWN_PAUSE_PCT*100:.0f}% limit"
            logger.warning("CIRCUIT BREAKER: %s — pausing entries", msg)
            self.state.set_paused(True)
            notifier.bot_paused(msg)
            notifier.risk_alert(msg, dd_pct)
            return

        vd2_contracts = self.state.get_vd2_contract_count()
        orders, decisions = run_entry_checks(snapshot, open_positions, vd2_contracts, self.mode)

        # Publish all decisions (enter or skip) to Supabase for the app's Trade Ideas screen
        from bot import publisher
        publisher.publish_bot_decisions(decisions, snapshot)

        if not orders:
            logger.info("No entry signals today")
            return

        for entry in orders:
            # Margin check before entry
            buying_power = self.orders.get_account_buying_power()
            required_bp = entry.n_contracts * (entry.k_put_short - entry.k_put_long) * 100
            from bot.config import MARGIN_BUFFER_MULT
            if buying_power > 0 and buying_power < required_bp * MARGIN_BUFFER_MULT:
                logger.warning(
                    "Insufficient buying power for %s: need $%.0f have $%.0f",
                    entry.config_id, required_bp * MARGIN_BUFFER_MULT, buying_power,
                )
                continue

            result = self.orders.place_iron_condor(entry)

            if result.success:
                from bot import metrics
                metrics.record_order_fill(entry.config_id, "entry")
                metrics.log_entry(entry.config_id, entry.expiration.isoformat(),
                                  entry.n_contracts, result.fill_price,
                                  entry.spx_at_decision, entry.vix_at_decision)
                from bot.state_manager import Position
                pos = Position(
                    id="",  # assigned by Supabase
                    config_id=entry.config_id,
                    entry_date=date.today(),
                    expiration=entry.expiration,
                    k_put_short=entry.k_put_short,
                    k_put_long=entry.k_put_long,
                    k_call_short=entry.k_call_short,
                    k_call_long=entry.k_call_long,
                    credit_collected=result.fill_price,
                    n_contracts=entry.n_contracts,
                    spx_entry=entry.spx_at_decision,
                    vix_entry=entry.vix_at_decision,
                )
                self.state.save_position(pos)
                notifier.trade_opened(
                    config_id=entry.config_id,
                    expiration=entry.expiration.isoformat(),
                    k_put_short=entry.k_put_short,
                    k_put_long=entry.k_put_long,
                    k_call_short=entry.k_call_short,
                    k_call_long=entry.k_call_long,
                    credit=entry.credit,
                    n_contracts=entry.n_contracts,
                    fill_price=result.fill_price,
                )
            else:
                from bot import metrics
                metrics.record_order_error(entry.config_id)
                logger.error("Entry order failed for %s: %s", entry.config_id, result.message)

    def run_exit_scan(self) -> None:
        """Every 15 min 9:30–15:45 — scan open positions for exit signals."""
        from bot.market_data import get_market_snapshot
        from bot.exit_engine import scan_all_exits
        from bot import notifier, metrics

        metrics.record_heartbeat()
        logger.debug("=== Exit scan ===")
        open_positions = self.state.get_open_positions()
        if not open_positions:
            return

        snapshot = get_market_snapshot()
        signals = scan_all_exits(open_positions, snapshot, self.mode)

        for signal in signals:
            pos = next((p for p in open_positions if p.id == signal.position_id), None)
            if pos is None:
                continue

            result = self.orders.close_iron_condor(pos, signal.reason)
            if result.success:
                self.state.close_position(
                    pos.id, date.today(), result.fill_price, signal.reason,
                )
                cumulative = self.state.get_cumulative_pnl()
                pnl = (pos.credit_collected - result.fill_price) * 100 * pos.n_contracts
                notifier.trade_closed(pos.config_id, signal.reason, pnl, cumulative,
                                      pos.n_contracts)
                from bot import metrics
                metrics.record_order_fill(pos.config_id, "exit")
                metrics.record_trade_closed(signal.reason)
                metrics.log_exit(pos.config_id, signal.reason, pnl, cumulative)
            else:
                logger.error(
                    "Close order failed for position %s: %s",
                    pos.id, result.message,
                )

    def run_eod(self) -> None:
        """3:45 PM — close any time-stop positions before market close."""
        logger.info("=== EOD sweep ===")
        # run_exit_scan handles all exit logic including time_stop_dte
        self.run_exit_scan()

    def record_closes(self) -> None:
        """4:05 PM — record close prices, run compound check, send daily summary."""
        from bot.config import TOTAL_ACCOUNT
        from bot import notifier

        logger.info("=== Record closes ===")

        # Update ATH equity
        cumulative_pnl = self.state.get_cumulative_pnl()
        equity = TOTAL_ACCOUNT + cumulative_pnl
        ath = self.state.get_ath_equity()
        if equity > ath:
            self.state.update_ath_equity(equity)
            logger.info("New ATH equity: $%.0f", equity)

        # Compound check
        promoted = self.state.check_and_apply_compound()
        if promoted:
            new_count = self.state.get_vd2_contract_count()
            notifier.compound_milestone(new_count, cumulative_pnl)

        # Daily summary notification
        open_positions = self.state.get_open_positions()
        ytd_roi = cumulative_pnl / TOTAL_ACCOUNT * 100
        notifier.daily_summary(
            open_positions=len(open_positions),
            day_pnl=0.0,  # Intraday tracking requires position repricing; simplified here
            cumulative_pnl=cumulative_pnl,
            ytd_roi_pct=ytd_roi,
            vd2_contracts=self.state.get_vd2_contract_count(),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_snapshot(self):
        """Return cached snapshot or fetch fresh if missing."""
        if self._snapshot is None:
            from bot.market_data import get_market_snapshot
            logger.warning("No cached snapshot — fetching fresh")
            self._snapshot = get_market_snapshot()
        return self._snapshot


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _confirm_live() -> bool:
    """Require explicit confirmation before live trading."""
    print("\n" + "=" * 60)
    print("  WARNING: LIVE TRADING MODE")
    print("  Real money will be at risk. Orders will be placed in")
    print("  your LIVE IBKR account.")
    print("=" * 60)
    answer = input("\nType 'yes I understand' to proceed: ")
    return answer.strip().lower() == "yes i understand"


def main():
    parser = argparse.ArgumentParser(description="SPX Iron Condor Bot")
    parser.add_argument(
        "--mode",
        choices=["dry-run", "paper", "live"],
        default="dry-run",
        help="Execution mode (default: dry-run)",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run entry+exit check once and exit (useful for Cloud Run jobs)",
    )
    args = parser.parse_args()

    if args.mode == "live" and not _confirm_live():
        print("Aborted.")
        sys.exit(0)

    logger.info("Starting SPX IC Bot | mode=%s | %s", args.mode, datetime.now().isoformat())

    runner = BotRunner(args.mode)

    # Start Prometheus metrics server (no-op if prometheus_client not installed)
    from bot.metrics import start_metrics_server
    start_metrics_server()

    if args.once:
        # Single-shot mode for Cloud Run invocations
        runner.morning_data_refresh()
        runner.run_entry()
        runner.run_exit_scan()
        logger.info("Single-shot run complete.")
        return

    # Daemon mode: build scheduler and start
    from bot.scheduler import build_scheduler
    scheduler = build_scheduler(runner)

    logger.info("Scheduler started. Press Ctrl+C to stop.")
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Bot stopped.")


if __name__ == "__main__":
    main()
