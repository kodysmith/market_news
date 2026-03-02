"""APScheduler job definitions for the SPX IC bot.

All times are America/New_York. Jobs only fire on market days
(NYSE calendar via pandas_market_calendars).
"""
from __future__ import annotations

import logging
from datetime import datetime
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

ET = ZoneInfo("America/New_York")


def _is_market_day() -> bool:
    """Return True if today is a NYSE trading day."""
    try:
        import pandas_market_calendars as mcal
        cal = mcal.get_calendar("NYSE")
        today_str = datetime.now(ET).strftime("%Y-%m-%d")
        schedule = cal.schedule(start_date=today_str, end_date=today_str)
        return not schedule.empty
    except Exception as e:
        logger.warning("Market day check failed: %s — assuming open", e)
        # Fallback: Mon-Fri only
        return datetime.now(ET).weekday() < 5


def build_scheduler(bot_runner) -> object:
    """Create and return a configured APScheduler BlockingScheduler.

    Args:
        bot_runner: Object with methods:
            - morning_data_refresh()
            - run_entry()
            - run_exit_scan()
            - run_eod()
            - record_closes()
    """
    from apscheduler.schedulers.blocking import BlockingScheduler
    from apscheduler.triggers.cron import CronTrigger

    scheduler = BlockingScheduler(timezone=ET)

    def guarded(fn):
        """Wrap a job so it skips on non-market days and logs exceptions."""
        def wrapper(*args, **kwargs):
            if not _is_market_day():
                logger.debug("Non-market day — skipping %s", fn.__name__)
                return
            try:
                fn(*args, **kwargs)
            except Exception as e:
                logger.error("Job %s failed: %s", fn.__name__, e, exc_info=True)
        wrapper.__name__ = fn.__name__
        return wrapper

    # 9:25 AM — fetch market data
    scheduler.add_job(
        guarded(bot_runner.morning_data_refresh),
        CronTrigger(hour=9, minute=25, timezone=ET),
        id="morning_data_refresh",
        name="Morning data refresh",
        misfire_grace_time=120,
    )

    # 9:30 AM — entry check
    scheduler.add_job(
        guarded(bot_runner.run_entry),
        CronTrigger(hour=9, minute=30, timezone=ET),
        id="entry_engine",
        name="Entry engine",
        misfire_grace_time=120,
    )

    # Every 15 min from 9:30 to 15:45 — exit scan
    scheduler.add_job(
        guarded(bot_runner.run_exit_scan),
        CronTrigger(hour="9-15", minute="0,15,30,45", timezone=ET),
        id="exit_scan",
        name="Exit engine scan",
        misfire_grace_time=60,
    )

    # 3:45 PM — EOD time-stop sweep
    scheduler.add_job(
        guarded(bot_runner.run_eod),
        CronTrigger(hour=15, minute=45, timezone=ET),
        id="eod_engine",
        name="EOD engine",
        misfire_grace_time=120,
    )

    # 4:05 PM — record close prices, update compound state
    scheduler.add_job(
        guarded(bot_runner.record_closes),
        CronTrigger(hour=16, minute=5, timezone=ET),
        id="record_closes",
        name="Record closes & compound check",
        misfire_grace_time=300,
    )

    return scheduler
