"""
Fisher Company Valuation Automation.

Philip Fisher-style qualitative scoring pipeline:
- SEC EDGAR ingestion (10-K, 10-Q, 8-K, DEF14A)
- XBRL-derived financial facts
- Market data (daily OHLCV)
- Transcripts (MVP-2)
- Per-point scoring with confidence and evidence
"""

__all__ = [
    "db",
    "config",
    "edgar_watcher",
    "xbrl_parser",
    "market_updater",
    "scoring_engine",
]
