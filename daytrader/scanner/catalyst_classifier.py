"""Layer 2: Catalyst Classifier — score each candidate's news catalyst.

Wraps the existing news_bot infrastructure to fetch and classify headlines
for scanner candidates. Maps importance levels to a 0-10 catalyst score.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from daytrader.config import CATALYST_SCORES, CATALYST_MIN_SCORE

logger = logging.getLogger("daytrader.catalyst")

# Keywords for catalyst classification
EARNINGS_KEYWORDS = [
    "earnings", "revenue", "eps", "quarterly results", "q1", "q2", "q3", "q4",
    "fiscal", "guidance", "outlook", "beat", "exceeded", "surpassed",
]
FDA_KEYWORDS = [
    "fda", "approval", "breakthrough", "clinical trial", "phase 3", "phase 2",
    "drug", "therapy", "pdufa", "nda", "eua",
]
CONTRACT_KEYWORDS = [
    "contract", "awarded", "deal", "billion", "million dollar", "partnership",
    "agreement", "strategic", "acquisition", "merger", "buyout", "tender offer",
]
ANALYST_KEYWORDS = [
    "upgrade", "downgrade", "price target", "initiated", "overweight",
    "buy rating", "outperform",
]
SOCIAL_KEYWORDS = [
    "reddit", "wallstreetbets", "meme", "short squeeze", "trending",
    "viral", "retail traders",
]


def classify_headline(headline: str) -> tuple[str, int]:
    """
    Classify a news headline into a catalyst type and score.

    Returns (catalyst_type, score).
    """
    if not headline:
        return "unknown", 0

    h = headline.lower()

    # Check in order of priority
    if any(k in h for k in FDA_KEYWORDS):
        if "approval" in h or "approved" in h:
            return "fda_approval", CATALYST_SCORES["fda_approval"]
        return "fda_approval", 7  # trial data, not approval

    if any(k in h for k in EARNINGS_KEYWORDS):
        if any(w in h for w in ["guidance", "raises", "outlook", "raised"]):
            return "earnings_beat_guidance", CATALYST_SCORES["earnings_beat_guidance"]
        if any(w in h for w in ["beat", "exceeded", "surpassed", "topped"]):
            return "earnings_beat", CATALYST_SCORES["earnings_beat"]
        return "earnings_beat", 5  # earnings mention but no clear beat

    if "acquisition" in h or "merger" in h or "buyout" in h or "tender" in h:
        return "acquisition", CATALYST_SCORES["acquisition"]

    if any(k in h for k in CONTRACT_KEYWORDS):
        return "major_contract", CATALYST_SCORES["major_contract"]

    if any(k in h for k in ANALYST_KEYWORDS):
        return "analyst_upgrade", CATALYST_SCORES["analyst_upgrade"]

    if any(k in h for k in SOCIAL_KEYWORDS):
        return "social_media", CATALYST_SCORES["social_media"]

    # Generic positive/negative
    if any(w in h for w in ["partnership", "collaboration", "strategic"]):
        return "partnership", CATALYST_SCORES["partnership"]

    return "unknown", CATALYST_SCORES["unknown"]


def fetch_and_classify(symbol: str) -> tuple[str, int, str]:
    """
    Fetch news for a symbol and classify the top catalyst.

    Returns (catalyst_type, score, headline).
    Uses the existing news_bot if available, falls back to FMP API directly.
    """
    # Try to use existing news bot infrastructure
    try:
        from news_bot.news_aggregator import NewsAggregator
        agg = NewsAggregator()
        articles = agg.fetch_fmp_news(ticker=symbol, limit=5)
        if articles:
            best_type, best_score, best_headline = "unknown", 0, ""
            for article in articles[:5]:
                title = article.get("title", "") or article.get("headline", "")
                cat_type, cat_score = classify_headline(title)
                if cat_score > best_score:
                    best_type = cat_type
                    best_score = cat_score
                    best_headline = title
            return best_type, best_score, best_headline
    except Exception as e:
        logger.debug("News bot unavailable: %s, trying FMP directly", e)

    # Fallback: direct FMP API call
    try:
        import os
        import requests

        fmp_key = os.getenv("FMP_API_KEY", "")
        if not fmp_key:
            env_file = ROOT / ".env"
            if env_file.is_file():
                for line in open(env_file):
                    if line.strip().startswith("FMP_API_KEY="):
                        fmp_key = line.strip().split("=", 1)[1].strip().strip('"')
                        break

        if fmp_key:
            url = f"https://financialmodelingprep.com/api/v3/stock_news?tickers={symbol}&limit=5&apikey={fmp_key}"
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                articles = resp.json()
                best_type, best_score, best_headline = "unknown", 0, ""
                for article in articles:
                    title = article.get("title", "")
                    cat_type, cat_score = classify_headline(title)
                    if cat_score > best_score:
                        best_type = cat_type
                        best_score = cat_score
                        best_headline = title
                return best_type, best_score, best_headline
    except Exception as e:
        logger.debug("FMP direct call failed: %s", e)

    return "unknown", 0, ""


def enrich_candidates(candidates: list) -> list:
    """
    Enrich scanner candidates with catalyst scores.
    Modifies candidates in place and returns them sorted by catalyst score.
    """
    for c in candidates:
        cat_type, cat_score, headline = fetch_and_classify(c.symbol)
        c.catalyst_type = cat_type
        c.catalyst_score = cat_score
        c.catalyst_headline = headline
        logger.info("  %s: catalyst=%s score=%d '%s'",
                    c.symbol, cat_type, cat_score, headline[:80])

    # Filter by minimum catalyst score
    qualified = [c for c in candidates if c.catalyst_score >= CATALYST_MIN_SCORE]
    qualified.sort(key=lambda c: (c.catalyst_score, c.gap_pct * c.rel_vol), reverse=True)

    logger.info("Catalyst filter: %d/%d candidates qualified (score >= %d)",
                len(qualified), len(candidates), CATALYST_MIN_SCORE)
    return qualified
