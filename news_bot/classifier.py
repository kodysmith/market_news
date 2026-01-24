"""
News importance classifier
Assigns importance scores based on keywords, tickers, and macro events
"""
import re
from typing import Dict, List, Tuple
import logging
from . import config
from .utils import extract_tickers, calculate_keyword_score, is_spam_or_promotional

logger = logging.getLogger(__name__)


class NewsClassifier:
    """Classify news articles by importance"""
    
    def __init__(self, qqq_constituents: List[str] = None):
        self.qqq_constituents = set(qqq_constituents or [])
        self.mega_caps = set(config.MEGA_CAP_TICKERS)
        
    def update_qqq_constituents(self, constituents: List[str]):
        """Update the QQQ constituents list"""
        self.qqq_constituents = set(constituents)
        logger.info(f"Updated QQQ constituents: {len(self.qqq_constituents)} tickers")
    
    def classify_article(self, article: Dict) -> Tuple[int, float, List[str], List[str], str]:
        """
        Classify article and return:
        (importance_level, keyword_score, mentioned_tickers, macro_events, reason)
        
        Importance levels:
        5 - Critical (Fed decisions, wars, market crashes)
        4 - High (interest rates, inflation, mega-cap earnings)
        3 - Medium (QQQ stocks, sector news)
        2 - Low (individual stock news)
        1 - Noise (promotional, spam)
        """
        headline = article.get('headline', '')
        text = article.get('raw_summary', '') + ' ' + article.get('full_text', '')
        combined = (headline + ' ' + text).lower()
        
        # Check for spam first
        if is_spam_or_promotional(headline, text):
            return (1, 0.0, [], [], "Spam or promotional content")
        
        # Extract tickers
        mentioned_tickers = extract_tickers(headline + ' ' + text)
        
        # Validate tickers against our watchlist
        valid_tickers = [t for t in mentioned_tickers 
                        if t in self.qqq_constituents or t in self.mega_caps]
        
        # Detect macro events
        macro_events = self._detect_macro_events(combined)
        
        # Calculate keyword scores
        critical_score = calculate_keyword_score(combined, config.CRITICAL_KEYWORDS)
        high_score = calculate_keyword_score(combined, config.HIGH_KEYWORDS)
        medium_score = calculate_keyword_score(combined, config.MEDIUM_KEYWORDS)
        
        # Determine importance level
        importance_level = 1
        reason = "Default low importance"
        keyword_score = 0.0
        
        # Critical level (5)
        if critical_score > 0.3 or len(macro_events) >= 2:
            importance_level = 5
            keyword_score = critical_score
            reason = f"Critical keywords or multiple macro events: {', '.join(macro_events)}"
        
        # High level (4)
        elif high_score > 0.4 or (len(macro_events) >= 1 and high_score > 0.2):
            importance_level = 4
            keyword_score = high_score
            reason = f"High importance keywords or macro event: {', '.join(macro_events)}"
        
        # Mega-cap earnings
        elif any(ticker in self.mega_caps for ticker in valid_tickers) and \
             any(word in combined for word in ['earnings', 'revenue', 'profit', 'guidance', 'results']):
            importance_level = 4
            keyword_score = high_score
            reason = f"Mega-cap earnings: {', '.join(t for t in valid_tickers if t in self.mega_caps)}"
        
        # Medium level (3)
        elif medium_score > 0.3 or (len(valid_tickers) >= 2 and high_score > 0.1):
            importance_level = 3
            keyword_score = medium_score
            reason = f"Medium keywords or multiple QQQ stocks: {', '.join(valid_tickers[:3])}"
        
        # QQQ constituent news
        elif len(valid_tickers) >= 1 and any(ticker in self.qqq_constituents for ticker in valid_tickers):
            importance_level = 3
            keyword_score = medium_score
            reason = f"QQQ constituent news: {', '.join(valid_tickers[:3])}"
        
        # Low level (2)
        elif len(mentioned_tickers) >= 1 or high_score > 0.05:
            importance_level = 2
            keyword_score = max(high_score, medium_score)
            reason = f"Stock mention or low keyword match"
        
        # Boost importance for breaking news indicators
        if any(word in headline.lower() for word in ['breaking', 'alert', 'urgent', 'just in']):
            importance_level = min(importance_level + 1, 5)
            reason += " (Breaking news)"
        
        return (importance_level, keyword_score, valid_tickers, macro_events, reason)
    
    def _detect_macro_events(self, text: str) -> List[str]:
        """Detect macro economic events in text"""
        events = []
        
        # Federal Reserve events
        if any(term in text for term in ['fomc', 'federal reserve decision', 'fed decision', 'powell speech']):
            events.append('FOMC_DECISION')
        elif any(term in text for term in ['interest rate', 'rate hike', 'rate cut', 'rate decision']):
            events.append('INTEREST_RATE')
        
        # Economic indicators
        if any(term in text for term in ['cpi ', 'inflation report', 'consumer price']):
            events.append('CPI_INFLATION')
        
        if any(term in text for term in ['ppi ', 'producer price']):
            events.append('PPI')
        
        if any(term in text for term in ['gdp ', 'economic growth', 'gross domestic']):
            events.append('GDP')
        
        if any(term in text for term in ['unemployment', 'jobs report', 'nonfarm payroll', 'employment']):
            events.append('EMPLOYMENT')
        
        # Geopolitical events
        if any(term in text for term in ['war', 'military conflict', 'invasion', 'missile', 'nuclear']):
            events.append('GEOPOLITICAL_CONFLICT')
        
        if any(term in text for term in ['tariff', 'trade war', 'trade deal', 'china trade']):
            events.append('TRADE_POLICY')
        
        # Market structure
        if any(term in text for term in ['market crash', 'circuit breaker', 'trading halt', 'market selloff']):
            events.append('MARKET_DISRUPTION')
        
        if any(term in text for term in ['bank failure', 'credit crisis', 'liquidity crisis', 'systemic risk']):
            events.append('FINANCIAL_CRISIS')
        
        # Energy/Commodities
        if any(term in text for term in ['oil price', 'crude oil', 'opec', 'energy crisis']):
            events.append('ENERGY_MARKET')
        
        return events
    
    def classify_batch(self, articles: List[Dict]) -> List[Dict]:
        """
        Classify a batch of articles
        Returns articles with classification data added
        """
        classified = []
        
        for article in articles:
            try:
                importance, score, tickers, events, reason = self.classify_article(article)
                
                article['importance_level'] = importance
                article['keyword_score'] = score
                article['mentioned_tickers'] = tickers
                article['macro_events'] = events
                article['classification_reason'] = reason
                
                classified.append(article)
            
            except Exception as e:
                logger.error(f"Error classifying article: {e}")
                # Add default classification
                article['importance_level'] = 2
                article['keyword_score'] = 0.0
                article['mentioned_tickers'] = []
                article['macro_events'] = []
                article['classification_reason'] = f"Classification error: {e}"
                classified.append(article)
        
        # Log statistics
        importance_counts = {}
        for article in classified:
            level = article['importance_level']
            importance_counts[level] = importance_counts.get(level, 0) + 1
        
        logger.info(f"Classified {len(classified)} articles: {importance_counts}")
        
        return classified


if __name__ == "__main__":
    # Test the classifier
    logging.basicConfig(level=logging.INFO)
    
    # Sample articles
    test_articles = [
        {
            'headline': 'Fed Announces Emergency Interest Rate Cut',
            'raw_summary': 'The Federal Reserve announced an emergency 50 basis point rate cut...',
            'full_text': 'Breaking: FOMC emergency meeting results in rate cut'
        },
        {
            'headline': 'Apple Reports Record Q4 Earnings',
            'raw_summary': 'AAPL beats earnings expectations with strong iPhone sales',
            'full_text': 'Apple Inc reported quarterly earnings...'
        },
        {
            'headline': 'Small Tech Company Announces New Product',
            'raw_summary': 'XYZ Corp launches new software product',
            'full_text': 'A small technology company...'
        }
    ]
    
    classifier = NewsClassifier(qqq_constituents=['AAPL', 'MSFT', 'NVDA', 'GOOGL'])
    
    for article in test_articles:
        importance, score, tickers, events, reason = classifier.classify_article(article)
        print(f"\nHeadline: {article['headline']}")
        print(f"Importance: {importance}/5")
        print(f"Score: {score:.2f}")
        print(f"Tickers: {tickers}")
        print(f"Events: {events}")
        print(f"Reason: {reason}")


