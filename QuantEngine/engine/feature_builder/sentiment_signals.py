"""
Sentiment Signals for AI Quant Trading System

Creates trading signals from news, filings, and text data.
Implements basic sentiment analysis for Phase 1.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
import re
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)


class SentimentSignalGenerator:
    """Generate sentiment-based trading signals"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

        # Sentiment keywords and weights
        self.positive_words = {
            'upgrade': 2.0, 'upgraded': 2.0, 'buy': 1.5, 'strong': 1.0, 'positive': 1.5,
            'beat': 1.8, 'beats': 1.8, 'exceed': 1.5, 'exceeds': 1.5, 'surprise': 1.2,
            'gains': 1.0, 'growth': 1.2, 'bullish': 1.8, 'optimistic': 1.3, 'rally': 1.0,
            'breakthrough': 1.5, 'achievement': 1.2, 'milestone': 1.0, 'partnership': 1.3,
            'acquisition': 1.4, 'merger': 1.2, 'expansion': 1.1, 'launch': 1.0
        }

        self.negative_words = {
            'downgrade': -2.0, 'downgraded': -2.0, 'sell': -1.5, 'weak': -1.0, 'negative': -1.5,
            'miss': -1.8, 'misses': -1.8, 'below': -1.5, 'disappoint': -1.5, 'disappointed': -1.5,
            'decline': -1.2, 'losses': -1.5, 'bearish': -1.8, 'pessimistic': -1.3, 'crash': -1.5,
            'lawsuit': -1.4, 'scandal': -1.8, 'investigation': -1.5, 'bankruptcy': -2.0,
            'layoffs': -1.3, 'restructuring': -1.2, 'delays': -1.0, 'concerns': -1.2
        }

        # Filing-specific keywords
        self.filing_positive = {
            'guidance': 0.5, 'forecast': 0.3, 'growth': 0.8, 'expansion': 0.6,
            'investment': 0.4, 'dividend': 1.0, 'buyback': 1.2, 'acquisition': 0.8
        }

        self.filing_negative = {
            'restructuring': -0.8, 'impairment': -1.0, 'losses': -1.2, 'write-down': -1.0,
            'litigation': -0.8, 'investigation': -0.6, 'delays': -0.4, 'challenges': -0.5
        }

    def generate_sentiment_signals(self, news_data: Dict[str, pd.DataFrame],
                                 filing_data: Optional[Dict[str, pd.DataFrame]] = None) -> Dict[str, pd.Series]:
        """
        Generate sentiment signals from news and filing data

        Args:
            news_data: Dictionary of news DataFrames keyed by ticker
            filing_data: Optional dictionary of filing DataFrames

        Returns:
            Dictionary of sentiment signals
        """

        signals = {}

        # Process news sentiment
        if news_data:
            news_signals = self._process_news_sentiment(news_data)
            signals.update(news_signals)

        # Process filing sentiment
        if filing_data:
            filing_signals = self._process_filing_sentiment(filing_data)
            signals.update(filing_signals)

        # Generate composite sentiment signals
        if news_data:
            composite_signals = self._generate_composite_signals(signals, list(news_data.keys()))
            signals.update(composite_signals)

        return signals

    def _process_news_sentiment(self, news_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.Series]:
        """Process news articles for sentiment signals"""

        signals = {}

        for ticker, news_df in news_data.items():
            if news_df.empty:
                continue

            try:
                # Calculate sentiment scores for each article
                sentiment_scores = []

                for idx, row in news_df.iterrows():
                    title = str(row.get('title', '')).lower()
                    content = str(row.get('content', '')).lower()

                    # Combine title and content
                    text = title + ' ' + content

                    # Calculate sentiment score
                    score = self._calculate_text_sentiment(text)
                    sentiment_scores.append(score)

                # Create time series of sentiment
                if sentiment_scores:
                    sentiment_series = pd.Series(sentiment_scores, index=news_df.index, name=f'{ticker}_news_sentiment')

                    # Resample to daily frequency (take mean sentiment per day)
                    daily_sentiment = sentiment_series.resample('D').mean().fillna(0)

                    signals[f'{ticker}_news_sentiment'] = daily_sentiment

                    # Generate binary signals based on sentiment thresholds
                    signals[f'{ticker}_news_positive'] = (daily_sentiment > 0.5).astype(int)
                    signals[f'{ticker}_news_negative'] = (daily_sentiment < -0.5).astype(int)

                    # Momentum in sentiment
                    signals[f'{ticker}_news_momentum'] = daily_sentiment.diff(3).fillna(0)

                    logger.info(f"Processed {len(sentiment_scores)} news items for {ticker}")

            except Exception as e:
                logger.error(f"Failed to process news sentiment for {ticker}: {e}")

        return signals

    def _process_filing_sentiment(self, filing_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.Series]:
        """Process SEC filings for sentiment signals"""

        signals = {}

        for ticker, filing_df in filing_data.items():
            if filing_df.empty:
                continue

            try:
                sentiment_scores = []

                for idx, row in filing_df.iterrows():
                    filing_type = str(row.get('type', '')).upper()
                    content = str(row.get('content', '')).lower()

                    # Different processing for different filing types
                    if filing_type in ['10-K', '10-Q']:
                        score = self._calculate_filing_sentiment(content, filing_type)
                    else:
                        score = self._calculate_text_sentiment(content)

                    sentiment_scores.append(score)

                if sentiment_scores:
                    sentiment_series = pd.Series(sentiment_scores, index=filing_df.index,
                                               name=f'{ticker}_filing_sentiment')

                    # Filings are less frequent, so forward fill
                    daily_sentiment = sentiment_series.resample('D').ffill().fillna(0)

                    signals[f'{ticker}_filing_sentiment'] = daily_sentiment
                    signals[f'{ticker}_filing_positive'] = (daily_sentiment > 0.3).astype(int)
                    signals[f'{ticker}_filing_negative'] = (daily_sentiment < -0.3).astype(int)

                    logger.info(f"Processed {len(sentiment_scores)} filings for {ticker}")

            except Exception as e:
                logger.error(f"Failed to process filing sentiment for {ticker}: {e}")

        return signals

    def _calculate_text_sentiment(self, text: str) -> float:
        """Calculate sentiment score for text using keyword matching"""

        if not text or len(text.strip()) < 10:
            return 0.0

        words = re.findall(r'\b\w+\b', text.lower())
        total_score = 0.0
        word_count = 0

        for word in words:
            if word in self.positive_words:
                total_score += self.positive_words[word]
                word_count += 1
            elif word in self.negative_words:
                total_score += self.negative_words[word]
                word_count += 1

        # Normalize by word count (avoid division by zero)
        if word_count > 0:
            normalized_score = total_score / np.sqrt(word_count)  # Diminishing returns
            return np.clip(normalized_score, -2.0, 2.0)  # Clip extreme values
        else:
            return 0.0

    def _calculate_filing_sentiment(self, content: str, filing_type: str) -> float:
        """Calculate sentiment for SEC filings"""

        base_sentiment = self._calculate_text_sentiment(content)

        # Filing-type specific adjustments
        type_multiplier = 1.0
        if filing_type == '10-K':
            type_multiplier = 1.2  # Annual reports are more important
        elif filing_type == '10-Q':
            type_multiplier = 1.0  # Quarterly reports
        elif filing_type == '8-K':
            type_multiplier = 1.5  # Current reports are very important

        # Look for specific filing sections
        if 'risk factors' in content.lower():
            base_sentiment -= 0.3  # Risk factors often highlight negatives

        if 'management discussion' in content.lower():
            # MD&A section often contains forward-looking statements
            mda_score = self._extract_mda_sentiment(content)
            base_sentiment = 0.7 * base_sentiment + 0.3 * mda_score

        return base_sentiment * type_multiplier

    def _extract_mda_sentiment(self, content: str) -> float:
        """Extract sentiment from Management Discussion & Analysis section"""

        # Simple keyword-based extraction for MDA
        mda_keywords = {
            'growth': 0.8, 'expansion': 0.6, 'investment': 0.5, 'opportunities': 0.7,
            'challenges': -0.5, 'uncertainty': -0.6, 'risks': -0.4, 'concerns': -0.5
        }

        text = content.lower()
        score = 0.0
        matches = 0

        for keyword, weight in mda_keywords.items():
            count = text.count(keyword)
            score += count * weight
            matches += count

        return score / max(matches, 1)

    def _generate_composite_signals(self, signals: Dict[str, pd.Series],
                                  tickers: List[str]) -> Dict[str, pd.Series]:
        """Generate composite sentiment signals across tickers"""

        composite_signals = {}

        # Market-wide sentiment (average across all tickers)
        sentiment_series = []
        for ticker in tickers:
            news_key = f'{ticker}_news_sentiment'
            if news_key in signals:
                sentiment_series.append(signals[news_key])

        if sentiment_series:
            # Align all series to common index
            aligned_sentiment = pd.concat(sentiment_series, axis=1).fillna(0)
            market_sentiment = aligned_sentiment.mean(axis=1)

            composite_signals['market_news_sentiment'] = market_sentiment
            composite_signals['market_sentiment_positive'] = (market_sentiment > 0.3).astype(int)
            composite_signals['market_sentiment_negative'] = (market_sentiment < -0.3).astype(int)

            # Sentiment divergence (individual ticker vs market)
            for ticker in tickers:
                news_key = f'{ticker}_news_sentiment'
                if news_key in signals:
                    divergence = signals[news_key] - market_sentiment
                    composite_signals[f'{ticker}_sentiment_divergence'] = divergence
                    composite_signals[f'{ticker}_sentiment_outperform'] = (divergence > 0.5).astype(int)

        return composite_signals

    def create_sentiment_features(self, sentiment_signals: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """Create additional sentiment-based features"""

        features = {}

        for signal_name, signal_series in sentiment_signals.items():
            if signal_series.empty:
                continue

            # Moving averages of sentiment
            features[f'{signal_name}_ma5'] = signal_series.rolling(5).mean()
            features[f'{signal_name}_ma20'] = signal_series.rolling(20).mean()

            # Sentiment volatility
            features[f'{signal_name}_volatility'] = signal_series.rolling(20).std()

            # Sentiment momentum
            features[f'{signal_name}_momentum'] = signal_series.diff(5)

            # Extreme sentiment periods
            if 'sentiment' in signal_name.lower():
                features[f'{signal_name}_extreme_positive'] = (signal_series > 1.0).astype(int)
                features[f'{signal_name}_extreme_negative'] = (signal_series < -1.0).astype(int)

        return features


class SentimentDataLoader:
    """Load sentiment data from various sources"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.data_path = Path(config.get('data_path', 'data'))

    def load_news_data(self, tickers: List[str], start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        """Load news data for tickers"""

        news_data = {}

        for ticker in tickers:
            # Mock news data for Phase 1
            # In production, this would connect to news APIs
            news_data[ticker] = self._generate_mock_news_data(ticker, start_date, end_date)

        return news_data

    def load_filing_data(self, tickers: List[str], start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        """Load SEC filing data for tickers"""

        filing_data = {}

        for ticker in tickers:
            # Mock filing data for Phase 1
            filing_data[ticker] = self._generate_mock_filing_data(ticker, start_date, end_date)

        return filing_data

    def _generate_mock_news_data(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Generate mock news data for testing"""

        dates = pd.date_range(start_date, end_date, freq='D')
        np.random.seed(hash(ticker) % 10000)  # Reproducible randomness

        news_items = []

        # Generate 0-3 news items per day
        for date in dates:
            num_items = np.random.poisson(0.5)  # Average 0.5 news items per day

            for i in range(num_items):
                # Random sentiment
                sentiment_type = np.random.choice(['positive', 'negative', 'neutral'],
                                                p=[0.3, 0.3, 0.4])

                if sentiment_type == 'positive':
                    title_templates = [
                        f"{ticker} Shows Strong Growth in Latest Quarter",
                        f"Analysts Upgrade {ticker} Following Earnings Beat",
                        f"{ticker} Announces New Partnership Deal",
                        f"Positive Market Reaction to {ticker}'s New Product"
                    ]
                elif sentiment_type == 'negative':
                    title_templates = [
                        f"{ticker} Faces Challenges in Current Market",
                        f"Analysts Downgrade {ticker} on Weak Guidance",
                        f"{ticker} Announces Cost Cutting Measures",
                        f"Concerns Rise Over {ticker}'s Competition"
                    ]
                else:
                    title_templates = [
                        f"{ticker} Reports Quarterly Results",
                        f"{ticker} CEO Comments on Market Conditions",
                        f"Industry Analysis Includes {ticker}",
                        f"{ticker} Stock Trading Update"
                    ]

                title = np.random.choice(title_templates)
                content = f"This is a mock news article about {ticker}. {title}"

                news_items.append({
                    'date': date,
                    'title': title,
                    'content': content,
                    'source': 'Mock News',
                    'sentiment_type': sentiment_type
                })

        return pd.DataFrame(news_items).set_index('date')

    def _generate_mock_filing_data(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Generate mock SEC filing data for testing"""

        dates = pd.date_range(start_date, end_date, freq='Q')  # Quarterly filings
        np.random.seed(hash(ticker + '_filings') % 10000)

        filings = []

        for date in dates:
            # 10-Q quarterly or 10-K annual
            filing_type = '10-K' if date.month == 12 else '10-Q'

            # Random content with some sentiment
            sentiment_words = []
            if np.random.random() > 0.5:
                sentiment_words.extend(['growth', 'expansion', 'positive', 'strong'])
            else:
                sentiment_words.extend(['challenges', 'uncertainty', 'risks'])

            content = f"This is a mock {filing_type} filing for {ticker}. The company reports {' '.join(sentiment_words)} in the current period."

            filings.append({
                'date': date,
                'type': filing_type,
                'content': content,
                'company': ticker
            })

        return pd.DataFrame(filings).set_index('date')


# Test functions
def test_sentiment_signals():
    """Test sentiment signal generation"""

    print("🧪 Testing Sentiment Signal Generation")

    config = {'data_path': 'data'}
    generator = SentimentSignalGenerator(config)

    # Create mock data
    loader = SentimentDataLoader(config)
    news_data = loader.load_news_data(['AAPL', 'GOOGL'], '2023-01-01', '2023-12-31')
    filing_data = loader.load_filing_data(['AAPL', 'GOOGL'], '2023-01-01', '2023-12-31')

    # Generate signals
    signals = generator.generate_sentiment_signals(news_data, filing_data)

    print(f"✅ Generated {len(signals)} sentiment signals")

    for signal_name, signal_series in list(signals.items())[:5]:  # Show first 5
        print(f"   • {signal_name}: {len(signal_series)} data points")

    # Test features
    features = generator.create_sentiment_features(signals)
    print(f"✅ Generated {len(features)} sentiment features")

    return signals, features


if __name__ == "__main__":
    test_sentiment_signals()


