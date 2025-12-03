"""
Backtest Data Loader

Loads historical data from CSV/Parquet files for backtesting.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import logging
from .interface import DataLoaderInterface

logger = logging.getLogger(__name__)


class BacktestDataLoader(DataLoaderInterface):
    """Loads historical data from files for backtesting"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.data_path = config.get('data_path', 'data/')
        self.data_cache = {}
        self.current_date = None

    def load_market_data(self, date: Optional[datetime] = None) -> Dict[str, Any]:
        """Load market data for a specific date"""
        if date is None:
            date = self.current_date

        if date is None:
            raise ValueError("No date specified and current_date not set")

        market_data = {}

        # Load SPY data
        spy_data = self._load_ticker_data('SPY', date)
        if spy_data is not None:
            market_data['spy'] = {
                'prices': spy_data.get('prices', []),
                'volumes': spy_data.get('volumes', [])
            }

        # Load VIX data (simplified - use historical average if not available)
        market_data['vix'] = {
            'level': spy_data.get('vix', 20) if spy_data else 20,
            'change': 0
        }

        return market_data

    def get_current_prices(self, tickers: list) -> Dict[str, float]:
        """Get current prices for tickers"""
        prices = {}

        for ticker in tickers:
            data = self._load_ticker_data(ticker, self.current_date)
            if data and 'current_price' in data:
                prices[ticker] = data['current_price']
            else:
                # Fallback: use last known price
                prices[ticker] = self._get_last_price(ticker)

        return prices

    def set_current_date(self, date: datetime):
        """Set the current date for point-in-time data access"""
        self.current_date = date

    def _load_ticker_data(self, ticker: str, date: datetime) -> Optional[Dict[str, Any]]:
        """Load historical data for a ticker up to a specific date"""
        try:
            # Try to load from CSV file
            file_path = f"{self.data_path}/{ticker.lower()}_historical.csv"

            if ticker not in self.data_cache:
                try:
                    df = pd.read_csv(file_path, parse_dates=['date'], index_col='date')
                    self.data_cache[ticker] = df
                except FileNotFoundError:
                    logger.warning(f"Data file not found for {ticker}, using synthetic data")
                    # Generate synthetic data for testing
                    self.data_cache[ticker] = self._generate_synthetic_data(ticker, date)

            df = self.data_cache[ticker]

            # Get data up to and including the date
            historical = df[df.index <= date].tail(50)  # Last 50 days

            if len(historical) == 0:
                return None

            return {
                'prices': historical['close'].tolist(),
                'volumes': historical['volume'].tolist() if 'volume' in historical.columns else [],
                'current_price': historical['close'].iloc[-1],
                'vix': 20  # Default VIX
            }

        except Exception as e:
            logger.error(f"Error loading data for {ticker}: {e}")
            return None

    def _get_last_price(self, ticker: str) -> float:
        """Get last known price for a ticker"""
        if ticker in self.data_cache:
            df = self.data_cache[ticker]
            if len(df) > 0:
                return df['close'].iloc[-1]
        return 100.0  # Default fallback

    def _generate_synthetic_data(self, ticker: str, end_date: datetime) -> pd.DataFrame:
        """Generate synthetic historical data for testing"""
        dates = pd.date_range(end_date - timedelta(days=365), end_date, freq='D')
        np.random.seed(hash(ticker) % 1000)

        # Generate random walk prices
        returns = np.random.normal(0.0005, 0.02, len(dates))
        prices = 100 * np.exp(np.cumsum(returns))

        df = pd.DataFrame({
            'close': prices,
            'volume': np.random.randint(1000000, 10000000, len(dates))
        }, index=dates)

        return df
