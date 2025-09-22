"""
Data Ingestion Manager for AI Quant Trading System

Handles:
- Market data acquisition from multiple sources
- Data normalization and caching
- Point-in-time data alignment
- Survivorship bias control
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from pathlib import Path
import logging
import yfinance as yf
import requests
from functools import lru_cache
import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)


class DataManager:
    """Central data management system"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.data_path = Path(config.get('data_path', 'data'))
        self.cache_dir = self.data_path / 'cache'
        self.raw_dir = self.data_path / 'raw'
        self.processed_dir = self.data_path / 'processed'

        # Create directories
        for dir_path in [self.cache_dir, self.raw_dir, self.processed_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Initialize DuckDB for fast queries
        self.db_path = self.cache_dir / 'market_data.db'
        self.conn = duckdb.connect(str(self.db_path))

        # Setup database tables
        self._setup_database()

    def _setup_database(self):
        """Initialize database schema"""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS market_data (
                ticker VARCHAR,
                date DATE,
                open DOUBLE,
                high DOUBLE,
                low DOUBLE,
                close DOUBLE,
                volume BIGINT,
                adj_close DOUBLE,
                source VARCHAR,
                last_updated TIMESTAMP,
                PRIMARY KEY (ticker, date)
            )
        """)

        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS options_data (
                underlying VARCHAR,
                symbol VARCHAR,
                date DATE,
                strike DOUBLE,
                expiry DATE,
                option_type VARCHAR, -- 'call' or 'put'
                bid DOUBLE,
                ask DOUBLE,
                last DOUBLE,
                volume BIGINT,
                open_interest BIGINT,
                iv DOUBLE,
                delta DOUBLE,
                gamma DOUBLE,
                theta DOUBLE,
                vega DOUBLE,
                rho DOUBLE,
                last_updated TIMESTAMP,
                PRIMARY KEY (symbol, date)
            )
        """)

    def get_market_data(self, tickers: List[str], start_date: str, end_date: str,
                       source: str = 'yahoo') -> Dict[str, pd.DataFrame]:
        """
        Get market data for tickers, downloading if necessary

        Args:
            tickers: List of ticker symbols
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            source: Data source ('yahoo', 'alpha_vantage', etc.)

        Returns:
            Dictionary of DataFrames keyed by ticker
        """

        data = {}

        for ticker in tickers:
            # Try cache first
            cached_data = self._get_cached_data(ticker, start_date, end_date)
            if cached_data is not None and not self._is_stale(cached_data, ticker):
                data[ticker] = cached_data
                continue

            # Download fresh data
            try:
                if source == 'yahoo':
                    df = self._download_yahoo_data(ticker, start_date, end_date)
                elif source == 'alpha_vantage':
                    df = self._download_alpha_vantage_data(ticker, start_date, end_date)
                else:
                    raise ValueError(f"Unsupported data source: {source}")

                if df is not None and not df.empty:
                    # Cache the data
                    self._cache_data(ticker, df, source)
                    data[ticker] = df

            except Exception as e:
                logger.error(f"Failed to get data for {ticker}: {e}")
                continue

        return data

    def _get_cached_data(self, ticker: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """Retrieve cached data from database"""
        try:
            query = """
                SELECT date, open, high, low, close, volume, adj_close
                FROM market_data
                WHERE ticker = ? AND date >= ? AND date <= ?
                ORDER BY date
            """

            df = self.conn.execute(query, [ticker, start_date, end_date]).fetchdf()

            if df.empty:
                return None

            # Set date index
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')

            return df

        except Exception as e:
            logger.warning(f"Cache retrieval failed for {ticker}: {e}")
            return None

    def _is_stale(self, data: pd.DataFrame, ticker: str, max_age_hours: int = 24) -> bool:
        """Check if cached data is stale"""
        if data.empty:
            return True

        last_date = data.index.max()
        hours_since_update = (pd.Timestamp.now() - last_date).total_seconds() / 3600

        # Consider data stale if more than max_age_hours old
        return hours_since_update > max_age_hours

    def _download_yahoo_data(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Download data from Yahoo Finance"""
        try:
            # Add .NS suffix for Indian stocks if needed
            yahoo_ticker = ticker

            # Download with yfinance
            data = yf.download(
                yahoo_ticker,
                start=start_date,
                end=end_date,
                progress=False,
                auto_adjust=True,
                prepost=True
            )

            if data.empty:
                logger.warning(f"No data found for {ticker}")
                return pd.DataFrame()

            # Ensure we have all required columns
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in data.columns for col in required_cols):
                logger.warning(f"Missing required columns for {ticker}")
                return pd.DataFrame()

            # Rename columns to standard format
            data = data.rename(columns={
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume',
                'Adj Close': 'adj_close'
            })

            # Add adjusted close if not present
            if 'adj_close' not in data.columns:
                data['adj_close'] = data['close']

            return data

        except Exception as e:
            logger.error(f"Yahoo download failed for {ticker}: {e}")
            return pd.DataFrame()

    def _download_alpha_vantage_data(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Download data from Alpha Vantage (requires API key)"""
        api_key = self.config.get('alpha_vantage_api_key')
        if not api_key:
            raise ValueError("Alpha Vantage API key required")

        try:
            url = f"https://www.alphavantage.co/query"
            params = {
                'function': 'TIME_SERIES_DAILY_ADJUSTED',
                'symbol': ticker,
                'apikey': api_key,
                'outputsize': 'full'
            }

            response = requests.get(url, params=params)
            response.raise_for_status()

            data = response.json()

            if 'Time Series (Daily)' not in data:
                logger.warning(f"No data in Alpha Vantage response for {ticker}")
                return pd.DataFrame()

            # Parse the response
            time_series = data['Time Series (Daily)']
            rows = []

            for date_str, values in time_series.items():
                if date_str < start_date or date_str > end_date:
                    continue

                row = {
                    'date': pd.to_datetime(date_str),
                    'open': float(values['1. open']),
                    'high': float(values['2. high']),
                    'low': float(values['3. low']),
                    'close': float(values['4. close']),
                    'adj_close': float(values['5. adjusted close']),
                    'volume': int(values['6. volume'])
                }
                rows.append(row)

            df = pd.DataFrame(rows)
            df = df.set_index('date').sort_index()

            return df

        except Exception as e:
            logger.error(f"Alpha Vantage download failed for {ticker}: {e}")
            return pd.DataFrame()

    def _cache_data(self, ticker: str, data: pd.DataFrame, source: str):
        """Cache data in database"""
        try:
            # Prepare data for insertion
            cache_df = data.reset_index()
            cache_df['ticker'] = ticker
            cache_df['source'] = source
            cache_df['last_updated'] = pd.Timestamp.now()

            # Rename columns to match schema
            cache_df = cache_df.rename(columns={
                'date': 'date',
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'volume': 'volume',
                'adj_close': 'adj_close'
            })

            # Insert to database (upsert)
            self.conn.execute("""
                INSERT OR REPLACE INTO market_data
                (ticker, date, open, high, low, close, volume, adj_close, source, last_updated)
                SELECT ticker, date, open, high, low, close, volume, adj_close, source, last_updated
                FROM cache_df
            """)

            logger.debug(f"Cached {len(cache_df)} rows for {ticker}")

        except Exception as e:
            logger.error(f"Failed to cache data for {ticker}: {e}")

    def get_options_data(self, underlying: str, date: str,
                        strikes: Optional[List[float]] = None) -> pd.DataFrame:
        """
        Get options chain data for underlying

        Note: This is a simplified implementation. Real options data
        would require a proper options data provider.
        """
        logger.warning("Options data not fully implemented - returning empty DataFrame")
        return pd.DataFrame()

    def save_to_parquet(self, data: Dict[str, pd.DataFrame], filename: str):
        """Save data dictionary to parquet format"""
        output_path = self.processed_dir / filename

        # Convert to pyarrow table
        tables = {}
        for ticker, df in data.items():
            df_copy = df.copy()
            df_copy['ticker'] = ticker
            tables[ticker] = pa.Table.from_pandas(df_copy)

        # Save as partitioned dataset
        if tables:
            pq.write_to_dataset(
                list(tables.values())[0],
                root_path=str(output_path),
                partition_cols=['ticker']
            )

    def load_from_parquet(self, filename: str) -> Dict[str, pd.DataFrame]:
        """Load data dictionary from parquet format"""
        input_path = self.processed_dir / filename

        if not input_path.exists():
            return {}

        # Load partitioned dataset
        dataset = pq.ParquetDataset(str(input_path))
        data = {}

        for fragment in dataset.fragments:
            table = fragment.to_table()
            df = table.to_pandas()
            tickers = df['ticker'].unique()

            for ticker in tickers:
                ticker_df = df[df['ticker'] == ticker].drop('ticker', axis=1)
                ticker_df = ticker_df.set_index('date').sort_index()
                data[ticker] = ticker_df

        return data

    def get_universe_data(self, universe_name: str = 'default') -> Dict[str, pd.DataFrame]:
        """Get data for a predefined universe"""
        universes = {
            'default': ['SPY', 'QQQ', 'IWM', 'VTI'],
            'leveraged': ['TQQQ', 'UVXY', 'SOXL', 'SOXS'],
            'tech': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA'],
            'sectors': ['XLE', 'XLF', 'XLK', 'XLV', 'XLY', 'XLB', 'XLI', 'XLRE', 'XLU']
        }

        tickers = universes.get(universe_name, universes['default'])

        # Get date range from config
        start_date = self.config.get('start_date', '2010-01-01')
        end_date = self.config.get('end_date', pd.Timestamp.now().strftime('%Y-%m-%d'))

        return self.get_market_data(tickers, start_date, end_date)

    def cleanup_old_cache(self, days_to_keep: int = 30):
        """Clean up old cached data"""
        cutoff_date = pd.Timestamp.now() - pd.Timedelta(days=days_to_keep)

        try:
            deleted = self.conn.execute("""
                DELETE FROM market_data
                WHERE last_updated < ?
            """, [cutoff_date]).fetchone()[0]

            logger.info(f"Cleaned up {deleted} old cache entries")

        except Exception as e:
            logger.error(f"Cache cleanup failed: {e}")

    def __del__(self):
        """Cleanup database connection"""
        if hasattr(self, 'conn'):
            self.conn.close()
