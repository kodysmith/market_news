#!/usr/bin/env python3
"""
Options Data Ingestion Flow

Downloads options chains and greeks from data providers and stores in Parquet format.
Supports multiple data sources: Tradier, Polygon, and others.
"""

import pandas as pd
import requests
import json
import pyarrow as pa
import pyarrow.parquet as pq
import os
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
import awswrangler as wr
from prefect import flow, task
from prefect.tasks import task_input_hash
import time

@task(cache_key_fn=task_input_hash, cache_expiration=timedelta(hours=1))
def download_tradier_options_chain(
    symbol: str,
    date: str,
    api_key: str
) -> pd.DataFrame:
    """
    Download options chain from Tradier API

    Args:
        symbol: Stock ticker symbol
        date: Expiration date in YYYY-MM-DD format
        api_key: Tradier API key

    Returns:
        DataFrame with options chain data
    """
    print(f"📊 Downloading {symbol} options chain for {date} from Tradier")

    base_url = "https://api.tradier.com/v1/markets/options/chains"
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Accept': 'application/json'
    }

    params = {
        'symbol': symbol,
        'expiration': date,
        'greeks': 'true'
    }

    response = requests.get(base_url, headers=headers, params=params)
    response.raise_for_status()

    data = response.json()

    if 'options' not in data or 'option' not in data['options']:
        print(f"⚠️ No options data available for {symbol} on {date}")
        return pd.DataFrame()

    options = data['options']['option']
    df = pd.DataFrame(options)

    # Clean up column names and data types
    df['symbol'] = symbol
    df['expiration_date'] = pd.to_datetime(df['expiration_date'])
    df['last_trade_date'] = pd.to_datetime(df['last_trade_date'])

    # Convert numeric columns
    numeric_cols = ['strike', 'last', 'change', 'bid', 'ask', 'volume', 'open_interest',
                   'implied_volatility', 'delta', 'gamma', 'theta', 'vega', 'rho']

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Add metadata
    df['data_provider'] = 'tradier'
    df['download_timestamp'] = datetime.now()

    print(f"✅ Downloaded {len(df)} options for {symbol} expiring {date}")
    return df

@task(cache_key_fn=task_input_hash, cache_expiration=timedelta(hours=1))
def download_polygon_options_chain(
    symbol: str,
    date: str,
    api_key: str
) -> pd.DataFrame:
    """
    Download options chain from Polygon API

    Args:
        symbol: Stock ticker symbol
        date: Expiration date in YYYY-MM-DD format
        api_key: Polygon API key

    Returns:
        DataFrame with options chain data
    """
    print(f"📊 Downloading {symbol} options chain for {date} from Polygon")

    # Format symbol for Polygon (O:SPY250315C00450000)
    formatted_symbol = symbol.upper()

    base_url = f"https://api.polygon.io/v3/snapshot/options/{formatted_symbol}"
    headers = {
        'Authorization': f'Bearer {api_key}'
    }

    response = requests.get(base_url, headers=headers)
    response.raise_for_status()

    data = response.json()

    if 'results' not in data:
        print(f"⚠️ No options data available for {symbol} on {date}")
        return pd.DataFrame()

    results = data['results']
    options_data = []

    for option in results:
        # Filter by expiration date
        if option.get('details', {}).get('expiration_date') == date:
            option_data = {
                'symbol': symbol,
                'option_symbol': option.get('ticker'),
                'strike': option.get('details', {}).get('strike_price'),
                'expiration_date': pd.to_datetime(option.get('details', {}).get('expiration_date')),
                'option_type': option.get('details', {}).get('contract_type'),
                'bid': option.get('last_quote', {}).get('bid'),
                'ask': option.get('last_quote', {}).get('ask'),
                'last': option.get('last_trade', {}).get('price'),
                'volume': option.get('last_trade', {}).get('size'),
                'open_interest': option.get('open_interest'),
                'implied_volatility': option.get('implied_volatility'),
                'delta': option.get('greeks', {}).get('delta'),
                'gamma': option.get('greeks', {}).get('gamma'),
                'theta': option.get('greeks', {}).get('theta'),
                'vega': option.get('greeks', {}).get('vega'),
                'rho': option.get('greeks', {}).get('rho'),
                'data_provider': 'polygon'
            }
            options_data.append(option_data)

    df = pd.DataFrame(options_data)

    if not df.empty:
        df['download_timestamp'] = datetime.now()
        print(f"✅ Downloaded {len(df)} options for {symbol} expiring {date}")

    return df

@task
def validate_options_data(data: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Validate and clean options data"""
    if data.empty:
        return data

    required_columns = ['symbol', 'strike', 'expiration_date', 'option_type']

    for col in required_columns:
        if col not in data.columns:
            raise ValueError(f"Missing required column: {col}")

    # Remove rows with missing essential data
    data = data.dropna(subset=required_columns)

    # Ensure proper data types
    data['strike'] = pd.to_numeric(data['strike'], errors='coerce')
    data['expiration_date'] = pd.to_datetime(data['expiration_date'])

    # Clean option types
    data['option_type'] = data['option_type'].str.lower().map({
        'call': 'call', 'c': 'call', 'put': 'put', 'p': 'put'
    })

    # Remove invalid strikes
    data = data[data['strike'] > 0]

    # Remove expired options
    current_date = datetime.now()
    data = data[data['expiration_date'] > current_date]

    print(f"✅ Validated {len(data)} options contracts for {symbol}")
    return data

@task
def calculate_additional_greeks(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate additional option greeks if not provided by data source"""
    if data.empty:
        return data

    # Calculate time to expiration in years
    current_date = datetime.now()
    data['dte'] = (data['expiration_date'] - current_date).dt.days
    data['time_to_expiry'] = data['dte'] / 365.0

    # Add basic moneyness metrics
    if 'last' in data.columns and 'strike' in data.columns:
        data['moneyness'] = data['last'] / data['strike']

    # Add basic option categories
    data['moneyness_category'] = pd.cut(
        data['moneyness'],
        bins=[0, 0.8, 0.95, 1.05, 1.2, float('inf')],
        labels=['deep_otm', 'otm', 'atm', 'itm', 'deep_itm']
    )

    print(f"✅ Calculated additional metrics for {len(data)} options")
    return data

@task
def store_options_data_parquet(
    data: pd.DataFrame,
    base_path: str,
    symbol: str,
    expiration_date: str,
    data_provider: str
) -> str:
    """
    Store options data in Parquet format with partitioning

    Args:
        data: DataFrame with options data
        base_path: Base directory path for storage
        symbol: Stock ticker symbol
        expiration_date: Expiration date
        data_provider: Data source provider

    Returns:
        Path where data was stored
    """
    if data.empty:
        print(f"⚠️ No data to store for {symbol}")
        return ""

    # Create directory structure: base_path/options/symbol/provider/expiration/
    output_dir = os.path.join(base_path, 'options', symbol.lower(), data_provider, expiration_date)
    os.makedirs(output_dir, exist_ok=True)

    # Convert to PyArrow table
    table = pa.Table.from_pandas(data)

    # Write to Parquet with symbol partitioning
    pq.write_to_dataset(
        table,
        root_path=output_dir,
        partition_cols=['symbol'],
        compression='snappy',
        coerce_timestamps='ms',
        allow_truncated_timestamps=True
    )

    print(f"💾 Stored options data in {output_dir}")
    return output_dir

@task
def store_options_data_s3(
    data: pd.DataFrame,
    s3_path: str,
    symbol: str,
    expiration_date: str,
    data_provider: str
) -> str:
    """
    Store options data in S3 using AWS Wrangler

    Args:
        data: DataFrame with options data
        s3_path: S3 path (s3://bucket/path/)
        symbol: Stock ticker symbol
        expiration_date: Expiration date
        data_provider: Data source provider

    Returns:
        S3 path where data was stored
    """
    if data.empty:
        print(f"⚠️ No data to store for {symbol}")
        return ""

    s3_output_path = f"{s3_path}/options/{symbol.lower()}/{data_provider}/{expiration_date}/"

    wr.s3.to_parquet(
        df=data,
        path=s3_output_path,
        dataset=True,
        partition_cols=['symbol'],
        compression='snappy',
        mode='append'
    )

    print(f"💾 Stored options data in S3: {s3_output_path}")
    return s3_output_path

@task
def get_expiration_dates(symbol: str, api_key: str, provider: str = 'tradier') -> List[str]:
    """
    Get available expiration dates for options

    Args:
        symbol: Stock ticker symbol
        api_key: API key for data provider
        provider: Data provider ('tradier' or 'polygon')

    Returns:
        List of expiration dates in YYYY-MM-DD format
    """
    if provider == 'tradier':
        base_url = "https://api.tradier.com/v1/markets/options/expirations"
        headers = {
            'Authorization': f'Bearer {api_key}',
            'Accept': 'application/json'
        }

        params = {'symbol': symbol}
        response = requests.get(base_url, headers=headers, params=params)
        response.raise_for_status()

        data = response.json()
        if 'expirations' in data and 'date' in data['expirations']:
            dates = data['expirations']['date']
            # Filter for next 6 months
            current_date = datetime.now()
            cutoff_date = current_date + timedelta(days=180)

            valid_dates = []
            for date_str in dates:
                exp_date = datetime.strptime(date_str, '%Y-%m-%d')
                if current_date <= exp_date <= cutoff_date:
                    valid_dates.append(date_str)

            return valid_dates

    elif provider == 'polygon':
        # Polygon doesn't have a direct expirations endpoint
        # We'll use a range of dates
        current_date = datetime.now()
        dates = []
        for i in range(0, 180, 7):  # Weekly expirations for 6 months
            exp_date = current_date + timedelta(days=i)
            if exp_date.weekday() == 4:  # Friday
                dates.append(exp_date.strftime('%Y-%m-%d'))
        return dates

    return []

@flow(name="ingest-options-data")
def ingest_options_flow(
    symbols: List[str],
    data_provider: str = 'tradier',
    api_key: Optional[str] = None,
    storage_path: Optional[str] = None,
    use_s3: bool = False,
    max_expirations: int = 12
):
    """
    Main options data ingestion flow

    Args:
        symbols: List of stock ticker symbols
        data_provider: Data provider ('tradier' or 'polygon')
        api_key: API key for data provider
        storage_path: Local path or S3 path for storage
        use_s3: Whether to use S3 for storage
        max_expirations: Maximum number of expiration dates to fetch per symbol
    """
    print(f"🚀 Starting options data ingestion for {len(symbols)} symbols using {data_provider}")

    if api_key is None:
        raise ValueError("API key is required for options data ingestion")

    if storage_path is None:
        storage_path = "./data"

    download_func = download_tradier_options_chain if data_provider == 'tradier' else download_polygon_options_chain

    for symbol in symbols:
        try:
            # Get available expiration dates
            expiration_dates = get_expiration_dates(symbol, api_key, data_provider)
            expiration_dates = expiration_dates[:max_expirations]  # Limit expirations

            print(f"📅 Found {len(expiration_dates)} expiration dates for {symbol}")

            for exp_date in expiration_dates:
                try:
                    # Download options chain
                    raw_data = download_func(symbol, exp_date, api_key)

                    if raw_data.empty:
                        continue

                    # Validate data
                    clean_data = validate_options_data(raw_data, symbol)

                    # Calculate additional metrics
                    enriched_data = calculate_additional_greeks(clean_data)

                    # Store data
                    if use_s3:
                        store_path = store_options_data_s3(
                            enriched_data, storage_path, symbol, exp_date, data_provider
                        )
                    else:
                        store_path = store_options_data_parquet(
                            enriched_data, storage_path, symbol, exp_date, data_provider
                        )

                    print(f"✅ Successfully processed {symbol} options for {exp_date}: {store_path}")

                    # Rate limiting
                    time.sleep(0.1)

                except Exception as e:
                    print(f"❌ Failed to process {symbol} options for {exp_date}: {e}")
                    continue

        except Exception as e:
            print(f"❌ Failed to process {symbol}: {e}")
            continue

if __name__ == "__main__":
    # Example usage (requires API key)
    symbols = ["SPY", "QQQ", "AAPL"]
    # api_key = os.getenv('TRADIER_API_KEY')  # Set your API key

    # Run the flow
    # ingest_options_flow(symbols, api_key=api_key)

