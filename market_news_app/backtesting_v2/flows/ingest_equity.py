#!/usr/bin/env python3
"""
Equity Data Ingestion Flow

Downloads equity price data from Yahoo Finance and stores in Parquet format
with date partitioning for efficient querying.
"""

import pandas as pd
import yfinance as yf
import pyarrow as pa
import pyarrow.parquet as pq
import os
from datetime import datetime, timedelta
from typing import List, Optional
import awswrangler as wr
from prefect import flow, task
from prefect.tasks import task_input_hash
from prefect.blocks.system import Secret

@task(cache_key_fn=task_input_hash, cache_expiration=timedelta(hours=6))
def download_equity_data(
    symbol: str, 
    start_date: str, 
    end_date: str, 
    interval: str = "1d"
) -> pd.DataFrame:
    """
    Download equity price data from Yahoo Finance
    
    Args:
        symbol: Stock ticker symbol
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        interval: Data interval (1d, 1h, 1m)
        
    Returns:
        DataFrame with OHLCV data
    """
    print(f"📊 Downloading {symbol} data from {start_date} to {end_date} ({interval})")
    
    ticker = yf.Ticker(symbol)
    data = ticker.history(start=start_date, end=end_date, interval=interval)
    
    if data.empty:
        raise ValueError(f"No data found for {symbol} in the specified date range")
    
    # Reset index and add symbol column
    data = data.reset_index()
    data['symbol'] = symbol
    data['interval'] = interval
    
    # Ensure proper datetime format
    if 'Date' in data.columns:
        data['date'] = pd.to_datetime(data['Date'])
        data = data.drop('Date', axis=1)
    elif 'Datetime' in data.columns:
        data['date'] = pd.to_datetime(data['Datetime'])
        data = data.drop('Datetime', axis=1)

    # Add year column for partitioning
    data['year'] = data['date'].dt.year
    
    print(f"✅ Downloaded {len(data)} rows of {symbol} data")
    return data

@task
def validate_equity_data(data: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Validate and clean equity data"""
    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    
    for col in required_columns:
        if col not in data.columns:
            raise ValueError(f"Missing required column: {col}")
    
    # Remove rows with missing essential data
    data = data.dropna(subset=required_columns)
    
    # Ensure numeric types
    for col in required_columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    
    # Remove any remaining NaN values
    data = data.dropna(subset=required_columns)
    
    print(f"✅ Validated {len(data)} rows of {symbol} data")
    return data

@task
def store_equity_data_parquet(
    data: pd.DataFrame, 
    base_path: str, 
    symbol: str,
    interval: str
) -> str:
    """
    Store equity data in Parquet format with partitioning
    
    Args:
        data: DataFrame with equity data
        base_path: Base directory path for storage
        symbol: Stock ticker symbol
        interval: Data interval
        
    Returns:
        Path where data was stored
    """
    # Create directory structure: base_path/symbol/interval/
    output_dir = os.path.join(base_path, symbol.lower(), interval)
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert to PyArrow table
    table = pa.Table.from_pandas(data)
    
    # Write to Parquet - partition by year for daily data to avoid too many partitions
    if interval == '1d':
        # For daily data, partition by year to keep partitions manageable
        pq.write_to_dataset(
            table,
            root_path=output_dir,
            partition_cols=['year'],
            compression='snappy',
            coerce_timestamps='ms',
            allow_truncated_timestamps=True
        )
    else:
        # For intraday data, no partitioning
        pq.write_table(
            table,
            f"{output_dir}/data.parquet",
            compression='snappy',
            coerce_timestamps='ms',
            allow_truncated_timestamps=True
        )
    
    print(f"💾 Stored {symbol} data in {output_dir}")
    return output_dir

@task
def store_equity_data_s3(
    data: pd.DataFrame, 
    s3_path: str, 
    symbol: str,
    interval: str
) -> str:
    """
    Store equity data in S3 using AWS Wrangler
    
    Args:
        data: DataFrame with equity data
        s3_path: S3 path (s3://bucket/path/)
        symbol: Stock ticker symbol
        interval: Data interval
        
    Returns:
        S3 path where data was stored
    """
    s3_output_path = f"{s3_path}/{symbol.lower()}/{interval}/"
    
    wr.s3.to_parquet(
        df=data,
        path=s3_output_path,
        dataset=True,
        partition_cols=['date'],
        compression='snappy',
        mode='append'
    )
    
    print(f"💾 Stored {symbol} data in S3: {s3_output_path}")
    return s3_output_path

@flow(name="ingest-equity-data")
def ingest_equity_flow(
    symbols: List[str],
    start_date: str,
    end_date: str,
    interval: str = "1d",
    storage_path: Optional[str] = None,
    use_s3: bool = False
):
    """
    Main equity data ingestion flow
    
    Args:
        symbols: List of stock ticker symbols
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        interval: Data interval (1d, 1h, 1m)
        storage_path: Local path or S3 path for storage
        use_s3: Whether to use S3 for storage
    """
    print(f"🚀 Starting equity data ingestion for {len(symbols)} symbols")
    
    if storage_path is None:
        storage_path = "./data/equity"
    
    for symbol in symbols:
        try:
            # Download data
            raw_data = download_equity_data(symbol, start_date, end_date, interval)
            
            # Validate data
            clean_data = validate_equity_data(raw_data, symbol)
            
            # Store data
            if use_s3:
                store_path = store_equity_data_s3(clean_data, storage_path, symbol, interval)
            else:
                store_path = store_equity_data_parquet(clean_data, storage_path, symbol, interval)
            
            print(f"✅ Successfully processed {symbol}: {store_path}")
            
        except Exception as e:
            print(f"❌ Failed to process {symbol}: {e}")
            continue

if __name__ == "__main__":
    # Example usage
    symbols = ["SPY", "QQQ", "DIA", "IWM"]
    start_date = "2020-01-01"
    end_date = "2024-01-01"
    
    # Run the flow
    ingest_equity_flow(symbols, start_date, end_date)
