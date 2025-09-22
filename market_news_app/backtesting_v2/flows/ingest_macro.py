#!/usr/bin/env python3
"""
Macro Data Ingestion Flow

Downloads macroeconomic data from FRED API, CFTC COT reports, and other sources.
Stores data in Parquet format for regime detection and macro factor modeling.
"""

import pandas as pd
import requests
import pyarrow as pa
import pyarrow.parquet as pq
import os
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
import awswrangler as wr
from prefect import flow, task
from prefect.tasks import task_input_hash
import time

@task(cache_key_fn=task_input_hash, cache_expiration=timedelta(days=1))
def download_fred_data(
    series_ids: List[str],
    api_key: str,
    start_date: str = "2000-01-01",
    end_date: Optional[str] = None
) -> pd.DataFrame:
    """
    Download macroeconomic data from FRED API

    Args:
        series_ids: List of FRED series IDs
        api_key: FRED API key
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format (defaults to today)

    Returns:
        DataFrame with macroeconomic data
    """
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')

    print(f"📊 Downloading {len(series_ids)} FRED series from {start_date} to {end_date}")

    base_url = "https://api.stlouisfed.org/fred/series/observations"
    all_data = []

    for series_id in series_ids:
        params = {
            'series_id': series_id,
            'api_key': api_key,
            'file_type': 'json',
            'observation_start': start_date,
            'observation_end': end_date
        }

        response = requests.get(base_url, params=params)
        response.raise_for_status()

        data = response.json()

        if 'observations' not in data:
            print(f"⚠️ No data available for series {series_id}")
            continue

        observations = data['observations']

        for obs in observations:
            if obs['value'] != '.' and obs['value'] != '':
                record = {
                    'series_id': series_id,
                    'date': pd.to_datetime(obs['date']),
                    'value': float(obs['value']),
                    'data_provider': 'fred'
                }
                all_data.append(record)

        print(f"✅ Downloaded {len(observations)} observations for {series_id}")
        time.sleep(0.3)  # Rate limiting

    df = pd.DataFrame(all_data)

    if not df.empty:
        df['download_timestamp'] = datetime.now()

    print(f"✅ Downloaded {len(df)} total macroeconomic observations")
    return df

@task(cache_key_fn=task_input_hash, cache_expiration=timedelta(days=1))
def download_cot_data(
    years: Optional[List[int]] = None
) -> pd.DataFrame:
    """
    Download CFTC Commitment of Traders (COT) reports

    Args:
        years: List of years to download (defaults to last 5 years)

    Returns:
        DataFrame with COT data
    """
    if years is None:
        current_year = datetime.now().year
        years = list(range(current_year - 5, current_year + 1))

    print(f"📊 Downloading CFTC COT data for years {years}")

    base_url = "https://www.cftc.gov/files/dea/history/deacot{year}.txt"
    all_data = []

    for year in years:
        try:
            url = base_url.format(year=year)
            response = requests.get(url)

            if response.status_code != 200:
                print(f"⚠️ COT data not available for {year}")
                continue

            # Parse the fixed-width format
            lines = response.text.split('\n')

            # Skip header lines and find data start
            data_started = False
            headers = []

            for line in lines:
                line = line.strip()
                if not line:
                    continue

                if line.startswith('Market and Exchange Names:'):
                    data_started = False
                    continue

                if not data_started:
                    if 'As of' in line or 'Report' in line:
                        continue
                    # This is likely a header line
                    if len(line.split()) > 5:  # Heuristic for header
                        headers = line.split()
                        data_started = True
                        continue
                else:
                    # Parse data line
                    if len(line) > 100:  # Minimum length for valid data
                        # Parse fixed-width fields (this is approximate)
                        try:
                            # Extract key fields - simplified parsing
                            parts = line.split()
                            if len(parts) >= 10:
                                record = {
                                    'report_date': pd.to_datetime(f"{year}-{parts[0]}-{parts[1]}") if len(parts) >= 2 else None,
                                    'market_code': parts[2] if len(parts) > 2 else '',
                                    'market_name': ' '.join(parts[3:-6]) if len(parts) > 9 else '',
                                    'producer_long': int(parts[-6]) if len(parts) > 6 else 0,
                                    'producer_short': int(parts[-5]) if len(parts) > 5 else 0,
                                    'swap_long': int(parts[-4]) if len(parts) > 4 else 0,
                                    'swap_short': int(parts[-3]) if len(parts) > 3 else 0,
                                    'managed_long': int(parts[-2]) if len(parts) > 2 else 0,
                                    'managed_short': int(parts[-1]) if len(parts) > 1 else 0,
                                    'data_provider': 'cftc_cot'
                                }
                                all_data.append(record)
                        except (ValueError, IndexError):
                            continue

            print(f"✅ Downloaded COT data for {year}")

        except Exception as e:
            print(f"❌ Failed to download COT data for {year}: {e}")
            continue

    df = pd.DataFrame(all_data)

    if not df.empty:
        df['download_timestamp'] = datetime.now()
        # Clean up dates
        df = df.dropna(subset=['report_date'])

    print(f"✅ Downloaded {len(df)} COT records")
    return df

@task(cache_key_fn=task_input_hash, cache_expiration=timedelta(hours=6))
def download_vix_data() -> pd.DataFrame:
    """
    Download VIX data from CBOE

    Returns:
        DataFrame with VIX historical data
    """
    print("📊 Downloading VIX data from CBOE")

    # Use CBOE's VIX historical data URL
    url = "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv"

    try:
        df = pd.read_csv(url)

        # Clean up the data
        df['DATE'] = pd.to_datetime(df['DATE'])
        df = df.rename(columns={
            'DATE': 'date',
            'OPEN': 'open',
            'HIGH': 'high',
            'LOW': 'low',
            'CLOSE': 'close'
        })

        df['symbol'] = 'VIX'
        df['data_provider'] = 'cboe'

        # Calculate additional metrics
        df['returns'] = df['close'].pct_change()
        df['realized_vol_20d'] = df['returns'].rolling(20).std() * (252 ** 0.5) * 100

        df['download_timestamp'] = datetime.now()

        print(f"✅ Downloaded {len(df)} VIX observations")
        return df

    except Exception as e:
        print(f"❌ Failed to download VIX data: {e}")
        return pd.DataFrame()

@task
def validate_macro_data(data: pd.DataFrame) -> pd.DataFrame:
    """Validate and clean macroeconomic data"""
    if data.empty:
        return data

    # Ensure date column exists and is datetime
    if 'date' in data.columns:
        data['date'] = pd.to_datetime(data['date'], errors='coerce')
        data = data.dropna(subset=['date'])

    # Ensure value column is numeric where applicable
    if 'value' in data.columns:
        data['value'] = pd.to_numeric(data['value'], errors='coerce')
        data = data.dropna(subset=['value'])

    # Remove duplicates
    data = data.drop_duplicates(subset=['series_id', 'date'] if 'series_id' in data.columns else ['date'])

    print(f"✅ Validated {len(data)} macroeconomic observations")
    return data

@task
def calculate_macro_features(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate derived macroeconomic features and regime indicators"""
    if data.empty:
        return data

    # Pivot FRED data for easier analysis
    if 'series_id' in data.columns:
        fred_pivot = data.pivot(index='date', columns='series_id', values='value')
        fred_pivot.columns = [f'fred_{col}' for col in fred_pivot.columns]

        # Calculate macroeconomic surprise indices
        key_series = {
            'fred_UNRATE': 'unemployment_rate',
            'fred_CPIAUCSL': 'cpi',
            'fred_GDP': 'gdp',
            'fred_DGS10': 'ten_year_yield',
            'fred_DGS2': 'two_year_yield',
            'fred_PAYEMS': 'payroll_change'
        }

        for fred_col, feature_name in key_series.items():
            if fred_col in fred_pivot.columns:
                # Calculate month-over-month changes
                fred_pivot[f'{feature_name}_mom'] = fred_pivot[fred_col].pct_change()

                # Calculate year-over-year changes
                fred_pivot[f'{feature_name}_yoy'] = fred_pivot[fred_col].pct_change(12)

        # Calculate yield curve slope
        if 'fred_DGS10' in fred_pivot.columns and 'fred_DGS2' in fred_pivot.columns:
            fred_pivot['yield_curve_slope'] = fred_pivot['fred_DGS10'] - fred_pivot['fred_DGS2']

        # Merge back
        data = data.merge(fred_pivot.reset_index(), on='date', how='left')

    # VIX regime calculation
    if 'close' in data.columns and data['symbol'].eq('VIX').any():
        vix_data = data[data['symbol'] == 'VIX'].copy()
        vix_data['vix_percentile_252d'] = vix_data['close'].rolling(252).rank(pct=True)
        vix_data['vix_regime'] = pd.cut(
            vix_data['vix_percentile_252d'],
            bins=[0, 0.2, 0.5, 0.8, 1.0],
            labels=['low_vol', 'normal_vol', 'high_vol', 'extreme_vol']
        )

        # Merge VIX features back
        vix_features = vix_data[['date', 'vix_percentile_252d', 'vix_regime']]
        data = data.merge(vix_features, on='date', how='left')

    print(f"✅ Calculated macro features for {len(data)} observations")
    return data

@task
def store_macro_data_parquet(
    data: pd.DataFrame,
    base_path: str,
    data_type: str
) -> str:
    """
    Store macroeconomic data in Parquet format with partitioning

    Args:
        data: DataFrame with macro data
        base_path: Base directory path for storage
        data_type: Type of macro data ('fred', 'cot', 'vix', etc.)

    Returns:
        Path where data was stored
    """
    if data.empty:
        print(f"⚠️ No {data_type} data to store")
        return ""

    # Create directory structure: base_path/macro/data_type/
    output_dir = os.path.join(base_path, 'macro', data_type)
    os.makedirs(output_dir, exist_ok=True)

    # Convert to PyArrow table
    table = pa.Table.from_pandas(data)

    # Write to Parquet with date partitioning
    pq.write_to_dataset(
        table,
        root_path=output_dir,
        partition_cols=['date'],
        compression='snappy',
        coerce_timestamps='ms',
        allow_truncated_timestamps=True
    )

    print(f"💾 Stored {data_type} macro data in {output_dir}")
    return output_dir

@task
def store_macro_data_s3(
    data: pd.DataFrame,
    s3_path: str,
    data_type: str
) -> str:
    """
    Store macroeconomic data in S3 using AWS Wrangler

    Args:
        data: DataFrame with macro data
        s3_path: S3 path (s3://bucket/path/)
        data_type: Type of macro data ('fred', 'cot', 'vix', etc.)

    Returns:
        S3 path where data was stored
    """
    if data.empty:
        print(f"⚠️ No {data_type} data to store")
        return ""

    s3_output_path = f"{s3_path}/macro/{data_type}/"

    wr.s3.to_parquet(
        df=data,
        path=s3_output_path,
        dataset=True,
        partition_cols=['date'],
        compression='snappy',
        mode='append'
    )

    print(f"💾 Stored {data_type} macro data in S3: {s3_output_path}")
    return s3_output_path

@flow(name="ingest-macro-data")
def ingest_macro_flow(
    fred_api_key: Optional[str] = None,
    storage_path: Optional[str] = None,
    use_s3: bool = False
):
    """
    Main macroeconomic data ingestion flow

    Args:
        fred_api_key: FRED API key
        storage_path: Local path or S3 path for storage
        use_s3: Whether to use S3 for storage
    """
    print("🚀 Starting macroeconomic data ingestion")

    if storage_path is None:
        storage_path = "./data"

    # FRED series to download
    fred_series = [
        'UNRATE',      # Unemployment Rate
        'CPIAUCSL',    # Consumer Price Index
        'GDP',         # Gross Domestic Product
        'DGS10',       # 10-Year Treasury Rate
        'DGS2',        # 2-Year Treasury Rate
        'PAYEMS',      # Total Nonfarm Payrolls
        'DEXUSEU',     # USD/EUR Exchange Rate
        'DEXJPUS',     # USD/JPY Exchange Rate
        'DCOILWTICO',  # WTI Crude Oil Price
        'GOLDAMGBD228NLBM',  # Gold Fixing Price
        'BAMLH0A0HYM2', # ICE BofA High Yield Index
        'VIXCLS'       # CBOE Volatility Index
    ]

    # Download FRED data
    if fred_api_key:
        fred_data = download_fred_data(fred_series, fred_api_key)
        fred_clean = validate_macro_data(fred_data)
        fred_enriched = calculate_macro_features(fred_clean)

        if use_s3:
            store_macro_data_s3(fred_enriched, storage_path, 'fred')
        else:
            store_macro_data_parquet(fred_enriched, storage_path, 'fred')
    else:
        print("⚠️ Skipping FRED data (no API key provided)")

    # Download COT data
    cot_data = download_cot_data()
    cot_clean = validate_macro_data(cot_data)
    cot_enriched = calculate_macro_features(cot_clean)

    if use_s3:
        store_macro_data_s3(cot_enriched, storage_path, 'cot')
    else:
        store_macro_data_parquet(cot_enriched, storage_path, 'cot')

    # Download VIX data
    vix_data = download_vix_data()
    vix_clean = validate_macro_data(vix_data)
    vix_enriched = calculate_macro_features(vix_clean)

    if use_s3:
        store_macro_data_s3(vix_enriched, storage_path, 'vix')
    else:
        store_macro_data_parquet(vix_enriched, storage_path, 'vix')

    print("✅ Macroeconomic data ingestion completed")

if __name__ == "__main__":
    # Example usage (requires FRED API key)
    # fred_api_key = os.getenv('FRED_API_KEY')

    # Run the flow
    # ingest_macro_flow(fred_api_key=fred_api_key)

