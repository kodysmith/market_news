#!/usr/bin/env python3
"""
Feature Factory

Builds trading signals and features using DuckDB for fast data processing.
Combines equity, options, and macroeconomic data to create regime-aware features.
"""

import duckdb
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
import pyarrow.parquet as pq
import pyarrow as pa
from prefect import flow, task
from prefect.tasks import task_input_hash
import warnings
warnings.filterwarnings('ignore')

class FeatureFactory:
    """Factory for building trading features using DuckDB"""

    def __init__(self, data_path: str = "./data"):
        """
        Initialize the feature factory

        Args:
            data_path: Path to the data directory
        """
        self.data_path = data_path
        self.con = duckdb.connect(database=':memory:')

        # Enable efficient Parquet reading
        self.con.execute("SET enable_object_cache=true")
        self.con.execute("SET max_memory='4GB'")

    def load_equity_data(self, symbols: List[str]) -> None:
        """Load equity data into DuckDB"""
        print(f"📊 Loading equity data for {len(symbols)} symbols")

        for symbol in symbols:
            parquet_path = os.path.join(self.data_path, 'equity', symbol.lower(), '**', '*.parquet')

            if os.path.exists(os.path.dirname(parquet_path)):
                try:
                    # Create table from Parquet files
                    table_name = f"equity_{symbol.lower()}"
                    self.con.execute(f"""
                        CREATE OR REPLACE TABLE {table_name} AS
                        SELECT * FROM read_parquet('{parquet_path}', hive_partitioning=true)
                    """)

                    # Create index on date for fast queries
                    self.con.execute(f"CREATE INDEX idx_{table_name}_date ON {table_name}(date)")

                    print(f"✅ Loaded {table_name}")
                except Exception as e:
                    print(f"❌ Failed to load {symbol}: {e}")
            else:
                print(f"⚠️ No data found for {symbol}")

    def load_options_data(self, symbols: List[str]) -> None:
        """Load options data into DuckDB"""
        print(f"📊 Loading options data for {len(symbols)} symbols")

        for symbol in symbols:
            parquet_path = os.path.join(self.data_path, 'options', symbol.lower(), '**', '**', '*.parquet')

            if os.path.exists(os.path.dirname(parquet_path)):
                try:
                    # Create table from Parquet files
                    table_name = f"options_{symbol.lower()}"
                    self.con.execute(f"""
                        CREATE OR REPLACE TABLE {table_name} AS
                        SELECT * FROM read_parquet('{parquet_path}', hive_partitioning=true)
                    """)

                    # Create indexes for fast queries
                    self.con.execute(f"CREATE INDEX idx_{table_name}_date ON {table_name}(date)")
                    self.con.execute(f"CREATE INDEX idx_{table_name}_expiration ON {table_name}(expiration_date)")
                    self.con.execute(f"CREATE INDEX idx_{table_name}_strike ON {table_name}(strike)")

                    print(f"✅ Loaded {table_name}")
                except Exception as e:
                    print(f"❌ Failed to load options for {symbol}: {e}")
            else:
                print(f"⚠️ No options data found for {symbol}")

    def load_macro_data(self) -> None:
        """Load macroeconomic data into DuckDB"""
        print("📊 Loading macroeconomic data")

        macro_types = ['fred', 'cot', 'vix']

        for macro_type in macro_types:
            parquet_path = os.path.join(self.data_path, 'macro', macro_type, '**', '*.parquet')

            if os.path.exists(os.path.dirname(parquet_path)):
                try:
                    # Create table from Parquet files
                    table_name = f"macro_{macro_type}"
                    self.con.execute(f"""
                        CREATE OR REPLACE TABLE {table_name} AS
                        SELECT * FROM read_parquet('{parquet_path}', hive_partitioning=true)
                    """)

                    # Create index on date
                    self.con.execute(f"CREATE INDEX idx_{table_name}_date ON {table_name}(date)")

                    print(f"✅ Loaded {table_name}")
                except Exception as e:
                    print(f"❌ Failed to load {macro_type}: {e}")
            else:
                print(f"⚠️ No {macro_type} data found")

    def build_regime_features(self, symbol: str) -> pd.DataFrame:
        """
        Build market regime features using VIX and macro data

        Args:
            symbol: Stock ticker symbol

        Returns:
            DataFrame with regime features
        """
        print(f"🏗️ Building regime features for {symbol}")

        query = f"""
        WITH equity_data AS (
            SELECT
                date,
                close,
                LAG(close, 20) OVER (ORDER BY date) as close_20d_ago,
                LAG(close, 200) OVER (ORDER BY date) as close_200d_ago,
                AVG(close) OVER (ORDER BY date ROWS BETWEEN 199 PRECEDING AND CURRENT ROW) as sma_200,
                STDDEV(close) OVER (ORDER BY date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) as vol_20d
            FROM equity_{symbol.lower()}
            WHERE date >= '2010-01-01'
        ),
        macro_data AS (
            SELECT
                date,
                vix_percentile_252d,
                vix_regime,
                yield_curve_slope,
                fred_UNRATE as unemployment_rate,
                fred_DGS10 as ten_year_yield,
                fred_DGS2 as two_year_yield
            FROM macro_vix vix
            FULL OUTER JOIN macro_fred fred USING (date)
            WHERE date >= '2010-01-01'
        )
        SELECT
            e.date,
            e.close,

            -- Trend features
            CASE WHEN e.close > e.sma_200 THEN 1 ELSE 0 END as trend_up,
            (e.close - e.sma_200) / NULLIF(e.sma_200, 0) as trend_strength,

            -- Momentum features
            (e.close - e.close_20d_ago) / NULLIF(e.close_20d_ago, 0) as mom_20d,
            (e.close - e.close_200d_ago) / NULLIF(e.close_200d_ago, 0) as mom_200d,

            -- Volatility regime
            CASE
                WHEN m.vix_regime = 'low_vol' THEN 0
                WHEN m.vix_regime = 'normal_vol' THEN 1
                WHEN m.vix_regime = 'high_vol' THEN 2
                WHEN m.vix_regime = 'extreme_vol' THEN 3
                ELSE 1
            END as vol_regime,

            -- Macro regime
            CASE
                WHEN m.yield_curve_slope < 0 THEN 0  -- Inverted
                WHEN m.yield_curve_slope < 0.5 THEN 1  -- Flat
                ELSE 2  -- Normal
            END as yield_curve_regime,

            CASE
                WHEN m.unemployment_rate > 6.0 THEN 0  -- High unemployment
                WHEN m.unemployment_rate > 4.0 THEN 1  -- Normal
                ELSE 2  -- Low unemployment
            END as employment_regime,

            -- Combined risk regime (0=low risk, 1=normal, 2=high risk, 3=extreme risk)
            CASE
                WHEN (m.vix_regime IN ('low_vol', 'normal_vol')) AND (e.trend_up = 1) AND (m.yield_curve_slope > 0) THEN 0
                WHEN (m.vix_regime = 'high_vol') OR (e.trend_up = 0) OR (m.yield_curve_slope < 0.5) THEN 1
                WHEN (m.vix_regime = 'extreme_vol') OR (m.unemployment_rate > 6.0) THEN 2
                ELSE 1
            END as risk_regime,

            -- VIX features
            m.vix_percentile_252d as vix_percentile,

            -- Macro features
            m.yield_curve_slope,
            m.unemployment_rate,
            m.ten_year_yield,
            m.two_year_yield

        FROM equity_data e
        LEFT JOIN macro_data m USING (date)
        ORDER BY e.date
        """

        try:
            result = self.con.execute(query).fetchdf()
            print(f"✅ Built regime features: {len(result)} rows")
            return result
        except Exception as e:
            print(f"❌ Failed to build regime features: {e}")
            return pd.DataFrame()

    def build_options_features(self, symbol: str) -> pd.DataFrame:
        """
        Build options-based features including skew, term structure, and positioning

        Args:
            symbol: Stock ticker symbol

        Returns:
            DataFrame with options features
        """
        print(f"🏗️ Building options features for {symbol}")

        query = f"""
        WITH equity_data AS (
            SELECT
                date,
                close,
                vol_20d
            FROM equity_{symbol.lower()}
            WHERE date >= '2010-01-01'
        ),
        options_data AS (
            SELECT
                date,
                expiration_date,
                strike,
                option_type,
                last as option_price,
                bid,
                ask,
                volume,
                open_interest,
                delta,
                gamma,
                theta,
                vega,
                rho,
                implied_volatility,
                dte,
                moneyness,
                moneyness_category,
                (expiration_date - date) as days_to_expiry
            FROM options_{symbol.lower()}
            WHERE date >= '2010-01-01'
            AND dte BETWEEN 1 AND 365
        ),
        atm_options AS (
            -- Find at-the-money options for each date/expiration
            SELECT
                date,
                expiration_date,
                option_type,
                option_price,
                implied_volatility,
                delta,
                ROW_NUMBER() OVER (
                    PARTITION BY date, expiration_date, option_type
                    ORDER BY ABS(delta - 0.5)
                ) as rn
            FROM options_data
            WHERE ABS(delta - 0.5) < 0.1  -- Close to ATM
        ),
        skew_data AS (
            -- Calculate put-call skew
            SELECT
                date,
                expiration_date,
                AVG(CASE WHEN option_type = 'put' THEN implied_volatility END) as put_vol,
                AVG(CASE WHEN option_type = 'call' THEN implied_volatility END) as call_vol,
                COUNT(CASE WHEN option_type = 'put' THEN 1 END) as put_count,
                COUNT(CASE WHEN option_type = 'call' THEN 1 END) as call_count
            FROM atm_options
            WHERE rn = 1
            GROUP BY date, expiration_date
        )
        SELECT
            e.date,
            e.close,

            -- ATM implied volatility by tenor
            AVG(CASE WHEN o.days_to_expiry BETWEEN 1 AND 30 THEN o.implied_volatility END) as iv_1m,
            AVG(CASE WHEN o.days_to_expiry BETWEEN 31 AND 60 THEN o.implied_volatility END) as iv_2m,
            AVG(CASE WHEN o.days_to_expiry BETWEEN 61 AND 90 THEN o.implied_volatility END) as iv_3m,
            AVG(CASE WHEN o.days_to_expiry BETWEEN 91 AND 180 THEN o.implied_volatility END) as iv_6m,

            -- Volatility term structure
            (iv_3m - iv_1m) as vol_term_structure,

            -- Put-call skew
            AVG(s.put_vol - s.call_vol) as put_call_skew,

            -- Open interest concentration
            SUM(CASE WHEN o.moneyness_category = 'atm' THEN o.open_interest ELSE 0 END) /
            NULLIF(SUM(o.open_interest), 0) as oi_atm_ratio,

            SUM(CASE WHEN o.moneyness_category IN ('otm', 'deep_otm') AND o.option_type = 'put' THEN o.open_interest ELSE 0 END) /
            NULLIF(SUM(o.open_interest), 0) as oi_put_otm_ratio,

            -- Gamma exposure
            SUM(CASE WHEN ABS(o.delta) BETWEEN 0.3 AND 0.7 THEN o.gamma * o.open_interest ELSE 0 END) as gamma_exposure,

            -- Realized vs implied volatility
            AVG(o.implied_volatility) / NULLIF(AVG(e.vol_20d), 0) as iv_rv_ratio,

            -- Options volume ratio
            SUM(CASE WHEN o.option_type = 'put' THEN o.volume ELSE 0 END) /
            NULLIF(SUM(CASE WHEN o.option_type = 'call' THEN o.volume ELSE 0 END), 0) as put_call_volume_ratio

        FROM equity_data e
        LEFT JOIN options_data o ON e.date = o.date
        LEFT JOIN skew_data s ON e.date = s.date
        GROUP BY e.date, e.close
        ORDER BY e.date
        """

        try:
            result = self.con.execute(query).fetchdf()
            print(f"✅ Built options features: {len(result)} rows")
            return result
        except Exception as e:
            print(f"❌ Failed to build options features: {e}")
            return pd.DataFrame()

    def build_positioning_features(self) -> pd.DataFrame:
        """
        Build positioning features from COT data

        Returns:
            DataFrame with positioning features
        """
        print("🏗️ Building positioning features")

        query = """
        SELECT
            date,
            report_date,

            -- Producer positioning (commercial hedgers)
            AVG(producer_long - producer_short) as producer_net_position,

            -- Swap dealer positioning
            AVG(swap_long - swap_short) as swap_net_position,

            -- Managed money positioning (speculators)
            AVG(managed_long - managed_short) as managed_net_position,

            -- Extreme positioning signals
            CASE
                WHEN AVG(producer_long - producer_short) > 0 THEN 1  -- Producers net long (bearish)
                ELSE 0
            END as producer_bearish,

            CASE
                WHEN AVG(managed_long - managed_short) > 0 THEN 1  -- Specs net long (bullish)
                ELSE 0
            END as spec_bullish,

            -- Positioning divergence
            AVG(managed_long - managed_short) - AVG(producer_long - producer_short) as positioning_divergence

        FROM macro_cot
        WHERE date >= '2010-01-01'
        GROUP BY date, report_date
        ORDER BY date
        """

        try:
            result = self.con.execute(query).fetchdf()
            print(f"✅ Built positioning features: {len(result)} rows")
            return result
        except Exception as e:
            print(f"❌ Failed to build positioning features: {e}")
            return pd.DataFrame()

    def build_combined_features(self, symbol: str) -> pd.DataFrame:
        """
        Build combined feature set for backtesting

        Args:
            symbol: Stock ticker symbol

        Returns:
            DataFrame with all features
        """
        print(f"🏗️ Building combined features for {symbol}")

        # Get individual feature sets
        regime_features = self.build_regime_features(symbol)
        options_features = self.build_options_features(symbol)
        positioning_features = self.build_positioning_features()

        if regime_features.empty:
            print("⚠️ No regime features available")
            return pd.DataFrame()

        # Merge all features
        features = regime_features.copy()

        if not options_features.empty:
            features = features.merge(options_features, on='date', how='left')

        if not positioning_features.empty:
            features = features.merge(positioning_features, on='date', how='left')

        # Fill missing values
        features = features.fillna(method='ffill').fillna(0)

        # Add derived features
        features['composite_risk_score'] = (
            features['vol_regime'] * 0.4 +
            features['yield_curve_regime'] * 0.3 +
            features['employment_regime'] * 0.3
        )

        features['momentum_signal'] = (
            (features['mom_20d'] > 0).astype(int) +
            (features['mom_200d'] > 0).astype(int) +
            (features['trend_up'] > 0).astype(int)
        )

        print(f"✅ Built combined features: {len(features)} rows, {len(features.columns)} features")
        return features

    def save_features(self, features: pd.DataFrame, symbol: str, output_path: str) -> str:
        """
        Save features to Parquet format

        Args:
            features: DataFrame with features
            symbol: Stock ticker symbol
            output_path: Output directory path

        Returns:
            Path where features were saved
        """
        if features.empty:
            print("⚠️ No features to save")
            return ""

        # Create output directory
        output_dir = os.path.join(output_path, 'features', symbol.lower())
        os.makedirs(output_dir, exist_ok=True)

        # Save to Parquet with date partitioning
        table = pa.Table.from_pandas(features)
        pq.write_to_dataset(
            table,
            root_path=output_dir,
            partition_cols=['date'],
            compression='snappy',
            coerce_timestamps='ms',
            allow_truncated_timestamps=True
        )

        print(f"💾 Saved features to {output_dir}")
        return output_dir

@task(cache_key_fn=task_input_hash, cache_expiration=timedelta(hours=1))
def build_features_for_symbol(symbol: str, data_path: str = "./data") -> str:
    """
    Build features for a single symbol

    Args:
        symbol: Stock ticker symbol
        data_path: Path to data directory

    Returns:
        Path where features were saved
    """
    factory = FeatureFactory(data_path)

    # Load data
    factory.load_equity_data([symbol])
    factory.load_options_data([symbol])
    factory.load_macro_data()

    # Build features
    features = factory.build_combined_features(symbol)

    if not features.empty:
        output_path = factory.save_features(features, symbol, data_path)
        return output_path
    else:
        print(f"⚠️ No features generated for {symbol}")
        return ""

@flow(name="build-features")
def build_features_flow(
    symbols: List[str],
    data_path: str = "./data"
) -> Dict[str, str]:
    """
    Build features for multiple symbols

    Args:
        symbols: List of stock ticker symbols
        data_path: Path to data directory

    Returns:
        Dictionary mapping symbols to feature file paths
    """
    print(f"🚀 Building features for {len(symbols)} symbols")

    results = {}

    for symbol in symbols:
        try:
            output_path = build_features_for_symbol(symbol, data_path)
            if output_path:
                results[symbol] = output_path
                print(f"✅ Built features for {symbol}")
            else:
                print(f"❌ Failed to build features for {symbol}")
        except Exception as e:
            print(f"❌ Error building features for {symbol}: {e}")
            continue

    print(f"🎉 Feature building completed for {len(results)} symbols")
    return results

if __name__ == "__main__":
    # Example usage
    symbols = ["SPY", "QQQ", "AAPL"]
    results = build_features_flow(symbols)
    print("Feature building results:", results)

