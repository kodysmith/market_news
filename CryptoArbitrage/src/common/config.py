"""
Configuration Management for Crypto Arbitrage System
Environment-aware configuration with validation
"""

import os
from pathlib import Path
from typing import List, Dict, Optional
from decimal import Decimal
import yaml
from pydantic import BaseModel, Field, field_validator
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class ExchangeConfig(BaseModel):
    """Configuration for a single exchange"""
    name: str
    api_key: str = Field(default="")
    api_secret: str = Field(default="")
    api_passphrase: Optional[str] = Field(default=None)  # For Coinbase
    
    # Settings
    enabled: bool = Field(default=True)
    testnet: bool = Field(default=False)
    rate_limit: int = Field(default=1000)  # Requests per minute
    
    # Fees (will be overridden by actual exchange data when available)
    maker_fee_pct: Decimal = Field(default=Decimal('0.1'))
    taker_fee_pct: Decimal = Field(default=Decimal('0.1'))


class StrategyConfig(BaseModel):
    """Arbitrage strategy parameters"""
    
    # Capital allocation
    total_capital_usd: Decimal = Field(default=Decimal('10000'))
    max_position_size_usd: Decimal = Field(default=Decimal('1000'))
    max_position_size_pct: Decimal = Field(default=Decimal('0.1'))  # 10% of capital
    
    # Profit thresholds
    min_spread_pct: Decimal = Field(default=Decimal('0.5'), ge=Decimal('0.1'))
    target_spread_pct: Decimal = Field(default=Decimal('1.0'))
    
    # Execution parameters
    execution_timeout_seconds: int = Field(default=2, ge=1, le=10)
    max_price_staleness_seconds: int = Field(default=5)
    
    # Pairs to monitor (Top 50 by market cap)
    trading_pairs: List[str] = Field(default_factory=lambda: [
        # Major pairs
        'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT', 'XRP/USDT',
        'ADA/USDT', 'DOGE/USDT', 'AVAX/USDT', 'DOT/USDT', 'MATIC/USDT',
        # DeFi
        'LINK/USDT', 'UNI/USDT', 'AAVE/USDT', 'ATOM/USDT', 'LTC/USDT',
        # Layer 1s
        'NEAR/USDT', 'FTM/USDT', 'ALGO/USDT', 'VET/USDT', 'ICP/USDT',
        # Others
        'XLM/USDT', 'FIL/USDT', 'ETC/USDT', 'APT/USDT', 'ARB/USDT',
        # Memecoins
        'SHIB/USDT', 'PEPE/USDT',
        # Additional top 50
        'OP/USDT', 'IMX/USDT', 'LDO/USDT', 'INJ/USDT', 'HBAR/USDT',
        'QNT/USDT', 'RUNE/USDT', 'GRT/USDT', 'SAND/USDT', 'MANA/USDT'
    ])
    
    # Volume requirements
    min_24h_volume_usd: Decimal = Field(default=Decimal('1000000'))  # $1M daily volume
    
    @field_validator('trading_pairs')
    @classmethod
    def validate_pairs(cls, v):
        """Ensure all pairs are valid format"""
        for pair in v:
            if '/' not in pair:
                raise ValueError(f"Invalid pair format: {pair}. Must be BASE/QUOTE")
        return v


class RiskConfig(BaseModel):
    """Risk management parameters"""
    
    # Position limits
    max_total_exposure_usd: Decimal = Field(default=Decimal('5000'))
    max_single_exchange_exposure_pct: Decimal = Field(default=Decimal('0.6'))
    
    # Circuit breakers
    max_daily_loss_usd: Decimal = Field(default=Decimal('500'))
    max_consecutive_failures: int = Field(default=5)
    
    # Rate limiting
    max_trades_per_hour: int = Field(default=20)
    max_trades_per_day: int = Field(default=100)
    
    # Balance requirements
    min_balance_per_exchange_usd: Decimal = Field(default=Decimal('100'))
    rebalance_threshold_pct: Decimal = Field(default=Decimal('0.3'))  # 30% imbalance


class MonitoringConfig(BaseModel):
    """Monitoring and alerting configuration"""
    
    # Performance tracking
    track_latency: bool = Field(default=True)
    track_opportunities: bool = Field(default=True)
    
    # Alerting
    alert_on_failure: bool = Field(default=True)
    alert_on_low_balance: bool = Field(default=True)
    alert_threshold_balance_usd: Decimal = Field(default=Decimal('50'))
    
    # Dashboard
    dashboard_enabled: bool = Field(default=True)
    dashboard_port: int = Field(default=5001)  # Different from HedgeFund


class DatabaseConfig(BaseModel):
    """Database configuration"""
    sqlite_path: str = Field(default="./data/arbitrage.db")
    enable_wal_mode: bool = Field(default=True)  # Write-ahead logging
    backup_enabled: bool = Field(default=True)
    backup_interval_hours: int = Field(default=24)


class AppConfig(BaseModel):
    """Complete application configuration"""
    
    # Environment
    environment: str = Field(default="development")
    
    # Sub-configurations
    exchanges: Dict[str, ExchangeConfig] = Field(default_factory=dict)
    strategy: StrategyConfig = Field(default_factory=StrategyConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    
    @field_validator('exchanges')
    @classmethod
    def validate_exchanges(cls, v):
        """Ensure at least 2 exchanges configured"""
        if len(v) < 2:
            raise ValueError("Need at least 2 exchanges for arbitrage")
        return v


def load_config(config_path: Optional[Path] = None,
                environment: Optional[str] = None) -> AppConfig:
    """
    Load configuration from YAML and environment variables
    
    Args:
        config_path: Path to config file
        environment: Override environment (development/simulation/production)
    
    Returns:
        AppConfig instance
    """
    # Determine environment
    env = environment or os.getenv('CRYPTO_ENV', 'development')
    
    # Default config path
    if config_path is None:
        config_path = Path(__file__).parent.parent.parent / 'config' / 'config.yaml'
    
    # Load base config
    config_dict = {}
    if config_path.exists():
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f) or {}
    
    # Override with environment-specific config
    env_config_path = config_path.parent / f'config.{env}.yaml'
    if env_config_path.exists():
        with open(env_config_path, 'r') as f:
            env_config = yaml.safe_load(f) or {}
            config_dict.update(env_config)
    
    # Set environment
    config_dict['environment'] = env
    
    # Override with environment variables
    
    # Binance
    if 'exchanges' not in config_dict:
        config_dict['exchanges'] = {}
    
    if 'binance' not in config_dict['exchanges']:
        config_dict['exchanges']['binance'] = {}
    
    if os.getenv('BINANCE_API_KEY'):
        config_dict['exchanges']['binance']['api_key'] = os.getenv('BINANCE_API_KEY')
    if os.getenv('BINANCE_API_SECRET'):
        config_dict['exchanges']['binance']['api_secret'] = os.getenv('BINANCE_API_SECRET')
    
    # Coinbase
    if 'coinbase' not in config_dict['exchanges']:
        config_dict['exchanges']['coinbase'] = {}
    
    if os.getenv('COINBASE_API_KEY'):
        config_dict['exchanges']['coinbase']['api_key'] = os.getenv('COINBASE_API_KEY')
    if os.getenv('COINBASE_API_SECRET'):
        config_dict['exchanges']['coinbase']['api_secret'] = os.getenv('COINBASE_API_SECRET')
    if os.getenv('COINBASE_PASSPHRASE'):
        config_dict['exchanges']['coinbase']['api_passphrase'] = os.getenv('COINBASE_PASSPHRASE')
    
    # Kraken
    if 'kraken' not in config_dict['exchanges']:
        config_dict['exchanges']['kraken'] = {}
    
    if os.getenv('KRAKEN_API_KEY'):
        config_dict['exchanges']['kraken']['api_key'] = os.getenv('KRAKEN_API_KEY')
    if os.getenv('KRAKEN_API_SECRET'):
        config_dict['exchanges']['kraken']['api_secret'] = os.getenv('KRAKEN_API_SECRET')
    
    # Create and validate
    config = AppConfig(**config_dict)
    
    return config


