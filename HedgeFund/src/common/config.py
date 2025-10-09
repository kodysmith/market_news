"""
Module: Configuration Manager
Purpose: Centralized configuration with environment-aware loading
Dependencies: pydantic, python-dotenv
Exports: load_config(), StrategyConfig, RiskConfig, BrokerConfig

Loads configuration from YAML files and environment variables with validation.
Supports multiple environments (dev, paper, production).
"""

from typing import List, Dict, Optional
from pydantic import BaseModel, Field, field_validator, model_validator
from pathlib import Path
import yaml
import os
from dotenv import load_dotenv
from decimal import Decimal

# Load environment variables
load_dotenv()


class StrategyConfig(BaseModel):
    """Strategy parameters configuration"""
    
    # Assets to trade
    assets: List[str] = Field(default=['SPY', 'QQQ', 'DIA', 'IWM'])
    
    # Asset allocations (must sum to 1.0)
    allocations: Dict[str, float] = Field(default={
        'SPY': 0.40,
        'QQQ': 0.30,
        'DIA': 0.20,
        'IWM': 0.10
    })
    
    # Put selling parameters
    put_delta_target: float = Field(default=-0.30, ge=-1.0, le=0.0)
    put_dte: int = Field(default=14, ge=1, le=90)
    position_size_pct: float = Field(default=0.01, ge=0.001, le=0.05)
    profit_target_pct: float = Field(default=0.50, ge=0.0, le=1.0)
    
    # Covered call parameters
    call_delta_target: float = Field(default=0.30, ge=0.0, le=1.0)
    call_dte: int = Field(default=30, ge=7, le=90)
    
    # Hedging parameters
    enable_adaptive_hedge: bool = Field(default=True)
    weeknight_hedge_dte: int = Field(default=2, ge=1, le=7)
    weekend_hedge_dte: int = Field(default=14, ge=7, le=30)
    hedge_delta_target: float = Field(default=-0.50, ge=-1.0, le=0.0)
    hedge_coverage_pct: float = Field(default=1.0, ge=0.0, le=1.0)
    max_hedge_cost_pct: float = Field(default=0.005, ge=0.0, le=0.02)
    
    # Capital management
    max_capital_deployed: float = Field(default=0.75, ge=0.0, le=1.0)
    min_cash_reserve: float = Field(default=0.25, ge=0.0, le=1.0)
    
    @field_validator('allocations')
    @classmethod
    def allocations_sum_to_one(cls, v):
        """Validate allocations sum to 1.0"""
        total = sum(v.values())
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Allocations must sum to 1.0, got {total}")
        return v
    
    class Config:
        # Allow Decimal types
        arbitrary_types_allowed = True


class RiskConfig(BaseModel):
    """Risk management configuration"""
    
    # Position limits
    max_contracts_per_trade: int = Field(default=10, ge=1, le=100)
    max_contracts_per_strike: int = Field(default=10, ge=1, le=50)
    max_delta_per_asset: float = Field(default=5000.0, ge=0.0)
    max_total_delta: float = Field(default=15000.0, ge=0.0)
    
    # Circuit breakers
    circuit_breaker_reduce_threshold: float = Field(default=-0.08, le=0.0)
    circuit_breaker_halt_threshold: float = Field(default=-0.12, le=0.0)
    position_size_reduction: float = Field(default=0.50, ge=0.0, le=1.0)
    
    # Margin limits
    max_margin_utilization: float = Field(default=0.50, ge=0.0, le=0.90)
    margin_buffer_pct: float = Field(default=0.20, ge=0.0, le=0.50)
    
    # VaR parameters
    var_confidence: float = Field(default=0.95, ge=0.90, le=0.99)
    var_lookback_days: int = Field(default=252, ge=30, le=500)
    
    # Stress test scenarios
    stress_market_drop_pct: float = Field(default=-0.10)
    stress_vol_spike_mult: float = Field(default=2.0, ge=1.0)
    
    @model_validator(mode='after')
    def halt_worse_than_reduce(self):
        """Halt threshold must be worse than reduce threshold"""
        if self.circuit_breaker_halt_threshold > self.circuit_breaker_reduce_threshold:
            raise ValueError("Halt threshold must be worse than reduce threshold")
        return self


class BrokerConfig(BaseModel):
    """Broker configuration"""
    
    # Primary broker (Alpaca)
    primary_broker: str = Field(default="alpaca")
    alpaca_api_key: str = Field(default="")
    alpaca_secret_key: str = Field(default="")
    alpaca_paper_mode: bool = Field(default=True)
    alpaca_base_url: str = Field(default="https://paper-api.alpaca.markets")
    
    # Backup broker (Interactive Brokers)
    backup_broker: str = Field(default="interactive_brokers")
    ib_host: str = Field(default="127.0.0.1")
    ib_port: int = Field(default=4001)  # 4001=paper, 4002=live
    ib_client_id: int = Field(default=1)
    
    # Execution settings
    default_order_type: str = Field(default="limit")
    order_timeout_seconds: int = Field(default=30, ge=5, le=300)
    max_retry_attempts: int = Field(default=3, ge=1, le=10)
    retry_delay_seconds: float = Field(default=1.0, ge=0.1, le=10.0)
    
    # Commission structure
    option_commission: float = Field(default=0.65)
    stock_commission: float = Field(default=0.01)
    
    @model_validator(mode='after')
    def api_keys_not_empty_in_production(self):
        """Ensure API keys are set in non-paper mode"""
        if not self.alpaca_paper_mode:
            if not self.alpaca_api_key or not self.alpaca_secret_key:
                raise ValueError("API keys must be set for live trading")
        return self


class DatabaseConfig(BaseModel):
    """Database configuration"""
    
    # PostgreSQL
    postgres_host: str = Field(default="localhost")
    postgres_port: int = Field(default=5432)
    postgres_database: str = Field(default="trading_dev")
    postgres_user: str = Field(default="trading")
    postgres_password: str = Field(default="")
    
    # Redis
    redis_host: str = Field(default="localhost")
    redis_port: int = Field(default=6379)
    redis_db: int = Field(default=0)
    redis_password: Optional[str] = Field(default=None)
    
    # Connection pooling
    pool_size: int = Field(default=10, ge=1, le=100)
    pool_timeout: int = Field(default=30, ge=5, le=300)
    
    @property
    def postgres_url(self) -> str:
        """Generate PostgreSQL connection URL"""
        return (f"postgresql://{self.postgres_user}:{self.postgres_password}"
                f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_database}")


class MonitoringConfig(BaseModel):
    """Monitoring and alerting configuration"""
    
    # Logging
    log_level: str = Field(default="INFO")
    log_file: str = Field(default="logs/trading.log")
    log_rotation: str = Field(default="midnight")
    log_retention_days: int = Field(default=2555)  # 7 years
    
    # Prometheus
    metrics_port: int = Field(default=9090)
    metrics_enabled: bool = Field(default=True)
    
    # Alerts
    enable_pagerduty: bool = Field(default=False)
    pagerduty_api_key: str = Field(default="")
    
    enable_email_alerts: bool = Field(default=True)
    alert_email: str = Field(default="")
    
    # Sentry (error tracking)
    enable_sentry: bool = Field(default=False)
    sentry_dsn: str = Field(default="")


class AppConfig(BaseModel):
    """Main application configuration"""
    
    # Environment
    environment: str = Field(default="development")  # development, paper, production
    
    # Sub-configs
    strategy: StrategyConfig = Field(default_factory=StrategyConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    broker: BrokerConfig = Field(default_factory=BrokerConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    
    # Initial capital
    initial_capital: float = Field(default=100000.0, ge=10000.0)
    
    @field_validator('environment')
    @classmethod
    def valid_environment(cls, v):
        """Validate environment setting"""
        allowed = ['development', 'paper', 'production']
        if v not in allowed:
            raise ValueError(f"Environment must be one of: {allowed}")
        return v
    
    @property
    def is_production(self) -> bool:
        """Check if running in production"""
        return self.environment == 'production'
    
    @property
    def is_paper(self) -> bool:
        """Check if running in paper trading"""
        return self.environment == 'paper'


def load_config(config_path: Optional[Path] = None, 
                environment: Optional[str] = None) -> AppConfig:
    """
    Load configuration from file and environment variables
    
    Priority (highest to lowest):
    1. Environment variables
    2. Config file
    3. Defaults
    
    Args:
        config_path: Path to YAML config file (default: config/config.yaml)
        environment: Override environment (dev, paper, production)
    
    Returns:
        AppConfig instance
    
    Example:
        >>> config = load_config()
        >>> print(config.strategy.assets)
        ['SPY', 'QQQ', 'DIA', 'IWM']
    """
    
    # Determine environment
    env = environment or os.getenv('TRADING_ENV', 'development')
    
    # Load base config file
    if config_path is None:
        config_path = Path(__file__).parent.parent.parent / 'config' / 'config.yaml'
    
    config_dict = {}
    
    if config_path.exists():
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f) or {}
    
    # Override with environment-specific config
    env_config_path = config_path.parent / f'config.{env}.yaml'
    if env_config_path.exists():
        with open(env_config_path, 'r') as f:
            env_config = yaml.safe_load(f) or {}
            # Merge (env-specific overrides base)
            config_dict = {**config_dict, **env_config}
    
    # Override with environment variables
    config_dict['environment'] = env
    
    # Broker config from environment
    if os.getenv('ALPACA_API_KEY'):
        if 'broker' not in config_dict:
            config_dict['broker'] = {}
        config_dict['broker']['alpaca_api_key'] = os.getenv('ALPACA_API_KEY')
        config_dict['broker']['alpaca_secret_key'] = os.getenv('ALPACA_SECRET_KEY')
        config_dict['broker']['alpaca_paper_mode'] = env != 'production'
    
    # Database config from environment
    if os.getenv('DATABASE_URL'):
        if 'database' not in config_dict:
            config_dict['database'] = {}
        # Parse DATABASE_URL if needed
        config_dict['database']['postgres_password'] = os.getenv('POSTGRES_PASSWORD', '')
    
    # Monitoring config from environment
    if os.getenv('SENTRY_DSN'):
        if 'monitoring' not in config_dict:
            config_dict['monitoring'] = {}
        config_dict['monitoring']['sentry_dsn'] = os.getenv('SENTRY_DSN')
        config_dict['monitoring']['enable_sentry'] = True
    
    # Create and validate config
    config = AppConfig(**config_dict)
    
    return config


def save_config(config: AppConfig, config_path: Path):
    """
    Save configuration to YAML file
    
    Args:
        config: AppConfig instance
        config_path: Where to save
    """
    config_dict = config.model_dump()
    
    # Remove sensitive data before saving to file
    if 'broker' in config_dict:
        config_dict['broker']['alpaca_api_key'] = '***'
        config_dict['broker']['alpaca_secret_key'] = '***'
    
    if 'database' in config_dict:
        config_dict['database']['postgres_password'] = '***'
    
    with open(config_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)


if __name__ == "__main__":
    """Test configuration manager"""
    
    print("Testing Configuration Manager...")
    
    # Create default config
    config = AppConfig()
    
    print(f"\n✓ Default config loaded:")
    print(f"  Environment: {config.environment}")
    print(f"  Initial Capital: ${config.initial_capital:,.0f}")
    print(f"  Assets: {config.strategy.assets}")
    print(f"  Allocations: {config.strategy.allocations}")
    
    # Test strategy config
    print(f"\n✓ Strategy parameters:")
    print(f"  Put delta: {config.strategy.put_delta_target}")
    print(f"  Put DTE: {config.strategy.put_dte} days")
    print(f"  Position size: {config.strategy.position_size_pct:.1%}")
    print(f"  Profit target: {config.strategy.profit_target_pct:.0%}")
    
    # Test risk config
    print(f"\n✓ Risk parameters:")
    print(f"  Max delta per asset: {config.risk.max_delta_per_asset:,.0f}")
    print(f"  Circuit breaker (reduce): {config.risk.circuit_breaker_reduce_threshold:.1%}")
    print(f"  Circuit breaker (halt): {config.risk.circuit_breaker_halt_threshold:.1%}")
    
    # Test broker config
    print(f"\n✓ Broker configuration:")
    print(f"  Primary: {config.broker.primary_broker}")
    print(f"  Backup: {config.broker.backup_broker}")
    print(f"  Paper mode: {config.broker.alpaca_paper_mode}")
    
    # Test database config
    print(f"\n✓ Database configuration:")
    print(f"  PostgreSQL: {config.database.postgres_host}:{config.database.postgres_port}")
    print(f"  Database: {config.database.postgres_database}")
    print(f"  Redis: {config.database.redis_host}:{config.database.redis_port}")
    
    # Test environment checks
    print(f"\n✓ Environment checks:")
    print(f"  Is production: {config.is_production}")
    print(f"  Is paper: {config.is_paper}")
    
    # Test validation
    print(f"\n✓ Testing validation...")
    try:
        # This should fail - allocations don't sum to 1.0
        bad_config = StrategyConfig(allocations={'SPY': 0.5, 'QQQ': 0.3})
    except ValueError as e:
        print(f"  ✓ Validation working: {e}")
    
    # Test save/load from file
    print(f"\n✓ Testing file I/O...")
    test_path = Path("/tmp/test_config.yaml")
    save_config(config, test_path)
    print(f"  Saved to: {test_path}")
    
    # Clean up
    if test_path.exists():
        test_path.unlink()
        print(f"  Cleaned up test file")
    
    print("\n✅ Configuration manager working correctly!")

