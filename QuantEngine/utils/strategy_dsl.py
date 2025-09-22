"""
Strategy Domain Specific Language (DSL) for AI Quant Trading System

This module defines the schema and validation for trading strategy specifications.
All strategies must conform to this DSL before being executed.
"""

from typing import Dict, List, Any, Optional, Union
import json
from enum import Enum
from datetime import datetime

# Import pydantic v2 compatible
try:
    from pydantic import BaseModel, Field, field_validator
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False

    # Basic fallback implementation
    class BaseModel:
        def __init__(self, **data):
            for key, value in data.items():
                setattr(self, key, value)

        def dict(self):
            return self.__dict__

    def Field(**kwargs):
        return None

    def field_validator(field_name):
        def decorator(func):
            return func
        return decorator


class SignalType(str, Enum):
    """Supported signal types in the strategy DSL"""
    MA_CROSS = "MA_cross"
    IV_PROXY = "IV_proxy"
    VOLATILITY = "volatility"
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    RSI = "RSI"
    MACD = "MACD"
    BOLLINGER = "bollinger"
    SENTIMENT = "sentiment"
    CUSTOM = "custom"


class RuleOperator(str, Enum):
    """Supported rule operators"""
    GT = ">"
    LT = "<"
    GTE = ">="
    LTE = "<="
    EQ = "=="
    NE = "!="
    AND = "and"
    OR = "or"


class SignalDefinition(BaseModel):
    """Definition of a single signal"""
    type: SignalType
    name: Optional[str] = None
    params: Dict[str, Any] = Field(default_factory=dict)
    rule: Optional[str] = None  # Rule expression like "fast>slow"

    @field_validator('params')
    @classmethod
    def validate_params(cls, v, info):
        signal_type = info.data.get('type')
        if signal_type == SignalType.MA_CROSS:
            required = ['fast', 'slow']
        elif signal_type == SignalType.IV_PROXY:
            required = ['method']
        elif signal_type == SignalType.RSI:
            required = ['period', 'overbought', 'oversold']
        elif signal_type == SignalType.BOLLINGER:
            required = ['period', 'std_dev']
        else:
            required = []

        for param in required:
            if param not in v:
                raise ValueError(f"Parameter '{param}' required for signal type {signal_type}")
        return v


class EntryCondition(BaseModel):
    """Entry condition specification"""
    all: Optional[List[str]] = None  # All conditions must be true
    any: Optional[List[str]] = None  # Any condition must be true
    none: Optional[List[str]] = None  # None of these conditions should be true


class ExitCondition(BaseModel):
    """Exit condition specification"""
    all: Optional[List[str]] = None
    any: Optional[List[str]] = None
    time_based: Optional[Dict[str, Any]] = None  # e.g., {"max_holding_days": 30}


class OverlayType(str, Enum):
    """Supported overlay types"""
    PUTS = "puts"
    CALLS = "calls"
    COLLARS = "collars"


class OptionOverlay(BaseModel):
    """Options overlay specification"""
    target_delta: float = Field(..., ge=-1.0, le=1.0)
    ratio: float = Field(..., gt=0)  # Size relative to underlying position
    dte_min: int = Field(..., ge=1)  # Minimum days to expiration
    dte_max: int = Field(..., le=365)  # Maximum days to expiration
    budget_pct_month: float = Field(..., ge=0, le=1)  # Monthly budget as % of portfolio
    roll_trigger: Optional[Dict[str, Any]] = None  # Auto-roll conditions


class PositionSizing(BaseModel):
    """Position sizing specification"""
    vol_target_ann: float = Field(..., ge=0, le=2.0)  # Annualized vol target
    max_weight: float = Field(..., ge=0, le=5.0)  # Max weight per position
    min_weight: float = Field(default=0.0, ge=0)  # Min weight per position
    kelly_fraction: Optional[float] = Field(default=0.5, ge=0, le=1)  # Kelly fraction


class CostModel(BaseModel):
    """Trading cost model"""
    commission_bps: float = Field(default=2.0, ge=0)  # Commission in basis points
    fee_per_option: float = Field(default=0.65, ge=0)  # Per-option fee
    slippage_bps: float = Field(default=1.0, ge=0)  # Slippage in basis points
    borrow_rate_ann: float = Field(default=0.02, ge=0)  # Annual borrow rate for shorts


class RiskLimits(BaseModel):
    """Risk management limits"""
    max_dd_pct: float = Field(..., ge=0, le=1)  # Max drawdown as fraction
    max_gross_exposure: float = Field(default=1.2, ge=0)  # Max gross exposure
    max_sector_weight: float = Field(default=0.3, ge=0)  # Max sector concentration
    max_single_position: float = Field(default=0.1, ge=0)  # Max single position weight
    circuit_breaker_dd: float = Field(default=0.05, ge=0)  # Halt threshold


class StrategySpec(BaseModel):
    """Complete strategy specification in DSL"""
    name: str
    universe: List[str] = Field(..., min_items=1)  # Tickers to trade
    description: Optional[str] = None
    signals: List[SignalDefinition] = Field(..., min_items=1)
    entry: EntryCondition
    exit: Optional[ExitCondition] = None
    sizing: PositionSizing
    overlays: Optional[Dict[str, OptionOverlay]] = None  # Options overlays
    costs: CostModel = Field(default_factory=CostModel)
    risk: RiskLimits
    metadata: Optional[Dict[str, Any]] = None  # Additional metadata

    @field_validator('name')
    @classmethod
    def validate_name(cls, v):
        if not v.replace('_', '').replace('-', '').isalnum():
            raise ValueError("Strategy name must be alphanumeric with underscores/hyphens only")
        return v

    @field_validator('universe')
    @classmethod
    def validate_universe(cls, v):
        # Basic ticker validation - could be enhanced
        for ticker in v:
            if not ticker or len(ticker) > 10:
                raise ValueError(f"Invalid ticker: {ticker}")
        return v


class StrategyValidator:
    """Validates strategy specifications against the DSL"""

    @staticmethod
    def validate_spec(spec_dict: Dict[str, Any]) -> StrategySpec:
        """Validate a strategy specification dictionary"""
        try:
            return StrategySpec(**spec_dict)
        except Exception as e:
            raise ValueError(f"Strategy validation failed: {e}")

    @staticmethod
    def validate_spec_json(json_str: str) -> StrategySpec:
        """Validate a strategy specification from JSON string"""
        try:
            spec_dict = json.loads(json_str)
            return StrategyValidator.validate_spec(spec_dict)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON: {e}")

    @staticmethod
    def validate_spec_file(filepath: str) -> StrategySpec:
        """Validate a strategy specification from file"""
        with open(filepath, 'r') as f:
            return StrategyValidator.validate_spec_json(f.read())


# Example strategy specifications
EXAMPLE_TQQQ_STRATEGY = {
    "name": "tqqq_regime_puts_v1",
    "universe": ["TQQQ"],
    "description": "Long TQQQ with put protection during low IV regimes",
    "signals": [
        {
            "type": "MA_cross",
            "name": "trend_filter",
            "params": {"fast": 20, "slow": 200},
            "rule": "fast>slow"
        },
        {
            "type": "IV_proxy",
            "name": "vol_regime",
            "params": {"method": "rv20_scaled", "low_thresh": 0.45}
        }
    ],
    "entry": {
        "all": ["trend_filter.rule", "vol_regime<low_thresh"]
    },
    "exit": {
        "any": ["trend_filter.fast<trend_filter.slow"]
    },
    "sizing": {
        "vol_target_ann": 0.15,
        "max_weight": 1.0,
        "kelly_fraction": 0.5
    },
    "overlays": {
        "puts": {
            "target_delta": -0.2,
            "ratio": 0.5,
            "dte_min": 30,
            "dte_max": 90,
            "budget_pct_month": 0.01,
            "roll_trigger": {"min_delta_abs": 0.06, "min_dte": 25}
        }
    },
    "costs": {
        "commission_bps": 2.0,
        "fee_per_option": 0.65,
        "slippage_bps": 1.0
    },
    "risk": {
        "max_dd_pct": 0.25,
        "max_gross_exposure": 1.2,
        "max_sector_weight": 0.3,
        "circuit_breaker_dd": 0.05
    }
}


if __name__ == "__main__":
    # Test validation
    try:
        spec = StrategyValidator.validate_spec(EXAMPLE_TQQQ_STRATEGY)
        print(f"✓ Strategy '{spec.name}' validated successfully")
        print(f"  Universe: {spec.universe}")
        print(f"  Signals: {len(spec.signals)}")
        print(f"  Has overlays: {spec.overlays is not None}")
    except Exception as e:
        print(f"✗ Validation failed: {e}")
