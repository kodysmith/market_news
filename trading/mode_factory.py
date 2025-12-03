"""
Mode Factory

Creates mode-specific adapters (data loaders, executors, state stores) based on mode.
"""

from typing import Dict, Any
import logging
from .data.backtest import BacktestDataLoader
from .data.live import LiveDataLoader
from .execution.backtest import BacktestExecutor
from .execution.paper import PaperExecutor
from .execution.real import RealExecutor
from .state.backtest import BacktestStore
from .state.supabase import SupabaseStateStore

logger = logging.getLogger(__name__)


class ModeFactory:
    """Factory for creating mode-specific components"""

    @staticmethod
    def create_data_loader(mode: str, config: Dict[str, Any]):
        """Create data loader for mode"""
        if mode == 'backtest':
            return BacktestDataLoader(config)
        elif mode in ['paper', 'live']:
            return LiveDataLoader(config)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    @staticmethod
    def create_executor(mode: str, config: Dict[str, Any], state_store=None, initial_capital: float = 100000):
        """Create executor for mode"""
        if mode == 'backtest':
            return BacktestExecutor(config, initial_capital)
        elif mode == 'paper':
            if state_store is None:
                raise ValueError("State store required for paper mode")
            return PaperExecutor(config, state_store)
        elif mode == 'live':
            return RealExecutor(config)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    @staticmethod
    def create_state_store(mode: str, config: Dict[str, Any]):
        """Create state store for mode"""
        if mode == 'backtest':
            return BacktestStore(config)
        elif mode in ['paper', 'live']:
            return SupabaseStateStore(config)
        else:
            raise ValueError(f"Unknown mode: {mode}")
