"""
State Store Interface

Abstract interface for state management in different modes.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional


class StateStoreInterface(ABC):
    """Abstract interface for state stores"""

    @abstractmethod
    def save_run(self, run_data: Dict[str, Any]) -> str:
        """
        Save a trading run

        Args:
            run_data: Run metadata and results

        Returns:
            Run ID
        """
        pass

    @abstractmethod
    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Get run data by ID"""
        pass

    @abstractmethod
    def save_positions(self, positions: Dict[str, float], run_id: str):
        """Save positions for a run"""
        pass

    @abstractmethod
    def get_positions(self, run_id: str) -> List[Dict[str, Any]]:
        """Get positions for a run"""
        pass
