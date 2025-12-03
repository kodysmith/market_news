"""
Backtest State Store

Saves backtest results to CSV/Parquet files.
"""

import pandas as pd
import os
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging
from .interface import StateStoreInterface

logger = logging.getLogger(__name__)


class BacktestStore(StateStoreInterface):
    """Saves backtest state to files"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.output_dir = config.get('output_dir', 'data/results/')
        os.makedirs(self.output_dir, exist_ok=True)
        self.runs = []
        self.positions_history = []

    def save_run(self, run_data: Dict[str, Any]) -> str:
        """Save run to CSV"""
        run_id = run_data.get('run_id', f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        run_data['run_id'] = run_id

        self.runs.append(run_data)

        # Save to CSV
        df = pd.DataFrame([run_data])
        file_path = os.path.join(self.output_dir, f"runs_{run_id}.csv")
        df.to_csv(file_path, index=False)

        logger.info(f"Saved run {run_id} to {file_path}")
        return run_id

    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Get run by ID"""
        for run in self.runs:
            if run.get('run_id') == run_id:
                return run
        return None

    def save_positions(self, positions: Dict[str, float], run_id: str):
        """Save positions to CSV"""
        timestamp = datetime.now()

        for ticker, value in positions.items():
            self.positions_history.append({
                'run_id': run_id,
                'timestamp': timestamp,
                'ticker': ticker,
                'value': value
            })

        # Save to CSV
        df = pd.DataFrame(self.positions_history)
        file_path = os.path.join(self.output_dir, f"positions_{run_id}.csv")
        df.to_csv(file_path, index=False)

    def get_positions(self, run_id: str) -> List[Dict[str, Any]]:
        """Get positions for a run"""
        return [p for p in self.positions_history if p.get('run_id') == run_id]

    def save_trades(self, trades: List[Dict[str, Any]], run_id: str):
        """Save trade history to CSV"""
        df = pd.DataFrame(trades)
        file_path = os.path.join(self.output_dir, f"trades_{run_id}.csv")
        df.to_csv(file_path, index=False)
        logger.info(f"Saved {len(trades)} trades to {file_path}")
