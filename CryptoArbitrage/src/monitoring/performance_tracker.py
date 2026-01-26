"""
Performance Tracker
Calculates and tracks arbitrage system performance metrics
"""

from typing import Dict, List
from decimal import Decimal
from datetime import datetime, timedelta
from collections import defaultdict
import logging

from ..database.repository import ArbitrageRepository

logger = logging.getLogger(__name__)


class PerformanceTracker:
    """
    Tracks arbitrage system performance
    
    Calculates metrics like:
    - Total P&L
    - Success rate
    - Average execution time
    - ROI
    """
    
    def __init__(self, repository: ArbitrageRepository):
        """
        Initialize performance tracker
        
        Args:
            repository: Database repository
        """
        self.repository = repository
        
        # Real-time metrics
        self.session_start = datetime.now()
        self.session_profit = Decimal('0')
        self.session_trades = 0
        
        logger.info("PerformanceTracker initialized")
    
    def calculate_current_performance(self) -> Dict:
        """
        Calculate current session performance
        
        Returns:
            Dict with performance metrics
        """
        # Get today's data from database
        today_data = self.repository.get_daily_pnl(days=1)
        
        if today_data:
            today = today_data[0]
            return {
                'session_duration_hours': (datetime.now() - self.session_start).total_seconds() / 3600,
                'total_trades': today.get('num_trades', 0),
                'successful_trades': today.get('successful_trades', 0),
                'success_rate': (
                    today.get('successful_trades', 0) / today.get('num_trades', 1)
                    if today.get('num_trades', 0) > 0 else 0
                ),
                'total_profit_usd': float(today.get('total_profit', 0) or 0),
                'avg_spread_pct': float(today.get('avg_spread', 0) or 0),
                'avg_execution_time_ms': float(today.get('avg_execution_time', 0) or 0),
                'trades_per_hour': (
                    today.get('num_trades', 0) / max(1, (datetime.now() - self.session_start).total_seconds() / 3600)
                )
            }
        
        return {
            'session_duration_hours': 0,
            'total_trades': 0,
            'successful_trades': 0,
            'success_rate': 0,
            'total_profit_usd': 0,
            'avg_spread_pct': 0,
            'avg_execution_time_ms': 0,
            'trades_per_hour': 0
        }
    
    def calculate_historical_performance(self, days: int = 30) -> Dict:
        """
        Calculate historical performance
        
        Args:
            days: Number of days to analyze
        
        Returns:
            Dict with historical metrics
        """
        daily_data = self.repository.get_daily_pnl(days)
        
        if not daily_data:
            return {
                'period_days': days,
                'total_profit': 0,
                'avg_daily_profit': 0,
                'total_trades': 0,
                'overall_success_rate': 0
            }
        
        total_profit = sum(d.get('total_profit', 0) or 0 for d in daily_data)
        total_trades = sum(d.get('num_trades', 0) for d in daily_data)
        successful_trades = sum(d.get('successful_trades', 0) for d in daily_data)
        
        return {
            'period_days': len(daily_data),
            'total_profit': total_profit,
            'avg_daily_profit': total_profit / len(daily_data),
            'total_trades': total_trades,
            'overall_success_rate': successful_trades / max(1, total_trades)
        }


