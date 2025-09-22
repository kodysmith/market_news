"""
Alert management and deduplication logic
"""

from typing import List, Dict, Any, Optional
from database import DatabaseManager
from spreads import BullPutSpread

class AlertManager:
    """Manage alert sending and deduplication"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
    
    def decide_alerts(self, spreads: List[BullPutSpread], 
                     min_ev_threshold: float = 0.10,
                     min_pop_threshold: float = 0.50) -> List[BullPutSpread]:
        """
        Decide which spreads should trigger alerts based on dedupe rules
        
        Args:
            spreads: List of bull put spreads
            min_ev_threshold: Minimum EV threshold for alerts
            min_pop_threshold: Minimum POP threshold for alerts
        
        Returns:
            List of spreads that should trigger alerts
        """
        alerts_to_send = []
        
        for spread in spreads:
            # Check basic thresholds
            if spread.ev < min_ev_threshold or spread.pop < min_pop_threshold:
                continue
            
            # Check if we should send alert based on dedupe rules
            if self.db_manager.should_send_alert(spread.spread_id, spread.ev):
                alerts_to_send.append(spread)
        
        return alerts_to_send
    
    def mark_alert_sent(self, spread: BullPutSpread):
        """Mark an alert as sent in the database"""
        self.db_manager.upsert_pushed_alert(spread.spread_id, spread.ev)
    
    def get_alert_summary(self, spreads: List[BullPutSpread]) -> Dict[str, Any]:
        """Get summary of alert decisions"""
        total_spreads = len(spreads)
        high_ev_spreads = [s for s in spreads if s.ev >= 0.10]
        high_pop_spreads = [s for s in spreads if s.pop >= 0.50]
        
        return {
            'total_spreads': total_spreads,
            'high_ev_spreads': len(high_ev_spreads),
            'high_pop_spreads': len(high_pop_spreads),
            'avg_ev': sum(s.ev for s in spreads) / len(spreads) if spreads else 0,
            'avg_pop': sum(s.pop for s in spreads) / len(spreads) if spreads else 0,
            'max_ev': max(s.ev for s in spreads) if spreads else 0,
            'max_pop': max(s.pop for s in spreads) if spreads else 0
        }
    
    def format_alert_message(self, spread: BullPutSpread) -> Dict[str, str]:
        """Format alert message for FCM"""
        title = f"New positive-EV bull put on {spread.ticker}"
        body = (f"POP {spread.pop:.0%}, EV ${spread.ev:.2f}, "
                f"credit ${spread.credit:.2f}, {spread.dte} DTE "
                f"({spread.short_strike:.0f}/{spread.long_strike:.0f})")
        
        data = {
            'type': 'opportunity',
            'id': spread.spread_id,
            'ticker': spread.ticker,
            'strategy': 'BULL_PUT',
            'expiry': spread.expiry,
            'shortK': str(spread.short_strike),
            'longK': str(spread.long_strike)
        }
        
        return {
            'title': title,
            'body': body,
            'data': data
        }
    
    def get_alert_stats(self) -> Dict[str, Any]:
        """Get alert statistics from database"""
        # This would query the database for alert statistics
        # For now, return basic info
        return {
            'total_alerts_sent': 0,  # Would query pushed_alerts table
            'unique_spreads_alerted': 0,  # Would count unique spread IDs
            'avg_ev_alerted': 0.0,  # Would calculate from spread_snapshots
            'last_alert_time': None  # Would get from most recent pushed_alert
        }

