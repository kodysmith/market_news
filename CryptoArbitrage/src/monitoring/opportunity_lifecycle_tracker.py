"""
Opportunity Lifecycle Tracker
Monitors how long arbitrage opportunities stay valid
"""

from typing import Dict, Optional
from datetime import datetime
from decimal import Decimal
import logging

from ..common.models import ArbitrageOpportunity
from ..database.repository import ArbitrageRepository

logger = logging.getLogger(__name__)


class OpportunityLifecycleTracker:
    """
    Tracks the lifecycle of arbitrage opportunities
    
    - Records when opportunities first appear
    - Updates when they're still valid
    - Marks when they expire/close
    - Calculates duration metrics
    """
    
    def __init__(self, repository: ArbitrageRepository):
        """
        Initialize lifecycle tracker
        
        Args:
            repository: Database repository for persistence
        """
        self.repository = repository
        
        # Active opportunities: {opportunity_key: (opportunity, first_seen_timestamp)}
        self.active_opportunities: Dict[str, tuple[ArbitrageOpportunity, datetime]] = {}
        
        # Statistics
        self.total_opportunities_tracked = 0
        self.total_expired = 0
        self.durations_ms = []
        
        logger.info("OpportunityLifecycleTracker initialized")
    
    def track_opportunity(self, opportunity: ArbitrageOpportunity) -> ArbitrageOpportunity:
        """
        Track or update an opportunity
        
        Args:
            opportunity: Newly detected opportunity
            
        Returns:
            Updated opportunity with lifecycle data
        """
        # Create unique key for this opportunity
        opp_key = self._get_opportunity_key(opportunity)
        now = datetime.now()
        
        # Check if this is a new or existing opportunity
        if opp_key in self.active_opportunities:
            # Existing opportunity - update it
            existing_opp, first_seen = self.active_opportunities[opp_key]
            
            # Update timestamps
            opportunity.first_seen_at = first_seen
            opportunity.last_seen_at = now
            opportunity.times_seen = existing_opp.times_seen + 1
            
            # Update in memory
            self.active_opportunities[opp_key] = (opportunity, first_seen)
            
            logger.debug(f"Updated opportunity {opp_key}: seen {opportunity.times_seen} times, "
                        f"duration {(now - first_seen).total_seconds():.2f}s")
        else:
            # New opportunity
            opportunity.first_seen_at = now
            opportunity.last_seen_at = now
            opportunity.times_seen = 1
            
            self.active_opportunities[opp_key] = (opportunity, now)
            self.total_opportunities_tracked += 1
            
            logger.info(f"🆕 New opportunity tracked: {opp_key} - "
                       f"{opportunity.pair} @ {opportunity.net_spread_pct:.2f}%")
        
        # Save to database
        self.repository.save_opportunity(opportunity)
        
        return opportunity
    
    def mark_opportunity_expired(self, opportunity: ArbitrageOpportunity) -> Optional[ArbitrageOpportunity]:
        """
        Mark an opportunity as expired (no longer valid)
        
        Args:
            opportunity: The expired opportunity
            
        Returns:
            Updated opportunity with expiry data
        """
        opp_key = self._get_opportunity_key(opportunity)
        
        if opp_key not in self.active_opportunities:
            return None
        
        existing_opp, first_seen = self.active_opportunities[opp_key]
        now = datetime.now()
        
        # Calculate total duration
        duration_seconds = (now - first_seen).total_seconds()
        duration_ms = duration_seconds * 1000
        
        # Update opportunity
        existing_opp.expired_at = now
        existing_opp.opportunity_duration_ms = duration_ms
        
        # Save to database
        self.repository.save_opportunity(existing_opp)
        
        # Remove from active tracking
        del self.active_opportunities[opp_key]
        
        # Update statistics
        self.total_expired += 1
        self.durations_ms.append(duration_ms)
        
        logger.info(f"⏱️  Opportunity expired: {opp_key} - "
                   f"Duration: {duration_ms:.0f}ms ({duration_seconds:.2f}s), "
                   f"Seen {existing_opp.times_seen} times")
        
        return existing_opp
    
    def check_and_expire_stale_opportunities(self, 
                                            current_opportunities: list[ArbitrageOpportunity],
                                            staleness_threshold_seconds: float = 1.0):
        """
        Check for opportunities that are no longer being detected and mark them as expired
        
        Args:
            current_opportunities: List of currently detected opportunities
            staleness_threshold_seconds: How long to wait before marking as expired
        """
        # Create set of current opportunity keys
        current_keys = {self._get_opportunity_key(opp) for opp in current_opportunities}
        
        # Find opportunities that are no longer present
        now = datetime.now()
        expired_keys = []
        
        for opp_key, (opp, first_seen) in self.active_opportunities.items():
            # If not in current list and last seen was more than threshold ago
            time_since_last_seen = (now - opp.last_seen_at).total_seconds()
            
            if opp_key not in current_keys and time_since_last_seen >= staleness_threshold_seconds:
                expired_keys.append(opp_key)
        
        # Mark expired opportunities
        for opp_key in expired_keys:
            opp, _ = self.active_opportunities[opp_key]
            self.mark_opportunity_expired(opp)
    
    def _get_opportunity_key(self, opportunity: ArbitrageOpportunity) -> str:
        """
        Generate unique key for opportunity based on pair and exchange combination
        
        Args:
            opportunity: The opportunity
            
        Returns:
            Unique key string
        """
        # Key based on pair + exchange pair (direction matters)
        return f"{opportunity.pair}:{opportunity.buy_exchange}->{opportunity.sell_exchange}"
    
    def get_statistics(self) -> Dict:
        """Get lifecycle statistics"""
        active_count = len(self.active_opportunities)
        
        avg_duration_ms = sum(self.durations_ms) / len(self.durations_ms) if self.durations_ms else 0
        min_duration_ms = min(self.durations_ms) if self.durations_ms else 0
        max_duration_ms = max(self.durations_ms) if self.durations_ms else 0
        
        # Calculate percentiles
        if self.durations_ms:
            sorted_durations = sorted(self.durations_ms)
            p50_idx = int(len(sorted_durations) * 0.5)
            p90_idx = int(len(sorted_durations) * 0.9)
            p95_idx = int(len(sorted_durations) * 0.95)
            
            p50_duration = sorted_durations[p50_idx]
            p90_duration = sorted_durations[p90_idx]
            p95_duration = sorted_durations[p95_idx]
        else:
            p50_duration = p90_duration = p95_duration = 0
        
        return {
            'total_tracked': self.total_opportunities_tracked,
            'active_now': active_count,
            'total_expired': self.total_expired,
            'avg_duration_ms': avg_duration_ms,
            'avg_duration_seconds': avg_duration_ms / 1000,
            'min_duration_ms': min_duration_ms,
            'max_duration_ms': max_duration_ms,
            'p50_duration_ms': p50_duration,
            'p90_duration_ms': p90_duration,
            'p95_duration_ms': p95_duration,
            'durations_recorded': len(self.durations_ms)
        }
    
    def get_active_opportunities(self) -> list[ArbitrageOpportunity]:
        """Get list of currently active opportunities"""
        return [opp for opp, _ in self.active_opportunities.values()]


