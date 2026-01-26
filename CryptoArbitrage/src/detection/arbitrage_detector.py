"""
Arbitrage Detector
Identifies profitable arbitrage opportunities across exchanges
"""

from typing import List, Optional, Dict
from decimal import Decimal
from datetime import datetime
import logging
import uuid

from ..common.models import ArbitrageOpportunity, PriceQuote
from ..common.config import StrategyConfig
from ..data.price_aggregator import PriceAggregator

logger = logging.getLogger(__name__)


class ArbitrageDetector:
    """
    Detects arbitrage opportunities in real-time
    
    Scans prices from all exchanges and identifies profitable spreads
    after accounting for fees, slippage, and execution risk.
    """
    
    def __init__(self, 
                 price_aggregator: PriceAggregator,
                 config: StrategyConfig,
                 exchange_fees: Dict[str, Dict[str, Decimal]]):
        """
        Initialize arbitrage detector
        
        Args:
            price_aggregator: Real-time price source
            config: Strategy configuration
            exchange_fees: Fee structure for each exchange
        """
        self.price_aggregator = price_aggregator
        self.config = config
        self.exchange_fees = exchange_fees
        
        # Statistics
        self.opportunities_detected = 0
        self.opportunities_filtered = 0
        
        logger.info("ArbitrageDetector initialized")
        logger.info(f"Min spread threshold: {config.min_spread_pct}%")
    
    def detect_opportunities(self) -> List[ArbitrageOpportunity]:
        """
        Scan all pairs and detect arbitrage opportunities
        
        Returns:
            List of profitable arbitrage opportunities
        """
        opportunities = []
        
        # Get all current prices
        all_prices = self.price_aggregator.get_all_pairs_prices()
        
        for pair, exchange_prices in all_prices.items():
            # Need at least 2 exchanges with valid prices
            if len(exchange_prices) < 2:
                continue
            
            # Find arbitrage opportunity
            opp = self._find_best_spread(pair, exchange_prices)
            
            if opp:
                opportunities.append(opp)
                self.opportunities_detected += 1
        
        return opportunities
    
    def _find_best_spread(self, 
                         pair: str,
                         exchange_prices: Dict[str, PriceQuote]) -> Optional[ArbitrageOpportunity]:
        """
        Find best arbitrage opportunity for a pair
        
        Args:
            pair: Trading pair
            exchange_prices: Prices from different exchanges
        
        Returns:
            ArbitrageOpportunity if profitable, None otherwise
        """
        # Find cheapest buy price (ask) and highest sell price (bid)
        best_buy_exchange = None
        best_buy_price = None
        best_sell_exchange = None
        best_sell_price = None
        
        for exchange_name, quote in exchange_prices.items():
            if not quote:
                continue
            
            # For buying, we pay the ASK price
            if best_buy_price is None or quote.ask < best_buy_price:
                best_buy_price = quote.ask
                best_buy_exchange = exchange_name
            
            # For selling, we receive the BID price
            if best_sell_price is None or quote.bid > best_sell_price:
                best_sell_price = quote.bid
                best_sell_exchange = exchange_name
        
        # Can't arbitrage if same exchange or missing prices
        if (not best_buy_exchange or not best_sell_exchange or 
            best_buy_exchange == best_sell_exchange or
            best_buy_price is None or best_sell_price is None):
            return None
        
        # Calculate spread
        if best_buy_price == 0:
            return None
        
        gross_spread_pct = ((best_sell_price - best_buy_price) / best_buy_price) * Decimal('100')
        
        # Calculate fees for both legs
        buy_fee = self.exchange_fees.get(best_buy_exchange, {}).get('taker_fee', Decimal('0.1'))
        sell_fee = self.exchange_fees.get(best_sell_exchange, {}).get('taker_fee', Decimal('0.1'))
        total_fees_pct = buy_fee + sell_fee
        
        # Net spread after fees
        net_spread_pct = gross_spread_pct - total_fees_pct
        
        # Check if meets minimum threshold
        if net_spread_pct < self.config.min_spread_pct:
            self.opportunities_filtered += 1
            return None
        
        # Calculate max quantity we can trade
        # For now, use configured max position size
        max_quantity = self.config.max_position_size_usd / best_buy_price
        
        # Estimate profit
        estimated_profit = (best_sell_price - best_buy_price) * max_quantity
        estimated_profit -= (best_buy_price * max_quantity * buy_fee / Decimal('100'))
        estimated_profit -= (best_sell_price * max_quantity * sell_fee / Decimal('100'))
        
        # Calculate price staleness (max age of both quotes)
        buy_quote = exchange_prices[best_buy_exchange]
        sell_quote = exchange_prices[best_sell_exchange]
        
        now = datetime.now()
        max_age = max(
            (now - buy_quote.timestamp).total_seconds(),
            (now - sell_quote.timestamp).total_seconds()
        )
        
        # Calculate confidence score
        confidence = self._calculate_confidence(
            net_spread_pct,
            max_age,
            buy_quote.volume_24h or Decimal('0')
        )
        
        # Create opportunity
        opportunity = ArbitrageOpportunity(
            id=f"ARB-{uuid.uuid4().hex[:12]}",
            detected_at=datetime.now(),
            pair=pair,
            base_asset=pair.split('/')[0],
            quote_asset=pair.split('/')[1],
            buy_exchange=best_buy_exchange,
            buy_price=best_buy_price,
            buy_fee_pct=buy_fee,
            sell_exchange=best_sell_exchange,
            sell_price=best_sell_price,
            sell_fee_pct=sell_fee,
            gross_spread_pct=gross_spread_pct,
            fees_total_pct=total_fees_pct,
            net_spread_pct=net_spread_pct,
            max_quantity=max_quantity,
            estimated_profit_usd=estimated_profit,
            price_staleness_ms=max_age * 1000,
            estimated_slippage_pct=Decimal('0.05'),  # Assume 0.05% slippage
            confidence_score=confidence,
            is_executed=False
        )
        
        logger.info(f"🎯 Opportunity detected: {pair} - "
                   f"Buy {best_buy_exchange} @ ${best_buy_price}, "
                   f"Sell {best_sell_exchange} @ ${best_sell_price}, "
                   f"Net: {net_spread_pct:.2f}%")
        
        return opportunity
    
    def _calculate_confidence(self,
                             net_spread_pct: Decimal,
                             price_age_seconds: float,
                             volume_24h: Decimal) -> float:
        """
        Calculate confidence score for opportunity
        
        Args:
            net_spread_pct: Net spread percentage
            price_age_seconds: Age of price data
            volume_24h: 24-hour trading volume
        
        Returns:
            Confidence score 0-1 (higher is better)
        """
        confidence = 1.0
        
        # Reduce confidence for stale prices
        if price_age_seconds > 2:
            confidence *= 0.8
        if price_age_seconds > 3:
            confidence *= 0.6
        
        # Reduce confidence for low volume
        if volume_24h < 1000000:  # < $1M volume
            confidence *= 0.7
        
        # Increase confidence for larger spreads
        if net_spread_pct > Decimal('1.0'):
            confidence *= 1.2
        
        # Cap at 1.0
        return min(confidence, 1.0)
    
    def get_statistics(self) -> Dict:
        """Get detector statistics"""
        return {
            'opportunities_detected': self.opportunities_detected,
            'opportunities_filtered': self.opportunities_filtered,
            'detection_rate': self.opportunities_detected / max(1, self.opportunities_detected + self.opportunities_filtered),
            'price_updates_received': self.price_aggregator.updates_received
        }


