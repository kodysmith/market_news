"""
Decision Cockpit - Single-Screen Trading State View
Provides regime, volatility, structure, and action filter data for instant decision-making.

Enhanced with:
- Multi-lens wall detection (Today/Tactical/Regime)
- Transition state detection
- OPEX-aware analysis
- Dealer-centric GEX interpretation

Design principles:
- One screen, no scrolling
- State > numbers
- Binary where possible
- Permissions, not predictions
- Readable in 3 seconds
"""

import logging
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
import json
import os

logger = logging.getLogger(__name__)


class DecisionCockpit:
    """
    The Decision Cockpit aggregates market state into four bands:
    1. REGIME - GEX state (positive/negative gamma) + transition flag
    2. VOLATILITY - IV direction and state
    3. STRUCTURE - Multi-lens walls (Today/Tactical/Regime) and zones
    4. ACTION FILTER - Allowed/forbidden actions based on regime + transition
    """
    
    # Supported tickers
    SUPPORTED_TICKERS = ['SPY', 'QQQ', 'IWM']
    
    def __init__(self, gex_state: Optional[Dict] = None, volatility_data: Optional[Dict] = None):
        """
        Initialize with optional pre-fetched data.
        
        Args:
            gex_state: Complete GEX cockpit state from compute_cockpit_state()
                       Contains: spot, flip, net_gex, regime, transition, walls_*
            volatility_data: Pre-calculated volatility metrics
        """
        self.gex_state = gex_state or {}
        self.volatility_data = volatility_data or {}
    
    def get_state(self, ticker: str = 'SPY') -> Dict[str, Any]:
        """
        Get the complete cockpit state for a ticker.
        
        Returns:
            Dict with regime, volatility, structure, and action_filter
        """
        ticker = ticker.upper()
        
        # Build each section
        regime = self._get_regime(ticker)
        volatility = self._get_volatility(ticker)
        structure = self._get_structure(ticker)
        action_filter = self._get_action_filter(regime, volatility)
        
        # Get diagnostics from gex_state
        raw_diagnostics = self.gex_state.get('diagnostics', {})
        
        return {
            'ticker': ticker,
            'timestamp': datetime.now().isoformat(),
            'regime': regime,
            'volatility': volatility,
            'structure': structure,
            'action_filter': action_filter,
            'net_series': self.gex_state.get('net_series', []),
            'opex': self.gex_state.get('opex'),
            'contracts_processed': self.gex_state.get('contracts_processed', 0),
            'contracts_total': self.gex_state.get('contracts_total', 0),
            'diagnostics': {
                'calls_in_target': raw_diagnostics.get('calls_in_target'),
                'puts_in_target': raw_diagnostics.get('puts_in_target'),
                'is_symmetric': raw_diagnostics.get('is_symmetric'),
                'target_strikes': raw_diagnostics.get('target_strikes'),
                'missing_from_api': raw_diagnostics.get('missing_from_api'),
                'returned_but_unusable': raw_diagnostics.get('returned_but_unusable'),
                'num_expiries': raw_diagnostics.get('num_expiries')
            }
        }
    
    def _get_regime(self, ticker: str) -> Dict[str, Any]:
        """
        Get the GEX regime state with transition detection.
        
        Returns:
            Dict with label, spot, flip_line, transition, and behavioral bias
        """
        spot = self.gex_state.get('spot', 0)
        flip_line = self.gex_state.get('flip')
        regime = self.gex_state.get('regime', 'UNKNOWN')
        net_gex = self.gex_state.get('net_gex', 0)
        transition_info = self.gex_state.get('transition', {})
        
        # Parse regime string (handles both formats)
        regime_upper = regime.upper() if regime else 'UNKNOWN'
        is_positive = 'POSITIVE' in regime_upper
        is_negative = 'NEGATIVE' in regime_upper
        
        # Determine regime label and bias
        if transition_info.get('transition', False):
            label = 'TRANSITION'
            bias = 'Reduce Size / Wait for Confirmation'
        elif is_positive:
            label = 'POSITIVE GAMMA'
            bias = 'Mean Reversion / Sell Premium'
        elif is_negative:
            label = 'NEGATIVE GAMMA'
            bias = 'Expansion / Follow Breaks'
        else:
            label = 'UNKNOWN'
            bias = 'Wait for Clarity'
        
        # Calculate distance to flip
        if spot and flip_line:
            distance_to_flip = spot - flip_line
            distance_pct = (distance_to_flip / spot) * 100 if spot else 0
        else:
            distance_to_flip = 0
            distance_pct = 0
        
        # Determine flip line status reason
        flip_line_reason = None
        if flip_line is None:
            if is_positive:
                flip_line_reason = "All positive gamma (no zero crossing)"
            elif is_negative:
                flip_line_reason = "All negative gamma (no zero crossing)"
            else:
                flip_line_reason = "Insufficient data"
        
        # Get GEX totals for transparency
        call_gex_total = self.gex_state.get('call_gex_total', 0)
        put_gex_total = self.gex_state.get('put_gex_total', 0)
        
        return {
            'label': label,
            'spot': round(spot, 2) if spot else None,
            'flip_line': round(flip_line, 2) if flip_line else None,
            'flip_line_reason': flip_line_reason,
            'bias': bias,
            'distance_to_flip': round(distance_to_flip, 2),
            'distance_to_flip_pct': round(distance_pct, 2),
            'net_gex': net_gex,
            'call_gex': call_gex_total,
            'put_gex': put_gex_total,
            'gex_ratio': round(abs(call_gex_total / put_gex_total), 2) if put_gex_total != 0 else None,
            'is_positive': is_positive,
            'is_negative': is_negative,
            'transition': transition_info.get('transition', False),
            'transition_reason': transition_info.get('reason')
        }
    
    def _get_volatility(self, ticker: str) -> Dict[str, Any]:
        """
        Get volatility state.
        
        Returns:
            Dict with front_iv, direction, term_structure, and state
        """
        front_iv = self.volatility_data.get('front_iv', 0)
        front_iv_change_1h = self.volatility_data.get('front_iv_change_1h', 0)
        front_iv_change_1d = self.volatility_data.get('front_iv_change_1d', 0)
        back_iv = self.volatility_data.get('back_iv', 0)
        vix = self.volatility_data.get('vix', 0)
        vix_change = self.volatility_data.get('vix_change', 0)
        
        # Determine direction
        if front_iv_change_1h > 0.5:
            direction = 'RISING'
            direction_symbol = '↑'
        elif front_iv_change_1h < -0.5:
            direction = 'FALLING'
            direction_symbol = '↓'
        else:
            direction = 'FLAT'
            direction_symbol = '→'
        
        # Determine term structure
        if front_iv and back_iv:
            iv_diff = front_iv - back_iv
            if iv_diff > 2:
                term_structure = 'INVERTED'  # Front > Back (fear)
            elif iv_diff < -2:
                term_structure = 'CONTANGO'  # Back > Front (normal)
            else:
                term_structure = 'FLAT'
        else:
            term_structure = 'UNKNOWN'
        
        # Determine overall vol state
        if direction == 'RISING' and (term_structure == 'INVERTED' or vix > 25):
            state = 'EXPANDING'
        elif direction == 'FALLING' and vix < 15:
            state = 'CONTRACTING'
        elif vix > 30:
            state = 'ELEVATED'
        elif vix < 12:
            state = 'COMPRESSED'
        else:
            state = 'NORMAL'
        
        return {
            'front_iv': round(front_iv, 2) if front_iv else None,
            'front_iv_change_1h': round(front_iv_change_1h, 2),
            'front_iv_change_1d': round(front_iv_change_1d, 2),
            'direction': direction,
            'direction_symbol': direction_symbol,
            'term_structure': term_structure,
            'state': state,
            'vix': round(vix, 2) if vix else None,
            'vix_change': round(vix_change, 2) if vix_change else None
        }
    
    def _get_structure(self, ticker: str) -> Dict[str, Any]:
        """
        Get market structure with multi-lens walls (GEX + OI).
        
        Returns:
            Dict with:
            - walls_regime/tactical/today: Multi-candidate wall data
            - gex_walls: Primary GEX walls for behavior decisions
            - oi_walls: Primary OI walls for position context
            - Zone metrics and distances
        """
        spot = self.gex_state.get('spot', 0)
        
        # Multi-lens walls (now include both GEX and OI)
        walls_regime = self.gex_state.get('walls_regime', {})
        walls_tactical = self.gex_state.get('walls_tactical', {})
        walls_today = self.gex_state.get('walls_today', {})
        
        # Extract GEX walls (primary for behavior)
        gex_walls = walls_tactical.get('gex_walls', {})
        gex_put_data = gex_walls.get('put', {})
        gex_call_data = gex_walls.get('call', {})
        
        # Extract OI walls (context/validation)
        oi_walls = walls_tactical.get('oi_walls', {})
        oi_put_data = oi_walls.get('put', {})
        oi_call_data = oi_walls.get('call', {})
        
        # Primary GEX walls for distance calculations
        put_wall = gex_put_data.get('primary')
        call_wall = gex_call_data.get('primary')
        
        # Format wall candidates for display
        def format_candidates(candidates):
            """Format candidate list for API response"""
            return [
                {
                    'strike': c['strike'],
                    'value': c['value'],
                    'distance': c['distance_from_spot']
                }
                for c in (candidates or [])
            ]
        
        # Build comprehensive wall structure
        result = {
            # Multi-lens with full GEX + OI data
            'walls_regime': {
                'gex': {
                    'call': gex_call_data if walls_regime.get('gex_walls') else {},
                    'put': gex_put_data if walls_regime.get('gex_walls') else {}
                },
                'oi': {
                    'call': oi_call_data if walls_regime.get('oi_walls') else {},
                    'put': oi_put_data if walls_regime.get('oi_walls') else {}
                }
            } if walls_regime else {},
            'walls_tactical': {
                'gex': {
                    'call': format_candidates(gex_call_data.get('candidates')),
                    'put': format_candidates(gex_put_data.get('candidates'))
                },
                'oi': {
                    'call': format_candidates(oi_call_data.get('candidates')),
                    'put': format_candidates(oi_put_data.get('candidates'))
                }
            },
            'walls_today': {
                'gex': {
                    'call': walls_today.get('gex_walls', {}).get('call', {}),
                    'put': walls_today.get('gex_walls', {}).get('put', {})
                },
                'oi': {
                    'call': walls_today.get('oi_walls', {}).get('call', {}),
                    'put': walls_today.get('oi_walls', {}).get('put', {})
                }
            } if walls_today else {},
            
            # Primary walls for quick reference (GEX-based)
            'primary_walls': {
                'call': call_wall,
                'put': put_wall,
                'type': 'GEX'  # Indicates these are GEX walls
            },
            
            # OI walls for validation
            'oi_primary_walls': {
                'call': oi_call_data.get('primary'),
                'put': oi_put_data.get('primary'),
                'type': 'OI'
            }
        }
        
        if not spot or not put_wall or not call_wall:
            result.update({
                'no_trade_zone': None,
                'in_no_trade_zone': False,
                'distance_to_put_wall': None,
                'distance_to_call_wall': None,
                'distance_to_nearest': None,
                'nearest_wall': None,
                'wall_range': None,
                'position_in_range_pct': None,
                'wall_agreement': None
            })
            return result
        
        # Calculate distances (using GEX walls for behavior)
        dist_to_put = spot - put_wall
        dist_to_call = call_wall - spot
        wall_range = call_wall - put_wall
        
        # Distance to nearest wall
        distance_to_nearest = min(abs(dist_to_put), abs(dist_to_call))
        nearest_wall = 'PUT' if abs(dist_to_put) < abs(dist_to_call) else 'CALL'
        
        # Position in range (0 = at put wall, 100 = at call wall)
        if wall_range > 0:
            position_pct = ((spot - put_wall) / wall_range) * 100
        else:
            position_pct = 50
        
        # No-trade zone: when walls are tight and spot is in the middle
        if wall_range > 0 and wall_range < spot * 0.02:  # Walls within 2% of spot
            zone_start = put_wall + (wall_range * 0.2)
            zone_end = call_wall - (wall_range * 0.2)
            no_trade_zone = [round(zone_start, 2), round(zone_end, 2)]
            in_no_trade_zone = zone_start <= spot <= zone_end
        else:
            no_trade_zone = None
            in_no_trade_zone = False
        
        # Check if GEX and OI walls agree (validation)
        oi_put = oi_put_data.get('primary')
        oi_call = oi_call_data.get('primary')
        wall_agreement = {
            'put_agrees': put_wall == oi_put if oi_put else None,
            'call_agrees': call_wall == oi_call if oi_call else None,
            'note': None
        }
        if not wall_agreement['put_agrees'] and oi_put:
            wall_agreement['note'] = f"GEX put wall ({put_wall}) differs from OI wall ({oi_put})"
        if not wall_agreement['call_agrees'] and oi_call:
            note = f"GEX call wall ({call_wall}) differs from OI wall ({oi_call})"
            wall_agreement['note'] = note if not wall_agreement['note'] else wall_agreement['note'] + "; " + note
        
        result.update({
            'no_trade_zone': no_trade_zone,
            'in_no_trade_zone': in_no_trade_zone,
            'distance_to_put_wall': round(dist_to_put, 2),
            'distance_to_call_wall': round(dist_to_call, 2),
            'distance_to_nearest': round(distance_to_nearest, 2),
            'nearest_wall': nearest_wall,
            'wall_range': round(wall_range, 2),
            'position_in_range_pct': round(position_pct, 1),
            'wall_agreement': wall_agreement,
            
            # Behavioral labels for each level
            'level_labels': self._get_level_labels(
                spot, put_wall, call_wall,
                gex_put_data.get('candidates', []),
                gex_call_data.get('candidates', []),
                oi_put_data.get('candidates', []),
                oi_call_data.get('candidates', []),
                self.gex_state.get('flip')
            )
        })
        
        return result
    
    def _get_level_labels(
        self,
        spot: float,
        put_wall: Optional[float],
        call_wall: Optional[float],
        gex_put_candidates: List[Dict],
        gex_call_candidates: List[Dict],
        oi_put_candidates: List[Dict],
        oi_call_candidates: List[Dict],
        flip_line: Optional[float]
    ) -> Dict[str, Any]:
        """
        Generate behavioral labels for key price levels.
        
        This is where the product becomes elite - actionable cognition, not just numbers.
        
        Args:
            spot: Current spot price
            put_wall: Primary put wall (GEX)
            call_wall: Primary call wall (GEX)
            gex_put_candidates: List of put GEX wall candidates
            gex_call_candidates: List of call GEX wall candidates
            oi_put_candidates: List of put OI wall candidates
            oi_call_candidates: List of call OI wall candidates
            flip_line: GEX flip line price
        
        Returns:
            Dict with labeled levels and behavioral descriptions
        """
        labels = {
            'put_levels': [],
            'call_levels': [],
            'flip_level': None,
            'zone_description': None
        }
        
        # Label put levels (support)
        if gex_put_candidates:
            primary = gex_put_candidates[0]
            primary_strike = primary.get('strike')
            
            # Check if OI agrees
            oi_primary = oi_put_candidates[0].get('strike') if oi_put_candidates else None
            
            label = "Structural support / hedging floor"
            if primary_strike and oi_primary:
                if abs(primary_strike - spot) < 1:
                    label = "Near-ATM support / pin risk"
                elif primary_strike == oi_primary:
                    label = "Strong support (GEX + OI aligned)"
                else:
                    label = "Hedging support (GEX-based)"
            
            labels['put_levels'].append({
                'strike': primary_strike,
                'type': 'primary',
                'source': 'GEX',
                'label': label,
                'distance': round(spot - primary_strike, 2) if primary_strike else None
            })
            
            # Add secondary levels
            for i, candidate in enumerate(gex_put_candidates[1:3], 1):
                strike = candidate.get('strike')
                if strike:
                    labels['put_levels'].append({
                        'strike': strike,
                        'type': 'secondary',
                        'source': 'GEX',
                        'label': f"Secondary support #{i}",
                        'distance': round(spot - strike, 2)
                    })
            
            # Add OI-based levels if different
            if oi_put_candidates and oi_primary != primary_strike:
                labels['put_levels'].append({
                    'strike': oi_primary,
                    'type': 'oi_reference',
                    'source': 'OI',
                    'label': "Position concentration (OI)",
                    'distance': round(spot - oi_primary, 2) if oi_primary else None
                })
        
        # Label call levels (resistance)
        if gex_call_candidates:
            primary = gex_call_candidates[0]
            primary_strike = primary.get('strike')
            
            oi_primary = oi_call_candidates[0].get('strike') if oi_call_candidates else None
            
            label = "Dealer resistance / acceleration trigger"
            if primary_strike and oi_primary:
                if abs(primary_strike - spot) < 1:
                    label = "Near-ATM resistance / pin risk"
                elif primary_strike == oi_primary:
                    label = "Strong resistance (GEX + OI aligned)"
                else:
                    label = "Hedging resistance (GEX-based)"
            
            labels['call_levels'].append({
                'strike': primary_strike,
                'type': 'primary',
                'source': 'GEX',
                'label': label,
                'distance': round(primary_strike - spot, 2) if primary_strike else None
            })
            
            # Add secondary levels
            for i, candidate in enumerate(gex_call_candidates[1:3], 1):
                strike = candidate.get('strike')
                if strike:
                    labels['call_levels'].append({
                        'strike': strike,
                        'type': 'secondary',
                        'source': 'GEX',
                        'label': f"Secondary resistance #{i}",
                        'distance': round(strike - spot, 2)
                    })
            
            # Add OI-based levels if different
            if oi_call_candidates and oi_primary != primary_strike:
                labels['call_levels'].append({
                    'strike': oi_primary,
                    'type': 'oi_reference',
                    'source': 'OI',
                    'label': "Position concentration (OI)",
                    'distance': round(oi_primary - spot, 2) if oi_primary else None
                })
        
        # Label flip line
        if flip_line:
            flip_distance = flip_line - spot
            if abs(flip_distance) < 2:
                flip_label = "⚠️ CRITICAL: Regime decision level (near spot)"
            elif flip_distance > 0:
                flip_label = "Trend decision level (above spot → breakout target)"
            else:
                flip_label = "Trend decision level (below spot → breakdown risk)"
            
            labels['flip_level'] = {
                'strike': round(flip_line, 2),
                'label': flip_label,
                'distance': round(flip_distance, 2),
                'significance': "Crossing this level changes dealer behavior"
            }
        
        # Zone description (actionable summary)
        if put_wall and call_wall and spot:
            range_size = call_wall - put_wall
            
            if range_size < 5:
                labels['zone_description'] = {
                    'type': 'TIGHT',
                    'summary': f"Tight range ({put_wall}-{call_wall}): Pin/chop expected",
                    'behavior': "No fading, wait for wall break"
                }
            elif flip_line and put_wall < flip_line < call_wall:
                labels['zone_description'] = {
                    'type': 'FLIP_IN_RANGE',
                    'summary': f"Flip line ({flip_line:.0f}) within range: Unstable",
                    'behavior': "Directional bias, trend following, no mean reversion"
                }
            elif flip_line and spot < flip_line:
                labels['zone_description'] = {
                    'type': 'BELOW_FLIP',
                    'summary': f"Below flip ({flip_line:.0f}): Negative gamma zone",
                    'behavior': "Expect acceleration, don't fight the trend"
                }
            else:
                labels['zone_description'] = {
                    'type': 'ABOVE_FLIP',
                    'summary': f"Above flip: Positive gamma zone",
                    'behavior': "Mean reversion, sell premium, fade extremes"
                }
        
        return labels
    
    def _get_action_filter(self, regime: Dict, volatility: Dict) -> Dict[str, List[str]]:
        """
        Get allowed and forbidden actions based on regime, transition, and volatility.
        
        This is the decision engine - maps state to permissions.
        Updated to handle transition states specially.
        """
        is_positive = regime.get('is_positive', False)
        is_transition = regime.get('transition', False)
        vol_state = volatility.get('state', 'NORMAL')
        vol_direction = volatility.get('direction', 'FLAT')
        term_structure = volatility.get('term_structure', 'FLAT')
        
        allowed = []
        forbidden = []
        
        # TRANSITION rules (highest priority)
        if is_transition:
            allowed = [
                'Small size only',
                'Defined-risk only',
                'Avoid new short premium',
                'Wait for regime confirmation'
            ]
            forbidden = [
                'Aggressive sizing',
                'Tight strikes',
                'Unhedged positions',
                'New premium selling',
                'Directional bets'
            ]
            return {'allowed': allowed, 'forbidden': forbidden}
        
        # NEGATIVE GAMMA rules
        if not is_positive:
            if vol_state in ['EXPANDING', 'ELEVATED']:
                # Negative gamma + expanding vol = trend following
                allowed = [
                    'Convex trades (debit spreads)',
                    'Calendars/diagonals (long vol)',
                    'Directional AFTER breaks',
                    'Defined-risk structures'
                ]
                forbidden = [
                    'Tight condors',
                    'Fade trades',
                    'Naked short premium',
                    '0DTE index trades'
                ]
            else:
                # Negative gamma + normal/contracting vol
                allowed = [
                    'Directional after confirmation',
                    'Wide spreads',
                    'Reduced position size',
                    'Wait for wall test'
                ]
                forbidden = [
                    'Short premium near walls',
                    'Mean reversion trades',
                    'Averaging down',
                    'Full position size'
                ]
        
        # POSITIVE GAMMA rules
        else:
            if vol_state in ['COMPRESSED', 'CONTRACTING']:
                # Positive gamma + low vol = sell premium
                allowed = [
                    'Sell premium (wide)',
                    'Mean reversion with stops',
                    'Fade moves to walls',
                    'Cash-secured puts',
                    'Iron condors'
                ]
                forbidden = [
                    'Breakout chasing',
                    'Trend following',
                    'Long premium',
                    'Overly tight strikes'
                ]
            else:
                # Positive gamma + normal/elevated vol = cautious selling
                allowed = [
                    'Sell premium with wide strikes',
                    'Cash-secured puts',
                    'Mean reversion with stops',
                    'Reduced size premium selling'
                ]
                forbidden = [
                    'Aggressive short premium',
                    'Tight strikes',
                    'Breakout plays',
                    'Undefined risk'
                ]
        
        # Unknown state
        if regime.get('label') == 'UNKNOWN':
            allowed = ['Wait for clarity', 'Paper trade only']
            forbidden = ['All real trades until regime confirmed']
        
        return {
            'allowed': allowed,
            'forbidden': forbidden
        }


def get_cockpit_state(
    ticker: str,
    gex_state: Optional[Dict] = None,
    volatility_data: Optional[Dict] = None
) -> Dict[str, Any]:
    """
    Convenience function to get cockpit state.
    
    Args:
        ticker: Stock ticker (SPY, QQQ, IWM)
        gex_state: Complete GEX cockpit state from compute_cockpit_state()
        volatility_data: Pre-calculated volatility data
    
    Returns:
        Complete cockpit state dict
    """
    cockpit = DecisionCockpit(gex_state, volatility_data)
    return cockpit.get_state(ticker)
