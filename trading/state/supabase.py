"""
Supabase State Store

Manages all database operations for paper and live trading.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
import logging
from .interface import StateStoreInterface

logger = logging.getLogger(__name__)


class SupabaseStateStore(StateStoreInterface):
    """Manages state in Supabase database"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.supabase_url = config.get('supabase_url')
        self.supabase_key = config.get('supabase_key')

        # Initialize Supabase client
        try:
            from supabase import create_client, Client
            self.client: Client = create_client(self.supabase_url, self.supabase_key)
        except ImportError:
            logger.error("supabase-py not installed. Install with: pip install supabase")
            self.client = None
        except Exception as e:
            logger.error(f"Error initializing Supabase client: {e}")
            self.client = None

    def save_run(self, run_data: Dict[str, Any]) -> str:
        """Save a trading run to database"""
        if not self.client:
            raise ValueError("Supabase client not initialized")

        try:
            # Prepare run data
            run_record = {
                'mode': run_data.get('mode', 'paper'),
                'status': run_data.get('status', 'running'),
                'portfolio_value': run_data.get('portfolio_value', 0),
                'cash': run_data.get('cash', 0),
                'started_at': run_data.get('started_at', datetime.now().isoformat()),
                'completed_at': run_data.get('completed_at'),
                'regime': run_data.get('regime', {}),
                'config': run_data.get('config', {})
            }

            result = self.client.table('runs').insert(run_record).execute()
            run_id = result.data[0]['id'] if result.data else None

            logger.info(f"Saved run {run_id} to Supabase")
            return run_id

        except Exception as e:
            logger.error(f"Error saving run to Supabase: {e}")
            raise

    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Get run by ID"""
        if not self.client:
            return None

        try:
            result = self.client.table('runs').select('*').eq('id', run_id).execute()
            return result.data[0] if result.data else None
        except Exception as e:
            logger.error(f"Error getting run from Supabase: {e}")
            return None

    def update_run(self, run_id: str, updates: Dict[str, Any]):
        """Update run with new data"""
        if not self.client:
            return

        try:
            self.client.table('runs').update(updates).eq('id', run_id).execute()
        except Exception as e:
            logger.error(f"Error updating run in Supabase: {e}")

    def save_positions(self, positions: Dict[str, float], run_id: str):
        """Save positions for a run"""
        if not self.client:
            return

        try:
            # Delete existing positions for this run
            self.client.table('positions').delete().eq('run_id', run_id).execute()

            # Insert new positions
            position_records = []
            for ticker, value in positions.items():
                position_records.append({
                    'run_id': run_id,
                    'ticker': ticker,
                    'value': value,
                    'updated_at': datetime.now().isoformat()
                })

            if position_records:
                self.client.table('positions').insert(position_records).execute()

        except Exception as e:
            logger.error(f"Error saving positions to Supabase: {e}")

    def get_positions(self, run_id: str) -> List[Dict[str, Any]]:
        """Get positions for a run"""
        if not self.client:
            return []

        try:
            result = self.client.table('positions').select('*').eq('run_id', run_id).execute()
            return result.data if result.data else []
        except Exception as e:
            logger.error(f"Error getting positions from Supabase: {e}")
            return []

    def update_position(self, run_id: str, ticker: str, quantity: float, price: float):
        """Update a position (add or subtract quantity)"""
        if not self.client:
            return

        try:
            # Get current position
            result = self.client.table('positions').select('*').eq('run_id', run_id).eq('ticker', ticker).execute()

            if result.data:
                # Update existing position
                current = result.data[0]
                new_quantity = current.get('quantity', 0) + quantity
                new_value = new_quantity * price

                if new_quantity == 0:
                    # Delete if zero
                    self.client.table('positions').delete().eq('id', current['id']).execute()
                else:
                    # Update
                    self.client.table('positions').update({
                        'quantity': new_quantity,
                        'value': new_value,
                        'updated_at': datetime.now().isoformat()
                    }).eq('id', current['id']).execute()
            else:
                # Create new position
                if quantity != 0:
                    self.client.table('positions').insert({
                        'run_id': run_id,
                        'ticker': ticker,
                        'quantity': quantity,
                        'value': quantity * price,
                        'updated_at': datetime.now().isoformat()
                    }).execute()

        except Exception as e:
            logger.error(f"Error updating position in Supabase: {e}")

    def create_order(self, order_data: Dict[str, Any]) -> str:
        """Create an order record"""
        if not self.client:
            return None

        try:
            order_record = {
                'run_id': order_data.get('run_id'),
                'ticker': order_data.get('ticker'),
                'quantity': order_data.get('quantity'),
                'order_type': order_data.get('order_type', 'market'),
                'fill_price': order_data.get('fill_price'),
                'commission': order_data.get('commission', 0),
                'status': order_data.get('status', 'pending'),
                'created_at': datetime.now().isoformat()
            }

            result = self.client.table('orders').insert(order_record).execute()
            return result.data[0]['id'] if result.data else None

        except Exception as e:
            logger.error(f"Error creating order in Supabase: {e}")
            return None

    def save_regime_history(self, run_id: str, regime: Dict[str, Any]):
        """Save regime detection result"""
        if not self.client:
            return

        try:
            regime_record = {
                'run_id': run_id,
                'regime': regime.get('regime'),
                'confidence': regime.get('confidence'),
                'indicators': regime.get('indicators', {}),
                'timestamp': regime.get('timestamp', datetime.now().isoformat())
            }

            self.client.table('regime_history').insert(regime_record).execute()

        except Exception as e:
            logger.error(f"Error saving regime history to Supabase: {e}")
