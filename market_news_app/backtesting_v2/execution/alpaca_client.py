#!/usr/bin/env python3
"""
Alpaca Trading API Client

Paper and live trading execution through Alpaca API.
Handles order placement, position management, and risk controls.
"""

import alpaca_trade_api as tradeapi
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AlpacaExecutionClient:
    """Alpaca trading API client for order execution"""

    def __init__(self,
                 api_key: str = None,
                 api_secret: str = None,
                 base_url: str = None,
                 paper: bool = True):
        """
        Initialize Alpaca client

        Args:
            api_key: Alpaca API key
            api_secret: Alpaca API secret
            base_url: API base URL (None for default)
            paper: Whether to use paper trading
        """
        self.api_key = api_key or os.getenv('ALPACA_API_KEY')
        self.api_secret = api_secret or os.getenv('ALPACA_API_SECRET')

        if not self.api_key or not self.api_secret:
            raise ValueError("Alpaca API credentials not provided")

        # Set base URL for paper/live trading
        if base_url is None:
            if paper:
                base_url = 'https://paper-api.alpaca.markets'
            else:
                base_url = 'https://api.alpaca.markets'

        self.api = tradeapi.REST(
            key_id=self.api_key,
            secret_key=self.api_secret,
            base_url=base_url
        )

        self.paper = paper
        self.account = None

        # Initialize account info
        self._refresh_account()

        logger.info(f"Alpaca client initialized ({'Paper' if paper else 'Live'} trading)")

    def _refresh_account(self):
        """Refresh account information"""
        try:
            self.account = self.api.get_account()
        except Exception as e:
            logger.error(f"Failed to refresh account: {e}")
            self.account = None

    def get_account_info(self) -> Dict[str, Any]:
        """Get current account information"""
        if self.account is None:
            self._refresh_account()

        if self.account is None:
            return {}

        return {
            'equity': float(self.account.equity),
            'cash': float(self.account.cash),
            'buying_power': float(self.account.buying_power),
            'daytrade_count': int(self.account.daytrade_count),
            'portfolio_value': float(self.account.portfolio_value),
            'status': self.account.status
        }

    def get_positions(self) -> Dict[str, Dict[str, Any]]:
        """Get current positions"""
        try:
            positions = self.api.list_positions()
            return {
                pos.symbol: {
                    'qty': float(pos.qty),
                    'avg_entry_price': float(pos.avg_entry_price),
                    'market_value': float(pos.market_value),
                    'unrealized_pl': float(pos.unrealized_pl),
                    'unrealized_plpc': float(pos.unrealized_plpc),
                    'current_price': float(pos.current_price)
                }
                for pos in positions
            }
        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            return {}

    def get_current_price(self, symbol: str) -> float:
        """Get current market price for a symbol"""
        try:
            quote = self.api.get_latest_quote(symbol)
            return float(quote.askprice)  # Use ask price for buying
        except Exception as e:
            logger.error(f"Failed to get price for {symbol}: {e}")
            return 0.0

    def place_market_order(self,
                          symbol: str,
                          qty: float,
                          side: str,
                          time_in_force: str = 'day') -> Optional[Dict[str, Any]]:
        """
        Place a market order

        Args:
            symbol: Stock symbol
            qty: Quantity (positive for buy, negative for sell)
            side: 'buy' or 'sell'
            time_in_force: Time in force ('day', 'gtc', etc.)

        Returns:
            Order information dictionary
        """
        try:
            order = self.api.submit_order(
                symbol=symbol,
                qty=abs(qty),
                side=side,
                type='market',
                time_in_force=time_in_force
            )

            order_info = {
                'id': order.id,
                'symbol': order.symbol,
                'qty': float(order.qty),
                'side': order.side,
                'type': order.type,
                'status': order.status,
                'submitted_at': order.submitted_at,
                'filled_qty': float(order.filled_qty or 0),
                'filled_avg_price': float(order.filled_avg_price or 0)
            }

            logger.info(f"Placed {side} order for {qty} {symbol} (ID: {order.id})")
            return order_info

        except Exception as e:
            logger.error(f"Failed to place market order: {e}")
            return None

    def place_limit_order(self,
                         symbol: str,
                         qty: float,
                         side: str,
                         limit_price: float,
                         time_in_force: str = 'day') -> Optional[Dict[str, Any]]:
        """
        Place a limit order

        Args:
            symbol: Stock symbol
            qty: Quantity
            side: 'buy' or 'sell'
            limit_price: Limit price
            time_in_force: Time in force

        Returns:
            Order information dictionary
        """
        try:
            order = self.api.submit_order(
                symbol=symbol,
                qty=abs(qty),
                side=side,
                type='limit',
                limit_price=limit_price,
                time_in_force=time_in_force
            )

            order_info = {
                'id': order.id,
                'symbol': order.symbol,
                'qty': float(order.qty),
                'side': order.side,
                'type': order.type,
                'limit_price': limit_price,
                'status': order.status,
                'submitted_at': order.submitted_at
            }

            logger.info(f"Placed {side} limit order for {qty} {symbol} at ${limit_price} (ID: {order.id})")
            return order_info

        except Exception as e:
            logger.error(f"Failed to place limit order: {e}")
            return None

    def place_stop_order(self,
                        symbol: str,
                        qty: float,
                        side: str,
                        stop_price: float,
                        time_in_force: str = 'gtc') -> Optional[Dict[str, Any]]:
        """
        Place a stop order

        Args:
            symbol: Stock symbol
            qty: Quantity
            side: 'buy' or 'sell'
            stop_price: Stop price
            time_in_force: Time in force

        Returns:
            Order information dictionary
        """
        try:
            order = self.api.submit_order(
                symbol=symbol,
                qty=abs(qty),
                side=side,
                type='stop',
                stop_price=stop_price,
                time_in_force=time_in_force
            )

            order_info = {
                'id': order.id,
                'symbol': order.symbol,
                'qty': float(order.qty),
                'side': order.side,
                'type': order.type,
                'stop_price': stop_price,
                'status': order.status,
                'submitted_at': order.submitted_at
            }

            logger.info(f"Placed {side} stop order for {qty} {symbol} at ${stop_price} (ID: {order.id})")
            return order_info

        except Exception as e:
            logger.error(f"Failed to place stop order: {e}")
            return None

    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an order

        Args:
            order_id: Order ID to cancel

        Returns:
            True if successful
        """
        try:
            self.api.cancel_order(order_id)
            logger.info(f"Cancelled order {order_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return False

    def cancel_all_orders(self) -> int:
        """
        Cancel all open orders

        Returns:
            Number of orders cancelled
        """
        try:
            orders = self.api.list_orders(status='open')
            cancelled_count = 0

            for order in orders:
                try:
                    self.api.cancel_order(order.id)
                    cancelled_count += 1
                except:
                    continue

            logger.info(f"Cancelled {cancelled_count} orders")
            return cancelled_count

        except Exception as e:
            logger.error(f"Failed to cancel all orders: {e}")
            return 0

    def get_order_status(self, order_id: str) -> Optional[Dict[str, Any]]:
        """
        Get order status

        Args:
            order_id: Order ID

        Returns:
            Order status information
        """
        try:
            order = self.api.get_order(order_id)
            return {
                'id': order.id,
                'status': order.status,
                'filled_qty': float(order.filled_qty or 0),
                'filled_avg_price': float(order.filled_avg_price or 0),
                'qty': float(order.qty),
                'side': order.side,
                'symbol': order.symbol
            }
        except Exception as e:
            logger.error(f"Failed to get order status for {order_id}: {e}")
            return None

    def get_open_orders(self) -> List[Dict[str, Any]]:
        """Get all open orders"""
        try:
            orders = self.api.list_orders(status='open')
            return [{
                'id': order.id,
                'symbol': order.symbol,
                'qty': float(order.qty),
                'side': order.side,
                'type': order.type,
                'status': order.status,
                'submitted_at': order.submitted_at
            } for order in orders]
        except Exception as e:
            logger.error(f"Failed to get open orders: {e}")
            return []

    def execute_portfolio_trades(self,
                                target_weights: Dict[str, float],
                                portfolio_value: float = None) -> List[Dict[str, Any]]:
        """
        Execute trades to achieve target portfolio weights

        Args:
            target_weights: Target portfolio weights (symbol -> weight)
            portfolio_value: Portfolio value to use (default: current equity)

        Returns:
            List of executed orders
        """
        if portfolio_value is None:
            account_info = self.get_account_info()
            portfolio_value = account_info.get('equity', 0)

        if portfolio_value <= 0:
            logger.error("Invalid portfolio value")
            return []

        current_positions = self.get_positions()
        executed_orders = []

        for symbol, target_weight in target_weights.items():
            try:
                target_value = portfolio_value * target_weight
                current_qty = current_positions.get(symbol, {}).get('qty', 0)
                current_price = self.get_current_price(symbol)

                if current_price <= 0:
                    logger.warning(f"Could not get price for {symbol}")
                    continue

                target_qty = target_value / current_price
                qty_diff = target_qty - current_qty

                # Minimum order size and rounding
                if abs(qty_diff) * current_price < 10:  # Minimum $10 order
                    continue

                qty_diff = round(qty_diff)

                if qty_diff > 0:  # Buy
                    order = self.place_market_order(symbol, qty_diff, 'buy')
                elif qty_diff < 0:  # Sell
                    order = self.place_market_order(symbol, abs(qty_diff), 'sell')
                else:
                    continue

                if order:
                    executed_orders.append(order)

                time.sleep(0.1)  # Rate limiting

            except Exception as e:
                logger.error(f"Failed to execute trade for {symbol}: {e}")
                continue

        logger.info(f"Executed {len(executed_orders)} portfolio rebalancing orders")
        return executed_orders

    def get_portfolio_history(self, days: int = 30) -> pd.DataFrame:
        """
        Get portfolio value history

        Args:
            days: Number of days of history

        Returns:
            DataFrame with portfolio history
        """
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)

            portfolio_history = self.api.get_portfolio_history(
                date_start=start_date.strftime('%Y-%m-%d'),
                date_end=end_date.strftime('%Y-%m-%d'),
                timeframe='1D'
            )

            if portfolio_history.equity:
                df = pd.DataFrame({
                    'timestamp': portfolio_history.timestamp,
                    'equity': portfolio_history.equity,
                    'profit_loss': portfolio_history.profit_loss,
                    'profit_loss_pct': portfolio_history.profit_loss_pct
                })
                df['date'] = pd.to_datetime(df['timestamp'], unit='s')
                return df.set_index('date')
            else:
                return pd.DataFrame()

        except Exception as e:
            logger.error(f"Failed to get portfolio history: {e}")
            return pd.DataFrame()

    def calculate_portfolio_metrics(self, days: int = 30) -> Dict[str, float]:
        """
        Calculate portfolio performance metrics

        Args:
            days: Lookback period in days

        Returns:
            Dictionary with performance metrics
        """
        history_df = self.get_portfolio_history(days)

        if history_df.empty:
            return {}

        # Calculate returns
        history_df['returns'] = history_df['equity'].pct_change()

        # Basic metrics
        total_return = (history_df['equity'].iloc[-1] / history_df['equity'].iloc[0]) - 1
        volatility = history_df['returns'].std() * np.sqrt(252)
        sharpe_ratio = history_df['returns'].mean() / history_df['returns'].std() * np.sqrt(252) if history_df['returns'].std() > 0 else 0

        # Drawdown
        cumulative = (1 + history_df['returns']).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()

        return {
            'total_return': float(total_return),
            'volatility': float(volatility),
            'sharpe_ratio': float(sharpe_ratio),
            'max_drawdown': float(max_drawdown),
            'current_equity': float(history_df['equity'].iloc[-1])
        }

class AlpacaRiskManager:
    """Risk management for Alpaca trading"""

    def __init__(self, client: AlpacaExecutionClient):
        self.client = client

    def check_position_limits(self,
                            symbol: str,
                            proposed_qty: float,
                            max_position_size: float = 0.10) -> Tuple[bool, str]:
        """
        Check if proposed position violates size limits

        Args:
            symbol: Stock symbol
            proposed_qty: Proposed quantity
            max_position_size: Maximum position size as fraction of portfolio

        Returns:
            Tuple of (allowed, reason)
        """
        account_info = self.client.get_account_info()
        portfolio_value = account_info.get('equity', 0)

        if portfolio_value <= 0:
            return False, "Invalid portfolio value"

        current_price = self.client.get_current_price(symbol)
        if current_price <= 0:
            return False, f"Could not get price for {symbol}"

        position_value = abs(proposed_qty) * current_price
        position_fraction = position_value / portfolio_value

        if position_fraction > max_position_size:
            return False, ".2%"

        return True, "OK"

    def check_daily_trade_limit(self, additional_trades: int = 1) -> Tuple[bool, str]:
        """
        Check Pattern Day Trading rule compliance

        Args:
            additional_trades: Number of additional trades planned

        Returns:
            Tuple of (allowed, reason)
        """
        account_info = self.client.get_account_info()
        daytrade_count = account_info.get('daytrade_count', 0)

        if daytrade_count + additional_trades > 3:
            return False, f"Would exceed PDT limit (current: {daytrade_count}, adding: {additional_trades})"

        return True, "OK"

    def validate_order(self,
                      symbol: str,
                      qty: float,
                      side: str,
                      order_type: str = 'market',
                      price: float = None) -> Tuple[bool, str]:
        """
        Validate an order before submission

        Args:
            symbol: Stock symbol
            qty: Quantity
            side: 'buy' or 'sell'
            order_type: Order type
            price: Price (for limit orders)

        Returns:
            Tuple of (valid, reason)
        """
        # Check account status
        account_info = self.client.get_account_info()
        if account_info.get('status') != 'ACTIVE':
            return False, f"Account not active: {account_info.get('status')}"

        # Check buying power for buys
        if side == 'buy':
            cost = qty * (price or self.client.get_current_price(symbol))
            buying_power = account_info.get('buying_power', 0)

            if cost > buying_power:
                return False, ".2f"

        # Check position exists for sells
        elif side == 'sell':
            positions = self.client.get_positions()
            current_qty = positions.get(symbol, {}).get('qty', 0)

            if abs(qty) > current_qty:
                return False, ".0f"

        # Check position limits
        allowed, reason = self.check_position_limits(symbol, qty)
        if not allowed:
            return False, reason

        # Check trade limits
        allowed, reason = self.check_daily_trade_limit()
        if not allowed:
            return False, reason

        return True, "Order validated"

# Convenience functions
def create_alpaca_client(paper: bool = True) -> AlpacaExecutionClient:
    """Create Alpaca execution client"""
    return AlpacaExecutionClient(paper=paper)

def create_alpaca_risk_manager(client: AlpacaExecutionClient) -> AlpacaRiskManager:
    """Create Alpaca risk manager"""
    return AlpacaRiskManager(client)

if __name__ == "__main__":
    # Example usage (requires API keys)
    try:
        client = create_alpaca_client(paper=True)
        risk_mgr = create_alpaca_risk_manager(client)

        # Get account info
        account = client.get_account_info()
        print("Account Info:", account)

        # Get positions
        positions = client.get_positions()
        print("Positions:", positions)

        # Get current price
        price = client.get_current_price('SPY')
        print(f"SPY Price: ${price}")

        # Calculate metrics
        metrics = client.calculate_portfolio_metrics(30)
        print("Portfolio Metrics:", metrics)

    except Exception as e:
        print(f"Alpaca client demo failed: {e}")
        print("Make sure ALPACA_API_KEY and ALPACA_API_SECRET are set")

