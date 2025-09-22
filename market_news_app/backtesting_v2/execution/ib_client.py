#!/usr/bin/env python3
"""
Interactive Brokers API Client

Live trading execution through IBKR TWS or Gateway.
Provides advanced order types, portfolio management, and institutional-grade execution.
"""

import ib_insync as ib
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import logging
import threading
import asyncio

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IBKRExecutionClient:
    """Interactive Brokers API client for advanced order execution"""

    def __init__(self,
                 host: str = '127.0.0.1',
                 port: int = 7497,  # Live: 7496, Paper: 7497
                 client_id: int = 1):
        """
        Initialize IBKR client

        Args:
            host: TWS/Gateway host
            port: TWS/Gateway port
            client_id: Client ID for connection
        """
        self.ib = ib.IB()
        self.host = host
        self.port = port
        self.client_id = client_id
        self.connected = False

        # Connect to IBKR
        self._connect()

        # Account and portfolio tracking
        self.account_info = {}
        self.positions = {}
        self.account_updates_thread = None

    def _connect(self):
        """Connect to IBKR TWS/Gateway"""
        try:
            self.ib.connect(self.host, self.port, clientId=self.client_id)
            self.connected = True
            logger.info(f"Connected to IBKR at {self.host}:{self.port}")

            # Start account updates in background
            self._start_account_updates()

        except Exception as e:
            logger.error(f"Failed to connect to IBKR: {e}")
            self.connected = False

    def _start_account_updates(self):
        """Start background account and position updates"""
        def account_update_loop():
            while self.connected:
                try:
                    # Update account info
                    self.account_info = self._get_account_info()

                    # Update positions
                    self.positions = self._get_positions()

                    time.sleep(5)  # Update every 5 seconds

                except Exception as e:
                    logger.error(f"Account update error: {e}")
                    time.sleep(10)

        self.account_updates_thread = threading.Thread(target=account_update_loop, daemon=True)
        self.account_updates_thread.start()

    def _get_account_info(self) -> Dict[str, Any]:
        """Get current account information"""
        try:
            account = self.ib.accountSummary()

            account_data = {}
            for item in account:
                key = item.tag
                value = item.value
                account_data[key] = float(value) if value.replace('.', '').replace('-', '').isdigit() else value

            return account_data

        except Exception as e:
            logger.error(f"Failed to get account info: {e}")
            return {}

    def _get_positions(self) -> Dict[str, Dict[str, Any]]:
        """Get current positions"""
        try:
            positions = self.ib.positions()

            position_data = {}
            for pos in positions:
                symbol = pos.contract.symbol
                position_data[symbol] = {
                    'contract': pos.contract,
                    'position': pos.position,
                    'avg_cost': pos.avgCost,
                    'account': pos.account
                }

            return position_data

        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            return {}

    def disconnect(self):
        """Disconnect from IBKR"""
        if self.connected:
            self.ib.disconnect()
            self.connected = False
            logger.info("Disconnected from IBKR")

    def get_account_info(self) -> Dict[str, Any]:
        """Get current account information"""
        return self.account_info.copy()

    def get_positions(self) -> Dict[str, Dict[str, Any]]:
        """Get current positions"""
        return self.positions.copy()

    def get_current_price(self, symbol: str, sec_type: str = 'STK', exchange: str = 'SMART') -> float:
        """
        Get current market price for a symbol

        Args:
            symbol: Stock symbol
            sec_type: Security type ('STK', 'OPT', etc.)
            exchange: Exchange

        Returns:
            Current market price
        """
        try:
            contract = ib.Contract()
            contract.symbol = symbol
            contract.secType = sec_type
            contract.exchange = exchange
            contract.currency = 'USD'

            self.ib.reqMktData(contract, '', False, False)

            # Wait for market data
            timeout = 10
            start_time = time.time()

            while time.time() - start_time < timeout:
                ticker = self.ib.ticker(contract)
                if ticker and ticker.last:
                    return float(ticker.last)
                time.sleep(0.1)

            # Fallback to close price
            if ticker and ticker.close:
                return float(ticker.close)

            logger.warning(f"Could not get live price for {symbol}")
            return 0.0

        except Exception as e:
            logger.error(f"Failed to get price for {symbol}: {e}")
            return 0.0

    def create_stock_contract(self, symbol: str, exchange: str = 'SMART') -> ib.Contract:
        """Create stock contract"""
        contract = ib.Contract()
        contract.symbol = symbol
        contract.secType = 'STK'
        contract.exchange = exchange
        contract.currency = 'USD'
        return contract

    def create_option_contract(self,
                             symbol: str,
                             expiration: str,
                             strike: float,
                             right: str,
                             exchange: str = 'SMART') -> ib.Contract:
        """Create options contract"""
        contract = ib.Contract()
        contract.symbol = symbol
        contract.secType = 'OPT'
        contract.exchange = exchange
        contract.currency = 'USD'
        contract.lastTradeDateOrContractMonth = expiration
        contract.strike = strike
        contract.right = right
        return contract

    def place_market_order(self,
                          contract: ib.Contract,
                          quantity: int,
                          action: str,
                          transmit: bool = True) -> Optional[ib.Trade]:
        """
        Place a market order

        Args:
            contract: IBKR contract
            quantity: Order quantity
            action: 'BUY' or 'SELL'
            transmit: Whether to transmit immediately

        Returns:
            Trade object
        """
        try:
            order = ib.MarketOrder(action, quantity, transmit=transmit)

            trade = self.ib.placeOrder(contract, order)

            logger.info(f"Placed {action} market order for {quantity} {contract.symbol}")
            return trade

        except Exception as e:
            logger.error(f"Failed to place market order: {e}")
            return None

    def place_limit_order(self,
                         contract: ib.Contract,
                         quantity: int,
                         action: str,
                         limit_price: float,
                         transmit: bool = True) -> Optional[ib.Trade]:
        """
        Place a limit order

        Args:
            contract: IBKR contract
            quantity: Order quantity
            action: 'BUY' or 'SELL'
            limit_price: Limit price
            transmit: Whether to transmit immediately

        Returns:
            Trade object
        """
        try:
            order = ib.LimitOrder(action, quantity, limit_price, transmit=transmit)

            trade = self.ib.placeOrder(contract, order)

            logger.info(f"Placed {action} limit order for {quantity} {contract.symbol} at ${limit_price}")
            return trade

        except Exception as e:
            logger.error(f"Failed to place limit order: {e}")
            return None

    def place_stop_order(self,
                        contract: ib.Contract,
                        quantity: int,
                        action: str,
                        stop_price: float,
                        transmit: bool = True) -> Optional[ib.Trade]:
        """
        Place a stop order

        Args:
            contract: IBKR contract
            quantity: Order quantity
            action: 'BUY' or 'SELL'
            stop_price: Stop price
            transmit: Whether to transmit immediately

        Returns:
            Trade object
        """
        try:
            order = ib.StopOrder(action, quantity, stop_price, transmit=transmit)

            trade = self.ib.placeOrder(contract, order)

            logger.info(f"Placed {action} stop order for {quantity} {contract.symbol} at ${stop_price}")
            return trade

        except Exception as e:
            logger.error(f"Failed to place stop order: {e}")
            return None

    def place_bracket_order(self,
                           contract: ib.Contract,
                           quantity: int,
                           action: str,
                           entry_price: float,
                           stop_loss: float,
                           take_profit: float) -> List[ib.Trade]:
        """
        Place a bracket order (entry + stop loss + take profit)

        Args:
            contract: IBKR contract
            quantity: Order quantity
            action: 'BUY' or 'SELL'
            entry_price: Entry price
            stop_loss: Stop loss price
            take_profit: Take profit price

        Returns:
            List of trade objects
        """
        try:
            # Create bracket orders
            bracket = self.ib.bracketOrder(
                action, quantity, entry_price, take_profit, stop_loss
            )

            # Place the bracket order
            trades = []
            for order in bracket:
                trade = self.ib.placeOrder(contract, order)
                trades.append(trade)

            logger.info(f"Placed bracket order for {quantity} {contract.symbol}")
            return trades

        except Exception as e:
            logger.error(f"Failed to place bracket order: {e}")
            return []

    def cancel_order(self, trade: ib.Trade) -> bool:
        """
        Cancel an order

        Args:
            trade: Trade object to cancel

        Returns:
            True if successful
        """
        try:
            self.ib.cancelOrder(trade.order)
            logger.info(f"Cancelled order for {trade.contract.symbol}")
            return True
        except Exception as e:
            logger.error(f"Failed to cancel order: {e}")
            return False

    def cancel_all_orders(self) -> int:
        """
        Cancel all open orders

        Returns:
            Number of orders cancelled
        """
        try:
            open_orders = self.ib.openOrders()
            cancelled_count = 0

            for order in open_orders:
                try:
                    self.ib.cancelOrder(order)
                    cancelled_count += 1
                except:
                    continue

            logger.info(f"Cancelled {cancelled_count} orders")
            return cancelled_count

        except Exception as e:
            logger.error(f"Failed to cancel all orders: {e}")
            return 0

    def get_order_status(self, trade: ib.Trade) -> Dict[str, Any]:
        """
        Get order status

        Args:
            trade: Trade object

        Returns:
            Order status information
        """
        try:
            return {
                'order_id': trade.order.orderId,
                'status': trade.orderState.status,
                'filled': trade.orderState.filled,
                'remaining': trade.orderState.remaining,
                'avg_fill_price': trade.orderState.avgFillPrice,
                'last_fill_price': trade.orderState.lastFillPrice,
                'why_held': trade.orderState.whyHeld
            }
        except Exception as e:
            logger.error(f"Failed to get order status: {e}")
            return {}

    def get_open_orders(self) -> List[Dict[str, Any]]:
        """Get all open orders"""
        try:
            open_orders = self.ib.openOrders()
            return [{
                'order_id': order.orderId,
                'symbol': order.contract.symbol,
                'action': order.action,
                'quantity': order.totalQuantity,
                'type': order.orderType,
                'status': 'open'
            } for order in open_orders]
        except Exception as e:
            logger.error(f"Failed to get open orders: {e}")
            return []

    def execute_portfolio_trades(self,
                                target_allocations: Dict[str, float],
                                portfolio_value: float = None) -> List[ib.Trade]:
        """
        Execute trades to achieve target portfolio allocations

        Args:
            target_allocations: Target allocations (symbol -> weight)
            portfolio_value: Portfolio value to use

        Returns:
            List of executed trades
        """
        if portfolio_value is None:
            account_info = self.get_account_info()
            portfolio_value = account_info.get('NetLiquidation', 0)

        if portfolio_value <= 0:
            logger.error("Invalid portfolio value")
            return []

        current_positions = self.get_positions()
        executed_trades = []

        for symbol, target_weight in target_allocations.items():
            try:
                target_value = portfolio_value * target_weight
                current_qty = current_positions.get(symbol, {}).get('position', 0)
                current_price = self.get_current_price(symbol)

                if current_price <= 0:
                    logger.warning(f"Could not get price for {symbol}")
                    continue

                target_qty = target_value / current_price
                qty_diff = int(target_qty - current_qty)

                if abs(qty_diff) == 0:
                    continue

                # Minimum order size
                if abs(qty_diff) * current_price < 100:  # Minimum $100 order
                    continue

                contract = self.create_stock_contract(symbol)

                if qty_diff > 0:  # Buy
                    trade = self.place_market_order(contract, qty_diff, 'BUY')
                else:  # Sell
                    trade = self.place_market_order(contract, abs(qty_diff), 'SELL')

                if trade:
                    executed_trades.append(trade)

                time.sleep(0.5)  # Rate limiting

            except Exception as e:
                logger.error(f"Failed to execute trade for {symbol}: {e}")
                continue

        logger.info(f"Executed {len(executed_trades)} portfolio rebalancing trades")
        return executed_trades

    def get_historical_data(self,
                           contract: ib.Contract,
                           duration: str = '1 Y',
                           bar_size: str = '1 day') -> pd.DataFrame:
        """
        Get historical market data

        Args:
            contract: IBKR contract
            duration: Duration string ('1 Y', '6 M', etc.)
            bar_size: Bar size ('1 day', '1 hour', etc.)

        Returns:
            DataFrame with historical data
        """
        try:
            bars = self.ib.reqHistoricalData(
                contract,
                endDateTime='',
                durationStr=duration,
                barSizeSetting=bar_size,
                whatToShow='TRADES',
                useRTH=True
            )

            df = ib.util.df(bars)
            if not df.empty:
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)

            return df

        except Exception as e:
            logger.error(f"Failed to get historical data: {e}")
            return pd.DataFrame()

    def calculate_portfolio_metrics(self, lookback_days: int = 252) -> Dict[str, float]:
        """
        Calculate portfolio performance metrics

        Args:
            lookback_days: Lookback period in days

        Returns:
            Dictionary with performance metrics
        """
        try:
            # Get account values history
            account_values = self.ib.accountValues()

            # Extract equity curve
            equity_values = []
            for av in account_values:
                if av.tag == 'NetLiquidation':
                    equity_values.append({
                        'date': av.currency,  # This might need adjustment
                        'equity': float(av.value)
                    })

            if not equity_values:
                return {}

            df = pd.DataFrame(equity_values)
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            df.sort_index(inplace=True)

            # Calculate returns
            df['returns'] = df['equity'].pct_change()

            # Metrics
            total_return = (df['equity'].iloc[-1] / df['equity'].iloc[0]) - 1
            volatility = df['returns'].std() * np.sqrt(252)
            sharpe_ratio = df['returns'].mean() / df['returns'].std() * np.sqrt(252) if df['returns'].std() > 0 else 0

            # Drawdown
            cumulative = (1 + df['returns']).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = drawdown.min()

            return {
                'total_return': float(total_return),
                'volatility': float(volatility),
                'sharpe_ratio': float(sharpe_ratio),
                'max_drawdown': float(max_drawdown),
                'current_equity': float(df['equity'].iloc[-1])
            }

        except Exception as e:
            logger.error(f"Failed to calculate portfolio metrics: {e}")
            return {}

class IBKRRiskManager:
    """Risk management for IBKR trading"""

    def __init__(self, client: IBKRExecutionClient):
        self.client = client

    def check_margin_requirements(self,
                                contract: ib.Contract,
                                quantity: int,
                                action: str) -> Tuple[bool, str]:
        """
        Check margin requirements for a trade

        Args:
            contract: IBKR contract
            quantity: Trade quantity
            action: 'BUY' or 'SELL'

        Returns:
            Tuple of (approved, reason)
        """
        try:
            # Get account info
            account_info = self.client.get_account_info()

            # Simplified margin check - in practice would be more sophisticated
            buying_power = account_info.get('BuyingPower', 0)
            current_price = self.client.get_current_price(contract.symbol)

            if current_price <= 0:
                return False, "Could not get current price"

            trade_value = abs(quantity) * current_price

            # Conservative margin requirement (50% of trade value)
            margin_required = trade_value * 0.5

            if margin_required > buying_power:
                return False, ".2f"

            return True, "Margin requirements met"

        except Exception as e:
            return False, f"Margin check failed: {e}"

    def check_position_limits(self,
                            symbol: str,
                            proposed_qty: int,
                            max_position_pct: float = 0.10) -> Tuple[bool, str]:
        """
        Check position size limits

        Args:
            symbol: Stock symbol
            proposed_qty: Proposed quantity
            max_position_pct: Maximum position size as percentage of portfolio

        Returns:
            Tuple of (allowed, reason)
        """
        try:
            account_info = self.client.get_account_info()
            portfolio_value = account_info.get('NetLiquidation', 0)

            if portfolio_value <= 0:
                return False, "Invalid portfolio value"

            current_price = self.client.get_current_price(symbol)
            if current_price <= 0:
                return False, "Could not get current price"

            position_value = abs(proposed_qty) * current_price
            position_pct = position_value / portfolio_value

            if position_pct > max_position_pct:
                return False, ".1%"

            return True, "Position size within limits"

        except Exception as e:
            return False, f"Position check failed: {e}"

    def validate_order(self,
                      contract: ib.Contract,
                      quantity: int,
                      action: str,
                      order_type: str = 'market',
                      price: float = None) -> Tuple[bool, str]:
        """
        Comprehensive order validation

        Args:
            contract: IBKR contract
            quantity: Order quantity
            action: 'BUY' or 'SELL'
            order_type: Order type
            price: Price (for limit orders)

        Returns:
            Tuple of (valid, reason)
        """
        # Check margin requirements
        margin_ok, margin_reason = self.check_margin_requirements(contract, quantity, action)
        if not margin_ok:
            return False, margin_reason

        # Check position limits
        position_ok, position_reason = self.check_position_limits(contract.symbol, quantity)
        if not position_ok:
            return False, position_reason

        # Check account status
        account_info = self.client.get_account_info()
        account_status = account_info.get('AccountType', '')

        # Additional validations could be added here
        # - Pattern day trading rules
        # - Position concentration
        # - Risk limits per symbol/sector

        return True, "Order validated"

# Convenience functions
def create_ibkr_client(host: str = '127.0.0.1',
                      port: int = 7497,
                      client_id: int = 1) -> IBKRExecutionClient:
    """Create IBKR execution client"""
    return IBKRExecutionClient(host, port, client_id)

def create_ibkr_risk_manager(client: IBKRExecutionClient) -> IBKRRiskManager:
    """Create IBKR risk manager"""
    return IBKRRiskManager(client)

if __name__ == "__main__":
    # Example usage (requires IBKR TWS/Gateway running)
    try:
        client = create_ibkr_client()
        risk_mgr = create_ibkr_risk_manager(client)

        if client.connected:
            # Get account info
            account = client.get_account_info()
            print("Account Info:", account)

            # Get positions
            positions = client.get_positions()
            print("Positions:", list(positions.keys()))

            # Get current price
            price = client.get_current_price('SPY')
            print(f"SPY Price: ${price}")

            # Create and validate order
            contract = client.create_stock_contract('SPY')
            valid, reason = risk_mgr.validate_order(contract, 100, 'BUY')
            print(f"Order validation: {valid} - {reason}")

        else:
            print("Could not connect to IBKR. Make sure TWS/Gateway is running.")

    except Exception as e:
        print(f"IBKR client demo failed: {e}")
        print("Make sure IBKR TWS/Gateway is running on the specified port.")

