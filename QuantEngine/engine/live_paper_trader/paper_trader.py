"""
Paper Trading Engine for AI Quant Trading System

Simulates live trading execution for approved strategies.
Tracks positions, fills, P&L, and generates trading reports.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime, timedelta
from pathlib import Path
import json
import sqlite3

logger = logging.getLogger(__name__)


class PaperTrade:
    """Represents a single paper trade"""

    def __init__(self, strategy_name: str, ticker: str, side: str, quantity: float,
                 price: float, timestamp: datetime, order_type: str = 'market'):
        self.strategy_name = strategy_name
        self.ticker = ticker
        self.side = side  # 'buy' or 'sell'
        self.quantity = quantity
        self.price = price
        self.timestamp = timestamp
        self.order_type = order_type
        self.fill_price = price  # In paper trading, orders fill immediately
        self.commission = 0.0
        self.status = 'filled'
        self.order_id = f"{strategy_name}_{ticker}_{timestamp.strftime('%Y%m%d_%H%M%S')}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert trade to dictionary"""
        return {
            'order_id': self.order_id,
            'strategy_name': self.strategy_name,
            'ticker': self.ticker,
            'side': self.side,
            'quantity': self.quantity,
            'price': self.price,
            'fill_price': self.fill_price,
            'commission': self.commission,
            'timestamp': self.timestamp.isoformat(),
            'order_type': self.order_type,
            'status': self.status
        }


class Position:
    """Represents a position in the paper portfolio"""

    def __init__(self, ticker: str, quantity: float = 0.0, avg_price: float = 0.0):
        self.ticker = ticker
        self.quantity = quantity
        self.avg_price = avg_price
        self.unrealized_pnl = 0.0
        self.realized_pnl = 0.0
        self.last_update = datetime.now()

    def update_unrealized_pnl(self, current_price: float):
        """Update unrealized P&L based on current price"""
        if self.quantity != 0:
            self.unrealized_pnl = (current_price - self.avg_price) * self.quantity
        else:
            self.unrealized_pnl = 0.0
        self.last_update = datetime.now()

    def add_trade(self, trade: PaperTrade):
        """Add a trade to this position"""
        if trade.side == 'buy':
            # Calculate new average price
            total_cost = (self.quantity * self.avg_price) + (trade.quantity * trade.fill_price)
            self.quantity += trade.quantity
            if self.quantity > 0:
                self.avg_price = total_cost / self.quantity
        elif trade.side == 'sell':
            # Realize P&L
            if self.quantity > 0:
                realized = (trade.fill_price - self.avg_price) * min(trade.quantity, self.quantity)
                self.realized_pnl += realized

            self.quantity -= trade.quantity

            # Close position if quantity goes to zero
            if abs(self.quantity) < 0.001:
                self.quantity = 0.0
                self.avg_price = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert position to dictionary"""
        return {
            'ticker': self.ticker,
            'quantity': self.quantity,
            'avg_price': self.avg_price,
            'unrealized_pnl': self.unrealized_pnl,
            'realized_pnl': self.realized_pnl,
            'total_pnl': self.unrealized_pnl + self.realized_pnl,
            'last_update': self.last_update.isoformat()
        }


class PaperPortfolio:
    """Paper trading portfolio"""

    def __init__(self, initial_capital: float = 100000.0):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: Dict[str, Position] = {}
        self.total_value = initial_capital
        self.daily_pnl = []
        self.creation_date = datetime.now()

    def execute_trade(self, trade: PaperTrade) -> bool:
        """Execute a trade in the portfolio"""

        # Calculate trade value
        trade_value = trade.quantity * trade.fill_price

        # Check if we have enough cash for buys
        if trade.side == 'buy':
            if trade_value > self.cash:
                logger.warning(f"Insufficient cash for trade: need ${trade_value:.2f}, have ${self.cash:.2f}")
                return False
            self.cash -= trade_value
        elif trade.side == 'sell':
            # Check if we have the position to sell
            if trade.ticker not in self.positions or self.positions[trade.ticker].quantity < trade.quantity:
                logger.warning(f"Insufficient position for sell: need {trade.quantity}, have {self.positions.get(trade.ticker, Position(trade.ticker)).quantity}")
                return False
            self.cash += trade_value

        # Update position
        if trade.ticker not in self.positions:
            self.positions[trade.ticker] = Position(trade.ticker)

        self.positions[trade.ticker].add_trade(trade)

        logger.info(f"Executed {trade.side} {trade.quantity} {trade.ticker} @ ${trade.fill_price:.2f}")
        return True

    def update_prices(self, prices: Dict[str, float]):
        """Update position prices and calculate unrealized P&L"""

        for ticker, price in prices.items():
            if ticker in self.positions:
                self.positions[ticker].update_unrealized_pnl(price)

        self._calculate_total_value(prices)

    def _calculate_total_value(self, prices: Dict[str, float]):
        """Calculate total portfolio value"""

        position_value = 0.0
        for ticker, position in self.positions.items():
            if ticker in prices:
                position_value += position.quantity * prices[ticker]

        self.total_value = self.cash + position_value

    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get portfolio summary"""

        total_unrealized_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
        total_realized_pnl = sum(pos.realized_pnl for pos in self.positions.values())
        total_pnl = total_unrealized_pnl + total_realized_pnl

        return {
            'cash': self.cash,
            'total_value': self.total_value,
            'total_pnl': total_pnl,
            'unrealized_pnl': total_unrealized_pnl,
            'realized_pnl': total_realized_pnl,
            'num_positions': len([p for p in self.positions.values() if p.quantity != 0]),
            'initial_capital': self.initial_capital,
            'return_pct': (self.total_value - self.initial_capital) / self.initial_capital * 100,
            'positions': [pos.to_dict() for pos in self.positions.values() if pos.quantity != 0]
        }

    def get_daily_pnl(self) -> List[Dict[str, Any]]:
        """Get daily P&L history"""
        return self.daily_pnl


class PaperTradingEngine:
    """Paper trading engine for strategy execution"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.portfolio = PaperPortfolio(config.get('initial_capital', 100000.0))
        self.approved_strategies: Dict[str, Dict[str, Any]] = {}
        self.active_strategies: Dict[str, bool] = {}
        self.trade_log: List[PaperTrade] = []
        self.daily_reports = []

        # Setup database
        self.db_path = Path(config.get('data_path', 'data')) / 'paper_trading.db'
        self._setup_database()

    def _setup_database(self):
        """Setup SQLite database for trade persistence"""

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    order_id TEXT PRIMARY KEY,
                    strategy_name TEXT,
                    ticker TEXT,
                    side TEXT,
                    quantity REAL,
                    price REAL,
                    fill_price REAL,
                    commission REAL,
                    timestamp TEXT,
                    order_type TEXT,
                    status TEXT
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS portfolio_snapshots (
                    date TEXT,
                    cash REAL,
                    total_value REAL,
                    total_pnl REAL,
                    unrealized_pnl REAL,
                    realized_pnl REAL,
                    num_positions INTEGER,
                    return_pct REAL
                )
            """)

    def register_strategy(self, strategy_spec: Dict[str, Any], approval_status: Dict[str, Any]) -> bool:
        """
        Register an approved strategy for paper trading

        Args:
            strategy_spec: Strategy specification
            approval_status: Approval details from robustness testing

        Returns:
            Success status
        """

        strategy_name = strategy_spec.get('name')

        if approval_status.get('approved', False):
            self.approved_strategies[strategy_name] = {
                'spec': strategy_spec,
                'approval': approval_status,
                'registration_date': datetime.now(),
                'status': 'registered'
            }

            logger.info(f"Registered strategy {strategy_name} for paper trading")
            return True
        else:
            logger.warning(f"Strategy {strategy_name} not approved for paper trading")
            return False

    def activate_strategy(self, strategy_name: str) -> bool:
        """Activate a registered strategy for live execution"""

        if strategy_name in self.approved_strategies:
            self.active_strategies[strategy_name] = True
            self.approved_strategies[strategy_name]['status'] = 'active'
            logger.info(f"Activated strategy {strategy_name}")
            return True
        else:
            logger.error(f"Strategy {strategy_name} not registered")
            return False

    def deactivate_strategy(self, strategy_name: str) -> bool:
        """Deactivate a strategy"""

        if strategy_name in self.active_strategies:
            self.active_strategies[strategy_name] = False
            self.approved_strategies[strategy_name]['status'] = 'inactive'
            logger.info(f"Deactivated strategy {strategy_name}")
            return True

        return False

    def run_daily_cycle(self, market_data: Dict[str, pd.DataFrame],
                       current_date: datetime) -> Dict[str, Any]:
        """
        Run daily trading cycle for all active strategies

        Args:
            market_data: Current market data
            current_date: Trading date

        Returns:
            Daily cycle results
        """

        daily_results = {
            'date': current_date,
            'trades_executed': 0,
            'strategies_run': 0,
            'errors': [],
            'portfolio_snapshot': None
        }

        # Update portfolio prices
        current_prices = {}
        for ticker, df in market_data.items():
            # Get price for current date
            date_mask = (df.index.date == current_date.date())
            if date_mask.any():
                current_prices[ticker] = df.loc[date_mask, 'close'].iloc[0]

        if current_prices:
            self.portfolio.update_prices(current_prices)

        # Run each active strategy
        for strategy_name, is_active in self.active_strategies.items():
            if not is_active:
                continue

            try:
                strategy_result = self._run_strategy_cycle(strategy_name, market_data, current_date)
                daily_results['trades_executed'] += strategy_result.get('trades_executed', 0)
                daily_results['strategies_run'] += 1

            except Exception as e:
                error_msg = f"Strategy {strategy_name} failed: {e}"
                daily_results['errors'].append(error_msg)
                logger.error(error_msg)

        # Save portfolio snapshot
        portfolio_summary = self.portfolio.get_portfolio_summary()
        daily_results['portfolio_snapshot'] = portfolio_summary

        # Save to database
        self._save_daily_snapshot(current_date, portfolio_summary)

        # Add to daily reports
        self.daily_reports.append(daily_results)

        logger.info(f"Completed daily cycle for {current_date.date()}: {daily_results['trades_executed']} trades")

        return daily_results

    def _run_strategy_cycle(self, strategy_name: str, market_data: Dict[str, pd.DataFrame],
                           current_date: datetime) -> Dict[str, Any]:
        """Run trading cycle for a single strategy"""

        strategy_info = self.approved_strategies[strategy_name]
        strategy_spec = strategy_info['spec']

        # Simplified strategy execution (in practice, this would use the full backtester)
        trades_executed = 0

        # Mock signal generation and trade execution
        # In production, this would evaluate the strategy's entry/exit conditions

        # Example: Random small trades for demonstration
        if np.random.random() < 0.1:  # 10% chance of trade per day
            ticker = strategy_spec.get('universe', ['SPY'])[0]
            if ticker in market_data:
                df = market_data[ticker]
                date_mask = (df.index.date == current_date.date())
                if date_mask.any():
                    price = df.loc[date_mask, 'close'].iloc[0]

                    # Random buy or sell
                    side = 'buy' if np.random.random() > 0.5 else 'sell'
                    quantity = np.random.uniform(10, 100)

                    # Create and execute trade
                    trade = PaperTrade(
                        strategy_name=strategy_name,
                        ticker=ticker,
                        side=side,
                        quantity=quantity,
                        price=price,
                        timestamp=current_date
                    )

                    if self.portfolio.execute_trade(trade):
                        self.trade_log.append(trade)
                        self._save_trade(trade)
                        trades_executed += 1

        return {'trades_executed': trades_executed}

    def _save_trade(self, trade: PaperTrade):
        """Save trade to database"""

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO trades (order_id, strategy_name, ticker, side, quantity,
                                  price, fill_price, commission, timestamp, order_type, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                trade.order_id, trade.strategy_name, trade.ticker, trade.side,
                trade.quantity, trade.price, trade.fill_price, trade.commission,
                trade.timestamp.isoformat(), trade.order_type, trade.status
            ))

    def _save_daily_snapshot(self, date: datetime, portfolio_summary: Dict[str, Any]):
        """Save daily portfolio snapshot"""

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO portfolio_snapshots (date, cash, total_value, total_pnl,
                                               unrealized_pnl, realized_pnl, num_positions, return_pct)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                date.date().isoformat(),
                portfolio_summary['cash'],
                portfolio_summary['total_value'],
                portfolio_summary['total_pnl'],
                portfolio_summary['unrealized_pnl'],
                portfolio_summary['realized_pnl'],
                portfolio_summary['num_positions'],
                portfolio_summary['return_pct']
            ))

    def get_performance_report(self, start_date: Optional[datetime] = None,
                             end_date: Optional[datetime] = None) -> Dict[str, Any]:
        """Generate performance report for paper trading"""

        if start_date is None:
            start_date = self.portfolio.creation_date
        if end_date is None:
            end_date = datetime.now()

        # Get portfolio snapshots from database
        with sqlite3.connect(self.db_path) as conn:
            snapshots = conn.execute("""
                SELECT * FROM portfolio_snapshots
                WHERE date >= ? AND date <= ?
                ORDER BY date
            """, (start_date.date().isoformat(), end_date.date().isoformat())).fetchall()

        # Calculate performance metrics
        if snapshots:
            values = [s[2] for s in snapshots]  # total_value
            returns = [0]
            for i in range(1, len(values)):
                ret = (values[i] - values[i-1]) / values[i-1]
                returns.append(ret)

            # Basic metrics
            total_return = (values[-1] - values[0]) / values[0]
            ann_return = total_return * 365 / (end_date - start_date).days

            # Volatility
            ann_vol = np.std(returns) * np.sqrt(252) if len(returns) > 1 else 0

            # Sharpe
            sharpe = ann_return / ann_vol if ann_vol > 0 else 0

            # Max drawdown
            running_max = np.maximum.accumulate(values)
            drawdown = (values - running_max) / running_max
            max_dd = np.min(drawdown)

        else:
            total_return = ann_return = ann_vol = sharpe = max_dd = 0

        # Get trade statistics
        total_trades = len(self.trade_log)
        winning_trades = sum(1 for trade in self.trade_log
                           if (trade.side == 'sell' and trade.fill_price > trade.price) or
                              (trade.side == 'buy' and trade.fill_price < trade.price))

        win_rate = winning_trades / total_trades if total_trades > 0 else 0

        return {
            'start_date': start_date,
            'end_date': end_date,
            'total_return': total_return,
            'ann_return': ann_return,
            'ann_vol': ann_vol,
            'sharpe': sharpe,
            'max_dd': max_dd,
            'total_trades': total_trades,
            'win_rate': win_rate,
            'portfolio_summary': self.portfolio.get_portfolio_summary(),
            'active_strategies': list(self.active_strategies.keys()),
            'approved_strategies': list(self.approved_strategies.keys())
        }

    def generate_daily_report(self) -> str:
        """Generate daily paper trading report"""

        portfolio = self.portfolio.get_portfolio_summary()

        report = f"""# Paper Trading Daily Report

**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Portfolio Summary

- **Total Value:** ${portfolio['total_value']:,.2f}
- **Cash:** ${portfolio['cash']:,.2f}
- **Total P&L:** ${portfolio['total_pnl']:,.2f}
- **Return:** {portfolio['return_pct']:.2f}%
- **Active Positions:** {portfolio['num_positions']}

## Active Strategies

"""

        for strategy_name, is_active in self.active_strategies.items():
            status = "✅ Active" if is_active else "⏸️ Inactive"
            report += f"- {strategy_name}: {status}\n"

        if portfolio['positions']:
            report += "\n## Current Positions\n\n"
            report += "| Ticker | Quantity | Avg Price | Unrealized P&L |\n"
            report += "|--------|----------|-----------|----------------|\n"

            for position in portfolio['positions']:
                report += f"| {position['ticker']} | {position['quantity']:.0f} | ${position['avg_price']:.2f} | ${position['unrealized_pnl']:.2f} |\n"

        # Recent trades
        recent_trades = self.trade_log[-5:]  # Last 5 trades
        if recent_trades:
            report += "\n## Recent Trades\n\n"
            report += "| Time | Strategy | Ticker | Side | Quantity | Price |\n"
            report += "|------|----------|--------|------|----------|-------|\n"

            for trade in recent_trades:
                report += f"| {trade.timestamp.strftime('%H:%M')} | {trade.strategy_name} | {trade.ticker} | {trade.side.upper()} | {trade.quantity:.0f} | ${trade.fill_price:.2f} |\n"

        return report


# Test functions
def test_paper_trading():
    """Test paper trading functionality"""

    print("🧪 Testing Paper Trading Engine")

    config = {
        'initial_capital': 100000.0,
        'data_path': 'data'
    }

    engine = PaperTradingEngine(config)

    # Test portfolio operations
    portfolio = engine.portfolio

    # Create some mock trades
    trade1 = PaperTrade('test_strategy', 'AAPL', 'buy', 100, 150.0, datetime.now())
    trade2 = PaperTrade('test_strategy', 'AAPL', 'sell', 50, 155.0, datetime.now())

    portfolio.execute_trade(trade1)
    portfolio.execute_trade(trade2)

    # Update prices
    portfolio.update_prices({'AAPL': 160.0})

    summary = portfolio.get_portfolio_summary()

    print(f"✅ Portfolio value: ${summary['total_value']:,.2f}")
    print(f"✅ Cash: ${summary['cash']:,.2f}")
    print(f"✅ P&L: ${summary['total_pnl']:.2f}")
    print(f"✅ Positions: {summary['num_positions']}")

    return summary


if __name__ == "__main__":
    test_paper_trading()


