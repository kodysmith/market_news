# Broker Integration Specification

**Complete guide to Alpaca and Interactive Brokers integration**

---

## Overview

### Dual-Broker Strategy

**Primary**: Alpaca (commission-free options, modern API)  
**Backup**: Interactive Brokers (institutional-grade, global reach)

**Rationale:**
- **Reliability**: If one fails, failover to other
- **Cost**: Alpaca has no commissions on options
- **Quality**: IB has better fills for large orders
- **Risk**: Don't depend on single counterparty

---

## Alpaca Integration (Primary)

### Why Alpaca

✅ **Pros:**
- Commission-free options trading
- Modern REST API (easy integration)
- Real-time market data included
- Paper trading environment
- Good documentation
- Active community

⚠️ **Cons:**
- Newer broker (less history)
- Limited to US markets
- Smaller than IB
- Options availability varies

### API Setup

**1. Account Creation:**
```
1. Visit alpaca.markets
2. Sign up for trading account
3. Complete verification (2-3 days)
4. Enable options trading (additional approval)
5. Fund account
6. Get API credentials
```

**2. API Credentials:**
```python
# Paper trading credentials
ALPACA_PAPER_API_KEY = "PK..."
ALPACA_PAPER_SECRET_KEY = "..."
ALPACA_PAPER_BASE_URL = "https://paper-api.alpaca.markets"

# Live trading credentials  
ALPACA_LIVE_API_KEY = "AK..."
ALPACA_LIVE_SECRET_KEY = "..."
ALPACA_LIVE_BASE_URL = "https://api.alpaca.markets"

# Store in AWS Secrets Manager, NOT in code!
```

### Core API Endpoints

**Account Management:**
```python
GET /v2/account
# Returns account info, buying power, cash, portfolio value

GET /v2/positions
# Returns all current positions

GET /v2/positions/{symbol}
# Get specific position details
```

**Options Trading:**
```python
POST /v2/orders
# Submit option order
{
  "symbol": "SPY241115P00400000",  # OCC format
  "qty": 1,
  "side": "sell",
  "type": "limit",
  "time_in_force": "day",
  "limit_price": "8.50",
  "order_class": "simple"
}

GET /v2/orders
# Get all orders (open, filled, canceled)

GET /v2/orders/{order_id}
# Get specific order status

DELETE /v2/orders/{order_id}
# Cancel open order
```

**Market Data:**
```python
GET /v2/stocks/{symbol}/quotes/latest
# Get latest quote (bid/ask)

GET /v2/stocks/{symbol}/bars
# Get historical bars

GET /v2/options/contracts
# Get options contracts for symbol

GET /v2/options/quotes/latest
# Get option quote
```

### Python SDK Usage

**Installation:**
```bash
pip install alpaca-py
```

**Basic Trading:**
```python
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import LimitOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

# Initialize client
trading_client = TradingClient(
    api_key=ALPACA_API_KEY,
    secret_key=ALPACA_SECRET_KEY,
    paper=True  # Set False for live trading
)

# Get account info
account = trading_client.get_account()
print(f"Cash: ${account.cash}")
print(f"Buying Power: ${account.buying_power}")

# Submit option order
order_request = LimitOrderRequest(
    symbol="QQQ241115P00350000",  # QQQ Nov 15 2024 350 Put
    qty=1,
    side=OrderSide.SELL,
    time_in_force=TimeInForce.DAY,
    limit_price=5.50
)

order = trading_client.submit_order(order_request)

# Check order status
order_status = trading_client.get_order_by_id(order.id)
print(f"Status: {order_status.status}")
```

**Options Symbol Format** (OCC standard):
```
Format: AAAAAAMMDDDDCCPPPPPPPP

AAAAAA - Underlying (6 chars, padded with spaces if needed)
MM     - Expiration month (01-12)
DD     - Expiration day (01-31)
DD     - Expiration year (last 2 digits)
CC     - Call/Put (C or P)
PPPPPPPP - Strike price (8 digits, including decimals)

Examples:
SPY241115P00400000   = SPY Nov 15 2024 $400 Put
QQQ  241122C00350000 = QQQ Nov 22 2024 $350 Call (note spaces)
```

**Helper Function:**
```python
def create_option_symbol(underlying: str, strike: float, 
                        option_type: str, expiration: date) -> str:
    """Create OCC-format option symbol"""
    # Pad underlying to 6 characters
    padded_underlying = underlying.ljust(6)
    
    # Format expiration
    exp_str = expiration.strftime('%y%m%d')
    
    # Format strike (multiply by 1000, pad to 8 digits)
    strike_str = f"{int(strike * 1000):08d}"
    
    # Put it together
    symbol = f"{padded_underlying}{exp_str}{option_type}{strike_str}"
    
    return symbol

# Usage
symbol = create_option_symbol('SPY', 400.0, 'P', date(2024, 11, 15))
# Returns: "SPY   241115P00400000"
```

### Error Handling

**Common Errors:**

```python
from alpaca.trading.requests import LimitOrderRequest
from alpaca.common.exceptions import APIError

def submit_order_with_retry(client, order_request, max_retries=3):
    """Submit order with automatic retry"""
    
    for attempt in range(max_retries):
        try:
            order = client.submit_order(order_request)
            return order
            
        except APIError as e:
            if e.status_code == 403:
                # Insufficient buying power
                logger.error("Insufficient buying power", error=str(e))
                raise  # Don't retry
            
            elif e.status_code == 422:
                # Invalid order (bad symbol, etc.)
                logger.error("Invalid order", error=str(e))
                raise  # Don't retry
            
            elif e.status_code == 429:
                # Rate limit - wait and retry
                wait_time = 2 ** attempt  # Exponential backoff
                logger.warning(f"Rate limited, waiting {wait_time}s")
                time.sleep(wait_time)
                continue
            
            elif e.status_code >= 500:
                # Server error - retry
                logger.warning(f"Server error, attempt {attempt+1}/{max_retries}")
                time.sleep(1)
                continue
            
            else:
                logger.error("Unknown API error", status=e.status_code)
                raise
    
    raise MaxRetriesExceeded("Failed after {max_retries} attempts")
```

### Rate Limits

**Alpaca Limits:**
- 200 requests per minute (trading)
- 10,000 requests per minute (market data)

**Handling:**
```python
import time
from collections import deque

class RateLimiter:
    """Enforce rate limits"""
    
    def __init__(self, max_calls: int, time_window: int):
        self.max_calls = max_calls
        self.time_window = time_window
        self.calls = deque()
    
    def acquire(self):
        """Block until rate limit allows"""
        now = time.time()
        
        # Remove calls outside window
        while self.calls and self.calls[0] < now - self.time_window:
            self.calls.popleft()
        
        # Wait if at limit
        if len(self.calls) >= self.max_calls:
            sleep_time = self.time_window - (now - self.calls[0])
            if sleep_time > 0:
                logger.debug(f"Rate limit reached, sleeping {sleep_time:.2f}s")
                time.sleep(sleep_time)
            self.calls.popleft()
        
        # Record this call
        self.calls.append(time.time())

# Usage
rate_limiter = RateLimiter(max_calls=200, time_window=60)

def make_api_call():
    rate_limiter.acquire()  # Blocks if needed
    return client.get_positions()
```

---

## Interactive Brokers Integration (Backup)

### Why Interactive Brokers

✅ **Pros:**
- Institutional-grade reliability
- Global markets access
- Best execution quality
- Huge capital base (safe)
- Advanced order types
- Professional support

⚠️ **Cons:**
- Commissions ($0.65/contract + fees)
- Complex API
- More setup required
- Desktop app needed (TWS/Gateway)

### API Setup

**1. Account Creation:**
```
1. Visit interactivebrokers.com
2. Open individual or entity account
3. Complete verification (3-5 days)
4. Enable options permissions
5. Fund account ($10K minimum)
6. Enable API access in account settings
```

**2. Install TWS Gateway:**
```bash
# Download IB Gateway (lighter than full TWS)
wget https://download2.interactivebrokers.com/installers/ibgateway/latest-standalone/ibgateway-latest-standalone-linux-x64.sh

# Install
chmod +x ibgateway-*.sh
./ibgateway-*.sh

# Configure
# - Enable API connections
# - Set port (default 4001 for paper, 4002 for live)
# - Disable auto-logoff
# - Allow connections from localhost
```

**3. Python Integration (ib_insync):**
```bash
pip install ib_insync
```

**Basic Usage:**
```python
from ib_insync import IB, Stock, Option, MarketOrder, LimitOrder

# Connect to IB Gateway
ib = IB()
ib.connect('127.0.0.1', 4001, clientId=1)  # 4001 = paper trading

# Check connection
if ib.isConnected():
    print("✅ Connected to IB")

# Get account info
account_values = ib.accountValues()
cash = [v for v in account_values if v.tag == 'CashBalance'][0].value

# Create option contract
option = Option(
    symbol='SPY',
    lastTradeDateOrContractMonth='20241115',
    strike=400,
    right='P',  # Put
    exchange='SMART'
)

# Get contract details (validate it exists)
ib.qualifyContracts(option)

# Submit order
order = LimitOrder(
    action='SELL',
    totalQuantity=1,
    lmtPrice=8.50
)

trade = ib.placeOrder(option, order)

# Monitor order status
while not trade.isDone():
    ib.waitOnUpdate()
    print(f"Status: {trade.orderStatus.status}")

# Get fill details
if trade.orderStatus.status == 'Filled':
    print(f"Filled at: ${trade.orderStatus.avgFillPrice}")
```

### IB-Specific Considerations

**Contract Qualification:**
```python
# IB requires "qualifying" contracts before trading

def qualify_option(symbol, expiration, strike, right):
    """Ensure contract exists and get complete details"""
    
    option = Option(
        symbol=symbol,
        lastTradeDateOrContractMonth=expiration.strftime('%Y%m%d'),
        strike=strike,
        right=right,
        exchange='SMART'
    )
    
    # Qualify with IB (fills in contract ID, multiplier, etc.)
    contracts = ib.qualifyContracts(option)
    
    if not contracts:
        raise ValueError(f"Option contract not found: {option}")
    
    return contracts[0]
```

**Market Data Subscriptions:**
```python
# IB charges for live market data
# Need subscriptions for real-time quotes

# Request market data
ticker = ib.reqMktData(contract)

# Wait for data
ib.sleep(1)

# Access data
print(f"Bid: {ticker.bid}, Ask: {ticker.ask}")

# Cancel subscription when done
ib.cancelMktData(contract)
```

**Connection Management:**
```python
class IBConnection:
    """Manage IB connection with auto-reconnect"""
    
    def __init__(self, host='127.0.0.1', port=4001, client_id=1):
        self.ib = IB()
        self.host = host
        self.port = port
        self.client_id = client_id
        self.connect()
    
    def connect(self):
        """Connect with error handling"""
        try:
            self.ib.connect(self.host, self.port, self.client_id)
            logger.info("Connected to IB Gateway")
        except Exception as e:
            logger.error(f"IB connection failed: {e}")
            raise
    
    def ensure_connected(self):
        """Reconnect if disconnected"""
        if not self.ib.isConnected():
            logger.warning("IB disconnected, reconnecting...")
            self.connect()
    
    def disconnect(self):
        """Clean disconnect"""
        if self.ib.isConnected():
            self.ib.disconnect()
            logger.info("Disconnected from IB")
```

---

## Unified Broker Interface

### Abstract Base Class

```python
# File: src/execution/broker_interface.py

from abc import ABC, abstractmethod
from typing import List, Optional
from dataclasses import dataclass

@dataclass
class Fill:
    """Standardized fill details"""
    order_id: str
    timestamp: datetime
    symbol: str
    quantity: int
    fill_price: float
    commission: float
    broker: str
    
    @property
    def total_value(self):
        return self.quantity * self.fill_price * 100  # Options are 100 multiplier

@dataclass
class Position:
    """Standardized position details"""
    symbol: str
    quantity: int
    avg_entry_price: float
    current_price: float
    market_value: float
    unrealized_pnl: float

class BrokerInterface(ABC):
    """Abstract broker interface - all brokers must implement"""
    
    @abstractmethod
    def submit_order(self, symbol: str, quantity: int, side: str, 
                    limit_price: Optional[float] = None) -> Fill:
        """Submit an order"""
        pass
    
    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order"""
        pass
    
    @abstractmethod
    def get_positions(self) -> List[Position]:
        """Get all current positions"""
        pass
    
    @abstractmethod
    def get_account_info(self) -> dict:
        """Get account information"""
        pass
    
    @abstractmethod
    def get_order_status(self, order_id: str) -> str:
        """Check order status"""
        pass
```

### Alpaca Implementation

```python
# File: src/execution/alpaca_broker.py

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import LimitOrderRequest, MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

class AlpacaBroker(BrokerInterface):
    """Alpaca-specific implementation"""
    
    def __init__(self, api_key: str, secret_key: str, paper: bool = True):
        self.client = TradingClient(api_key, secret_key, paper=paper)
        self.broker_name = 'alpaca'
    
    def submit_order(self, symbol, quantity, side, limit_price=None) -> Fill:
        """Submit order to Alpaca"""
        
        # Create order request
        if limit_price:
            request = LimitOrderRequest(
                symbol=symbol,
                qty=quantity,
                side=OrderSide.BUY if side == 'buy' else OrderSide.SELL,
                time_in_force=TimeInForce.DAY,
                limit_price=limit_price
            )
        else:
            request = MarketOrderRequest(
                symbol=symbol,
                qty=quantity,
                side=OrderSide.BUY if side == 'buy' else OrderSide.SELL,
                time_in_force=TimeInForce.DAY
            )
        
        # Submit
        order = self.client.submit_order(request)
        
        # Wait for fill (with timeout)
        fill = self._wait_for_fill(order.id, timeout=30)
        
        return fill
    
    def _wait_for_fill(self, order_id, timeout=30) -> Fill:
        """Wait for order to fill"""
        start = time.time()
        
        while time.time() - start < timeout:
            order = self.client.get_order_by_id(order_id)
            
            if order.status == 'filled':
                return Fill(
                    order_id=order.id,
                    timestamp=order.filled_at,
                    symbol=order.symbol,
                    quantity=int(order.filled_qty),
                    fill_price=float(order.filled_avg_price),
                    commission=0.0,  # Alpaca is commission-free
                    broker=self.broker_name
                )
            
            elif order.status in ['canceled', 'rejected']:
                raise OrderFailedException(f"Order {order_id} status: {order.status}")
            
            time.sleep(0.5)  # Poll every 500ms
        
        raise TimeoutError(f"Order {order_id} did not fill within {timeout}s")
    
    def get_positions(self) -> List[Position]:
        """Get all positions from Alpaca"""
        alpaca_positions = self.client.get_all_positions()
        
        return [
            Position(
                symbol=p.symbol,
                quantity=int(p.qty),
                avg_entry_price=float(p.avg_entry_price),
                current_price=float(p.current_price),
                market_value=float(p.market_value),
                unrealized_pnl=float(p.unrealized_pl)
            )
            for p in alpaca_positions
        ]
    
    def get_account_info(self) -> dict:
        """Get account details"""
        account = self.client.get_account()
        
        return {
            'cash': float(account.cash),
            'buying_power': float(account.buying_power),
            'portfolio_value': float(account.portfolio_value),
            'equity': float(account.equity),
            'margin_used': float(account.portfolio_value) - float(account.cash)
        }
```

### IB Implementation

```python
# File: src/execution/ib_broker.py

from ib_insync import IB, Option, LimitOrder, MarketOrder

class IBBroker(BrokerInterface):
    """Interactive Brokers implementation"""
    
    def __init__(self, host='127.0.0.1', port=4001, client_id=1):
        self.ib = IB()
        self.ib.connect(host, port, client_id)
        self.broker_name = 'interactive_brokers'
    
    def submit_order(self, symbol, quantity, side, limit_price=None) -> Fill:
        """Submit order to IB"""
        
        # Parse option symbol (convert OCC to IB format)
        contract = self._parse_option_symbol(symbol)
        
        # Qualify contract
        self.ib.qualifyContracts(contract)
        
        # Create order
        if limit_price:
            order = LimitOrder(
                action='SELL' if side == 'sell' else 'BUY',
                totalQuantity=quantity,
                lmtPrice=limit_price
            )
        else:
            order = MarketOrder(
                action='SELL' if side == 'sell' else 'BUY',
                totalQuantity=quantity
            )
        
        # Submit
        trade = self.ib.placeOrder(contract, order)
        
        # Wait for fill
        while not trade.isDone():
            self.ib.waitOnUpdate(timeout=1)
        
        # Convert to standard Fill format
        return Fill(
            order_id=str(trade.order.orderId),
            timestamp=trade.log[-1].time,
            symbol=symbol,
            quantity=trade.orderStatus.filled,
            fill_price=trade.orderStatus.avgFillPrice,
            commission=trade.orderStatus.commission,
            broker=self.broker_name
        )
    
    def _parse_option_symbol(self, occ_symbol: str) -> Option:
        """Convert OCC symbol to IB Option contract"""
        # Parse OCC format: SPY241115P00400000
        underlying = occ_symbol[:6].strip()
        year = int('20' + occ_symbol[6:8])
        month = int(occ_symbol[8:10])
        day = int(occ_symbol[10:12])
        right = occ_symbol[12]  # C or P
        strike = int(occ_symbol[13:]) / 1000.0
        
        expiration = date(year, month, day)
        
        return Option(
            symbol=underlying,
            lastTradeDateOrContractMonth=expiration.strftime('%Y%m%d'),
            strike=strike,
            right=right,
            exchange='SMART'
        )
```

---

## Failover Logic

### Automatic Broker Failover

```python
# File: src/execution/broker_manager.py

class BrokerManager:
    """Manage multiple brokers with automatic failover"""
    
    def __init__(self):
        self.alpaca = AlpacaBroker(api_key, secret_key, paper=False)
        self.ib = IBBroker(host='localhost', port=4002)  # Live port
        
        self.primary = self.alpaca
        self.backup = self.ib
        
        self.failover_count = 0
        self.last_failover = None
    
    def submit_order(self, symbol, quantity, side, limit_price=None) -> Fill:
        """Try primary, failover to backup if needed"""
        
        # Try primary broker
        try:
            fill = self.primary.submit_order(symbol, quantity, side, limit_price)
            logger.info(f"Order filled via primary ({self.primary.broker_name})")
            return fill
            
        except Exception as e:
            logger.error(f"Primary broker failed: {e}")
            
            # Alert operations
            self.alert_failover(e)
            
            # Try backup broker
            try:
                fill = self.backup.submit_order(symbol, quantity, side, limit_price)
                logger.warning(f"Order filled via backup ({self.backup.broker_name})")
                
                self.failover_count += 1
                self.last_failover = datetime.now()
                
                return fill
                
            except Exception as e2:
                logger.critical(f"Both brokers failed! Primary: {e}, Backup: {e2}")
                
                # This is CRITICAL - alert immediately
                alert_critical("Complete broker failure", primary_error=e, backup_error=e2)
                
                raise BothBrokersFailed(primary_error=e, backup_error=e2)
    
    def alert_failover(self, error):
        """Alert team that failover occurred"""
        message = f"""
        BROKER FAILOVER ACTIVATED
        
        Primary: {self.primary.broker_name}
        Error: {error}
        Time: {datetime.now()}
        
        Attempting backup broker: {self.backup.broker_name}
        """
        
        # Send PagerDuty alert
        pagerduty.trigger_incident(
            title="Broker Failover",
            description=message,
            severity='high'
        )
        
        # Log to audit trail
        audit_logger.log_event('FAILOVER', {
            'primary_broker': self.primary.broker_name,
            'backup_broker': self.backup.broker_name,
            'error': str(error),
            'failover_count_today': self.failover_count
        })
```

---

## Data Reconciliation

### Daily Reconciliation

**Match Our Records to Broker**:

```python
# File: src/operations/reconciliation.py

def daily_reconciliation(date: date, broker: BrokerInterface):
    """Reconcile our records to broker's records"""
    
    # Get our positions
    our_positions = position_manager.get_all_positions()
    
    # Get broker's positions
    broker_positions = broker.get_positions()
    
    # Compare
    discrepancies = []
    
    for our_pos in our_positions:
        broker_pos = find_matching_position(broker_positions, our_pos)
        
        if broker_pos is None:
            discrepancies.append({
                'type': 'missing_at_broker',
                'position': our_pos,
                'impact': 'critical'
            })
        
        elif our_pos.quantity != broker_pos.quantity:
            discrepancies.append({
                'type': 'quantity_mismatch',
                'our_qty': our_pos.quantity,
                'broker_qty': broker_pos.quantity,
                'position': our_pos,
                'impact': 'high'
            })
    
    # Check for positions at broker we don't have
    for broker_pos in broker_positions:
        if not find_matching_position(our_positions, broker_pos):
            discrepancies.append({
                'type': 'unexpected_position',
                'position': broker_pos,
                'impact': 'critical'
            })
    
    # Log results
    if discrepancies:
        logger.error("Reconciliation failed", discrepancies=discrepancies)
        alert_ops("RECONCILIATION FAILURE", discrepancies)
        return False
    else:
        logger.info("Reconciliation successful")
        return True
```

**Automated Resolution:**
```python
def resolve_discrepancy(discrepancy):
    """Attempt automatic resolution"""
    
    if discrepancy['type'] == 'quantity_mismatch':
        # Check if trade is pending
        pending = check_pending_trades(discrepancy['position'])
        if pending:
            # Wait for trade to settle
            return 'PENDING_SETTLEMENT'
    
    if discrepancy['type'] == 'missing_at_broker':
        # Check if position was assigned/exercised
        if check_for_exercise(discrepancy['position']):
            # Update our records
            handle_exercise(discrepancy['position'])
            return 'RESOLVED_EXERCISE'
    
    # Cannot auto-resolve
    return 'MANUAL_REVIEW_REQUIRED'
```

---

## Options-Specific Considerations

### Assignment Handling

**Check for Assignments:**

```python
def check_assignments(broker: BrokerInterface):
    """Check if any short options were assigned"""
    
    # Get yesterday's short options
    yesterday_shorts = db.query("""
        SELECT * FROM positions
        WHERE position_type IN ('short_put', 'covered_call')
          AND status = 'open'
          AND expiration <= CURRENT_DATE
    """)
    
    # Get today's equity positions
    current_equities = broker.get_positions()
    
    # Look for new shares (assignments)
    for short in yesterday_shorts:
        # Check if we now own shares
        shares = get_shares_for_asset(current_equities, short.asset)
        
        if shares > 0 and not had_shares_yesterday(short.asset):
            # Assignment detected!
            handle_assignment(short, shares)

def handle_assignment(short_option: Position, shares_received: int):
    """Process an assignment"""
    
    logger.info(
        "assignment_detected",
        symbol=short_option.symbol,
        strike=short_option.strike,
        shares=shares_received,
        cost=short_option.strike * shares_received
    )
    
    # Update positions
    position_manager.remove_position(short_option.id)
    position_manager.add_shares(short_option.asset, shares_received)
    
    # Immediately sell covered call (wheel)
    sell_covered_call(short_option.asset, shares_received)
    
    # Log for compliance
    audit_logger.log_trade({
        'action': 'ASSIGNMENT',
        'option': short_option.symbol,
        'shares_received': shares_received,
        'cost': short_option.strike * shares_received
    })
```

### Expiration Management

**Auto-Exercise Check:**

```python
def check_expirations(expiration_date: date):
    """Process options expiring today"""
    
    expiring = db.query("""
        SELECT * FROM positions
        WHERE expiration = %s
          AND status = 'open'
    """, [expiration_date])
    
    for position in expiring:
        asset_price = get_closing_price(position.asset, expiration_date)
        
        if position.position_type == 'short_put':
            if asset_price < position.strike:
                # In the money - will be assigned
                logger.warning(
                    "assignment_expected",
                    symbol=position.symbol,
                    strike=position.strike,
                    asset_price=asset_price,
                    estimated_shares=position.quantity * 100
                )
                # Reserve cash for assignment
                reserve_cash(position.strike * position.quantity * 100)
            else:
                # Expired worthless - we keep premium
                logger.info(
                    "option_expired_worthless",
                    symbol=position.symbol,
                    premium_kept=position.premium_received
                )
                close_position(position, reason='EXPIRED_WORTHLESS')
        
        elif position.position_type == 'covered_call':
            if asset_price >= position.strike:
                # In the money - shares will be called away
                logger.info(
                    "call_assignment_expected",
                    symbol=position.symbol,
                    shares_sold=position.quantity * 100,
                    proceeds=position.strike * position.quantity * 100
                )
            else:
                # Expired worthless - we keep premium AND shares
                logger.info(
                    "call_expired_worthless",
                    symbol=position.symbol,
                    premium_kept=position.premium_received,
                    shares_retained=position.quantity * 100
                )
                close_position(position, reason='EXPIRED_WORTHLESS')
```

---

## Testing

### Paper Trading Tests

```python
# File: tests/integration/test_broker_integration.py

def test_alpaca_option_round_trip():
    """Test selling and buying back an option"""
    
    broker = AlpacaBroker(paper=True)
    
    # Sell a put
    symbol = create_option_symbol('SPY', 400, 'P', date.today() + timedelta(days=14))
    
    sell_fill = broker.submit_order(
        symbol=symbol,
        quantity=1,
        side='sell',
        limit_price=8.50
    )
    
    assert sell_fill.quantity == 1
    assert sell_fill.broker == 'alpaca'
    
    # Wait a bit
    time.sleep(5)
    
    # Buy it back
    buy_fill = broker.submit_order(
        symbol=symbol,
        quantity=1,
        side='buy',
        limit_price=9.00  # Willing to pay more to close
    )
    
    assert buy_fill.quantity == 1
    
    # Calculate P&L
    pnl = (sell_fill.fill_price - buy_fill.fill_price) * 100
    print(f"Round-trip P&L: ${pnl:.2f}")

def test_broker_failover():
    """Test automatic failover to backup broker"""
    
    manager = BrokerManager()
    
    # Simulate primary failure
    manager.primary.simulate_failure()
    
    # Try to execute order (should failover)
    fill = manager.submit_order('SPY241115P00400000', 1, 'sell', 8.50)
    
    assert fill.broker == 'interactive_brokers'  # Used backup
    assert manager.failover_count == 1
```

---

## Production Deployment

### Broker Account Setup

**Alpaca Production:**
```
1. Verify live account approved
2. Fund account ($10K minimum recommended)
3. Enable options level 2 or higher
4. Generate production API keys
5. Store keys in AWS Secrets Manager
6. Configure alerts for account issues
7. Set up 2FA on account
```

**IB Production:**
```
1. Verify live account approved
2. Fund account ($10K minimum)
3. Enable options permissions
4. Install IB Gateway on production server
5. Configure auto-login (secure credentials)
6. Set up connection monitoring
7. Configure 2FA on account
```

### Connection Monitoring

```python
# Monitor broker connections continuously

async def monitor_broker_health():
    """Check broker connectivity every 30 seconds"""
    
    while True:
        try:
            # Check Alpaca
            alpaca_account = alpaca_broker.get_account_info()
            alpaca_health.set(1)  # Prometheus metric
            
            # Check IB
            ib_account = ib_broker.get_account_info()
            ib_health.set(1)
            
        except Exception as e:
            logger.error(f"Broker health check failed: {e}")
            
            if 'alpaca' in str(e).lower():
                alpaca_health.set(0)
                alert_ops("Alpaca connection lost")
            
            if 'ib' in str(e).lower():
                ib_health.set(0)
                alert_ops("IB connection lost")
        
        await asyncio.sleep(30)
```

---

## Cost Optimization

### Commission Comparison

**Alpaca:**
- Options: **$0.00** per contract
- Stocks: **$0.00** per share
- Regulatory fees: ~$0.02 per contract (unavoidable)
- **Total per trade**: ~$0.02

**Interactive Brokers:**
- Options: **$0.65** per contract
- Stocks: **$0.005** per share (min $1)
- Regulatory fees: ~$0.02 per contract
- **Total per trade**: ~$0.67 per contract

**Annual Cost Estimate**:
```
Strategy executes ~260 trades/year (5/week * 52 weeks)

Alpaca only:
260 trades × $0.02 = $5.20/year

If using IB:
260 trades × $0.67 = $174.20/year

Savings with Alpaca: $169/year (at small scale)
At $100M AUM (higher volume): $500-1000/year savings
```

**Strategy**: Use Alpaca primary, IB for backup only

---

## Best Practices

### Do's

✓ Always validate fills match orders  
✓ Log every API call  
✓ Handle all exceptions gracefully  
✓ Test failover regularly  
✓ Monitor connection health  
✓ Reconcile daily  
✓ Keep API keys secure  
✓ Rate limit API calls  
✓ Have manual override procedures  

### Don'ts

✗ Never hardcode API keys  
✗ Don't assume orders always fill  
✗ Don't ignore reconciliation errors  
✗ Don't exceed rate limits  
✗ Don't deploy untested code to production  
✗ Don't rely on single broker  
✗ Don't skip error handling  
✗ Don't forget to log  

---

## Conclusion

**Reliable broker integration is CRITICAL for fund success.**

**Our Approach:**
- ✓ Dual brokers (Alpaca + IB)
- ✓ Automatic failover
- ✓ Comprehensive error handling
- ✓ Daily reconciliation
- ✓ Extensive logging
- ✓ Paper trading first
- ✓ Monitoring and alerts

**This provides institutional-grade reliability even with retail brokers.**

---

*Document Version 1.0*  
*Last Updated: October 2025*  
*Technical Specification*

