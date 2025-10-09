# Logging & Audit Trail Specification

**Institutional-Grade Logging for Compliance and Operations**

---

## Requirements Overview

### Regulatory Requirements (SEC Rule 204-2)

**Must Maintain Records of:**
1. All trade orders (7 years)
2. Trade confirmations and statements (7 years)
3. Written communications (7 years)
4. Financial records (7 years)
5. Compliance documents (life of firm + 7 years)

**Accessibility**: First 2 years must be immediately accessible

### Our Requirements

**Beyond Regulatory Minimum:**
- Immutable audit trail (cannot be altered)
- Real-time logging (sub-second timestamps)
- Structured format (machine-readable)
- Searchable (for investigations)
- Distributed (multiple locations)
- Monitored (alerts on anomalies)

---

## Log Architecture

### Multi-Tier Logging System

```
Application Events
        ↓
┌───────────────────┐
│  Log Aggregator   │
│   (structlog)     │
└────────┬──────────┘
         ↓
    ┌────┴────┐
    ↓         ↓
┌─────┐   ┌──────┐
│File │   │ DB   │
│Logs │   │Audit │
└──┬──┘   └───┬──┘
   ↓          ↓
┌──────┐  ┌─────────┐
│ S3   │  │Immutable│
│Archive│  │ Table  │
└──────┘  └─────────┘
```

### Log Levels

```python
CRITICAL = 50  # System failure, trading halted
ERROR    = 40  # Failed trades, data errors
WARNING  = 30  # Unusual conditions, hedge triggers
INFO     = 20  # Normal operations, trades executed
DEBUG    = 10  # Detailed diagnostics (dev only)
```

**Production Log Level**: INFO (WARNING in production initially)

---

## Log Types & Formats

### 1. Trade Logs

**Purpose**: Record every trade execution (regulatory requirement)

**Format** (JSON):
```json
{
  "timestamp": "2024-10-08T10:32:45.123456Z",
  "log_type": "TRADE",
  "trade_id": "TRD-2024-10-08-0001",
  "event": "trade_executed",
  "asset": "QQQ",
  "action": "SELL_PUT",
  "symbol": "QQQ241122P00400000",
  "quantity": 5,
  "strike": 400.00,
  "expiration": "2024-11-22",
  "order_type": "limit",
  "limit_price": 8.50,
  "fill_price": 8.52,
  "fill_timestamp": "2024-10-08T10:32:47.891234Z",
  "broker": "alpaca",
  "commission": 3.25,
  "premium_collected": 4260.00,
  "net_proceeds": 4256.75,
  "portfolio_nav_before": 152340.12,
  "portfolio_nav_after": 156596.87,
  "executing_user": "portfolio_manager",
  "strategy_component": "weekly_put_sale",
  "risk_check_passed": true,
  "slippage_pct": 0.0024
}
```

**Implementation**:
```python
def log_trade(trade: Trade):
    """Log trade execution with full context"""
    logger.info(
        "trade_executed",
        trade_id=trade.id,
        asset=trade.asset,
        action=trade.action,
        symbol=trade.option_symbol,
        quantity=trade.quantity,
        fill_price=trade.fill_price,
        broker=trade.broker,
        commission=trade.commission,
        pnl=trade.pnl,
        timestamp=trade.timestamp.isoformat()
    )
    
    # Also write to database (immutable)
    db.execute("""
        INSERT INTO trade_log (
            trade_id, timestamp, event_type, event_data
        ) VALUES (%s, %s, %s, %s)
    """, (trade.id, trade.timestamp, 'TRADE_EXECUTED', 
          json.dumps(trade.to_dict())))
```

### 2. Risk Logs

**Purpose**: Record all risk management decisions

**Events to Log:**
- Risk check approvals/rejections
- Circuit breaker activations
- Position limit violations
- Margin warnings
- VaR breaches

**Format**:
```json
{
  "timestamp": "2024-10-08T14:22:15.123456Z",
  "log_type": "RISK",
  "event": "circuit_breaker_triggered",
  "severity": "HIGH",
  "trigger": "drawdown_threshold",
  "current_drawdown": -0.082,
  "threshold": -0.08,
  "action_taken": "reduce_position_sizes_50pct",
  "portfolio_nav": 145230.45,
  "peak_nav": 158000.12,
  "positions_affected": 12,
  "delta_before": 8543.2,
  "delta_after": 4271.6,
  "notified": ["portfolio_manager", "cco"],
  "auto_executed": true
}
```

### 3. System Logs

**Purpose**: Monitor system health and performance

**Events to Log:**
- Application start/stop
- API calls (count, latency)
- Database queries (slow queries)
- Errors and exceptions
- Performance metrics

**Format**:
```json
{
  "timestamp": "2024-10-08T09:30:01.234567Z",
  "log_type": "SYSTEM",
  "event": "api_call",
  "api": "alpaca",
  "endpoint": "/v2/positions",
  "method": "GET",
  "status_code": 200,
  "latency_ms": 145,
  "response_size_bytes": 2048,
  "retry_count": 0,
  "success": true
}
```

### 4. NAV Logs

**Purpose**: Audit trail for valuation (compliance critical)

**Format**:
```json
{
  "timestamp": "2024-10-08T16:15:00.000000Z",
  "log_type": "NAV",
  "event": "nav_calculated",
  "nav": 156789.45,
  "cash": 35420.12,
  "equity_value": 89234.56,
  "options_value": 32134.77,
  "methodology": "black_scholes",
  "pricing_source": "market_close_prices",
  "components": {
    "SPY_shares": 200,
    "SPY_value": 80400.00,
    "QQQ_short_puts": -5,
    "QQQ_puts_value": -4250.00,
    "..."
  },
  "verified_by": "administrator",
  "verification_timestamp": "2024-10-08T18:30:00.000000Z",
  "discrepancy": 0.02
}
```

**Implementation**:
```python
def log_nav_calculation(nav_result):
    """Log NAV with full calculation details"""
    logger.info(
        "nav_calculated",
        nav=nav_result.nav,
        cash=nav_result.cash,
        methodology="black_scholes",
        components=nav_result.components,
        verified_by="administrator"
    )
    
    # Immutable database record
    db.execute("""
        INSERT INTO nav_history (
            timestamp, nav, methodology, components, verified_by
        ) VALUES (%s, %s, %s, %s, %s)
    """, (datetime.now(), nav_result.nav, "black_scholes",
          json.dumps(nav_result.components), "administrator"))
```

### 5. Compliance Logs

**Purpose**: Document compliance activities

**Events to Log:**
- Personal trading submissions
- Personal trading approvals/rejections
- Marketing material reviews
- Policy violations
- Training completions
- Annual reviews

**Format**:
```json
{
  "timestamp": "2024-10-08T11:00:00.000000Z",
  "log_type": "COMPLIANCE",
  "event": "personal_trade_preclearance",
  "employee": "john_doe",
  "security": "AAPL",
  "quantity": 100,
  "side": "buy",
  "request_id": "PCT-2024-0045",
  "reviewer": "cco",
  "decision": "approved",
  "reasoning": "no conflict, not restricted",
  "review_timestamp": "2024-10-08T11:05:23.456789Z"
}
```

---

## Implementation Details

### Structured Logging

**Configuration**:

```python
# File: src/common/logging_config.py

import structlog
import logging.config

def configure_logging(log_level='INFO', log_file='logs/trading.log'):
    """Configure structured logging for the application"""
    
    # Shared processors
    shared_processors = [
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]
    
    # Configure structlog
    structlog.configure(
        processors=shared_processors + [
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Configure standard logging
    logging.config.dictConfig({
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'json': {
                '()': structlog.stdlib.ProcessorFormatter,
                'processor': structlog.processors.JSONRenderer(),
                'foreign_pre_chain': shared_processors,
            },
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'formatter': 'json',
            },
            'file': {
                'class': 'logging.handlers.TimedRotatingFileHandler',
                'filename': log_file,
                'when': 'midnight',
                'interval': 1,
                'backupCount': 2555,  # 7 years
                'formatter': 'json',
            },
        },
        'loggers': {
            '': {
                'handlers': ['console', 'file'],
                'level': log_level,
            },
        },
    })
```

### Database Audit Log

**Schema**:

```sql
-- Immutable audit log (append-only)

CREATE TABLE audit_log (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    log_type VARCHAR(50) NOT NULL,
    event VARCHAR(100) NOT NULL,
    event_data JSONB NOT NULL,
    user_id VARCHAR(100),
    ip_address VARCHAR(45),
    session_id VARCHAR(100),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for fast searching
CREATE INDEX idx_audit_log_timestamp ON audit_log(timestamp);
CREATE INDEX idx_audit_log_type ON audit_log(log_type);
CREATE INDEX idx_audit_log_event ON audit_log(event);
CREATE INDEX idx_audit_log_user ON audit_log(user_id);

-- Prevent deletions and updates (immutable)
CREATE RULE audit_log_no_delete AS ON DELETE TO audit_log DO INSTEAD NOTHING;
CREATE RULE audit_log_no_update AS ON UPDATE TO audit_log DO INSTEAD NOTHING;

-- Only allow inserts
GRANT INSERT ON audit_log TO trading_user;
REVOKE UPDATE, DELETE ON audit_log FROM trading_user;
```

### Log Aggregation

**Collect from Multiple Sources:**

```python
# File: src/monitoring/log_aggregator.py

import logging
from datetime import datetime

class LogAggregator:
    """Aggregate logs from all components"""
    
    def __init__(self):
        self.file_logger = logging.getLogger('file')
        self.db_logger = DatabaseLogger()
        self.metrics_logger = PrometheusLogger()
    
    def log_event(self, log_type, event, data):
        """Write to all log destinations"""
        timestamp = datetime.now()
        
        log_entry = {
            'timestamp': timestamp.isoformat(),
            'log_type': log_type,
            'event': event,
            **data
        }
        
        # File log (JSON)
        self.file_logger.info(json.dumps(log_entry))
        
        # Database log (searchable)
        self.db_logger.insert(log_type, event, data)
        
        # Metrics (for monitoring)
        self.metrics_logger.increment(f'{log_type}_{event}_total')
```

---

## Compliance Requirements

### Trade Blotter

**Daily Trade Blotter** (SEC requirement):

```sql
-- Generate daily trade blotter

SELECT 
    timestamp,
    trade_id,
    asset,
    action,
    quantity,
    price,
    commission,
    broker,
    executing_user
FROM trade_log
WHERE timestamp::date = CURRENT_DATE
ORDER BY timestamp;

-- Export as CSV for compliance
COPY (SELECT ...) TO '/compliance/blotters/blotter_2024-10-08.csv' CSV HEADER;
```

**Retention**: 7 years, first 2 years readily accessible

### Position Snapshots

**End-of-Day Positions** (compliance requirement):

```python
def save_position_snapshot(date: date, positions: List[Position]):
    """Save daily position snapshot"""
    snapshot = {
        'date': date,
        'timestamp': datetime.now(),
        'positions': [p.to_dict() for p in positions],
        'total_count': len(positions),
        'total_value': sum(p.market_value for p in positions),
        'total_delta': sum(p.delta for p in positions)
    }
    
    # Database
    db.execute("""
        INSERT INTO position_snapshots (date, snapshot_data)
        VALUES (%s, %s)
    """, (date, json.dumps(snapshot)))
    
    # File backup
    filepath = f'compliance/positions/positions_{date}.json'
    with open(filepath, 'w') as f:
        json.dump(snapshot, f, indent=2)
    
    # Upload to S3
    s3_client.upload_file(filepath, 'compliance-bucket', 
                          f'positions/{date.year}/positions_{date}.json')
```

### Communications Log

**Email Archiving** (SEC requirement):

```python
# All investor emails must be archived

import email
import imaplib

class EmailArchiver:
    """Archive all business emails"""
    
    def archive_email(self, message):
        """Archive email to compliance storage"""
        
        email_record = {
            'timestamp': datetime.now(),
            'from': message['From'],
            'to': message['To'],
            'cc': message.get('Cc', ''),
            'subject': message['Subject'],
            'body': get_email_body(message),
            'attachments': list_attachments(message),
            'message_id': message['Message-ID']
        }
        
        # Store in database
        db.execute("""
            INSERT INTO email_archive (timestamp, email_data)
            VALUES (%s, %s)
        """, (email_record['timestamp'], json.dumps(email_record)))
        
        # Also store raw .eml file
        filepath = f"compliance/emails/{timestamp.date()}/{message_id}.eml"
        save_raw_email(message, filepath)
```

**Third-Party Solutions**: Smarsh, Global Relay ($3-5K/year)

---

## Audit Trail Features

### Immutability

**Append-Only Database**:
```sql
-- Prevent any modifications to audit log

ALTER TABLE audit_log ADD CONSTRAINT audit_log_immutable
    CHECK (created_at IS NOT NULL);

-- Revoke UPDATE and DELETE
REVOKE UPDATE, DELETE ON audit_log FROM ALL;
GRANT INSERT ON audit_log TO trading_app;
```

**Blockchain-Style Hash Chain** (optional but impressive):
```python
import hashlib

class HashChainLogger:
    """Create tamper-evident log chain"""
    
    def __init__(self):
        self.previous_hash = '0' * 64  # Genesis
    
    def log_event(self, event_data):
        """Log event with hash chain"""
        # Create hash of: previous_hash + event_data
        data_str = json.dumps(event_data, sort_keys=True)
        combined = self.previous_hash + data_str
        current_hash = hashlib.sha256(combined.encode()).hexdigest()
        
        # Store with hash
        log_entry = {
            **event_data,
            'previous_hash': self.previous_hash,
            'current_hash': current_hash
        }
        
        db.insert_audit_log(log_entry)
        
        # Update chain
        self.previous_hash = current_hash
        
        return current_hash
    
    def verify_chain(self):
        """Verify log chain integrity"""
        logs = db.get_all_audit_logs()
        
        for i, log in enumerate(logs):
            if i == 0:
                expected_prev = '0' * 64
            else:
                expected_prev = logs[i-1]['current_hash']
            
            if log['previous_hash'] != expected_prev:
                raise TamperingDetected(f"Log {log['id']} hash mismatch!")
        
        return True  # Chain intact
```

### Searchability

**Full-Text Search** (for investigations):

```sql
-- Add full-text search index

CREATE INDEX idx_audit_log_fts ON audit_log 
    USING gin(to_tsvector('english', event_data::text));

-- Search for specific events
SELECT * FROM audit_log
WHERE to_tsvector('english', event_data::text) @@ to_tsquery('SPY & assignment');

-- Find all trades for an asset
SELECT * FROM audit_log
WHERE log_type = 'TRADE'
  AND event_data->>'asset' = 'QQQ'
  AND timestamp >= '2024-10-01'
ORDER BY timestamp;
```

**Search Interface** (for compliance):

```python
class AuditSearchEngine:
    """Search audit logs for compliance"""
    
    def search_trades(self, 
                     asset: str = None,
                     start_date: date = None,
                     end_date: date = None,
                     action: str = None) -> List[Dict]:
        """Search trade logs"""
        
        query = "SELECT * FROM audit_log WHERE log_type = 'TRADE'"
        params = []
        
        if asset:
            query += " AND event_data->>'asset' = %s"
            params.append(asset)
        
        if start_date:
            query += " AND timestamp >= %s"
            params.append(start_date)
        
        if end_date:
            query += " AND timestamp <= %s"
            params.append(end_date)
        
        if action:
            query += " AND event_data->>'action' = %s"
            params.append(action)
        
        query += " ORDER BY timestamp"
        
        return db.query(query, params)
```

---

## Monitoring Integration

### Metrics from Logs

**Prometheus Metrics** (auto-generated from logs):

```python
from prometheus_client import Counter, Histogram, Gauge

# Define metrics
trades_total = Counter('trades_total', 'Total trades', ['asset', 'action'])
trade_latency = Histogram('trade_latency_seconds', 'Trade execution time')
nav_value = Gauge('portfolio_nav', 'Current NAV')
error_total = Counter('errors_total', 'Total errors', ['error_type'])

# Log wrapper that also updates metrics
def log_and_metric_trade(trade):
    # Log to audit trail
    audit_logger.log_trade(trade)
    
    # Update metrics
    trades_total.labels(asset=trade.asset, action=trade.action).inc()
    trade_latency.observe(trade.execution_time)
    
    # Calculate and update NAV
    new_nav = calculate_nav()
    nav_value.set(new_nav)
```

### Alerting from Logs

**Pattern-Based Alerts**:

```python
class LogMonitor:
    """Monitor logs for alert conditions"""
    
    def __init__(self):
        self.error_counts = {}
    
    def process_log(self, log_entry):
        """Process each log entry for alert conditions"""
        
        # Alert on any ERROR level
        if log_entry['level'] == 'ERROR':
            self.send_alert('high', f"Error logged: {log_entry['event']}")
        
        # Alert on rapid error increase
        if log_entry['level'] >= 'WARNING':
            self.error_counts[log_entry['event']] = \
                self.error_counts.get(log_entry['event'], 0) + 1
            
            # >5 errors of same type in 5 minutes
            if self.error_counts[log_entry['event']] > 5:
                self.send_alert('critical', 
                    f"Repeated errors: {log_entry['event']} x5")
        
        # Alert on circuit breakers
        if log_entry['event'] == 'circuit_breaker_triggered':
            self.send_alert('critical', "Circuit breaker activated!")
        
        # Alert on failed trades
        if log_entry['event'] == 'trade_failed':
            self.send_alert('high', f"Trade execution failed: {log_entry['trade_id']}")
```

---

## Storage & Retention

### Hot Storage (0-90 Days)

**Local SSD** (fast access):
- All application logs
- Trade logs
- Performance data
- Searchable via database

**Access Time**: <1 second

### Warm Storage (90 Days - 2 Years)

**AWS S3 Standard** (readily accessible):
- Compressed daily logs
- Monthly aggregated reports
- Position snapshots
- NAV calculations

**Access Time**: <5 seconds  
**Cost**: ~$25/TB/month

### Cold Storage (2-7 Years)

**AWS S3 Glacier** (compliance archive):
- All logs older than 2 years
- Still compliant with SEC (retrievable)
- Compressed and encrypted

**Access Time**: 3-5 hours (Glacier retrieval)  
**Cost**: ~$4/TB/month

**Lifecycle Policy**:
```python
# Automatic transition to cheaper storage

s3_lifecycle = {
    'Rules': [
        {
            'Id': 'archive-old-logs',
            'Status': 'Enabled',
            'Transitions': [
                {
                    'Days': 90,
                    'StorageClass': 'STANDARD_IA'  # Infrequent Access
                },
                {
                    'Days': 730,  # 2 years
                    'StorageClass': 'GLACIER'
                }
            ],
            'Expiration': {
                'Days': 2555  # 7 years
            }
        }
    ]
}
```

---

## Query & Analysis Tools

### Log Analysis Scripts

**Common Queries**:

```python
# File: scripts/log_analysis.py

class LogAnalyzer:
    """Analyze logs for insights"""
    
    def daily_trading_summary(self, date: date):
        """Generate daily trading summary"""
        trades = db.query("""
            SELECT 
                event_data->>'asset' as asset,
                event_data->>'action' as action,
                COUNT(*) as count,
                SUM((event_data->>'pnl')::numeric) as total_pnl,
                AVG((event_data->>'slippage_pct')::numeric) as avg_slippage
            FROM audit_log
            WHERE log_type = 'TRADE'
              AND timestamp::date = %s
            GROUP BY asset, action
        """, [date])
        
        return pd.DataFrame(trades)
    
    def risk_event_history(self, days: int = 30):
        """Get recent risk events"""
        events = db.query("""
            SELECT 
                timestamp,
                event,
                event_data->>'severity' as severity,
                event_data->>'action_taken' as action
            FROM audit_log
            WHERE log_type = 'RISK'
              AND timestamp > NOW() - INTERVAL '%s days'
            ORDER BY timestamp DESC
        """, [days])
        
        return pd.DataFrame(events)
    
    def execution_quality_report(self, start_date: date, end_date: date):
        """Analyze execution quality"""
        metrics = db.query("""
            SELECT 
                AVG((event_data->>'latency_ms')::numeric) as avg_latency,
                PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY (event_data->>'latency_ms')::numeric) as p95_latency,
                AVG((event_data->>'slippage_pct')::numeric) as avg_slippage,
                SUM(CASE WHEN event = 'trade_executed' THEN 1 ELSE 0 END) as success_count,
            SUM(CASE WHEN event = 'trade_failed' THEN 1 ELSE 0 END) as failure_count
            FROM audit_log
            WHERE log_type = 'SYSTEM'
              AND timestamp BETWEEN %s AND %s
        """, [start_date, end_date])
        
        return metrics
```

---

## Real-Time Monitoring

### Live Log Streaming

**WebSocket Stream for Dashboards**:

```python
import asyncio
import websockets

class LogStreamer:
    """Stream logs to monitoring dashboard"""
    
    def __init__(self):
        self.subscribers = set()
    
    async def publish_log(self, log_entry):
        """Publish log to all subscribers"""
        message = json.dumps(log_entry)
        
        # Send to all connected websockets
        if self.subscribers:
            await asyncio.wait([
                subscriber.send(message)
                for subscriber in self.subscribers
            ])
    
    async def subscribe(self, websocket):
        """Subscribe to log stream"""
        self.subscribers.add(websocket)
        try:
            await websocket.wait_closed()
        finally:
            self.subscribers.remove(websocket)
```

### Dashboard Integration

**Grafana Dashboard** (from logs):

```
Dashboard: Trading Operations
├── Panel: Live Trade Feed (last 50 trades)
├── Panel: Error Rate (last hour)
├── Panel: API Latency (p50, p95, p99)
├── Panel: Execution Success Rate
└── Panel: Recent Alerts
```

---

## Alert Configuration

### Alert Rules

```yaml
# File: config/alerts.yaml

alerts:
  critical:
    - name: trading_halted
      condition: log_type == 'RISK' AND event == 'circuit_breaker_triggered'
      notification: [pagerduty, sms, phone]
      response_time: immediate
    
    - name: execution_failure
      condition: log_type == 'TRADE' AND event == 'trade_failed'
      notification: [pagerduty, sms]
      response_time: 5_minutes
  
  high:
    - name: high_slippage
      condition: log_type == 'TRADE' AND slippage_pct > 0.02
      notification: [email, slack]
      response_time: 30_minutes
    
    - name: position_limit_breach
      condition: log_type == 'RISK' AND event == 'limit_exceeded'
      notification: [email, slack]
      response_time: 15_minutes
  
  medium:
    - name: slow_api
      condition: log_type == 'SYSTEM' AND latency_ms > 1000
      notification: [slack]
      response_time: 1_hour
  
  low:
    - name: daily_summary
      condition: timestamp == '16:30:00'  # After NAV calc
      notification: [email]
      response_time: eod
```

---

## Compliance Audit Support

### Pre-Audit Preparation

**Generate Compliance Package**:

```python
def generate_compliance_package(start_date, end_date):
    """
    Generate all records for compliance audit
    
    Typically for SEC examination
    """
    
    package = {
        'trade_blotters': export_trade_blotters(start_date, end_date),
        'position_snapshots': export_position_snapshots(start_date, end_date),
        'nav_calculations': export_nav_history(start_date, end_date),
        'fee_calculations': export_fee_calculations(start_date, end_date),
        'risk_events': export_risk_events(start_date, end_date),
        'compliance_logs': export_compliance_logs(start_date, end_date),
        'email_archive': export_emails(start_date, end_date)
    }
    
    # Create ZIP file
    zip_path = f'compliance/audits/audit_package_{start_date}_{end_date}.zip'
    create_zip(package, zip_path)
    
    return zip_path
```

### Search Examples

**Common Compliance Queries**:

```sql
-- Find all trades for a specific date
SELECT * FROM audit_log 
WHERE log_type = 'TRADE' 
  AND timestamp::date = '2024-10-08';

-- Find all position changes for an asset
SELECT * FROM audit_log
WHERE event_data->>'asset' = 'SPY'
  AND event IN ('position_opened', 'position_closed', 'assignment')
ORDER BY timestamp;

-- Find all circuit breaker activations
SELECT * FROM audit_log
WHERE event = 'circuit_breaker_triggered'
ORDER BY timestamp DESC;

-- Calculate total premium collected
SELECT 
    SUM((event_data->>'premium_collected')::numeric) as total_premium
FROM audit_log
WHERE event_data->>'action' IN ('SELL_PUT', 'SELL_CALL')
  AND timestamp >= '2024-01-01';
```

---

## Data Retention Schedule

| Record Type | Hot Storage | Warm Storage | Cold Storage | Total Retention |
|-------------|-------------|--------------|--------------|-----------------|
| Trade logs | 90 days | 2 years | 5 years | 7 years |
| NAV calculations | 90 days | 2 years | 5 years | 7 years |
| Position snapshots | 90 days | 2 years | 5 years | 7 years |
| System logs | 30 days | 1 year | 0 | 1 year |
| Email archive | 90 days | 2 years | 5 years | 7 years |
| Compliance docs | Permanent | Permanent | Permanent | Permanent |

**Automated Archival**:
```bash
#!/bin/bash
# Cron job: Daily at 2 AM

# Move logs >90 days to S3
find /logs -name "*.log" -mtime +90 -exec \
    aws s3 cp {} s3://compliance-archive/logs/ \;

# Move logs >730 days to Glacier
aws s3 ls s3://compliance-archive/logs/ | while read -r line; do
    file=$(echo $line | awk '{print $4}')
    age=$((($(date +%s) - $(date -d "$(echo $line | awk '{print $1}')" +%s)) / 86400))
    
    if [ $age -gt 730 ]; then
        aws s3 cp s3://compliance-archive/logs/$file \
                  s3://compliance-archive-glacier/logs/$file \
                  --storage-class GLACIER
    fi
done
```

---

## Disaster Recovery

### Log Backup Strategy

**Hourly Incremental**:
```bash
# Cron: Every hour
0 * * * * pg_dump trading_db | gzip > /backups/hourly/db_$(date +\%Y\%m\%d_\%H).sql.gz

# Sync to S3
0 * * * * aws s3 sync /backups/hourly/ s3://backups/hourly/ --delete
```

**Daily Full Backup**:
```bash
# Cron: Daily at 1 AM
0 1 * * * pg_dump trading_db | gzip > /backups/daily/db_$(date +\%Y\%m\%d).sql.gz
0 1 * * * tar czf /backups/daily/logs_$(date +\%Y\%m\%d).tar.gz /logs/
0 2 * * * aws s3 sync /backups/daily/ s3://backups/daily/
```

**Weekly Archive**:
```bash
# Cron: Sunday at 3 AM
0 3 * * 0 tar czf /backups/weekly/full_backup_$(date +\%Y\%W).tar.gz /opt/trading/
0 4 * * 0 aws s3 cp /backups/weekly/full_backup_$(date +\%Y\%W).tar.gz \
                     s3://backups/weekly/
```

### Recovery Procedures

**Restore from Backup**:

```bash
#!/bin/bash
# restore_from_backup.sh

BACKUP_DATE=$1  # Format: 2024-10-08

echo "Restoring from backup: $BACKUP_DATE"

# 1. Stop trading
sudo systemctl stop trading-engine

# 2. Download backup from S3
aws s3 cp s3://backups/daily/db_${BACKUP_DATE}.sql.gz /tmp/

# 3. Restore database
gunzip /tmp/db_${BACKUP_DATE}.sql.gz
psql -U postgres -d trading_db < /tmp/db_${BACKUP_DATE}.sql

# 4. Verify restoration
psql -U postgres -d trading_db -c "SELECT COUNT(*) FROM trades;"

# 5. Restart trading (WITH APPROVAL)
echo "⚠️  MANUAL VERIFICATION REQUIRED BEFORE RESTART"
echo "Verify data integrity, then run: sudo systemctl start trading-engine"
```

---

## Performance Optimization

### Log Batching

**Reduce Database Load**:

```python
class BatchLogger:
    """Batch logs for efficiency"""
    
    def __init__(self, batch_size=100, flush_interval=5):
        self.buffer = []
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self.last_flush = time.time()
    
    def log(self, log_entry):
        """Add to batch"""
        self.buffer.append(log_entry)
        
        # Flush if batch full or interval elapsed
        if len(self.buffer) >= self.batch_size or \
           time.time() - self.last_flush > self.flush_interval:
            self.flush()
    
    def flush(self):
        """Write batch to database"""
        if not self.buffer:
            return
        
        # Bulk insert
        db.executemany("""
            INSERT INTO audit_log (timestamp, log_type, event, event_data)
            VALUES (%s, %s, %s, %s)
        """, [(l['timestamp'], l['log_type'], l['event'], json.dumps(l))
              for l in self.buffer])
        
        self.buffer = []
        self.last_flush = time.time()
```

### Async Logging

**Non-Blocking Logs**:

```python
import asyncio
from queue import Queue
from threading import Thread

class AsyncLogger:
    """Asynchronous logging to not block trading"""
    
    def __init__(self):
        self.queue = Queue()
        self.worker = Thread(target=self._process_queue, daemon=True)
        self.worker.start()
    
    def log(self, log_entry):
        """Add to queue (returns immediately)"""
        self.queue.put(log_entry)
    
    def _process_queue(self):
        """Background worker processes queue"""
        while True:
            log_entry = self.queue.get()
            
            # Write to file
            file_logger.info(json.dumps(log_entry))
            
            # Write to database
            db_logger.insert(log_entry)
            
            self.queue.task_done()
```

---

## Compliance Certification

### Daily Attestation

```python
# At end of each day, CCO certifies logs

def certify_daily_logs(date: date, cco_signature: str):
    """CCO certification that logs are complete and accurate"""
    
    certification = {
        'date': date,
        'certified_by': cco_signature,
        'certification_timestamp': datetime.now(),
        'log_count': count_logs_for_date(date),
        'trades_logged': count_trades(date),
        'trades_reconciled': verify_trade_reconciliation(date),
        'nav_calculated': verify_nav_exists(date),
        'statement': 'I certify that the audit logs for this date are complete and accurate'
    }
    
    db.execute("""
        INSERT INTO log_certifications (date, certification_data)
        VALUES (%s, %s)
    """, (date, json.dumps(certification)))
```

---

## Conclusion

**Logging is not optional - it's the foundation of compliance and operations.**

**Our Approach:**
- ✓ **Comprehensive** - Log everything that matters
- ✓ **Immutable** - Cannot be altered after creation
- ✓ **Searchable** - Find anything quickly
- ✓ **Compliant** - Meets all SEC requirements
- ✓ **Monitored** - Real-time alerts
- ✓ **Retained** - 7+ years
- ✓ **Backed Up** - Multiple locations

**Investment**: $10-15K for tools + $5K/month storage  
**Payoff**: Clean audits, quick investigations, regulatory confidence

---

*Document Version 1.0*  
*Last Updated: October 2025*  
*Technical Specification*

