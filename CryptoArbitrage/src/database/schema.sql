-- Crypto Arbitrage System Database Schema
-- SQLite Database for trade logging and analytics

-- ============================================================
-- ARBITRAGE OPPORTUNITIES
-- ============================================================

CREATE TABLE IF NOT EXISTS opportunities (
    id TEXT PRIMARY KEY,
    detected_at DATETIME NOT NULL,
    
    -- Trading pair
    pair TEXT NOT NULL,
    base_asset TEXT NOT NULL,
    quote_asset TEXT NOT NULL,
    
    -- Buy side
    buy_exchange TEXT NOT NULL,
    buy_price REAL NOT NULL,
    buy_fee_pct REAL NOT NULL,
    
    -- Sell side  
    sell_exchange TEXT NOT NULL,
    sell_price REAL NOT NULL,
    sell_fee_pct REAL NOT NULL,
    
    -- Spread calculations
    gross_spread_pct REAL NOT NULL,
    fees_total_pct REAL NOT NULL,
    net_spread_pct REAL NOT NULL,
    
    -- Execution parameters
    max_quantity REAL NOT NULL,
    estimated_profit_usd REAL NOT NULL,
    
    -- Quality metrics
    price_staleness_ms REAL,
    estimated_slippage_pct REAL,
    confidence_score REAL,
    
    -- Execution status
    is_executed BOOLEAN DEFAULT 0,
    execution_id TEXT,
    
    -- Lifecycle tracking
    first_seen_at DATETIME,
    last_seen_at DATETIME,
    expired_at DATETIME,
    opportunity_duration_ms REAL,
    times_seen INTEGER DEFAULT 1,
    
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_opp_detected ON opportunities(detected_at DESC);
CREATE INDEX IF NOT EXISTS idx_opp_pair ON opportunities(pair, detected_at);
CREATE INDEX IF NOT EXISTS idx_opp_spread ON opportunities(net_spread_pct DESC);
CREATE INDEX IF NOT EXISTS idx_opp_executed ON opportunities(is_executed, detected_at);
CREATE INDEX IF NOT EXISTS idx_opp_duration ON opportunities(opportunity_duration_ms DESC);

-- ============================================================
-- ARBITRAGE EXECUTIONS
-- ============================================================

CREATE TABLE IF NOT EXISTS executions (
    id TEXT PRIMARY KEY,
    opportunity_id TEXT NOT NULL,
    started_at DATETIME NOT NULL,
    completed_at DATETIME,
    
    -- Trade details
    pair TEXT NOT NULL,
    quantity REAL NOT NULL,
    
    -- Buy leg
    buy_exchange TEXT NOT NULL,
    buy_order_id TEXT NOT NULL,
    buy_status TEXT NOT NULL,
    buy_filled_price REAL,
    buy_filled_quantity REAL DEFAULT 0,
    buy_fee REAL DEFAULT 0,
    
    -- Sell leg
    sell_exchange TEXT NOT NULL,
    sell_order_id TEXT NOT NULL,
    sell_status TEXT NOT NULL,
    sell_filled_price REAL,
    sell_filled_quantity REAL DEFAULT 0,
    sell_fee REAL DEFAULT 0,
    
    -- Results
    is_successful BOOLEAN DEFAULT 0,
    realized_profit_usd REAL,
    realized_spread_pct REAL,
    execution_time_ms REAL,
    
    -- Risk tracking
    had_failures BOOLEAN DEFAULT 0,
    failure_reason TEXT,
    required_reversal BOOLEAN DEFAULT 0,
    
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (opportunity_id) REFERENCES opportunities(id)
);

CREATE INDEX IF NOT EXISTS idx_exec_time ON executions(started_at DESC);
CREATE INDEX IF NOT EXISTS idx_exec_pair ON executions(pair, started_at);
CREATE INDEX IF NOT EXISTS idx_exec_success ON executions(is_successful, started_at);
CREATE INDEX IF NOT EXISTS idx_exec_profit ON executions(realized_profit_usd DESC);

-- ============================================================
-- EXCHANGE BALANCES
-- ============================================================

CREATE TABLE IF NOT EXISTS exchange_balances (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    exchange TEXT NOT NULL,
    timestamp DATETIME NOT NULL,
    
    -- Balances (JSON stored as TEXT)
    balances_json TEXT NOT NULL,
    
    -- USD total
    total_usd REAL NOT NULL,
    
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_balance_time ON exchange_balances(exchange, timestamp DESC);

-- ============================================================
-- PRICE SNAPSHOTS
-- ============================================================

CREATE TABLE IF NOT EXISTS price_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME NOT NULL,
    pair TEXT NOT NULL,
    exchange TEXT NOT NULL,
    
    bid REAL NOT NULL,
    ask REAL NOT NULL,
    mid REAL NOT NULL,
    
    volume_24h REAL,
    latency_ms REAL,
    
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_price_time ON price_snapshots(pair, exchange, timestamp DESC);

-- ============================================================
-- PERFORMANCE METRICS
-- ============================================================

CREATE TABLE IF NOT EXISTS daily_performance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date DATE UNIQUE NOT NULL,
    
    -- Opportunity metrics
    opportunities_detected INTEGER DEFAULT 0,
    opportunities_executed INTEGER DEFAULT 0,
    execution_rate REAL DEFAULT 0,
    
    -- Financial metrics
    total_profit_usd REAL DEFAULT 0,
    total_fees_usd REAL DEFAULT 0,
    net_profit_usd REAL DEFAULT 0,
    roi_pct REAL DEFAULT 0,
    
    -- Execution quality
    avg_execution_time_ms REAL DEFAULT 0,
    avg_slippage_pct REAL DEFAULT 0,
    success_rate REAL DEFAULT 0,
    
    -- Trading activity
    total_trades INTEGER DEFAULT 0,
    total_volume_usd REAL DEFAULT 0,
    
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_perf_date ON daily_performance(date DESC);

-- ============================================================
-- SYSTEM LOGS
-- ============================================================

CREATE TABLE IF NOT EXISTS system_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME NOT NULL,
    level TEXT NOT NULL,
    component TEXT NOT NULL,
    message TEXT NOT NULL,
    data TEXT,
    
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_logs_time ON system_logs(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_logs_level ON system_logs(level, timestamp DESC);

-- ============================================================
-- LATENCY METRICS
-- ============================================================

CREATE TABLE IF NOT EXISTS latency_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME NOT NULL,
    exchange TEXT NOT NULL,
    
    -- Latency measurements
    rest_api_latency_ms REAL,
    websocket_latency_ms REAL,
    order_submission_latency_ms REAL,
    
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_latency_time ON latency_metrics(exchange, timestamp DESC);

