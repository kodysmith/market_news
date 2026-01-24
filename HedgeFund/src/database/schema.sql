-- Trading System Database Schema
-- PostgreSQL 15+
-- Purpose: Store all trading data, positions, audit logs, NAV history

-- ============================================================
-- TRADES TABLE (Immutable Audit Log)
-- ============================================================

CREATE TABLE IF NOT EXISTS trades (
    id BIGSERIAL PRIMARY KEY,
    trade_id VARCHAR(50) UNIQUE NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    
    -- Trade details
    trade_type VARCHAR(50) NOT NULL,  -- 'option', 'stock', 'assignment'
    asset VARCHAR(10) NOT NULL,
    action VARCHAR(50) NOT NULL,      -- 'SELL_PUT', 'BUY_CALL', etc.
    
    -- Quantities and prices
    quantity INTEGER NOT NULL,
    fill_price NUMERIC(12, 4) NOT NULL,
    commission NUMERIC(10, 2) DEFAULT 0.00,
    
    -- Options-specific
    strike NUMERIC(10, 2),
    expiration DATE,
    option_type CHAR(1),  -- 'C' or 'P'
    option_symbol VARCHAR(50),
    dte INTEGER,
    
    -- Execution details
    broker VARCHAR(50) NOT NULL,
    order_id VARCHAR(100),
    fill_timestamp TIMESTAMPTZ,
    
    -- Financial tracking
    premium_collected NUMERIC(12, 2),
    pnl NUMERIC(12, 2),
    
    -- Portfolio impact
    nav_before NUMERIC(12, 2),
    nav_after NUMERIC(12, 2),
    delta_before NUMERIC(10, 2),
    delta_after NUMERIC(10, 2),
    
    -- Metadata
    strategy_component VARCHAR(100),
    reasoning TEXT,
    metadata JSONB,
    
    -- Audit
    created_at TIMESTAMPTZ DEFAULT NOW(),
    created_by VARCHAR(100) DEFAULT 'trading_system'
);

-- Indexes for fast querying
CREATE INDEX idx_trades_timestamp ON trades(timestamp);
CREATE INDEX idx_trades_asset ON trades(asset);
CREATE INDEX idx_trades_action ON trades(action);
CREATE INDEX idx_trades_trade_type ON trades(trade_type);
CREATE INDEX idx_trades_expiration ON trades(expiration) WHERE expiration IS NOT NULL;

-- Prevent modifications (immutable audit log)
CREATE RULE trades_no_delete AS ON DELETE TO trades DO INSTEAD NOTHING;
CREATE RULE trades_no_update AS ON UPDATE TO trades DO INSTEAD NOTHING;

-- ============================================================
-- POSITIONS TABLE (Current Holdings)
-- ============================================================

CREATE TABLE IF NOT EXISTS positions (
    id BIGSERIAL PRIMARY KEY,
    position_id VARCHAR(50) UNIQUE NOT NULL,
    
    -- Position details
    asset VARCHAR(10) NOT NULL,
    position_type VARCHAR(50) NOT NULL,  -- 'short_put', 'covered_call', 'shares', etc.
    quantity INTEGER NOT NULL,
    
    -- Entry details
    entry_price NUMERIC(12, 4) NOT NULL,
    entry_date TIMESTAMPTZ NOT NULL,
    
    -- Options-specific
    strike NUMERIC(10, 2),
    expiration DATE,
    option_type CHAR(1),
    option_symbol VARCHAR(50),
    dte INTEGER,
    
    -- Current values (updated real-time)
    current_price NUMERIC(12, 4),
    market_value NUMERIC(12, 2),
    unrealized_pnl NUMERIC(12, 2),
    
    -- Greeks (for options)
    delta NUMERIC(10, 4) DEFAULT 0,
    gamma NUMERIC(10, 6) DEFAULT 0,
    theta NUMERIC(10, 4) DEFAULT 0,
    vega NUMERIC(10, 4) DEFAULT 0,
    
    -- Status
    status VARCHAR(20) DEFAULT 'open',  -- 'open', 'closed', 'assigned', 'expired'
    close_date TIMESTAMPTZ,
    close_price NUMERIC(12, 4),
    realized_pnl NUMERIC(12, 2),
    
    -- Metadata
    notes TEXT,
    metadata JSONB,
    
    -- Timestamps
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_positions_asset ON positions(asset);
CREATE INDEX idx_positions_status ON positions(status);
CREATE INDEX idx_positions_type ON positions(position_type);
CREATE INDEX idx_positions_expiration ON positions(expiration) WHERE expiration IS NOT NULL;

-- Update timestamp trigger
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_positions_updated_at BEFORE UPDATE ON positions
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- ============================================================
-- NAV HISTORY (Daily Valuations)
-- ============================================================

CREATE TABLE IF NOT EXISTS nav_history (
    timestamp TIMESTAMPTZ PRIMARY KEY,
    nav NUMERIC(12, 2) NOT NULL,
    
    -- NAV components
    cash NUMERIC(12, 2) NOT NULL,
    equity_value NUMERIC(12, 2) DEFAULT 0,
    options_value NUMERIC(12, 2) DEFAULT 0,
    
    -- Risk metrics
    total_delta NUMERIC(10, 2) DEFAULT 0,
    margin_used NUMERIC(12, 2) DEFAULT 0,
    
    -- Valuation details
    methodology VARCHAR(50) DEFAULT 'black_scholes',
    pricing_source VARCHAR(50) DEFAULT 'market_close',
    
    -- Verification
    verified_by VARCHAR(100),
    verification_timestamp TIMESTAMPTZ,
    discrepancy NUMERIC(10, 6) DEFAULT 0,
    
    -- Components breakdown
    components JSONB,
    
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_nav_history_date ON nav_history(DATE(timestamp));

-- ============================================================
-- AUDIT LOG (Comprehensive Event Log)
-- ============================================================

CREATE TABLE IF NOT EXISTS audit_log (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    
    -- Event classification
    log_type VARCHAR(50) NOT NULL,  -- 'TRADE', 'RISK', 'SYSTEM', 'COMPLIANCE', 'NAV'
    event VARCHAR(100) NOT NULL,     -- Specific event name
    severity VARCHAR(20) DEFAULT 'INFO',  -- 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'
    
    -- Event data (flexible JSONB for different event types)
    event_data JSONB NOT NULL,
    
    -- Audit trail
    user_id VARCHAR(100),
    ip_address VARCHAR(45),
    session_id VARCHAR(100),
    
    -- Immutability support (hash chain)
    previous_hash VARCHAR(64),
    current_hash VARCHAR(64),
    
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for searching
CREATE INDEX idx_audit_log_timestamp ON audit_log(timestamp);
CREATE INDEX idx_audit_log_type ON audit_log(log_type);
CREATE INDEX idx_audit_log_event ON audit_log(event);
CREATE INDEX idx_audit_log_severity ON audit_log(severity);

-- GIN index for JSONB searching
CREATE INDEX idx_audit_log_event_data ON audit_log USING gin(event_data);

-- Prevent modifications (immutable audit log)
CREATE RULE audit_log_no_delete AS ON DELETE TO audit_log DO INSTEAD NOTHING;
CREATE RULE audit_log_no_update AS ON UPDATE TO audit_log DO INSTEAD NOTHING;

-- ============================================================
-- PORTFOLIO SNAPSHOTS (Daily State)
-- ============================================================

CREATE TABLE IF NOT EXISTS portfolio_snapshots (
    snapshot_date DATE NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    
    -- Aggregate metrics
    nav NUMERIC(12, 2) NOT NULL,
    cash NUMERIC(12, 2) NOT NULL,
    total_equity_value NUMERIC(12, 2),
    total_options_value NUMERIC(12, 2),
    total_delta NUMERIC(10, 2),
    margin_used NUMERIC(12, 2),
    
    -- Position counts
    position_count INTEGER,
    short_puts_count INTEGER,
    covered_calls_count INTEGER,
    shares_positions_count INTEGER,
    hedge_positions_count INTEGER,
    
    -- Risk metrics
    current_drawdown NUMERIC(10, 4),
    peak_nav NUMERIC(12, 2),
    var_95 NUMERIC(12, 2),
    
    -- Full position details
    positions JSONB NOT NULL,
    
    PRIMARY KEY (snapshot_date)
);

CREATE INDEX idx_snapshots_timestamp ON portfolio_snapshots(timestamp);

-- ============================================================
-- ORDERS TABLE (Order Tracking)
-- ============================================================

CREATE TABLE IF NOT EXISTS orders (
    id BIGSERIAL PRIMARY KEY,
    order_id VARCHAR(50) UNIQUE NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    
    -- Order details
    asset VARCHAR(10) NOT NULL,
    action VARCHAR(50) NOT NULL,
    quantity INTEGER NOT NULL,
    
    -- Options-specific
    strike NUMERIC(10, 2),
    expiration DATE,
    option_type CHAR(1),
    option_symbol VARCHAR(50),
    
    -- Order parameters
    order_type VARCHAR(20) NOT NULL,  -- 'market', 'limit'
    limit_price NUMERIC(12, 4),
    time_in_force VARCHAR(20) DEFAULT 'day',
    
    -- Status tracking
    status VARCHAR(20) NOT NULL,  -- 'pending', 'submitted', 'filled', 'cancelled', 'rejected'
    broker VARCHAR(50),
    broker_order_id VARCHAR(100),
    
    -- Fill details (when filled)
    fill_price NUMERIC(12, 4),
    fill_quantity INTEGER,
    fill_timestamp TIMESTAMPTZ,
    commission NUMERIC(10, 2),
    
    -- Risk context
    risk_approved BOOLEAN DEFAULT false,
    risk_reason TEXT,
    
    -- Strategy context
    strategy_component VARCHAR(100),
    reasoning TEXT,
    signal_id VARCHAR(50),
    
    -- Metadata
    metadata JSONB,
    
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_orders_timestamp ON orders(timestamp);
CREATE INDEX idx_orders_status ON orders(status);
CREATE INDEX idx_orders_asset ON orders(asset);

-- ============================================================
-- RISK EVENTS (Risk Management Actions)
-- ============================================================

CREATE TABLE IF NOT EXISTS risk_events (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    
    -- Event details
    event_type VARCHAR(50) NOT NULL,  -- 'circuit_breaker', 'position_limit', 'margin_warning', etc.
    severity VARCHAR(20) NOT NULL,
    
    -- Context
    portfolio_nav NUMERIC(12, 2),
    current_drawdown NUMERIC(10, 4),
    total_delta NUMERIC(10, 2),
    
    -- Action taken
    action_taken VARCHAR(100),
    positions_affected INTEGER,
    
    -- Details
    details JSONB,
    
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_risk_events_timestamp ON risk_events(timestamp);
CREATE INDEX idx_risk_events_type ON risk_events(event_type);

-- ============================================================
-- PERFORMANCE METRICS (Time Series)
-- ============================================================

CREATE TABLE IF NOT EXISTS performance_metrics (
    timestamp TIMESTAMPTZ NOT NULL,
    
    -- Returns
    daily_return NUMERIC(10, 6),
    cumulative_return NUMERIC(10, 6),
    
    -- Risk metrics
    sharpe_ratio NUMERIC(10, 4),
    sortino_ratio NUMERIC(10, 4),
    max_drawdown NUMERIC(10, 4),
    current_drawdown NUMERIC(10, 4),
    volatility NUMERIC(10, 6),
    
    -- Portfolio metrics
    nav NUMERIC(12, 2),
    cash NUMERIC(12, 2),
    total_delta NUMERIC(10, 2),
    position_count INTEGER,
    
    -- Trading metrics
    trades_today INTEGER,
    premium_collected_today NUMERIC(12, 2),
    hedge_cost_today NUMERIC(12, 2),
    
    PRIMARY KEY (timestamp)
);

CREATE INDEX idx_perf_metrics_date ON performance_metrics(DATE(timestamp));

-- ============================================================
-- HELPER VIEWS
-- ============================================================

-- Current open positions view
CREATE OR REPLACE VIEW v_open_positions AS
SELECT 
    position_id,
    asset,
    position_type,
    quantity,
    strike,
    expiration,
    entry_price,
    current_price,
    market_value,
    unrealized_pnl,
    delta,
    (expiration - CURRENT_DATE) AS days_to_expiration
FROM positions
WHERE status = 'open'
ORDER BY asset, expiration;

-- Daily performance view
CREATE OR REPLACE VIEW v_daily_performance AS
SELECT 
    DATE(timestamp) as date,
    MAX(nav) as nav,
    SUM(CASE WHEN event = 'trade_executed' THEN 1 ELSE 0 END) as trades,
    SUM(CASE WHEN event_data->>'action' = 'SELL_PUT' 
        THEN (event_data->>'premium_collected')::numeric 
        ELSE 0 END) as put_premium,
    SUM(CASE WHEN event_data->>'action' = 'SELL_CALL' 
        THEN (event_data->>'premium_collected')::numeric 
        ELSE 0 END) as call_premium
FROM audit_log
WHERE log_type = 'TRADE'
GROUP BY DATE(timestamp)
ORDER BY date DESC;

-- Position summary by asset
CREATE OR REPLACE VIEW v_position_summary AS
SELECT 
    asset,
    COUNT(*) as position_count,
    SUM(quantity) as total_quantity,
    SUM(market_value) as total_value,
    SUM(unrealized_pnl) as total_unrealized_pnl,
    SUM(delta) as total_delta
FROM positions
WHERE status = 'open'
GROUP BY asset;

-- ============================================================
-- HELPER FUNCTIONS
-- ============================================================

-- Calculate current NAV from positions
CREATE OR REPLACE FUNCTION calculate_current_nav()
RETURNS NUMERIC AS $$
DECLARE
    total_nav NUMERIC;
BEGIN
    SELECT 
        COALESCE((SELECT nav FROM nav_history ORDER BY timestamp DESC LIMIT 1), 0) +
        COALESCE(SUM(market_value), 0)
    INTO total_nav
    FROM positions
    WHERE status = 'open';
    
    RETURN total_nav;
END;
$$ LANGUAGE plpgsql;

-- Get net delta for asset
CREATE OR REPLACE FUNCTION get_net_delta(p_asset VARCHAR DEFAULT NULL)
RETURNS NUMERIC AS $$
BEGIN
    IF p_asset IS NULL THEN
        -- Total delta across all assets
        RETURN COALESCE((SELECT SUM(delta) FROM positions WHERE status = 'open'), 0);
    ELSE
        -- Delta for specific asset
        RETURN COALESCE((SELECT SUM(delta) FROM positions WHERE asset = p_asset AND status = 'open'), 0);
    END IF;
END;
$$ LANGUAGE plpgsql;

-- ============================================================
-- DATA RETENTION POLICIES
-- ============================================================

-- Function to archive old logs to S3 (called by cron)
-- Keeps recent data in main table for fast access

COMMENT ON TABLE trades IS 'Immutable trade log - never delete, archive after 2 years to S3';
COMMENT ON TABLE audit_log IS 'Immutable audit trail - never delete, archive after 2 years';
COMMENT ON TABLE nav_history IS 'Daily NAV calculations - keep forever';
COMMENT ON TABLE positions IS 'Current positions - archive closed positions after 90 days';

-- ============================================================
-- GRANTS (Security)
-- ============================================================

-- Trading application user (limited permissions)
-- CREATE USER trading_app WITH PASSWORD 'secure_password_here';

-- Grant necessary permissions
GRANT SELECT, INSERT ON trades TO trading_app;
GRANT SELECT, INSERT ON audit_log TO trading_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON positions TO trading_app;
GRANT SELECT, INSERT ON nav_history TO trading_app;
GRANT SELECT, INSERT ON orders TO trading_app;
GRANT SELECT, INSERT ON risk_events TO trading_app;
GRANT SELECT, INSERT ON performance_metrics TO trading_app;

-- Grant sequence usage
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO trading_app;

-- Readonly user for reporting
-- CREATE USER trading_readonly WITH PASSWORD 'readonly_password';
GRANT SELECT ON ALL TABLES IN SCHEMA public TO trading_readonly;

-- ============================================================
-- INITIAL DATA / SETUP
-- ============================================================

-- Insert initial cash position (run once on new database)
-- INSERT INTO nav_history (timestamp, nav, cash, methodology)
-- VALUES (NOW(), 100000.00, 100000.00, 'initial_capital');

COMMENT ON DATABASE trading_dev IS 'Trading system database - development environment';



