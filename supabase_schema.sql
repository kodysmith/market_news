-- Supabase Database Schema for Live Trading System
-- Run this SQL in your Supabase SQL editor to create the tables

-- Runs table - tracks each trading run
CREATE TABLE IF NOT EXISTS runs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    mode VARCHAR(20) NOT NULL CHECK (mode IN ('backtest', 'paper', 'live')),
    status VARCHAR(20) NOT NULL DEFAULT 'running' CHECK (status IN ('running', 'completed', 'failed')),
    portfolio_value NUMERIC(15, 2) DEFAULT 0,
    cash NUMERIC(15, 2) DEFAULT 0,
    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMPTZ,
    regime JSONB,
    config JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Positions table - tracks current positions for each run
CREATE TABLE IF NOT EXISTS positions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id UUID NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    ticker VARCHAR(10) NOT NULL,
    quantity NUMERIC(15, 4) NOT NULL,
    value NUMERIC(15, 2) NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(run_id, ticker)
);

-- Orders table - tracks all executed orders
CREATE TABLE IF NOT EXISTS orders (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id UUID NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    ticker VARCHAR(10) NOT NULL,
    quantity NUMERIC(15, 4) NOT NULL,
    order_type VARCHAR(20) NOT NULL DEFAULT 'market',
    fill_price NUMERIC(10, 2),
    commission NUMERIC(10, 2) DEFAULT 0,
    status VARCHAR(20) NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'filled', 'cancelled', 'rejected')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    filled_at TIMESTAMPTZ
);

-- Regime history table - tracks regime detection results
CREATE TABLE IF NOT EXISTS regime_history (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id UUID NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    regime VARCHAR(50) NOT NULL,
    confidence NUMERIC(5, 4) NOT NULL,
    indicators JSONB,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Run logs table - tracks execution logs and errors
CREATE TABLE IF NOT EXISTS run_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id UUID NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    level VARCHAR(20) NOT NULL,
    message TEXT NOT NULL,
    metadata JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_runs_mode ON runs(mode);
CREATE INDEX IF NOT EXISTS idx_runs_status ON runs(status);
CREATE INDEX IF NOT EXISTS idx_runs_started_at ON runs(started_at);

CREATE INDEX IF NOT EXISTS idx_positions_run_id ON positions(run_id);
CREATE INDEX IF NOT EXISTS idx_positions_ticker ON positions(ticker);

CREATE INDEX IF NOT EXISTS idx_orders_run_id ON orders(run_id);
CREATE INDEX IF NOT EXISTS idx_orders_ticker ON orders(ticker);
CREATE INDEX IF NOT EXISTS idx_orders_created_at ON orders(created_at);

CREATE INDEX IF NOT EXISTS idx_regime_history_run_id ON regime_history(run_id);
CREATE INDEX IF NOT EXISTS idx_regime_history_timestamp ON regime_history(timestamp);

CREATE INDEX IF NOT EXISTS idx_run_logs_run_id ON run_logs(run_id);
CREATE INDEX IF NOT EXISTS idx_run_logs_created_at ON run_logs(created_at);

-- Row Level Security (RLS) Policies
-- Enable RLS on all tables
ALTER TABLE runs ENABLE ROW LEVEL SECURITY;
ALTER TABLE positions ENABLE ROW LEVEL SECURITY;
ALTER TABLE orders ENABLE ROW LEVEL SECURITY;
ALTER TABLE regime_history ENABLE ROW LEVEL SECURITY;
ALTER TABLE run_logs ENABLE ROW LEVEL SECURITY;

-- Policy: Service role can do everything
CREATE POLICY "Service role full access" ON runs
    FOR ALL USING (auth.role() = 'service_role');

CREATE POLICY "Service role full access" ON positions
    FOR ALL USING (auth.role() = 'service_role');

CREATE POLICY "Service role full access" ON orders
    FOR ALL USING (auth.role() = 'service_role');

CREATE POLICY "Service role full access" ON regime_history
    FOR ALL USING (auth.role() = 'service_role');

CREATE POLICY "Service role full access" ON run_logs
    FOR ALL USING (auth.role() = 'service_role');

-- Policy: Anon users (dashboard) can read
CREATE POLICY "Anon read access" ON runs
    FOR SELECT USING (true);

CREATE POLICY "Anon read access" ON positions
    FOR SELECT USING (true);

CREATE POLICY "Anon read access" ON orders
    FOR SELECT USING (true);

CREATE POLICY "Anon read access" ON regime_history
    FOR SELECT USING (true);

CREATE POLICY "Anon read access" ON run_logs
    FOR SELECT USING (true);

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Trigger to auto-update updated_at
CREATE TRIGGER update_runs_updated_at BEFORE UPDATE ON runs
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_positions_updated_at BEFORE UPDATE ON positions
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
