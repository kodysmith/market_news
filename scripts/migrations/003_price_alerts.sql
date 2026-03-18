-- Price alert configuration and state
-- Used by the cloud-based price monitor (no IBKR dependency)

CREATE TABLE IF NOT EXISTS price_alerts (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    name TEXT NOT NULL,                    -- "SPX_IBF_6655_lower", "SPX_IBF_6655_upper"
    ticker TEXT NOT NULL,                  -- "^GSPC", "^VIX", "SCHD", etc.
    condition TEXT NOT NULL,               -- "below", "above", "between"
    threshold NUMERIC NOT NULL,            -- trigger price
    threshold_upper NUMERIC,               -- for "between" condition
    severity TEXT DEFAULT 'warning',       -- "info", "warning", "critical"
    message TEXT,                          -- custom alert message
    enabled BOOLEAN DEFAULT true,
    -- Cooldown: don't re-alert within N minutes
    cooldown_minutes INTEGER DEFAULT 15,
    last_triggered_at TIMESTAMPTZ,
    trigger_count INTEGER DEFAULT 0,
    -- Link to position/strategy
    strategy TEXT,                         -- "0DTE_IBF", "SCHD_COLLAR", etc.
    expiry DATE,                           -- auto-disable after expiry
    -- Metadata
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);

-- Track price snapshots for debugging/analysis
CREATE TABLE IF NOT EXISTS price_snapshots (
    id BIGSERIAL PRIMARY KEY,
    ticker TEXT NOT NULL,
    price NUMERIC NOT NULL,
    source TEXT DEFAULT 'yahoo',           -- "yahoo", "massive", "alphavantage"
    captured_at TIMESTAMPTZ DEFAULT now()
);

-- Index for fast lookups
CREATE INDEX idx_price_alerts_enabled ON price_alerts (enabled) WHERE enabled = true;
CREATE INDEX idx_price_alerts_ticker ON price_alerts (ticker);
CREATE INDEX idx_price_snapshots_ticker_time ON price_snapshots (ticker, captured_at DESC);

-- Auto-update updated_at
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER price_alerts_updated
    BEFORE UPDATE ON price_alerts
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();
