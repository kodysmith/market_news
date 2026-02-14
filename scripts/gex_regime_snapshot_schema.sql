-- Optional: store last GEX regime per symbol for flip detection (alternative to data/gex_regime_last.json).
-- The precompute script currently uses data/gex_regime_last.json; run this if you want DB-backed storage instead.
-- Run in Supabase SQL editor.

CREATE TABLE IF NOT EXISTS gex_regime_snapshot (
    symbol VARCHAR(10) PRIMARY KEY,
    regime VARCHAR(20) NOT NULL CHECK (regime IN ('POSITIVE', 'NEGATIVE')),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

COMMENT ON TABLE gex_regime_snapshot IS 'Last GEX regime per symbol for intraday flip alerts (SPX, SPY, XSP). Precompute script can use this instead of data/gex_regime_last.json when Supabase is used.';
