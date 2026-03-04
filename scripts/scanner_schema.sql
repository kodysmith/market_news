-- Premarket gap scanner: scan runs, ranked ideas per run, symbol baselines for relvol.
-- Run in Supabase SQL editor. Optional: use for scanner API and run_premarket_scanner.py.

-- One row per scanner run (e.g. each 1-5 min during premarket).
CREATE TABLE IF NOT EXISTS scan_runs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    as_of_et TIMESTAMPTZ NOT NULL,
    config_hash VARCHAR(64),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_scan_runs_as_of_et ON scan_runs (as_of_et DESC);

-- Ranked ideas per run; payload = full PremarketTradeIdea JSON.
CREATE TABLE IF NOT EXISTS scanner_trade_ideas (
    run_id UUID NOT NULL REFERENCES scan_runs(id) ON DELETE CASCADE,
    symbol VARCHAR(10) NOT NULL,
    rank INT NOT NULL,
    total_score NUMERIC(10,4) NOT NULL,
    payload JSONB NOT NULL,
    PRIMARY KEY (run_id, symbol)
);

CREATE INDEX IF NOT EXISTS idx_scanner_trade_ideas_run_id ON scanner_trade_ideas (run_id);
CREATE INDEX IF NOT EXISTS idx_scanner_trade_ideas_symbol ON scanner_trade_ideas (symbol);

-- Baselines for relvol and liquidity (avg premarket vol 20d, avg dollar vol 20d).
CREATE TABLE IF NOT EXISTS symbol_baselines (
    symbol VARCHAR(10) NOT NULL PRIMARY KEY,
    avg_pm_vol_20d DOUBLE PRECISION,
    avg_dollar_vol_20d DOUBLE PRECISION,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- RLS: anon can read scanner data; service role can write (runner script).
ALTER TABLE scan_runs ENABLE ROW LEVEL SECURITY;
ALTER TABLE scanner_trade_ideas ENABLE ROW LEVEL SECURITY;
ALTER TABLE symbol_baselines ENABLE ROW LEVEL SECURITY;

CREATE POLICY "scan_runs_anon_select" ON scan_runs FOR SELECT USING (true);
CREATE POLICY "scan_runs_service_full" ON scan_runs FOR ALL USING (auth.role() = 'service_role');

CREATE POLICY "scanner_trade_ideas_anon_select" ON scanner_trade_ideas FOR SELECT USING (true);
CREATE POLICY "scanner_trade_ideas_service_full" ON scanner_trade_ideas FOR ALL USING (auth.role() = 'service_role');

CREATE POLICY "symbol_baselines_anon_select" ON symbol_baselines FOR SELECT USING (true);
CREATE POLICY "symbol_baselines_service_full" ON symbol_baselines FOR ALL USING (auth.role() = 'service_role');
