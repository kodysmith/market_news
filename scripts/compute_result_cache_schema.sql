-- Precomputed result cache: one row per (symbol, task_type). Filled by run_gex_cockpit_precompute.py every 5 min.
-- Run in Supabase SQL editor after compute_job_queue_schema.sql (or in same project).

CREATE TABLE IF NOT EXISTS compute_result_cache (
    symbol VARCHAR(10) NOT NULL,
    task_type VARCHAR(30) NOT NULL
        CHECK (task_type IN ('gex', 'valuation', 'cockpit', 'probability', 'trade_ideas', 'congressional_trades')),
    result JSONB NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (symbol, task_type)
);

CREATE INDEX IF NOT EXISTS idx_compute_result_cache_task_type
    ON compute_result_cache (task_type);

-- RLS: anon can read. Service role (precompute script) can write.
ALTER TABLE compute_result_cache ENABLE ROW LEVEL SECURITY;
CREATE POLICY "compute_result_cache_service_full" ON compute_result_cache FOR ALL USING (auth.role() = 'service_role');
CREATE POLICY "compute_result_cache_anon_select" ON compute_result_cache FOR SELECT USING (true);
