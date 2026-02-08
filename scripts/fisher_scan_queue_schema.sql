-- Fisher full-market scan queue (run after supabase_schema_fisher.sql)
-- For monthly or manual scan of all SEC tickers with Fisher pipeline.

CREATE TABLE IF NOT EXISTS fisher_scan_queue (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    ticker VARCHAR(10) NOT NULL UNIQUE,
    status VARCHAR(20) NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending', 'processing', 'done', 'failed')),
    source VARCHAR(20) NOT NULL DEFAULT 'sec',
    queued_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    error_text TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_fisher_scan_queue_status_queued
    ON fisher_scan_queue (status, queued_at)
    WHERE status = 'pending';

-- RLS (Supabase); for plain Postgres ensure auth.role() is stubbed if needed
ALTER TABLE fisher_scan_queue ENABLE ROW LEVEL SECURITY;
CREATE POLICY "fisher_service_full" ON fisher_scan_queue FOR ALL USING (auth.role() = 'service_role');
CREATE POLICY "fisher_anon_read" ON fisher_scan_queue FOR SELECT USING (true);
