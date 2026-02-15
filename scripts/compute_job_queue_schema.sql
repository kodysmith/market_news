-- On-demand compute job queue (Option 1: table as queue).
-- App inserts a row (symbol, task_type, status=pending). Server claims, runs task, sets result and status=done.
-- Run in Supabase SQL editor after supabase_schema.sql (or in same project).

CREATE TABLE IF NOT EXISTS compute_job_queue (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    symbol VARCHAR(10) NOT NULL,
    task_type VARCHAR(30) NOT NULL
        CHECK (task_type IN ('gex', 'valuation', 'cockpit', 'probability', 'trade_ideas', 'congressional_trades')),
    status VARCHAR(20) NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending', 'processing', 'done', 'failed')),
    requested_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    claimed_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    result JSONB,
    error_text TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_compute_job_queue_status_requested
    ON compute_job_queue (status, requested_at)
    WHERE status = 'pending';

-- RLS (Supabase)
ALTER TABLE compute_job_queue ENABLE ROW LEVEL SECURITY;
CREATE POLICY "compute_job_service_full" ON compute_job_queue FOR ALL USING (auth.role() = 'service_role');
CREATE POLICY "compute_job_anon_insert_select" ON compute_job_queue FOR INSERT WITH CHECK (true);
CREATE POLICY "compute_job_anon_select" ON compute_job_queue FOR SELECT USING (true);

-- Optional: enable Supabase Realtime so the compute worker can wake on INSERT (see docs/COMPUTE_QUEUE.md).
-- Run once in Supabase SQL editor. Skip if the table is already in the publication.
-- ALTER PUBLICATION supabase_realtime ADD TABLE compute_job_queue;
