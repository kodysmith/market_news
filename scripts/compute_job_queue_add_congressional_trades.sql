-- Migration: add 'congressional_trades' to task_type CHECK on compute_job_queue (and compute_result_cache).
-- Run this in Supabase SQL editor if you get: "new row for relation compute_job_queue violates check constraint".
-- Your table was likely created before congressional_trades was added to the schema.

-- 1) compute_job_queue: drop existing task_type check and add new one
DO $$
DECLARE
  c RECORD;
BEGIN
  FOR c IN
    SELECT conname FROM pg_constraint
    WHERE conrelid = 'public.compute_job_queue'::regclass
      AND contype = 'c'
      AND pg_get_constraintdef(oid) LIKE '%task_type%'
  LOOP
    EXECUTE format('ALTER TABLE public.compute_job_queue DROP CONSTRAINT %I', c.conname);
  END LOOP;
END $$;

ALTER TABLE public.compute_job_queue
  ADD CONSTRAINT compute_job_queue_task_type_check
  CHECK (task_type IN ('gex', 'valuation', 'cockpit', 'probability', 'trade_ideas', 'congressional_trades'));

-- 2) compute_result_cache: same if table exists and has a task_type check
DO $$
DECLARE
  c RECORD;
  tbl regclass;
BEGIN
  tbl := to_regclass('public.compute_result_cache');
  IF tbl IS NULL THEN
    RETURN;
  END IF;
  FOR c IN
    SELECT conname FROM pg_constraint
    WHERE conrelid = tbl
      AND contype = 'c'
      AND pg_get_constraintdef(oid) LIKE '%task_type%'
  LOOP
    EXECUTE format('ALTER TABLE public.compute_result_cache DROP CONSTRAINT %I', c.conname);
  END LOOP;
  EXECUTE 'ALTER TABLE public.compute_result_cache ADD CONSTRAINT compute_result_cache_task_type_check CHECK (task_type IN (''gex'', ''valuation'', ''cockpit'', ''probability'', ''trade_ideas'', ''congressional_trades''))';
END $$;
