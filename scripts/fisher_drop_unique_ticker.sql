-- Drop UNIQUE on ticker if it exists (e.g. from older schema).
-- Run in Supabase SQL editor to allow multiple rows per ticker over time.

DO $$
BEGIN
  IF EXISTS (
    SELECT 1 FROM pg_constraint
    WHERE conrelid = 'fisher_company'::regclass AND conname = 'fisher_company_ticker_key'
  ) THEN
    ALTER TABLE fisher_company DROP CONSTRAINT fisher_company_ticker_key;
  END IF;

  IF EXISTS (
    SELECT 1 FROM pg_constraint
    WHERE conrelid = 'fisher_scan_queue'::regclass AND conname = 'fisher_scan_queue_ticker_key'
  ) THEN
    ALTER TABLE fisher_scan_queue DROP CONSTRAINT fisher_scan_queue_ticker_key;
  END IF;
END $$;
