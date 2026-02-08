-- Stub for Supabase auth.role() so RLS policies work on local Postgres.
-- Run this once before supabase_schema.sql when using local Docker Postgres.

CREATE SCHEMA IF NOT EXISTS auth;
CREATE OR REPLACE FUNCTION auth.role() RETURNS text AS $$
  SELECT 'service_role';
$$ LANGUAGE sql;
