-- Migration: 002_trade_ideas_published.sql
-- Table for trade ideas published when SPX (or other symbols) meet entry criteria.
-- Filled by scripts/publish_trade_ideas.py (cron). App reads by valid_date (today/yesterday).
-- Run once in Supabase SQL editor. Safe to re-run: uses IF NOT EXISTS.

CREATE TABLE IF NOT EXISTS trade_ideas_published (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    valid_date DATE NOT NULL,
    as_of_et TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(10) NOT NULL,
    idea_id VARCHAR(128) NOT NULL,
    payload JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(valid_date, symbol, idea_id)
);

CREATE INDEX IF NOT EXISTS idx_trade_ideas_published_valid_date
    ON trade_ideas_published (valid_date DESC);
CREATE INDEX IF NOT EXISTS idx_trade_ideas_published_symbol
    ON trade_ideas_published (symbol);
CREATE INDEX IF NOT EXISTS idx_trade_ideas_published_as_of
    ON trade_ideas_published (as_of_et DESC);

COMMENT ON TABLE trade_ideas_published IS 'Trade ideas published when scanner/criteria detect SPX (or symbol) meets entry; app reads by valid_date (e.g. today, yesterday).';

ALTER TABLE trade_ideas_published ENABLE ROW LEVEL SECURITY;
CREATE POLICY "trade_ideas_published_anon_select" ON trade_ideas_published FOR SELECT USING (true);
CREATE POLICY "trade_ideas_published_service_full" ON trade_ideas_published FOR ALL USING (auth.role() = 'service_role');
