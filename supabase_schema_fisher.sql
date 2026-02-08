-- Fisher Valuation Automation - Supabase/Postgres Schema
-- Run this SQL in your Supabase SQL editor after supabase_schema.sql
-- Creates: company, filing, financial_fact, segment_fact, transcript, transcript_chunk,
--          market_bar_daily, peer_set, fisher_score_snapshot

-- company: S&P 500 (or universe) companies with CIK for EDGAR
CREATE TABLE IF NOT EXISTS fisher_company (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    ticker VARCHAR(10) NOT NULL UNIQUE,
    name TEXT,
    cik VARCHAR(20),
    sector TEXT,
    industry TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- filing: SEC filings (10-K, 10-Q, 8-K, DEF14A)
CREATE TABLE IF NOT EXISTS fisher_filing (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    company_id UUID NOT NULL REFERENCES fisher_company(id) ON DELETE CASCADE,
    form_type VARCHAR(10) NOT NULL,
    filing_date DATE NOT NULL,
    period_end DATE,
    url TEXT,
    content_hash VARCHAR(64),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(company_id, form_type, period_end)
);

-- financial_fact: XBRL-derived facts per filing
CREATE TABLE IF NOT EXISTS fisher_financial_fact (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    filing_id UUID NOT NULL REFERENCES fisher_filing(id) ON DELETE CASCADE,
    fact_name VARCHAR(100) NOT NULL,
    value NUMERIC(20, 4) NOT NULL,
    unit VARCHAR(20),
    period DATE,
    dimensions JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- segment_fact: segment revenue/margin from 10-K/10-Q
CREATE TABLE IF NOT EXISTS fisher_segment_fact (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    filing_id UUID NOT NULL REFERENCES fisher_filing(id) ON DELETE CASCADE,
    segment_name TEXT NOT NULL,
    revenue NUMERIC(20, 4),
    margin_pct NUMERIC(10, 4),
    period DATE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- transcript: earnings call transcripts
CREATE TABLE IF NOT EXISTS fisher_transcript (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    company_id UUID NOT NULL REFERENCES fisher_company(id) ON DELETE CASCADE,
    event_date DATE NOT NULL,
    source TEXT,
    raw_text TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(company_id, event_date)
);

-- transcript_chunk: speaker-tagged chunks
CREATE TABLE IF NOT EXISTS fisher_transcript_chunk (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    transcript_id UUID NOT NULL REFERENCES fisher_transcript(id) ON DELETE CASCADE,
    speaker VARCHAR(100),
    section VARCHAR(50),
    content TEXT NOT NULL,
    sequence INT NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- market_bar_daily: daily OHLCV by company
CREATE TABLE IF NOT EXISTS fisher_market_bar_daily (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    company_id UUID NOT NULL REFERENCES fisher_company(id) ON DELETE CASCADE,
    date DATE NOT NULL,
    open NUMERIC(15, 4),
    high NUMERIC(15, 4),
    low NUMERIC(15, 4),
    close NUMERIC(15, 4) NOT NULL,
    volume BIGINT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(company_id, date)
);

-- peer_set: peer tickers for comparison (e.g. same sector)
CREATE TABLE IF NOT EXISTS fisher_peer_set (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    company_id UUID NOT NULL REFERENCES fisher_company(id) ON DELETE CASCADE UNIQUE,
    peer_tickers JSONB NOT NULL DEFAULT '[]',
    source TEXT,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- fisher_score_snapshot: per-company snapshot of Fisher point scores
CREATE TABLE IF NOT EXISTS fisher_score_snapshot (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    company_id UUID NOT NULL REFERENCES fisher_company(id) ON DELETE CASCADE,
    snapshot_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    points JSONB NOT NULL DEFAULT '{}',
    category_scores JSONB DEFAULT '{}',
    total_score NUMERIC(5, 2),
    version INT NOT NULL DEFAULT 1,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_fisher_company_ticker ON fisher_company(ticker);
CREATE INDEX IF NOT EXISTS idx_fisher_company_cik ON fisher_company(cik);

CREATE INDEX IF NOT EXISTS idx_fisher_filing_company_id ON fisher_filing(company_id);
CREATE INDEX IF NOT EXISTS idx_fisher_filing_company_date ON fisher_filing(company_id, filing_date DESC);
CREATE INDEX IF NOT EXISTS idx_fisher_filing_form_type ON fisher_filing(form_type);

CREATE INDEX IF NOT EXISTS idx_fisher_financial_fact_filing_id ON fisher_financial_fact(filing_id);
CREATE INDEX IF NOT EXISTS idx_fisher_financial_fact_name ON fisher_financial_fact(fact_name);

CREATE INDEX IF NOT EXISTS idx_fisher_segment_fact_filing_id ON fisher_segment_fact(filing_id);

CREATE INDEX IF NOT EXISTS idx_fisher_transcript_company_id ON fisher_transcript(company_id);
CREATE INDEX IF NOT EXISTS idx_fisher_transcript_event_date ON fisher_transcript(event_date);

CREATE INDEX IF NOT EXISTS idx_fisher_transcript_chunk_transcript_id ON fisher_transcript_chunk(transcript_id);

CREATE INDEX IF NOT EXISTS idx_fisher_market_bar_company_date ON fisher_market_bar_daily(company_id, date DESC);

CREATE INDEX IF NOT EXISTS idx_fisher_score_snapshot_company_at ON fisher_score_snapshot(company_id, snapshot_at DESC);

-- RLS
ALTER TABLE fisher_company ENABLE ROW LEVEL SECURITY;
ALTER TABLE fisher_filing ENABLE ROW LEVEL SECURITY;
ALTER TABLE fisher_financial_fact ENABLE ROW LEVEL SECURITY;
ALTER TABLE fisher_segment_fact ENABLE ROW LEVEL SECURITY;
ALTER TABLE fisher_transcript ENABLE ROW LEVEL SECURITY;
ALTER TABLE fisher_transcript_chunk ENABLE ROW LEVEL SECURITY;
ALTER TABLE fisher_market_bar_daily ENABLE ROW LEVEL SECURITY;
ALTER TABLE fisher_peer_set ENABLE ROW LEVEL SECURITY;
ALTER TABLE fisher_score_snapshot ENABLE ROW LEVEL SECURITY;

-- Service role full access
CREATE POLICY "fisher_service_full" ON fisher_company FOR ALL USING (auth.role() = 'service_role');
CREATE POLICY "fisher_service_full" ON fisher_filing FOR ALL USING (auth.role() = 'service_role');
CREATE POLICY "fisher_service_full" ON fisher_financial_fact FOR ALL USING (auth.role() = 'service_role');
CREATE POLICY "fisher_service_full" ON fisher_segment_fact FOR ALL USING (auth.role() = 'service_role');
CREATE POLICY "fisher_service_full" ON fisher_transcript FOR ALL USING (auth.role() = 'service_role');
CREATE POLICY "fisher_service_full" ON fisher_transcript_chunk FOR ALL USING (auth.role() = 'service_role');
CREATE POLICY "fisher_service_full" ON fisher_market_bar_daily FOR ALL USING (auth.role() = 'service_role');
CREATE POLICY "fisher_service_full" ON fisher_peer_set FOR ALL USING (auth.role() = 'service_role');
CREATE POLICY "fisher_service_full" ON fisher_score_snapshot FOR ALL USING (auth.role() = 'service_role');

-- Anon read (dashboard)
CREATE POLICY "fisher_anon_read" ON fisher_company FOR SELECT USING (true);
CREATE POLICY "fisher_anon_read" ON fisher_filing FOR SELECT USING (true);
CREATE POLICY "fisher_anon_read" ON fisher_financial_fact FOR SELECT USING (true);
CREATE POLICY "fisher_anon_read" ON fisher_segment_fact FOR SELECT USING (true);
CREATE POLICY "fisher_anon_read" ON fisher_transcript FOR SELECT USING (true);
CREATE POLICY "fisher_anon_read" ON fisher_transcript_chunk FOR SELECT USING (true);
CREATE POLICY "fisher_anon_read" ON fisher_market_bar_daily FOR SELECT USING (true);
CREATE POLICY "fisher_anon_read" ON fisher_peer_set FOR SELECT USING (true);
CREATE POLICY "fisher_anon_read" ON fisher_score_snapshot FOR SELECT USING (true);

-- Trigger for fisher_company updated_at
CREATE TRIGGER update_fisher_company_updated_at BEFORE UPDATE ON fisher_company
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_fisher_peer_set_updated_at BEFORE UPDATE ON fisher_peer_set
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
