-- News feed for app + backtesting. Filled by publish_news_to_supabase.py (from data/news.json or API).
-- Run in Supabase SQL editor.

CREATE TABLE IF NOT EXISTS news_feed (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    headline TEXT NOT NULL,
    source TEXT,
    url TEXT,
    summary TEXT,
    published_date DATE,
    category TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_news_feed_published_date ON news_feed (published_date DESC);
CREATE INDEX IF NOT EXISTS idx_news_feed_created_at ON news_feed (created_at DESC);

-- RLS: anon can read; service role (publish script) can insert/update.
ALTER TABLE news_feed ENABLE ROW LEVEL SECURITY;
CREATE POLICY "news_feed_service_full" ON news_feed FOR ALL USING (auth.role() = 'service_role');
CREATE POLICY "news_feed_anon_select" ON news_feed FOR SELECT USING (true);
