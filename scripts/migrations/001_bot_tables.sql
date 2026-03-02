-- Migration: 001_bot_tables.sql
-- Creates all tables needed for the SPX Iron Condor Bot.
-- Run once against your Supabase project via the SQL editor or psql.
-- Safe to re-run: uses IF NOT EXISTS / ON CONFLICT.

-- -------------------------------------------------------------------------
-- bot_positions: one row per open/closed iron condor position
-- -------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS bot_positions (
  id                uuid         PRIMARY KEY DEFAULT gen_random_uuid(),
  config_id         text         NOT NULL,          -- '15_VIX_SWEET_SPOT' | 'VD2_ENTRY_105'
  entry_date        date         NOT NULL,
  expiration        date         NOT NULL,
  k_put_short       numeric      NOT NULL,
  k_put_long        numeric      NOT NULL,
  k_call_short      numeric      NOT NULL,
  k_call_long       numeric      NOT NULL,
  credit_collected  numeric      NOT NULL,           -- per-contract credit ($/unit, NOT × 100)
  n_contracts       int          NOT NULL,
  spx_entry         numeric,                         -- SPX price at entry (for % drop check)
  vix_entry         numeric,                         -- VIX at entry (for spike exit check)
  status            text         NOT NULL DEFAULT 'open', -- open | closed | expired
  exit_date         date,
  exit_debit        numeric,                         -- per-contract closing debit
  pnl               numeric,                         -- total P&L in dollars (all contracts)
  exit_reason       text,                            -- profit_target | stop_loss | time_stop | ...
  created_at        timestamptz  NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS bot_positions_status_idx ON bot_positions(status);
CREATE INDEX IF NOT EXISTS bot_positions_config_id_idx ON bot_positions(config_id);
CREATE INDEX IF NOT EXISTS bot_positions_entry_date_idx ON bot_positions(entry_date);

-- -------------------------------------------------------------------------
-- bot_state: key/value store for scalar bot state
-- -------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS bot_state (
  key        text        PRIMARY KEY,
  value      text        NOT NULL,
  updated_at timestamptz NOT NULL DEFAULT now()
);

-- Seed initial values (safe: ON CONFLICT DO NOTHING)
INSERT INTO bot_state (key, value) VALUES
  ('vd2_contracts',  '6'),
  ('cumulative_pnl', '0'),
  ('ath_equity',     '700000'),
  ('entries_paused', '0')
ON CONFLICT (key) DO NOTHING;

-- -------------------------------------------------------------------------
-- bot_trades: append-only log of completed trades
-- -------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS bot_trades (
  id          uuid         PRIMARY KEY DEFAULT gen_random_uuid(),
  config_id   text         NOT NULL,
  entry_date  date,
  exit_date   date,
  n_contracts int          NOT NULL,
  credit      numeric,                    -- per-contract entry credit ($/unit)
  debit       numeric,                    -- per-contract exit debit ($/unit)
  pnl         numeric,                    -- total P&L in dollars (all contracts)
  exit_reason text,
  created_at  timestamptz  NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS bot_trades_exit_date_idx ON bot_trades(exit_date);
CREATE INDEX IF NOT EXISTS bot_trades_config_id_idx ON bot_trades(config_id);

-- -------------------------------------------------------------------------
-- Row-level security (optional but recommended for Supabase)
-- Enable RLS and restrict to service_role only for bot tables.
-- -------------------------------------------------------------------------
-- ALTER TABLE bot_positions ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE bot_state     ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE bot_trades    ENABLE ROW LEVEL SECURITY;
--
-- CREATE POLICY bot_service_only ON bot_positions USING (auth.role() = 'service_role');
-- CREATE POLICY bot_service_only ON bot_state     USING (auth.role() = 'service_role');
-- CREATE POLICY bot_service_only ON bot_trades    USING (auth.role() = 'service_role');
