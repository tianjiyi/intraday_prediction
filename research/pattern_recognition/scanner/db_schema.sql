-- Scanner Module Database Schema
-- TimescaleDB tables for watchlist management and pattern findings
--
-- Run this script to create the tables:
--   psql -h localhost -U kronos -d market_data -f db_schema.sql

-- ============================================================================
-- DEFAULT WATCHLIST
-- Static list of symbols to always scan (QQQ, NVDA, TSLA, etc.)
-- ============================================================================
CREATE TABLE IF NOT EXISTS default_watchlist (
    id          SERIAL PRIMARY KEY,
    symbol      VARCHAR(10) NOT NULL UNIQUE,
    category    VARCHAR(50),          -- 'index_etf', 'tech_mega', 'momentum', etc.
    enabled     BOOLEAN DEFAULT TRUE,
    notes       TEXT,
    added_at    TIMESTAMPTZ DEFAULT NOW(),
    updated_at  TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_default_watchlist_enabled
    ON default_watchlist (enabled) WHERE enabled = TRUE;

-- Insert default symbols
INSERT INTO default_watchlist (symbol, category, enabled) VALUES
    ('QQQ',   'index_etf',  TRUE),
    ('SPY',   'index_etf',  TRUE),
    ('NVDA',  'tech_mega',  TRUE),
    ('TSLA',  'momentum',   TRUE),
    ('AMD',   'tech_mega',  TRUE),
    ('AAPL',  'tech_mega',  TRUE),
    ('MSFT',  'tech_mega',  TRUE),
    ('META',  'tech_mega',  TRUE),
    ('GOOGL', 'tech_mega',  TRUE),
    ('AMZN',  'tech_mega',  TRUE)
ON CONFLICT (symbol) DO NOTHING;


-- ============================================================================
-- FLOATING WATCHLIST
-- Dynamic list of unusual premarket movers (cleared daily)
-- ============================================================================
CREATE TABLE IF NOT EXISTS floating_watchlist (
    id              SERIAL PRIMARY KEY,
    date            DATE NOT NULL,
    symbol          VARCHAR(10) NOT NULL,
    gap_percent     DECIMAL(8,4),         -- Gap up/down % from prev close
    premarket_volume BIGINT,              -- Total premarket volume
    avg_volume      BIGINT,               -- 20-day average volume
    volume_ratio    DECIMAL(8,4),         -- premarket_vol / avg_vol
    market_cap      DECIMAL(16,2),        -- Market cap in dollars
    prev_close      DECIMAL(12,4),        -- Previous day close
    premarket_high  DECIMAL(12,4),        -- Premarket high
    premarket_low   DECIMAL(12,4),        -- Premarket low
    premarket_open  DECIMAL(12,4),        -- Premarket open (4am price)
    reason          TEXT,                 -- 'gap_up', 'volume_spike', 'both'
    scanned_at      TIMESTAMPTZ DEFAULT NOW(),

    UNIQUE(date, symbol)
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_floating_watchlist_date
    ON floating_watchlist (date DESC);
CREATE INDEX IF NOT EXISTS idx_floating_watchlist_symbol
    ON floating_watchlist (symbol, date DESC);


-- ============================================================================
-- PATTERN FINDINGS
-- Ascending triangle patterns detected during real-time scanning
-- Uses TimescaleDB hypertable for time-series optimization
-- ============================================================================
CREATE TABLE IF NOT EXISTS pattern_findings (
    time                TIMESTAMPTZ NOT NULL,
    symbol              VARCHAR(10) NOT NULL,
    timeframe           VARCHAR(10) NOT NULL,     -- '1min', '5min', '15min'
    pattern_type        VARCHAR(50) NOT NULL,     -- 'ascending_triangle', etc.
    resistance_level    DECIMAL(12,4),
    support_slope       DECIMAL(12,8),
    support_intercept   DECIMAL(12,4),
    confidence          DECIMAL(5,4),             -- 0.0 to 1.0
    pattern_start_idx   INTEGER,
    pattern_end_idx     INTEGER,
    pattern_start_time  TIMESTAMPTZ,
    pattern_end_time    TIMESTAMPTZ,
    breakout_status     VARCHAR(20) DEFAULT 'pending',  -- 'pending', 'success', 'failure', 'expired'
    breakout_price      DECIMAL(12,4),
    breakout_time       TIMESTAMPTZ,
    bars_to_breakout    INTEGER,
    compression_ratio   DECIMAL(5,4),
    pattern_height      DECIMAL(12,4),
    peaks_count         INTEGER,
    valleys_count       INTEGER,
    pattern_data        JSONB,                    -- Full pattern data (peaks, valleys, etc.)

    PRIMARY KEY (time, symbol, timeframe)
);

-- Convert to hypertable (TimescaleDB)
-- Only run if TimescaleDB extension is enabled
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'timescaledb') THEN
        PERFORM create_hypertable('pattern_findings', 'time',
            chunk_time_interval => INTERVAL '7 days',
            if_not_exists => TRUE
        );
    END IF;
END $$;

-- Indexes
CREATE INDEX IF NOT EXISTS idx_pattern_findings_symbol_time
    ON pattern_findings (symbol, time DESC);
CREATE INDEX IF NOT EXISTS idx_pattern_findings_type
    ON pattern_findings (pattern_type, time DESC);
CREATE INDEX IF NOT EXISTS idx_pattern_findings_status
    ON pattern_findings (breakout_status, time DESC);


-- ============================================================================
-- PREMARKET SCAN LOG
-- Tracks daily premarket scan execution
-- ============================================================================
CREATE TABLE IF NOT EXISTS premarket_scan_log (
    id              SERIAL PRIMARY KEY,
    date            DATE NOT NULL UNIQUE,
    scan_start      TIMESTAMPTZ,
    scan_end        TIMESTAMPTZ,
    symbols_scanned INTEGER DEFAULT 0,    -- Total symbols checked
    symbols_passed  INTEGER DEFAULT 0,    -- Symbols meeting criteria
    status          VARCHAR(20),          -- 'completed', 'failed', 'partial', 'running'
    error_msg       TEXT,
    config_snapshot JSONB                 -- Scanner config used
);

-- Index
CREATE INDEX IF NOT EXISTS idx_premarket_scan_log_date
    ON premarket_scan_log (date DESC);


-- ============================================================================
-- REALTIME SCAN LOG
-- Tracks each real-time scan cycle during RTH
-- ============================================================================
CREATE TABLE IF NOT EXISTS realtime_scan_log (
    id              SERIAL PRIMARY KEY,
    scan_time       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    symbols_scanned INTEGER DEFAULT 0,
    patterns_found  INTEGER DEFAULT 0,
    scan_duration_ms INTEGER,             -- Milliseconds to complete scan
    status          VARCHAR(20),          -- 'completed', 'failed', 'timeout'
    error_msg       TEXT
);

-- Convert to hypertable
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'timescaledb') THEN
        PERFORM create_hypertable('realtime_scan_log', 'scan_time',
            chunk_time_interval => INTERVAL '1 day',
            if_not_exists => TRUE
        );
    END IF;
END $$;


-- ============================================================================
-- STOCK FUNDAMENTALS
-- Cached fundamental data from Yahoo Finance (refreshed weekly/monthly)
-- ============================================================================
CREATE TABLE IF NOT EXISTS stock_fundamentals (
    symbol              VARCHAR(10) PRIMARY KEY,
    market_cap          DECIMAL(18,2),        -- Market capitalization in dollars
    shares_outstanding  BIGINT,               -- Total shares outstanding
    float_shares        BIGINT,               -- Tradable float
    sector              VARCHAR(100),         -- e.g., 'Technology'
    industry            VARCHAR(100),         -- e.g., 'Semiconductors'
    updated_at          TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_fundamentals_market_cap
    ON stock_fundamentals (market_cap);
CREATE INDEX IF NOT EXISTS idx_fundamentals_sector
    ON stock_fundamentals (sector);


-- ============================================================================
-- VIEWS
-- ============================================================================

-- Combined watchlist view (default + floating for today)
CREATE OR REPLACE VIEW combined_watchlist AS
SELECT symbol, 'default' as source, NULL::DECIMAL as gap_percent
FROM default_watchlist
WHERE enabled = TRUE
UNION ALL
SELECT symbol, 'floating' as source, gap_percent
FROM floating_watchlist
WHERE date = CURRENT_DATE;


-- Today's pattern summary
CREATE OR REPLACE VIEW today_patterns AS
SELECT
    symbol,
    timeframe,
    COUNT(*) as pattern_count,
    MAX(confidence) as max_confidence,
    AVG(confidence) as avg_confidence,
    SUM(CASE WHEN breakout_status = 'success' THEN 1 ELSE 0 END) as successes,
    SUM(CASE WHEN breakout_status = 'failure' THEN 1 ELSE 0 END) as failures,
    MAX(time) as latest_pattern
FROM pattern_findings
WHERE time::date = CURRENT_DATE
GROUP BY symbol, timeframe
ORDER BY pattern_count DESC;


-- Pattern success rate by symbol (last 30 days)
CREATE OR REPLACE VIEW pattern_success_rates AS
SELECT
    symbol,
    COUNT(*) as total_patterns,
    SUM(CASE WHEN breakout_status = 'success' THEN 1 ELSE 0 END) as successes,
    SUM(CASE WHEN breakout_status = 'failure' THEN 1 ELSE 0 END) as failures,
    ROUND(
        SUM(CASE WHEN breakout_status = 'success' THEN 1 ELSE 0 END)::DECIMAL /
        NULLIF(SUM(CASE WHEN breakout_status IN ('success', 'failure') THEN 1 ELSE 0 END), 0),
        3
    ) as success_rate,
    ROUND(AVG(confidence), 3) as avg_confidence,
    ROUND(AVG(bars_to_breakout), 1) as avg_bars_to_breakout
FROM pattern_findings
WHERE time > NOW() - INTERVAL '30 days'
GROUP BY symbol
HAVING COUNT(*) >= 3
ORDER BY success_rate DESC NULLS LAST;


-- ============================================================================
-- FUNCTIONS
-- ============================================================================

-- Function to clean old floating watchlist entries (keep 30 days)
CREATE OR REPLACE FUNCTION cleanup_floating_watchlist()
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM floating_watchlist
    WHERE date < CURRENT_DATE - INTERVAL '30 days';
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;


-- Function to check if pattern already logged (deduplication)
CREATE OR REPLACE FUNCTION is_pattern_duplicate(
    p_symbol VARCHAR(10),
    p_timeframe VARCHAR(10),
    p_resistance DECIMAL(12,4),
    p_time_tolerance INTERVAL DEFAULT '30 minutes',
    p_price_tolerance DECIMAL DEFAULT 0.01
)
RETURNS BOOLEAN AS $$
BEGIN
    RETURN EXISTS (
        SELECT 1 FROM pattern_findings
        WHERE symbol = p_symbol
          AND timeframe = p_timeframe
          AND time > NOW() - p_time_tolerance
          AND ABS(resistance_level - p_resistance) / resistance_level < p_price_tolerance
    );
END;
$$ LANGUAGE plpgsql;
