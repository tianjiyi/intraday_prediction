-- TimescaleDB Schema for Market Data Storage
-- This schema is auto-loaded when the Docker container starts

-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- ============================================
-- TRADES TABLE
-- Stores tick-level trade data from Polygon
-- ============================================
CREATE TABLE IF NOT EXISTS trades (
    id          BIGSERIAL,
    time        TIMESTAMPTZ NOT NULL,
    symbol      VARCHAR(10) NOT NULL,
    price       DECIMAL(12,4) NOT NULL,
    size        INTEGER NOT NULL,
    exchange    VARCHAR(10),
    conditions  TEXT[]
);

-- Convert to hypertable (partitioned by time)
SELECT create_hypertable('trades', 'time',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

-- Index for efficient symbol + time range queries
CREATE INDEX IF NOT EXISTS idx_trades_symbol_time
    ON trades (symbol, time DESC);

-- ============================================
-- QUOTES TABLE
-- Stores NBBO quote data from Polygon
-- ============================================
CREATE TABLE IF NOT EXISTS quotes (
    id          BIGSERIAL,
    time        TIMESTAMPTZ NOT NULL,
    symbol      VARCHAR(10) NOT NULL,
    bid_price   DECIMAL(12,4),
    bid_size    INTEGER,
    ask_price   DECIMAL(12,4),
    ask_size    INTEGER
);

-- Convert to hypertable
SELECT create_hypertable('quotes', 'time',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

-- Index for efficient symbol + time range queries
CREATE INDEX IF NOT EXISTS idx_quotes_symbol_time
    ON quotes (symbol, time DESC);

-- ============================================
-- FEATURES TABLE
-- Stores computed features (OFI, VPIN, HMM state)
-- for different timeframes
-- ============================================
CREATE TABLE IF NOT EXISTS features (
    time        TIMESTAMPTZ NOT NULL,
    symbol      VARCHAR(10) NOT NULL,
    timeframe   INTEGER NOT NULL,  -- minutes: 1, 5, 15, 30
    ofi         DECIMAL(16,4),
    vpin        DECIMAL(8,4),
    hmm_state   INTEGER,
    close       DECIMAL(12,4),
    volume      BIGINT,

    PRIMARY KEY (time, symbol, timeframe)
);

-- Convert to hypertable
SELECT create_hypertable('features', 'time',
    chunk_time_interval => INTERVAL '7 days',
    if_not_exists => TRUE
);

-- Index for efficient queries by symbol and timeframe
CREATE INDEX IF NOT EXISTS idx_features_symbol_tf_time
    ON features (symbol, timeframe, time DESC);

-- ============================================
-- DATA INGESTION LOG
-- Tracks which dates have been ingested
-- ============================================
CREATE TABLE IF NOT EXISTS ingestion_log (
    id          SERIAL PRIMARY KEY,
    symbol      VARCHAR(10) NOT NULL,
    date        DATE NOT NULL,
    data_type   VARCHAR(20) NOT NULL,  -- 'trades' or 'quotes'
    record_count INTEGER NOT NULL,
    ingested_at TIMESTAMPTZ DEFAULT NOW(),
    source      VARCHAR(50),  -- 'polygon_api', 'parquet_migration', etc.

    UNIQUE(symbol, date, data_type)
);

-- ============================================
-- COMPRESSION POLICY (for old data)
-- Compress data older than 7 days
-- ============================================
-- Note: Run these manually after initial data load
-- ALTER TABLE trades SET (timescaledb.compress);
-- ALTER TABLE quotes SET (timescaledb.compress);
-- SELECT add_compression_policy('trades', INTERVAL '7 days');
-- SELECT add_compression_policy('quotes', INTERVAL '7 days');

-- ============================================
-- RETENTION POLICY (optional)
-- Uncomment to auto-delete data older than 1 year
-- ============================================
-- SELECT add_retention_policy('trades', INTERVAL '1 year');
-- SELECT add_retention_policy('quotes', INTERVAL '1 year');

-- ============================================
-- USEFUL VIEWS
-- ============================================

-- View: Daily trade summary per symbol
CREATE OR REPLACE VIEW daily_trade_summary AS
SELECT
    time_bucket('1 day', time) AS date,
    symbol,
    COUNT(*) as trade_count,
    SUM(size) as total_volume,
    MIN(price) as low,
    MAX(price) as high,
    FIRST(price, time) as open,
    LAST(price, time) as close
FROM trades
GROUP BY time_bucket('1 day', time), symbol;

-- View: Check data coverage
CREATE OR REPLACE VIEW data_coverage AS
SELECT
    symbol,
    data_type,
    MIN(date) as first_date,
    MAX(date) as last_date,
    COUNT(DISTINCT date) as days_covered,
    SUM(record_count) as total_records
FROM ingestion_log
GROUP BY symbol, data_type;

-- Grant permissions (if needed for non-owner users)
-- GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO kronos;
