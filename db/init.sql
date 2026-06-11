-- ================================================================
-- NSE Smart Money Screener
-- Database Schema — TimescaleDB
-- ================================================================
-- Tables:
--   1. daily_prices    → EOD OHLCV + Delivery% (EQ series only)
--   2. weekly_prices   → Weekly OHLCV + Delivery% (resampled)
-- ================================================================

-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;


-- ================================================================
-- TABLE 1 — daily_prices
-- ----------------------------------------------------------------
-- One row per stock per trading day.
--
-- Sources:
--   OHLCV + volume → NSE Bhavcopy CSV
--   delivery_qty   → NSE MTO DAT file
--   delivery_pct   → NSE MTO DAT file
--
-- Series filter applied at ingestion:
--   EQ series only
--   Excludes BE, BZ, SM, ST, SME, IL
-- ================================================================

CREATE TABLE IF NOT EXISTS daily_prices (

    -- Identity
    symbol              TEXT            NOT NULL,
    date                DATE            NOT NULL,

    -- Price — from Bhavcopy
    open                NUMERIC(12, 2)  NOT NULL,
    high                NUMERIC(12, 2)  NOT NULL,
    low                 NUMERIC(12, 2)  NOT NULL,
    close               NUMERIC(12, 2)  NOT NULL,

    -- Volume — from Bhavcopy
    -- Total traded quantity in shares (not rupee value)
    volume              BIGINT          NOT NULL,

    -- Delivery — from MTO file
    -- NULL when MTO file unavailable for that date (rare)
    delivery_qty        BIGINT,
    delivery_pct        NUMERIC(6, 2),

    -- Primary key — one row per stock per day
    PRIMARY KEY (symbol, date)
);

-- Convert to TimescaleDB hypertable
-- Partitions by month automatically
-- Makes date range queries significantly faster on 5yr data
SELECT create_hypertable(
    'daily_prices',
    'date',
    chunk_time_interval => INTERVAL '1 month',
    if_not_exists       => TRUE
);

-- Index 1 — full history of one symbol
-- Used by: OBV calculation, indicator engine, pattern scanner
-- Query: WHERE symbol = 'RELIANCE' ORDER BY date DESC
CREATE INDEX IF NOT EXISTS idx_daily_symbol
    ON daily_prices (symbol, date DESC);

-- Index 2 — all stocks for one date
-- Used by: daily screener full market scan
-- Query: WHERE date = '2025-06-09'
CREATE INDEX IF NOT EXISTS idx_daily_date
    ON daily_prices (date DESC);


-- ================================================================
-- TABLE 2 — weekly_prices
-- ----------------------------------------------------------------
-- One row per stock per week.
-- Resampled from daily_prices — computed every weekend.
-- Never written directly from NSE — always derived from daily.
--
-- Week boundary:
--   week_start = Monday of that week
--   open       = Monday's open price
--   high       = highest high across Mon–Fri
--   low        = lowest low across Mon–Fri
--   close      = Friday's close price
--   volume     = sum of all daily volumes Mon–Fri
--   delivery_qty  = sum of all daily delivery qty
--   delivery_pct  = average delivery % across the week
--
-- Why store weekly separately?
--   Pattern scanner (Phase 3) runs base formation +
--   cup and handle detection on weekly candles across
--   5 years × 1500 stocks.
--   Pre-aggregating weekly saves compute at scan time.
-- ================================================================

CREATE TABLE IF NOT EXISTS weekly_prices (

    -- Identity
    symbol              TEXT            NOT NULL,
    week_start          DATE            NOT NULL,   -- always a Monday

    -- Price
    open                NUMERIC(12, 2)  NOT NULL,   -- Monday open
    high                NUMERIC(12, 2)  NOT NULL,   -- week high
    low                 NUMERIC(12, 2)  NOT NULL,   -- week low
    close               NUMERIC(12, 2)  NOT NULL,   -- Friday close

    -- Volume
    volume              BIGINT          NOT NULL,   -- sum Mon-Fri

    -- Delivery
    delivery_qty        BIGINT,                     -- sum Mon-Fri
    delivery_pct        NUMERIC(6, 2),              -- avg Mon-Fri

    -- Primary key — one row per stock per week
    PRIMARY KEY (symbol, week_start)
);

-- Convert to TimescaleDB hypertable
SELECT create_hypertable(
    'weekly_prices',
    'week_start',
    chunk_time_interval => INTERVAL '6 months',
    if_not_exists       => TRUE
);

-- Index 1 — full weekly history of one symbol
-- Used by: weekly base formation scanner, OBV on weekly
CREATE INDEX IF NOT EXISTS idx_weekly_symbol
    ON weekly_prices (symbol, week_start DESC);

-- Index 2 — all stocks for one week
-- Used by: weekly screener scan
CREATE INDEX IF NOT EXISTS idx_weekly_date
    ON weekly_prices (week_start DESC);

-- ================================================================
-- TABLE 3 — yearly_open_levels
-- ----------------------------------------------------------------
-- Stores the yearly open price for every stock every year.
-- One row per stock per year.
-- ================================================================

CREATE TABLE IF NOT EXISTS yearly_open_levels (
    symbol          TEXT        NOT NULL,
    year            SMALLINT    NOT NULL,
    yearly_open     NUMERIC(12, 2) NOT NULL,
    first_trade_date DATE       NOT NULL,
    PRIMARY KEY (symbol, year)
);

CREATE INDEX IF NOT EXISTS idx_yearly_open_symbol
    ON yearly_open_levels (symbol, year DESC);


-- ================================================================
-- TABLE 4 — yearly_open_tests
-- ----------------------------------------------------------------
-- Every instance where price tested a yearly open level.
-- Records test type, delivery context, and 8-week outcome.
-- ================================================================

CREATE TABLE IF NOT EXISTS yearly_open_tests (
    id                  SERIAL PRIMARY KEY,
    symbol              TEXT            NOT NULL,
    year                SMALLINT        NOT NULL,
    yearly_open         NUMERIC(12, 2)  NOT NULL,
    test_date           DATE            NOT NULL,
    test_number         SMALLINT        NOT NULL,  -- 1st, 2nd, 3rd test

    -- Test type
    test_type           TEXT            NOT NULL,
    -- 'intraday_touch' / 'close_below' / 'false_breakdown'

    -- Price context at test
    close_at_test       NUMERIC(12, 2),
    low_at_test         NUMERIC(12, 2),
    volume_at_test      BIGINT,
    delivery_at_test    NUMERIC(6, 2),
    vol_ratio_at_test   NUMERIC(6, 2),  -- vs 22D avg

    -- Outcome (filled after 8 weeks)
    outcome             TEXT,
    -- 'strong_support' / 'weak_support' /
    -- 'false_breakdown' / 'breakdown' / 'base_formation'

    return_1w           NUMERIC(8, 2),
    return_2w           NUMERIC(8, 2),
    return_4w           NUMERIC(8, 2),
    return_8w           NUMERIC(8, 2),
    max_gain_8w         NUMERIC(8, 2),
    max_drawdown_8w     NUMERIC(8, 2),
    gave_10pct          BOOLEAN DEFAULT FALSE
);

CREATE INDEX IF NOT EXISTS idx_yot_symbol
    ON yearly_open_tests (symbol, test_date DESC);

CREATE INDEX IF NOT EXISTS idx_yot_year
    ON yearly_open_tests (year, test_date DESC);

-- ================================================================
-- VIEWS
-- ----------------------------------------------------------------
-- Frequently used queries — computed on the fly, no data stored
-- ================================================================


-- Latest trading day data for every symbol
-- Used by: screener to get today's snapshot per stock
CREATE OR REPLACE VIEW vw_latest_daily AS
SELECT DISTINCT ON (symbol)
    symbol,
    date,
    open,
    high,
    low,
    close,
    volume,
    delivery_qty,
    delivery_pct
FROM daily_prices
ORDER BY symbol, date DESC;


-- Latest week data for every symbol
CREATE OR REPLACE VIEW vw_latest_weekly AS
SELECT DISTINCT ON (symbol)
    symbol,
    week_start,
    open,
    high,
    low,
    close,
    volume,
    delivery_qty,
    delivery_pct
FROM weekly_prices
ORDER BY symbol, week_start DESC;


-- Volume averages per symbol from daily data
-- Excludes today so ratios compare TODAY vs PRIOR periods
-- Used by: screener volume awakening calculation
CREATE OR REPLACE VIEW vw_volume_averages AS
WITH latest AS (
    SELECT symbol, MAX(date) AS today
    FROM daily_prices
    GROUP BY symbol
)
SELECT
    d.symbol,
    l.today,

    -- Prior 5 trading days (1 week)
    ROUND(AVG(d.volume) FILTER (
        WHERE d.date <  l.today
        AND   d.date >= l.today - INTERVAL '9 days'
    ), 0)                                   AS vol_avg_5d,

    -- Prior 22 trading days (1 month)
    ROUND(AVG(d.volume) FILTER (
        WHERE d.date <  l.today
        AND   d.date >= l.today - INTERVAL '32 days'
    ), 0)                                   AS vol_avg_22d,

    -- Prior 65 trading days (1 quarter)
    ROUND(AVG(d.volume) FILTER (
        WHERE d.date <  l.today
        AND   d.date >= l.today - INTERVAL '95 days'
    ), 0)                                   AS vol_avg_65d,

    -- Prior 180 trading days (6 months)
    ROUND(AVG(d.volume) FILTER (
        WHERE d.date <  l.today
        AND   d.date >= l.today - INTERVAL '260 days'
    ), 0)                                   AS vol_avg_180d,

    -- Delivery % averages
    ROUND(AVG(d.delivery_pct) FILTER (
        WHERE d.date <  l.today
        AND   d.date >= l.today - INTERVAL '9 days'
    ), 2)                                   AS deliv_avg_5d,

    ROUND(AVG(d.delivery_pct) FILTER (
        WHERE d.date <  l.today
        AND   d.date >= l.today - INTERVAL '32 days'
    ), 2)                                   AS deliv_avg_22d

FROM daily_prices d
JOIN latest l ON d.symbol = l.symbol
GROUP BY d.symbol, l.today;


-- ================================================================
-- COMMENTS
-- ================================================================

COMMENT ON TABLE daily_prices
    IS 'NSE EOD OHLCV + delivery. EQ series only. Sources: Bhavcopy + MTO.';

COMMENT ON TABLE weekly_prices
    IS 'Weekly resampled OHLCV + delivery. Derived from daily_prices every weekend.';

COMMENT ON COLUMN daily_prices.delivery_qty
    IS 'Shares actually delivered (not squared off intraday). Source: NSE MTO file.';

COMMENT ON COLUMN daily_prices.delivery_pct
    IS 'delivery_qty as % of total volume. High % = institutional holding. Source: NSE MTO.';

COMMENT ON COLUMN weekly_prices.delivery_pct
    IS 'Average daily delivery % across the week.';

COMMENT ON COLUMN weekly_prices.week_start
    IS 'Monday of the trading week. Always a Monday date.';