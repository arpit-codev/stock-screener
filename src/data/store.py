# ================================================================
# src/data/store.py
# ----------------------------------------------------------------
# Database read/write for daily_prices and weekly_prices tables.
#
# Two responsibilities:
#   WRITE — save downloaded data into TimescaleDB
#   READ  — load historical data for indicator calculations
#
# Uses psycopg2 for writes (fast batch insert)
# Uses pandas + SQLAlchemy for reads (returns DataFrame directly)
# ================================================================

from datetime import date
from typing import Optional

import pandas as pd
import psycopg2
import psycopg2.extras
from sqlalchemy import create_engine, text

from config.settings import DB_CONFIG, DB_URL
from src.utils.logger import get_logger

log = get_logger(__name__)


# ================================================================
# CONNECTION HELPERS
# ================================================================

def get_connection():
    """
    Returns a raw psycopg2 connection.
    Used for writes — faster than SQLAlchemy for bulk inserts.
    """
    return psycopg2.connect(**DB_CONFIG)


def get_engine():
    """
    Returns a SQLAlchemy engine.
    Used for reads — pandas read_sql works directly with it.
    """
    return create_engine(DB_URL)


# ================================================================
# WRITE — daily_prices
# ================================================================

def save_daily_prices(df: pd.DataFrame) -> int:
    """
    Inserts daily price rows into daily_prices table.
    Skips rows that already exist (ON CONFLICT DO NOTHING).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame from downloader.download_daily_data()
        Must have columns:
            symbol, date, open, high, low, close,
            volume, delivery_qty, delivery_pct

    Returns
    -------
    int
        Number of rows actually inserted.
        (Less than len(df) if some dates already exist)
    """
    if df is None or df.empty:
        log.warning("save_daily_prices called with empty DataFrame")
        return 0

    # Convert DataFrame rows to list of tuples for bulk insert
    records = [
        (
            str(row["symbol"]),
            row["date"],
            float(row["open"]),
            float(row["high"]),
            float(row["low"]),
            float(row["close"]),
            int(row["volume"]),
            int(row["delivery_qty"]) if pd.notna(row["delivery_qty"]) else None,
            float(row["delivery_pct"]) if pd.notna(row["delivery_pct"]) else None,
        )
        for _, row in df.iterrows()
    ]

    sql = """
        INSERT INTO daily_prices (
            symbol, date, open, high, low, close,
            volume, delivery_qty, delivery_pct
        )
        VALUES %s
        ON CONFLICT (symbol, date) DO NOTHING
    """

    try:
        conn = get_connection()
        cursor = conn.cursor()

        # execute_values — fastest way to bulk insert in psycopg2
        psycopg2.extras.execute_values(
            cursor,
            sql,
            records,
            page_size=500       # insert 500 rows per batch
        )

        inserted = len(records)
        conn.commit()
        cursor.close()
        conn.close()

        log.info(f"Saved {inserted} rows to daily_prices")
        return inserted

    except Exception as e:
        log.error(f"Failed to save daily_prices: {e}")
        raise


# ================================================================
# WRITE — weekly_prices
# ================================================================

def save_weekly_prices(df: pd.DataFrame) -> int:
    """
    Inserts weekly price rows into weekly_prices table.
    Skips rows that already exist (ON CONFLICT DO NOTHING).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame from resample.build_weekly_prices()
        Must have columns:
            symbol, week_start, open, high, low, close,
            volume, delivery_qty, delivery_pct

    Returns
    -------
    int
        Number of rows actually inserted.
    """
    if df is None or df.empty:
        log.warning("save_weekly_prices called with empty DataFrame")
        return 0

    records = [
        (
            str(row["symbol"]),
            row["week_start"],
            float(row["open"]),
            float(row["high"]),
            float(row["low"]),
            float(row["close"]),
            int(row["volume"]),
            int(row["delivery_qty"]) if pd.notna(row["delivery_qty"]) else None,
            float(row["delivery_pct"]) if pd.notna(row["delivery_pct"]) else None,
        )
        for _, row in df.iterrows()
    ]

    sql = """
        INSERT INTO weekly_prices (
            symbol, week_start, open, high, low, close,
            volume, delivery_qty, delivery_pct
        )
        VALUES %s
        ON CONFLICT (symbol, week_start) DO NOTHING
    """

    try:
        conn = get_connection()
        cursor = conn.cursor()

        psycopg2.extras.execute_values(
            cursor,
            sql,
            records,
            page_size=500
        )

        conn.commit()
        cursor.close()
        conn.close()

        inserted = len(records)

        log.info(f"Saved {inserted} rows to weekly_prices")
        return inserted

    except Exception as e:
        log.error(f"Failed to save weekly_prices: {e}")
        raise


# ================================================================
# READ — daily_prices
# ================================================================

def load_daily_prices(
    symbol: str,
    from_date: date,
    to_date: date
) -> pd.DataFrame:
    """
    Loads daily prices for a single symbol between two dates.
    Used by indicator calculations — OBV, volume ratios, etc.

    Parameters
    ----------
    symbol : str
        NSE trading symbol e.g. 'RELIANCE'
    from_date : date
        Start date (inclusive)
    to_date : date
        End date (inclusive)

    Returns
    -------
    pd.DataFrame
        Sorted by date ascending.
        Empty DataFrame if no data found.
    """
    sql = """
        SELECT
            symbol, date, open, high, low, close,
            volume, delivery_qty, delivery_pct
        FROM daily_prices
        WHERE symbol   = :symbol
        AND   date     >= :from_date
        AND   date     <= :to_date
        ORDER BY date ASC
    """

    try:
        engine = get_engine()
        df = pd.read_sql(
            text(sql),
            engine,
            params={
                "symbol"    : symbol,
                "from_date" : from_date,
                "to_date"   : to_date,
            }
        )
        log.debug(f"Loaded {len(df)} rows for {symbol} ({from_date} → {to_date})")
        return df

    except Exception as e:
        log.error(f"Failed to load daily_prices for {symbol}: {e}")
        return pd.DataFrame()


def load_all_symbols_for_date(trading_date: date) -> pd.DataFrame:
    """
    Loads all stocks for a single trading date.
    Used by the screener — full market scan for one day.

    Parameters
    ----------
    trading_date : date
        The date to load.

    Returns
    -------
    pd.DataFrame
        All EQ stocks for that date.
        Empty DataFrame if no data found.
    """
    sql = """
        SELECT
            symbol, date, open, high, low, close,
            volume, delivery_qty, delivery_pct
        FROM daily_prices
        WHERE date = :trading_date
        ORDER BY symbol ASC
    """

    try:
        engine = get_engine()
        df = pd.read_sql(
            text(sql),
            engine,
            params={"trading_date": trading_date}
        )
        log.info(f"Loaded {len(df)} stocks for {trading_date}")
        return df

    except Exception as e:
        log.error(f"Failed to load daily_prices for {trading_date}: {e}")
        return pd.DataFrame()


def load_recent_history(
    symbol: str,
    days: int = 365
) -> pd.DataFrame:
    """
    Loads last N days of daily prices for a symbol.
    Used by indicator engine — OBV, volume averages, price changes.

    Parameters
    ----------
    symbol : str
        NSE trading symbol
    days : int
        Number of calendar days to look back. Default 365.

    Returns
    -------
    pd.DataFrame
        Sorted by date ascending.
    """
    sql = """
        SELECT
            symbol, date, open, high, low, close,
            volume, delivery_qty, delivery_pct
        FROM daily_prices
        WHERE symbol = :symbol
        AND   date  >= CURRENT_DATE - (:days * INTERVAL '1 day')
        ORDER BY date ASC
    """

    try:
        engine = get_engine()
        df = pd.read_sql(
            text(sql),
            engine,
            params={
                "symbol" : symbol,
                "days"   : days,
            }
        )
        log.debug(f"Loaded {len(df)} rows for {symbol} (last {days} days)")
        return df

    except Exception as e:
        log.error(f"Failed to load recent history for {symbol}: {e}")
        return pd.DataFrame()


# ================================================================
# UTILITIES
# ================================================================

def get_latest_date() -> Optional[date]:
    """
    Returns the most recent date available in daily_prices.
    Used by sync script to know where to resume from.

    Returns
    -------
    date or None
        Latest date in DB. None if table is empty.
    """
    sql = "SELECT MAX(date) AS latest FROM daily_prices"

    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute(sql)
        result = cursor.fetchone()
        cursor.close()
        conn.close()

        if result and result[0]:
            log.info(f"Latest date in DB: {result[0]}")
            return result[0]
        else:
            log.info("daily_prices table is empty")
            return None

    except Exception as e:
        log.error(f"Failed to get latest date: {e}")
        return None


def get_all_symbols() -> list[str]:
    """
    Returns list of all distinct symbols in daily_prices.
    Used by screener to iterate over all stocks.

    Returns
    -------
    list[str]
        Sorted list of NSE symbols.
    """
    sql = "SELECT DISTINCT symbol FROM daily_prices ORDER BY symbol"

    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute(sql)
        symbols = [row[0] for row in cursor.fetchall()]
        cursor.close()
        conn.close()
        log.info(f"Total symbols in DB: {len(symbols)}")
        return symbols

    except Exception as e:
        log.error(f"Failed to get symbols: {e}")
        return []


def date_exists(trading_date: date) -> bool:
    sql = "SELECT 1 FROM daily_prices WHERE date = %s LIMIT 1"

    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute(sql, (trading_date,))
        exists = cursor.fetchone() is not None
        cursor.close()
        conn.close()
        return exists

    except Exception as e:
        log.error(f"Failed to check date existence: {e}")
        return False


def save_yearly_open_levels(df: pd.DataFrame) -> int:
    """Saves yearly open levels to DB."""
    if df is None or df.empty:
        return 0

    records = [
        (
            str(row["symbol"]),
            int(row["year"]),
            float(row["yearly_open"]),
            row["first_trade_date"],
        )
        for _, row in df.iterrows()
    ]

    sql = """
        INSERT INTO yearly_open_levels (
            symbol, year, yearly_open, first_trade_date
        )
        VALUES %s
        ON CONFLICT (symbol, year) DO NOTHING
    """

    try:
        conn = get_connection()
        cursor = conn.cursor()
        psycopg2.extras.execute_values(cursor, sql, records, page_size=500)
        conn.commit()
        cursor.close()
        conn.close()
        inserted = len(records)
        log.info(f"Saved {inserted} yearly open levels")
        return inserted
    except Exception as e:
        log.error(f"Failed to save yearly open levels: {e}")
        raise


def save_yearly_open_tests(df: pd.DataFrame) -> int:
    """Saves yearly open test events to DB."""
    if df is None or df.empty:
        return 0

    records = [
        (
            str(row["symbol"]),
            int(row["year"]),
            float(row["yearly_open"]),
            row["test_date"],
            int(row["test_number"]),
            str(row["test_type"]),
            float(row["close_at_test"])      if pd.notna(row.get("close_at_test"))      else None,
            float(row["low_at_test"])        if pd.notna(row.get("low_at_test"))        else None,
            int(row["volume_at_test"])       if pd.notna(row.get("volume_at_test"))     else None,
            float(row["delivery_at_test"])   if pd.notna(row.get("delivery_at_test"))   else None,
            float(row["vol_ratio_at_test"])  if pd.notna(row.get("vol_ratio_at_test")) else None,
            str(row["outcome"])              if pd.notna(row.get("outcome"))            else None,
            float(row["return_1w"])          if pd.notna(row.get("return_1w"))          else None,
            float(row["return_2w"])          if pd.notna(row.get("return_2w"))          else None,
            float(row["return_4w"])          if pd.notna(row.get("return_4w"))          else None,
            float(row["return_8w"])          if pd.notna(row.get("return_8w"))          else None,
            float(row["max_gain_8w"])        if pd.notna(row.get("max_gain_8w"))        else None,
            float(row["max_drawdown_8w"])    if pd.notna(row.get("max_drawdown_8w"))    else None,
            bool(row["gave_10pct"])          if pd.notna(row.get("gave_10pct"))         else False,
        )
        for _, row in df.iterrows()
    ]

    sql = """
        INSERT INTO yearly_open_tests (
            symbol, year, yearly_open, test_date, test_number,
            test_type, close_at_test, low_at_test, volume_at_test,
            delivery_at_test, vol_ratio_at_test, outcome,
            return_1w, return_2w, return_4w, return_8w,
            max_gain_8w, max_drawdown_8w, gave_10pct
        )
        VALUES %s
        ON CONFLICT DO NOTHING
    """

    try:
        conn = get_connection()
        cursor = conn.cursor()
        psycopg2.extras.execute_values(cursor, sql, records, page_size=500)
        conn.commit()
        cursor.close()
        conn.close()
        inserted = len(records)
        log.info(f"Saved {inserted} yearly open tests")
        return inserted
    except Exception as e:
        log.error(f"Failed to save yearly open tests: {e}")
        raise