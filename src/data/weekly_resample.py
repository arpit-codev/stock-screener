# ================================================================
# src/data/resample.py
# ----------------------------------------------------------------
# Resamples daily_prices into weekly_prices.
#
# Weekly candle logic:
#   open         = first trading day's open of the week
#   high         = highest high across all days of the week
#   low          = lowest low across all days of the week
#   close        = last trading day's close of the week
#   volume       = sum of all days
#   delivery_qty = sum of all days
#   delivery_pct = average across all days (weighted by volume)
#
# Run every weekend after Friday's data is downloaded.
# ================================================================

from datetime import date

import pandas as pd
from sqlalchemy import text

from config.settings import DB_URL
from src.data.store import save_weekly_prices, get_engine
from src.utils.calendar import get_week_start
from src.utils.logger import get_logger

log = get_logger(__name__)


# ================================================================
# BUILD WEEKLY CANDLES
# ================================================================

def build_weekly_prices(
    from_date: date = None,
    to_date: date = None
) -> pd.DataFrame:
    """
    Reads daily_prices from DB and aggregates into weekly candles.

    Parameters
    ----------
    from_date : date, optional
        Start date for resampling. Defaults to earliest in DB.
    to_date : date, optional
        End date for resampling. Defaults to latest in DB.

    Returns
    -------
    pd.DataFrame
        Weekly candles with columns:
            symbol, week_start, open, high, low, close,
            volume, delivery_qty, delivery_pct
    """
    log.info(f"Building weekly prices: {from_date} → {to_date}")

    # ── Load daily data from DB ────────────────────────────────
    engine = get_engine()

    if from_date and to_date:
        sql = """
            SELECT
                symbol, date, open, high, low, close,
                volume, delivery_qty, delivery_pct
            FROM daily_prices
            WHERE date >= :from_date
            AND   date <= :to_date
            ORDER BY symbol, date ASC
        """
        df = pd.read_sql(
            text(sql),
            engine,
            params={"from_date": from_date, "to_date": to_date}
        )
    else:
        sql = """
            SELECT
                symbol, date, open, high, low, close,
                volume, delivery_qty, delivery_pct
            FROM daily_prices
            ORDER BY symbol, date ASC
        """
        df = pd.read_sql(text(sql), engine)

    if df.empty:
        log.warning("No daily data found for resampling")
        return pd.DataFrame()

    log.info(f"Loaded {len(df)} daily rows for resampling")

    # ── Add week_start column ──────────────────────────────────
    # week_start = Monday of that week
    df["date"]       = pd.to_datetime(df["date"])
    df["week_start"] = df["date"].apply(
        lambda d: get_week_start(d.date())
    )

    # ── Aggregate by symbol + week_start ──────────────────────
    weekly_rows = []

    for (symbol, week_start), week_df in df.groupby(["symbol", "week_start"]):
        week_df = week_df.sort_values("date")

        # Price
        open_price  = week_df.iloc[0]["open"]    # first day open
        high_price  = week_df["high"].max()       # week high
        low_price   = week_df["low"].min()        # week low
        close_price = week_df.iloc[-1]["close"]   # last day close

        # Volume
        total_volume = week_df["volume"].sum()

        # Delivery
        total_delivery_qty = week_df["delivery_qty"].sum() \
            if week_df["delivery_qty"].notna().any() else None

        # Delivery % — weighted average by volume
        # (days with higher volume have more weight)
        valid_deliv = week_df[week_df["delivery_pct"].notna()]
        if not valid_deliv.empty and valid_deliv["volume"].sum() > 0:
            weighted_deliv_pct = (
                (valid_deliv["delivery_pct"] * valid_deliv["volume"]).sum()
                / valid_deliv["volume"].sum()
            )
            weighted_deliv_pct = round(float(weighted_deliv_pct), 2)
        else:
            weighted_deliv_pct = None

        weekly_rows.append({
            "symbol"       : symbol,
            "week_start"   : week_start,
            "open"         : float(open_price),
            "high"         : float(high_price),
            "low"          : float(low_price),
            "close"        : float(close_price),
            "volume"       : int(total_volume),
            "delivery_qty" : int(total_delivery_qty) if total_delivery_qty else None,
            "delivery_pct" : weighted_deliv_pct,
        })

    weekly_df = pd.DataFrame(weekly_rows)
    weekly_df = weekly_df.sort_values(["symbol", "week_start"])
    weekly_df = weekly_df.reset_index(drop=True)

    log.info(f"Built {len(weekly_df)} weekly candles "
             f"for {weekly_df['symbol'].nunique()} symbols")

    return weekly_df


# ================================================================
# RESAMPLE AND SAVE
# ================================================================

def resample_and_save(
    from_date: date = None,
    to_date: date = None
) -> int:
    """
    Builds weekly candles and saves them to weekly_prices table.
    Main function called by scripts.

    Parameters
    ----------
    from_date : date, optional
        Start date. Defaults to all available data.
    to_date : date, optional
        End date. Defaults to all available data.

    Returns
    -------
    int
        Number of weekly rows inserted.
    """
    # Build weekly candles
    weekly_df = build_weekly_prices(from_date, to_date)

    if weekly_df.empty:
        log.warning("No weekly candles built — nothing to save")
        return 0

    # Save to DB
    inserted = save_weekly_prices(weekly_df)
    log.info(f"Weekly resample complete — {inserted} rows saved")
    return inserted


# ================================================================
# INCREMENTAL UPDATE
# ================================================================

def resample_last_week() -> int:
    """
    Resamples only the most recent completed week.
    Called every Monday morning to add last week's candle.

    Returns
    -------
    int
        Number of rows inserted (should be ~2040 — one per symbol).
    """
    from datetime import timedelta

    today      = date.today()
    # Go back to find last completed Friday
    days_back  = (today.weekday() + 3) % 7 + 1
    last_friday = today - timedelta(days=days_back)
    last_monday = get_week_start(last_friday)

    log.info(f"Resampling last week: {last_monday} → {last_friday}")
    return resample_and_save(
        from_date = last_monday,
        to_date   = last_friday
    )