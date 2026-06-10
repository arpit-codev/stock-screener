# ================================================================
# src/data/downloader.py
# ----------------------------------------------------------------
# Downloads EOD data from NSE using jugaad-data library.
#
# Uses full_bhavcopy_save() which downloads OHLCV + delivery %
# in a single call — handles NSE session/cookies internally.
#
# One function: download_daily_data(date) → pd.DataFrame | None
# ================================================================

import os
import tempfile
from datetime import date

import pandas as pd
from jugaad_data.nse import full_bhavcopy_save

from config.settings import VALID_SERIES, MIN_PRICE
from src.utils.logger import get_logger

log = get_logger(__name__)

# CSV columns → our DB column names
COLUMN_MAP = {
    "SYMBOL"       : "symbol",
    " SERIES"      : "series",
    " DATE1"       : "date",
    " OPEN_PRICE"  : "open",
    " HIGH_PRICE"  : "high",
    " LOW_PRICE"   : "low",
    " CLOSE_PRICE" : "close",
    " TTL_TRD_QNTY": "volume",
    " DELIV_QTY"   : "delivery_qty",
    " DELIV_PER"   : "delivery_pct",
}


def download_daily_data(trading_date: date) -> pd.DataFrame | None:
    """
    Downloads full bhavcopy for a given date.
    Returns cleaned DataFrame ready to insert into daily_prices.

    Parameters
    ----------
    trading_date : date
        The trading date to download.

    Returns
    -------
    pd.DataFrame or None
        Columns: symbol, date, open, high, low, close,
                 volume, delivery_qty, delivery_pct
        Returns None if data unavailable (holiday/weekend).
    """
    log.info(f"Downloading bhavcopy: {trading_date}")

    with tempfile.TemporaryDirectory() as tmp_dir:
        try:
            full_bhavcopy_save(trading_date, tmp_dir)
        except Exception as e:
            log.warning(f"Download failed for {trading_date}: {e}")
            return None

        # Find downloaded CSV
        files = os.listdir(tmp_dir)
        csv_files = [f for f in files if f.endswith(".csv")]

        if not csv_files:
            log.info(f"No data available for {trading_date} — likely holiday/weekend")
            return None

        csv_path = os.path.join(tmp_dir, csv_files[0])
        df = pd.read_csv(csv_path)

    log.info(f"Downloaded {len(df)} rows for {trading_date}")

    # ── Rename columns ─────────────────────────────────────────
    df = df.rename(columns=COLUMN_MAP)

    # ── Keep only columns we need ──────────────────────────────
    keep = ["symbol", "series", "date", "open", "high",
            "low", "close", "volume", "delivery_qty", "delivery_pct"]
    df = df[keep].copy()

    # ── Filter EQ series only ──────────────────────────────────
    df["series"] = df["series"].str.strip()
    df = df[df["series"].isin(VALID_SERIES)].copy()
    df = df.drop(columns=["series"])
    log.info(f"After EQ filter: {len(df)} stocks")

    # ── Clean data types ───────────────────────────────────────
    df["symbol"] = df["symbol"].str.strip()
    df["date"] = pd.to_datetime(df["date"].str.strip(), format="%d-%b-%Y").dt.date

    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["volume"]       = pd.to_numeric(df["volume"],       errors="coerce")
    df["delivery_qty"] = pd.to_numeric(df["delivery_qty"], errors="coerce")
    df["delivery_pct"] = pd.to_numeric(df["delivery_pct"], errors="coerce")

    # ── Filter minimum price ───────────────────────────────────
    before = len(df)
    df = df[df["close"] >= MIN_PRICE]
    removed = before - len(df)
    if removed > 0:
        log.info(f"Removed {removed} stocks below ₹{MIN_PRICE}")

    # ── Drop rows with missing critical values ─────────────────
    df = df.dropna(subset=["symbol", "date", "close", "volume"])
    df = df.reset_index(drop=True)

    log.info(f"Clean data ready: {len(df)} stocks for {trading_date}")
    return df