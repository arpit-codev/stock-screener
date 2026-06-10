# ================================================================
# scripts/run_daily_sync.py
# ----------------------------------------------------------------
# Daily data sync script.
# Run this every evening after 6 PM market close.
#
# What it does:
#   1. Checks what's the latest date in DB
#   2. Downloads any missing trading days
#   3. Saves to daily_prices
#   4. Resamples last week into weekly_prices (if Friday)
#
# Usage:
#   python scripts/run_daily_sync.py
#   python scripts/run_daily_sync.py --date 2026-06-09  (specific date)
#   python scripts/run_daily_sync.py --force            (re-download today)
# ================================================================

import sys
import argparse
import time
from datetime import date, timedelta

sys.path.insert(0, '.')

from src.data.downloader import download_daily_data
from src.data.store import (
    save_daily_prices,
    date_exists,
    get_latest_date
)
from src.data.weekly_resample import resample_last_week
from src.utils.calendar import (
    get_trading_days,
    is_trading_day
)
from src.utils.logger import get_logger
from config.settings import REQUEST_DELAY

log = get_logger("daily_sync")


def run_sync(target_date: date = None, force: bool = False):
    """
    Syncs NSE data up to target_date.

    Parameters
    ----------
    target_date : date, optional
        Date to sync up to. Defaults to today.
    force : bool
        If True, re-downloads even if date exists in DB.
    """
    if target_date is None:
        target_date = date.today()

    log.info("=" * 50)
    log.info("DAILY SYNC STARTED")
    log.info(f"Target date : {target_date}")
    log.info("=" * 50)

    # ── Find missing dates ─────────────────────────────────────
    latest_in_db = get_latest_date()

    if latest_in_db is None:
        log.warning("DB is empty — run backfill.py first")
        return

    log.info(f"Latest date in DB : {latest_in_db}")

    # Get all trading days between latest in DB and target
    start = latest_in_db + timedelta(days=1)
    missing_days = get_trading_days(start, target_date)

    if not missing_days:
        log.info("DB is already up to date — nothing to sync")
        return

    log.info(f"Missing trading days: {len(missing_days)}")

    # ── Download and save missing days ─────────────────────────
    downloaded = 0
    failed     = 0

    for trading_date in missing_days:

        # Skip if exists and not forcing
        if not force and date_exists(trading_date):
            log.info(f"Already exists — skipping {trading_date}")
            continue

        log.info(f"Downloading {trading_date}...")
        df = download_daily_data(trading_date)

        if df is None or df.empty:
            log.info(f"No data for {trading_date} — holiday or unavailable")
            failed += 1
            time.sleep(REQUEST_DELAY)
            continue

        try:
            inserted = save_daily_prices(df)
            log.info(f"Saved {inserted} stocks for {trading_date}")
            downloaded += 1
        except Exception as e:
            log.error(f"Failed to save {trading_date}: {e}")
            failed += 1

        time.sleep(REQUEST_DELAY)

    # ── Resample weekly if today is Friday ─────────────────────
    # Builds weekly candle for the completed week
    if target_date.weekday() == 4:   # 4 = Friday
        log.info("Friday detected — resampling last week...")
        try:
            weekly_inserted = resample_last_week()
            log.info(f"Weekly resample: {weekly_inserted} rows saved")
        except Exception as e:
            log.error(f"Weekly resample failed: {e}")

    # ── Summary ────────────────────────────────────────────────
    log.info("=" * 50)
    log.info("DAILY SYNC COMPLETE")
    log.info(f"Downloaded : {downloaded} days")
    log.info(f"Failed     : {failed} days")
    log.info("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NSE Daily Data Sync")
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Target date to sync to (YYYY-MM-DD). Defaults to today."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if date already exists in DB"
    )
    args = parser.parse_args()

    target = None
    if args.date:
        from datetime import datetime
        target = datetime.strptime(args.date, "%Y-%m-%d").date()

    run_sync(target_date=target, force=args.force)