# ================================================================
# scripts/backfill.py
# ----------------------------------------------------------------
# One-time script to download historical NSE data.
# Run this once to populate daily_prices for the last N years.
#
# Usage:
#   python scripts/backfill.py            → downloads 1 year
#   python scripts/backfill.py --years 5  → downloads 5 years
#   python scripts/backfill.py --years 2  → downloads 2 years
#
# Skips dates already in DB — safe to re-run anytime.
# ================================================================

import sys
import argparse
import time
from datetime import date

sys.path.insert(0, '.')

from src.data.downloader import download_daily_data
from src.data.store import save_daily_prices, date_exists
from src.utils.calendar import get_trading_days, get_backfill_start_date
from src.utils.logger import get_logger
from config.settings import REQUEST_DELAY

log = get_logger("backfill")


def run_backfill(years: int = 1):
    """
    Downloads and stores historical bhavcopy data.

    Parameters
    ----------
    years : int
        How many years back to download. Default 1.
    """
    end_date   = date.today()
    start_date = get_backfill_start_date(years)

    trading_days = get_trading_days(start_date, end_date)

    log.info(f"Backfill started")
    log.info(f"Range     : {start_date} → {end_date}")
    log.info(f"Total days: {len(trading_days)}")
    log.info(f"Years back: {years}")

    total      = len(trading_days)
    downloaded = 0
    skipped    = 0
    failed     = 0

    for i, trading_date in enumerate(trading_days, 1):

        # Progress indicator
        pct = round(i / total * 100, 1)
        log.info(f"[{i}/{total} — {pct}%] Processing {trading_date}")

        # Skip if already in DB
        if date_exists(trading_date):
            log.info(f"Already exists — skipping {trading_date}")
            skipped += 1
            continue

        # Download
        df = download_daily_data(trading_date)

        if df is None or df.empty:
            log.info(f"No data for {trading_date} — holiday or unavailable")
            failed += 1
            time.sleep(REQUEST_DELAY)
            continue

        # Save to DB
        try:
            inserted = save_daily_prices(df)
            log.info(f"Saved {inserted} stocks for {trading_date}")
            downloaded += 1
        except Exception as e:
            log.error(f"Failed to save {trading_date}: {e}")
            failed += 1

        # Polite delay — avoid hammering NSE servers
        time.sleep(REQUEST_DELAY)

    # ── Summary ────────────────────────────────────────────────
    log.info("=" * 50)
    log.info("BACKFILL COMPLETE")
    log.info(f"Downloaded : {downloaded} days")
    log.info(f"Skipped    : {skipped} days (already in DB)")
    log.info(f"Failed     : {failed} days (holidays/unavailable)")
    log.info(f"Total      : {total} days processed")
    log.info("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NSE Historical Data Backfill")
    parser.add_argument(
        "--years",
        type=int,
        default=1,
        help="Number of years to backfill (default: 1)"
    )
    args = parser.parse_args()
    run_backfill(years=args.years)