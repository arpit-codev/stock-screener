# ================================================================
# config/settings.py
# ----------------------------------------------------------------
# Central configuration for the NSE Smart Money Screener.
# All constants, URLs, DB config, and paths live here.
# No other file should hardcode any of these values.
# ================================================================
import os
import sys
from pathlib import Path

# Add project root to Python path
# Ensures imports work from any directory
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from dotenv import load_dotenv

# ----------------------------------------------------------------
# Paths
# ----------------------------------------------------------------

# Project root — two levels up from this file
# config/settings.py → config/ → project root
ROOT_DIR = Path(__file__).resolve().parent.parent

# Log files directory
LOG_DIR = ROOT_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

# ----------------------------------------------------------------
# Environment variables
# Load from .env file at project root
# ----------------------------------------------------------------

load_dotenv(ROOT_DIR / ".env")

# ----------------------------------------------------------------
# Database
# ----------------------------------------------------------------

DB_CONFIG = {
    "host"    : os.getenv("DB_HOST",     "localhost"),
    "port"    : int(os.getenv("DB_PORT", "5432")),
    "dbname"  : os.getenv("DB_NAME",     "stockmarket"),
    "user"    : os.getenv("DB_USER",     "postgres"),
    "password": os.getenv("DB_PASSWORD", "password"),
}

# SQLAlchemy connection URL
# Used by pandas read_sql and SQLAlchemy engine
DB_URL = (
    f"postgresql+psycopg2://"
    f"{DB_CONFIG['user']}:{DB_CONFIG['password']}"
    f"@{DB_CONFIG['host']}:{DB_CONFIG['port']}"
    f"/{DB_CONFIG['dbname']}"
)

# ----------------------------------------------------------------
# NSE URLs
# ----------------------------------------------------------------

# Bhavcopy — daily OHLCV for all NSE stocks
# Variables: {year} = 2025, {month} = JUN, {date} = 09JUN2025
BHAVCOPY_URL = (
    "https://nsearchives.nseindia.com/content/historical/EQUITIES"
    "/{year}/{month}/cm{date}bhav.csv.zip"
)

# MTO — delivery data for all NSE stocks
# Variables: {date} = 09062025 (DDMMYYYY format)
MTO_URL = (
    "https://nsearchives.nseindia.com"
    "/archives/equities/mto/MTO_{date}.DAT"
)

# NSE homepage — must be visited first to get session cookies
# Without this, all archive downloads return 403
NSE_BASE_URL = "https://www.nseindia.com"

# ----------------------------------------------------------------
# NSE Request Headers
# NSE blocks requests that don't look like a real browser
# These headers simulate a Chrome browser visit
# ----------------------------------------------------------------

NSE_HEADERS = {
    "User-Agent"                : (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept"                    : (
        "text/html,application/xhtml+xml,application/xml;"
        "q=0.9,image/avif,image/webp,image/apng,*/*;"
        "q=0.8,application/signed-exchange;v=b3;q=0.7"
    ),
    "Accept-Language"           : "en-IN,en-GB;q=0.9,en-US;q=0.8,en;q=0.7",
    "Accept-Encoding"           : "gzip, deflate, br",
    "Connection"                : "keep-alive",
    "Upgrade-Insecure-Requests" : "1",
    "Sec-Fetch-Dest"            : "document",
    "Sec-Fetch-Mode"            : "navigate",
    "Sec-Fetch-Site"            : "none",
    "Sec-Fetch-User"            : "?1",
    "Cache-Control"             : "max-age=0",
}

# ----------------------------------------------------------------
# Data Constants
# ----------------------------------------------------------------

# Only these NSE series are stored
# EQ  = standard exchange traded stocks
# BE  = trade-for-trade segment (excluded — settlement risk)
# BZ  = trade-for-trade (excluded)
# SM  = small and medium enterprises (excluded)
# ST  = small and medium enterprises (excluded)
VALID_SERIES = ["EQ"]

# Minimum price filter
# Stocks below this are too illiquid / penny stocks
MIN_PRICE = 10.0

# Minimum average daily volume (shares)
# Stocks below this cannot be entered/exited cleanly
MIN_AVG_VOLUME = 50000

# How many years of historical data to backfill
BACKFILL_YEARS = 5

# Delay between NSE requests (seconds)
# NSE rate limits aggressive scrapers
# Keep at 1.5 minimum to avoid getting blocked
REQUEST_DELAY = 1.5

# ----------------------------------------------------------------
# Date Format Constants
# NSE uses specific date formats in URLs and CSV files
# ----------------------------------------------------------------

# Format for bhavcopy URL
# e.g. 09JUN2025
BHAVCOPY_DATE_FORMAT = "%d%b%Y"

# Format for MTO URL
# e.g. 09062025
MTO_DATE_FORMAT = "%d%m%Y"

# Format stored in database
# e.g. 2025-06-09
DB_DATE_FORMAT = "%Y-%m-%d"

# ----------------------------------------------------------------
# Telegram (optional)
# Leave blank to disable alerts
# ----------------------------------------------------------------

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID",   "")