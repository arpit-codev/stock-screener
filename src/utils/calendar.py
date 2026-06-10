# ================================================================
# src/utils/calendar.py
# ----------------------------------------------------------------
# NSE trading day helpers.
# Tells the downloader which dates to download
# and which to skip (weekends + NSE holidays).
#
# Usage:
#   from src.utils.calendar import get_trading_days
#   days = get_trading_days(start_date, end_date)
# ================================================================

from datetime import date, timedelta
from src.utils.logger import get_logger

log = get_logger(__name__)

# ----------------------------------------------------------------
# NSE Holidays
# ----------------------------------------------------------------
# NSE publishes holidays annually.
# Add each year's holidays here as they are announced.
# Format: date(YYYY, MM, DD)
#
# Source: https://www.nseindia.com/global/content/market_timings_holidays/market_timings_holidays.htm
# ----------------------------------------------------------------

NSE_HOLIDAYS = {

    # 2021
    date(2021, 1, 26),   # Republic Day
    date(2021, 3, 11),   # Mahashivratri
    date(2021, 3, 29),   # Holi
    date(2021, 4, 2),    # Good Friday
    date(2021, 4, 14),   # Dr. Ambedkar Jayanti
    date(2021, 4, 21),   # Ram Navami
    date(2021, 5, 13),   # Id-Ul-Fitr
    date(2021, 7, 21),   # Bakri Id
    date(2021, 8, 19),   # Muharram
    date(2021, 10, 15),  # Dussehra
    date(2021, 11, 4),   # Diwali Laxmi Pujan
    date(2021, 11, 5),   # Diwali Balipratipada
    date(2021, 11, 19),  # Gurunanak Jayanti

    # 2022
    date(2022, 1, 26),   # Republic Day
    date(2022, 3, 1),    # Mahashivratri
    date(2022, 3, 18),   # Holi
    date(2022, 4, 14),   # Dr. Ambedkar Jayanti
    date(2022, 4, 15),   # Good Friday
    date(2022, 5, 3),    # Id-Ul-Fitr
    date(2022, 8, 9),    # Muharram
    date(2022, 8, 15),   # Independence Day
    date(2022, 8, 31),   # Ganesh Chaturthi
    date(2022, 10, 2),   # Gandhi Jayanti
    date(2022, 10, 5),   # Dussehra
    date(2022, 10, 24),  # Diwali Laxmi Pujan
    date(2022, 10, 26),  # Diwali Balipratipada
    date(2022, 11, 8),   # Gurunanak Jayanti

    # 2023
    date(2023, 1, 26),   # Republic Day
    date(2023, 3, 7),    # Holi
    date(2023, 3, 30),   # Ram Navami
    date(2023, 4, 4),    # Mahavir Jayanti
    date(2023, 4, 7),    # Good Friday
    date(2023, 4, 14),   # Dr. Ambedkar Jayanti
    date(2023, 4, 21),   # Id-Ul-Fitr
    date(2023, 6, 28),   # Bakri Id
    date(2023, 8, 15),   # Independence Day
    date(2023, 9, 19),   # Ganesh Chaturthi
    date(2023, 10, 2),   # Gandhi Jayanti
    date(2023, 10, 24),  # Dussehra
    date(2023, 11, 13),  # Diwali Laxmi Pujan
    date(2023, 11, 14),  # Diwali Balipratipada
    date(2023, 11, 27),  # Gurunanak Jayanti
    date(2023, 12, 25),  # Christmas

    # 2024
    date(2024, 1, 22),   # Ram Mandir Consecration
    date(2024, 1, 26),   # Republic Day
    date(2024, 3, 25),   # Holi
    date(2024, 3, 29),   # Good Friday
    date(2024, 4, 14),   # Dr. Ambedkar Jayanti
    date(2024, 4, 17),   # Ram Navami
    date(2024, 4, 21),   # Mahavir Jayanti
    date(2024, 5, 23),   # Buddha Purnima
    date(2024, 6, 17),   # Bakri Id
    date(2024, 7, 17),   # Muharram
    date(2024, 8, 15),   # Independence Day
    date(2024, 10, 2),   # Gandhi Jayanti
    date(2024, 10, 12),  # Dussehra
    date(2024, 11, 1),   # Diwali Laxmi Pujan
    date(2024, 11, 15),  # Gurunanak Jayanti
    date(2024, 12, 25),  # Christmas

    # 2025
    date(2025, 1, 26),   # Republic Day
    date(2025, 2, 26),   # Mahashivratri
    date(2025, 3, 14),   # Holi
    date(2025, 3, 31),   # Id-Ul-Fitr
    date(2025, 4, 10),   # Shri Ram Navami
    date(2025, 4, 14),   # Dr. Ambedkar Jayanti
    date(2025, 4, 18),   # Good Friday
    date(2025, 5, 1),    # Maharashtra Day
    date(2025, 8, 15),   # Independence Day
    date(2025, 8, 27),   # Ganesh Chaturthi
    date(2025, 10, 2),   # Gandhi Jayanti
    date(2025, 10, 2),   # Dussehra
    date(2025, 10, 20),  # Diwali Laxmi Pujan
    date(2025, 10, 21),  # Diwali Balipratipada
    date(2025, 11, 5),   # Gurunanak Jayanti
    date(2025, 12, 25),  # Christmas

# 2026
    date(2026, 1, 26),   # Republic Day
    date(2026, 2, 26),   # Mahashivratri
    date(2026, 3, 20),   # Holi
    date(2026, 3, 31),   # Id-Ul-Fitr (Eid) — tentative
    date(2026, 4, 2),    # Ram Navami
    date(2026, 4, 3),    # Good Friday
    date(2026, 4, 14),   # Dr. Ambedkar Jayanti
    date(2026, 5, 1),    # Maharashtra Day
    date(2026, 8, 15),   # Independence Day
    date(2026, 8, 17),   # Ganesh Chaturthi — tentative
    date(2026, 10, 2),   # Gandhi Jayanti
    date(2026, 10, 22),  # Dussehra — tentative
    date(2026, 11, 9),   # Diwali Laxmi Pujan — tentative
    date(2026, 11, 10),  # Diwali Balipratipada — tentative
    date(2026, 12, 25),  # Christmas
}


# ----------------------------------------------------------------
# Core Functions
# ----------------------------------------------------------------

def is_trading_day(d: date) -> bool:
    """
    Returns True if the given date is a valid NSE trading day.

    A date is NOT a trading day if:
    - It is a Saturday or Sunday
    - It is in the NSE_HOLIDAYS set

    Parameters
    ----------
    d : date
        The date to check.

    Returns
    -------
    bool
    """
    if d.weekday() >= 5:        # 5 = Saturday, 6 = Sunday
        return False
    if d in NSE_HOLIDAYS:
        return False
    return True


def get_trading_days(
    start_date: date,
    end_date: date
) -> list[date]:
    """
    Returns all valid NSE trading days between
    start_date and end_date inclusive.

    Parameters
    ----------
    start_date : date
        Start of the date range.
    end_date : date
        End of the date range.

    Returns
    -------
    list[date]
        Sorted list of trading days oldest to newest.

    Example
    -------
    from src.utils.calendar import get_trading_days
    from datetime import date

    days = get_trading_days(date(2025, 6, 2), date(2025, 6, 9))
    # Returns: [date(2025,6,2), date(2025,6,3), date(2025,6,4),
    #           date(2025,6,5), date(2025,6,6), date(2025,6,9)]
    # Note: June 7 (Sat) and June 8 (Sun) are excluded
    """
    if start_date > end_date:
        log.warning(f"start_date {start_date} is after end_date {end_date} — returning empty list")
        return []

    days = []
    current = start_date
    while current <= end_date:
        if is_trading_day(current):
            days.append(current)
        current += timedelta(days=1)

    log.debug(f"Trading days {start_date} → {end_date}: {len(days)} days")
    return days


def get_last_n_trading_days(
    n: int,
    end_date: date = None
) -> list[date]:
    """
    Returns the last N trading days ending on end_date.
    If end_date not provided, uses today.

    Parameters
    ----------
    n : int
        Number of trading days to return.
    end_date : date, optional
        Last date in the range. Defaults to today.

    Returns
    -------
    list[date]
        Sorted list oldest to newest.

    Example
    -------
    days = get_last_n_trading_days(5)
    # Returns last 5 trading days up to today
    """
    if end_date is None:
        end_date = date.today()

    days = []
    current = end_date

    while len(days) < n:
        if is_trading_day(current):
            days.append(current)
        current -= timedelta(days=1)

        # Safety guard — don't loop forever
        if (end_date - current).days > n * 3:
            log.warning(f"Could not find {n} trading days — stopping search")
            break

    # Return oldest to newest
    return sorted(days)


def get_backfill_start_date(years: int = 5) -> date:
    """
    Returns the start date for a historical backfill.
    Goes back `years` from today.

    Parameters
    ----------
    years : int
        Number of years to go back. Default 5.

    Returns
    -------
    date
        Approximate start date for backfill.
    """
    today = date.today()
    # Approximate — 365 days per year
    start = date(today.year - years, today.month, today.day)
    log.info(f"Backfill start date ({years} years): {start}")
    return start


def get_week_start(d: date) -> date:
    """
    Returns the Monday of the week containing date d.
    Used when building weekly_prices rows.

    Parameters
    ----------
    d : date
        Any date.

    Returns
    -------
    date
        The Monday of that week.

    Example
    -------
    get_week_start(date(2025, 6, 11))  # Wednesday
    # Returns: date(2025, 6, 9)        # Monday
    """
    return d - timedelta(days=d.weekday())