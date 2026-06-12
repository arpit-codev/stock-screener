# ================================================================
# src/indicators/price.py
# ----------------------------------------------------------------
# Price-based indicator calculations.
#
# Detects 6 scenarios:
#   1. Consolidation    — tight range over last 4 weeks
#   2. Higher Lows      — each pullback higher than previous
#   3. Near 20 EMA      — optimal pullback entry zone
#   4. Below 200 EMA    — potential explosive move setup
#   5. Near 52W High    — stock showing relative strength
#   6. Multi Year Base  — long dormancy, stored energy
#
# Input:  DataFrame of daily prices for ONE symbol
# Output: dict of computed values + scenario flags
# ================================================================

import pandas as pd
import numpy as np
from src.utils.logger import get_logger

log = get_logger(__name__)


# ================================================================
# MAIN FUNCTION
# ================================================================

def calculate_price_indicators(df: pd.DataFrame) -> dict:
    """
    Calculates all price indicators for a single symbol.

    Parameters
    ----------
    df : pd.DataFrame
        Daily prices for one symbol, sorted oldest to newest.
        Must have: date, open, high, low, close
        Minimum 22 rows required.

    Returns
    -------
    dict
        All computed values and scenario flags.
        Returns empty dict if insufficient data.
    """
    if df is None or len(df) < 22:
        log.debug("Insufficient data for price indicators")
        return {}

    df = df.sort_values("date").reset_index(drop=True)

    today   = df.iloc[-1]
    history = df.iloc[:-1]

    close_today = float(today["close"])
    high_today  = float(today["high"])
    low_today   = float(today["low"])

    if close_today <= 0:
        return {}

    # ── Price change calculations ──────────────────────────────
    def pct_change(days_back):
        if len(df) < days_back + 1:
            return None
        past_close = float(df.iloc[-(days_back + 1)]["close"])
        if past_close <= 0:
            return None
        return round((close_today - past_close) / past_close * 100, 2)

    chg_1d  = pct_change(1)
    chg_1w  = pct_change(5)
    chg_1m  = pct_change(22)
    chg_3m  = pct_change(65)
    chg_6m  = pct_change(130)
    chg_1y  = pct_change(252)
    chg_3y  = pct_change(756)

    # ── 52 week high and low ───────────────────────────────────
    lookback_52w = min(252, len(df))
    high_52w     = float(df.tail(lookback_52w)["high"].max())
    low_52w      = float(df.tail(lookback_52w)["low"].min())

    pct_from_52w_high = round(
        (close_today - high_52w) / high_52w * 100, 2
    ) if high_52w > 0 else None

    pct_from_52w_low = round(
        (close_today - low_52w) / low_52w * 100, 2
    ) if low_52w > 0 else None

    # ── EMA calculations ───────────────────────────────────────
    ema_20  = float(df["close"].ewm(span=20,  adjust=False).mean().iloc[-1])
    ema_50  = float(df["close"].ewm(span=50,  adjust=False).mean().iloc[-1])
    ema_200 = float(df["close"].ewm(span=200, adjust=False).mean().iloc[-1]) \
              if len(df) >= 200 else None

    pct_from_ema20  = round((close_today - ema_20)  / ema_20  * 100, 2)
    pct_from_ema50  = round((close_today - ema_50)  / ema_50  * 100, 2)
    pct_from_ema200 = round((close_today - ema_200) / ema_200 * 100, 2) \
                      if ema_200 else None

    # ── Range width — consolidation check ─────────────────────
    # How wide is the price range over last 4 weeks?
    last_22 = df.tail(22)
    range_high_4w = float(last_22["high"].max())
    range_low_4w  = float(last_22["low"].min())
    range_pct_4w  = round(
        (range_high_4w - range_low_4w) / range_low_4w * 100, 2
    ) if range_low_4w > 0 else None

    # 8 week range
    last_40       = df.tail(40)
    range_high_8w = float(last_40["high"].max())
    range_low_8w  = float(last_40["low"].min())
    range_pct_8w  = round(
        (range_high_8w - range_low_8w) / range_low_8w * 100, 2
    ) if range_low_8w > 0 else None

    # ── Higher lows detection ──────────────────────────────────
    # Check last 3 significant lows are ascending
    # Use rolling 5-day minimum to find local lows

    if len(df) >= 60:
        # Find actual swing lows over last 90 days
        # A swing low = lower than surrounding 3 days on each side
        swing_lows = []
        lookback = min(90, len(df) - 6)
        start_idx = len(df) - lookback

        for idx in range(start_idx + 3, len(df) - 3):
            window_low = float(df["low"].iloc[idx])
            left_min = float(df["low"].iloc[idx - 3:idx].min())
            right_min = float(df["low"].iloc[idx + 1:idx + 4].min())
            if window_low < left_min and window_low < right_min:
                swing_lows.append((idx, window_low))

        # Find best ascending sequence among last 4 swing lows
        # Not just last 3 — allows skipping one bad point
        higher_lows = False
        if len(swing_lows) >= 2:
            # Check last 2 swing lows ascending
            last2 = swing_lows[-2:]
            if last2[-1][1] > last2[-2][1]:
                higher_lows = True

        if higher_lows and len(swing_lows) >= 3:
            # Confirm with 3rd last as well
            last3 = swing_lows[-3:]
            if last3[-1][1] > last3[-2][1]:
                higher_lows = True
            else:
                # Middle low broke sequence — check if overall trend up
                # First and last should still be ascending
                if last3[-1][1] > last3[-3][1]:
                    higher_lows = True
                else:
                    higher_lows = False
    else:
        higher_lows = False

    # ── Today's candle size ────────────────────────────────────
    candle_range     = high_today - low_today
    candle_body      = abs(close_today - float(today["open"]))
    candle_body_pct  = round(
        candle_body / candle_range * 100, 2
    ) if candle_range > 0 else 0

    # ================================================================
    # SCENARIO FLAGS
    # ================================================================

    # SCENARIO 1 — Consolidation
    # Price in tight range over last 4 weeks
    # Range < 8% = coiling, potential breakout building
    scenario_consolidating = bool(
        range_pct_4w is not None and
        range_pct_4w <= 8.0
    )

    # SCENARIO 2 — Higher Lows Structure
    # Each pullback low is higher than the previous
    # Buyers stepping in at higher prices = accumulation
    scenario_higher_lows = bool(higher_lows)

    # SCENARIO 3 — Price vs 20 EMA
    # near_20ema = within ±3% of 20 EMA (pullback zone)
    # above_20ema = price is above 20 EMA (uptrend intact)
    scenario_near_20ema = bool(
        abs(pct_from_ema20) <= 3.0
    )
    scenario_above_20ema = bool(
        pct_from_ema20 > 0
    )

    # SCENARIO 4 — Below 200 EMA But Basing
    # Price below long term average
    # But not in freefall — range is tight
    # These can have explosive breakout moves
    scenario_below_200ema = bool(
        ema_200 is not None and
        close_today < ema_200 and
        range_pct_4w is not None and
        range_pct_4w <= 12.0    # must be basing, not crashing
    )

    # SCENARIO 5 — Near 52 Week High
    # Within 10% of yearly high
    # Stock showing relative strength — not broken
    scenario_near_52w_high = bool(
        pct_from_52w_high is not None and
        pct_from_52w_high >= -10.0    # within 10% of high
    )

    # SCENARIO 6 — Multi Year Base
    # Price hasn't moved much in 3 years
    # Long dormancy = stored energy waiting to release
    scenario_multi_year_base = bool(
        chg_3y is not None and
        -30.0 <= chg_3y <= 40.0 and   # flat over 3 years
        chg_1y is not None and
        -15.0 <= chg_1y <= 20.0        # also flat over 1 year
    )

    # ================================================================
    # RETURN ALL VALUES
    # ================================================================

    return {
        # ── Price changes ───────────────────────────────────────
        "close"              : round(close_today, 2),
        "chg_1d"             : chg_1d,
        "chg_1w"             : chg_1w,
        "chg_1m"             : chg_1m,
        "chg_3m"             : chg_3m,
        "chg_6m"             : chg_6m,
        "chg_1y"             : chg_1y,
        "chg_3y"             : chg_3y,

        # ── 52 week levels ──────────────────────────────────────
        "high_52w"           : round(high_52w, 2),
        "low_52w"            : round(low_52w,  2),
        "pct_from_52w_high"  : pct_from_52w_high,
        "pct_from_52w_low"   : pct_from_52w_low,

        # ── EMA levels ──────────────────────────────────────────
        "ema_20"             : round(ema_20, 2),
        "ema_50"             : round(ema_50, 2),
        "ema_200"            : round(ema_200, 2) if ema_200 else None,
        "pct_from_ema20"     : pct_from_ema20,
        "pct_from_ema50"     : pct_from_ema50,
        "pct_from_ema200"    : pct_from_ema200,

        # ── Range ───────────────────────────────────────────────
        "range_pct_4w"       : range_pct_4w,
        "range_pct_8w"       : range_pct_8w,
        "range_high_4w"      : round(range_high_4w, 2),
        "range_low_4w"       : round(range_low_4w,  2),

        # ── Structure ───────────────────────────────────────────
        "higher_lows"        : higher_lows,
        "candle_body_pct"    : candle_body_pct,

        # ── Scenario flags ──────────────────────────────────────
        "scenario_consolidating"    : scenario_consolidating,
        "scenario_higher_lows"      : scenario_higher_lows,
        "scenario_near_20ema"       : scenario_near_20ema,
        "scenario_above_20ema"      : scenario_above_20ema,
        "scenario_below_200ema"     : scenario_below_200ema,
        "scenario_near_52w_high"    : scenario_near_52w_high,
        "scenario_multi_year_base"  : scenario_multi_year_base,
    }