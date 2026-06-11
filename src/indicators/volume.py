# ================================================================
# src/indicators/volume.py
# ----------------------------------------------------------------
# Volume-based indicator calculations.
#
# Detects 9 scenarios:
#   1. Volume Awakening        — sudden spike vs own dormant history
#   2. Structural Volume Rise  — sustained multi-period expansion
#   3. Volume Contraction      — drying up (base building signal)
#   4. Delivery Confirmation   — high delivery on spike day
#   5. Delivery Acceleration   — delivery % rising over time
#   6. Volume Climax (Up)      — extended move, too late to enter
#   7. Volume Expansion        — weekly volume > quarterly baseline
#   8. Delivery Progression    — rising delivery on consecutive up days
#   9. Absorption Event        — high volume + flat price + high delivery
#
# Input:  DataFrame of daily prices for ONE symbol
# Output: dict of computed values + scenario flags
# ================================================================

import pandas as pd
import numpy as np
from src.utils.logger import get_logger

log = get_logger(__name__)


# ================================================================
# HELPER FUNCTIONS
# ================================================================

def _is_green_candle(row: pd.Series) -> bool:
    """Returns True if close > open (green/bullish candle)."""
    return float(row["close"]) > float(row["open"])


def _candle_body_pct(row: pd.Series) -> float:
    """
    Returns candle body size as % of total range.
    Small body (< 30%) = indecision / absorption.
    Large body (> 70%) = strong directional move.
    """
    total_range = float(row["high"]) - float(row["low"])
    if total_range <= 0:
        return 0.0
    body = abs(float(row["close"]) - float(row["open"]))
    return round(body / total_range * 100, 2)


# ================================================================
# MAIN FUNCTION
# ================================================================

def calculate_volume_indicators(df: pd.DataFrame) -> dict:
    """
    Calculates all volume indicators for a single symbol.

    Parameters
    ----------
    df : pd.DataFrame
        Daily prices for one symbol, sorted oldest to newest.
        Must have columns:
            date, open, high, low, close, volume, delivery_pct
        Minimum 22 rows required.

    Returns
    -------
    dict
        All computed values and scenario flags.
        Returns empty dict if insufficient data.
    """
    if df is None or len(df) < 22:
        log.debug("Insufficient data for volume indicators")
        return {}

    df = df.sort_values("date").reset_index(drop=True)

    # ── Today and history ──────────────────────────────────────
    today   = df.iloc[-1]
    history = df.iloc[:-1]          # exclude today for avg calculations

    vol_today   = float(today["volume"])
    close_today = float(today["close"])
    open_today  = float(today["open"])
    high_today  = float(today["high"])
    low_today   = float(today["low"])
    deliv_today = today.get("delivery_pct", np.nan)

    # ── Volume averages (prior periods, excluding today) ───────
    vol_5d   = history.tail(5)["volume"].mean()     # ~1 week
    vol_22d  = history.tail(22)["volume"].mean()    # ~1 month
    vol_65d  = history.tail(65)["volume"].mean()    # ~1 quarter
    vol_180d = history.tail(180)["volume"].mean()   # ~6 months

    # Weekly volume sum (last 5 days including today)
    vol_week_sum = df.tail(5)["volume"].sum()

    # Average weekly volume over last 13 weeks
    # Each week = 5 trading days → 13 weeks = 65 days
    weekly_vol_13w_avg = vol_65d * 5   # avg weekly sum

    # ── Volume ratios ──────────────────────────────────────────
    day_vol_ratio   = round(vol_today / vol_5d,  2) if vol_5d  > 0 else 0
    week_vol_ratio  = round(vol_5d    / vol_22d, 2) if vol_22d > 0 else 0
    month_vol_ratio = round(vol_22d   / vol_65d, 2) if vol_65d > 0 else 0

    # ── Volume contraction ─────────────────────────────────────
    if len(history) >= 10:
        last_10     = history.tail(10)["volume"]
        first_half  = last_10.iloc[:5].mean()
        second_half = last_10.iloc[5:].mean()
        is_contracting = bool(second_half < first_half * 0.85)
    else:
        is_contracting = False

    # ── Delivery averages ──────────────────────────────────────
    deliv_5d_avg  = history.tail(5)["delivery_pct"].mean()
    deliv_22d_avg = history.tail(22)["delivery_pct"].mean()
    deliv_ratio   = round(
        float(deliv_today) / float(deliv_5d_avg), 2
    ) if pd.notna(deliv_today) and deliv_5d_avg > 0 else 0

    deliv_accelerating = bool(
        pd.notna(deliv_today) and
        float(deliv_5d_avg) > float(deliv_22d_avg) * 1.1
    )

    # ── Price move context ─────────────────────────────────────
    prev_close    = float(df.iloc[-2]["close"]) if len(df) >= 2 else close_today
    price_chg_pct = round(
        (close_today - prev_close) / prev_close * 100, 2
    ) if prev_close > 0 else 0

    # ── Candle body ────────────────────────────────────────────
    today_body_pct = _candle_body_pct(today)

    # ================================================================
    # SCENARIO 1 — Volume Awakening
    # ----------------------------------------------------------------
    # Stock was dormant for months — volume just woke up.
    # Key: the dormancy check separates this from a stock
    # that is always active and just had a normal big day.
    # ================================================================
    scenario_awakening = bool(
        day_vol_ratio  >= 1.5 and          # today bigger than recent week
        week_vol_ratio >= 1.2 and          # week already elevated vs month
        vol_180d > 0   and
        vol_22d <= vol_180d * 1.3          # was quiet for 6 months (dormancy)
    )

    # ================================================================
    # SCENARIO 2 — Structural Volume Rise
    # ----------------------------------------------------------------
    # Not just today — the entire recent period is more active.
    # Multi-period confirmation = sustained institutional interest.
    # ================================================================
    scenario_structural_rise = bool(
        week_vol_ratio  >= 1.3 and         # week > month
        month_vol_ratio >= 1.1             # month > quarter
    )

    # ================================================================
    # SCENARIO 3 — Volume Contraction
    # ----------------------------------------------------------------
    # Volume drying up = supply exhausted = base building.
    # Most powerful when price is also flat during contraction.
    # ================================================================
    scenario_contraction = bool(
        is_contracting and
        day_vol_ratio <= 1.2               # today also not spiking
    )

    # ================================================================
    # SCENARIO 4 — Delivery Confirmation
    # ----------------------------------------------------------------
    # High volume + high delivery = real institutional buying.
    # Low delivery on high volume = intraday operators, ignore.
    # ================================================================
    scenario_delivery_confirm = bool(
        day_vol_ratio >= 1.5 and
        pd.notna(deliv_today) and
        float(deliv_today) >= 45.0
    )

    # ================================================================
    # SCENARIO 5 — Delivery Acceleration
    # ----------------------------------------------------------------
    # Delivery % has been rising over multiple days.
    # Smart money quietly increasing position over time.
    # ================================================================
    scenario_delivery_accel = bool(
        deliv_accelerating and
        pd.notna(deliv_today) and
        float(deliv_today) >= 40.0
    )

    # ================================================================
    # SCENARIO 6 — Volume Climax (Up Move — Too Late)
    # ----------------------------------------------------------------
    # Massive volume + large UP price move = move already happened.
    # Note: DOWN climax (selling exhaustion) is actually bullish —
    # we do NOT flag that as too late.
    # ================================================================
    scenario_climax = bool(
        day_vol_ratio  >= 3.0 and
        price_chg_pct  >= 3.0 and          # significant UP move
        close_today    > prev_close         # confirmed up direction
    )

    # ================================================================
    # SCENARIO 7 — Volume Expansion
    # ----------------------------------------------------------------
    # This week's total volume has crossed above the 13-week
    # average weekly volume.
    # Refinement: at least 3 of last 5 days were above their
    # own 22-day average — confirms breadth of the expansion.
    # ================================================================

    # Count how many of last 5 days had above-average volume
    last_5_days = df.tail(5)
    above_avg_days = sum(
        1 for _, row in last_5_days.iterrows()
        if float(row["volume"]) > vol_22d
    )

    scenario_volume_expansion = bool(
        vol_week_sum > weekly_vol_13w_avg and    # week sum > 13W avg week
        above_avg_days >= 3                       # at least 3 of 5 days active
    )

    # ================================================================
    # SCENARIO 8 — Delivery Progression
    # ----------------------------------------------------------------
    # Rising delivery % on consecutive green (up-close) days.
    # Tight price range is mandatory — if price is running 3%+
    # each day it's momentum, not accumulation.
    #
    # Conditions:
    #   - Last 3 days all closed green (close > open)
    #   - Each day's delivery % > previous day's delivery %
    #   - Each day's delivery % > 45%
    #   - Total 3-day price range < 3% (tight accumulation)
    # ================================================================
    scenario_delivery_progression = False

    if len(df) >= 4:
        last_3 = df.tail(3).reset_index(drop=True)
        d1, d2, d3 = last_3.iloc[0], last_3.iloc[1], last_3.iloc[2]

        deliv_d1 = d1.get("delivery_pct", np.nan)
        deliv_d2 = d2.get("delivery_pct", np.nan)
        deliv_d3 = d3.get("delivery_pct", np.nan)

        # 3-day price range
        three_day_high = last_3["high"].max()
        three_day_low  = last_3["low"].min()
        three_day_range_pct = (
            (three_day_high - three_day_low) / three_day_low * 100
        ) if three_day_low > 0 else 999

        all_green = (
            _is_green_candle(d1) and
            _is_green_candle(d2) and
            _is_green_candle(d3)
        )

        delivery_rising = (
            pd.notna(deliv_d1) and
            pd.notna(deliv_d2) and
            pd.notna(deliv_d3) and
            float(deliv_d3) > float(deliv_d2) > float(deliv_d1)
        )

        all_delivery_high = (
            pd.notna(deliv_d1) and
            pd.notna(deliv_d2) and
            pd.notna(deliv_d3) and
            float(deliv_d1) >= 45.0 and
            float(deliv_d2) >= 45.0 and
            float(deliv_d3) >= 45.0
        )

        scenario_delivery_progression = bool(
            all_green and
            delivery_rising and
            all_delivery_high and
            three_day_range_pct < 3.0      # price tight — not chasing
        )

    # ================================================================
    # SCENARIO 9 — Absorption Event
    # ----------------------------------------------------------------
    # High volume + price went nowhere + high delivery.
    # Classic Wyckoff Phase B — smart money absorbing all supply.
    #
    # The candle body check is critical:
    #   Small body = fierce battle, buyers absorbed sellers
    #   Large body = directional move, not absorption
    #
    # Conditions:
    #   - Volume >= 1.5x monthly average (heavy supply present)
    #   - Price close within 0.5% of open (went nowhere)
    #   - Delivery % >= 55% (real buying, not intraday flipping)
    #   - Candle body < 35% of total range (absorption candle shape)
    #   - Occurs at or near a support level (price near recent lows)
    # ================================================================

    # Check if price is near recent support (within 5% of 22-day low)
    recent_low      = history.tail(22)["low"].min() if len(history) >= 22 else low_today
    near_support    = bool(close_today <= recent_low * 1.05)

    # Price barely moved (close within 0.5% of open)
    open_close_diff = abs(close_today - open_today) / open_today * 100 \
        if open_today > 0 else 999

    scenario_absorption = bool(
        day_vol_ratio  >= 1.5 and           # heavy volume
        open_close_diff <= 0.5 and          # price went nowhere
        pd.notna(deliv_today) and
        float(deliv_today) >= 55.0 and      # real buying
        today_body_pct  <= 35.0             # small body candle
    )

    # ================================================================
    # RETURN ALL VALUES
    # ================================================================

    return {
        # ── Raw values ─────────────────────────────────────────
        "vol_today"           : int(vol_today),
        "vol_5d_avg"          : round(vol_5d,   0),
        "vol_22d_avg"         : round(vol_22d,  0),
        "vol_65d_avg"         : round(vol_65d,  0),
        "vol_180d_avg"        : round(vol_180d, 0),
        "vol_week_sum"        : int(vol_week_sum),

        # ── Ratios ─────────────────────────────────────────────
        "day_vol_ratio"       : day_vol_ratio,
        "week_vol_ratio"      : week_vol_ratio,
        "month_vol_ratio"     : month_vol_ratio,

        # ── Delivery ───────────────────────────────────────────
        "delivery_pct_today"  : round(float(deliv_today), 2) \
                                if pd.notna(deliv_today) else None,
        "deliv_5d_avg"        : round(float(deliv_5d_avg),  2) \
                                if pd.notna(deliv_5d_avg)  else None,
        "deliv_22d_avg"       : round(float(deliv_22d_avg), 2) \
                                if pd.notna(deliv_22d_avg) else None,
        "deliv_ratio"         : deliv_ratio,
        "deliv_accelerating"  : deliv_accelerating,

        # ── Candle context ─────────────────────────────────────
        "price_chg_pct"       : price_chg_pct,
        "today_body_pct"      : today_body_pct,
        "near_support"        : near_support,

        # ── Scenario flags ─────────────────────────────────────
        "scenario_awakening"            : scenario_awakening,
        "scenario_structural_rise"      : scenario_structural_rise,
        "scenario_contraction"          : scenario_contraction,
        "scenario_delivery_confirm"     : scenario_delivery_confirm,
        "scenario_delivery_accel"       : scenario_delivery_accel,
        "scenario_climax"               : scenario_climax,
        "scenario_volume_expansion"     : scenario_volume_expansion,
        "scenario_delivery_progression" : scenario_delivery_progression,
        "scenario_absorption"           : scenario_absorption,
    }