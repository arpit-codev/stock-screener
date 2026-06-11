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
#                                checked over last 15 days
#
# Key fix: Uses MEDIAN not mean for volume baselines
#          A single spike day should not inflate averages
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

    # ── Volume averages — MEDIAN to avoid spike distortion ─────
    # Single spike days (e.g. 41M vs normal 2M) inflate mean badly
    # Median gives the true "typical" volume level
    vol_5d   = history.tail(5)["volume"].median()
    vol_22d  = history.tail(22)["volume"].median()
    vol_65d  = history.tail(65)["volume"].median()
    vol_180d = history.tail(180)["volume"].median()

    # Weekly volume sum (last 5 days including today)
    vol_week_sum = df.tail(5)["volume"].sum()

    # Average weekly volume over last 13 weeks
    weekly_vol_13w_avg = vol_65d * 5

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

    # ── Check if price near support ────────────────────────────
    recent_low   = history.tail(22)["low"].min() \
                   if len(history) >= 22 else low_today
    near_support = bool(close_today <= recent_low * 1.05)

    # ── Open close diff today ──────────────────────────────────
    open_close_diff = abs(close_today - open_today) / open_today * 100 \
                      if open_today > 0 else 999

    # ================================================================
    # SCENARIO 1 — Volume Awakening
    # ================================================================
    scenario_awakening = bool(
        day_vol_ratio  >= 1.5 and
        week_vol_ratio >= 1.2 and
        vol_180d > 0   and
        vol_22d <= vol_180d * 1.3
    )

    # ================================================================
    # SCENARIO 2 — Structural Volume Rise
    # ================================================================
    scenario_structural_rise = bool(
        week_vol_ratio  >= 1.3 and
        month_vol_ratio >= 1.1
    )

    # ================================================================
    # SCENARIO 3 — Volume Contraction
    # ================================================================
    scenario_contraction = bool(
        is_contracting and
        day_vol_ratio <= 1.2
    )

    # ================================================================
    # SCENARIO 4 — Delivery Confirmation
    # ================================================================
    scenario_delivery_confirm = bool(
        day_vol_ratio >= 1.5 and
        pd.notna(deliv_today) and
        float(deliv_today) >= 45.0
    )

    # ================================================================
    # SCENARIO 5 — Delivery Acceleration
    # ================================================================
    scenario_delivery_accel = bool(
        deliv_accelerating and
        pd.notna(deliv_today) and
        float(deliv_today) >= 40.0
    )

    # ================================================================
    # SCENARIO 6 — Volume Climax
    # ================================================================
    scenario_climax = bool(
        day_vol_ratio  >= 3.0 and
        price_chg_pct  >= 3.0 and
        close_today    > prev_close
    )

    # ================================================================
    # SCENARIO 7 — Volume Expansion
    # ================================================================
    last_5_days    = df.tail(5)
    above_avg_days = sum(
        1 for _, row in last_5_days.iterrows()
        if float(row["volume"]) > vol_22d
    )

    scenario_volume_expansion = bool(
        vol_week_sum > weekly_vol_13w_avg and
        above_avg_days >= 3
    )

    # ================================================================
    # SCENARIO 8 — Delivery Progression
    # ================================================================
    scenario_delivery_progression = False

    if len(df) >= 4:
        last_3 = df.tail(3).reset_index(drop=True)
        d1, d2, d3 = last_3.iloc[0], last_3.iloc[1], last_3.iloc[2]

        deliv_d1 = d1.get("delivery_pct", np.nan)
        deliv_d2 = d2.get("delivery_pct", np.nan)
        deliv_d3 = d3.get("delivery_pct", np.nan)

        three_day_high      = last_3["high"].max()
        three_day_low       = last_3["low"].min()
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
            three_day_range_pct < 3.0
        )

    # ================================================================
    # SCENARIO 9 — Absorption Event (last 15 days)
    # ----------------------------------------------------------------
    # Check if ANY day in last 15 days was an absorption candle.
    # High volume + flat price + high delivery = smart money
    # absorbing all available supply at a key level.
    #
    # Uses median volume BEFORE the window as baseline
    # so a spike day doesn't inflate its own comparison.
    # ================================================================
    scenario_absorption = False
    absorption_date = None
    absorption_tier = None
    absorption_vol_ratio = None
    absorption_delivery_pct = None

    # Baseline = median of all data before the 15-day window
    # This ensures spike days in window don't inflate baseline
    vol_baseline = df.iloc[:-16]["volume"].median() \
                   if len(df) > 16 else vol_22d

    check_window = df.tail(16)   # last 15 days + today

    for _, abs_row in check_window.iterrows():
        abs_vol   = float(abs_row["volume"])
        abs_open  = float(abs_row["open"])
        abs_close = float(abs_row["close"])
        abs_high  = float(abs_row["high"])
        abs_low   = float(abs_row["low"])
        abs_deliv = abs_row.get("delivery_pct", np.nan)

        abs_vol_ratio  = abs_vol / vol_baseline \
                         if vol_baseline > 0 else 0
        abs_open_close = abs(abs_close - abs_open) / abs_open * 100 \
                         if abs_open > 0 else 999
        abs_range      = abs_high - abs_low
        abs_body       = abs(abs_close - abs_open)
        abs_body_pct   = abs_body / abs_range * 100 \
                         if abs_range > 0 else 0

        # Tier S — SUPER absorption
        # Massive unusual volume + very high delivery + flat price
        # This is the strongest possible accumulation signal
        tier_s = bool(
            abs_vol_ratio >= 5.0 and  # 5x+ baseline volume
            abs_open_close <= 1.0 and  # price barely moved
            pd.notna(abs_deliv) and
            float(abs_deliv) >= 65.0 and  # very high delivery
            abs_body_pct <= 40.0  # small body
        )

        # Tier A — strong absorption
        tier_a = bool(
            abs_vol_ratio >= 2.0 and
            abs_open_close <= 0.5 and
            pd.notna(abs_deliv) and
            float(abs_deliv) >= 60.0 and
            abs_body_pct <= 35.0
        )

        # Tier B — moderate absorption
        tier_b = bool(
            abs_vol_ratio >= 1.5 and
            abs_open_close <= 0.8 and
            pd.notna(abs_deliv) and
            float(abs_deliv) >= 55.0 and
            abs_body_pct <= 40.0
        )

        if tier_s or tier_a or tier_b:
            scenario_absorption = True
            absorption_date = abs_row["date"]
            absorption_tier = "S" if tier_s else \
                "A" if tier_a else "B"
            absorption_vol_ratio = round(abs_vol_ratio, 1)
            absorption_delivery_pct = round(float(abs_deliv), 1) \
                if pd.notna(abs_deliv) else None
            break

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
        "scenario_absorption": scenario_absorption,
        "absorption_date": str(absorption_date) \
            if absorption_date else None,
        "absorption_tier": absorption_tier \
            if scenario_absorption else None,
        "absorption_vol_ratio": absorption_vol_ratio \
            if scenario_absorption else None,
        "absorption_delivery_pct": absorption_delivery_pct \
            if scenario_absorption else None,
    }