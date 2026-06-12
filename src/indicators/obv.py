# ================================================================
# src/indicators/obv.py
# ----------------------------------------------------------------
# OBV (On Balance Volume) — Complete Analysis
#
# Architecture:
#   TIER 1 — Must pass filters (liquidity + not broken)
#   TIER 2 — Conviction scoring (13 scenarios, max 100 pts)
#   TIER 3 — Ranked output with tier label
#
# 13 Scenarios:
#   GROUP A — OBV Divergence (max 35 pts)
#     S1:  OBV new 50D high, price not             (12 pts)
#     S4:  OBV 100D breakout + consolidating       (8 pts)
#     S6:  Price flat 20D + OBV rising             (15 pts) ← CORE
#
#   GROUP B — Delivery Confirmation (max 30 pts)
#     S3:  OBV rising + delivery >= 60%            (15 pts)
#     S7:  Delivery spike + OBV spike              (10 pts)
#     S8:  4+ high delivery days in 10D            (5 pts)
#
#   GROUP C — OBV Momentum (max 20 pts)
#     S2:  OBV rising sustained 20 sessions        (8 pts)
#     S5:  OBV slope accelerating                  (7 pts)
#     S9:  Tight range + OBV rising                (5 pts)
#
#   GROUP D — Special Signals (max 15 pts)
#     S10: Shakeout — price down, OBV flat         (10 pts)
#     S11: Weekly OBV divergence                   (5 pts)
#     S12: Both daily + weekly divergence          (bonus 3 pts)
#     S13: OBV higher highs                        (bonus 2 pts)
#
# Conviction Tiers:
#   HIGH   >= 70 pts  → 5-15 stocks per day
#   MEDIUM 50-69 pts  → 20-40 stocks per day
#   LOW    30-49 pts  → monitor only
#   SKIP   < 30 pts
# ================================================================

import pandas as pd
import numpy as np
from src.utils.logger import get_logger

log = get_logger(__name__)


# ================================================================
# CALCULATE OBV
# ================================================================

def _calculate_obv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates raw OBV line.
    Up day   → add volume
    Down day → subtract volume
    Flat day → no change
    """
    obv    = [0]
    closes = df["close"].values
    vols   = df["volume"].values

    for i in range(1, len(df)):
        if closes[i] > closes[i - 1]:
            obv.append(obv[-1] + int(vols[i]))
        elif closes[i] < closes[i - 1]:
            obv.append(obv[-1] - int(vols[i]))
        else:
            obv.append(obv[-1])

    df = df.copy()
    df["obv"] = obv
    return df


def _resample_to_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """Resamples daily OHLCV to weekly."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")

    weekly = df.resample("W-FRI").agg({
        "open"   : "first",
        "high"   : "max",
        "low"    : "min",
        "close"  : "last",
        "volume" : "sum",
    }).dropna()

    if "delivery_pct" in df.columns:
        weekly_deliv = df["delivery_pct"].resample("W-FRI").mean()
        weekly["delivery_pct"] = weekly_deliv

    weekly = weekly.reset_index()
    weekly = weekly.rename(columns={"date": "week_date"})
    weekly["date"] = weekly["week_date"]

    return weekly


# ================================================================
# TIER 1 — MUST PASS FILTERS
# ================================================================

def check_tier1_filters(
    df: pd.DataFrame,
    min_avg_volume: int = 100000,
    min_price: float = 20.0
) -> dict:
    """
    Checks minimum quality filters.
    Stock must pass ALL to be scored.

    Falling knife logic:
      Stock near 52W low + OBV falling  → BLOCKED
      Stock near 52W low + OBV rising   → ALLOWED (accumulation)
      Stock above 52W low               → ALLOWED

    Returns dict with passed=True/False and reason if failed.
    """
    if df is None or len(df) < 30:
        return {"passed": False, "reason": "insufficient_data"}

    close_today = float(df["close"].iloc[-1])
    avg_vol     = float(df["volume"].tail(22).mean())

    # ── Liquidity ──────────────────────────────────────────────
    if avg_vol < min_avg_volume:
        return {
            "passed": False,
            "reason": f"low_volume_{int(avg_vol)}"
        }

    # ── Price floor ────────────────────────────────────────────
    if close_today < min_price:
        return {
            "passed": False,
            "reason": f"low_price_{close_today}"
        }

    # ── Falling knife check ────────────────────────────────────
    # Only block if BOTH price near 52W low AND OBV falling
    # If OBV rising near lows → accumulation setup → allow through
    low_52w      = float(df.tail(252)["low"].min()) \
                   if len(df) >= 252 else float(df["low"].min())
    near_52w_low = close_today < low_52w * 1.15

    if near_52w_low:
        df_temp    = _calculate_obv(df.copy())
        obv_now    = float(df_temp["obv"].iloc[-1])
        obv_20d    = float(df_temp["obv"].iloc[-21]) \
                     if len(df_temp) >= 21 else obv_now
        obv_falling = obv_now < obv_20d

        if obv_falling:
            return {
                "passed": False,
                "reason": "falling_knife_price_and_obv_down"
            }

    # ── OBV not collapsing ─────────────────────────────────────
    df_obv     = _calculate_obv(df.copy())
    obv_now    = float(df_obv["obv"].iloc[-1])
    obv_30d    = float(df_obv["obv"].iloc[-31]) \
                 if len(df_obv) >= 31 else obv_now
    avg_vol_30 = float(df["volume"].tail(30).mean())

    obv_chg_ratio = (obv_now - obv_30d) / avg_vol_30 \
                    if avg_vol_30 > 0 else 0

    if obv_chg_ratio < -10:
        return {
            "passed": False,
            "reason": "obv_collapsing"
        }

    # ── Delivery data check ────────────────────────────────────
    if "delivery_pct" in df.columns:
        valid_deliv = df["delivery_pct"].dropna()
        if len(valid_deliv) < 10:
            return {
                "passed": False,
                "reason": "insufficient_delivery_data"
            }

    return {"passed": True, "reason": None}


# ================================================================
# TIER 2 — SCORING — ALL 13 SCENARIOS
# ================================================================

def _score_scenarios(
    df_d: pd.DataFrame,
    df_w: pd.DataFrame
) -> dict:
    """
    Scores all 13 OBV scenarios.
    Returns scenario flags + individual scores + total.
    """
    scores   = {}
    flags    = {}
    avg_vol  = float(df_d["volume"].tail(22).mean())
    close_td = float(df_d["close"].iloc[-1])

    # ── Pre-calculations ───────────────────────────────────────

    obv_today    = float(df_d["obv"].iloc[-1])
    obv_5d_ago   = float(df_d["obv"].iloc[-6])  if len(df_d) >= 6  else obv_today
    obv_10d_ago  = float(df_d["obv"].iloc[-11]) if len(df_d) >= 11 else obv_today
    obv_20d_ago  = float(df_d["obv"].iloc[-21]) if len(df_d) >= 21 else obv_today
    obv_30d_ago  = float(df_d["obv"].iloc[-31]) if len(df_d) >= 31 else obv_today
    obv_50d_ago  = float(df_d["obv"].iloc[-51]) if len(df_d) >= 51 else obv_today
    obv_100d_ago = float(df_d["obv"].iloc[-101])if len(df_d) >= 101 else obv_today

    close_5d_ago  = float(df_d["close"].iloc[-6])  if len(df_d) >= 6  else close_td
    close_20d_ago = float(df_d["close"].iloc[-21]) if len(df_d) >= 21 else close_td
    close_50d_ago = float(df_d["close"].iloc[-51]) if len(df_d) >= 51 else close_td

    # 50-day high
    high_50d  = float(df_d["close"].tail(50).max())  if len(df_d) >= 50 else close_td
    obv_high_50d = float(df_d["obv"].tail(50).max()) if len(df_d) >= 50 else obv_today
    obv_high_100d= float(df_d["obv"].tail(100).max())if len(df_d) >= 100 else obv_today

    # Price changes
    price_chg_5d  = (close_td - close_5d_ago)  / close_5d_ago  * 100 \
                    if close_5d_ago  > 0 else 0
    price_chg_20d = (close_td - close_20d_ago) / close_20d_ago * 100 \
                    if close_20d_ago > 0 else 0

    # OBV change normalised by avg volume
    obv_chg_20d_norm = (obv_today - obv_20d_ago) / (avg_vol * 20) * 100 \
                       if avg_vol > 0 else 0

    # Delivery data
    has_delivery    = "delivery_pct" in df_d.columns
    deliv_today     = float(df_d["delivery_pct"].iloc[-1]) \
                      if has_delivery and pd.notna(df_d["delivery_pct"].iloc[-1]) \
                      else 0
    deliv_avg_22d   = float(df_d["delivery_pct"].tail(22).mean()) \
                      if has_delivery else 0
    deliv_avg_5d    = float(df_d["delivery_pct"].tail(5).mean()) \
                      if has_delivery else 0

    # Delivery volume (shares delivered today)
    has_deliv_qty   = "delivery_qty" in df_d.columns
    deliv_qty_today = float(df_d["delivery_qty"].iloc[-1]) \
                      if has_deliv_qty and pd.notna(df_d["delivery_qty"].iloc[-1]) \
                      else 0
    deliv_qty_avg   = float(df_d["delivery_qty"].tail(22).mean()) \
                      if has_deliv_qty else 0

    # 4-week range
    last_20       = df_d.tail(20)
    range_high_20 = float(last_20["high"].max())
    range_low_20  = float(last_20["low"].min())
    range_pct_20  = (range_high_20 - range_low_20) / range_low_20 * 100 \
                    if range_low_20 > 0 else 999

    # ── GROUP A — OBV Divergence (max 35 pts) ─────────────────

    # S6 — CORE: Price flat 20D + OBV rising (15 pts)
    s6 = bool(
        abs(price_chg_20d) <= 5.0 and
        obv_today > obv_20d_ago and
        obv_chg_20d_norm >= 10.0        # OBV up 10%+ of monthly volume
    )
    scores["s6_price_flat_obv_rising"] = 15 if s6 else 0
    flags["s6_price_flat_obv_rising"]  = s6

    # S1 — OBV new 50D high, price not (12 pts)
    s1 = bool(
        obv_today > obv_high_50d * 0.99 and   # OBV at/near 50D high
        close_td  < high_50d * 0.98  and        # price not at 50D high
        close_td  >= high_50d * 0.80            # but not broken either
    )
    scores["s1_obv_new_high_price_not"] = 12 if s1 else 0
    flags["s1_obv_new_high_price_not"]  = s1

    # S4 — OBV 100D breakout + price consolidating (8 pts)
    s4 = bool(
        len(df_d) >= 100 and
        obv_today > obv_high_100d * 0.99 and   # OBV at 100D high
        range_pct_20 <= 10.0                    # price still consolidating
    )
    scores["s4_obv_100d_breakout"] = 8 if s4 else 0
    flags["s4_obv_100d_breakout"]  = s4

    # ── GROUP B — Delivery Confirmation (max 30 pts) ──────────

    # S3 — OBV rising + high delivery (15 pts)
    obv_rising_20d = obv_today > obv_20d_ago
    s3_tier_a = bool(
        obv_rising_20d and
        deliv_today >= 60.0 and
        deliv_today > deliv_avg_22d
    )
    s3_tier_b = bool(
        obv_rising_20d and
        deliv_today >= 40.0 and
        deliv_today > deliv_avg_22d * 1.2
    )
    s3 = s3_tier_a or s3_tier_b
    s3_pts = 15 if s3_tier_a else (10 if s3_tier_b else 0)
    scores["s3_obv_delivery_confirm"] = s3_pts
    flags["s3_obv_delivery_confirm"]  = s3
    flags["s3_tier"]                  = "A" if s3_tier_a else ("B" if s3_tier_b else None)

    # S7 — Delivery spike + OBV spike (10 pts)
    # Check last 5 days for a delivery+OBV spike day
    s7 = False
    if has_deliv_qty and deliv_qty_avg > 0:
        last_5 = df_d.tail(5)
        for _, row in last_5.iterrows():
            dq  = float(row["delivery_qty"]) \
                  if pd.notna(row["delivery_qty"]) else 0
            dp  = float(row["delivery_pct"]) \
                  if has_delivery and pd.notna(row["delivery_pct"]) else 0
            vol = float(row["volume"])
            avg_v = avg_vol
            # Delivery volume > 2x avg AND volume spike AND OBV rising
            if (dq > deliv_qty_avg * 2.0 and
                    vol > avg_vol * 1.3 and
                    dp >= 50.0):
                s7 = True
                break
    scores["s7_delivery_obv_spike"] = 10 if s7 else 0
    flags["s7_delivery_obv_spike"]  = s7

    # S8 — 4+ high delivery days in last 10D + OBV rising (5 pts)
    s8 = False
    if has_delivery and len(df_d) >= 10:
        last_10      = df_d.tail(10)
        high_d_days  = 0
        for j in range(1, len(last_10)):
            dp  = float(last_10["delivery_pct"].iloc[j]) \
                  if pd.notna(last_10["delivery_pct"].iloc[j]) else 0
            c   = float(last_10["close"].iloc[j])
            pc  = float(last_10["close"].iloc[j-1])
            # High delivery on an up day
            if dp >= 50.0 and c >= pc:
                high_d_days += 1
        s8 = bool(high_d_days >= 4 and obv_rising_20d)
    scores["s8_multi_high_delivery"] = 5 if s8 else 0
    flags["s8_multi_high_delivery"]  = s8

    # ── GROUP C — OBV Momentum (max 20 pts) ───────────────────

    # S2 — OBV rising sustained 20 sessions (8 pts)
    # At least 12 of last 20 days were up-volume days
    # AND overall OBV direction positive
    s2 = False
    if len(df_d) >= 20:
        last_20_df   = df_d.tail(20)
        up_vol_days  = sum(
            1 for j in range(1, len(last_20_df))
            if float(last_20_df["close"].iloc[j]) >=
               float(last_20_df["close"].iloc[j-1])
        )
        s2 = bool(
            up_vol_days >= 12 and
            obv_today > obv_20d_ago and
            obv_today > obv_10d_ago     # still rising recently
        )
    scores["s2_obv_sustained_rise"] = 8 if s2 else 0
    flags["s2_obv_sustained_rise"]  = s2

    # S5 — OBV slope accelerating (7 pts)
    # Recent OBV change > 1.5x avg daily change
    s5 = False
    if len(df_d) >= 22:
        daily_changes = df_d["obv"].diff().tail(20)
        avg_daily_chg = float(daily_changes.mean())
        today_chg     = float(df_d["obv"].iloc[-1]) - \
                        float(df_d["obv"].iloc[-2])
        if avg_daily_chg > 0:
            s5 = bool(today_chg > avg_daily_chg * 1.5)
        elif avg_daily_chg <= 0 and today_chg > 0:
            s5 = True   # turned positive from negative
    scores["s5_obv_slope_accel"] = 7 if s5 else 0
    flags["s5_obv_slope_accel"]  = s5

    # S9 — Tight range + OBV rising + volume shrinking (5 pts)
    s9 = False
    if len(df_d) >= 15:
        vol_first_half  = float(df_d.tail(15).head(7)["volume"].mean())
        vol_second_half = float(df_d.tail(8)["volume"].mean())
        vol_contracting = vol_second_half < vol_first_half * 0.85
        s9 = bool(
            range_pct_20 <= 6.0 and    # tight range
            obv_rising_20d and          # OBV rising inside range
            vol_contracting             # volume drying up
        )
    scores["s9_tight_range_obv_rising"] = 5 if s9 else 0
    flags["s9_tight_range_obv_rising"]  = s9

    # ── GROUP D — Special Signals (max 15 pts) ────────────────

    # S10 — Shakeout: price down, OBV flat (10 pts)
    s10 = False
    if len(df_d) >= 5:
        obv_chg_5d_norm = abs(obv_today - obv_5d_ago) / (avg_vol * 5) * 100 \
                          if avg_vol > 0 else 0
        s10 = bool(
            price_chg_5d <= -3.0 and     # price fell 3%+
            obv_chg_5d_norm <= 5.0       # OBV barely moved (flat)
        )
    scores["s10_shakeout_detected"] = 10 if s10 else 0
    flags["s10_shakeout_detected"]  = s10

    # S11 — Weekly OBV divergence (5 pts)
    s11 = False
    if df_w is not None and len(df_w) >= 8:
        w_obv_now = float(df_w["obv"].iloc[-1])
        w_obv_8w  = float(df_w["obv"].iloc[-9]) \
                    if len(df_w) >= 9 else w_obv_now
        w_close_now = float(df_w["close"].iloc[-1])
        w_close_8w  = float(df_w["close"].iloc[-9]) \
                      if len(df_w) >= 9 else w_close_now
        w_price_chg = (w_close_now - w_close_8w) / w_close_8w * 100 \
                      if w_close_8w > 0 else 0
        s11 = bool(
            w_obv_now > w_obv_8w and
            abs(w_price_chg) <= 8.0
        )
    scores["s11_weekly_divergence"] = 5 if s11 else 0
    flags["s11_weekly_divergence"]  = s11

    # S12 — Both daily + weekly divergence (bonus 3 pts)
    s12 = bool(s6 and s11)
    scores["s12_both_timeframes"] = 3 if s12 else 0
    flags["s12_both_timeframes"]  = s12

    # S13 — OBV higher highs (bonus 2 pts)
    s13 = False
    if len(df_d) >= 30:
        rolling_max = df_d["obv"].rolling(5).max()
        p1 = float(rolling_max.iloc[-5])
        p2 = float(rolling_max.iloc[-15])
        p3 = float(rolling_max.iloc[-25])
        s13 = bool(p1 > p2 > p3)
    scores["s13_obv_higher_highs"] = 2 if s13 else 0
    flags["s13_obv_higher_highs"]  = s13

    # ── S14 — Pullback Divergence ─────────────────────────────
    # Price pulled back from recent high
    # BUT OBV did not fall — smart money held positions
    # Uptrend context checked via 4 conditions
    s14 = False
    pullback_detected = False
    pullback_depth_pct = 0.0
    obv_held = False
    uptrend_conditions = {"A": False, "B": False,
                          "C": False, "D": False}
    uptrend_score = 0

    if len(df_d) >= 20:
        # Find recent high in last 20 days
        recent_high_idx = df_d["close"].tail(20).idxmax()
        recent_high_price = float(df_d.loc[recent_high_idx, "close"])
        recent_high_obv = float(df_d.loc[recent_high_idx, "obv"])

        pullback_depth = (recent_high_price - close_td) \
                         / recent_high_price * 100
        pullback_detected = pullback_depth >= 5.0

        if pullback_detected:
            pullback_depth_pct = round(pullback_depth, 2)

            # Check if OBV held during pullback
            obv_chg_during_pull = obv_today - recent_high_obv
            obv_chg_norm = obv_chg_during_pull / avg_vol \
                if avg_vol > 0 else 0
            # OBV held = didn't fall more than 5x avg daily vol
            obv_held = obv_chg_norm >= -5.0

            # Uptrend context — check all 4 conditions
            ema_20_val = float(
                df_d["close"].ewm(span=20, adjust=False)
                .mean().iloc[-1]
            )
            ema_50_val = float(
                df_d["close"].ewm(span=50, adjust=False)
                .mean().iloc[-1]
            ) if len(df_d) >= 50 else None

            avg_50d = float(df_d["close"].tail(50).mean()) \
                if len(df_d) >= 50 else None

            # Condition A — price above 20 EMA
            uptrend_conditions["A"] = close_td > ema_20_val

            # Condition B — price above 50 EMA
            uptrend_conditions["B"] = bool(
                ema_50_val and close_td > ema_50_val
            )

            # Condition C — made higher high in last 60 days
            if len(df_d) >= 60:
                high_30_60d = float(
                    df_d["close"].iloc[-60:-30].max()
                )
                high_last_30d = float(
                    df_d["close"].tail(30).max()
                )
                uptrend_conditions["C"] = \
                    high_last_30d > high_30_60d
            else:
                uptrend_conditions["C"] = False

            # Condition D — price above 50-day avg close
            uptrend_conditions["D"] = bool(
                avg_50d and close_td > avg_50d
            )

            uptrend_score = sum(
                1 for v in uptrend_conditions.values() if v
            )

            # S14 fires if pullback detected + OBV held
            # regardless of uptrend score
            s14 = pullback_detected and obv_held

    # Points based on uptrend score (more context = more pts)
    s14_pts = 0
    if s14:
        if uptrend_score == 4:
            s14_pts = 20  # all 4 uptrend conditions + OBV held
        elif uptrend_score == 3:
            s14_pts = 15  # strong uptrend + OBV held
        elif uptrend_score == 2:
            s14_pts = 10  # moderate uptrend
        elif uptrend_score == 1:
            s14_pts = 6  # weak uptrend context
        else:
            s14_pts = 3  # no uptrend but OBV held pullback

    scores["s14_pullback_divergence"] = s14_pts
    flags["s14_pullback_divergence"] = s14
    flags["s14_pullback_depth_pct"] = pullback_depth_pct \
        if pullback_detected else 0
    flags["s14_obv_held"] = obv_held
    flags["s14_uptrend_conditions"] = uptrend_conditions
    flags["s14_uptrend_score"] = uptrend_score

    # ── Total score ────────────────────────────────────────────
    total_score = sum(scores.values())
    total_score = min(total_score, 100)   # cap at 100

    # ── Group scores ───────────────────────────────────────────
    group_a = scores["s6_price_flat_obv_rising"] + \
              scores["s1_obv_new_high_price_not"] + \
              scores["s4_obv_100d_breakout"]
    group_b = scores["s3_obv_delivery_confirm"] + \
              scores["s7_delivery_obv_spike"] + \
              scores["s8_multi_high_delivery"]
    group_c = scores["s2_obv_sustained_rise"] + \
              scores["s5_obv_slope_accel"] + \
              scores["s9_tight_range_obv_rising"]
    group_d = scores["s10_shakeout_detected"] + \
              scores["s11_weekly_divergence"] + \
              scores["s12_both_timeframes"] + \
              scores["s13_obv_higher_highs"] + \
              scores["s14_pullback_divergence"]

    # ── Conviction tier ────────────────────────────────────────
    if total_score >= 70:
        conviction_tier = "HIGH"
    elif total_score >= 50:
        conviction_tier = "MEDIUM"
    elif total_score >= 30:
        conviction_tier = "LOW"
    else:
        conviction_tier = "SKIP"

    return {
        "total_score"    : total_score,
        "conviction_tier": conviction_tier,
        "group_a_score"  : group_a,
        "group_b_score"  : group_b,
        "group_c_score"  : group_c,
        "group_d_score"  : group_d,
        "scores"         : scores,
        "flags"          : flags,
        # Key values for display
        "obv_today"            : int(obv_today),
        "obv_chg_20d_norm"     : round(obv_chg_20d_norm, 2),
        "obv_rising_20d"       : bool(obv_rising_20d),
        "price_chg_20d"        : round(price_chg_20d, 2),
        "deliv_today"          : round(deliv_today, 2),
        "deliv_avg_22d"        : round(deliv_avg_22d, 2),
        "range_pct_20"         : round(range_pct_20, 2),
    }


# ================================================================
# MAIN FUNCTION
# ================================================================

def calculate_obv_indicators(
    df_daily: pd.DataFrame,
    df_weekly: pd.DataFrame = None
) -> dict:
    """
    Complete OBV analysis — Tier 1 filters + Tier 2 scoring.

    Parameters
    ----------
    df_daily : pd.DataFrame
        Daily prices sorted oldest to newest.
        Must have: date, close, volume
        Optional but recommended: delivery_pct, delivery_qty

    df_weekly : pd.DataFrame, optional
        Weekly prices. Resampled from daily if not provided.

    Returns
    -------
    dict
        tier1_passed    : bool
        tier1_reason    : str (why failed, if applicable)
        total_score     : int (0-100)
        conviction_tier : str (HIGH/MEDIUM/LOW/SKIP)
        group scores    : per group breakdown
        scenario flags  : all 13 scenario results
        scenario scores : points per scenario
    """
    if df_daily is None or len(df_daily) < 30:
        return {
            "tier1_passed"   : False,
            "tier1_reason"   : "insufficient_data",
            "total_score"    : 0,
            "conviction_tier": "SKIP",
        }

    # ── Tier 1 ─────────────────────────────────────────────────
    tier1 = check_tier1_filters(df_daily)

    if not tier1["passed"]:
        return {
            "tier1_passed"   : False,
            "tier1_reason"   : tier1["reason"],
            "total_score"    : 0,
            "conviction_tier": "SKIP",
        }

    # ── Calculate OBV ──────────────────────────────────────────
    df_d = df_daily.copy()
    df_d = df_d.sort_values("date").reset_index(drop=True)
    df_d = _calculate_obv(df_d)

    # ── Weekly OBV ─────────────────────────────────────────────
    if df_weekly is not None and len(df_weekly) >= 8:
        df_w = df_weekly.copy()
        df_w = df_w.sort_values("date").reset_index(drop=True)
        df_w = _calculate_obv(df_w)
    else:
        df_w = _resample_to_weekly(df_d)
        df_w = _calculate_obv(df_w)

    # ── Tier 2 scoring ─────────────────────────────────────────
    scoring = _score_scenarios(df_d, df_w)

    result = {
        "tier1_passed"   : True,
        "tier1_reason"   : None,
    }
    result.update(scoring)

    return result


# ================================================================
# CONVENIENCE — OBV SERIES ONLY
# ================================================================

def get_obv_series(df: pd.DataFrame) -> pd.Series:
    """Returns OBV series for charting or further analysis."""
    if df is None or len(df) < 2:
        return pd.Series()
    df = df.sort_values("date").reset_index(drop=True)
    df = _calculate_obv(df)
    return df["obv"]