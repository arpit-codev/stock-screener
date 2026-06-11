# ================================================================
# src/indicators/ema_pre_cross.py
# ----------------------------------------------------------------
# EMA Pre-Cross Momentum Strategy
#
# Detects when 20/50 EMA cross is IMMINENT — 3-5 days away.
# Enter BEFORE the crowd notices the crossover.
#
# Entry conditions (all 5 must be true):
#   1. 20 EMA still below 50 EMA (cross not happened yet)
#   2. Gap closing rapidly (shrunk 60%+ in 5 days OR < 0.5%)
#   3. Price above BOTH EMAs
#   4. 20 EMA slope strongly positive (> 0.15% in 3 days)
#   5. Volume confirming (>= 1.2x 22-day avg)
#
# Entry price: close of signal candle
# SL level:    50 EMA at time of entry
# ================================================================

import pandas as pd
import numpy as np
from src.indicators.ema import calculate_emas
from src.utils.logger import get_logger

log = get_logger(__name__)

# ── Constants ──────────────────────────────────────────────────
GAP_CLOSE_PCT       = 0.4    # gap must shrink to 40% of 5-day ago gap
GAP_NEAR_PCT        = 0.5    # OR gap < 0.5% of price = almost touching
SLOPE_MIN_PCT       = 0.15   # 20 EMA must rise 0.15%+ in 3 days
VOLUME_MIN_RATIO    = 1.2    # volume >= 1.2x 22-day avg
COOLDOWN_DAYS       = 10     # min days between signals same stock
MIN_DATA_REQUIRED   = 60


# ================================================================
# FIND PRE-CROSS SIGNALS
# ================================================================

def find_pre_cross_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Finds all pre-cross momentum signals for one symbol.

    Parameters
    ----------
    df : pd.DataFrame
        Daily prices for one symbol.
        Must have: date, open, high, low, close, volume

    Returns
    -------
    pd.DataFrame
        One row per signal.
        Columns: symbol, signal_date, signal_index,
                 entry_price, sl_level,
                 ema_fast, ema_slow,
                 gap_pct, gap_5d_pct, gap_close_speed,
                 slope_fast_pct, vol_ratio,
                 signal_strength (strong/moderate)
    """
    if df is None or len(df) < MIN_DATA_REQUIRED:
        return pd.DataFrame()

    symbol = df["symbol"].iloc[0] if "symbol" in df.columns else ""

    # Calculate EMAs
    df = calculate_emas(df)
    df = df.sort_values("date").reset_index(drop=True)

    signals      = []
    last_signal  = -COOLDOWN_DAYS  # track last signal index

    for i in range(10, len(df)):
        row      = df.iloc[i]
        ema_fast = float(row["ema_fast"])
        ema_slow = float(row["ema_slow"])
        close    = float(row["close"])
        volume   = float(row["volume"])

        # ── Condition 1 — Not crossed yet ─────────────────────
        if ema_fast >= ema_slow:
            continue

        # ── Cooldown — avoid repeated signals ─────────────────
        if i - last_signal < COOLDOWN_DAYS:
            continue

        # ── Gap calculation ────────────────────────────────────
        gap_today = ema_slow - ema_fast      # positive = not crossed
        gap_pct   = gap_today / close * 100  # gap as % of price

        # Gap 5 days ago
        row_5d_ago  = df.iloc[i - 5]
        gap_5d_ago  = float(row_5d_ago["ema_slow"]) - \
                      float(row_5d_ago["ema_fast"])
        gap_5d_pct  = gap_5d_ago / float(row_5d_ago["close"]) * 100

        # Gap closing speed
        gap_close_speed = round(gap_today / gap_5d_ago, 3) \
                          if gap_5d_ago > 0 else 1.0

        # ── Condition 2 — Gap closing rapidly ─────────────────
        gap_closing = (
            gap_close_speed <= GAP_CLOSE_PCT or    # shrunk 60%+
            gap_pct <= GAP_NEAR_PCT                 # almost touching
        )
        if not gap_closing:
            continue

        # ── Condition 3 — Price above BOTH EMAs ───────────────
        price_above_both = close > ema_slow and close > ema_fast
        if not price_above_both:
            continue

        # ── Condition 4 — 20 EMA slope strongly positive ──────
        slope_fast_pct = float(row["ema_slope_fast_pct"]) \
                         if pd.notna(row["ema_slope_fast_pct"]) else 0
        if slope_fast_pct < SLOPE_MIN_PCT:
            continue

        # ── Condition 5 — Volume confirmation ─────────────────
        vol_22d   = df.iloc[max(0, i-22):i]["volume"].mean()
        vol_ratio = round(volume / vol_22d, 2) if vol_22d > 0 else 0
        if vol_ratio < VOLUME_MIN_RATIO:
            continue

        # ── All 5 conditions met — valid signal ───────────────

        # Signal strength
        # Strong: gap < 0.3% AND slope > 0.25% AND vol > 1.5x
        strong = (
            gap_pct        <= 0.3  and
            slope_fast_pct >= 0.25 and
            vol_ratio      >= 1.5
        )
        signal_strength = "strong" if strong else "moderate"

        # SL = 50 EMA at entry
        sl_level = round(ema_slow, 2)

        signals.append({
            "symbol"           : symbol,
            "signal_date"      : row["date"],
            "signal_index"     : i,
            "entry_price"      : round(close, 2),
            "sl_level"         : sl_level,
            "ema_fast"         : round(ema_fast, 2),
            "ema_slow"         : round(ema_slow, 2),
            "gap_pct"          : round(gap_pct,          3),
            "gap_5d_pct"       : round(gap_5d_pct,       3),
            "gap_close_speed"  : gap_close_speed,
            "slope_fast_pct"   : round(slope_fast_pct,   3),
            "vol_ratio"        : vol_ratio,
            "signal_strength"  : signal_strength,
        })

        last_signal = i

    if not signals:
        log.debug(f"{symbol} — no pre-cross signals found")
        return pd.DataFrame()

    result = pd.DataFrame(signals)
    log.info(
        f"{symbol} — {len(result)} pre-cross signals "
        f"({len(result[result['signal_strength']=='strong'])} strong)"
    )
    return result