# ================================================================
# src/indicators/monthly.py
# ----------------------------------------------------------------
# Monthly chart indicator calculations.
# Resamples daily data to monthly candles.
#
# Checks monthly timeframe for:
#   1. Monthly OBV direction and trend
#   2. Monthly EMA position (20, 50 month EMA)
#   3. Monthly candle structure (bullish/bearish/doji)
#   4. Monthly higher lows (uptrend structure)
#   5. Monthly volume trend
#   6. Position in 52W range
#
# Output:
#   STRONG → monthly chart bullish, all signals positive
#   MIXED  → some signals positive, some negative
#   WEAK   → monthly chart still bearish
#
# Key principle:
#   Monthly chart confirms the BIG PICTURE
#   Daily/weekly signals are more reliable when
#   monthly agrees
# ================================================================

import pandas as pd
import numpy as np
from src.utils.logger import get_logger

log = get_logger(__name__)


# ================================================================
# RESAMPLE DAILY TO MONTHLY
# ================================================================

def resample_to_monthly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resamples daily OHLCV to monthly candles.

    Parameters
    ----------
    df : pd.DataFrame
        Daily prices sorted oldest to newest.
        Must have: date, open, high, low, close, volume
        Optional: delivery_pct, delivery_qty

    Returns
    -------
    pd.DataFrame
        Monthly candles with date = last trading day of month.
    """
    if df is None or len(df) < 22:
        return pd.DataFrame()

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df = df.set_index("date")

    # Resample to monthly
    agg_dict = {
        "open"  : "first",
        "high"  : "max",
        "low"   : "min",
        "close" : "last",
        "volume": "sum",
    }

    if "delivery_pct" in df.columns:
        agg_dict["delivery_pct"] = "mean"
    if "delivery_qty" in df.columns:
        agg_dict["delivery_qty"] = "sum"

    monthly = df.resample("ME").agg(agg_dict).dropna(subset=["close"])
    monthly = monthly.reset_index()
    monthly = monthly.rename(columns={"date": "month_end"})
    monthly["date"] = monthly["month_end"]

    return monthly


# ================================================================
# CALCULATE MONTHLY OBV
# ================================================================

def _calculate_monthly_obv(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates OBV on monthly candles."""
    df = df.copy()
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

    df["obv"] = obv
    return df


# ================================================================
# MAIN FUNCTION
# ================================================================

def calculate_monthly_indicators(
    df_daily: pd.DataFrame
) -> dict:
    """
    Calculates monthly chart indicators from daily data.

    Parameters
    ----------
    df_daily : pd.DataFrame
        Daily prices. Needs at least 6 months for meaningful
        monthly analysis. 3 years ideal.

    Returns
    -------
    dict
        Monthly indicators and strength assessment.
        Keys:
          monthly_strength     : STRONG / MIXED / WEAK
          strength_score       : 0-100
          monthly_signals      : list of firing signals
          monthly_candle       : BULLISH / BEARISH / DOJI
          obv_direction        : UP / DOWN / FLAT
          above_monthly_20ema  : bool
          above_monthly_50ema  : bool
          monthly_higher_lows  : bool
          monthly_vol_rising   : bool
          months_in_base       : int
          pct_from_ath         : float
          monthly_return_3m    : float
          monthly_return_6m    : float
    """
    if df_daily is None or len(df_daily) < 66:  # 3 months minimum
        return {
            "monthly_strength" : "INSUFFICIENT_DATA",
            "strength_score"   : 0,
            "monthly_signals"  : [],
        }

    # Resample to monthly
    df_m = resample_to_monthly(df_daily)

    if df_m is None or len(df_m) < 3:
        return {
            "monthly_strength" : "INSUFFICIENT_DATA",
            "strength_score"   : 0,
            "monthly_signals"  : [],
        }

    # Calculate monthly OBV
    df_m = _calculate_monthly_obv(df_m)

    # Current values
    curr         = df_m.iloc[-1]
    curr_close   = float(curr["close"])
    curr_open    = float(curr["open"])
    curr_high    = float(curr["high"])
    curr_low     = float(curr["low"])
    curr_vol     = float(curr["volume"])
    curr_obv     = float(curr["obv"])

    # ── Monthly EMAs ───────────────────────────────────────────
    df_m["ema20"] = df_m["close"].ewm(span=20, adjust=False).mean()
    df_m["ema50"] = df_m["close"].ewm(span=50, adjust=False).mean()

    ema20_curr = float(df_m["ema20"].iloc[-1])
    ema50_curr = float(df_m["ema50"].iloc[-1]) \
                 if len(df_m) >= 50 else None

    above_ema20 = curr_close > ema20_curr
    above_ema50 = (curr_close > ema50_curr) \
                  if ema50_curr else None

    # ── Monthly candle type ────────────────────────────────────
    candle_range = curr_high - curr_low
    candle_body  = abs(curr_close - curr_open)
    body_pct     = candle_body / candle_range * 100 \
                   if candle_range > 0 else 0

    if body_pct < 20:
        monthly_candle = "DOJI"
    elif curr_close > curr_open:
        monthly_candle = "BULLISH"
    else:
        monthly_candle = "BEARISH"

    # ── Monthly OBV direction ──────────────────────────────────
    obv_3m_ago = float(df_m["obv"].iloc[-4]) \
                 if len(df_m) >= 4 else curr_obv
    obv_6m_ago = float(df_m["obv"].iloc[-7]) \
                 if len(df_m) >= 7 else curr_obv

    obv_chg_3m = curr_obv - obv_3m_ago
    obv_chg_6m = curr_obv - obv_6m_ago

    avg_monthly_vol = float(df_m["volume"].tail(6).mean())
    obv_chg_ratio   = obv_chg_3m / avg_monthly_vol \
                      if avg_monthly_vol > 0 else 0

    if obv_chg_ratio > 0.5:
        obv_direction = "UP"
    elif obv_chg_ratio < -0.5:
        obv_direction = "DOWN"
    else:
        obv_direction = "FLAT"

    # ── Monthly higher lows ────────────────────────────────────
    monthly_higher_lows = False
    if len(df_m) >= 4:
        lows = df_m["low"].tail(4).values
        monthly_higher_lows = bool(
            lows[-1] > lows[-2] > lows[-3]
        )

    # ── Monthly volume trend ───────────────────────────────────
    monthly_vol_rising = False
    if len(df_m) >= 6:
        vol_3m_avg = float(df_m["volume"].tail(3).mean())
        vol_6m_avg = float(df_m["volume"].tail(6).mean())
        monthly_vol_rising = vol_3m_avg > vol_6m_avg * 1.1

    # ── Monthly returns ────────────────────────────────────────
    close_3m_ago = float(df_m["close"].iloc[-4]) \
                   if len(df_m) >= 4 else curr_close
    close_6m_ago = float(df_m["close"].iloc[-7]) \
                   if len(df_m) >= 7 else curr_close
    close_12m_ago= float(df_m["close"].iloc[-13]) \
                   if len(df_m) >= 13 else curr_close

    monthly_return_3m  = round(
        (curr_close - close_3m_ago) / close_3m_ago * 100, 2
    ) if close_3m_ago > 0 else None

    monthly_return_6m  = round(
        (curr_close - close_6m_ago) / close_6m_ago * 100, 2
    ) if close_6m_ago > 0 else None

    monthly_return_12m = round(
        (curr_close - close_12m_ago) / close_12m_ago * 100, 2
    ) if close_12m_ago > 0 else None

    # ── ATH and range position ─────────────────────────────────
    ath         = float(df_m["high"].max())
    pct_from_ath= round(
        (curr_close - ath) / ath * 100, 2
    ) if ath > 0 else None

    # All time low
    atl         = float(df_m["low"].min())
    total_range = ath - atl
    position_in_range = round(
        (curr_close - atl) / total_range * 100, 1
    ) if total_range > 0 else 50

    # ── Months in base ─────────────────────────────────────────
    # Count consecutive months where price range < 15%
    months_in_base = 0
    for i in range(len(df_m) - 1, 0, -1):
        row  = df_m.iloc[i]
        r_hi = float(row["high"])
        r_lo = float(row["low"])
        r_pct= (r_hi - r_lo) / r_lo * 100 if r_lo > 0 else 999
        if r_pct < 15:
            months_in_base += 1
        else:
            break

    # ── Monthly OBV divergence ─────────────────────────────────
    # OBV rising while price declining on monthly
    price_chg_3m = monthly_return_3m or 0
    obv_monthly_divergence = bool(
        obv_direction == "UP" and
        price_chg_3m <= 5.0
    )

    # ================================================================
    # STRENGTH SCORING
    # ================================================================

    signals      = []
    score        = 0

    # OBV direction (most important — 30 pts)
    if obv_direction == "UP":
        score += 30
        signals.append("monthly_obv_rising")
    elif obv_direction == "FLAT":
        score += 10
        signals.append("monthly_obv_flat")

    # Monthly candle (20 pts)
    if monthly_candle == "BULLISH":
        score += 20
        signals.append("monthly_candle_bullish")
    elif monthly_candle == "DOJI":
        score += 8
        signals.append("monthly_candle_doji")

    # EMA position (15 pts)
    if above_ema20:
        score += 10
        signals.append("above_monthly_20ema")
    if above_ema50:
        score += 5
        signals.append("above_monthly_50ema")

    # Higher lows (15 pts)
    if monthly_higher_lows:
        score += 15
        signals.append("monthly_higher_lows")

    # Volume rising (10 pts)
    if monthly_vol_rising:
        score += 10
        signals.append("monthly_vol_rising")

    # OBV divergence bonus (10 pts)
    if obv_monthly_divergence:
        score += 10
        signals.append("monthly_obv_divergence")

    # Deep correction context (informational — no pts)
    if pct_from_ath is not None and pct_from_ath <= -30:
        signals.append(f"deep_correction_{abs(int(pct_from_ath))}pct_off_ath")

    if months_in_base >= 3:
        signals.append(f"basing_{months_in_base}_months")

    # ── Strength tier ──────────────────────────────────────────
    if score >= 65:
        monthly_strength = "STRONG"
    elif score >= 40:
        monthly_strength = "MIXED"
    else:
        monthly_strength = "WEAK"

    return {
        # Overall assessment
        "monthly_strength"       : monthly_strength,
        "strength_score"         : score,
        "monthly_signals"        : signals,

        # Candle
        "monthly_candle"         : monthly_candle,
        "monthly_candle_body_pct": round(body_pct, 1),

        # OBV
        "obv_direction"          : obv_direction,
        "obv_chg_ratio_3m"       : round(obv_chg_ratio, 2),
        "monthly_obv_divergence" : obv_monthly_divergence,

        # EMA
        "above_monthly_20ema"    : above_ema20,
        "above_monthly_50ema"    : above_ema50,
        "monthly_ema20"          : round(ema20_curr, 2),
        "monthly_ema50"          : round(ema50_curr, 2) \
                                   if ema50_curr else None,

        # Structure
        "monthly_higher_lows"    : monthly_higher_lows,
        "monthly_vol_rising"     : monthly_vol_rising,
        "months_in_base"         : months_in_base,

        # Returns
        "monthly_return_3m"      : monthly_return_3m,
        "monthly_return_6m"      : monthly_return_6m,
        "monthly_return_12m"     : monthly_return_12m,

        # Range
        "pct_from_ath"           : pct_from_ath,
        "position_in_range"      : position_in_range,
        "ath"                    : round(ath, 2),

        # Monthly count
        "total_months"           : len(df_m),
    }