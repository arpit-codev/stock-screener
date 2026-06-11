# ================================================================
# src/indicators/ema_momentum.py
# ----------------------------------------------------------------
# EMA Momentum Entry Strategy
#
# Different from ema.py (pullback strategy).
# This file finds MOMENTUM entries — two consecutive bullish
# candles with higher highs and higher lows after crossover.
#
# Entry:  Close of second bullish HH+HL candle
# SL:     Candle 1 low (3% buffer + 2 day non-recovery rule)
#
# Non-negotiable: 9 EMA must be above 15 EMA at entry
# Max wait: 10 days after crossover
# Crossover candle cannot be Candle 1
# ================================================================

import pandas as pd
import numpy as np
from src.indicators.ema import calculate_emas, find_crossovers
from src.utils.logger import get_logger

log = get_logger(__name__)

# ── Constants ──────────────────────────────────────────────────
MAX_WAIT_DAYS    = 10     # signal expires after 10 days
SL_BUFFER_PCT    = 3.0    # 3% below candle 1 low = SL breach
MIN_DATA_REQUIRED = 50


# ================================================================
# FIND MOMENTUM ENTRY
# ================================================================

def find_momentum_entry(
    df: pd.DataFrame,
    crossover_idx: int
) -> dict | None:
    """
    Finds two consecutive bullish HH+HL candles after crossover.

    Parameters
    ----------
    df : pd.DataFrame
        Full daily prices with EMA columns. Sorted oldest to newest.
    crossover_idx : int
        Index of the crossover candle.

    Returns
    -------
    dict or None
        Entry details if found, None if signal expired.

        Keys:
          candle1_date    — date of first bullish candle
          candle1_high    — high of candle 1
          candle1_low     — low of candle 1 (SL reference)
          candle2_date    — date of second bullish candle
          entry_price     — close of candle 2
          entry_date      — date of candle 2
          sl_level        — candle1_low × 0.97
          ema_9_at_entry  — 9 EMA on entry day
          ema_15_at_entry — 15 EMA on entry day
    """
    # Scan from Day 1 after crossover — crossover candle excluded
    start = crossover_idx + 1
    end   = min(crossover_idx + MAX_WAIT_DAYS + 1, len(df))

    if start >= len(df):
        return None

    window = df.iloc[start:end].reset_index(drop=True)

    if len(window) < 2:
        return None

    for i in range(len(window) - 1):
        c1    = window.iloc[i]
        c2    = window.iloc[i + 1]

        # ── Non-negotiable: 9 EMA > 15 EMA on both days ───────
        if float(c1["ema_9"]) <= float(c1["ema_15"]):
            log.debug("Signal expired — 9 EMA crossed below 15 EMA")
            return None

        if float(c2["ema_9"]) <= float(c2["ema_15"]):
            log.debug("Signal expired — 9 EMA crossed below 15 EMA")
            return None

        # ── Previous day for Candle 1 reference ───────────────
        # Candle 1 needs to be higher high + higher low vs prev day
        if i == 0:
            # Prev day is the crossover candle
            prev = df.iloc[crossover_idx]
        else:
            prev = window.iloc[i - 1]

        prev_high = float(prev["high"])
        prev_low  = float(prev["low"])

        # ── CANDLE 1 conditions ────────────────────────────────
        c1_open  = float(c1["open"])
        c1_high  = float(c1["high"])
        c1_low   = float(c1["low"])
        c1_close = float(c1["close"])

        c1_green = c1_close > c1_open           # green candle
        c1_hh    = c1_high  > prev_high         # higher high
        c1_hl    = c1_low   > prev_low          # higher low

        if not (c1_green and c1_hh and c1_hl):
            continue

        # ── CANDLE 2 conditions ────────────────────────────────
        c2_open  = float(c2["open"])
        c2_high  = float(c2["high"])
        c2_low   = float(c2["low"])
        c2_close = float(c2["close"])

        c2_green = c2_close > c2_open           # green candle
        c2_hh    = c2_high  > c1_high           # higher high vs C1
        c2_hl    = c2_low   > c1_low            # higher low vs C1

        if not (c2_green and c2_hh and c2_hl):
            continue

        # ── Both conditions met — valid entry ──────────────────
        sl_level = round(c1_low * (1 - SL_BUFFER_PCT / 100), 2)

        log.debug(
            f"Momentum entry found: "
            f"C1={c1['date']} low={c1_low} "
            f"C2={c2['date']} close={c2_close}"
        )

        return {
            "candle1_date"   : c1["date"],
            "candle1_high"   : round(c1_high,  2),
            "candle1_low"    : round(c1_low,   2),
            "candle2_date"   : c2["date"],
            "entry_date"     : c2["date"],
            "entry_price"    : round(c2_close, 2),
            "sl_level"       : sl_level,
            "ema_9_at_entry" : round(float(c2["ema_9"]),  2),
            "ema_15_at_entry": round(float(c2["ema_15"]), 2),
        }

    log.debug("No momentum entry found within wait window")
    return None


# ================================================================
# FIND ALL MOMENTUM SIGNALS FOR ONE SYMBOL
# ================================================================

def find_momentum_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Complete pipeline — finds all valid momentum signals.

    Parameters
    ----------
    df : pd.DataFrame
        Daily prices for one symbol.
        Must have: date, open, high, low, close, volume, symbol

    Returns
    -------
    pd.DataFrame
        One row per valid signal with entry details.
    """
    if df is None or len(df) < MIN_DATA_REQUIRED:
        return pd.DataFrame()

    symbol = df["symbol"].iloc[0] if "symbol" in df.columns else ""

    # Calculate EMAs
    df = calculate_emas(df)

    # Find all crossovers
    crossovers = find_crossovers(df)

    if crossovers.empty:
        log.debug(f"{symbol} — no crossovers found")
        return pd.DataFrame()

    signals = []

    for _, cross in crossovers.iterrows():
        cross_idx = int(cross["crossover_index"])

        entry = find_momentum_entry(df, cross_idx)

        if entry is None:
            continue

        signal = {
            "symbol"           : symbol,
            # Crossover details
            "crossover_date"   : cross["crossover_date"],
            "crossover_type"   : cross["crossover_type"],
            "ema_9_at_cross"   : cross["ema_9_at_cross"],
            "ema_15_at_cross"  : cross["ema_15_at_cross"],
            "close_at_cross"   : cross["close_at_cross"],
            "vol_ratio_at_cross": cross["vol_ratio"],
            # Entry details
            "candle1_date"     : entry["candle1_date"],
            "candle1_high"     : entry["candle1_high"],
            "candle1_low"      : entry["candle1_low"],
            "candle2_date"     : entry["candle2_date"],
            "entry_date"       : entry["entry_date"],
            "entry_price"      : entry["entry_price"],
            "sl_level"         : entry["sl_level"],
            "ema_9_at_entry"   : entry["ema_9_at_entry"],
            "ema_15_at_entry"  : entry["ema_15_at_entry"],
        }
        signals.append(signal)

    if not signals:
        return pd.DataFrame()

    result = pd.DataFrame(signals)
    log.info(
        f"{symbol} — {len(crossovers)} crossovers → "
        f"{len(result)} momentum signals"
    )
    return result