# ================================================================
# src/indicators/ema.py
# ----------------------------------------------------------------
# EMA crossover indicator — 20 EMA / 50 EMA strategy
#
# Detects:
#   1. Every 20 EMA cross above 50 EMA (buy signal)
#   2. Classifies crossover strength (strong/moderate/weak)
#   3. Finds pullback entry after crossover — 3 scenarios:
#        Scenario 1 — Clean pullback to 20 EMA
#        Scenario 2 — False break below 20 EMA + recovery
#        Scenario 3 — Tight consolidation near 20 EMA
#   4. Validates cross still intact at entry
#
# Non-negotiable: 20 EMA must be above 50 EMA at entry
# Maximum wait for pullback: 10 trading days
# ================================================================

import pandas as pd
import numpy as np
from src.utils.logger import get_logger

log = get_logger(__name__)

# ── Constants ──────────────────────────────────────────────────
EMA_FAST              = 20     # fast EMA period
EMA_SLOW              = 50     # slow EMA period
PULLBACK_TOLERANCE    = 0.5    # within 0.5% of 20 EMA = touching
FALSE_BREAK_DAYS      = 3      # max days below 20 EMA for false break
CONSOLIDATION_DAYS    = 3      # min days hugging 20 EMA
CONSOLIDATION_RANGE   = 1.0    # within 1% of 20 EMA = hugging
MAX_PULLBACK_WAIT     = 10     # signal expires after 10 days
MIN_EMA_SEPARATION    = 0.3    # 20 EMA must be 0.3% above 50 EMA
MIN_DATA_REQUIRED     = 60     # minimum rows needed (increased for 50 EMA)


# ================================================================
# STEP 1 — CALCULATE EMAs
# ================================================================

def calculate_emas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds EMA columns to the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Daily prices with at least 'close' column.
        Sorted oldest to newest.

    Returns
    -------
    pd.DataFrame
        Original df with added columns:
            ema_fast, ema_slow,
            ema_slope_fast, ema_slope_slow,
            ema_slope_fast_pct, ema_slope_slow_pct,
            ema_separation
    """
    df = df.copy()
    df = df.sort_values("date").reset_index(drop=True)

    # EMA calculation
    df["ema_fast"] = df["close"].ewm(span=EMA_FAST, adjust=False).mean()
    df["ema_slow"] = df["close"].ewm(span=EMA_SLOW, adjust=False).mean()

    # Absolute slope — compare today vs 3 days ago
    df["ema_slope_fast"] = df["ema_fast"] - df["ema_fast"].shift(3)
    df["ema_slope_slow"] = df["ema_slow"] - df["ema_slow"].shift(3)

    # Percentage slope — price range independent
    df["ema_slope_fast_pct"] = (
        (df["ema_fast"] - df["ema_fast"].shift(3)) /
         df["ema_fast"].shift(3) * 100
    ).round(4)

    df["ema_slope_slow_pct"] = (
        (df["ema_slow"] - df["ema_slow"].shift(3)) /
         df["ema_slow"].shift(3) * 100
    ).round(4)

    # EMA separation %
    df["ema_separation"] = (
            (df["ema_fast"] - df["ema_slow"]) / df["ema_slow"] * 100
    ).round(3)

    # Legacy column names — backward compatibility
    # Any code referencing ema_9/ema_15/ema_slope_9/ema_slope_15
    df["ema_9"] = df["ema_fast"]
    df["ema_15"] = df["ema_slow"]
    df["ema_slope_9"] = df["ema_slope_fast"]
    df["ema_slope_15"] = df["ema_slope_slow"]

    return df


# ================================================================
# STEP 2 — FIND CROSSOVERS
# ================================================================

def find_crossovers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Finds every instance where 20 EMA crosses above 50 EMA.

    Parameters
    ----------
    df : pd.DataFrame
        Output from calculate_emas()

    Returns
    -------
    pd.DataFrame
        One row per crossover event.
    """
    if df is None or len(df) < MIN_DATA_REQUIRED:
        return pd.DataFrame()

    crossovers = []

    for i in range(1, len(df)):
        prev = df.iloc[i - 1]
        curr = df.iloc[i]

        # Detect upward cross
        prev_below = float(prev["ema_fast"]) <= float(prev["ema_slow"])
        curr_above = float(curr["ema_fast"]) >  float(curr["ema_slow"])

        if not (prev_below and curr_above):
            continue

        ema_fast_val = float(curr["ema_fast"])
        ema_slow_val = float(curr["ema_slow"])
        close_val    = float(curr["close"])
        volume_val   = float(curr["volume"])

        # Volume ratio vs 22-day avg
        vol_22d   = df.iloc[max(0, i-22):i]["volume"].mean()
        vol_ratio = round(volume_val / vol_22d, 2) \
                    if vol_22d > 0 else 1.0

        # EMA separation at cross
        separation = (ema_fast_val - ema_slow_val) \
                     / ema_slow_val * 100

        crossover_type = _classify_crossover(
            close_val, ema_fast_val, ema_slow_val,
            vol_ratio, separation, df, i
        )

        crossovers.append({
            "symbol"           : df.iloc[i].get("symbol", ""),
            "crossover_date"   : curr["date"],
            "crossover_index"  : i,
            "ema_fast_at_cross": round(ema_fast_val, 2),
            "ema_slow_at_cross": round(ema_slow_val, 2),
            "close_at_cross"   : round(close_val,    2),
            "volume_at_cross"  : int(volume_val),
            "vol_ratio"        : vol_ratio,
            "separation_pct"   : round(separation,   3),
            "crossover_type"   : crossover_type,
            # Legacy names for compatibility
            "ema_9_at_cross"   : round(ema_fast_val, 2),
            "ema_15_at_cross"  : round(ema_slow_val, 2),
        })

    return pd.DataFrame(crossovers)


def _classify_crossover(
    close: float,
    ema_fast: float,
    ema_slow: float,
    vol_ratio: float,
    separation: float,
    df: pd.DataFrame,
    idx: int
) -> str:
    """
    Classifies crossover as strong / moderate / weak.

    WEAK:
      - Multiple crosses in last 10 days (whipsawing)
      - EMAs essentially flat (< 0.05% change in 3 days)
      - Price below fast EMA at crossover

    STRONG:
      - Both EMAs rising meaningfully (> 0.1% in 3 days)
      - Price above both EMAs
      - Separation >= 0.3% at crossover

    MODERATE:
      - Everything else
    """
    # Percentage slope
    fast_3d_ago = float(df.iloc[max(0, idx-3)]["ema_fast"])
    slow_3d_ago = float(df.iloc[max(0, idx-3)]["ema_slow"])

    slope_fast_pct = (ema_fast - fast_3d_ago) / fast_3d_ago * 100 \
                     if fast_3d_ago > 0 else 0
    slope_slow_pct = (ema_slow - slow_3d_ago) / slow_3d_ago * 100 \
                     if slow_3d_ago > 0 else 0

    # Whipsaw check
    recent_window  = df.iloc[max(0, idx-10):idx]
    recent_crosses = 0
    for j in range(1, len(recent_window)):
        p = recent_window.iloc[j-1]
        c = recent_window.iloc[j]
        if (float(p["ema_fast"]) > float(p["ema_slow"])) != \
           (float(c["ema_fast"]) > float(c["ema_slow"])):
            recent_crosses += 1

    if recent_crosses >= 2:
        return "weak"

    # WEAK
    ema_flat    = abs(slope_fast_pct) < 0.05 and \
                  abs(slope_slow_pct) < 0.05
    price_below = close < ema_fast

    if ema_flat or price_below:
        return "weak"

    # STRONG
    both_rising     = slope_fast_pct > 0.10 and slope_slow_pct > 0.05
    price_above     = close > ema_fast > ema_slow
    good_separation = separation >= MIN_EMA_SEPARATION

    if both_rising and price_above and good_separation:
        return "strong"

    return "moderate"


# ================================================================
# STEP 3 — FIND PULLBACK ENTRY
# ================================================================

def find_pullback_entry(
    df: pd.DataFrame,
    crossover_idx: int
) -> dict | None:
    """
    Finds first valid entry after a crossover.
    Checks all 3 scenarios within MAX_PULLBACK_WAIT days.

    Non-negotiable: 20 EMA must stay above 50 EMA at entry.
    """
    start  = crossover_idx + 1
    end    = min(crossover_idx + MAX_PULLBACK_WAIT + 1, len(df))
    window = df.iloc[start:end].reset_index(drop=True)

    if window.empty:
        return None

    for i in range(len(window)):
        row      = window.iloc[i]
        ema_fast = float(row["ema_fast"])
        ema_slow = float(row["ema_slow"])

        # Non-negotiable: 20 EMA must be above 50 EMA
        if ema_fast <= ema_slow:
            log.debug("Signal expired — 20 EMA crossed back below 50 EMA")
            return None

        entry = _clean_pullback(row, window, i)
        if entry:
            return entry

        entry = _false_break_recovery(row, window, i)
        if entry:
            return entry

        entry = _tight_consolidation(row, window, i)
        if entry:
            return entry

    log.debug("No pullback entry found within wait window")
    return None


def _clean_pullback(
    row: pd.Series,
    window: pd.DataFrame,
    idx: int
) -> dict | None:
    """
    Scenario 1 — Clean Pullback To 20 EMA.
      Low touches 20 EMA (within 0.5%)
      Close stays above 20 EMA
      20 EMA still above 50 EMA
    """
    ema_fast = float(row["ema_fast"])
    ema_slow = float(row["ema_slow"])
    low      = float(row["low"])
    close    = float(row["close"])

    if ema_fast <= ema_slow:
        return None

    touched_ema = low   <= ema_fast * (1 + PULLBACK_TOLERANCE / 100)
    close_above = close >= ema_fast * 0.998

    if not (touched_ema and close_above):
        return None

    if idx + 1 >= len(window):
        return None

    next_row = window.iloc[idx + 1]

    if float(next_row["ema_fast"]) <= float(next_row["ema_slow"]):
        return None

    return {
        "entry_scenario"   : "clean_pullback",
        "pullback_date"    : row["date"],
        "entry_date"       : next_row["date"],
        "entry_price"      : round(float(next_row["open"]), 2),
        "ema_fast_at_entry": round(float(next_row["ema_fast"]), 2),
        "ema_slow_at_entry": round(float(next_row["ema_slow"]), 2),
        # Legacy names
        "ema_9_at_entry"   : round(float(next_row["ema_fast"]), 2),
        "ema_15_at_entry"  : round(float(next_row["ema_slow"]), 2),
    }


def _false_break_recovery(
    row: pd.Series,
    window: pd.DataFrame,
    idx: int
) -> dict | None:
    """
    Scenario 2 — False Break Below 20 EMA + Recovery.
      Close goes below 20 EMA
      Recovers back above within 3 days
      20 EMA still above 50 EMA throughout
    """
    ema_fast = float(row["ema_fast"])
    ema_slow = float(row["ema_slow"])
    close    = float(row["close"])

    if ema_fast <= ema_slow:
        return None

    if close >= ema_fast:
        return None

    recovery_window = window.iloc[idx + 1 : idx + 1 + FALSE_BREAK_DAYS]

    for _, rec_row in recovery_window.iterrows():
        rec_fast  = float(rec_row["ema_fast"])
        rec_slow  = float(rec_row["ema_slow"])
        rec_close = float(rec_row["close"])

        if rec_fast <= rec_slow:
            return None

        if rec_close >= rec_fast * 0.998:
            rec_idx  = recovery_window.index.get_loc(rec_row.name)
            next_idx = rec_idx + 1

            if next_idx >= len(window):
                return None

            next_row = window.iloc[next_idx]

            if float(next_row["ema_fast"]) <= float(next_row["ema_slow"]):
                return None

            return {
                "entry_scenario"   : "false_break_recovery",
                "pullback_date"    : row["date"],
                "entry_date"       : next_row["date"],
                "entry_price"      : round(float(next_row["open"]), 2),
                "ema_fast_at_entry": round(float(next_row["ema_fast"]), 2),
                "ema_slow_at_entry": round(float(next_row["ema_slow"]), 2),
                "ema_9_at_entry"   : round(float(next_row["ema_fast"]), 2),
                "ema_15_at_entry"  : round(float(next_row["ema_slow"]), 2),
            }

    return None


def _tight_consolidation(
    row: pd.Series,
    window: pd.DataFrame,
    idx: int
) -> dict | None:
    """
    Scenario 3 — Tight Consolidation Near 20 EMA.
      Price within 1% of 20 EMA for 3+ days
      Close above 20 EMA on majority of days
      20 EMA still above 50 EMA throughout
      Breakout: close > previous day high
    """
    if idx < CONSOLIDATION_DAYS - 1:
        return None

    consol_window = window.iloc[
        max(0, idx - CONSOLIDATION_DAYS + 1) : idx + 1
    ]

    if len(consol_window) < CONSOLIDATION_DAYS:
        return None

    all_cross_valid = all(
        float(r["ema_fast"]) > float(r["ema_slow"])
        for _, r in consol_window.iterrows()
    )
    if not all_cross_valid:
        return None

    all_near_ema = all(
        abs(float(r["close"]) - float(r["ema_fast"]))
        / float(r["ema_fast"]) * 100 <= CONSOLIDATION_RANGE
        for _, r in consol_window.iterrows()
    )
    if not all_near_ema:
        return None

    above_count = sum(
        1 for _, r in consol_window.iterrows()
        if float(r["close"]) >= float(r["ema_fast"]) * 0.998
    )
    if above_count < CONSOLIDATION_DAYS - 1:
        return None

    if idx == 0:
        return None

    prev_row   = window.iloc[idx - 1]
    curr_close = float(row["close"])
    prev_high  = float(prev_row["high"])

    if curr_close <= prev_high:
        return None

    if idx + 1 >= len(window):
        return None

    next_row = window.iloc[idx + 1]

    if float(next_row["ema_fast"]) <= float(next_row["ema_slow"]):
        return None

    return {
        "entry_scenario"   : "tight_consolidation",
        "pullback_date"    : row["date"],
        "entry_date"       : next_row["date"],
        "entry_price"      : round(float(next_row["open"]), 2),
        "ema_fast_at_entry": round(float(next_row["ema_fast"]), 2),
        "ema_slow_at_entry": round(float(next_row["ema_slow"]), 2),
        "ema_9_at_entry"   : round(float(next_row["ema_fast"]), 2),
        "ema_15_at_entry"  : round(float(next_row["ema_slow"]), 2),
    }


# ================================================================
# MAIN — FIND ALL SIGNALS FOR ONE SYMBOL
# ================================================================

def find_ema_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Complete pipeline — finds all valid EMA crossover signals
    with pullback entries for one symbol.
    """
    if df is None or len(df) < MIN_DATA_REQUIRED:
        return pd.DataFrame()

    symbol = df["symbol"].iloc[0] if "symbol" in df.columns else ""

    df = calculate_emas(df)

    crossovers = find_crossovers(df)

    if crossovers.empty:
        log.debug(f"{symbol} — no crossovers found")
        return pd.DataFrame()

    signals = []

    for _, cross in crossovers.iterrows():
        cross_idx = int(cross["crossover_index"])
        entry     = find_pullback_entry(df, cross_idx)

        if entry is None:
            continue

        signal = {
            "symbol"            : symbol,
            "crossover_date"    : cross["crossover_date"],
            "crossover_type"    : cross["crossover_type"],
            "ema_fast_at_cross" : cross["ema_fast_at_cross"],
            "ema_slow_at_cross" : cross["ema_slow_at_cross"],
            "close_at_cross"    : cross["close_at_cross"],
            "vol_ratio_at_cross": cross["vol_ratio"],
            "separation_pct"    : cross["separation_pct"],
            "entry_scenario"    : entry["entry_scenario"],
            "pullback_date"     : entry["pullback_date"],
            "entry_date"        : entry["entry_date"],
            "entry_price"       : entry["entry_price"],
            "ema_fast_at_entry" : entry["ema_fast_at_entry"],
            "ema_slow_at_entry" : entry["ema_slow_at_entry"],
            # Legacy names — keeps ema_crossover.py working
            "ema_9_at_cross"    : cross["ema_9_at_cross"],
            "ema_15_at_cross"   : cross["ema_15_at_cross"],
            "ema_9_at_entry"    : entry["ema_9_at_entry"],
            "ema_15_at_entry"   : entry["ema_15_at_entry"],
        }
        signals.append(signal)

    if not signals:
        return pd.DataFrame()

    result = pd.DataFrame(signals)
    log.info(
        f"{symbol} — {len(crossovers)} crossovers → "
        f"{len(result)} valid signals"
    )
    return result