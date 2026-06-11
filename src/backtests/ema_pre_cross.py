# ================================================================
# src/backtests/ema_pre_cross.py
# ----------------------------------------------------------------
# EMA Pre-Cross Strategy Backtester
#
# SL Tiers (first triggers = exit):
#   Hard SL:    3% below entry price (always active)
#
#   Pre-cross:
#     Structure SL: 2 consecutive closes below 50 EMA
#     (20 EMA SL removed pre-cross — normal to be below 20 EMA
#      before cross happens)
#
#   Post-cross:
#     EMA Reversal: 20 EMA crosses back below 50 EMA
#
# Two exit approaches:
#   Approach A — Cross aware, exit on EMA reversal post-cross
#   Approach B — Hold through cross, exit on target or reversal
# ================================================================

import pandas as pd
import numpy as np
from src.indicators.ema_pre_cross import find_pre_cross_signals
from src.indicators.ema import calculate_emas
from src.utils.logger import get_logger

log = get_logger(__name__)

TARGET_PCT   = 10.0
TARGET_PCT_B = 15.0    # Approach B target — pre-cross gets more room
FORWARD_DAYS = 40
CONSEC_BELOW = 2


# ================================================================
# APPROACH A — CROSS AWARE, EMA REVERSAL EXIT
# ================================================================

def _approach_a(
    forward_df: pd.DataFrame,
    entry_price: float,
    sl_50ema: float,
    signal_gap: float
) -> dict:
    """
    Approach A — Cross aware strategy.

    Pre-cross exits:
      1. Target 10%
      2. Hard SL — 3% below entry
      3. Structure SL — 2 consecutive closes below 50 EMA

    Post-cross exits:
      1. Target 10%
      2. Hard SL — 3% below entry
      3. EMA reversal — 20 EMA crosses back below 50 EMA

    Time exit: 40 trading days
    """
    target_price   = entry_price * (1 + TARGET_PCT / 100)
    hard_sl        = sl_50ema * 0.97    # 3% below 50 EMA level
    below_50_count = 0
    cross_happened = False
    exit_reason    = None
    exit_price     = None
    exit_return    = None
    days_to_exit   = None

    for j, row in forward_df.iterrows():
        high     = float(row["high"])
        close    = float(row["close"])
        ema_fast = float(row["ema_fast"])
        ema_slow = float(row["ema_slow"])
        is_last  = (j == len(forward_df) - 1)

        # Track if cross happened
        if not cross_happened and ema_fast > ema_slow:
            cross_happened = True
            log.debug(f"Approach A — cross happened day {j+1}")

        # Exit 1 — Target (always active)
        if high >= target_price:
            exit_reason  = "target_10pct"
            exit_price   = target_price
            exit_return  = TARGET_PCT
            days_to_exit = j + 1
            break

        # Exit 2 — Hard SL (always active)
        if close <= hard_sl:
            exit_reason  = "sl_hard_3pct"
            exit_price   = close
            exit_return  = round(
                (close - entry_price) / entry_price * 100, 2
            )
            days_to_exit = j + 1
            break

        if not cross_happened:
            # ── Pre-cross: structure SL only ───────────────────
            # 20 EMA SL NOT used here — price below 20 EMA is
            # normal behaviour before cross happens
            if close < ema_slow:
                below_50_count += 1
                if below_50_count >= CONSEC_BELOW:
                    exit_reason  = "sl_consec_below_50ema"
                    exit_price   = close
                    exit_return  = round(
                        (close - entry_price) / entry_price * 100, 2
                    )
                    days_to_exit = j + 1
                    break
            else:
                below_50_count = 0

        else:
            # ── Post-cross: EMA reversal SL ────────────────────
            if ema_fast < ema_slow:
                exit_reason  = "sl_ema_reversal_post_cross"
                exit_price   = close
                exit_return  = round(
                    (close - entry_price) / entry_price * 100, 2
                )
                days_to_exit = j + 1
                break

        # Time exit
        if is_last:
            exit_reason  = "time_8w"
            exit_price   = close
            exit_return  = round(
                (close - entry_price) / entry_price * 100, 2
            )
            days_to_exit = j + 1

    trade_result = "WIN" if (exit_return or 0) > 0 else "LOSS"

    return {
        "a_exit_reason" : exit_reason,
        "a_exit_price"  : round(exit_price, 2) if exit_price else None,
        "a_exit_return" : exit_return,
        "a_days"        : days_to_exit,
        "a_result"      : trade_result,
        "a_cross"       : cross_happened,
    }


# ================================================================
# APPROACH B — HOLD THROUGH CROSS, TARGET FOCUSED
# ================================================================

def _approach_b(
    forward_df: pd.DataFrame,
    entry_price: float,
    sl_50ema: float,
    signal_gap: float
) -> dict:
    """
    Approach B — Hold through cross, target focused.

    Same pre-cross SL as Approach A.

    Post-cross:
      Only exit on 10% target OR hard SL OR 8 weeks
      Does NOT exit on EMA reversal post-cross
      Gives price more room to run past the cross

    This tests: is it better to hold for full target
    or exit when EMA reversal signals trend ended?
    """
    target_price   = entry_price * (1 + TARGET_PCT_B / 100)
    hard_sl = sl_50ema * 0.97  # 3% below 50 EMA level
    below_50_count = 0
    cross_happened = False
    exit_reason    = None
    exit_price     = None
    exit_return    = None
    days_to_exit   = None

    for j, row in forward_df.iterrows():
        high     = float(row["high"])
        close    = float(row["close"])
        ema_fast = float(row["ema_fast"])
        ema_slow = float(row["ema_slow"])
        is_last  = (j == len(forward_df) - 1)

        # Track cross
        if not cross_happened and ema_fast > ema_slow:
            cross_happened = True
            log.debug(f"Approach B — cross happened day {j+1}")

        # Exit 1 — Target (always active)
        if high >= target_price:
            exit_reason  = "target_10pct"
            exit_price   = target_price
            exit_return  = TARGET_PCT_B
            days_to_exit = j + 1
            break

        # Exit 2 — Hard SL (always active)
        if close <= hard_sl:
            exit_reason  = "sl_hard_3pct"
            exit_price   = close
            exit_return  = round(
                (close - entry_price) / entry_price * 100, 2
            )
            days_to_exit = j + 1
            break

        if not cross_happened:
            # ── Pre-cross: structure SL only ───────────────────
            if close < ema_slow:
                below_50_count += 1
                if below_50_count >= CONSEC_BELOW:
                    exit_reason  = "sl_consec_below_50ema"
                    exit_price   = close
                    exit_return  = round(
                        (close - entry_price) / entry_price * 100, 2
                    )
                    days_to_exit = j + 1
                    break
            else:
                below_50_count = 0

        # Post-cross: no EMA reversal exit
        # Hold until target or hard SL or time

        # Time exit
        if is_last:
            exit_reason  = "time_8w"
            exit_price   = close
            exit_return  = round(
                (close - entry_price) / entry_price * 100, 2
            )
            days_to_exit = j + 1

    trade_result = "WIN" if (exit_return or 0) > 0 else "LOSS"

    return {
        "b_exit_reason" : exit_reason,
        "b_exit_price"  : round(exit_price, 2) if exit_price else None,
        "b_exit_return" : exit_return,
        "b_days"        : days_to_exit,
        "b_result"      : trade_result,
        "cross_happened": cross_happened,
    }


# ================================================================
# MEASURE ALL OUTCOMES
# ================================================================

def measure_outcomes(
    signals_df: pd.DataFrame,
    df: pd.DataFrame
) -> pd.DataFrame:
    """Measures both approaches for every signal."""
    if signals_df is None or signals_df.empty:
        return pd.DataFrame()

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = calculate_emas(df)
    df = df.sort_values("date").reset_index(drop=True)

    results = []

    for _, signal in signals_df.iterrows():
        signal_date = pd.Timestamp(signal["signal_date"])
        entry_price = float(signal["entry_price"])
        sl_50ema    = float(signal["sl_level"])
        signal_gap  = abs(
            float(signal["ema_slow"]) - float(signal["ema_fast"])
        )

        forward_df = df[
            df["date"] > signal_date
        ].head(FORWARD_DAYS).reset_index(drop=True)

        if forward_df.empty:
            continue

        a = _approach_a(forward_df, entry_price, sl_50ema, signal_gap)
        b = _approach_b(forward_df, entry_price, sl_50ema, signal_gap)

        def get_weekly_return(weeks):
            subset = forward_df.head(weeks * 5)
            if subset.empty:
                return None
            return round(
                (float(subset.iloc[-1]["close"]) - entry_price)
                / entry_price * 100, 2
            )

        max_gain = round(
            (float(forward_df["high"].max()) - entry_price)
            / entry_price * 100, 2
        )
        max_dd = round(
            (float(forward_df["low"].min()) - entry_price)
            / entry_price * 100, 2
        )

        result = signal.to_dict()
        result.update(a)
        result.update(b)
        result.update({
            "target_price" : round(entry_price * 1.10, 2),
            "return_1w"    : get_weekly_return(1),
            "return_2w"    : get_weekly_return(2),
            "return_4w"    : get_weekly_return(4),
            "return_8w"    : get_weekly_return(8),
            "max_gain_8w"  : max_gain,
            "max_dd_8w"    : max_dd,
        })
        results.append(result)

    return pd.DataFrame(results)


# ================================================================
# GENERATE REPORT
# ================================================================

def generate_report(outcomes_df: pd.DataFrame) -> dict:
    """Generates statistics for both approaches."""
    if outcomes_df is None or outcomes_df.empty:
        return {}

    def approach_stats(result_col, return_col):
        total  = len(outcomes_df)
        wins   = len(outcomes_df[outcomes_df[result_col] == "WIN"])
        losses = len(outcomes_df[outcomes_df[result_col] == "LOSS"])
        wr     = round(wins / total * 100, 1) if total > 0 else 0

        win_ret  = outcomes_df[
            outcomes_df[result_col] == "WIN"
        ][return_col].dropna()
        loss_ret = outcomes_df[
            outcomes_df[result_col] == "LOSS"
        ][return_col].dropna()

        avg_win  = round(win_ret.mean(),  2) if not win_ret.empty  else None
        avg_loss = round(loss_ret.mean(), 2) if not loss_ret.empty else None
        rr       = round(abs(avg_win / avg_loss), 2) \
                   if avg_win and avg_loss and avg_loss != 0 else None

        loss_rate = 100 - wr
        ev = round(
            (wr / 100 * (avg_win or 0)) +
            (loss_rate / 100 * (avg_loss or 0)), 2
        )
        composite = round(ev * rr, 2) if rr else None

        return {
            "total"    : total,
            "wins"     : wins,
            "losses"   : losses,
            "win_rate" : wr,
            "avg_win"  : avg_win,
            "avg_loss" : avg_loss,
            "rr"       : rr,
            "ev"       : ev,
            "composite": composite,
        }

    a_stats = approach_stats("a_result", "a_exit_return")
    b_stats = approach_stats("b_result", "b_exit_return")

    a_exits = outcomes_df["a_exit_reason"].value_counts().to_dict()
    b_exits = outcomes_df["b_exit_reason"].value_counts().to_dict()

    cross_rate = round(
        outcomes_df["cross_happened"].mean() * 100, 1
    ) if "cross_happened" in outcomes_df.columns else None

    by_strength = {}
    for strength, grp in outcomes_df.groupby("signal_strength"):
        w = len(grp[grp["a_result"] == "WIN"])
        by_strength[strength] = {
            "total"   : len(grp),
            "wins"    : w,
            "win_rate": round(w / len(grp) * 100, 1)
        }

    return {
        "approach_a" : a_stats,
        "approach_b" : b_stats,
        "a_exits"    : a_exits,
        "b_exits"    : b_exits,
        "cross_rate" : cross_rate,
        "by_strength": by_strength,
    }


# ================================================================
# FULL BACKTEST
# ================================================================

def run_pre_cross_backtest(
    symbol: str,
    df: pd.DataFrame
) -> dict:
    """Runs complete pre-cross backtest for one symbol."""
    log.info(f"Running pre-cross backtest: {symbol}")

    signals = find_pre_cross_signals(df)

    if signals is None or signals.empty:
        return {}

    df_ema   = calculate_emas(df)
    outcomes = measure_outcomes(signals, df_ema)

    if outcomes.empty:
        return {"signals": signals}

    report = generate_report(outcomes)

    a = report.get("approach_a", {})
    b = report.get("approach_b", {})

    log.info(
        f"{symbol} — signals:{len(signals)} | "
        f"A: WR:{a.get('win_rate')}% EV:{a.get('ev')}% | "
        f"B: WR:{b.get('win_rate')}% EV:{b.get('ev')}% | "
        f"cross_rate:{report.get('cross_rate')}%"
    )

    return {
        "signals" : signals,
        "outcomes": outcomes,
        "report"  : report,
    }