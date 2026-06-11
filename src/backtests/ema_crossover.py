# ================================================================
# src/backtests/ema_crossover.py
# ----------------------------------------------------------------
# EMA Crossover Strategy Backtester
#
# Entry:  First pullback to 9 EMA after 9/15 EMA crossover
#         Three entry scenarios:
#           1. Clean pullback
#           2. False break recovery
#           3. Tight consolidation
#
# Exit (first of these triggers):
#   TARGET  → intraday high hits entry * 1.10
#   SL-A    → 9 EMA crosses back below 15 EMA (close basis)
#   SL-B    → close goes 5% below entry price
#   TIME    → 8 weeks (40 trading days) elapsed
#
# Input:  Daily prices for one symbol
# Output: Trade-by-trade results + summary statistics
# ================================================================

import pandas as pd
import numpy as np
from src.indicators.ema import find_ema_signals, calculate_emas
from src.utils.logger import get_logger

log = get_logger(__name__)

# ── Strategy Constants ─────────────────────────────────────────
TARGET_PCT    = 10.0    # 10% from entry
SL_PRICE_PCT  = 5.0     # 5% below entry = hard SL
FORWARD_WEEKS = 8       # max hold = 8 weeks
FORWARD_DAYS  = 40      # 8 weeks in trading days


# ================================================================
# MEASURE OUTCOMES — UNIFIED EXIT
# ================================================================

def measure_outcomes(
    signals_df: pd.DataFrame,
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Measures outcome for each signal.
    Unified exit — first of target/SL-A/SL-B/time triggers.

    Parameters
    ----------
    signals_df : pd.DataFrame
        Output from find_ema_signals()
    df : pd.DataFrame
        Full daily prices with EMA columns for the symbol.

    Returns
    -------
    pd.DataFrame
        signals_df with outcome columns added.
    """
    if signals_df is None or signals_df.empty:
        return pd.DataFrame()

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = calculate_emas(df)
    df = df.sort_values("date").reset_index(drop=True)

    results = []

    for _, signal in signals_df.iterrows():
        entry_date   = pd.Timestamp(signal["entry_date"])
        entry_price  = float(signal["entry_price"])
        target_price = entry_price * (1 + TARGET_PCT  / 100)
        sl_price     = entry_price * (1 - SL_PRICE_PCT / 100)

        # Forward window from entry date
        forward_df = df[
            df["date"] > entry_date
        ].head(FORWARD_DAYS).reset_index(drop=True)

        if forward_df.empty:
            continue

        # ── Day by day unified scan ────────────────────────────
        exit_reason      = None
        exit_date        = None
        exit_price       = None
        exit_return_pct  = None
        days_to_exit     = None
        sl_hit_price     = False
        sl_hit_ema       = False
        target_hit       = False
        time_exit        = False

        for j, row in forward_df.iterrows():
            high        = float(row["high"])
            low         = float(row["low"])
            close       = float(row["close"])
            ema_9       = float(row["ema_9"])
            ema_15      = float(row["ema_15"])
            is_last_day = (j == len(forward_df) - 1)

            # ── CHECK 1: TARGET ───────────────────────────────
            # Intraday high hits 10% above entry
            if high >= target_price:
                target_hit      = True
                exit_reason     = "target_10pct"
                exit_date       = row["date"].date()
                exit_price      = target_price
                exit_return_pct = TARGET_PCT
                days_to_exit    = j + 1
                break

            # ── CHECK 2: SL-B — Price SL ─────────────────────
            # Close goes 5% below entry price
            if close <= sl_price:
                sl_hit_price    = True
                exit_reason     = "sl_price_5pct"
                exit_date       = row["date"].date()
                exit_price      = close
                exit_return_pct = round(
                    (close - entry_price) / entry_price * 100, 2
                )
                days_to_exit    = j + 1
                break

            # ── CHECK 3: SL-A — EMA Reversal ─────────────────
            # 9 EMA crosses back below 15 EMA
            # Exit at close of that day
            if ema_9 < ema_15:
                sl_hit_ema      = True
                exit_reason     = "sl_ema_cross"
                exit_date       = row["date"].date()
                exit_price      = close
                exit_return_pct = round(
                    (close - entry_price) / entry_price * 100, 2
                )
                days_to_exit    = j + 1
                break

            # ── CHECK 4: TIME EXIT ────────────────────────────
            # 8 weeks elapsed — exit at close
            if is_last_day:
                time_exit       = True
                exit_reason     = "time_8w"
                exit_date       = row["date"].date()
                exit_price      = close
                exit_return_pct = round(
                    (close - entry_price) / entry_price * 100, 2
                )
                days_to_exit    = j + 1

        if exit_reason is None:
            continue

        # ── Classify trade result ──────────────────────────────
        # Fixed — any exit with positive return = WIN
        if target_hit:
            trade_result = "WIN"
        elif exit_return_pct is not None and exit_return_pct > 0:
            trade_result = "WIN"
        elif exit_return_pct is not None and exit_return_pct <= 0:
            trade_result = "LOSS"
        else:
            trade_result = "OPEN"

        # ── Weekly return snapshots ────────────────────────────
        def get_weekly_return(weeks):
            subset = forward_df.head(weeks * 5)
            if subset.empty:
                return None
            return round(
                (float(subset.iloc[-1]["close"]) - entry_price)
                / entry_price * 100, 2
            )

        # ── Max gain and drawdown ──────────────────────────────
        max_gain = round(
            (float(forward_df["high"].max()) - entry_price)
            / entry_price * 100, 2
        )
        max_dd = round(
            (float(forward_df["low"].min()) - entry_price)
            / entry_price * 100, 2
        )

        result = signal.to_dict()
        result.update({
            "target_price"    : round(target_price, 2),
            "sl_price"        : round(sl_price, 2),
            "exit_reason"     : exit_reason,
            "exit_date"       : exit_date,
            "exit_price"      : round(float(exit_price), 2)
                                if exit_price else None,
            "exit_return_pct" : exit_return_pct,
            "days_to_exit"    : days_to_exit,
            "trade_result"    : trade_result,
            "sl_hit_price"    : sl_hit_price,
            "sl_hit_ema"      : sl_hit_ema,
            "target_hit"      : target_hit,
            "time_exit"       : time_exit,
            "return_1w"       : get_weekly_return(1),
            "return_2w"       : get_weekly_return(2),
            "return_4w"       : get_weekly_return(4),
            "return_8w"       : get_weekly_return(8),
            "max_gain_8w"     : max_gain,
            "max_drawdown_8w" : max_dd,
        })
        results.append(result)

    return pd.DataFrame(results)


# ================================================================
# GENERATE REPORT
# ================================================================

def generate_report(outcomes_df: pd.DataFrame) -> dict:
    """
    Generates statistics for one symbol.

    Parameters
    ----------
    outcomes_df : pd.DataFrame
        Output from measure_outcomes()

    Returns
    -------
    dict
        Complete statistics.
    """
    if outcomes_df is None or outcomes_df.empty:
        return {}

    total     = len(outcomes_df)
    wins      = len(outcomes_df[outcomes_df["trade_result"] == "WIN"])
    losses    = len(outcomes_df[outcomes_df["trade_result"] == "LOSS"])
    win_rate  = round(wins / total * 100, 1) if total > 0 else 0

    # SL breakdown
    sl_price = int(outcomes_df["sl_hit_price"].sum())
    sl_ema   = int(outcomes_df["sl_hit_ema"].sum())
    targets  = int(outcomes_df["target_hit"].sum())
    time_ex  = int(outcomes_df["time_exit"].sum())

    # Return stats
    win_returns  = outcomes_df[
        outcomes_df["trade_result"] == "WIN"
    ]["exit_return_pct"].dropna()

    loss_returns = outcomes_df[
        outcomes_df["trade_result"] == "LOSS"
    ]["exit_return_pct"].dropna()

    avg_win  = round(win_returns.mean(),  2) if not win_returns.empty  else None
    avg_loss = round(loss_returns.mean(), 2) if not loss_returns.empty else None

    rr = round(
        abs(avg_win / avg_loss), 2
    ) if avg_win and avg_loss and avg_loss != 0 else None

    # EV and composite
    loss_rate = 100 - win_rate
    ev = round(
        (win_rate  / 100 * (avg_win  or 0)) +
        (loss_rate / 100 * (avg_loss or 0)), 2
    )
    composite = round(ev * rr, 2) if rr else None

    # 8 week hold stats
    valid_8w = outcomes_df["return_8w"].dropna()
    avg_8w   = round(valid_8w.mean(), 2) if not valid_8w.empty else None
    best_8w  = round(valid_8w.max(),  2) if not valid_8w.empty else None
    worst_8w = round(valid_8w.min(),  2) if not valid_8w.empty else None

    # By crossover type
    by_cross_type = {}
    for ctype, grp in outcomes_df.groupby("crossover_type"):
        w = len(grp[grp["trade_result"] == "WIN"])
        by_cross_type[ctype] = {
            "total"   : len(grp),
            "wins"    : w,
            "win_rate": round(w / len(grp) * 100, 1)
        }

    # By entry scenario
    by_entry = {}
    for etype, grp in outcomes_df.groupby("entry_scenario"):
        w = len(grp[grp["trade_result"] == "WIN"])
        by_entry[etype] = {
            "total"   : len(grp),
            "wins"    : w,
            "win_rate": round(w / len(grp) * 100, 1)
        }

    return {
        "total_trades"    : total,
        "wins"            : wins,
        "losses"          : losses,
        "win_rate_pct"    : win_rate,
        "targets_hit"     : targets,
        "sl_price_hits"   : sl_price,
        "sl_ema_hits"     : sl_ema,
        "time_exits"      : time_ex,
        "avg_win"         : avg_win,
        "avg_loss"        : avg_loss,
        "risk_reward"     : rr,
        "ev_per_trade"    : ev,
        "composite_score" : composite,
        "avg_return_8w"   : avg_8w,
        "best_return_8w"  : best_8w,
        "worst_return_8w" : worst_8w,
        "by_crossover_type" : by_cross_type,
        "by_entry_scenario" : by_entry,
    }


# ================================================================
# FULL BACKTEST FOR ONE SYMBOL
# ================================================================

def run_ema_backtest(symbol: str, df: pd.DataFrame) -> dict:
    """
    Runs complete EMA crossover backtest for one symbol.

    Parameters
    ----------
    symbol : str
        NSE trading symbol
    df : pd.DataFrame
        Full daily prices for the symbol

    Returns
    -------
    dict
        Keys: signals, outcomes, report
    """
    log.info(f"Running EMA backtest: {symbol}")

    # Find signals
    signals = find_ema_signals(df)

    if signals is None or signals.empty:
        log.info(f"{symbol} — no signals found")
        return {}

    # Calculate EMAs on full df for outcome measurement
    df_ema = calculate_emas(df)

    # Measure outcomes
    outcomes = measure_outcomes(signals, df_ema)

    if outcomes.empty:
        return {"signals": signals}

    # Generate report
    report = generate_report(outcomes)

    log.info(
        f"{symbol} — signals: {report.get('total_trades', 0)} | "
        f"WR: {report.get('win_rate_pct', 0)}% | "
        f"RR: {report.get('risk_reward', 'N/A')} | "
        f"EV: {report.get('ev_per_trade', 'N/A')}%"
    )

    return {
        "signals"  : signals,
        "outcomes" : outcomes,
        "report"   : report,
    }