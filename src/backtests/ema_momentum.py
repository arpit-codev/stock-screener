# ================================================================
# src/backtests/ema_momentum.py
# ----------------------------------------------------------------
# EMA Momentum Strategy Backtester
#
# Entry:  Close of second consecutive bullish HH+HL candle
# SL:     3% below Candle 1 low + 2 day non-recovery rule
# Target: 10% from entry
# Time:   8 weeks max hold
# ================================================================

import pandas as pd
import numpy as np
from src.indicators.ema_momentum import find_momentum_signals
from src.indicators.ema import calculate_emas
from src.utils.logger import get_logger

log = get_logger(__name__)

TARGET_PCT   = 10.0
FORWARD_DAYS = 40
SL_BUFFER    = 3.0


# ================================================================
# MEASURE OUTCOMES
# ================================================================

def measure_outcomes(
    signals_df: pd.DataFrame,
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Measures outcome for each momentum signal.

    SL logic:
      Day X:   close < candle1_low × 0.97  (breach day)
      Day X+1: close still below candle1_low (no recovery)
               → EXIT at Day X+1 close

    If Day X+1 recovers above candle1_low → SL cancelled
    Trade continues.
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
        candle1_low  = float(signal["candle1_low"])
        sl_level     = float(signal["sl_level"])      # 3% below c1 low
        target_price = entry_price * (1 + TARGET_PCT / 100)

        # Forward window
        forward_df = df[
            df["date"] > entry_date
        ].head(FORWARD_DAYS).reset_index(drop=True)

        if forward_df.empty:
            continue

        # ── Day by day scan ────────────────────────────────────
        exit_reason          = None
        exit_date            = None
        exit_price           = None
        exit_return_pct      = None
        days_to_exit         = None
        sl_hit_structure     = False
        sl_hit_ema           = False
        target_hit           = False
        time_exit            = False

        sl_breach_day        = None   # day when 3% breach happened
        sl_breach_close      = None

        for j, row in forward_df.iterrows():
            high        = float(row["high"])
            low         = float(row["low"])
            close       = float(row["close"])
            ema_9       = float(row["ema_9"])
            ema_15      = float(row["ema_15"])
            is_last_day = (j == len(forward_df) - 1)

            # ── CHECK 1: TARGET ───────────────────────────────
            if high >= target_price:
                target_hit      = True
                exit_reason     = "target_10pct"
                exit_date       = row["date"].date()
                exit_price      = target_price
                exit_return_pct = TARGET_PCT
                days_to_exit    = j + 1
                break

            # ── CHECK 2: SL — Structure (2 day rule) ─────────
            # Day 1: close breaches 3% below candle 1 low
            if sl_breach_day is not None:
                # We had a breach yesterday — check recovery
                if close < candle1_low:
                    # No recovery — SL confirmed → EXIT today
                    sl_hit_structure = True
                    exit_reason      = "sl_structure"
                    exit_date        = row["date"].date()
                    exit_price       = close
                    exit_return_pct  = round(
                        (close - entry_price) / entry_price * 100, 2
                    )
                    days_to_exit     = j + 1
                    break
                else:
                    # Recovered above candle 1 low → SL cancelled
                    log.debug(
                        f"SL breach cancelled — recovered above "
                        f"candle1_low {candle1_low}"
                    )
                    sl_breach_day   = None
                    sl_breach_close = None

            # Check for new breach
            if close < sl_level:   # 3% below candle 1 low
                sl_breach_day   = row["date"]
                sl_breach_close = close
                log.debug(
                    f"SL breach detected at {row['date'].date()} "
                    f"close={close} sl_level={sl_level}"
                )
                # Don't exit yet — wait for next day

            # ── CHECK 3: EMA REVERSAL ─────────────────────────
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

        # ── Classify result ────────────────────────────────────
        if exit_return_pct is not None and exit_return_pct > 0:
            trade_result = "WIN"
        elif exit_return_pct is not None and exit_return_pct <= 0:
            trade_result = "LOSS"
        else:
            trade_result = "OPEN"

        # ── Weekly returns ─────────────────────────────────────
        def get_weekly_return(weeks):
            subset = forward_df.head(weeks * 5)
            if subset.empty:
                return None
            return round(
                (float(subset.iloc[-1]["close"]) - entry_price)
                / entry_price * 100, 2
            )

        # ── Max gain/drawdown ──────────────────────────────────
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
            "exit_reason"     : exit_reason,
            "exit_date"       : exit_date,
            "exit_price"      : round(float(exit_price), 2)
                                if exit_price else None,
            "exit_return_pct" : exit_return_pct,
            "days_to_exit"    : days_to_exit,
            "trade_result"    : trade_result,
            "sl_hit_structure": sl_hit_structure,
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
    """Generates statistics for one symbol."""
    if outcomes_df is None or outcomes_df.empty:
        return {}

    total    = len(outcomes_df)
    wins     = len(outcomes_df[outcomes_df["trade_result"] == "WIN"])
    losses   = len(outcomes_df[outcomes_df["trade_result"] == "LOSS"])
    win_rate = round(wins / total * 100, 1) if total > 0 else 0

    sl_struct = int(outcomes_df["sl_hit_structure"].sum())
    sl_ema    = int(outcomes_df["sl_hit_ema"].sum())
    targets   = int(outcomes_df["target_hit"].sum())
    time_ex   = int(outcomes_df["time_exit"].sum())

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

    loss_rate = 100 - win_rate
    ev = round(
        (win_rate  / 100 * (avg_win  or 0)) +
        (loss_rate / 100 * (avg_loss or 0)), 2
    )
    composite = round(ev * rr, 2) if rr else None

    valid_8w = outcomes_df["return_8w"].dropna()
    avg_8w   = round(valid_8w.mean(), 2) if not valid_8w.empty else None
    best_8w  = round(valid_8w.max(),  2) if not valid_8w.empty else None
    worst_8w = round(valid_8w.min(),  2) if not valid_8w.empty else None

    by_cross = {}
    for ct, grp in outcomes_df.groupby("crossover_type"):
        w = len(grp[grp["trade_result"] == "WIN"])
        by_cross[ct] = {
            "total"   : len(grp),
            "wins"    : w,
            "win_rate": round(w / len(grp) * 100, 1)
        }

    return {
        "total_trades"   : total,
        "wins"           : wins,
        "losses"         : losses,
        "win_rate_pct"   : win_rate,
        "targets_hit"    : targets,
        "sl_struct_hits" : sl_struct,
        "sl_ema_hits"    : sl_ema,
        "time_exits"     : time_ex,
        "avg_win"        : avg_win,
        "avg_loss"       : avg_loss,
        "risk_reward"    : rr,
        "ev_per_trade"   : ev,
        "composite_score": composite,
        "avg_return_8w"  : avg_8w,
        "best_return_8w" : best_8w,
        "worst_return_8w": worst_8w,
        "by_crossover_type": by_cross,
    }


# ================================================================
# FULL BACKTEST
# ================================================================

def run_momentum_backtest(symbol: str, df: pd.DataFrame) -> dict:
    """Runs complete momentum backtest for one symbol."""
    log.info(f"Running momentum backtest: {symbol}")

    signals = find_momentum_signals(df)

    if signals is None or signals.empty:
        return {}

    df_ema   = calculate_emas(df)
    outcomes = measure_outcomes(signals, df_ema)

    if outcomes.empty:
        return {"signals": signals}

    report = generate_report(outcomes)

    log.info(
        f"{symbol} — signals: {report.get('total_trades', 0)} | "
        f"WR: {report.get('win_rate_pct', 0)}% | "
        f"RR: {report.get('risk_reward', 'N/A')} | "
        f"EV: {report.get('ev_per_trade', 'N/A')}%"
    )

    return {
        "signals" : signals,
        "outcomes": outcomes,
        "report"  : report,
    }