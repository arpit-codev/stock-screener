# ================================================================
# src/indicators/yearly_open.py
# ----------------------------------------------------------------
# Yearly Open Strategy — Full Backtester
#
# Entry:  Intraday touch OR false breakdown (close below + recovered)
#         NO entry on genuine close below (no recovery in 3 days)
#
# SL:     5% below yearly open (price SL)
#         OR 3 consecutive meaningfully lower closes below yearly open
#         OR 8 weeks elapsed — whichever comes first
#
# Target: 10% from entry
#         OR 8 weeks elapsed — whichever comes first
#
# Tracks same year AND subsequent year tests.
# Tolerance band: 0.5% around yearly open level.
# ================================================================

import pandas as pd
import numpy as np
from datetime import date
from src.utils.logger import get_logger

log = get_logger(__name__)

# ── Strategy Constants ─────────────────────────────────────────
TOUCH_TOLERANCE_PCT = 0.5    # within 0.5% = touching the level
SL_PCT              = 5.0    # 5% below yearly open = price SL
STRUCTURE_SL_DAYS   = 3      # 3 consecutive lower lows = structure SL
TARGET_PCT          = 10.0   # 10% from entry = target
FORWARD_WEEKS       = 8      # measure hold return over 8 weeks
RECOVERY_DAYS       = 3      # days to recover from close below


# ================================================================
# STEP 1 — BUILD YEARLY OPENS
# ================================================================

def build_yearly_opens(df: pd.DataFrame) -> pd.DataFrame:
    """
    Finds yearly open price for each year in the data.
    Only includes years where the first trade date is in January.
    This ensures we only use genuine yearly opens, not listing dates.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year

    rows = []
    for year, year_df in df.groupby("year"):
        year_df   = year_df.sort_values("date")
        first_row = year_df.iloc[0]
        first_date = first_row["date"]

        # Only include if first trade is in January
        # Stocks listed mid-year don't have genuine yearly opens
        if first_date.month != 1:
            log.debug(
                f"Skipping {first_row['symbol']} year {year} — "
                f"first trade {first_date.date()} is not January"
            )
            continue

        rows.append({
            "symbol"          : first_row["symbol"],
            "year"            : int(year),
            "yearly_open"     : float(first_row["open"]),
            "first_trade_date": first_date.date(),
        })

    return pd.DataFrame(rows)


# ================================================================
# STEP 2 — FIND TESTS + ENTRY
# ================================================================

def find_yearly_open_tests(
    df: pd.DataFrame,
    yearly_opens_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Finds every valid test of yearly open level.

    Entry rules:
      intraday_touch  → low touched yearly open, close stayed above
                        entry = next day open
      false_breakdown → closed below but recovered within 3 days
                        entry = next day open
      close_below     → NO ENTRY — price closed below and stayed below
                        skip completely
    """
    if df is None or df.empty or yearly_opens_df.empty:
        return pd.DataFrame()

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    all_tests = []

    for _, level_row in yearly_opens_df.iterrows():
        year        = level_row["year"]
        yearly_open = float(level_row["yearly_open"])
        symbol      = level_row["symbol"]

        # Bands
        upper_band = yearly_open * (1 + TOUCH_TOLERANCE_PCT / 100)
        sl_level   = yearly_open * (1 - SL_PCT / 100)

        # Scan from day after yearly open date
        start_date = pd.Timestamp(level_row["first_trade_date"])
        scan_df    = df[df["date"] > start_date].reset_index(drop=True)

        if len(scan_df) < 2:
            continue

        # Volume baseline for ratio
        vol_22d_avg = df[
            df["date"] <= start_date
        ].tail(22)["volume"].mean()

        test_number = 0
        skip_until  = None

        for i in range(len(scan_df) - 1):
            row      = scan_df.iloc[i]
            next_row = scan_df.iloc[i + 1]
            row_date = row["date"]

            if skip_until and row_date <= skip_until:
                continue

            low_price   = float(row["low"])
            close_price = float(row["close"])
            volume      = float(row["volume"])
            delivery    = row.get("delivery_pct", np.nan)

            # ── Detect test ────────────────────────────────────
            intraday_touch = (
                low_price <= upper_band and
                low_price >= sl_level and      # not already in SL zone
                close_price > yearly_open      # closed above — held
            )

            closed_below = (
                close_price < yearly_open and
                close_price >= sl_level        # not already in SL zone
            )

            if not (intraday_touch or closed_below):
                continue

            # ── Classify test type ─────────────────────────────
            if intraday_touch and not closed_below:
                test_type = "intraday_touch"

            elif closed_below:
                # Check if recovered within RECOVERY_DAYS
                recovery_window = scan_df[
                    scan_df["date"] > row_date
                ].head(RECOVERY_DAYS)

                recovered = any(
                    float(r["close"]) > yearly_open
                    for _, r in recovery_window.iterrows()
                )

                if recovered:
                    test_type = "false_breakdown"
                else:
                    # Genuine breakdown — NO ENTRY
                    log.debug(
                        f"{symbol} {row_date.date()} — "
                        f"genuine close_below, no recovery — skipping"
                    )
                    skip_until = row_date + pd.Timedelta(days=7)
                    continue
            else:
                continue

            # ── Entry = next day open ──────────────────────────
            entry_price = float(next_row["open"])

            # ── Volume ratio ───────────────────────────────────
            vol_ratio = round(volume / vol_22d_avg, 2) \
                if vol_22d_avg > 0 else None

            test_number += 1

            all_tests.append({
                "symbol"           : symbol,
                "year"             : year,
                "yearly_open"      : yearly_open,
                "sl_level"         : round(sl_level, 2),
                "test_date"        : row_date.date(),
                "entry_date"       : next_row["date"].date(),
                "test_number"      : test_number,
                "test_type"        : test_type,
                "close_at_test"    : round(close_price, 2),
                "low_at_test"      : round(low_price, 2),
                "entry_price"      : round(entry_price, 2),
                "volume_at_test"   : int(volume),
                "delivery_at_test" : round(float(delivery), 2)
                                     if pd.notna(delivery) else None,
                "vol_ratio_at_test": vol_ratio,
            })

            # Cooldown — avoid duplicate signals
            skip_until = row_date + pd.Timedelta(days=7)

    return pd.DataFrame(all_tests)


# ================================================================
# STEP 3 — MEASURE OUTCOMES — UNIFIED EXIT
# ================================================================

def measure_outcomes(
    tests_df: pd.DataFrame,
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Unified exit — first of these triggers:

    1. TARGET      → intraday high hits entry * 1.10
    2. SL-A        → daily close hits yearly_open * 0.95
    3. SL-B        → 3 consecutive meaningfully lower closes
                     below yearly open (lower lows)
    4. TIME EXIT   → 8 weeks (40 trading days) elapsed
                     exit at close of day 40
    """
    if tests_df is None or tests_df.empty:
        return pd.DataFrame()

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    results = []

    for _, test in tests_df.iterrows():
        entry_date   = pd.Timestamp(test["entry_date"])
        entry_price  = float(test["entry_price"])
        yearly_open  = float(test["yearly_open"])
        sl_level     = float(test["sl_level"])
        target_price = entry_price * (1 + TARGET_PCT / 100)

        # Forward window — 40 trading days = 8 weeks
        forward_df = df[
            df["date"] > entry_date
        ].head(40).reset_index(drop=True)

        if forward_df.empty:
            continue

        # ── Day by day unified scan ────────────────────────────
        exit_reason            = None
        exit_date              = None
        exit_price             = None
        exit_return_pct        = None
        days_to_exit           = None
        sl_hit_price           = False
        sl_hit_structure       = False
        target_hit             = False
        time_exit              = False

        consecutive_lower_lows = 0
        prev_lower_low_close   = None

        for j, row in forward_df.iterrows():
            high        = float(row["high"])
            low         = float(row["low"])
            close       = float(row["close"])
            is_last_day = (j == len(forward_df) - 1)

            # ── CHECK 1: TARGET ───────────────────────────────
            # Intraday high reaches 10% above entry
            if high >= target_price:
                target_hit      = True
                exit_reason     = "target_10pct"
                exit_date       = row["date"].date()
                exit_price      = target_price
                exit_return_pct = TARGET_PCT
                days_to_exit    = j + 1
                break

            # ── CHECK 2: SL-A — Price SL ─────────────────────
            # Close goes 5% below yearly open level
            if close <= sl_level:
                sl_hit_price    = True
                exit_reason     = "sl_price_5pct"
                exit_date       = row["date"].date()
                exit_price      = close
                exit_return_pct = round(
                    (close - entry_price) / entry_price * 100, 2
                )
                days_to_exit    = j + 1
                break

            # ── CHECK 3: SL-B — Structure SL ─────────────────
            # 3 consecutive meaningfully lower closes below
            # yearly open — sellers in control
            if close < yearly_open:
                if (prev_lower_low_close is not None and
                        close < prev_lower_low_close * 0.999):
                    # Meaningfully lower close — increment counter
                    consecutive_lower_lows += 1
                else:
                    # Not a lower low — reset counter
                    consecutive_lower_lows = 1
                    prev_lower_low_close   = close

                if consecutive_lower_lows >= STRUCTURE_SL_DAYS:
                    sl_hit_structure = True
                    exit_reason      = "sl_structure_3ll"
                    exit_date        = row["date"].date()
                    exit_price       = close
                    exit_return_pct  = round(
                        (close - entry_price) / entry_price * 100, 2
                    )
                    days_to_exit     = j + 1
                    break
            else:
                # Price back above yearly open — reset structure SL
                consecutive_lower_lows = 0
                prev_lower_low_close   = None

            # ── CHECK 4: TIME EXIT ────────────────────────────
            # 8 weeks elapsed — exit at close regardless
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
        if target_hit:
            trade_result = "WIN"
        elif sl_hit_price or sl_hit_structure:
            trade_result = "LOSS"
        elif time_exit:
            trade_result = "WIN" if exit_return_pct >= 0 else "LOSS"
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
        max_gain_8w = round(
            (float(forward_df["high"].max()) - entry_price)
            / entry_price * 100, 2
        )
        max_dd_8w = round(
            (float(forward_df["low"].min()) - entry_price)
            / entry_price * 100, 2
        )

        result = test.to_dict()
        result.update({
            "target_price"    : round(target_price, 2),
            "exit_reason"     : exit_reason,
            "exit_date"       : exit_date,
            "exit_price"      : round(exit_price, 2)
                                if exit_price else None,
            "exit_return_pct" : exit_return_pct,
            "days_to_exit"    : days_to_exit,
            "trade_result"    : trade_result,
            "sl_hit_price"    : sl_hit_price,
            "sl_hit_structure": sl_hit_structure,
            "target_hit"      : target_hit,
            "time_exit"       : time_exit,
            "return_1w"       : get_weekly_return(1),
            "return_2w"       : get_weekly_return(2),
            "return_4w"       : get_weekly_return(4),
            "return_8w"       : get_weekly_return(8),
            "max_gain_8w"     : max_gain_8w,
            "max_drawdown_8w" : max_dd_8w,
        })
        results.append(result)

    return pd.DataFrame(results)


# ================================================================
# STEP 4 — SYMBOL REPORT
# ================================================================

def generate_symbol_report(outcomes_df: pd.DataFrame) -> dict:
    """Generates statistics for one symbol."""
    if outcomes_df is None or outcomes_df.empty:
        return {}

    total     = len(outcomes_df)
    wins      = len(outcomes_df[outcomes_df["trade_result"] == "WIN"])
    losses    = len(outcomes_df[outcomes_df["trade_result"] == "LOSS"])
    sl_price  = int(outcomes_df["sl_hit_price"].sum())
    sl_struct = int(outcomes_df["sl_hit_structure"].sum())
    win_rate  = round(wins / total * 100, 1) if total > 0 else 0

    # Return stats — 8 week hold
    valid_8w  = outcomes_df["return_8w"].dropna()
    avg_8w    = round(valid_8w.mean(),  2) if not valid_8w.empty else None
    best_8w   = round(valid_8w.max(),   2) if not valid_8w.empty else None
    worst_8w  = round(valid_8w.min(),   2) if not valid_8w.empty else None

    # Actual trade returns
    win_returns  = outcomes_df[
        outcomes_df["trade_result"] == "WIN"
    ]["exit_return_pct"].dropna()

    loss_returns = outcomes_df[
        outcomes_df["trade_result"] == "LOSS"
    ]["exit_return_pct"].dropna()

    avg_win  = round(win_returns.mean(),  2) if not win_returns.empty  else None
    avg_loss = round(loss_returns.mean(), 2) if not loss_returns.empty else None

    rr_ratio = round(
        abs(avg_win / avg_loss), 2
    ) if avg_win and avg_loss and avg_loss != 0 else None

    # EV and composite score
    loss_rate = 100 - win_rate
    ev = round(
        (win_rate  / 100 * (avg_win  or 0)) +
        (loss_rate / 100 * (avg_loss or 0)),
        2
    )
    composite = round(ev * rr_ratio, 2) if rr_ratio else None

    # First touch vs later
    first = outcomes_df[outcomes_df["test_number"] == 1]
    later = outcomes_df[outcomes_df["test_number"] > 1]

    first_wr = round(
        len(first[first["trade_result"] == "WIN"]) / len(first) * 100, 1
    ) if not first.empty else None

    later_wr = round(
        len(later[later["trade_result"] == "WIN"]) / len(later) * 100, 1
    ) if not later.empty else None

    # High delivery tests
    high_d = outcomes_df[
        outcomes_df["delivery_at_test"].notna() &
        (outcomes_df["delivery_at_test"] >= 50)
    ]
    high_d_wr = round(
        len(high_d[high_d["trade_result"] == "WIN"]) / len(high_d) * 100, 1
    ) if not high_d.empty else None

    # By test type
    by_type = {}
    for ttype, grp in outcomes_df.groupby("test_type"):
        w = len(grp[grp["trade_result"] == "WIN"])
        by_type[ttype] = {
            "total"   : len(grp),
            "wins"    : w,
            "win_rate": round(w / len(grp) * 100, 1)
        }

    return {
        "total_tests"    : total,
        "wins"           : wins,
        "losses"         : losses,
        "win_rate_pct"   : win_rate,
        "sl_price_hits"  : sl_price,
        "sl_struct_hits" : sl_struct,
        "avg_return_8w"  : avg_8w,
        "best_return_8w" : best_8w,
        "worst_return_8w": worst_8w,
        "avg_win"        : avg_win,
        "avg_loss"       : avg_loss,
        "risk_reward"    : rr_ratio,
        "ev_per_trade"   : ev,
        "composite_score": composite,
        "first_touch_wr" : first_wr,
        "later_touch_wr" : later_wr,
        "high_deliv_wr"  : high_d_wr,
        "by_test_type"   : by_type,
    }


# ================================================================
# STEP 5 — FULL ANALYSIS FOR ONE SYMBOL
# ================================================================

def analyse_symbol(symbol: str, df: pd.DataFrame) -> dict:
    """
    Runs complete yearly open analysis for one symbol.
    Chains all steps together.
    """
    log.info(f"Analysing {symbol}")

    yearly_opens = build_yearly_opens(df)
    if yearly_opens.empty:
        return {}

    tests = find_yearly_open_tests(df, yearly_opens)
    if tests.empty:
        return {"yearly_opens": yearly_opens, "tests": pd.DataFrame()}

    outcomes = measure_outcomes(tests, df)
    report   = generate_symbol_report(outcomes)

    log.info(
        f"{symbol} — tests: {report.get('total_tests', 0)} | "
        f"WR: {report.get('win_rate_pct', 0)}% | "
        f"RR: {report.get('risk_reward', 'N/A')} | "
        f"EV: {report.get('ev_per_trade', 'N/A')}%"
    )

    return {
        "yearly_opens" : yearly_opens,
        "tests"        : tests,
        "outcomes"     : outcomes,
        "report"       : report,
    }