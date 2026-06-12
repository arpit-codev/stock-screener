# ================================================================
# src/scanner/active.py
# ================================================================
# Active Scanner — runs every evening after market close.
#
# Pipeline for each stock:
#   1. Load last 365 days daily + all weekly prices
#   2. Run volume indicators
#   3. Run OBV indicators (with Tier 1 filter)
#   4. Run price indicators
#   5. Score with unified conviction engine
#   6. Return ranked DataFrame — all tiers shown
#
# Usage:
#   from src.scanner.active import run_scanner
#   results = run_scanner()
# ================================================================

import pandas as pd
import numpy as np
from datetime import date, timedelta
from src.data.store       import get_all_symbols, \
                                 load_daily_prices, \
                                 load_weekly_prices
from src.indicators.volume import calculate_volume_indicators
from src.indicators.obv    import calculate_obv_indicators
from src.indicators.price  import calculate_price_indicators
from src.scanner.scoring   import calculate_conviction_score
from src.utils.logger      import get_logger

log = get_logger(__name__)

# ── Scanner constants ──────────────────────────────────────────
MIN_PRICE         = 20.0     # skip penny stocks
MIN_AVG_VOLUME    = 100000   # skip illiquid stocks
LOOKBACK_DAYS     = 365      # 1 year of daily data

ETF_KEYWORDS = [
    'LIQUID', 'GOLD', 'SILVER', 'NIFTY', 'SENSEX',
    'BANKBEES', 'JUNIORBEE', 'BEES', 'SETF', 'NETF',
    'GETF', 'IETF', 'BETA', 'HANG', 'MAFANG',
    'CPSEETF', 'MOM', 'QUAL', 'ALPHA', 'VALUE',
    'LOWVOL', 'GILT', 'GSEC', 'CASH', 'GROWW',
    'MON100', 'MOVALUE', 'DEFENCE', 'PHARMABEES',
    'MASPTOP50', 'SBILIQETF',
    'INFRAIETF', 'AUTOIETF', 'MIDCAPIETF', 'EVIETF',
    'HEALTHIETF', 'PVTBANIETF', 'HDFCNIFBAN',
]

def _is_etf(symbol: str) -> bool:
    s = symbol.upper()
    return any(kw in s for kw in ETF_KEYWORDS)
# ================================================================
# MAIN SCANNER
# ================================================================

def run_scanner(
    scan_date: date = None,
    min_score: int  = 0,
) -> pd.DataFrame:
    """
    Runs the full active scanner across all NSE stocks.

    Parameters
    ----------
    scan_date : date, optional
        Date to scan for. Defaults to today.
    min_score : int, optional
        Minimum score to include in results. Default 0 (all).

    Returns
    -------
    pd.DataFrame
        All qualifying stocks ranked by conviction score.
        Columns:
            symbol, final_score, conviction_tier,
            volume_score, obv_score, price_score,
            bonus_pts, red_flag_pts,
            close, chg_1d, chg_1w, chg_1m,
            delivery_pct, delivery_tier,
            obv_divergence, weekly_divergence,
            absorption, absorption_tier, absorption_date,
            shakeout, bonuses_fired, red_flags_fired,
            obv_tier, blocked_reason
    """
    if scan_date is None:
        scan_date = date.today()

    from_date = scan_date - timedelta(days=LOOKBACK_DAYS)

    log.info(f"Scanner starting — date: {scan_date}")

    symbols = get_all_symbols()
    log.info(f"Total symbols to scan: {len(symbols)}")

    results      = []
    blocked      = []
    processed    = 0
    errors       = 0

    for i, symbol in enumerate(symbols, 1):

        if i % 200 == 0:
            log.info(
                f"Progress: {i}/{len(symbols)} | "
                f"passed: {len(results)} | "
                f"blocked: {len(blocked)}"
            )

        try:
            # ── Load data ──────────────────────────────────────
            df_daily = load_daily_prices(
                symbol,
                from_date=from_date,
                to_date=scan_date
            )

            if df_daily is None or len(df_daily) < 22:
                blocked.append({
                    "symbol": symbol,
                    "reason": "insufficient_data"
                })
                continue

            # Basic liquidity filter before heavy computation
            latest_close = float(df_daily["close"].iloc[-1])
            avg_vol      = float(df_daily["volume"].tail(22).mean())

            if latest_close < MIN_PRICE:
                blocked.append({
                    "symbol": symbol,
                    "reason": f"low_price_{latest_close:.1f}"
                })
                continue

            if avg_vol < MIN_AVG_VOLUME:
                blocked.append({
                    "symbol": symbol,
                    "reason": "low_volume"
                })
                continue

            # Skip ETFs and liquid funds
            if _is_etf(symbol):
                blocked.append({
                    "symbol": symbol,
                    "reason": "etf"
                })
                continue

            # Load weekly prices
            df_weekly = load_weekly_prices(symbol)

            # ── Run indicators ─────────────────────────────────
            vol = calculate_volume_indicators(df_daily)
            obv = calculate_obv_indicators(df_daily, df_weekly)
            price = calculate_price_indicators(df_daily)

            # Skip low delivery stocks — intraday operators not institutions
            deliv = vol.get("delivery_pct_today")
            if deliv is not None and float(deliv) < 25.0:
                blocked.append({
                    "symbol": symbol,
                    "reason": "low_delivery"
                })
                continue

            # ── OBV Tier 1 filter ──────────────────────────────
            if not obv.get("tier1_passed", True):
                blocked.append({
                    "symbol": symbol,
                    "reason": obv.get("tier1_reason", "obv_filter")
                })
                continue

            # ── Unified scoring ────────────────────────────────
            score = calculate_conviction_score(
                vol, obv, price, symbol
            )

            final_score = score.get("final_score", 0)

            if final_score < min_score:
                continue

            # ── Build result row ───────────────────────────────
            row = {
                # Identity
                "symbol"           : symbol,
                "scan_date"        : scan_date,

                # Scores
                "final_score"      : final_score,
                "conviction_tier"  : score.get("conviction_tier"),
                "volume_score"     : score.get("volume_score", 0),
                "obv_score"        : score.get("obv_score", 0),
                "price_score"      : score.get("price_score", 0),
                "base_score"       : score.get("base_score", 0),
                "bonus_pts"        : score.get("bonus_pts", 0),
                "red_flag_pts"     : score.get("red_flag_pts", 0),

                # Price context
                "close"            : latest_close,
                "chg_1d"           : price.get("chg_1d"),
                "chg_1w"           : price.get("chg_1w"),
                "chg_1m"           : price.get("chg_1m"),
                "chg_3m"           : price.get("chg_3m"),
                "pct_from_52w_high": price.get("pct_from_52w_high"),
                "range_pct_4w"     : price.get("range_pct_4w"),

                # Volume context
                "vol_today"        : vol.get("vol_today"),
                "vol_22d_avg"      : vol.get("vol_22d_avg"),
                "day_vol_ratio"    : vol.get("day_vol_ratio"),
                "delivery_pct"     : vol.get("delivery_pct_today"),
                "deliv_22d_avg"    : vol.get("deliv_22d_avg"),

                # Key signals
                "obv_divergence"   : score.get("obv_divergence", False),
                "weekly_divergence": score.get("weekly_divergence", False),
                "absorption"       : score.get("absorption", False),
                "absorption_tier"  : vol.get("absorption_tier"),
                "absorption_date"  : vol.get("absorption_date"),
                "shakeout"         : score.get("shakeout", False),

                # OBV details
                "obv_tier"         : score.get("obv_tier"),
                "obv_chg_22d"      : obv.get("daily_obv_chg_22d"),

                # Bonuses and flags
                "bonuses_fired"    : score.get("bonuses_fired", []),
                "red_flags_fired"  : score.get("red_flags_fired", []),

                # Price scenarios
                "consolidating"    : price.get("scenario_consolidating", False),
                "higher_lows"      : price.get("scenario_higher_lows", False),
                "near_20ema"       : price.get("scenario_near_20ema", False),
                "multi_year_base"  : price.get("scenario_multi_year_base", False),
            }

            results.append(row)
            processed += 1

        except Exception as e:
            log.error(f"Error processing {symbol}: {e}")
            errors += 1
            continue

    log.info(
        f"Scanner complete — "
        f"processed: {processed} | "
        f"blocked: {len(blocked)} | "
        f"errors: {errors}"
    )

    if not results:
        log.warning("No stocks passed scanner filters")
        return pd.DataFrame()

    # ── Build ranked DataFrame ─────────────────────────────────
    df_results = pd.DataFrame(results)

    # Sort by final score descending
    df_results = df_results.sort_values(
        "final_score",
        ascending=False
    ).reset_index(drop=True)
    df_results.index += 1

    # ── Summary stats ──────────────────────────────────────────
    _log_summary(df_results, blocked)

    return df_results


# ================================================================
# SUMMARY LOGGING
# ================================================================

def _log_summary(
    df: pd.DataFrame,
    blocked: list
) -> None:
    """Logs scanner summary statistics."""

    total      = len(df)
    high_tier  = len(df[df["conviction_tier"] == "HIGH"])
    medium_tier= len(df[df["conviction_tier"] == "MEDIUM"])
    low_tier   = len(df[df["conviction_tier"] == "LOW"])
    skip_tier  = len(df[df["conviction_tier"] == "SKIP"])

    log.info(f"{'='*50}")
    log.info(f"SCANNER RESULTS SUMMARY")
    log.info(f"{'='*50}")
    log.info(f"Total scored      : {total}")
    log.info(f"HIGH  (>=70)      : {high_tier}")
    log.info(f"MEDIUM (50-69)    : {medium_tier}")
    log.info(f"LOW   (30-49)     : {low_tier}")
    log.info(f"SKIP  (<30)       : {skip_tier}")
    log.info(f"Blocked           : {len(blocked)}")

    # Block reason breakdown
    if blocked:
        reasons = {}
        for b in blocked:
            r = b["reason"].split("_")[0] \
                if "_" in b["reason"] else b["reason"]
            reasons[r] = reasons.get(r, 0) + 1
        log.info(f"Block reasons     : {reasons}")

    log.info(f"{'='*50}")