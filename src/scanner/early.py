# ================================================================
# src/scanner/early.py
# ----------------------------------------------------------------
# Early Accumulation Scanner
#
# Finds stocks in EARLY accumulation phase — before the main
# scanner catches them. These are 1-6 month setups.
#
# Philosophy:
#   Active scanner needs volume confirmation NOW
#   Early scanner needs OBV signals ONLY
#   Best entries are BEFORE volume confirms
#
# Criteria (lower bar than active scanner):
#   MUST: OBV rising on daily OR weekly
#   MUST: Not a falling knife
#   MUST: Price not in freefall (some structure)
#
# Scoring (0-100):
#   OBV signals      → max 60 pts
#   Price structure  → max 25 pts
#   Context bonus    → max 15 pts
#
# Output tiers:
#   STRONG  >= 60  → both TF OBV + structure
#   WATCH   40-59  → single TF OBV + some structure
#   RADAR   20-39  → early OBV signs only
# ================================================================

import pandas as pd
import numpy as np
from datetime import date, timedelta
from src.data.store        import get_all_symbols, \
                                  load_daily_prices, \
                                  load_weekly_prices
from src.indicators.obv    import calculate_obv_indicators
from src.indicators.price  import calculate_price_indicators
from src.utils.logger      import get_logger

log = get_logger(__name__)

# ── Constants ──────────────────────────────────────────────────
MIN_PRICE      = 20.0
MIN_AVG_VOLUME = 50000
LOOKBACK_DAYS  = 365

ETF_KEYWORDS = [
    'LIQUID', 'GOLD', 'SILVER', 'NIFTY', 'SENSEX',
    'BANKBEES', 'JUNIORBEE', 'BEES', 'SETF', 'NETF',
    'GETF', 'IETF', 'BETA', 'HANG', 'MAFANG',
    'CPSEETF', 'MOM', 'QUAL', 'ALPHA', 'VALUE',
    'LOWVOL', 'GILT', 'GSEC', 'CASH', 'GROWW',
    'MON100', 'MOVALUE', 'DEFENCE', 'PHARMABEES',
    'INFRAIETF', 'AUTOIETF', 'MIDCAPIETF', 'EVIETF',
    'HEALTHIETF', 'PVTBANIETF', 'HDFCNIFBAN',
    'MASPTOP', 'SBILIQ', 'LIQUIDETF', 'LIQUIDBETF',
    'LIQUIDPLUS', 'LIQUIDADD', 'LIQUIDCASE',
    'LTGILT', 'SILVRETF', 'SILVERETF', 'ESILVER',
    'SILVERAG', 'GOLDSHARE', 'GOLDETF', 'BSLGOLD',
    'HDFCMID', 'HDFCSML', 'HDFCSENSEX', 'JUNIORBEES',
    'MID150', 'MIDSMALL', 'NEXT50', 'MOMENTUM30',
    'AXISVALUE', 'ALPHAETF', 'PVTBAN', 'TRUSTMF',
]


def _is_etf(symbol: str) -> bool:
    return any(kw in symbol.upper() for kw in ETF_KEYWORDS)


# ================================================================
# EARLY SCORE CALCULATION
# ================================================================

def calculate_early_score(
    obv:    dict,
    price:  dict,
    symbol: str = ""
) -> dict:
    """
    Calculates early accumulation score.
    Focused on OBV signals and price structure.
    Does not require volume events.
    """
    if not obv or not price:
        return _empty_early_score()

    flags   = obv.get("flags", {})
    score   = 0
    signals = []

    # ================================================================
    # OBV SIGNALS (max 60 pts)
    # ================================================================

    # Both timeframes diverging (most powerful — 25 pts)
    if flags.get("s12_both_timeframes"):
        score += 25
        signals.append("BOTH_TF_OBV_DIV")

    # Weekly OBV divergence (18 pts)
    elif flags.get("s11_weekly_divergence"):
        score += 18
        signals.append("WEEKLY_OBV_DIV")

    # Daily OBV divergence (12 pts)
    if flags.get("s6_price_flat_obv_rising"):
        score += 12
        signals.append("DAILY_OBV_DIV")

    # OBV new 50D high while price not (10 pts)
    if flags.get("s1_obv_new_high_price_not"):
        score += 10
        signals.append("OBV_LEADING_PRICE")

    # OBV sustained rise (8 pts)
    if flags.get("s2_obv_sustained_rise"):
        score += 8
        signals.append("OBV_SUSTAINED")

    # OBV 100D breakout (8 pts)
    if flags.get("s4_obv_100d_breakout"):
        score += 8
        signals.append("OBV_100D_BREAKOUT")

    # OBV slope accelerating (5 pts)
    if flags.get("s5_obv_slope_accel"):
        score += 5
        signals.append("OBV_ACCEL")

    # OBV higher highs (4 pts)
    if flags.get("s13_obv_higher_highs"):
        score += 4
        signals.append("OBV_HH")

    # Shakeout detected (5 pts)
    if flags.get("s10_shakeout_detected"):
        score += 5
        signals.append("SHAKEOUT")

    # Pullback divergence (up to 15 pts based on uptrend context)
    if flags.get("s14_pullback_divergence"):
        s14_pts  = obv.get("scores", {}).get(
            "s14_pullback_divergence", 0
        )
        depth    = flags.get("s14_pullback_depth_pct", 0)
        ut_conds = flags.get("s14_uptrend_conditions", {})
        passing  = [k for k, v in ut_conds.items() if v]
        signal   = f"PULLBACK_DIV_{int(depth)}PCT"
        if passing:
            signal += f"_UT({''.join(passing)})"
        score   += s14_pts
        signals.append(signal)

    # Cap OBV section at 60
    score = min(score, 60)

    # ================================================================
    # PRICE STRUCTURE (max 25 pts)
    # ================================================================

    # Higher lows (10 pts)
    if price.get("scenario_higher_lows"):
        score += 10
        signals.append("HIGHER_LOWS")

    # Consolidating (8 pts)
    if price.get("scenario_consolidating"):
        score += 8
        signals.append("CONSOLIDATING")

    # Near 20 EMA (7 pts)
    if price.get("scenario_near_20ema"):
        score += 7
        signals.append("NEAR_20EMA")

    # ================================================================
    # CONTEXT BONUS (max 15 pts)
    # ================================================================

    pct_from_high = price.get("pct_from_52w_high") or 0
    chg_3m        = price.get("chg_3m") or 0

    # Deep correction 30-60% (8 pts)
    if -60.0 <= pct_from_high <= -30.0:
        score += 8
        signals.append(f"DEEP_CORR_{abs(int(pct_from_high))}PCT")

    # Very deep correction 60%+ (5 pts)
    elif pct_from_high < -60.0:
        score += 5
        signals.append(f"VERY_DEEP_{abs(int(pct_from_high))}PCT")

    # Price stabilising (4 pts)
    if -10.0 <= chg_3m <= 10.0:
        score += 4
        signals.append("PRICE_STABILISING")

    # Multi year base (3 pts)
    if price.get("scenario_multi_year_base"):
        score += 3
        signals.append("MULTI_YEAR_BASE")

    # ── Conviction tier ────────────────────────────────────────
    if score >= 60:
        tier = "STRONG"
    elif score >= 40:
        tier = "WATCH"
    elif score >= 20:
        tier = "RADAR"
    else:
        tier = "SKIP"

    return {
        "early_score"  : score,
        "early_tier"   : tier,
        "early_signals": signals,
    }


def _empty_early_score() -> dict:
    return {
        "early_score"  : 0,
        "early_tier"   : "SKIP",
        "early_signals": [],
    }


# ================================================================
# MAIN EARLY SCANNER
# ================================================================

def run_early_scanner(
    scan_date: date = None,
    min_score: int  = 20,
) -> pd.DataFrame:
    """
    Runs early accumulation scanner across all NSE stocks.
    """
    if scan_date is None:
        scan_date = date.today()

    from_date = scan_date - timedelta(days=LOOKBACK_DAYS)

    log.info(f"Early scanner starting — {scan_date}")

    symbols = get_all_symbols()
    results = []
    blocked = []
    errors  = 0

    for i, symbol in enumerate(symbols, 1):

        if i % 200 == 0:
            log.info(
                f"Progress: {i}/{len(symbols)} | "
                f"found: {len(results)}"
            )

        try:
            # ── ETF filter ─────────────────────────────────────
            if _is_etf(symbol):
                continue

            # ── Load daily data ────────────────────────────────
            df_daily = load_daily_prices(
                symbol,
                from_date=from_date,
                to_date=scan_date
            )

            if df_daily is None or len(df_daily) < 30:
                continue

            latest_close = float(df_daily["close"].iloc[-1])
            avg_vol      = float(
                df_daily["volume"].tail(22).mean()
            )

            # ── Basic filters ──────────────────────────────────
            if latest_close < MIN_PRICE:
                continue

            if avg_vol < MIN_AVG_VOLUME:
                continue

            # ── Price indicators ───────────────────────────────
            price    = calculate_price_indicators(df_daily)
            range_4w = price.get("range_pct_4w") or 0

            # Skip ETF-like tight range
            if range_4w < 2.0:
                continue

            # Skip extremely wide range
            if range_4w > 35.0:
                continue

            # ── Delivery filter ────────────────────────────────
            # Need at least 5 days with delivery >= 40%
            # in last 22 days
            if "delivery_pct" in df_daily.columns:
                last_22_deliv   = df_daily["delivery_pct"].tail(22)
                valid_deliv     = last_22_deliv.dropna()
                if len(valid_deliv) >= 10:
                    high_deliv_days = int(
                        (valid_deliv >= 40.0).sum()
                    )
                    if high_deliv_days < 5:
                        continue

                # Also check today's delivery — skip if < 15%
                # Very low today = operator driven today
                today_deliv = df_daily["delivery_pct"].iloc[-1]
                if pd.notna(today_deliv) and float(today_deliv) < 15.0:
                    continue

            # ── Load weekly data ───────────────────────────────
            df_weekly = load_weekly_prices(symbol)

            # ── OBV indicators ─────────────────────────────────
            obv = calculate_obv_indicators(df_daily, df_weekly)

            # Block genuine falling knives only
            if not obv.get("tier1_passed", True):
                reason = obv.get("tier1_reason", "")
                if "falling_knife" in str(reason):
                    blocked.append({
                        "symbol": symbol,
                        "reason": reason
                    })
                    continue

            # ── Must have at least one OBV signal ─────────────
            flags = obv.get("flags", {})
            has_obv_signal = any([
                flags.get("s6_price_flat_obv_rising"),
                flags.get("s11_weekly_divergence"),
                flags.get("s12_both_timeframes"),
                flags.get("s1_obv_new_high_price_not"),
                flags.get("s2_obv_sustained_rise"),
                flags.get("s4_obv_100d_breakout"),
                flags.get("s13_obv_higher_highs"),
                flags.get("s14_pullback_divergence"),
            ])

            if not has_obv_signal:
                continue

            # ── Calculate early score ──────────────────────────
            early = calculate_early_score(obv, price, symbol)

            if early["early_score"] < min_score:
                continue

            # ── Delivery for display ───────────────────────────
            deliv_today = None
            if "delivery_pct" in df_daily.columns:
                d = df_daily["delivery_pct"].iloc[-1]
                if pd.notna(d):
                    deliv_today = round(float(d), 1)

            results.append({
                "symbol"           : symbol,
                "scan_date"        : scan_date,
                "early_score"      : early["early_score"],
                "early_tier"       : early["early_tier"],
                "early_signals"    : early["early_signals"],
                "close"            : latest_close,
                "chg_1d"           : price.get("chg_1d"),
                "chg_1w"           : price.get("chg_1w"),
                "chg_1m"           : price.get("chg_1m"),
                "chg_3m"           : price.get("chg_3m"),
                "pct_from_52w_high": price.get("pct_from_52w_high"),
                "range_pct_4w"     : price.get("range_pct_4w"),
                "obv_score"        : obv.get("total_score"),
                "obv_rising_20d"   : obv.get("obv_rising_20d"),
                "weekly_div"       : flags.get("s11_weekly_divergence"),
                "both_tf_div"      : flags.get("s12_both_timeframes"),
                "daily_div"        : flags.get("s6_price_flat_obv_rising"),
                "shakeout"         : flags.get("s10_shakeout_detected"),
                "pullback_div"     : flags.get("s14_pullback_divergence"),
                "pullback_depth"   : flags.get("s14_pullback_depth_pct"),
                "uptrend_score"    : flags.get("s14_uptrend_score"),
                "avg_vol"          : round(avg_vol),
                "delivery_pct"     : deliv_today,
            })

        except Exception as e:
            log.error(f"Error on {symbol}: {e}")
            errors += 1
            continue

    log.info(
        f"Early scanner complete — "
        f"found:{len(results)} | "
        f"blocked:{len(blocked)} | "
        f"errors:{errors}"
    )

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)
    df = df.sort_values(
        "early_score", ascending=False
    ).reset_index(drop=True)
    df.index += 1

    strong = len(df[df["early_tier"] == "STRONG"])
    watch  = len(df[df["early_tier"] == "WATCH"])
    radar  = len(df[df["early_tier"] == "RADAR"])
    log.info(f"STRONG:{strong} WATCH:{watch} RADAR:{radar}")

    return df