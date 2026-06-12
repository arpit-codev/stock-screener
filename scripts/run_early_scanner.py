# ================================================================
# scripts/run_early_scanner.py
# ----------------------------------------------------------------
# Early Accumulation Scanner — finds pre-signal stocks
#
# Usage:
#   python scripts/run_early_scanner.py
#   python scripts/run_early_scanner.py --symbol TARIL
#   python scripts/run_early_scanner.py --tier STRONG
#   python scripts/run_early_scanner.py --min-score 40
# ================================================================

import sys
import argparse
from datetime import date, datetime, timedelta

sys.path.insert(0, '.')

import pandas as pd
from src.scanner.early import run_early_scanner, \
                               calculate_early_score
from src.utils.logger  import get_logger

log = get_logger("early_scanner")


# ================================================================
# SINGLE STOCK CHECK
# ================================================================

def run_single_stock(symbol: str, scan_date: date = None):
    """Early accumulation analysis for one stock."""
    from src.data.store     import load_daily_prices, \
                                   load_weekly_prices
    from src.indicators.obv import calculate_obv_indicators
    from src.indicators.price import calculate_price_indicators

    if scan_date is None:
        scan_date = date.today()

    from_date = scan_date - timedelta(days=365)

    df_d = load_daily_prices(symbol,
        from_date=from_date,
        to_date=scan_date)
    df_w = load_weekly_prices(symbol)

    if df_d is None or df_d.empty:
        print(f"\nNo data for {symbol}")
        return

    obv   = calculate_obv_indicators(df_d, df_w)
    price = calculate_price_indicators(df_d)
    early = calculate_early_score(obv, price, symbol)
    flags = obv.get("flags", {})

    print(f"\n{'='*60}")
    print(f"  EARLY SCANNER — {symbol}")
    print(f"  Date: {scan_date}")
    print(f"{'='*60}")

    print(f"\n  Early Score    : {early['early_score']}/100")
    print(f"  Early Tier     : {early['early_tier']}")
    print(f"  OBV Score      : {obv.get('total_score')}/100")
    print(f"  OBV Tier       : {obv.get('conviction_tier')}")

    print(f"\n{'─'*60}")
    print(f"  PRICE CONTEXT")
    print(f"{'─'*60}")
    print(f"  Close          : ₹{price.get('close')}")
    print(f"  Chg 1M         : {price.get('chg_1m')}%")
    print(f"  Chg 3M         : {price.get('chg_3m')}%")
    print(f"  From 52W High  : {price.get('pct_from_52w_high')}%")
    print(f"  Range 4W       : {price.get('range_pct_4w')}%")
    print(f"  Higher Lows    : {price.get('scenario_higher_lows')}")
    print(f"  Consolidating  : {price.get('scenario_consolidating')}")
    print(f"  Near 20 EMA    : {price.get('scenario_near_20ema')}")

    print(f"\n{'─'*60}")
    print(f"  OBV SIGNALS")
    print(f"{'─'*60}")
    print(f"  OBV Rising 20D : {obv.get('obv_rising_20d')}")
    print(f"  Price Chg 20D  : {obv.get('price_chg_20d')}%")
    print(f"  Daily Div      : {flags.get('s6_price_flat_obv_rising')}")
    print(f"  Weekly Div     : {flags.get('s11_weekly_divergence')}")
    print(f"  Both TF Div    : {flags.get('s12_both_timeframes')}")
    print(f"  OBV New High   : {flags.get('s1_obv_new_high_price_not')}")
    print(f"  OBV Sustained  : {flags.get('s2_obv_sustained_rise')}")
    print(f"  OBV Accel      : {flags.get('s5_obv_slope_accel')}")
    print(f"  Shakeout       : {flags.get('s10_shakeout_detected')}")
    print(f"  Pullback Div   : {flags.get('s14_pullback_divergence')}")

    print(f"\n{'─'*60}")
    print(f"  SIGNALS FIRED")
    print(f"{'─'*60}")
    for s in early['early_signals']:
        print(f"  ✅ {s}")

    if not early['early_signals']:
        print(f"  ❌ No signals")

    # S14 pullback divergence detail
    if flags.get("s14_pullback_divergence"):
        ut_conds = flags.get("s14_uptrend_conditions", {})
        print(f"\n{'─'*60}")
        print(f"  PULLBACK DIVERGENCE DETAIL")
        print(f"{'─'*60}")
        print(f"  Pullback depth   : "
              f"{flags.get('s14_pullback_depth_pct')}%")
        print(f"  OBV held         : {flags.get('s14_obv_held')}")
        print(f"  Uptrend score    : "
              f"{flags.get('s14_uptrend_score')}/4")
        print(f"  Cond A (>20 EMA) : "
              f"{'✅' if ut_conds.get('A') else '❌'}")
        print(f"  Cond B (>50 EMA) : "
              f"{'✅' if ut_conds.get('B') else '❌'}")
        print(f"  Cond C (HH 60D)  : "
              f"{'✅' if ut_conds.get('C') else '❌'}")
        print(f"  Cond D (>50D avg): "
              f"{'✅' if ut_conds.get('D') else '❌'}")

    # Tier explanation
    print(f"\n{'─'*60}")
    print(f"  WHAT THIS MEANS")
    print(f"{'─'*60}")
    tier = early['early_tier']
    if tier == "STRONG":
        print(f"  HIGH PRIORITY watchlist")
        print(f"  Strong OBV signals — institutional accumulation")
        print(f"  Watch for absorption event to confirm entry")
    elif tier == "WATCH":
        print(f"  SECONDARY watchlist")
        print(f"  OBV signals building — needs more confirmation")
        print(f"  Monitor weekly for volume pickup")
    elif tier == "RADAR":
        print(f"  EARLY RADAR only")
        print(f"  First signs of accumulation — too early to act")
        print(f"  Add to watchlist, check chart manually")
    else:
        print(f"  SKIP — insufficient OBV signals")

    print(f"{'='*60}\n")


# ================================================================
# PRINT MARKET RESULTS
# ================================================================

def print_results(df: pd.DataFrame, tier_filter: str = None):
    """Prints early scanner results."""
    if df is None or df.empty:
        print("\nNo stocks found.")
        return

    if tier_filter:
        df = df[df["early_tier"] == tier_filter.upper()]
        if df.empty:
            print(f"\nNo {tier_filter.upper()} tier stocks.")
            return

    scan_date = df["scan_date"].iloc[0]

    print(f"\n{'='*80}")
    print(f"  NSE EARLY ACCUMULATION SCANNER — {scan_date}")
    print(f"  {len(df)} stocks in early accumulation phase")
    print(f"{'='*80}")

    for tier in ["STRONG", "WATCH", "RADAR"]:
        count = len(df[df["early_tier"] == tier])
        if count > 0:
            bar = "█" * min(count, 40)
            print(f"  {tier:<8} {count:>4}  {bar}")
    print()

    for tier in ["STRONG", "WATCH", "RADAR"]:
        tier_df = df[df["early_tier"] == tier]
        if tier_df.empty:
            continue

        print(f"{'─'*80}")
        if tier == "STRONG":
            print(f"  STRONG — High conviction (1-3 month setup)")
        elif tier == "WATCH":
            print(f"  WATCH  — Building signals (2-6 month setup)")
        else:
            print(f"  RADAR  — Early signs only (3-12 month setup)")
        print(f"{'─'*80}")

        for _, row in tier_df.iterrows():
            signals_str = " | ".join(
                row["early_signals"][:4]
            ) if row["early_signals"] else "—"

            # Uptrend conditions if pullback div
            ut_str = ""
            if row.get("pullback_div"):
                ut_str = f" UT:{row.get('uptrend_score')}/4"

            print(
                f"  {row.name:>4}. "
                f"{row['symbol']:<15} "
                f"₹{row['close']:<8.2f} "
                f"Score:{row['early_score']:>3}  "
                f"52WH:{str(row['pct_from_52w_high'])+'%':<10} "
                f"1M:{str(row['chg_1m'])+'%':<8}"
            )
            print(
                f"         "
                f"Deliv:{str(row['delivery_pct'])+'%' if row['delivery_pct'] else 'N/A':<8} "
                f"Range:{str(row['range_pct_4w'])+'%':<8}"
                f"{ut_str}  "
                f"{signals_str}"
            )
            print()

    print(f"{'='*80}\n")


# ================================================================
# ENTRY POINT
# ================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="NSE Early Accumulation Scanner"
    )
    parser.add_argument(
        "--symbol",
        type=str,
        help="Check up to 5 stocks e.g. --symbol TARIL,AIAENG,TCS"
    )
    parser.add_argument(
        "--date",
        type=str,
        help="Scan date YYYY-MM-DD (default: today)"
    )
    parser.add_argument(
        "--min-score",
        type=int,
        default=20,
        help="Minimum early score (default: 20)"
    )
    parser.add_argument(
        "--tier",
        type=str,
        choices=["STRONG", "WATCH", "RADAR"],
        help="Filter by tier"
    )
    args = parser.parse_args()

    scan_date = date.today()
    if args.date:
        try:
            scan_date = datetime.strptime(
                args.date, "%Y-%m-%d"
            ).date()
        except ValueError:
            print(f"Invalid date: {args.date}")
            sys.exit(1)

    if args.symbol:
        symbols = [s.strip().upper() for s in args.symbol.split(",")]
        symbols = symbols[:5]  # max 5
        for symbol in symbols:
            run_single_stock(symbol, scan_date)
    else:
        results = run_early_scanner(
            scan_date=scan_date,
            min_score=args.min_score,
        )
        print_results(results, tier_filter=args.tier)