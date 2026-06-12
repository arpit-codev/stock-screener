# ================================================================
# scripts/run_scanner.py
# ----------------------------------------------------------------
# Daily scanner entry point.
# Run every evening after market close + daily sync.
#
# Usage:
#   python scripts/run_scanner.py
#   python scripts/run_scanner.py --date 2026-06-10
#   python scripts/run_scanner.py --min-score 50
#   python scripts/run_scanner.py --tier HIGH
#   python scripts/run_scanner.py --no-alert
# ================================================================

import sys
import argparse
from datetime import date, datetime

sys.path.insert(0, '.')

import pandas as pd
from src.scanner.active  import run_scanner
from src.utils.logger    import get_logger

log = get_logger("run_scanner")


def print_results(df: pd.DataFrame, tier_filter: str = None):
    """Prints ranked results to terminal."""

    if df is None or df.empty:
        print("\nNo stocks found matching criteria.")
        return

    # Apply tier filter if specified
    if tier_filter:
        df = df[df["conviction_tier"] == tier_filter.upper()]
        if df.empty:
            print(f"\nNo {tier_filter.upper()} tier stocks found.")
            return

    scan_date = df["scan_date"].iloc[0] \
                if "scan_date" in df.columns else date.today()

    print(f"\n{'='*80}")
    print(f"  NSE SMART MONEY SCANNER — {scan_date}")
    print(f"  {len(df)} stocks found")
    print(f"{'='*80}")

    # Tier summary
    for tier in ["HIGH", "MEDIUM", "LOW", "SKIP"]:
        count = len(df[df["conviction_tier"] == tier])
        if count > 0:
            bar = "█" * min(count, 30)
            print(f"  {tier:<8} {count:>4}  {bar}")
    print()

    # Detailed results by tier
    for tier in ["HIGH", "MEDIUM", "LOW"]:
        tier_df = df[df["conviction_tier"] == tier]
        if tier_df.empty:
            continue

        print(f"{'─'*80}")
        print(f"  {tier} CONVICTION")
        print(f"{'─'*80}")

        for _, row in tier_df.iterrows():
            # Build signal string
            signals = []
            if row.get("absorption"):
                t = row.get("absorption_tier", "")
                d = row.get("absorption_date", "")
                signals.append(f"ABS-{t}({d})")
            if row.get("obv_divergence"):
                signals.append("OBV-DIV")
            if row.get("weekly_divergence"):
                signals.append("W-DIV")
            if row.get("shakeout"):
                signals.append("SHAKEOUT")
            if row.get("consolidating"):
                signals.append("COIL")
            if row.get("higher_lows"):
                signals.append("HL")
            if row.get("bonuses_fired"):
                for b in row["bonuses_fired"]:
                    signals.append(f"BONUS:{b[:8]}")

            signal_str = " | ".join(signals) if signals else "—"

            print(
                f"  {row.name:>4}. "
                f"{row['symbol']:<15} "
                f"₹{row['close']:<8.2f} "
                f"Score:{row['final_score']:>3} "
                f"[V:{row['volume_score']:>2} "
                f"O:{row['obv_score']:>2} "
                f"P:{row['price_score']:>2}] "
                f"Deliv:{str(row['delivery_pct'])+'%' if row['delivery_pct'] else 'N/A':<8} "
                f"1M:{str(row['chg_1m'])+'%' if row['chg_1m'] else 'N/A':<8}"
            )
            print(
                f"         "
                f"Vol:{row['day_vol_ratio']}x  "
                f"52WH:{row['pct_from_52w_high']}%  "
                f"{signal_str}"
            )
            print()

    print(f"{'='*80}\n")

def run_single_stock(symbol: str, scan_date: date = None):
    """Full indicator breakdown for one stock."""
    from src.data.store        import load_daily_prices, \
                                      load_weekly_prices
    from src.indicators.volume import calculate_volume_indicators
    from src.indicators.obv    import calculate_obv_indicators
    from src.indicators.price  import calculate_price_indicators
    from src.scanner.scoring   import calculate_conviction_score
    from datetime import timedelta

    if scan_date is None:
        scan_date = date.today()

    from_date = scan_date - timedelta(days=365)

    df_d = load_daily_prices(symbol,
        from_date=from_date,
        to_date=scan_date)
    df_w = load_weekly_prices(symbol)

    if df_d is None or df_d.empty:
        print(f"\nNo data found for {symbol}")
        return

    vol   = calculate_volume_indicators(df_d)
    obv   = calculate_obv_indicators(df_d, df_w)
    price = calculate_price_indicators(df_d)

    if not obv.get("tier1_passed", True):
        print(f"\n{symbol} — BLOCKED")
        print(f"Reason: {obv.get('tier1_reason')}")
        return

    score = calculate_conviction_score(vol, obv, price, symbol)

    print(f"\n{'='*60}")
    print(f"  SMART MONEY SCANNER — {symbol}")
    print(f"  Date: {scan_date}")
    print(f"{'='*60}")
    print(f"\n  Final Score    : {score['final_score']}/100")
    print(f"  Tier           : {score['conviction_tier']}")
    print(f"  Volume score   : {score['volume_score']}/35")
    print(f"  OBV score      : {score['obv_score']}/40")
    print(f"  Price score    : {score['price_score']}/25")
    print(f"  Bonus pts      : +{score['bonus_pts']}")
    print(f"  Red flag pts   : {score['red_flag_pts']}")

    print(f"\n{'─'*60}")
    print(f"  PRICE")
    print(f"{'─'*60}")
    print(f"  Close          : ₹{price.get('close')}")
    print(f"  Chg 1D         : {price.get('chg_1d')}%")
    print(f"  Chg 1W         : {price.get('chg_1w')}%")
    print(f"  Chg 1M         : {price.get('chg_1m')}%")
    print(f"  From 52W High  : {price.get('pct_from_52w_high')}%")
    print(f"  From 20 EMA    : {price.get('pct_from_ema20')}%")
    print(f"  Range 4W       : {price.get('range_pct_4w')}%")

    print(f"\n{'─'*60}")
    print(f"  VOLUME")
    print(f"{'─'*60}")
    print(f"  Vol today      : {vol.get('vol_today'):,}")
    print(f"  Vol 22D median : {vol.get('vol_22d_avg'):,.0f}")
    print(f"  Day vol ratio  : {vol.get('day_vol_ratio')}x")
    print(f"  Delivery today : {vol.get('delivery_pct_today')}%")
    print(f"  Delivery 22D   : {vol.get('deliv_22d_avg')}%")
    if vol.get('scenario_absorption'):
        print(f"  Absorption     : ✅ Tier-{vol.get('absorption_tier')} "
              f"on {vol.get('absorption_date')} "
              f"({vol.get('absorption_vol_ratio')}x vol, "
              f"{vol.get('absorption_delivery_pct')}% deliv)")
    else:
        print(f"  Absorption     : ❌ not detected")

    print(f"\n{'─'*60}")
    print(f"  OBV")
    print(f"{'─'*60}")
    print(f"  OBV score      : {obv.get('total_score')}/100")
    print(f"  OBV tier       : {obv.get('conviction_tier')}")
    print(f"  OBV rising 20D : {obv.get('obv_rising_20d')}")
    print(f"  Price chg 20D  : {obv.get('price_chg_20d')}%")
    print(f"  Daily div 22D  : {obv.get('daily_div_22d')}")
    print(f"  Weekly div 8W  : {obv.get('weekly_div_8w')}")

    print(f"\n{'─'*60}")
    print(f"  SCENARIOS FIRED")
    print(f"{'─'*60}")

    for k, v in vol.items():
        if k.startswith('scenario_') and v:
            pts = score['component_scores'].get(f'vol_{k}', 0)
            print(f"  ✅ VOL  {k:<38} +{pts}")
    if vol.get('scenario_absorption'):
        pts = score['component_scores'].get('vol_absorption_tiered', 0)
        print(f"  ✅ VOL  absorption_tier_{vol.get('absorption_tier')}"
              f"{'':30} +{pts}")

    for k, v in obv.get('flags', {}).items():
        if isinstance(v, bool) and v:
            pts = obv.get('scores', {}).get(k, 0)
            print(f"  ✅ OBV  {k:<38} +{pts}")

    for k, v in price.items():
        if k.startswith('scenario_') and v:
            pts = score['component_scores'].get(f'price_{k}', 0)
            print(f"  ✅ PRC  {k:<38} +{pts}")

    if score['bonuses_fired']:
        for b in score['bonuses_fired']:
            print(f"  🎯 BONUS {b}")

    if score['red_flags_fired']:
        for r in score['red_flags_fired']:
            print(f"  ⚠️  FLAG  {r}")

    print(f"{'='*60}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="NSE Smart Money Scanner"
    )
    parser.add_argument(
        "--date",
        type=str,
        help="Scan date YYYY-MM-DD (default: today)"
    )
    parser.add_argument(
        "--symbol",
        type=str,
        help="Check up to 5 stocks e.g. --symbol JUBLFOOD,TCS,RELIANCE"
    )
    parser.add_argument(
        "--min-score",
        type=int,
        default=0,
        help="Minimum score to include (default: 0)"
    )
    parser.add_argument(
        "--tier",
        type=str,
        choices=["HIGH", "MEDIUM", "LOW", "SKIP"],
        help="Filter by conviction tier"
    )
    parser.add_argument(
        "--no-alert",
        action="store_true",
        help="Skip Telegram alert"
    )
    args = parser.parse_args()

    # Parse date
    scan_date = date.today()
    if args.date:
        try:
            scan_date = datetime.strptime(args.date, "%Y-%m-%d").date()
        except ValueError:
            print(f"Invalid date format: {args.date}. Use YYYY-MM-DD")
            sys.exit(1)

    log.info(f"Starting scanner for {scan_date}")

    # Run scanner
    # multi stock mode
    if args.symbol:
        symbols = [s.strip().upper() for s in args.symbol.split(",")]
        symbols = symbols[:5]  # max 5
        for symbol in symbols:
            run_single_stock(symbol, scan_date)
    else:
        # Full market scan
        results = run_scanner(
            scan_date=scan_date,
            min_score=args.min_score,
        )
        print_results(results, tier_filter=args.tier)

        if not args.symbol and not results.empty:
            high = len(results[results["conviction_tier"] == "HIGH"])
            medium = len(results[results["conviction_tier"] == "MEDIUM"])
            low = len(results[results["conviction_tier"] == "LOW"])
            log.info(
                f"Final: HIGH={high} MEDIUM={medium} LOW={low}"
            )

