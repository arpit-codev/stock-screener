# ================================================================
# scripts/run_yearly_open_analysis.py
# ----------------------------------------------------------------
# Runs yearly open strategy analysis.
#
# Usage:
#   python scripts/run_yearly_open_analysis.py --symbol TCS
#   python scripts/run_yearly_open_analysis.py --top 20
#   python scripts/run_yearly_open_analysis.py --all
# ================================================================

import sys
import argparse

sys.path.insert(0, '.')

import pandas as pd
from datetime import date

from src.data.store import load_daily_prices, get_all_symbols
from src.indicators.yearly_open import analyse_symbol
from src.utils.logger import get_logger

log = get_logger("yearly_open_analysis")


# ================================================================
# ETF FILTER
# ================================================================

ETF_KEYWORDS = [
    'LIQUID', 'GOLD', 'SILVER', 'NIFTY', 'SENSEX',
    'BANKBEES', 'JUNIORBEE', 'BEES', 'SETF', 'NETF',
    'GETF', 'IETF', 'BETA', 'HANG', 'MAFANG',
    'CPSEETF', 'MONQ', 'AUTOBEES', 'CONSUMBEES',
    'FINBEES', 'PHARMABEES', 'ITBEES', 'INFRABEES',
    'MOM', 'QUAL', 'ALPHA', 'VALUE', 'LOWVOL',
]

def is_etf(symbol: str) -> bool:
    """
    Returns True if symbol looks like an ETF or liquid fund.
    These are excluded from analysis — not tradeable stocks.
    """
    s = symbol.upper()
    return any(kw in s for kw in ETF_KEYWORDS)


# ================================================================
# SINGLE SYMBOL ANALYSIS
# ================================================================

def run_single_symbol(symbol: str):
    """Full analysis for one symbol."""
    df = load_daily_prices(
        symbol,
        from_date=date(2023, 1, 1),
        to_date=date.today()
    )

    if df.empty:
        print(f"No data for {symbol}")
        return

    result = analyse_symbol(symbol, df)

    if not result or result.get("tests", pd.DataFrame()).empty:
        print(f"\n{symbol} — No yearly open tests found")
        return

    outcomes = result["outcomes"]
    report   = result["report"]
    yo       = result["yearly_opens"]

    print(f"\n{'='*70}")
    print(f"  YEARLY OPEN STRATEGY — {symbol}")
    print(f"{'='*70}")

    # Yearly opens
    print(f"\nYearly Opens:")
    for _, r in yo.iterrows():
        print(f"  {int(r['year'])}  ₹{r['yearly_open']:.2f}"
              f"  (first trade: {r['first_trade_date']})")

    # All trades
    print(f"\nAll Trades:")
    print(f"{'─'*70}")
    cols = [
        "test_date", "year", "yearly_open", "test_number",
        "test_type", "entry_price", "sl_level", "target_price",
        "delivery_at_test", "vol_ratio_at_test",
        "exit_reason", "exit_return_pct", "days_to_exit",
        "trade_result"
    ]
    cols = [c for c in cols if c in outcomes.columns]
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 200)
    print(outcomes[cols].to_string(index=False))

    # Summary
    print(f"\n{'─'*70}")
    print(f"  SUMMARY")
    print(f"{'─'*70}")
    print(f"  Total trades       : {report['total_tests']}")
    print(f"  Wins               : {report['wins']}")
    print(f"  Losses             : {report['losses']}")
    print(f"  Win rate           : {report['win_rate_pct']}%")
    print(f"  SL hit (price 5%)  : {report['sl_price_hits']}")
    print(f"  SL hit (structure) : {report['sl_struct_hits']}")
    print(f"  Avg 8W return      : {report['avg_return_8w']}%")
    print(f"  Best 8W return     : {report['best_return_8w']}%")
    print(f"  Worst 8W return    : {report['worst_return_8w']}%")
    print(f"  Avg win            : {report['avg_win']}%")
    print(f"  Avg loss           : {report['avg_loss']}%")
    print(f"  Risk/Reward        : {report['risk_reward']}")
    print(f"  EV per trade       : {report.get('ev_per_trade')}%")
    print(f"  Composite score    : {report.get('composite_score')}")
    print(f"  First touch WR     : {report['first_touch_wr']}%")
    print(f"  Later touch WR     : {report['later_touch_wr']}%")
    print(f"  High delivery WR   : {report['high_deliv_wr']}%")

    print(f"\n  Exit reasons:")
    if "exit_reason" in outcomes.columns:
        for reason, count in outcomes["exit_reason"].value_counts().items():
            print(f"    {reason:<25} : {count}")

    print(f"\n  By test type:")
    for ttype, stats in report["by_test_type"].items():
        print(f"    {ttype:<20} "
              f"total:{stats['total']}  "
              f"wins:{stats['wins']}  "
              f"WR:{stats['win_rate']}%")
    print(f"{'='*70}\n")


# ================================================================
# MARKET WIDE ANALYSIS
# ================================================================

def run_market_analysis(top_n: int = None):
    """Run across all stocks — find best performers."""
    symbols = get_all_symbols()
    log.info(f"Analysing {len(symbols)} symbols...")

    all_results = []

    for i, symbol in enumerate(symbols, 1):
        if i % 100 == 0:
            log.info(f"Progress: {i}/{len(symbols)}")

        # Skip ETFs and liquid funds
        if is_etf(symbol):
            log.debug(f"Skipping ETF: {symbol}")
            continue

        df = load_daily_prices(
            symbol,
            from_date=date(2023, 1, 1),
            to_date=date.today()
        )

        if df.empty or len(df) < 100:
            continue

        if df.empty or len(df) < 100:
            continue

            # Skip if average volume too low — illiquid / operator stocks
        avg_vol = df["volume"].tail(22).mean()
        if avg_vol < 50000:
            log.debug(f"Skipping {symbol} — low volume {avg_vol:.0f}")
            continue

        # Skip if price too low — penny stocks
        latest_close = df["close"].iloc[-1]
        if latest_close < 50:
            log.debug(f"Skipping {symbol} — price too low {latest_close}")
            continue

        result = analyse_symbol(symbol, df)
        if not result:
            continue

        tests = result.get("tests", pd.DataFrame())
        if tests.empty:
            continue

        report = result["report"]

        # Minimum quality filters
        if report.get("total_tests", 0) < 3:
            continue

        # Skip if no losses — 100% win rate = data issue
        if report.get("losses", 0) == 0:
            continue

        # Calculate Expected Value
        avg_win   = report.get("avg_win")  or 0
        avg_loss  = report.get("avg_loss") or 0
        win_rate  = report.get("win_rate_pct") or 0
        loss_rate = 100 - win_rate

        ev = round(
            (win_rate  / 100 * avg_win) +
            (loss_rate / 100 * avg_loss),
            2
        )

        rr       = report.get("risk_reward") or 0
        n_trades = report.get("total_tests") or 0

        composite_score = round(ev * rr, 2) if rr else None

        all_results.append({
            "symbol"          : symbol,
            "total_trades"    : n_trades,
            "wins"            : report["wins"],
            "losses"          : report["losses"],
            "win_rate"        : win_rate,
            "avg_win"         : avg_win,
            "avg_loss"        : avg_loss,
            "risk_reward"     : rr,
            "ev_per_trade"    : ev,
            "composite_score" : composite_score,
            "sl_price_hits"   : report["sl_price_hits"],
            "sl_struct_hits"  : report["sl_struct_hits"],
            "avg_return_8w"   : report["avg_return_8w"],
            "first_touch_wr"  : report["first_touch_wr"],
            "high_deliv_wr"   : report["high_deliv_wr"],
        })

    if not all_results:
        print("No results")
        return

    df_results = pd.DataFrame(all_results)

    # Sort by composite score — NaN pushed to bottom
    df_results = df_results.sort_values(
        "composite_score",
        ascending=False,
        na_position='last'
    ).reset_index(drop=True)
    df_results.index += 1

    if top_n:
        df_results = df_results.head(top_n)

    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 200)

    print(f"\n{'='*80}")
    print(f"  YEARLY OPEN STRATEGY — MARKET ANALYSIS")
    print(f"  Stocks analysed : {len(all_results)}")
    print(f"  Entry: intraday touch + false breakdown only")
    print(f"  SL: 5% below yearly open OR 3 consecutive lower lows")
    print(f"  Target: 10% from entry OR 8 weeks whichever first")
    print(f"  Ranked by: EV × R:R composite score")
    print(f"{'='*80}")
    print(df_results.to_string())

    # Market wide stats
    total_trades = int(df_results["total_trades"].sum())
    total_wins   = int(df_results["wins"].sum())
    overall_wr   = round(total_wins / total_trades * 100, 1) \
                   if total_trades > 0 else 0

    print(f"\n{'─'*60}")
    print(f"  MARKET WIDE STATISTICS")
    print(f"{'─'*60}")
    print(f"  Stocks qualifying      : {len(all_results)}")
    print(f"  Total trades           : {total_trades}")
    print(f"  Overall win rate       : {overall_wr}%")
    print(f"  Avg win rate           : "
          f"{df_results['win_rate'].mean():.1f}%")
    print(f"  Avg EV per trade       : "
          f"{df_results['ev_per_trade'].mean():.2f}%")
    print(f"  Avg Risk/Reward        : "
          f"{df_results['risk_reward'].mean():.2f}")
    print(f"  Avg composite score    : "
          f"{df_results['composite_score'].mean():.2f}")
    print(f"  Stocks EV > 0          : "
          f"{len(df_results[df_results['ev_per_trade'] > 0])}")
    print(f"  Stocks EV > 2%         : "
          f"{len(df_results[df_results['ev_per_trade'] > 2])}")
    print(f"  Stocks WR > 60%        : "
          f"{len(df_results[df_results['win_rate'] >= 60])}")
    print(f"  Stocks RR > 2.0        : "
          f"{len(df_results[df_results['risk_reward'] >= 2.0])}")
    print(f"{'='*80}\n")


# ================================================================
# ENTRY POINT
# ================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="NSE Yearly Open Strategy Backtester"
    )
    parser.add_argument("--symbol", type=str, help="Single symbol")
    parser.add_argument("--top",    type=int, help="Top N by composite score")
    parser.add_argument("--all",    action="store_true",
                        help="Run all stocks")
    args = parser.parse_args()

    if args.symbol:
        run_single_symbol(args.symbol.upper())
    else:
        run_market_analysis(top_n=args.top)