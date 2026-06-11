# ================================================================
# scripts/run_ema_backtest.py
# ----------------------------------------------------------------
# EMA Crossover Strategy — Market Backtester
#
# Usage:
#   python scripts/run_ema_backtest.py --symbol JUBLFOOD
#   python scripts/run_ema_backtest.py --top 20
#   python scripts/run_ema_backtest.py --all
# ================================================================

import sys
import argparse

sys.path.insert(0, '.')

import pandas as pd
from datetime import date

from src.data.store import load_daily_prices, get_all_symbols
from src.backtests.ema_crossover import run_ema_backtest
from src.utils.logger import get_logger

log = get_logger("ema_backtest")

# ── Same filters as yearly open ────────────────────────────────
ETF_KEYWORDS = [
    'LIQUID', 'GOLD', 'SILVER', 'NIFTY', 'SENSEX',
    'BANKBEES', 'JUNIORBEE', 'BEES', 'SETF', 'NETF',
    'GETF', 'IETF', 'BETA', 'HANG', 'MAFANG',
    'CPSEETF', 'MOM', 'QUAL', 'ALPHA', 'VALUE', 'LOWVOL',
]

def is_etf(symbol: str) -> bool:
    return any(kw in symbol.upper() for kw in ETF_KEYWORDS)


# ================================================================
# SINGLE SYMBOL
# ================================================================

def run_single_symbol(symbol: str):
    """Full backtest for one symbol."""
    df = load_daily_prices(
        symbol,
        from_date=date(2023, 1, 1),
        to_date=date.today()
    )

    if df.empty:
        print(f"No data for {symbol}")
        return

    result = run_ema_backtest(symbol, df)

    if not result or "outcomes" not in result:
        print(f"\n{symbol} — No signals found")
        return

    outcomes = result["outcomes"]
    report   = result["report"]

    print(f"\n{'='*70}")
    print(f"  EMA 9/15 CROSSOVER STRATEGY — {symbol}")
    print(f"{'='*70}")

    # All trades
    print(f"\nAll Trades:")
    print(f"{'─'*70}")
    cols = [
        "crossover_date", "crossover_type", "entry_scenario",
        "entry_date", "entry_price", "sl_price", "target_price",
        "exit_reason", "exit_return_pct", "days_to_exit",
        "trade_result"
    ]
    cols = [c for c in cols if c in outcomes.columns]
    pd.set_option('display.width', 250)
    pd.set_option('display.max_columns', None)
    print(outcomes[cols].to_string(index=False))

    # Summary
    print(f"\n{'─'*70}")
    print(f"  SUMMARY")
    print(f"{'─'*70}")
    print(f"  Total trades       : {report['total_trades']}")
    print(f"  Wins               : {report['wins']}")
    print(f"  Losses             : {report['losses']}")
    print(f"  Win rate           : {report['win_rate_pct']}%")
    print(f"  Avg win            : {report['avg_win']}%")
    print(f"  Avg loss           : {report['avg_loss']}%")
    print(f"  Risk/Reward        : {report['risk_reward']}")
    print(f"  EV per trade       : {report['ev_per_trade']}%")
    print(f"  Composite score    : {report['composite_score']}")
    print(f"  Avg 8W return      : {report['avg_return_8w']}%")
    print(f"  Best 8W return     : {report['best_return_8w']}%")
    print(f"  Worst 8W return    : {report['worst_return_8w']}%")

    print(f"\n  Exit breakdown:")
    print(f"    Target 10%       : {report['targets_hit']}")
    print(f"    SL EMA cross     : {report['sl_ema_hits']}")
    print(f"    SL price 5%      : {report['sl_price_hits']}")
    print(f"    Time 8W          : {report['time_exits']}")

    print(f"\n  By crossover type:")
    for ct, s in report["by_crossover_type"].items():
        print(f"    {ct:<12} "
              f"total:{s['total']}  "
              f"wins:{s['wins']}  "
              f"WR:{s['win_rate']}%")

    print(f"\n  By entry scenario:")
    for es, s in report["by_entry_scenario"].items():
        print(f"    {es:<25} "
              f"total:{s['total']}  "
              f"wins:{s['wins']}  "
              f"WR:{s['win_rate']}%")

    print(f"{'='*70}\n")


# ================================================================
# MARKET ANALYSIS
# ================================================================

def run_market_analysis(top_n: int = None):
    """Run across all stocks — find best performers."""
    symbols = get_all_symbols()
    log.info(f"Running EMA backtest for {len(symbols)} symbols...")

    all_results = []

    for i, symbol in enumerate(symbols, 1):
        if i % 100 == 0:
            log.info(f"Progress: {i}/{len(symbols)}")

        if is_etf(symbol):
            continue

        df = load_daily_prices(
            symbol,
            from_date=date(2023, 1, 1),
            to_date=date.today()
        )

        if df.empty or len(df) < 100:
            continue

        # Liquidity filters
        avg_vol = df["volume"].tail(22).mean()
        if avg_vol < 50000:
            continue

        latest_close = df["close"].iloc[-1]
        if latest_close < 50:
            continue

        result = run_ema_backtest(symbol, df)
        if not result or "report" not in result:
            continue

        report = result["report"]

        # Minimum quality filters
        if report.get("total_trades", 0) < 3:
            continue
        if report.get("losses", 0) == 0:
            continue

        avg_win  = report.get("avg_win")  or 0
        avg_loss = report.get("avg_loss") or 0
        win_rate = report.get("win_rate_pct") or 0
        rr       = report.get("risk_reward") or 0
        ev       = report.get("ev_per_trade") or 0
        composite = report.get("composite_score")

        all_results.append({
            "symbol"          : symbol,
            "total_trades"    : report["total_trades"],
            "wins"            : report["wins"],
            "losses"          : report["losses"],
            "win_rate"        : win_rate,
            "avg_win"         : avg_win,
            "avg_loss"        : avg_loss,
            "risk_reward"     : rr,
            "ev_per_trade"    : ev,
            "composite_score" : composite,
            "sl_ema_hits"     : report["sl_ema_hits"],
            "sl_price_hits"   : report["sl_price_hits"],
            "targets_hit"     : report["targets_hit"],
            "avg_return_8w"   : report["avg_return_8w"],
        })

    if not all_results:
        print("No results")
        return

    df_results   = pd.DataFrame(all_results)
    df_full      = df_results.copy()

    df_results = df_results.sort_values(
        "composite_score",
        ascending=False,
        na_position='last'
    ).reset_index(drop=True)
    df_results.index += 1

    if top_n:
        df_results = df_results.head(top_n)

    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 250)

    print(f"\n{'='*80}")
    print(f"  EMA 9/15 CROSSOVER — MARKET ANALYSIS")
    print(f"  Entry: first pullback to 9 EMA after crossover")
    print(f"  SL: EMA reversal OR 5% price SL")
    print(f"  Target: 10% OR 8 weeks whichever first")
    print(f"  Ranked by: EV × R:R composite score")
    print(f"{'='*80}")
    print(df_results.to_string())

    # Market wide stats — all qualifying stocks
    total_trades = int(df_full["total_trades"].sum())
    total_wins   = int(df_full["wins"].sum())
    overall_wr   = round(total_wins / total_trades * 100, 1) \
                   if total_trades > 0 else 0

    print(f"\n{'─'*60}")
    print(f"  MARKET WIDE STATISTICS")
    print(f"{'─'*60}")
    print(f"  Stocks qualifying      : {len(all_results)}")
    print(f"  Total trades           : {total_trades}")
    print(f"  Overall win rate       : {overall_wr}%")
    print(f"  Avg win rate           : "
          f"{df_full['win_rate'].mean():.1f}%")
    print(f"  Avg EV per trade       : "
          f"{df_full['ev_per_trade'].mean():.2f}%")
    print(f"  Avg Risk/Reward        : "
          f"{df_full['risk_reward'].mean():.2f}")
    print(f"  Stocks EV > 0          : "
          f"{len(df_full[df_full['ev_per_trade'] > 0])}")
    print(f"  Stocks EV > 2%         : "
          f"{len(df_full[df_full['ev_per_trade'] > 2])}")
    print(f"  Stocks WR > 50%        : "
          f"{len(df_full[df_full['win_rate'] >= 50])}")
    print(f"  Stocks WR > 60%        : "
          f"{len(df_full[df_full['win_rate'] >= 60])}")
    print(f"  Stocks RR > 2.0        : "
          f"{len(df_full[df_full['risk_reward'] >= 2.0])}")
    print(f"{'='*80}\n")


# ================================================================
# ENTRY POINT
# ================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="EMA 9/15 Crossover Strategy Backtester"
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