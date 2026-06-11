# ================================================================
# scripts/run_ema_pre_cross.py
# ----------------------------------------------------------------
# EMA Pre-Cross Strategy — Backtester
#
# Usage:
#   python scripts/run_ema_pre_cross.py --symbol BHARTIARTL
#   python scripts/run_ema_pre_cross.py --top 20
#   python scripts/run_ema_pre_cross.py --all
# ================================================================

import sys
import argparse

sys.path.insert(0, '.')

import pandas as pd
from datetime import date

from src.data.store import load_daily_prices, get_all_symbols
from src.backtests.ema_pre_cross import run_pre_cross_backtest
from src.utils.logger import get_logger

log = get_logger("ema_pre_cross")

ETF_KEYWORDS = [
    'LIQUID', 'GOLD', 'SILVER', 'NIFTY', 'SENSEX',
    'BANKBEES', 'JUNIORBEE', 'BEES', 'SETF', 'NETF',
    'GETF', 'IETF', 'BETA', 'HANG', 'MAFANG',
    'CPSEETF', 'MOM', 'QUAL', 'ALPHA', 'VALUE', 'LOWVOL',
]

def is_etf(symbol: str) -> bool:
    return any(kw in symbol.upper() for kw in ETF_KEYWORDS)


def run_single_symbol(symbol: str):
    df = load_daily_prices(
        symbol,
        from_date=date(2023, 1, 1),
        to_date=date.today()
    )

    if df.empty:
        print(f"No data for {symbol}")
        return

    result = run_pre_cross_backtest(symbol, df)

    if not result or "outcomes" not in result:
        print(f"\n{symbol} — No signals found")
        return

    outcomes = result["outcomes"]
    report   = result["report"]
    signals  = result["signals"]
    a        = report["approach_a"]
    b        = report["approach_b"]

    print(f"\n{'='*80}")
    print(f"  EMA PRE-CROSS STRATEGY — {symbol}")
    print(f"  Entry: close when 20/50 cross imminent")
    print(f"  SL: below 50 EMA OR 2 consec below 20 EMA OR gap expanding")
    print(f"{'='*80}")

    # Signal details
    print(f"\nSignals Detected: {len(signals)}")
    sig_cols = [
        "signal_date", "entry_price", "sl_level",
        "gap_pct", "gap_close_speed",
        "slope_fast_pct", "vol_ratio", "signal_strength"
    ]
    sig_cols = [c for c in sig_cols if c in signals.columns]
    pd.set_option('display.width', 250)
    print(signals[sig_cols].to_string(index=False))

    # Trade results
    print(f"\nAll Trades:")
    print(f"{'─'*80}")
    trade_cols = [
        "signal_date", "signal_strength",
        "entry_price", "sl_level", "target_price",
        "a_exit_reason", "a_exit_return", "a_days", "a_result",
        "b_exit_reason", "b_exit_return", "b_days", "b_result",
        "cross_happened"
    ]
    trade_cols = [c for c in trade_cols if c in outcomes.columns]
    print(outcomes[trade_cols].to_string(index=False))

    # Summary
    print(f"\n{'─'*80}")
    print(f"  APPROACH A — Fixed Target (10% OR SL OR 8W)")
    print(f"{'─'*80}")
    print(f"  Total trades     : {a['total']}")
    print(f"  Wins             : {a['wins']}")
    print(f"  Losses           : {a['losses']}")
    print(f"  Win rate         : {a['win_rate']}%")
    print(f"  Avg win          : {a['avg_win']}%")
    print(f"  Avg loss         : {a['avg_loss']}%")
    print(f"  Risk/Reward      : {a['rr']}")
    print(f"  EV per trade     : {a['ev']}%")
    print(f"  Composite score  : {a['composite']}")
    print(f"\n  Exit reasons:")
    for reason, count in report["a_exits"].items():
        print(f"    {reason:<35} : {count}")

    print(f"\n{'─'*80}")
    print(f"  APPROACH B — Hold Through Cross (15% target)")
    print(f"{'─'*80}")
    print(f"  Total trades     : {b['total']}")
    print(f"  Wins             : {b['wins']}")
    print(f"  Losses           : {b['losses']}")
    print(f"  Win rate         : {b['win_rate']}%")
    print(f"  Avg win          : {b['avg_win']}%")
    print(f"  Avg loss         : {b['avg_loss']}%")
    print(f"  Risk/Reward      : {b['rr']}")
    print(f"  EV per trade     : {b['ev']}%")
    print(f"  Composite score  : {b['composite']}")
    print(f"  Cross happened   : {report['cross_rate']}% of trades")
    print(f"\n  Exit reasons:")
    for reason, count in report["b_exits"].items():
        print(f"    {reason:<35} : {count}")

    print(f"\n{'─'*80}")
    print(f"  BY SIGNAL STRENGTH:")
    for strength, s in report["by_strength"].items():
        print(f"    {strength:<10} "
              f"total:{s['total']}  "
              f"wins:{s['wins']}  "
              f"WR(A):{s['win_rate']}%")

    print(f"\n{'─'*80}")
    print(f"  APPROACH COMPARISON:")
    print(f"  {'Metric':<20} {'Approach A':>15} {'Approach B':>15}")
    print(f"  {'─'*50}")
    print(f"  {'Win Rate':<20} {str(a['win_rate'])+'%':>15} "
          f"{str(b['win_rate'])+'%':>15}")
    print(f"  {'Avg Win':<20} {str(a['avg_win'])+'%':>15} "
          f"{str(b['avg_win'])+'%':>15}")
    print(f"  {'Avg Loss':<20} {str(a['avg_loss'])+'%':>15} "
          f"{str(b['avg_loss'])+'%':>15}")
    print(f"  {'R:R':<20} {str(a['rr']):>15} "
          f"{str(b['rr']):>15}")
    print(f"  {'EV per trade':<20} {str(a['ev'])+'%':>15} "
          f"{str(b['ev'])+'%':>15}")
    print(f"  {'Composite':<20} {str(a['composite']):>15} "
          f"{str(b['composite']):>15}")
    print(f"{'='*80}\n")


def run_market_analysis(top_n: int = None):
    symbols = get_all_symbols()
    log.info(f"Running pre-cross backtest for {len(symbols)} symbols...")

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

        if df["volume"].tail(22).mean() < 50000:
            continue

        if df["close"].iloc[-1] < 50:
            continue

        result = run_pre_cross_backtest(symbol, df)
        if not result or "report" not in result:
            continue

        report = result["report"]
        a      = report.get("approach_a", {})
        b      = report.get("approach_b", {})

        if a.get("total", 0) < 3:
            continue
        if a.get("losses", 0) == 0:
            continue

        all_results.append({
            "symbol"      : symbol,
            "signals"     : a.get("total", 0),
            "cross_rate"  : report.get("cross_rate"),
            # Approach A
            "a_wr"        : a.get("win_rate"),
            "a_avg_win"   : a.get("avg_win"),
            "a_avg_loss"  : a.get("avg_loss"),
            "a_rr"        : a.get("rr"),
            "a_ev"        : a.get("ev"),
            "a_composite" : a.get("composite"),
            # Approach B
            "b_wr"        : b.get("win_rate"),
            "b_avg_win"   : b.get("avg_win"),
            "b_avg_loss"  : b.get("avg_loss"),
            "b_rr"        : b.get("rr"),
            "b_ev"        : b.get("ev"),
            "b_composite" : b.get("composite"),
        })

    if not all_results:
        print("No results")
        return

    df_results = pd.DataFrame(all_results)
    df_full    = df_results.copy()

    # Sort by Approach A composite (primary)
    df_results = df_results.sort_values(
        "a_composite",
        ascending=False,
        na_position='last'
    ).reset_index(drop=True)
    df_results.index += 1

    if top_n:
        df_results = df_results.head(top_n)

    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 250)

    print(f"\n{'='*80}")
    print(f"  EMA PRE-CROSS STRATEGY — MARKET ANALYSIS")
    print(f"  Ranked by Approach A composite score")
    print(f"{'='*80}")
    print(df_results.to_string())

    # Market stats
    total    = int(df_full["signals"].sum())
    a_wins   = int((df_full["a_wr"] / 100 * df_full["signals"]).sum())
    overall  = round(a_wins / total * 100, 1) if total > 0 else 0

    print(f"\n{'─'*60}")
    print(f"  MARKET WIDE — APPROACH A")
    print(f"{'─'*60}")
    print(f"  Stocks qualifying  : {len(all_results)}")
    print(f"  Total signals      : {total}")
    print(f"  Overall WR         : {overall}%")
    print(f"  Avg WR             : {df_full['a_wr'].mean():.1f}%")
    print(f"  Avg EV             : {df_full['a_ev'].mean():.2f}%")
    print(f"  Avg R:R            : {df_full['a_rr'].mean():.2f}")
    print(f"  Avg cross rate     : {df_full['cross_rate'].mean():.1f}%")
    print(f"  Stocks EV > 0      : "
          f"{len(df_full[df_full['a_ev'] > 0])}")
    print(f"  Stocks EV > 2%     : "
          f"{len(df_full[df_full['a_ev'] > 2])}")
    print(f"  Stocks WR > 60%    : "
          f"{len(df_full[df_full['a_wr'] >= 60])}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="EMA Pre-Cross Strategy Backtester"
    )
    parser.add_argument("--symbol", type=str)
    parser.add_argument("--top",    type=int)
    parser.add_argument("--all",    action="store_true")
    args = parser.parse_args()

    if args.symbol:
        run_single_symbol(args.symbol.upper())
    else:
        run_market_analysis(top_n=args.top)