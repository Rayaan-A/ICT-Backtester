import sys

import config
from backtest.engine import PBBlakeBacktester
from backtest.metrics import calculate_metrics
from data.database import load_ohlcv, save_ohlcv
from data.fetcher import fetch_ohlcv, resample_ohlcv


def run_backtest(symbol: str = config.DEFAULT_SYMBOL) -> None:
    """Fetch multi-TF data, run the PB Blake backtest, and print a summary."""
    print(f"\n{'=' * 60}")
    print(f"  PB Blake Mech Model  ·  {symbol}  ·  base {config.BASE_TIMEFRAME}")
    print(f"{'=' * 60}")

    # ── Fetch / load base TF (5m) ─────────────────────────────────────────
    df_base = load_ohlcv(symbol, config.BASE_TIMEFRAME)
    if df_base is None:
        print("Fetching from yfinance …")
        df_base = fetch_ohlcv(symbol, config.BASE_TIMEFRAME, config.BASE_PERIOD)
        save_ohlcv(df_base, symbol, config.BASE_TIMEFRAME)
        print(f"  {len(df_base)} × {config.BASE_TIMEFRAME} candles fetched and cached.")
    else:
        print(f"  {len(df_base)} × {config.BASE_TIMEFRAME} candles loaded from database.")

    # ── Build multi-TF dict by resampling ────────────────────────────────
    dfs = {config.BASE_TIMEFRAME: df_base}
    all_tfs = set(config.SETUP_TIMEFRAMES) | set(config.HTF_BIAS_TIMEFRAMES)
    for tf in all_tfs:
        if tf == config.BASE_TIMEFRAME:
            continue
        dfs[tf] = resample_ohlcv(df_base, tf)
        print(f"  {len(dfs[tf])} × {tf} candles (resampled)")

    # ── Run backtest ──────────────────────────────────────────────────────
    print("\nRunning backtest …")
    backtester = PBBlakeBacktester(dfs)
    trades     = backtester.run()
    metrics    = calculate_metrics(trades, backtester.equity_curve)

    # ── Trade-type breakdown ──────────────────────────────────────────────
    cont = [t for t in trades if t.setup_type == "continuation"]
    rev  = [t for t in trades if t.setup_type == "reversal"]
    print(f"\nTrade breakdown")
    print(f"  Continuation  {len(cont)}")
    print(f"  Reversal      {len(rev)}")

    # ── Metrics ───────────────────────────────────────────────────────────
    print(f"\nResults")
    for key, value in metrics.items():
        label = key.replace("_", " ").title()
        print(f"  {label:<28} {value}")


if __name__ == "__main__":
    symbol = sys.argv[1] if len(sys.argv) > 1 else config.DEFAULT_SYMBOL
    run_backtest(symbol)
