import sys

import config
from backtest.engine import Backtester
from backtest.metrics import calculate_metrics
from data.database import load_ohlcv, save_ohlcv
from data.fetcher import fetch_ohlcv
from signals.fvg import detect_fvgs
from signals.liquidity import detect_sweeps
from signals.order_blocks import detect_order_blocks


def run_backtest(
    symbol: str = config.DEFAULT_SYMBOL,
    timeframe: str = config.DEFAULT_TIMEFRAME,
) -> None:
    """Run the full ICT backtest pipeline and print a summary."""
    print(f"\n{'=' * 52}")
    print(f"  {symbol}  {timeframe}")
    print(f"{'=' * 52}")

    df = load_ohlcv(symbol, timeframe)
    if df is None:
        print("Fetching from yfinance...")
        df = fetch_ohlcv(symbol, timeframe)
        save_ohlcv(df, symbol, timeframe)
        print(f"  {len(df)} candles fetched and cached.")
    else:
        print(f"  {len(df)} candles loaded from database.")

    fvgs = detect_fvgs(df)
    obs = detect_order_blocks(df)
    sweeps = detect_sweeps(df)

    bull_fvg = sum(1 for f in fvgs if f.direction == "bullish")
    bear_fvg = len(fvgs) - bull_fvg
    bull_ob = sum(1 for o in obs if o.direction == "bullish")
    bear_ob = len(obs) - bull_ob
    sweep_low = sum(1 for s in sweeps if s.direction == "low")
    sweep_high = len(sweeps) - sweep_low

    print(f"\nSignals")
    print(f"  FVGs          {bull_fvg} bullish  {bear_fvg} bearish")
    print(f"  Order Blocks  {bull_ob} bullish  {bear_ob} bearish")
    print(f"  Liq Sweeps    {sweep_low} low  {sweep_high} high")

    backtester = Backtester(df)
    trades = backtester.run()
    metrics = calculate_metrics(trades, backtester.equity_curve)

    print(f"\nResults")
    for key, value in metrics.items():
        label = key.replace("_", " ").title()
        print(f"  {label:<28} {value}")


if __name__ == "__main__":
    symbol = sys.argv[1] if len(sys.argv) > 1 else config.DEFAULT_SYMBOL
    timeframe = sys.argv[2] if len(sys.argv) > 2 else config.DEFAULT_TIMEFRAME
    run_backtest(symbol, timeframe)
