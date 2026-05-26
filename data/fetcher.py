import pandas as pd
import yfinance as yf

import config


def fetch_ohlcv(
    symbol: str,
    timeframe: str = config.DEFAULT_TIMEFRAME,
) -> pd.DataFrame:
    """Fetch OHLCV data from yfinance and return with ATR column."""
    period = config.TIMEFRAME_PERIODS.get(timeframe, "20d")
    ticker = yf.Ticker(symbol)
    df = ticker.history(period=period, interval=timeframe)
    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.columns = ["open", "high", "low", "close", "volume"]
    df.dropna(inplace=True)
    df = _add_atr(df, config.ATR_PERIOD)
    return df


def _add_atr(df: pd.DataFrame, period: int) -> pd.DataFrame:
    """Compute ATR using Wilder's EMA and append as 'atr' column."""
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    df = df.copy()
    df["atr"] = tr.ewm(span=period, adjust=False).mean()
    return df
