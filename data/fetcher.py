import pandas as pd
import yfinance as yf

import config


def fetch_ohlcv(
    symbol: str,
    timeframe: str = config.BASE_TIMEFRAME,
    period: str = config.BASE_PERIOD,
) -> pd.DataFrame:
    """Fetch OHLCV data from yfinance and return with ATR column added."""
    ticker = yf.Ticker(symbol)
    df = ticker.history(period=period, interval=timeframe)
    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_convert("UTC").tz_localize(None)
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.columns = ["open", "high", "low", "close", "volume"]
    df.dropna(inplace=True)
    df = add_atr(df, config.ATR_PERIOD)
    return df


def resample_ohlcv(df_base: pd.DataFrame, target_tf: str) -> pd.DataFrame:
    """Resample a base OHLCV dataframe to a higher timeframe and recompute ATR."""
    rule = _tf_to_pandas_rule(target_tf)
    df = (
        df_base.resample(rule)
        .agg({
            "open":   "first",
            "high":   "max",
            "low":    "min",
            "close":  "last",
            "volume": "sum",
        })
        .dropna()
    )
    df = add_atr(df, config.ATR_PERIOD)
    return df


def add_atr(df: pd.DataFrame, period: int = config.ATR_PERIOD) -> pd.DataFrame:
    """Compute ATR using Wilder's EMA and append as the 'atr' column."""
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


def _tf_to_pandas_rule(tf: str) -> str:
    """Convert a yfinance-style interval string to a pandas resample offset alias."""
    mapping: dict = {
        "1m":  "1min",
        "2m":  "2min",
        "5m":  "5min",
        "15m": "15min",
        "30m": "30min",
        "1h":  "1h",
        "4h":  "4h",
        "1d":  "1D",
    }
    return mapping.get(tf, tf)
