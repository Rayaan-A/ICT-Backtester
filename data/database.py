import os
import sqlite3
from typing import Optional

import pandas as pd

import config


def _ensure_dir() -> None:
    """Create the data directory if it does not exist."""
    os.makedirs(os.path.dirname(config.DB_PATH), exist_ok=True)


def _table_name(symbol: str, timeframe: str) -> str:
    """Build a safe SQLite table name from symbol and timeframe."""
    safe = symbol.replace("=", "_").replace("/", "_").replace("-", "_")
    return f"ohlcv_{safe}_{timeframe}"


def save_ohlcv(df: pd.DataFrame, symbol: str, timeframe: str) -> None:
    """Persist an OHLCV dataframe to SQLite, replacing any existing data."""
    _ensure_dir()
    table = _table_name(symbol, timeframe)
    with sqlite3.connect(config.DB_PATH) as conn:
        df.to_sql(table, conn, if_exists="replace", index=True, index_label="datetime")


def load_ohlcv(symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
    """Load an OHLCV dataframe from SQLite. Returns None if the table does not exist."""
    if not os.path.exists(config.DB_PATH):
        return None
    table = _table_name(symbol, timeframe)
    try:
        with sqlite3.connect(config.DB_PATH) as conn:
            df = pd.read_sql(
                f"SELECT * FROM [{table}]",
                conn,
                index_col="datetime",
                parse_dates=["datetime"],
            )
        return df
    except Exception:
        return None
