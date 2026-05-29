"""
Market structure bias via Break of Structure (BOS) tracking.

Rules (applied to confirmed swing points on the given timeframe):
  - When price closes above the most recent confirmed swing HIGH → bullish BOS → bias = 'bullish'
  - When price closes below the most recent confirmed swing LOW  → bearish BOS → bias = 'bearish'

The bias at bar i is determined from all bars up to and including i,
so there is no lookahead.  Bias starts as None until the first BOS occurs.

Usage:
    ms = compute_bias(df_15m)   # Series['bullish'|'bearish'|None], same index as df_15m
    bias_at_5m_bar = get_bias(ms, ts_5m)   # look up the last known 15m bias
"""

from typing import List, Optional

import pandas as pd

import config
from signals.swing import SwingPoint, detect_swings


def compute_bias(df: pd.DataFrame, lookback: Optional[int] = None) -> pd.Series:
    """
    Return a Series (same index as *df*) where each value is
    'bullish', 'bearish', or None — the market structure bias as of that bar.

    *lookback* overrides config.SWING_LOOKBACK for swing detection on this df.
    Pass a smaller value (e.g. 2) when working with sparse daily candles.
    """
    swings: List[SwingPoint] = detect_swings(df, lookback=lookback or config.SWING_LOOKBACK)
    bias_values = [None] * len(df)

    # Sort swings by index so we walk forward in time
    swings_sorted = sorted(swings, key=lambda s: s.index)

    current_bias: Optional[str] = None
    last_sh: Optional[float] = None   # most recent confirmed swing high price
    last_sl: Optional[float] = None   # most recent confirmed swing low price
    swing_ptr = 0

    for i in range(len(df)):
        # Absorb all swings confirmed by bar i
        while swing_ptr < len(swings_sorted) and swings_sorted[swing_ptr].index <= i:
            sp = swings_sorted[swing_ptr]
            if sp.swing_type == "high":
                last_sh = sp.price
            else:
                last_sl = sp.price
            swing_ptr += 1

        close = df["close"].iloc[i]

        if last_sh is not None and close > last_sh:
            current_bias = "bullish"
        elif last_sl is not None and close < last_sl:
            current_bias = "bearish"

        bias_values[i] = current_bias

    return pd.Series(bias_values, index=df.index, name="ms_bias")


def get_bias(bias: pd.Series, ts: pd.Timestamp) -> Optional[str]:
    """
    Return the most recent bias value that is at or before *ts*.
    *bias* is a UTC-naive Series; *ts* is compared naively.
    Returns None if no bias is available yet.
    """
    if ts.tzinfo is not None:
        ts = ts.tz_localize(None) if hasattr(ts, "tz_localize") else ts.replace(tzinfo=None)

    idx = bias.index
    if idx.tzinfo is not None:
        idx = idx.tz_localize(None)

    past = idx[idx <= ts]
    if past.empty:
        return None
    return bias.iloc[len(past) - 1]
