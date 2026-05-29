"""
Turtle Soup detection.

A turtle soup is a failed breakout of a confirmed swing high or low:
  Bullish: price wicks BELOW a confirmed swing low but closes back ABOVE it
           → fake-out of sell-side liquidity → bullish distribution expected
  Bearish: price wicks ABOVE a confirmed swing high but closes back BELOW it
           → fake-out of buy-side liquidity → bearish distribution expected

This is the micro-level "manipulation" signal the PB Blake model uses to
sponsor continuation entries when no session-open AMD sweep is present.
"""

from dataclasses import dataclass
from typing import List

import pandas as pd

import config
from signals.swing import SwingPoint


@dataclass
class TurtleSoup:
    """A failed breakout of a confirmed swing high or low."""
    index:       int
    timestamp:   pd.Timestamp
    direction:   str    # 'bullish' (swept lows → long) or 'bearish' (swept highs → short)
    sweep_level: float
    candle_high: float
    candle_low:  float


def detect_turtle_soups(
    df:     pd.DataFrame,
    swings: List[SwingPoint],
) -> List[TurtleSoup]:
    """
    Detect turtle soups using confirmed swing points (no lookahead).

    For each bar i, find the nearest confirmed swing low below price and the
    nearest confirmed swing high above price — both confirmed before bar i.
    If the bar wicks past one and closes back inside, record a turtle soup.

    Args:
        df:     5m OHLCV DataFrame.
        swings: Pre-computed swing points from detect_swings().

    Returns:
        List of TurtleSoup events in chronological order.
    """
    soups: List[TurtleSoup] = []
    min_start = config.SWING_LOOKBACK * 2 + 2

    for i in range(min_start, len(df)):
        hi    = df["high"].iloc[i]
        lo    = df["low"].iloc[i]
        close = df["close"].iloc[i]

        # Swings confirmed before this bar (confirmed_index = swing.index + SWING_LOOKBACK <= i)
        confirmed_lows  = [
            s for s in swings
            if s.swing_type == "low"
            and s.index + config.SWING_LOOKBACK <= i
        ]
        confirmed_highs = [
            s for s in swings
            if s.swing_type == "high"
            and s.index + config.SWING_LOOKBACK <= i
        ]

        # Bullish turtle soup: wick below nearest confirmed swing low, close back above
        if confirmed_lows:
            nearest_low = max(confirmed_lows, key=lambda s: s.price)   # highest swing low ≈ nearest below
            if lo < nearest_low.price and close > nearest_low.price:
                soups.append(TurtleSoup(
                    index       = i,
                    timestamp   = df.index[i],
                    direction   = "bullish",
                    sweep_level = nearest_low.price,
                    candle_high = hi,
                    candle_low  = lo,
                ))

        # Bearish turtle soup: wick above nearest confirmed swing high, close back below
        if confirmed_highs:
            nearest_high = min(confirmed_highs, key=lambda s: s.price)  # lowest swing high ≈ nearest above
            if hi > nearest_high.price and close < nearest_high.price:
                soups.append(TurtleSoup(
                    index       = i,
                    timestamp   = df.index[i],
                    direction   = "bearish",
                    sweep_level = nearest_high.price,
                    candle_high = hi,
                    candle_low  = lo,
                ))

    return sorted(soups, key=lambda s: s.index)
