"""
Swing high / swing low detection.

A bar at index i is a confirmed swing high when its high is the highest
value in [i-lookback .. i+lookback].  Swing lows are analogous.

Because we need `lookback` bars to the right to confirm a swing, these
points are only usable in the backtester once bar  i + lookback  has
passed — the helpers below enforce this via the `confirmed_before`
argument (pass the current bar index i).
"""

from dataclasses import dataclass
from typing import List, Optional

import pandas as pd

import config


# ── Dataclass ─────────────────────────────────────────────────────────────────

@dataclass
class SwingPoint:
    """A confirmed swing high or swing low."""
    index:      int
    timestamp:  pd.Timestamp
    price:      float
    swing_type: str   # 'high' or 'low'


# ── Detection ─────────────────────────────────────────────────────────────────

def detect_swings(
    df: pd.DataFrame,
    lookback: int = config.SWING_LOOKBACK,
) -> List[SwingPoint]:
    """Return all confirmed swing highs and lows in *df*."""
    swings: List[SwingPoint] = []
    n = len(df)

    for i in range(lookback, n - lookback):
        window_h = df["high"].iloc[i - lookback : i + lookback + 1]
        window_l = df["low"].iloc[i - lookback : i + lookback + 1]

        if df["high"].iloc[i] >= window_h.max():
            swings.append(SwingPoint(
                index      = i,
                timestamp  = df.index[i],
                price      = df["high"].iloc[i],
                swing_type = "high",
            ))

        if df["low"].iloc[i] <= window_l.min():
            swings.append(SwingPoint(
                index      = i,
                timestamp  = df.index[i],
                price      = df["low"].iloc[i],
                swing_type = "low",
            ))

    return swings


# ── Target helpers (draw on liquidity) ────────────────────────────────────────

def nearest_target_above(
    swings: List[SwingPoint],
    price: float,
    confirmed_before: int,
) -> Optional[float]:
    """
    Nearest confirmed swing HIGH above *price* that was confirmed before
    bar *confirmed_before* (no lookahead bias).
    Returns the price level, or None if not found.
    """
    candidates = [
        s for s in swings
        if s.swing_type == "high"
        and s.price > price
        and s.index + config.SWING_LOOKBACK <= confirmed_before
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda s: s.price).price


def nearest_target_below(
    swings: List[SwingPoint],
    price: float,
    confirmed_before: int,
) -> Optional[float]:
    """
    Nearest confirmed swing LOW below *price* that was confirmed before
    bar *confirmed_before*.
    Returns the price level, or None if not found.
    """
    candidates = [
        s for s in swings
        if s.swing_type == "low"
        and s.price < price
        and s.index + config.SWING_LOOKBACK <= confirmed_before
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda s: s.price).price


# ── Stop-loss helpers ─────────────────────────────────────────────────────────

def nearest_swing_low_below(
    swings: List[SwingPoint],
    price: float,
    confirmed_before: int,
) -> Optional[float]:
    """Nearest confirmed swing low below *price* — used for long SL placement."""
    return nearest_target_below(swings, price, confirmed_before)


def nearest_swing_high_above(
    swings: List[SwingPoint],
    price: float,
    confirmed_before: int,
) -> Optional[float]:
    """Nearest confirmed swing high above *price* — used for short SL placement."""
    return nearest_target_above(swings, price, confirmed_before)
