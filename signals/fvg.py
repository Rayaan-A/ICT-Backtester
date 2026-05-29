"""
Fair Value Gap (FVG) detection with full lifecycle state tracking.

State machine for each FVG
──────────────────────────
Bullish FVG  (support zone: bottom = high[i-2], top = low[i])
  UNFILLED     → price has not yet entered the zone
  RESPECTED    → price entered (low ≤ top) and closed back above bottom
  DISRESPECTED → price closed below bottom  →  becomes a bearish IFVG

Bearish FVG  (resistance zone: bottom = high[i], top = low[i-2])
  UNFILLED     → price has not yet entered the zone
  RESPECTED    → price entered (high ≥ bottom) and closed back below top
  DISRESPECTED → price closed above top  →  becomes a bullish IFVG
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

import pandas as pd

import config


# ── State enum ────────────────────────────────────────────────────────────────

class FVGState(str, Enum):
    UNFILLED     = "unfilled"
    RESPECTED    = "respected"
    DISRESPECTED = "disrespected"   # price closed through the zone → IFVG


# ── Dataclass ─────────────────────────────────────────────────────────────────

@dataclass
class FVG:
    """A Fair Value Gap with lifecycle state."""

    index:     int                        # index of the middle candle (i-1)
    timestamp: pd.Timestamp
    direction: str                        # 'bullish' or 'bearish'
    top:       float
    bottom:    float
    timeframe: str

    state:           FVGState               = FVGState.UNFILLED
    state_index:     Optional[int]          = None
    state_timestamp: Optional[pd.Timestamp] = None
    is_ifvg:         bool                   = False   # True when DISRESPECTED


# ── Detection ─────────────────────────────────────────────────────────────────

def detect_fvgs(df: pd.DataFrame, timeframe: str = "5m") -> List[FVG]:
    """
    Scan *df* for all bullish and bearish FVGs, then walk forward through
    price to determine whether each one was respected or disrespected.
    """
    fvgs: List[FVG] = []

    for i in range(2, len(df)):
        atr     = df["atr"].iloc[i] if "atr" in df.columns else 0.0
        min_gap = atr * config.FVG_MIN_SIZE_ATR

        # Bullish FVG: gap between high[i-2] and low[i]
        bullish_gap = df["low"].iloc[i] - df["high"].iloc[i - 2]
        if bullish_gap > min_gap:
            fvgs.append(FVG(
                index     = i - 1,
                timestamp = df.index[i - 1],
                direction = "bullish",
                top       = df["low"].iloc[i],
                bottom    = df["high"].iloc[i - 2],
                timeframe = timeframe,
            ))

        # Bearish FVG: gap between low[i-2] and high[i]
        bearish_gap = df["low"].iloc[i - 2] - df["high"].iloc[i]
        if bearish_gap > min_gap:
            fvgs.append(FVG(
                index     = i - 1,
                timestamp = df.index[i - 1],
                direction = "bearish",
                top       = df["low"].iloc[i - 2],
                bottom    = df["high"].iloc[i],
                timeframe = timeframe,
            ))

    _update_states(df, fvgs)
    return fvgs


# ── State machine ─────────────────────────────────────────────────────────────

def _update_states(df: pd.DataFrame, fvgs: List[FVG]) -> None:
    """Walk forward for each FVG and assign its final state."""
    for fvg in fvgs:
        entered = False   # True once price has touched the zone

        for i in range(fvg.index + 2, len(df)):
            hi    = df["high"].iloc[i]
            lo    = df["low"].iloc[i]
            close = df["close"].iloc[i]

            if fvg.direction == "bullish":
                if lo <= fvg.top:
                    entered = True
                if entered:
                    if close < fvg.bottom:
                        # Price closed through support → bearish IFVG
                        _set(fvg, FVGState.DISRESPECTED, i, df.index[i], is_ifvg=True)
                        break
                    if close >= fvg.bottom:
                        # Price bounced within zone → respected
                        _set(fvg, FVGState.RESPECTED, i, df.index[i])
                        break

            else:  # bearish
                if hi >= fvg.bottom:
                    entered = True
                if entered:
                    if close > fvg.top:
                        # Price closed through resistance → bullish IFVG
                        _set(fvg, FVGState.DISRESPECTED, i, df.index[i], is_ifvg=True)
                        break
                    if close <= fvg.top:
                        # Price rejected within zone → respected
                        _set(fvg, FVGState.RESPECTED, i, df.index[i])
                        break


def _set(
    fvg:     FVG,
    state:   FVGState,
    idx:     int,
    ts:      pd.Timestamp,
    is_ifvg: bool = False,
) -> None:
    """Mutate fvg in-place with state outcome."""
    fvg.state           = state
    fvg.state_index     = idx
    fvg.state_timestamp = ts
    fvg.is_ifvg         = is_ifvg
