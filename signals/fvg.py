from dataclasses import dataclass, field
from typing import List, Optional

import pandas as pd

import config


@dataclass
class FVG:
    """A Fair Value Gap: the imbalance zone between candle[i-2] and candle[i]."""

    index: int                    # index of the middle candle (i-1)
    timestamp: pd.Timestamp
    direction: str                # 'bullish' or 'bearish'
    top: float
    bottom: float
    filled: bool = False
    fill_index: Optional[int] = None


def detect_fvgs(df: pd.DataFrame) -> List[FVG]:
    """Detect all bullish and bearish FVGs and mark which ones are filled."""
    fvgs: List[FVG] = []

    for i in range(2, len(df)):
        atr = df["atr"].iloc[i]
        min_gap = atr * config.FVG_MIN_SIZE_ATR

        # Bullish FVG: gap between top of candle[i-2] and bottom of candle[i]
        bullish_gap = df["low"].iloc[i] - df["high"].iloc[i - 2]
        if bullish_gap > min_gap:
            fvgs.append(
                FVG(
                    index=i - 1,
                    timestamp=df.index[i - 1],
                    direction="bullish",
                    top=df["low"].iloc[i],
                    bottom=df["high"].iloc[i - 2],
                )
            )

        # Bearish FVG: gap between bottom of candle[i-2] and top of candle[i]
        bearish_gap = df["low"].iloc[i - 2] - df["high"].iloc[i]
        if bearish_gap > min_gap:
            fvgs.append(
                FVG(
                    index=i - 1,
                    timestamp=df.index[i - 1],
                    direction="bearish",
                    top=df["low"].iloc[i - 2],
                    bottom=df["high"].iloc[i],
                )
            )

    _mark_filled(df, fvgs)
    return fvgs


def _mark_filled(df: pd.DataFrame, fvgs: List[FVG]) -> None:
    """Mutate FVGs in-place, marking fill_index when price trades through the gap."""
    for fvg in fvgs:
        for i in range(fvg.index + 2, len(df)):
            if fvg.direction == "bullish" and df["low"].iloc[i] <= fvg.bottom:
                fvg.filled = True
                fvg.fill_index = i
                break
            if fvg.direction == "bearish" and df["high"].iloc[i] >= fvg.top:
                fvg.filled = True
                fvg.fill_index = i
                break
