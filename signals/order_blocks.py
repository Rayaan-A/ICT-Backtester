from dataclasses import dataclass
from typing import List, Optional

import pandas as pd

import config


@dataclass
class OrderBlock:
    """The last opposing candle before a displacement move."""

    index: int               # candle index of the OB itself
    timestamp: pd.Timestamp
    direction: str           # 'bullish' or 'bearish'
    top: float
    bottom: float
    displaced_by_index: int  # index of the displacement candle
    mitigated: bool = False
    mitigation_index: Optional[int] = None


def detect_order_blocks(df: pd.DataFrame) -> List[OrderBlock]:
    """Detect bullish and bearish order blocks and mark mitigated ones."""
    obs: List[OrderBlock] = []

    for i in range(1, len(df)):
        candle_range = df["high"].iloc[i] - df["low"].iloc[i]
        atr = df["atr"].iloc[i]

        if candle_range < atr * config.OB_DISPLACEMENT_ATR:
            continue

        is_bullish_displacement = df["close"].iloc[i] > df["open"].iloc[i]
        is_bearish_displacement = df["close"].iloc[i] < df["open"].iloc[i]

        if is_bullish_displacement:
            # Last bearish candle before this move is the bullish OB
            for j in range(i - 1, -1, -1):
                if df["close"].iloc[j] < df["open"].iloc[j]:
                    obs.append(
                        OrderBlock(
                            index=j,
                            timestamp=df.index[j],
                            direction="bullish",
                            top=df["high"].iloc[j],
                            bottom=df["low"].iloc[j],
                            displaced_by_index=i,
                        )
                    )
                    break

        elif is_bearish_displacement:
            # Last bullish candle before this move is the bearish OB
            for j in range(i - 1, -1, -1):
                if df["close"].iloc[j] > df["open"].iloc[j]:
                    obs.append(
                        OrderBlock(
                            index=j,
                            timestamp=df.index[j],
                            direction="bearish",
                            top=df["high"].iloc[j],
                            bottom=df["low"].iloc[j],
                            displaced_by_index=i,
                        )
                    )
                    break

    _mark_mitigated(df, obs)
    return obs


def _mark_mitigated(df: pd.DataFrame, obs: List[OrderBlock]) -> None:
    """Mutate OBs in-place: bullish mitigated on close below its low, bearish on close above its high."""
    for ob in obs:
        start = ob.displaced_by_index + 1
        for i in range(start, len(df)):
            if ob.direction == "bullish" and df["close"].iloc[i] < ob.bottom:
                ob.mitigated = True
                ob.mitigation_index = i
                break
            if ob.direction == "bearish" and df["close"].iloc[i] > ob.top:
                ob.mitigated = True
                ob.mitigation_index = i
                break
