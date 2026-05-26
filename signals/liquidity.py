from dataclasses import dataclass
from typing import List

import pandas as pd

import config


@dataclass
class LiquiditySweep:
    """A wick that pierces a swing high/low but closes back inside the range."""

    index: int
    timestamp: pd.Timestamp
    direction: str    # 'high' = swept highs (bearish signal), 'low' = swept lows (bullish signal)
    sweep_level: float
    candle_high: float
    candle_low: float


def detect_sweeps(df: pd.DataFrame) -> List[LiquiditySweep]:
    """Detect liquidity sweeps of swing highs and swing lows."""
    sweeps: List[LiquiditySweep] = []
    lookback = config.LIQ_LOOKBACK

    for i in range(lookback, len(df)):
        window = df.iloc[i - lookback : i]
        swing_high = window["high"].max()
        swing_low = window["low"].min()

        high = df["high"].iloc[i]
        low = df["low"].iloc[i]
        close = df["close"].iloc[i]

        # Wick above swing high, close back below it
        if high > swing_high and close < swing_high:
            sweeps.append(
                LiquiditySweep(
                    index=i,
                    timestamp=df.index[i],
                    direction="high",
                    sweep_level=swing_high,
                    candle_high=high,
                    candle_low=low,
                )
            )
        # Wick below swing low, close back above it
        elif low < swing_low and close > swing_low:
            sweeps.append(
                LiquiditySweep(
                    index=i,
                    timestamp=df.index[i],
                    direction="low",
                    sweep_level=swing_low,
                    candle_high=high,
                    candle_low=low,
                )
            )

    return sweeps
