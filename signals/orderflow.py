"""
Order flow analysis using OHLCV data.

Since yfinance provides OHLCV (not tick data), we approximate:
  - Delta: estimated buy/sell pressure from close position within the bar's range.
           delta = volume × (2 × (close − low) / (high − low) − 1)
           → +volume when close == high (all buying), −volume when close == low (all selling)
  - Session VWAP: typical-price weighted average that resets at midnight ET each day,
                  with ±1σ and ±2σ standard deviation bands.
  - Cumulative delta: running net buy/sell pressure since the day's session open.
  - Volume imbalance: bars where volume exceeds the rolling average by a threshold.
"""

from typing import Optional

import numpy as np
import pandas as pd
import pytz

import config

_ET = pytz.timezone("America/New_York")


# ── ET date helper ────────────────────────────────────────────────────────────

def _et_date(ts: pd.Timestamp) -> str:
    """Return the ET calendar date of *ts* as an ISO string for groupby keys."""
    if ts.tzinfo is None:
        ts_aware = pytz.utc.localize(ts)
    else:
        ts_aware = ts
    et_dt = ts_aware.astimezone(_ET)
    return f"{et_dt.year}-{et_dt.month:02d}-{et_dt.day:02d}"


# ── Core computation ──────────────────────────────────────────────────────────

def compute_orderflow(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-bar order flow metrics from OHLCV data.

    Args:
        df: OHLCV DataFrame with columns [open, high, low, close, volume].
            Index must be UTC-naive or tz-aware timestamps.

    Returns:
        DataFrame with the same index as *df* and columns:
            vwap         – session VWAP (resets midnight ET each calendar day)
            vwap_upper1  – VWAP + 1σ
            vwap_lower1  – VWAP − 1σ
            vwap_upper2  – VWAP + 2σ
            vwap_lower2  – VWAP − 2σ
            delta        – estimated buy minus sell volume for the bar
            cum_delta    – cumulative delta since the day's session open
            volume_ratio – bar volume / rolling-average volume
            is_imbalance – True where volume_ratio > VOLUME_IMBALANCE_MULTIPLIER
    """
    typical = (df["high"] + df["low"] + df["close"]) / 3
    volume  = df["volume"].replace(0, np.nan).fillna(1.0)

    # Delta: close position within bar range → buying or selling pressure
    hl_range    = (df["high"] - df["low"]).replace(0, np.nan).ffill()
    delta_ratio = (df["close"] - df["low"]) / hl_range   # 0 (all selling) → 1 (all buying)
    delta       = volume * (2 * delta_ratio - 1)          # range: [−volume, +volume]

    # Group bars by their ET calendar date for session-scoped cumulation
    et_dates = pd.Series(
        [_et_date(ts) for ts in df.index],
        index=df.index,
        name="et_date",
    )

    # Session VWAP = cumsum(typical × volume) / cumsum(volume) per day
    tp_vol    = typical * volume
    cum_tpvol = tp_vol.groupby(et_dates).cumsum()
    cum_vol   = volume.groupby(et_dates).cumsum()
    vwap      = cum_tpvol / cum_vol

    # VWAP standard deviation: sqrt(cumsum((typical − vwap)² × volume) / cumsum(volume))
    sq_dev  = ((typical - vwap) ** 2 * volume).groupby(et_dates).cumsum()
    std_dev = (sq_dev / cum_vol).apply(np.sqrt)

    vwap_upper1 = vwap + 1.0 * std_dev
    vwap_lower1 = vwap - 1.0 * std_dev
    vwap_upper2 = vwap + 2.0 * std_dev
    vwap_lower2 = vwap - 2.0 * std_dev

    # Cumulative delta per session day
    cum_delta = delta.groupby(et_dates).cumsum()

    # Volume imbalance flag
    avg_vol      = volume.rolling(config.VOLUME_IMBALANCE_LOOKBACK, min_periods=1).mean()
    volume_ratio = volume / avg_vol
    is_imbalance = volume_ratio > config.VOLUME_IMBALANCE_MULTIPLIER

    return pd.DataFrame(
        {
            "vwap":         vwap,
            "vwap_upper1":  vwap_upper1,
            "vwap_lower1":  vwap_lower1,
            "vwap_upper2":  vwap_upper2,
            "vwap_lower2":  vwap_lower2,
            "delta":        delta,
            "cum_delta":    cum_delta,
            "volume_ratio": volume_ratio,
            "is_imbalance": is_imbalance,
        },
        index=df.index,
    )


# ── Entry confirmation ────────────────────────────────────────────────────────

def is_order_flow_confirmed(
    of:          pd.DataFrame,
    i:           int,
    direction:   str,
    entry_price: float,
) -> bool:
    """
    Return True if order flow at bar *i* supports a trade in *direction*.

    Two conditions must both hold:

    Long confirmation:
      1. Cumulative delta is higher now than ORDER_FLOW_DELTA_WINDOW bars ago
         → net buying pressure is building within the session.
      2. Entry price is at or above the VWAP −1σ band
         → not chasing price deep into bearish VWAP territory.

    Short confirmation:
      1. Cumulative delta is lower now than ORDER_FLOW_DELTA_WINDOW bars ago
         → net selling pressure is building.
      2. Entry price is at or below the VWAP +1σ band
         → not shorting into bullish VWAP territory.

    Falls back to True when there are too few bars for the delta window,
    so early-session setups are not unfairly excluded.
    """
    window = config.ORDER_FLOW_DELTA_WINDOW

    if i < window:
        return True  # not enough history; don't penalise early bars

    cum_now  = of["cum_delta"].iloc[i]
    cum_prev = of["cum_delta"].iloc[i - window]

    if direction == "long":
        delta_ok = cum_now > cum_prev
        vwap_ok  = entry_price >= of["vwap_lower1"].iloc[i]
    else:
        delta_ok = cum_now < cum_prev
        vwap_ok  = entry_price <= of["vwap_upper1"].iloc[i]

    return delta_ok and vwap_ok
