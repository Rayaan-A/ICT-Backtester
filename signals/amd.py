"""
AMD (Accumulation → Manipulation → Distribution) model.

For each trading session (London, NY AM):
  1. ACCUMULATION — price consolidates in the pre-session window defined in
     config.ACCUMULATION_WINDOWS, forming a clear high and low.
  2. MANIPULATION — within the first AMD_MANIPULATION_WINDOW_BARS bars of the
     session, price sweeps one side of the accumulation range (the Judas swing).
  3. DISTRIBUTION — after the sweep, price reverses and trends in the opposite
     direction.  The engine looks for an IFVG entry during this phase.

Only sweeps of the *accumulation range* are returned — not random mid-session
swing sweeps.  This makes reversals far more precise than generic detect_sweeps.
"""

import datetime
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd
import pytz

import config

_ET = pytz.timezone("America/New_York")


# ── Dataclasses ───────────────────────────────────────────────────────────────

@dataclass
class AccumulationRange:
    """The consolidation range that forms before a kill-zone session."""
    session:      str
    date:         datetime.date          # calendar date of the *session* (not acc window)
    high:         float
    low:          float
    session_open: pd.Timestamp           # UTC-naive timestamp when the kill zone opens


@dataclass
class AMDManipulation:
    """A Judas sweep of the accumulation range at session open — the M phase."""
    index:        int
    timestamp:    pd.Timestamp
    session:      str
    direction:    str                    # 'high' → swept highs (short dist), 'low' → swept lows (long dist)
    sweep_level:  float                  # the acc high/low that was taken out
    candle_high:  float
    candle_low:   float
    acc_range:    AccumulationRange


# ── Public API ────────────────────────────────────────────────────────────────

def detect_amd(df: pd.DataFrame) -> List[AMDManipulation]:
    """
    Full AMD detection pipeline on *df* (5m base-TF OHLCV with ATR column).
    Returns all confirmed Judas sweeps across all sessions in the data.
    """
    acc_ranges   = _compute_accumulation_ranges(df)
    manipulations = _detect_manipulations(df, acc_ranges)
    return manipulations


# ── Accumulation range detection ──────────────────────────────────────────────

def _compute_accumulation_ranges(df: pd.DataFrame) -> List[AccumulationRange]:
    """
    For each session defined in config.ACCUMULATION_WINDOWS, find the
    OHLCV range of the pre-session window and record its high/low.
    """
    if df.index.tzinfo is None:
        idx_et = df.index.tz_localize("UTC").tz_convert(_ET)
    else:
        idx_et = df.index.tz_convert(_ET)

    ranges: List[AccumulationRange] = []

    for session, (acc_start_str, acc_end_str) in config.ACCUMULATION_WINDOWS.items():
        acc_start_t = pd.Timestamp(acc_start_str).time()
        acc_end_t   = pd.Timestamp(acc_end_str).time()
        sess_open_t = pd.Timestamp(config.KILL_ZONES[session][0]).time()

        # Group bars that fall in the accumulation window
        # Handle windows that cross midnight (e.g. 22:00–02:00)
        crosses_midnight = acc_start_t > acc_end_t

        if crosses_midnight:
            in_acc = (idx_et.time >= acc_start_t) | (idx_et.time < acc_end_t)
        else:
            in_acc = (idx_et.time >= acc_start_t) & (idx_et.time < acc_end_t)

        acc_bars = df[in_acc].copy()
        acc_bars_et = idx_et[in_acc]

        if acc_bars.empty:
            continue

        # Each accumulation window belongs to the *session date* it precedes.
        # For windows that cross midnight, the session date is the day after acc start.
        def _session_date(et_ts: pd.Timestamp) -> datetime.date:
            if crosses_midnight and et_ts.time() >= acc_start_t:
                return (et_ts + datetime.timedelta(days=1)).date()
            return et_ts.date()

        acc_bars["_sess_date"] = [_session_date(t) for t in acc_bars_et]

        for sess_date, group in acc_bars.groupby("_sess_date"):
            if len(group) < 3:
                continue

            acc_high = group["high"].max()
            acc_low  = group["low"].min()

            # Build the UTC-naive session open timestamp
            naive_open = pd.Timestamp(
                datetime.datetime.combine(sess_date, sess_open_t),
            )
            # Convert ET→UTC→naive
            et_open   = _ET.localize(naive_open.to_pydatetime())
            utc_open  = et_open.astimezone(pytz.utc).replace(tzinfo=None)
            sess_open = pd.Timestamp(utc_open)

            ranges.append(AccumulationRange(
                session      = session,
                date         = sess_date,
                high         = acc_high,
                low          = acc_low,
                session_open = sess_open,
            ))

    return sorted(ranges, key=lambda r: r.session_open)


# ── Manipulation detection ────────────────────────────────────────────────────

def _detect_manipulations(
    df: pd.DataFrame,
    acc_ranges: List[AccumulationRange],
) -> List[AMDManipulation]:
    """
    For each accumulation range, scan the first AMD_MANIPULATION_WINDOW_BARS
    bars of the session for a wick past the accumulation high or low.
    """
    manipulations: List[AMDManipulation] = []

    for acc in acc_ranges:
        # Find the bar index just at or after session open
        open_locs = df.index.searchsorted(acc.session_open)
        if open_locs >= len(df):
            continue

        end_bar = min(open_locs + config.AMD_MANIPULATION_WINDOW_BARS, len(df))

        swept_high = False
        swept_low  = False

        for i in range(open_locs, end_bar):
            candle = df.iloc[i]
            hi, lo = candle["high"], candle["low"]

            # Swept the accumulation HIGH (bearish Judas swing → short distribution)
            if not swept_high and hi > acc.high:
                manipulations.append(AMDManipulation(
                    index       = i,
                    timestamp   = df.index[i],
                    session     = acc.session,
                    direction   = "high",
                    sweep_level = acc.high,
                    candle_high = hi,
                    candle_low  = lo,
                    acc_range   = acc,
                ))
                swept_high = True

            # Swept the accumulation LOW (bullish Judas swing → long distribution)
            elif not swept_low and lo < acc.low:
                manipulations.append(AMDManipulation(
                    index       = i,
                    timestamp   = df.index[i],
                    session     = acc.session,
                    direction   = "low",
                    sweep_level = acc.low,
                    candle_high = hi,
                    candle_low  = lo,
                    acc_range   = acc,
                ))
                swept_low = True

    return sorted(manipulations, key=lambda m: m.index)
