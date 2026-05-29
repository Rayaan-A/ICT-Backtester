"""
Kill-zone and trading-session helpers.

yfinance futures data is returned as UTC-naive timestamps.
All conversions go through America/New_York so that London and NY kill
zones align with the clock the user sees on TradingView.
"""

from datetime import time as dtime
from typing import Optional

import pandas as pd
import pytz

import config

_ET = pytz.timezone("America/New_York")


def _to_et_time(ts: pd.Timestamp) -> dtime:
    """Return the wall-clock time in ET for a UTC-naive (or tz-aware) timestamp."""
    if ts.tzinfo is None:
        ts_aware = pytz.utc.localize(ts)
    else:
        ts_aware = ts
    return ts_aware.astimezone(_ET).time()


def session_at(ts: pd.Timestamp) -> Optional[str]:
    """
    Return the kill-zone name active at *ts*, or None if outside all zones.

    Zones are defined in config.KILL_ZONES as (start, end) strings in
    24-hour America/New_York time.  Zones that cross midnight are supported.
    """
    t = _to_et_time(ts)

    for name, (start_str, end_str) in config.KILL_ZONES.items():
        sh, sm = map(int, start_str.split(":"))
        eh, em = map(int, end_str.split(":"))
        s = dtime(sh, sm)
        e = dtime(eh, em)

        if s <= e:          # same-day zone (e.g. 09:30 – 11:00)
            if s <= t < e:
                return name
        else:               # crosses midnight (e.g. 20:00 – 00:00)
            if t >= s or t < e:
                return name

    return None


def is_reversal_session(ts: pd.Timestamp) -> bool:
    """True if *ts* falls within a session valid for reversal trades."""
    return session_at(ts) in config.REVERSAL_SESSIONS


def is_continuation_session(ts: pd.Timestamp) -> bool:
    """True if *ts* falls within a session valid for continuation trades."""
    return session_at(ts) in config.CONTINUATION_SESSIONS
