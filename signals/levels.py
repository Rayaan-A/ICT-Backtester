"""
Key liquidity levels: Prior Day High/Low and prior session High/Low.

ICT draws on liquidity cluster around these levels — price is far more
likely to reach a PDH/PDL than a random swing point.

RTH session: 09:30 – 16:00 America/New_York.
A bar at index i receives the PDH/PDL from the last *completed* RTH day,
so there is no lookahead bias.
"""

from typing import Optional

import pandas as pd
import pytz

_ET = pytz.timezone("America/New_York")
_RTH_START = "09:30"
_RTH_END   = "16:00"


def compute_levels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a DataFrame (same index as *df*) with columns:
        pdh, pdl          — prior RTH day high / low
        psh, psl          — prior session (London or NY AM) high / low
    All values are forward-filled so every bar has a level.
    NaN until the first full session has completed.
    """
    if df.index.tzinfo is None:
        idx_et = df.index.tz_localize("UTC").tz_convert(_ET)
    else:
        idx_et = df.index.tz_convert(_ET)

    df_et = df.copy()
    df_et.index = idx_et

    # ── Prior Day High / Low (RTH only) ──────────────────────────────────
    rth_mask = (
        (idx_et.time >= pd.Timestamp(_RTH_START).time()) &
        (idx_et.time <  pd.Timestamp(_RTH_END).time())
    )
    rth = df_et[rth_mask]
    daily_hl = rth.groupby(rth.index.date).agg(day_high=("high", "max"), day_low=("low", "min"))

    # Map each bar to yesterday's completed RTH session
    bar_dates   = pd.Series(idx_et.date, index=df.index)
    prior_dates = bar_dates.map(lambda d: _prior_date(d, daily_hl.index))

    pdh = prior_dates.map(lambda d: daily_hl.loc[d, "day_high"] if d is not None else float("nan"))
    pdl = prior_dates.map(lambda d: daily_hl.loc[d, "day_low"]  if d is not None else float("nan"))

    # ── Prior Session High / Low ──────────────────────────────────────────
    # Sessions: London 02:00-05:00 ET, NY AM 09:30-11:00 ET
    sessions = [
        ("london", "02:00", "05:00"),
        ("ny_am",  "09:30", "11:00"),
    ]
    session_highs: pd.Series = pd.Series(float("nan"), index=df.index)
    session_lows:  pd.Series = pd.Series(float("nan"), index=df.index)

    for _name, start_str, end_str in sessions:
        mask = (
            (idx_et.time >= pd.Timestamp(start_str).time()) &
            (idx_et.time <  pd.Timestamp(end_str).time())
        )
        sess = df_et[mask]
        if sess.empty:
            continue

        # Group by date within each session
        sess_hl = sess.groupby(sess.index.date).agg(
            sess_high=("high", "max"),
            sess_low=("low",  "min"),
        )

        # Each bar gets the *prior completed* session high/low
        for orig_idx, date in zip(df.index[mask], idx_et[mask].date):
            prior = _prior_date(date, sess_hl.index)
            if prior is not None:
                # Only overwrite if this session's prior level is closer
                ph = sess_hl.loc[prior, "sess_high"]
                pl = sess_hl.loc[prior, "sess_low"]
                if pd.isna(session_highs[orig_idx]) or abs(ph - df.loc[orig_idx, "close"]) < abs(session_highs[orig_idx] - df.loc[orig_idx, "close"]):
                    session_highs[orig_idx] = ph
                if pd.isna(session_lows[orig_idx]) or abs(pl - df.loc[orig_idx, "close"]) < abs(session_lows[orig_idx] - df.loc[orig_idx, "close"]):
                    session_lows[orig_idx] = pl

    session_highs = session_highs.ffill()
    session_lows  = session_lows.ffill()

    return pd.DataFrame({
        "pdh": pdh.values,
        "pdl": pdl.values,
        "psh": session_highs.values,
        "psl": session_lows.values,
    }, index=df.index)


def nearest_key_level_above(
    levels: pd.DataFrame,
    price: float,
    i: int,
    min_distance: float = 0.0,
    max_distance: float = float("inf"),
) -> Optional[float]:
    """
    Return the closest PDH or prior session high above *price* at bar *i*
    within the distance band [min_distance, max_distance].
    Returns None if none qualify.
    """
    row = levels.iloc[i]
    candidates = []
    for col in ("pdh", "psh"):
        v = row[col]
        if not pd.isna(v) and price + min_distance < v <= price + max_distance:
            candidates.append(v)
    return min(candidates) if candidates else None


def nearest_key_level_below(
    levels: pd.DataFrame,
    price: float,
    i: int,
    min_distance: float = 0.0,
    max_distance: float = float("inf"),
) -> Optional[float]:
    """
    Return the closest PDL or prior session low below *price* at bar *i*
    within the distance band [min_distance, max_distance].
    Returns None if none qualify.
    """
    row = levels.iloc[i]
    candidates = []
    for col in ("pdl", "psl"):
        v = row[col]
        if not pd.isna(v) and price - max_distance <= v < price - min_distance:
            candidates.append(v)
    return max(candidates) if candidates else None


# ── Internal helpers ──────────────────────────────────────────────────────────

def _prior_date(date, available_dates):
    """Return the most recent date in *available_dates* that is strictly before *date*."""
    prior = [d for d in available_dates if d < date]
    return max(prior) if prior else None
