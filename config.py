from typing import Dict, List, Tuple

# ── Symbols ───────────────────────────────────────────────────────────────────
SYMBOLS: List[str] = ["NQ=F", "ES=F"]
DEFAULT_SYMBOL: str = "NQ=F"

# ── Multi-timeframe ───────────────────────────────────────────────────────────
# Base TF: the backtester loop runs on 5m candles (60 days available via yfinance)
BASE_TIMEFRAME: str = "5m"
BASE_PERIOD: str = "60d"

# TFs to detect *setup* FVGs on (respected / disrespected)
# Listed highest → lowest; the engine checks them in this order
SETUP_TIMEFRAMES: List[str] = ["15m", "5m"]

# Higher timeframes used purely for bias (market structure direction filter)
# Listed highest → lowest; ALL must agree with the entry direction
HTF_BIAS_TIMEFRAMES: List[str] = ["1d"]

# TFs to look for *entry* IFVGs on (highest TF first = preferred entry)
ENTRY_IFVG_TIMEFRAMES: List[str] = ["5m"]

# ── ATR ───────────────────────────────────────────────────────────────────────
ATR_PERIOD: int = 14

# ── ICT signal parameters ─────────────────────────────────────────────────────
FVG_MIN_SIZE_ATR: float = 0.1    # minimum FVG size as fraction of ATR
LIQ_LOOKBACK: int = 20           # bars back for swing high/low in sweep detection
OB_DISPLACEMENT_ATR: float = 1.5 # kept for order_blocks.py compatibility

# ── Kill zones (America/New_York, 24-hour strings) ────────────────────────────
KILL_ZONES: Dict[str, Tuple[str, str]] = {
    "london": ("02:00", "05:00"),
    "ny_am":  ("09:30", "11:00"),
    "ny_10":  ("10:00", "10:30"),   # intra-session PO3 window; AMD uses 09:30–10:00 range
    "ny_pm":  ("13:00", "16:00"),
}

# Sessions where reversal trades are valid
REVERSAL_SESSIONS: List[str] = ["ny_am", "london", "ny_pm"]

# Sessions where continuation trades are valid (same kill zones)
CONTINUATION_SESSIONS: List[str] = ["ny_am", "london", "ny_pm"]

# Pre-session accumulation windows (America/New_York, 24-hour strings)
# These define the consolidation range that gets swept at each session open
ACCUMULATION_WINDOWS: Dict[str, Tuple[str, str]] = {
    "london": ("00:00", "02:00"),   # Asian session → swept at London open
    "ny_am":  ("07:00", "09:30"),   # post-London pre-market → swept at NY open
    "ny_10":  ("09:30", "10:00"),   # first 30 min of NY → swept at 10:00 open
    "ny_pm":  ("11:00", "13:00"),   # NY midday consolidation → swept at PM open
}

# How many 5m bars into the session the manipulation (Judas sweep) must occur
AMD_MANIPULATION_WINDOW_BARS: int = 12   # first 60 min of session

# Maximum trades allowed per calendar day
MAX_TRADES_PER_DAY: int = 3

# ── Swing detection ───────────────────────────────────────────────────────────
SWING_LOOKBACK: int = 5       # bars on each side required to confirm a swing point (intraday)
HTF_SWING_LOOKBACK: int = 2   # bars on each side for daily/HTF swing confirmation

# ── Engine timing windows (in 5m base-TF bars) ───────────────────────────────
# How recently a setup FVG must have been *respected* to trigger an IFVG search
SETUP_RESPECT_WINDOW_BARS: int = 48   # ~4 hours

# How recently a sweep must have occurred for a reversal entry
REVERSAL_SWEEP_WINDOW_BARS: int = 12  # ~1 hour

# Continuation entries require a recent liquidity sweep as a "sponsor"
# (either an AMD session sweep or a turtle soup of a confirmed swing)
REQUIRE_CONTINUATION_SPONSOR: bool = True
CONTINUATION_SPONSOR_WINDOW_BARS: int = 48  # ~4 hours

# Window to look back for a turtle soup confirmation
TURTLE_SOUP_WINDOW_BARS: int = 12  # ~1 hour

# ── Risk management ───────────────────────────────────────────────────────────
RISK_PER_TRADE: float = 0.01      # fraction of capital risked per trade
INITIAL_CAPITAL: float = 10_000.0
FALLBACK_RR: float = 2.0          # R:R used when no swing target is found
MIN_RR: float = 1.0               # skip trade if best target is closer than this
MAX_RR: float = 2.5               # skip trade if no target within this many R
MIN_SL_ATR: float = 0.5           # SL must be at least this many ATRs from entry

# ── Order flow ────────────────────────────────────────────────────────────────
# Toggle to A/B test order flow confirmation vs raw signal entries
ORDER_FLOW_FILTER: bool = True

# VWAP standard deviation multipliers (used for band labelling in compute_orderflow)
VWAP_STD_BANDS: List[float] = [1.0, 2.0]

# Volume imbalance: bar volume must exceed rolling avg by this multiple to be flagged
VOLUME_IMBALANCE_MULTIPLIER: float = 2.0

# Rolling window (bars) used to compute the average volume baseline
VOLUME_IMBALANCE_LOOKBACK: int = 20

# How many bars back to measure cumulative-delta momentum direction
ORDER_FLOW_DELTA_WINDOW: int = 3

# ── Database ──────────────────────────────────────────────────────────────────
DB_PATH: str = "data/market_data.db"
