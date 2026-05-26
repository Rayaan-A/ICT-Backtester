from typing import Dict, List

SYMBOLS: List[str] = ["NQ=F", "ES=F"]

TIMEFRAMES: Dict[str, str] = {
    "1h": "1h",
    "4h": "4h",
    "1d": "1d",
}

# Data lookback per timeframe
TIMEFRAME_PERIODS: Dict[str, str] = {
    "1h": "20d",
    "4h": "60d",
    "1d": "180d",
}

DEFAULT_SYMBOL: str = "NQ=F"
DEFAULT_TIMEFRAME: str = "1h"

# ICT parameters
OB_DISPLACEMENT_ATR: float = 1.5
FVG_MIN_SIZE_ATR: float = 0.3
LIQ_LOOKBACK: int = 20
RISK_PER_TRADE: float = 0.01
RR_RATIO: float = 2.0

# ATR
ATR_PERIOD: int = 14

# Database
DB_PATH: str = "data/market_data.db"

# Account
INITIAL_CAPITAL: float = 10000.0
