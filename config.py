from typing import Dict, List

SYMBOLS: List[str] = ["EURUSD=X", "GBPUSD=X", "NQ=F"]

TIMEFRAMES: Dict[str, str] = {
    "1h": "1h",
    "4h": "4h",
    "1d": "1d",
}

DEFAULT_SYMBOL: str = "EURUSD=X"
DEFAULT_TIMEFRAME: str = "1h"
DEFAULT_PERIOD: str = "60d"

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
