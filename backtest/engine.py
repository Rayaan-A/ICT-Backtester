from dataclasses import dataclass
from typing import List, Optional, Tuple

import pandas as pd

import config
from signals.fvg import FVG, detect_fvgs
from signals.liquidity import LiquiditySweep, detect_sweeps


@dataclass
class Trade:
    """A single simulated trade with entry, SL, TP, and result."""

    entry_index: int
    entry_time: pd.Timestamp
    direction: str          # 'long' or 'short'
    entry_price: float
    sl: float
    tp: float
    exit_index: Optional[int] = None
    exit_time: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    result: Optional[str] = None   # 'win', 'loss', 'open'
    pnl_r: Optional[float] = None  # pnl in units of R (risk)


class Backtester:
    """Simulates ICT entries: sweep of liquidity → price enters opposing FVG."""

    def __init__(
        self,
        df: pd.DataFrame,
        initial_capital: float = config.INITIAL_CAPITAL,
    ) -> None:
        self.df = df.copy()
        self.initial_capital = initial_capital
        self.trades: List[Trade] = []
        self.equity_curve: List[float] = []

    def run(self) -> List[Trade]:
        """Execute the full backtest loop and return all trades."""
        fvgs = detect_fvgs(self.df)
        sweeps = detect_sweeps(self.df)

        capital = self.initial_capital
        active: Optional[Trade] = None
        start = config.LIQ_LOOKBACK + 2

        # Pre-fill equity curve for the warm-up candles
        self.equity_curve = [capital] * start

        for i in range(start, len(self.df)):
            if active is not None:
                active, pnl_r = self._manage(active, i)
                if active.result is not None:
                    risk_amount = capital * config.RISK_PER_TRADE
                    capital += risk_amount * pnl_r
                    self.trades.append(active)
                    active = None

            if active is None:
                active = self._check_entry(i, fvgs, sweeps)

            self.equity_curve.append(capital)

        if active is not None:
            active.result = "open"
            active.exit_index = len(self.df) - 1
            active.exit_price = self.df["close"].iloc[-1]
            active.exit_time = self.df.index[-1]
            self.trades.append(active)

        return self.trades

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _check_entry(
        self,
        i: int,
        fvgs: List[FVG],
        sweeps: List[LiquiditySweep],
    ) -> Optional[Trade]:
        """Return a new Trade if the current candle triggers an entry, else None."""
        recent_sweeps = [s for s in sweeps if i - 10 <= s.index < i]
        if not recent_sweeps:
            return None

        latest_sweep = max(recent_sweeps, key=lambda s: s.index)
        candle = self.df.iloc[i]
        atr = self.df["atr"].iloc[i]

        if latest_sweep.direction == "low":
            return self._long_entry(i, candle, atr, fvgs, latest_sweep.index)

        if latest_sweep.direction == "high":
            return self._short_entry(i, candle, atr, fvgs, latest_sweep.index)

        return None

    def _long_entry(
        self,
        i: int,
        candle: pd.Series,
        atr: float,
        fvgs: List[FVG],
        sweep_idx: int,
    ) -> Optional[Trade]:
        """Enter long when price dips into a bullish FVG formed after the sweep."""
        for fvg in fvgs:
            if fvg.direction != "bullish" or fvg.index <= sweep_idx or fvg.filled:
                continue
            if fvg.index >= i:
                continue
            if candle["low"] <= fvg.top and candle["close"] >= fvg.bottom:
                entry = fvg.bottom
                sl = entry - atr * 0.5
                risk = entry - sl
                tp = entry + risk * config.RR_RATIO
                return Trade(
                    entry_index=i,
                    entry_time=self.df.index[i],
                    direction="long",
                    entry_price=entry,
                    sl=sl,
                    tp=tp,
                )
        return None

    def _short_entry(
        self,
        i: int,
        candle: pd.Series,
        atr: float,
        fvgs: List[FVG],
        sweep_idx: int,
    ) -> Optional[Trade]:
        """Enter short when price rallies into a bearish FVG formed after the sweep."""
        for fvg in fvgs:
            if fvg.direction != "bearish" or fvg.index <= sweep_idx or fvg.filled:
                continue
            if fvg.index >= i:
                continue
            if candle["high"] >= fvg.bottom and candle["close"] <= fvg.top:
                entry = fvg.top
                sl = entry + atr * 0.5
                risk = sl - entry
                tp = entry - risk * config.RR_RATIO
                return Trade(
                    entry_index=i,
                    entry_time=self.df.index[i],
                    direction="short",
                    entry_price=entry,
                    sl=sl,
                    tp=tp,
                )
        return None

    def _manage(self, trade: Trade, i: int) -> Tuple[Trade, float]:
        """Check SL/TP for an open trade. Returns (trade, pnl_r)."""
        candle = self.df.iloc[i]

        if trade.direction == "long":
            if candle["low"] <= trade.sl:
                return self._close(trade, i, trade.sl, "loss", -1.0)
            if candle["high"] >= trade.tp:
                return self._close(trade, i, trade.tp, "win", config.RR_RATIO)

        else:  # short
            if candle["high"] >= trade.sl:
                return self._close(trade, i, trade.sl, "loss", -1.0)
            if candle["low"] <= trade.tp:
                return self._close(trade, i, trade.tp, "win", config.RR_RATIO)

        return trade, 0.0

    def _close(
        self,
        trade: Trade,
        i: int,
        price: float,
        result: str,
        pnl_r: float,
    ) -> Tuple[Trade, float]:
        """Mutate trade with exit details and return it."""
        trade.exit_index = i
        trade.exit_time = self.df.index[i]
        trade.exit_price = price
        trade.result = result
        trade.pnl_r = pnl_r
        return trade, pnl_r
