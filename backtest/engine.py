"""
PB Blake Mechanical-Model Backtester
=====================================

Two setups are modelled:

CONTINUATION
  1. A setup-TF (15m or 5m) FVG is *respected*: price taps into the zone
     and closes back out in the original direction.
  2. During the pullback move *toward* that FVG, one or more lower-TF FVGs
     formed.  The highest-timeframe one of these becomes the IFVG entry zone.
  3. Entry: current bar enters the IFVG zone.
  4. SL  : nearest confirmed swing low (long) / high (short) below/above entry.
  5. TP  : nearest confirmed swing high (long) / low (short) — draw on liquidity.

REVERSAL  (London / NY AM sessions only)
  1. A liquidity sweep of a recent swing high or low.
  2. Shortly after the sweep, a setup-TF FVG is *disrespected* (price closes
     through it), creating an IFVG.
  3. Entry: current bar enters the IFVG zone.
  4. SL  : the wick extreme of the sweep candle.
  5. TP  : nearest confirmed swing high/low in trade direction.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd

import config
from signals.amd import AMDManipulation, detect_amd
from signals.fvg import FVG, FVGState, detect_fvgs
from signals.levels import compute_levels, nearest_key_level_above, nearest_key_level_below
from signals.market_structure import compute_bias, get_bias
from signals.orderflow import compute_orderflow, is_order_flow_confirmed
from signals.sessions import is_continuation_session, is_reversal_session
from signals.swing import (
    SwingPoint,
    detect_swings,
    nearest_swing_high_above,
    nearest_swing_low_below,
    nearest_target_above,
    nearest_target_below,
)
from signals.turtle_soup import TurtleSoup, detect_turtle_soups



# ── Trade dataclass ───────────────────────────────────────────────────────────

@dataclass
class Trade:
    """A single simulated trade."""
    entry_index:  int
    entry_time:   pd.Timestamp
    direction:    str                      # 'long' or 'short'
    entry_price:  float
    sl:           float
    tp:           float
    setup_type:   str                      # 'continuation' or 'reversal'
    exit_index:   Optional[int]          = None
    exit_time:    Optional[pd.Timestamp] = None
    exit_price:   Optional[float]        = None
    result:       Optional[str]          = None   # 'win' / 'loss' / 'open'
    pnl_r:        Optional[float]        = None   # profit in units of R
    be_triggered: bool                   = False  # True once SL moved to break-even


# ── Backtester ────────────────────────────────────────────────────────────────

class PBBlakeBacktester:
    """Simulates PB Blake's mechanical model on multi-timeframe OHLCV data."""

    def __init__(
        self,
        dfs: Dict[str, pd.DataFrame],
        initial_capital: float = config.INITIAL_CAPITAL,
    ) -> None:
        """
        Args:
            dfs: dict mapping timeframe label → OHLCV+ATR DataFrame.
                 Must include config.BASE_TIMEFRAME (e.g. '5m').
        """
        self.dfs             = dfs
        self.df              = dfs[config.BASE_TIMEFRAME]
        self.initial_capital = initial_capital
        self.trades:       List[Trade] = []
        self.equity_curve: List[float] = []
        self.levels    = compute_levels(self.df)
        self.orderflow = compute_orderflow(self.df)

        # 15m bias (setup TF)
        _setup_bias_tf = next(
            (tf for tf in config.SETUP_TIMEFRAMES if tf != config.BASE_TIMEFRAME),
            config.BASE_TIMEFRAME,
        )
        self.ms_bias = compute_bias(dfs[_setup_bias_tf])

        # HTF biases — one Series per timeframe, all must agree for an entry
        self.htf_biases = {
            tf: compute_bias(dfs[tf], lookback=config.HTF_SWING_LOOKBACK)
            for tf in config.HTF_BIAS_TIMEFRAMES
            if tf in dfs
        }

    # ── Public entry point ────────────────────────────────────────────────

    def run(self) -> List[Trade]:
        """Pre-compute all signals then step through every base-TF bar."""
        fvgs_by_tf: Dict[str, List[FVG]] = {
            tf: detect_fvgs(df, timeframe=tf)
            for tf, df in self.dfs.items()
        }
        amd_manips = detect_amd(self.df)
        swings     = detect_swings(self.df)

        # Store on self so _manage/_check_be/_has_continuation_sponsor can access without extra args
        self._run_fvgs        = fvgs_by_tf
        self._run_swings      = swings
        self._run_amd         = amd_manips
        self._run_turtle_soups = detect_turtle_soups(self.df, swings)

        capital = self.initial_capital
        active:  Optional[Trade] = None
        start    = max(config.LIQ_LOOKBACK + 2, config.SWING_LOOKBACK * 2 + 2)

        self.equity_curve = [capital] * start

        trades_today: int = 0
        current_day:  Optional[pd.Timestamp] = None

        for i in range(start, len(self.df)):
            ts  = self.df.index[i]
            day = ts.normalize() if ts.tzinfo is None else ts.tz_localize(None).normalize()

            # Reset daily counter on a new calendar day
            if day != current_day:
                current_day  = day
                trades_today = 0

            # Manage any open trade first
            if active is not None:
                active, pnl_r = self._manage(active, i)
                if active.result is not None:
                    capital += capital * config.RISK_PER_TRADE * pnl_r
                    self.trades.append(active)
                    active = None

            # Look for a new entry (only if daily limit not reached)
            if active is None and trades_today < config.MAX_TRADES_PER_DAY:
                bias     = get_bias(self.ms_bias, ts)
                htf_bias = self._htf_bias(ts)
                trade    = None
                # Enter when 15m bias exists and HTF either agrees or is not yet established
                if bias is not None and (htf_bias is None or bias == htf_bias):
                    if is_continuation_session(ts):
                        trade = self._check_continuation(i, fvgs_by_tf, swings, bias)
                    if trade is None and is_reversal_session(ts):
                        trade = self._check_reversal(i, fvgs_by_tf, amd_manips, swings, bias)
                if trade is not None:
                    trades_today += 1
                active = trade

            self.equity_curve.append(capital)

        # Close any position still open at end of data
        if active is not None:
            active.result     = "open"
            active.exit_index = len(self.df) - 1
            active.exit_price = self.df["close"].iloc[-1]
            active.exit_time  = self.df.index[-1]
            self.trades.append(active)

        return self.trades

    # ── HTF bias consensus ────────────────────────────────────────────────

    def _htf_bias(self, ts: pd.Timestamp) -> Optional[str]:
        """
        Return the single bias shared by all HTF timeframes at *ts*, or None
        if they disagree or any is undetermined.
        """
        biases = [get_bias(b, ts) for b in self.htf_biases.values()]
        if not biases:
            return None
        if any(b is None for b in biases):
            return None
        return biases[0] if len(set(biases)) == 1 else None

    # ── Continuation sponsor check ────────────────────────────────────────

    def _has_continuation_sponsor(self, i: int, direction: str) -> bool:
        """
        Return True if a liquidity sweep in *direction* has occurred recently,
        acting as the manipulation 'sponsor' required before a continuation entry.

        Two sources count as a sponsor:
          1. An AMD session-open sweep (Judas swing) in the correct direction.
          2. A turtle soup of a confirmed swing high/low in the correct direction.

        For a long trade: sweep of lows (sell-side swept → bullish).
        For a short trade: sweep of highs (buy-side swept → bearish).
        """
        window       = config.CONTINUATION_SPONSOR_WINDOW_BARS
        amd_dir      = "low"  if direction == "long" else "high"
        ts_dir       = "bullish" if direction == "long" else "bearish"

        # AMD sweep sponsor
        for m in self._run_amd:
            if 0 < i - m.index <= window and m.direction == amd_dir:
                return True

        # Turtle soup sponsor
        for ts in self._run_turtle_soups:
            if 0 < i - ts.index <= window and ts.direction == ts_dir:
                return True

        return False

    # ── Continuation setup ────────────────────────────────────────────────

    def _check_continuation(
        self,
        i: int,
        fvgs_by_tf: Dict[str, List[FVG]],
        swings: List[SwingPoint],
        bias: str,
    ) -> Optional[Trade]:
        """
        Scan setup-TF FVGs that were recently respected, filtered to the 15m bias.
        Requires a recent AMD sweep or turtle soup as a manipulation 'sponsor'
        (per the PB Blake rule: no manipulation = no trade).
        For each qualifying FVG, find the highest-TF IFVG and check bar i entry.
        """
        ts_now    = self.df.index[i]
        direction = "long" if bias == "bullish" else "short"

        # Require a sponsor sweep before scanning for entries
        if config.REQUIRE_CONTINUATION_SPONSOR and not self._has_continuation_sponsor(i, direction):
            return None

        for setup_tf in config.SETUP_TIMEFRAMES:
            for sfvg in fvgs_by_tf.get(setup_tf, []):
                if sfvg.state != FVGState.RESPECTED:
                    continue
                if sfvg.state_timestamp is None:
                    continue
                if sfvg.direction != bias:
                    continue

                bars_since = _bars_elapsed(sfvg.state_timestamp, ts_now)
                if bars_since < 0 or bars_since > config.SETUP_RESPECT_WINDOW_BARS:
                    continue

                trade = self._find_ifvg_entry(
                    i          = i,
                    fvgs_by_tf = fvgs_by_tf,
                    setup_dir  = sfvg.direction,
                    pb_start   = sfvg.timestamp,
                    pb_end     = sfvg.state_timestamp,
                    swings     = swings,
                )
                if trade is not None:
                    return trade

        return None

    def _find_ifvg_entry(
        self,
        i:          int,
        fvgs_by_tf: Dict[str, List[FVG]],
        setup_dir:  str,
        pb_start:   pd.Timestamp,
        pb_end:     pd.Timestamp,
        swings:     List[SwingPoint],
    ) -> Optional[Trade]:
        """
        Find the highest-TF FVG that was created during the pullback move
        [pb_start, pb_end] and check if bar i is entering it as an IFVG.

        For a bullish setup (price came DOWN to the setup FVG):
          - Pullback FVGs are bearish (created by bearish displacement within pullback)
          - After the setup FVG respect, price rallies back up into these zones
          - Entry: bar's high enters the bearish FVG bottom while close is still inside

        For a bearish setup (price came UP to the setup FVG):
          - Pullback FVGs are bullish
          - Entry: bar's low enters the bullish FVG top while close is still inside
        """
        candle    = self.df.iloc[i]
        atr       = self.df["atr"].iloc[i]
        ts_now    = self.df.index[i]
        long_bias = setup_dir == "bullish"
        ifvg_dir  = "bearish" if long_bias else "bullish"

        for entry_tf in config.ENTRY_IFVG_TIMEFRAMES:
            candidates = [
                f for f in fvgs_by_tf.get(entry_tf, [])
                if f.direction == ifvg_dir
                and pb_start <= f.timestamp <= pb_end
            ]
            if not candidates:
                continue

            # Prefer most recently formed (= highest up the pullback = closest to setup)
            candidates.sort(key=lambda f: f.timestamp, reverse=True)

            for ifvg in candidates:
                if long_bias:
                    # Entering bearish FVG from below: high touches bottom, close still ≤ top
                    if candle["high"] >= ifvg.bottom and candle["close"] <= ifvg.top:
                        entry    = ifvg.bottom
                        min_sl   = entry - atr * config.MIN_SL_ATR
                        swing_sl = nearest_swing_low_below(swings, entry, i)
                        sl       = min(swing_sl, min_sl) if swing_sl is not None else min_sl
                        risk     = entry - sl
                        tp    = (
                            nearest_key_level_above(self.levels, entry, i,
                                                    min_distance=risk * config.MIN_RR,
                                                    max_distance=risk * config.MAX_RR)
                            or _swing_tp_in_band(nearest_target_above(swings, entry, i),
                                                 entry, risk, above=True)
                        )
                        if tp is not None and tp > entry > sl:
                            if config.ORDER_FLOW_FILTER and not is_order_flow_confirmed(
                                self.orderflow, i, "long", entry
                            ):
                                continue
                            return Trade(
                                entry_index = i,
                                entry_time  = ts_now,
                                direction   = "long",
                                entry_price = entry,
                                sl          = sl,
                                tp          = tp,
                                setup_type  = "continuation",
                            )
                else:
                    # Entering bullish FVG from above: low touches top, close still ≥ bottom
                    if candle["low"] <= ifvg.top and candle["close"] >= ifvg.bottom:
                        entry    = ifvg.top
                        min_sl   = entry + atr * config.MIN_SL_ATR
                        swing_sl = nearest_swing_high_above(swings, entry, i)
                        sl       = max(swing_sl, min_sl) if swing_sl is not None else min_sl
                        risk     = sl - entry
                        tp    = (
                            nearest_key_level_below(self.levels, entry, i,
                                                    min_distance=risk * config.MIN_RR,
                                                    max_distance=risk * config.MAX_RR)
                            or _swing_tp_in_band(nearest_target_below(swings, entry, i),
                                                 entry, risk, above=False)
                        )
                        if tp is not None and tp < entry < sl:
                            if config.ORDER_FLOW_FILTER and not is_order_flow_confirmed(
                                self.orderflow, i, "short", entry
                            ):
                                continue
                            return Trade(
                                entry_index = i,
                                entry_time  = ts_now,
                                direction   = "short",
                                entry_price = entry,
                                sl          = sl,
                                tp          = tp,
                                setup_type  = "continuation",
                            )

        return None

    # ── Reversal setup ────────────────────────────────────────────────────

    def _check_reversal(
        self,
        i:          int,
        fvgs_by_tf: Dict[str, List[FVG]],
        amd_manips: List[AMDManipulation],
        swings:     List[SwingPoint],
        bias:       str,
    ) -> Optional[Trade]:
        """
        AMD reversal entry: an accumulation-range Judas sweep (M phase) followed
        by an IFVG entry in the distribution direction (D phase).
        The sweep must have occurred within REVERSAL_SWEEP_WINDOW_BARS of bar i.
        """
        candle = self.df.iloc[i]
        atr    = self.df["atr"].iloc[i]
        ts_now = self.df.index[i]

        recent_manips = [
            m for m in amd_manips
            if 0 < i - m.index <= config.REVERSAL_SWEEP_WINDOW_BARS
        ]
        if not recent_manips:
            return None

        latest_manip = max(recent_manips, key=lambda m: m.index)

        for setup_tf in config.SETUP_TIMEFRAMES:
            for sfvg in fvgs_by_tf.get(setup_tf, []):
                if sfvg.state != FVGState.DISRESPECTED:
                    continue
                if sfvg.state_timestamp is None:
                    continue
                # Disrespect must have happened after the Judas sweep
                if sfvg.state_timestamp < latest_manip.timestamp:
                    continue

                bars_since = _bars_elapsed(sfvg.state_timestamp, ts_now)
                if bars_since < 0 or bars_since > config.REVERSAL_SWEEP_WINDOW_BARS:
                    continue

                # Judas swept lows → long distribution: bias was bearish before the sweep
                if latest_manip.direction == "low" and sfvg.direction == "bearish" and bias == "bearish":
                    entry = sfvg.top
                    if not (candle["low"] <= sfvg.top and candle["close"] >= sfvg.bottom):
                        continue
                    sl   = min(latest_manip.candle_low - atr * 0.1, entry - atr * config.MIN_SL_ATR)
                    risk = entry - sl
                    tp   = (
                        nearest_key_level_above(self.levels, entry, i,
                                                min_distance=risk * config.MIN_RR,
                                                max_distance=risk * config.MAX_RR)
                        or _swing_tp_in_band(nearest_target_above(swings, entry, i),
                                             entry, risk, above=True)
                    )
                    if tp is not None and tp > entry > sl:
                        if config.ORDER_FLOW_FILTER and not is_order_flow_confirmed(
                            self.orderflow, i, "long", entry
                        ):
                            continue
                        return Trade(
                            entry_index = i,
                            entry_time  = ts_now,
                            direction   = "long",
                            entry_price = entry,
                            sl          = sl,
                            tp          = tp,
                            setup_type  = "reversal",
                        )

                # Judas swept highs → short distribution: bias was bullish before the sweep
                elif latest_manip.direction == "high" and sfvg.direction == "bullish" and bias == "bullish":
                    entry = sfvg.bottom
                    if not (candle["high"] >= sfvg.bottom and candle["close"] <= sfvg.top):
                        continue
                    sl   = max(latest_manip.candle_high + atr * 0.1, entry + atr * config.MIN_SL_ATR)
                    risk = sl - entry
                    tp   = (
                        nearest_key_level_below(self.levels, entry, i,
                                                min_distance=risk * config.MIN_RR,
                                                max_distance=risk * config.MAX_RR)
                        or _swing_tp_in_band(nearest_target_below(swings, entry, i),
                                             entry, risk, above=False)
                    )
                    if tp is not None and tp < entry < sl:
                        if config.ORDER_FLOW_FILTER and not is_order_flow_confirmed(
                            self.orderflow, i, "short", entry
                        ):
                            continue
                        return Trade(
                            entry_index = i,
                            entry_time  = ts_now,
                            direction   = "short",
                            entry_price = entry,
                            sl          = sl,
                            tp          = tp,
                            setup_type  = "reversal",
                        )

        return None

    # ── Trade management ──────────────────────────────────────────────────

    def _check_be(self, trade: Trade, i: int) -> bool:
        """
        Return True if break-even should be triggered at bar i.

        Two conditions — either is sufficient:
          1. Price reaches the nearest swing high (long) / low (short) above/below
             the entry that was confirmed before the entry bar.
          2. A new IFVG in the trade direction has formed and been confirmed
             between entry and bar i, with its zone beyond the entry price.
        """
        if trade.be_triggered:
            return False

        candle = self.df.iloc[i]

        # Condition 1: price reaches a swing target beyond entry
        if trade.direction == "long":
            swing_target = nearest_target_above(self._run_swings, trade.entry_price, trade.entry_index)
            if swing_target is not None and candle["high"] >= swing_target:
                return True
        else:
            swing_target = nearest_target_below(self._run_swings, trade.entry_price, trade.entry_index)
            if swing_target is not None and candle["low"] <= swing_target:
                return True

        # Condition 2: a new IFVG in trade direction confirmed since entry
        for fvgs in self._run_fvgs.values():
            for fvg in fvgs:
                if fvg.state != FVGState.DISRESPECTED:
                    continue
                if fvg.index <= trade.entry_index:
                    continue  # must have formed after trade entry
                if fvg.state_index is None or fvg.state_index > i:
                    continue  # must already be disrespected by this bar
                # Long: bearish FVG disrespected above entry → bullish IFVG confirmed overhead
                if trade.direction == "long" and fvg.direction == "bearish" and fvg.bottom > trade.entry_price:
                    return True
                # Short: bullish FVG disrespected below entry → bearish IFVG confirmed below
                if trade.direction == "short" and fvg.direction == "bullish" and fvg.top < trade.entry_price:
                    return True

        return False

    def _manage(self, trade: Trade, i: int) -> Tuple[Trade, float]:
        """Check break-even, then SL and TP for an open trade. Returns (trade, pnl_r)."""
        # Move SL to break-even before checking exit conditions
        if self._check_be(trade, i):
            trade.sl          = trade.entry_price
            trade.be_triggered = True

        candle = self.df.iloc[i]
        risk   = abs(trade.entry_price - trade.sl)
        reward = abs(trade.tp - trade.entry_price)
        rr     = (reward / risk) if risk > 0 else config.FALLBACK_RR

        if trade.direction == "long":
            if candle["low"] <= trade.sl:
                if trade.be_triggered:
                    return self._close(trade, i, trade.entry_price, "breakeven", 0.0)
                return self._close(trade, i, trade.sl, "loss", -1.0)
            if candle["high"] >= trade.tp:
                return self._close(trade, i, trade.tp, "win", rr)
        else:
            if candle["high"] >= trade.sl:
                if trade.be_triggered:
                    return self._close(trade, i, trade.entry_price, "breakeven", 0.0)
                return self._close(trade, i, trade.sl, "loss", -1.0)
            if candle["low"] <= trade.tp:
                return self._close(trade, i, trade.tp, "win", rr)

        return trade, 0.0

    def _close(
        self,
        trade:  Trade,
        i:      int,
        price:  float,
        result: str,
        pnl_r:  float,
    ) -> Tuple[Trade, float]:
        """Mutate trade with exit details and return it."""
        trade.exit_index = i
        trade.exit_time  = self.df.index[i]
        trade.exit_price = price
        trade.result     = result
        trade.pnl_r      = pnl_r
        return trade, pnl_r


# ── Utilities ─────────────────────────────────────────────────────────────────

def _swing_tp_in_band(
    tp: Optional[float],
    entry: float,
    risk: float,
    above: bool,
) -> Optional[float]:
    """Return *tp* only if it falls within the MIN_RR/MAX_RR band, else None."""
    if tp is None:
        return None
    distance = (tp - entry) if above else (entry - tp)
    if risk * config.MIN_RR <= distance <= risk * config.MAX_RR:
        return tp
    return None


def _bars_elapsed(earlier: pd.Timestamp, later: pd.Timestamp) -> int:
    """
    Approximate number of 5-min base-TF bars between two timestamps.
    Returns a negative number if *later* is before *earlier*.
    """
    delta_minutes = (later - earlier).total_seconds() / 60
    return int(delta_minutes / 5)
