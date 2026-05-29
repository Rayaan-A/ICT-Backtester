"""
Unit tests for signals/orderflow.py

All tests use synthetic DataFrames — no yfinance calls, no network access.
Timestamps are UTC-naive to match what yfinance returns for futures data.
"""

import numpy as np
import pandas as pd
import pytest

from signals.orderflow import compute_orderflow, is_order_flow_confirmed


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_df(
    opens:   list,
    highs:   list,
    lows:    list,
    closes:  list,
    volumes: list,
    start:   str = "2024-01-02 09:30:00",   # a Monday
    freq:    str = "5min",
) -> pd.DataFrame:
    """Build a minimal OHLCV DataFrame with a DatetimeIndex."""
    idx = pd.date_range(start=start, periods=len(closes), freq=freq)
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": volumes},
        index=idx,
    )


def _bullish_bars(n: int = 10, base: float = 100.0) -> pd.DataFrame:
    """All bars close near the high → strongly positive delta."""
    return _make_df(
        opens   = [base + i      for i in range(n)],
        highs   = [base + i + 1  for i in range(n)],
        lows    = [base + i - 0.1 for i in range(n)],
        closes  = [base + i + 0.9 for i in range(n)],
        volumes = [1000] * n,
    )


def _bearish_bars(n: int = 10, base: float = 110.0) -> pd.DataFrame:
    """Prices trending down, close near the low → negative delta and price below VWAP."""
    return _make_df(
        opens   = [base - i       for i in range(n)],
        highs   = [base - i + 0.1 for i in range(n)],
        lows    = [base - i - 1.0 for i in range(n)],
        closes  = [base - i - 0.9 for i in range(n)],
        volumes = [1000] * n,
    )


# ── Delta calculation ─────────────────────────────────────────────────────────

class TestDelta:
    def test_close_at_high_gives_positive_delta(self):
        """Close == high → delta == +volume (pure buying)."""
        df = _make_df([99], [101], [99], [101], [500])
        of = compute_orderflow(df)
        assert of["delta"].iloc[0] == pytest.approx(500.0, rel=1e-6)

    def test_close_at_low_gives_negative_delta(self):
        """Close == low → delta == −volume (pure selling)."""
        df = _make_df([101], [101], [99], [99], [500])
        of = compute_orderflow(df)
        assert of["delta"].iloc[0] == pytest.approx(-500.0, rel=1e-6)

    def test_close_at_midpoint_gives_zero_delta(self):
        """Close == midpoint → delta == 0 (balanced)."""
        df = _make_df([100], [102], [98], [100], [400])
        of = compute_orderflow(df)
        assert of["delta"].iloc[0] == pytest.approx(0.0, abs=1e-9)

    def test_bullish_sequence_has_positive_cum_delta(self):
        """Sequence of bullish bars → cumulative delta stays positive."""
        df  = _bullish_bars(n=10)
        of  = compute_orderflow(df)
        assert (of["cum_delta"] > 0).all()

    def test_bearish_sequence_has_negative_cum_delta(self):
        """Sequence of bearish bars → cumulative delta stays negative."""
        df  = _bearish_bars(n=10)
        of  = compute_orderflow(df)
        assert (of["cum_delta"] < 0).all()


# ── VWAP ─────────────────────────────────────────────────────────────────────

class TestVWAP:
    def test_vwap_equals_typical_price_when_single_bar(self):
        """With one bar, VWAP must equal its own typical price."""
        df = _make_df([99], [102], [98], [101], [1000])
        of = compute_orderflow(df)
        expected_typical = (102 + 98 + 101) / 3
        assert of["vwap"].iloc[0] == pytest.approx(expected_typical, rel=1e-6)

    def test_vwap_resets_on_new_et_day(self):
        """VWAP must reset at midnight ET: bar on day 2 should not carry day 1 history."""
        # Day 1: one 5m bar at 14:00 UTC (= 09:00 ET on 2024-01-02)
        # Day 2: one 5m bar at 14:00 UTC next day (= 09:00 ET on 2024-01-03)
        idx = pd.DatetimeIndex([
            pd.Timestamp("2024-01-02 14:00:00"),
            pd.Timestamp("2024-01-03 14:00:00"),
        ])
        df = pd.DataFrame(
            {
                "open":   [100.0, 200.0],
                "high":   [110.0, 210.0],
                "low":    [90.0,  190.0],
                "close":  [105.0, 205.0],
                "volume": [1000.0, 2000.0],
            },
            index=idx,
        )
        of = compute_orderflow(df)
        typical_d2 = (210 + 190 + 205) / 3
        # Day 2 VWAP must equal day 2's own typical price (reset)
        assert of["vwap"].iloc[1] == pytest.approx(typical_d2, rel=1e-6)

    def test_vwap_upper1_above_vwap(self):
        """Upper 1σ band must always be ≥ VWAP (std dev is non-negative)."""
        df = _bullish_bars(n=20)
        of = compute_orderflow(df)
        assert (of["vwap_upper1"] >= of["vwap"]).all()

    def test_vwap_lower1_below_vwap(self):
        """Lower 1σ band must always be ≤ VWAP."""
        df = _bullish_bars(n=20)
        of = compute_orderflow(df)
        assert (of["vwap_lower1"] <= of["vwap"]).all()

    def test_vwap_bands_order(self):
        """Band ordering: upper2 ≥ upper1 ≥ vwap ≥ lower1 ≥ lower2."""
        df = _bullish_bars(n=20)
        of = compute_orderflow(df)
        # Use iloc[1:] to skip the first bar where std == 0 and all bands equal VWAP
        assert (of["vwap_upper2"].iloc[1:] >= of["vwap_upper1"].iloc[1:]).all()
        assert (of["vwap_lower1"].iloc[1:] >= of["vwap_lower2"].iloc[1:]).all()


# ── Volume imbalance ─────────────────────────────────────────────────────────

class TestVolumeImbalance:
    def test_high_volume_bar_flagged(self):
        """A bar with volume 3× the rolling avg should be flagged as an imbalance."""
        volumes = [1000] * 19 + [3500]   # last bar is 3.5× avg
        df = _make_df(
            opens   = [100.0] * 20,
            highs   = [101.0] * 20,
            lows    = [99.0]  * 20,
            closes  = [100.5] * 20,
            volumes = volumes,
        )
        of = compute_orderflow(df)
        assert of["is_imbalance"].iloc[-1]

    def test_normal_volume_not_flagged(self):
        """All identical-volume bars should not produce any imbalance flag."""
        df = _bullish_bars(n=20)
        of = compute_orderflow(df)
        assert not of["is_imbalance"].any()

    def test_volume_ratio_one_for_constant_volume(self):
        """Constant volume → volume_ratio == 1.0 for every bar."""
        df = _bullish_bars(n=20)
        of = compute_orderflow(df)
        assert np.allclose(of["volume_ratio"], 1.0)


# ── Order flow confirmation ───────────────────────────────────────────────────

class TestIsOrderFlowConfirmed:
    def test_long_confirmed_when_cum_delta_rising_above_lower_band(self):
        """
        Long should be confirmed: bullish bars → cum_delta rising, price near
        the high of each bar → above the lower 1σ band.
        """
        df = _bullish_bars(n=10)
        of = compute_orderflow(df)
        # By bar 5, there should be enough history and positive delta momentum
        entry = df["close"].iloc[5]
        assert is_order_flow_confirmed(of, 5, "long", entry)

    def test_short_confirmed_when_cum_delta_falling_below_upper_band(self):
        """
        Short should be confirmed: bearish bars → cum_delta falling, price near
        the low of each bar → below the upper 1σ band.
        """
        df = _bearish_bars(n=10)
        of = compute_orderflow(df)
        entry = df["close"].iloc[5]
        assert is_order_flow_confirmed(of, 5, "short", entry)

    def test_long_rejected_when_cum_delta_falling(self):
        """Long should be rejected when cumulative delta is declining (selling pressure)."""
        df  = _bearish_bars(n=10)
        of  = compute_orderflow(df)
        # In a bearish sequence, delta is falling — long should be rejected
        entry = df["close"].iloc[5]
        assert not is_order_flow_confirmed(of, 5, "long", entry)

    def test_short_rejected_when_cum_delta_rising(self):
        """Short should be rejected when cumulative delta is rising (buying pressure)."""
        df  = _bullish_bars(n=10)
        of  = compute_orderflow(df)
        entry = df["close"].iloc[5]
        assert not is_order_flow_confirmed(of, 5, "short", entry)

    def test_early_bars_always_confirmed(self):
        """When i < ORDER_FLOW_DELTA_WINDOW, fall back to True (not enough history)."""
        df  = _bearish_bars(n=10)
        of  = compute_orderflow(df)
        # i=1 is below the default window (3) → should return True regardless of direction
        assert is_order_flow_confirmed(of, 1, "long",  df["close"].iloc[1])
        assert is_order_flow_confirmed(of, 1, "short", df["close"].iloc[1])

    def test_long_rejected_when_price_below_lower_band(self):
        """
        Long should be rejected even with rising delta if price is deeply below VWAP −1σ.
        We simulate this by manually patching the band to be very high.
        """
        df = _bullish_bars(n=10)
        of = compute_orderflow(df)
        # Force lower band far above any realistic entry price
        of = of.copy()
        of["vwap_lower1"] = 1e9
        entry = df["close"].iloc[5]
        assert not is_order_flow_confirmed(of, 5, "long", entry)

    def test_short_rejected_when_price_above_upper_band(self):
        """Short should be rejected when price is above VWAP +1σ (too extended)."""
        df = _bearish_bars(n=10)
        of = compute_orderflow(df)
        of = of.copy()
        of["vwap_upper1"] = -1e9   # force upper band far below any realistic price
        entry = df["close"].iloc[5]
        assert not is_order_flow_confirmed(of, 5, "short", entry)
