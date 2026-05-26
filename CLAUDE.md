# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

ICT backtesting engine for Fair Value Gaps, Order Blocks, Liquidity Sweeps, and Market Structure, with a Plotly Dash dashboard.

## Commands

```bash
# Install dependencies
pip install pandas numpy yfinance plotly dash sqlite3

# Run the backtester
python main.py

# Run the dashboard
python dashboard/app.py

# Run tests
pytest

# Run a single test
pytest tests/test_signals.py::test_fvg_detection -v
```

## Architecture

```
config.py          # All settings and constants (never hardcode values elsewhere)
main.py            # Entry point: orchestrates data → signals → backtest pipeline
data/              # Data fetching and storage (yfinance → SQLite via sqlite3)
signals/           # ICT pattern detection (FVG, OB, liquidity sweeps, market structure)
backtest/          # Strategy execution: applies signals to OHLCV data, tracks PnL
dashboard/         # Plotly Dash app: visualizes equity curve, trades, and patterns on chart
```

**Data flow:** `data/` fetches OHLCV → `signals/` detects ICT patterns → `backtest/` simulates trades → `dashboard/` renders results.

## ICT Concepts

- **FVG (Fair Value Gap):** 3-candle imbalance — gap between candle[i-2].high and candle[i].low (bullish) or candle[i-2].low and candle[i].high (bearish)
- **Order Block (OB):** Last opposing candle before a displacement move (displacement = range > 1.5× ATR)
- **Liquidity Sweep:** Wick past a swing high/low with the candle closing back inside the prior range
- **Market Structure:** Track Break of Structure (BOS) and Change of Character (CHoCH) using swing highs/lows

## Coding Style

- Type hints on every function signature
- Docstring on every function
- Functions should be small and single-purpose
- **All settings go in `config.py`** — no hardcoded values anywhere else (symbols, timeframes, thresholds, ATR multipliers, etc.)
