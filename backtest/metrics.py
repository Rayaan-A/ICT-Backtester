from typing import Dict, List, Union

import numpy as np

import config
from backtest.engine import Trade


def calculate_metrics(
    trades: List[Trade],
    equity_curve: List[float],
) -> Dict[str, Union[int, float, str]]:
    """Compute summary statistics for a completed backtest."""
    closed = [t for t in trades if t.result in ("win", "loss")]

    if not closed:
        return {"error": "No closed trades to evaluate"}

    wins = [t for t in closed if t.result == "win"]
    equity = np.array(equity_curve, dtype=float)

    return {
        "total_trades": len(closed),
        "wins": len(wins),
        "losses": len(closed) - len(wins),
        "win_rate": round(len(wins) / len(closed), 4),
        "profit_factor": round(_profit_factor(closed), 4),
        "sharpe_ratio": round(_sharpe(equity), 4),
        "max_drawdown_pct": round(_max_drawdown(equity) * 100, 2),
        "final_equity": round(float(equity[-1]), 2),
        "total_return_pct": round((float(equity[-1]) / equity[0] - 1) * 100, 2),
    }


def equity_series(equity_curve: List[float]) -> np.ndarray:
    """Return the equity curve as a numpy array."""
    return np.array(equity_curve, dtype=float)


# ------------------------------------------------------------------
# Internal calculations
# ------------------------------------------------------------------


def _profit_factor(trades: List[Trade]) -> float:
    """Gross profit divided by gross loss (in R units)."""
    gross_profit = sum(t.pnl_r for t in trades if t.result == "win" and t.pnl_r)
    gross_loss = abs(sum(t.pnl_r for t in trades if t.result == "loss" and t.pnl_r))
    return gross_profit / gross_loss if gross_loss > 0 else float("inf")


def _sharpe(equity: np.ndarray, periods_per_year: int = 252) -> float:
    """Annualised Sharpe ratio, assuming zero risk-free rate."""
    returns = np.diff(equity) / equity[:-1]
    if returns.std() == 0:
        return 0.0
    return float(returns.mean() / returns.std() * np.sqrt(periods_per_year))


def _max_drawdown(equity: np.ndarray) -> float:
    """Maximum peak-to-trough drawdown as a negative fraction."""
    peak = np.maximum.accumulate(equity)
    drawdown = (equity - peak) / peak
    return float(drawdown.min())
