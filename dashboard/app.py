import calendar
import os
import sys
from typing import List

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import dash
import plotly.graph_objects as go
from dash import ClientsideFunction, Input, Output, dcc, html
import pandas as pd

import config
from backtest.engine import Backtester
from backtest.metrics import calculate_metrics
from data.fetcher import fetch_ohlcv
from signals.fvg import FVG, detect_fvgs
from signals.liquidity import LiquiditySweep, detect_sweeps
from signals.order_blocks import OrderBlock, detect_order_blocks

app = dash.Dash(
    __name__,
    title="ICT Backtester",
    external_scripts=[
        "https://unpkg.com/lightweight-charts@4.2.0/dist/lightweight-charts.standalone.production.js"
    ],
)

app.layout = html.Div(
    style={"fontFamily": "Arial, sans-serif", "backgroundColor": "#1a1a2e", "color": "white"},
    children=[
        html.H1("ICT Backtester", style={"textAlign": "center", "padding": "20px 0 10px"}),
        html.Div(
            style={"padding": "0 20px 10px", "display": "flex", "alignItems": "flex-end", "gap": "20px"},
            children=[
                html.Div([
                    html.Label("Symbol", style={"display": "block", "marginBottom": "4px"}),
                    dcc.Dropdown(
                        id="symbol-dd",
                        options=[{"label": s, "value": s} for s in config.SYMBOLS],
                        value=config.DEFAULT_SYMBOL,
                        clearable=False,
                        style={"width": "180px", "color": "#000"},
                    ),
                ]),
                html.Div([
                    html.Label("Timeframe", style={"display": "block", "marginBottom": "4px"}),
                    dcc.Dropdown(
                        id="timeframe-dd",
                        options=[{"label": tf, "value": tf} for tf in config.TIMEFRAMES],
                        value=config.DEFAULT_TIMEFRAME,
                        clearable=False,
                        style={"width": "120px", "color": "#000"},
                    ),
                ]),
                html.Button(
                    "Run Backtest",
                    id="run-btn",
                    n_clicks=0,
                    style={
                        "padding": "8px 20px",
                        "cursor": "pointer",
                        "backgroundColor": "#7986cb",
                        "border": "none",
                        "color": "white",
                        "borderRadius": "4px",
                    },
                ),
            ],
        ),
        html.Div(
            id="tv-chart",
            style={"height": "580px", "width": "100%", "backgroundColor": "#1a1a2e"},
        ),
        dcc.Store(id="chart-data"),
        dcc.Store(id="chart-rendered"),
        dcc.Graph(id="equity-chart", style={"height": "280px"}),
        html.Div(id="metrics-panel", style={"padding": "20px"}),
    ],
)


@app.callback(
    Output("chart-data", "data"),
    Output("equity-chart", "figure"),
    Output("metrics-panel", "children"),
    Input("run-btn", "n_clicks"),
    Input("symbol-dd", "value"),
    Input("timeframe-dd", "value"),
)
def update_data(n_clicks: int, symbol: str, timeframe: str) -> tuple:
    """Compute signals and backtest; serialize chart data for the clientside renderer."""
    df = fetch_ohlcv(symbol, timeframe)
    fvgs = detect_fvgs(df)
    obs = detect_order_blocks(df)
    sweeps = detect_sweeps(df)

    backtester = Backtester(df)
    trades = backtester.run()
    metrics = calculate_metrics(trades, backtester.equity_curve)

    return (
        _serialize_chart_data(df, fvgs, obs, sweeps),
        _equity_chart(backtester.equity_curve),
        _metrics_panel(metrics),
    )


app.clientside_callback(
    ClientsideFunction(namespace="clientside", function_name="renderChart"),
    Output("chart-rendered", "data"),
    Input("chart-data", "data"),
)


# ------------------------------------------------------------------
# Serialization helpers
# ------------------------------------------------------------------


def _to_unix(ts: pd.Timestamp) -> int:
    """Convert a UTC-naive pandas Timestamp to a UTC Unix timestamp (seconds)."""
    return calendar.timegm(ts.timetuple())


def _serialize_chart_data(
    df: pd.DataFrame,
    fvgs: List[FVG],
    obs: List[OrderBlock],
    sweeps: List[LiquiditySweep],
) -> dict:
    """Produce a JSON-serialisable dict consumed by the clientside chart renderer."""
    candles = [
        {
            "time": _to_unix(ts),
            "open": round(float(row["open"]), 6),
            "high": round(float(row["high"]), 6),
            "low": round(float(row["low"]), 6),
            "close": round(float(row["close"]), 6),
        }
        for ts, row in df.iterrows()
    ]

    volume = [
        {
            "time": _to_unix(ts),
            "value": float(row["volume"]),
            "color": "#26a69a" if row["close"] >= row["open"] else "#ef5350",
        }
        for ts, row in df.iterrows()
    ]

    last_time = _to_unix(df.index[-1])

    fvg_boxes = [
        {
            "left": _to_unix(df.index[fvg.index]),
            "right": last_time,
            "top": round(float(fvg.top), 6),
            "bottom": round(float(fvg.bottom), 6),
            "fillColor": "rgba(38,166,154,0.18)" if fvg.direction == "bullish" else "rgba(239,83,80,0.18)",
            "borderColor": "#26a69a" if fvg.direction == "bullish" else "#ef5350",
            "extendRight": True,
            "dashed": False,
        }
        for fvg in fvgs
        if not fvg.filled and fvg.index < len(df)
    ]

    ob_boxes = [
        {
            "left": _to_unix(df.index[ob.index]),
            "right": last_time,
            "top": round(float(ob.top), 6),
            "bottom": round(float(ob.bottom), 6),
            "fillColor": "rgba(255,214,0,0.10)" if ob.direction == "bullish" else "rgba(255,87,34,0.10)",
            "borderColor": "#FFD600" if ob.direction == "bullish" else "#FF5722",
            "extendRight": True,
            "dashed": True,
        }
        for ob in obs
        if not ob.mitigated and ob.index < len(df)
    ]

    sweep_lines = [
        {
            "price": round(float(s.sweep_level), 6),
            "direction": s.direction,
        }
        for s in sweeps
    ]

    return {
        "candles": candles,
        "volume": volume,
        "fvgs": fvg_boxes,
        "obs": ob_boxes,
        "sweeps": sweep_lines,
    }


# ------------------------------------------------------------------
# Equity chart + metrics panel (still Plotly)
# ------------------------------------------------------------------


def _equity_chart(equity_curve: List[float]) -> go.Figure:
    """Line chart of the equity curve."""
    fig = go.Figure(
        go.Scatter(
            x=list(range(len(equity_curve))),
            y=equity_curve,
            mode="lines",
            line=dict(color="#7986cb", width=2),
            fill="tozeroy",
            fillcolor="rgba(121,134,203,0.12)",
        )
    )
    fig.update_layout(
        template="plotly_dark",
        title="Equity Curve",
        paper_bgcolor="#1a1a2e",
        plot_bgcolor="#1a1a2e",
        margin=dict(l=50, r=20, t=40, b=20),
        yaxis_title="Capital ($)",
        xaxis_title="Candle",
    )
    return fig


def _metrics_panel(metrics: dict) -> html.Div:
    """Render metrics as a responsive grid of cards."""
    if "error" in metrics:
        return html.P(metrics["error"], style={"color": "#ef5350"})
    cells = [
        html.Div(
            [
                html.Span(
                    k.replace("_", " ").title(),
                    style={"color": "#aaa", "fontSize": "12px"},
                ),
                html.Div(str(v), style={"fontSize": "18px", "fontWeight": "bold"}),
            ],
            style={"padding": "10px 20px", "backgroundColor": "#16213e", "borderRadius": "6px"},
        )
        for k, v in metrics.items()
    ]
    return html.Div(
        cells,
        style={
            "display": "grid",
            "gridTemplateColumns": "repeat(auto-fill, minmax(180px, 1fr))",
            "gap": "12px",
        },
    )


if __name__ == "__main__":
    app.run(debug=True)
