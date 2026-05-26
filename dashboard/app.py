from typing import List

import dash
import plotly.graph_objects as go
from dash import Input, Output, dcc, html
from plotly.subplots import make_subplots
import pandas as pd

import config
from backtest.engine import Backtester, Trade
from backtest.metrics import calculate_metrics
from data.fetcher import fetch_ohlcv
from signals.fvg import FVG, detect_fvgs
from signals.liquidity import detect_sweeps
from signals.order_blocks import OrderBlock, detect_order_blocks

app = dash.Dash(__name__, title="ICT Backtester")

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
                    style={"padding": "8px 20px", "cursor": "pointer", "backgroundColor": "#7986cb", "border": "none", "color": "white", "borderRadius": "4px"},
                ),
            ],
        ),
        dcc.Graph(id="main-chart", style={"height": "580px"}),
        dcc.Graph(id="equity-chart", style={"height": "280px"}),
        html.Div(id="metrics-panel", style={"padding": "20px"}),
    ],
)


@app.callback(
    Output("main-chart", "figure"),
    Output("equity-chart", "figure"),
    Output("metrics-panel", "children"),
    Input("run-btn", "n_clicks"),
    Input("symbol-dd", "value"),
    Input("timeframe-dd", "value"),
)
def update(n_clicks: int, symbol: str, timeframe: str):
    """Fetch data, detect signals, run backtest, and render all charts."""
    df = fetch_ohlcv(symbol, timeframe)
    fvgs = detect_fvgs(df)
    obs = detect_order_blocks(df)

    backtester = Backtester(df)
    trades = backtester.run()
    metrics = calculate_metrics(trades, backtester.equity_curve)

    return (
        _main_chart(df, fvgs, obs, trades),
        _equity_chart(backtester.equity_curve),
        _metrics_panel(metrics),
    )


# ------------------------------------------------------------------
# Chart builders
# ------------------------------------------------------------------


def _main_chart(
    df: pd.DataFrame,
    fvgs: List[FVG],
    obs: List[OrderBlock],
    trades: List[Trade],
) -> go.Figure:
    """Candlestick + volume chart with FVG and OB overlays."""
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.75, 0.25], vertical_spacing=0.02,
    )

    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["open"], high=df["high"], low=df["low"], close=df["close"],
            name="Price",
            increasing_line_color="#26a69a",
            decreasing_line_color="#ef5350",
        ),
        row=1, col=1,
    )

    fig.add_trace(
        go.Bar(
            x=df.index, y=df["volume"],
            marker_color="#7986cb", opacity=0.5, name="Volume",
        ),
        row=2, col=1,
    )

    # FVG rectangles
    for fvg in fvgs:
        if fvg.index >= len(df):
            continue
        x0 = df.index[fvg.index]
        x1 = df.index[fvg.fill_index] if fvg.filled and fvg.fill_index else df.index[-1]
        bull = fvg.direction == "bullish"
        fig.add_shape(
            type="rect", x0=x0, x1=x1, y0=fvg.bottom, y1=fvg.top,
            fillcolor="rgba(38,166,154,0.15)" if bull else "rgba(239,83,80,0.15)",
            line=dict(color="#26a69a" if bull else "#ef5350", width=1),
            row=1, col=1,
        )

    # OB rectangles
    for ob in obs:
        if ob.index >= len(df) or ob.displaced_by_index >= len(df):
            continue
        x0 = df.index[ob.index]
        x1 = df.index[ob.displaced_by_index]
        bull = ob.direction == "bullish"
        fig.add_shape(
            type="rect", x0=x0, x1=x1, y0=ob.bottom, y1=ob.top,
            fillcolor="rgba(255,214,0,0.10)" if bull else "rgba(255,87,34,0.10)",
            line=dict(color="#FFD600" if bull else "#FF5722", width=1, dash="dot"),
            row=1, col=1,
        )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#1a1a2e",
        plot_bgcolor="#1a1a2e",
        xaxis_rangeslider_visible=False,
        showlegend=False,
        margin=dict(l=50, r=20, t=20, b=10),
    )
    return fig


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
    """Render metrics as a two-column grid."""
    if "error" in metrics:
        return html.P(metrics["error"], style={"color": "#ef5350"})

    cells = [
        html.Div(
            [
                html.Span(k.replace("_", " ").title(), style={"color": "#aaa", "fontSize": "12px"}),
                html.Div(str(v), style={"fontSize": "18px", "fontWeight": "bold"}),
            ],
            style={"padding": "10px 20px", "backgroundColor": "#16213e", "borderRadius": "6px"},
        )
        for k, v in metrics.items()
    ]
    return html.Div(
        cells,
        style={"display": "grid", "gridTemplateColumns": "repeat(auto-fill, minmax(180px, 1fr))", "gap": "12px"},
    )


if __name__ == "__main__":
    app.run(debug=True)
