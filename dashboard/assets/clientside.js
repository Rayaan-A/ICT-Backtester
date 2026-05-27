/* ICT Backtester — TradingView Lightweight Charts renderer */

// ---------------------------------------------------------------------------
// Box drawing primitive — renders FVG and OB rectangles directly on canvas
// ---------------------------------------------------------------------------

class BoxRenderer {
    constructor(boxes, series, chart) {
        this._boxes = boxes;
        this._series = series;
        this._chart = chart;
    }

    draw(target) {
        if (!this._series || !this._chart) return;
        const timeScale = this._chart.timeScale();

        target.useMediaCoordinateSpace(scope => {
            const ctx = scope.context;
            const chartWidth = scope.mediaSize.width;

            for (const box of this._boxes) {
                const x0 = timeScale.timeToCoordinate(box.left);
                if (x0 === null) continue;

                const x1 = box.extendRight
                    ? chartWidth
                    : timeScale.timeToCoordinate(box.right);
                if (!box.extendRight && x1 === null) continue;

                const y0 = this._series.priceToCoordinate(box.top);
                const y1 = this._series.priceToCoordinate(box.bottom);
                if (y0 === null || y1 === null) continue;

                const left   = Math.min(x0, x1 !== null ? x1 : chartWidth);
                const top    = Math.min(y0, y1);
                const width  = Math.abs((x1 !== null ? x1 : chartWidth) - x0);
                const height = Math.abs(y1 - y0);

                ctx.save();
                ctx.fillStyle = box.fillColor;
                ctx.fillRect(left, top, width, height);
                ctx.strokeStyle = box.borderColor;
                ctx.lineWidth = 1;
                if (box.dashed) ctx.setLineDash([4, 4]);
                ctx.strokeRect(left, top, width, height);
                ctx.setLineDash([]);
                ctx.restore();
            }
        });
    }
}

class BoxPaneView {
    constructor(renderer) { this._renderer = renderer; }
    renderer() { return this._renderer; }
    zOrder() { return 'normal'; }
}

class BoxSeriesPrimitive {
    constructor(boxes) {
        this._boxes = boxes;
        this._views = [];
    }

    attached(params) {
        this._views = [
            new BoxPaneView(
                new BoxRenderer(this._boxes, params.series, params.chart)
            )
        ];
    }

    detached() { this._views = []; }
    updateAllViews() {}
    paneViews() { return this._views; }
    autoscaleInfo() { return null; }
}

// ---------------------------------------------------------------------------
// Dash clientside callback
// ---------------------------------------------------------------------------

window.dash_clientside = window.dash_clientside || {};
window.dash_clientside.clientside = {

    renderChart: function (data) {
        if (!data || !data.candles || !data.candles.length) {
            return window.dash_clientside.no_update;
        }

        const container = document.getElementById('tv-chart');
        if (!container) return window.dash_clientside.no_update;

        // Destroy previous chart instance
        if (window._lwChart) {
            window._lwChart.remove();
            window._lwChart = null;
        }

        const chart = LightweightCharts.createChart(container, {
            autoSize: true,
            layout: {
                background: { color: '#1a1a2e' },
                textColor: '#d1d4dc',
            },
            grid: {
                vertLines: { color: '#2a2e39' },
                horzLines: { color: '#2a2e39' },
            },
            crosshair: {
                mode: LightweightCharts.CrosshairMode.Normal,
            },
            timeScale: {
                borderColor: '#485c7b',
                timeVisible: true,
                secondsVisible: false,
                rightOffset: 8,
            },
            rightPriceScale: {
                borderColor: '#485c7b',
            },
        });
        window._lwChart = chart;

        // Volume — overlaid in the bottom 18% of the price pane
        const volSeries = chart.addHistogramSeries({
            priceFormat: { type: 'volume' },
            priceScaleId: 'vol',
        });
        volSeries.priceScale().applyOptions({
            scaleMargins: { top: 0.82, bottom: 0 },
            visible: false,
        });
        volSeries.setData(data.volume || []);

        // Candlestick series
        const candleSeries = chart.addCandlestickSeries({
            upColor: '#26a69a',
            downColor: '#ef5350',
            borderUpColor: '#26a69a',
            borderDownColor: '#ef5350',
            wickUpColor: '#26a69a',
            wickDownColor: '#ef5350',
        });
        candleSeries.setData(data.candles);

        // FVG + OB rectangles via canvas primitive
        const allBoxes = (data.fvgs || []).concat(data.obs || []);
        if (allBoxes.length) {
            candleSeries.attachPrimitive(new BoxSeriesPrimitive(allBoxes));
        }

        // Liquidity sweep horizontal lines
        (data.sweeps || []).forEach(s => {
            candleSeries.createPriceLine({
                price: s.price,
                color: s.direction === 'high' ? '#FF6B6B' : '#4FC3F7',
                lineWidth: 1,
                lineStyle: LightweightCharts.LineStyle.Dashed,
                axisLabelVisible: false,
            });
        });

        chart.timeScale().fitContent();
        return true;
    }

};
