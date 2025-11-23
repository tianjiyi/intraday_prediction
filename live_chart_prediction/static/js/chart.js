/**
 * QQQ Live Prediction Chart with TradingView Lightweight Charts
 */

// Global variables
let chart = null;
let volumeChart = null;
let candlestickSeries = null;
let volumeSeries = null;
let predictionLineSeries = null;
let confidenceBandSeries = {};
let indicatorSeries = {};
let socket = null;
let showPredictions = true;
let showConfidence = true;
let showIndicators = true;
let showSMAs = true;
let isStreamingActive = false;
let currentBar = null;
let barCompletedCount = 0;
let lastCandleTime = null; // track last bar time (used for internal state)
const PREDICTION_UPDATE_INTERVAL = 5; // Generate new prediction every 5 bars
let currentTimeframe = 1; // Current timeframe in minutes
let aggregatedBar = null; // Track current aggregated bar for multi-minute timeframes

// Chart configuration
const VOLUME_CHART_HEIGHT = 120;
const chartOptions = {
    width: 800,
    height: Math.max(window.innerHeight - 140 - VOLUME_CHART_HEIGHT, 420),
    layout: {
        backgroundColor: '#1e222d',
        textColor: '#d1d4dc',
    },
    grid: {
        vertLines: {
            color: '#2B2B43',
        },
        horzLines: {
            color: '#2B2B43',
        },
    },
    crosshair: {
        mode: LightweightCharts.CrosshairMode.Normal,
    },
    rightPriceScale: {
        borderColor: '#2B2B43',
    },
    timeScale: {
        borderColor: '#2B2B43',
        timeVisible: true,
        secondsVisible: false,
        // Show time in local timezone with proper formatting
        tickMarkFormatter: (time) => {
            const date = new Date(time * 1000);
            return date.toLocaleTimeString('en-US', {
                hour: '2-digit',
                minute: '2-digit',
                hour12: false
            });
        },
    },
    // Add localization for tooltips and crosshair
    localization: {
        timeFormatter: (time) => {
            const date = new Date(time * 1000);
            return date.toLocaleString('en-US', {
                month: 'short',
                day: 'numeric',
                hour: '2-digit',
                minute: '2-digit',
                hour12: false
            });
        },
    },
};

// Initialize the chart
function initChart() {
    const chartContainer = document.getElementById('chart');
    
    // Check if LightweightCharts is loaded
    if (typeof LightweightCharts === 'undefined') {
        console.error('LightweightCharts library not loaded!');
        return;
    }
    
    chart = LightweightCharts.createChart(chartContainer, chartOptions);
    
    // Create candlestick series for historical data
    candlestickSeries = chart.addCandlestickSeries({
        upColor: '#26a69a',
        downColor: '#ef5350',
        borderVisible: false,
        wickUpColor: '#26a69a',
        wickDownColor: '#ef5350',
    });

    // Create separate volume chart
    const volumeChartContainer = document.getElementById('volume-chart');
    volumeChart = LightweightCharts.createChart(volumeChartContainer, {
        width: volumeChartContainer.clientWidth,
        height: VOLUME_CHART_HEIGHT,
        layout: {
            backgroundColor: '#1e222d',
            textColor: '#d1d4dc',
        },
        grid: {
            vertLines: { color: '#2B2B43' },
            horzLines: { color: '#2B2B43' },
        },
        rightPriceScale: {
            borderColor: '#2B2B43',
            scaleMargins: {
                top: 0.1,
                bottom: 0,
            },
        },
        timeScale: {
            borderColor: '#2B2B43',
            timeVisible: false,
            secondsVisible: false,
            visible: false, // Hide time axis on volume chart (synced with main)
        },
        crosshair: {
            mode: LightweightCharts.CrosshairMode.Normal,
        },
    });

    // Create volume series in volume chart
    volumeSeries = volumeChart.addHistogramSeries({
        color: '#26a69a',
        priceFormat: {
            type: 'volume',
        },
        priceScaleId: 'right',
    });

    // Sync time scales between main chart and volume chart
    chart.timeScale().subscribeVisibleLogicalRangeChange((logicalRange) => {
        if (logicalRange) {
            volumeChart.timeScale().setVisibleLogicalRange(logicalRange);
        }
    });

    volumeChart.timeScale().subscribeVisibleLogicalRangeChange((logicalRange) => {
        if (logicalRange) {
            chart.timeScale().setVisibleLogicalRange(logicalRange);
        }
    });

    // Create line series for mean prediction
    predictionLineSeries = chart.addLineSeries({
        color: '#2962FF',
        lineWidth: 3,
        title: 'Prediction',
    });
    
    // Create area series for confidence bands
    confidenceBandSeries.p90 = chart.addLineSeries({
        color: 'rgba(41, 98, 255, 0.1)',
        lineWidth: 1,
        lineStyle: LightweightCharts.LineStyle.Dotted,
        title: 'P90',
    });
    
    confidenceBandSeries.p75 = chart.addLineSeries({
        color: 'rgba(41, 98, 255, 0.2)',
        lineWidth: 1,
        lineStyle: LightweightCharts.LineStyle.Dotted,
        title: 'P75',
    });
    
    confidenceBandSeries.p25 = chart.addLineSeries({
        color: 'rgba(41, 98, 255, 0.2)',
        lineWidth: 1,
        lineStyle: LightweightCharts.LineStyle.Dotted,
        title: 'P25',
    });
    
    confidenceBandSeries.p10 = chart.addLineSeries({
        color: 'rgba(41, 98, 255, 0.1)',
        lineWidth: 1,
        lineStyle: LightweightCharts.LineStyle.Dotted,
        title: 'P10',
    });
    
    // Create indicator series
    indicatorSeries.vwap = chart.addLineSeries({
        color: '#FF6B6B',
        lineWidth: 2,
        lineStyle: LightweightCharts.LineStyle.Dashed,
        title: 'VWAP',
    });
    
    indicatorSeries.bbUpper = chart.addLineSeries({
        color: 'rgba(255, 193, 7, 0.5)',
        lineWidth: 1,
        title: 'BB Upper',
    });
    
    indicatorSeries.bbMiddle = chart.addLineSeries({
        color: 'rgba(255, 193, 7, 0.7)',
        lineWidth: 1,
        title: 'BB Middle',
    });
    
    indicatorSeries.bbLower = chart.addLineSeries({
        color: 'rgba(255, 193, 7, 0.5)',
        lineWidth: 1,
        title: 'BB Lower',
    });

    // Create SMA series
    indicatorSeries.sma5 = chart.addLineSeries({
        color: '#FF8C00',
        lineWidth: 2,
        title: 'SMA 5',
    });

    indicatorSeries.sma21 = chart.addLineSeries({
        color: '#FF0000',
        lineWidth: 2,
        title: 'SMA 21',
    });

    indicatorSeries.sma233 = chart.addLineSeries({
        color: '#808080',
        lineWidth: 3,
        title: 'SMA 233',
    });

    // Fit content
    chart.timeScale().fitContent();

    // Add crosshair move event listener for mouse tracking
    chart.subscribeCrosshairMove((param) => {
        handleCrosshairMove(param);
    });
}

// Convert data for chart
// Helper function to add timeframe offset to a date
function addTimeframeOffset(date, offset) {
    const newDate = new Date(date);
    if (currentTimeframe === 1440) {
        // Daily timeframe - add days
        newDate.setDate(newDate.getDate() + offset);
    } else if (currentTimeframe === 10080) {
        // Weekly timeframe - add weeks
        newDate.setDate(newDate.getDate() + (offset * 7));
    } else {
        // Intraday timeframes - add minutes
        newDate.setMinutes(newDate.getMinutes() + (offset * currentTimeframe));
    }
    return newDate;
}

function convertToChartData(data, baseDate = null, timeOffset = 0) {
    if (!data || !Array.isArray(data)) return [];

    const startDate = baseDate || new Date();

    return data.map((value, index) => {
        const date = addTimeframeOffset(startDate, index + timeOffset);
        const unixTime = Math.floor(date.getTime() / 1000);

        return {
            time: unixTime,
            value: typeof value === 'number' ? value : parseFloat(value)
        };
    });
}

// Convert candlestick data with proper OHLC aggregation for timeframe buckets
// Returns { candles: [], volumes: [] }
function convertCandlestickData(data) {
    if (!data || !Array.isArray(data)) return { candles: [], volumes: [] };

    const timeframeSeconds = currentTimeframe * 60;
    const buckets = new Map();

    // Debug: log first 3 raw timestamps and timeframe info
    console.log(`Converting ${data.length} bars with timeframe=${currentTimeframe}min (${timeframeSeconds}s)`);
    console.log('Raw timestamps (first 3):', data.slice(0, 3).map(c => c.timestamp));

    // Group bars by bucketed time and aggregate OHLC properly
    data.forEach((candle, idx) => {
        const date = new Date(candle.timestamp);
        const unixTime = Math.floor(date.getTime() / 1000);
        // Snap to nearest bucket (floor)
        const bucketedTime = Math.floor(unixTime / timeframeSeconds) * timeframeSeconds;

        // Debug: log first 3 conversions
        if (idx < 3) {
            console.log(`Bar ${idx}: "${candle.timestamp}" -> Unix: ${unixTime} -> Bucket: ${bucketedTime} (${new Date(bucketedTime * 1000).toISOString()})`);
        }

        if (!buckets.has(bucketedTime)) {
            // First bar in this bucket - initialize with its values
            buckets.set(bucketedTime, {
                time: bucketedTime,
                open: parseFloat(candle.open),
                high: parseFloat(candle.high),
                low: parseFloat(candle.low),
                close: parseFloat(candle.close),
                volume: parseFloat(candle.volume) || 0
            });
        } else {
            // Aggregate with existing bar in bucket
            const existing = buckets.get(bucketedTime);
            existing.high = Math.max(existing.high, parseFloat(candle.high));
            existing.low = Math.min(existing.low, parseFloat(candle.low));
            existing.close = parseFloat(candle.close); // Last close wins
            existing.volume += parseFloat(candle.volume) || 0; // Sum volume
        }
    });

    // Debug: log conversion result
    console.log(`Conversion result: ${data.length} bars -> ${buckets.size} unique buckets`);

    // Convert to sorted array (TradingView requires ascending time order)
    const sortedBuckets = Array.from(buckets.values()).sort((a, b) => a.time - b.time);

    // Separate candles and volumes
    const candles = sortedBuckets.map(bar => ({
        time: bar.time,
        open: bar.open,
        high: bar.high,
        low: bar.low,
        close: bar.close
    }));

    const volumes = sortedBuckets.map(bar => ({
        time: bar.time,
        value: bar.volume,
        color: bar.close >= bar.open ? '#26a69a80' : '#ef535080' // Green/Red with transparency
    }));

    return { candles, volumes };
}

// Convert SMA data to chart format with proper bucket aggregation
function convertSmaToChartData(smaValues, historicalData) {
    if (!smaValues || !Array.isArray(smaValues) || !historicalData || !Array.isArray(historicalData)) {
        return [];
    }

    const timeframeSeconds = currentTimeframe * 60;
    const buckets = new Map();  // bucket timestamp -> last SMA value

    for (let i = 0; i < Math.min(smaValues.length, historicalData.length); i++) {
        const smaValue = smaValues[i];
        if (smaValue !== null && smaValue !== undefined && !isNaN(smaValue)) {
            const date = new Date(historicalData[i].timestamp);
            const unixTime = Math.floor(date.getTime() / 1000);
            // Apply the same snapping logic as candlesticks to ensure alignment
            const bucketedTime = Math.floor(unixTime / timeframeSeconds) * timeframeSeconds;

            // Use last value in bucket (overwrite) - ensures one point per candlestick
            buckets.set(bucketedTime, parseFloat(smaValue));
        }
    }

    // Convert to sorted array (TradingView requires ascending time order)
    return Array.from(buckets.entries())
        .map(([time, value]) => ({ time, value }))
        .sort((a, b) => a.time - b.time);
}

// Update chart with new data
function updateChart(data) {
    if (!data) return;
    
    console.log('Updating chart with data:', data);
    
    // Update historical candlestick data
    if (data.historical && data.historical.length > 0) {
        const { candles: candleData, volumes: volumeData } = convertCandlestickData(data.historical);

        // Clear and repopulate barBoundaries with historical data to prevent duplicates
        barBoundaries.clear();
        aggregatedBar = null; // Reset aggregation state
        candleData.forEach(candle => {
            barBoundaries.add(candle.time);
        });

        console.log('Setting candlestick data:', candleData.slice(0, 5)); // Log first 5 for debugging
        console.log('Initialized barBoundaries with', barBoundaries.size, 'historical bars');

        candlestickSeries.setData(candleData);

        // Set volume data
        if (volumeSeries && volumeData.length > 0) {
            console.log('Setting volume data:', volumeData.slice(0, 5));
            volumeSeries.setData(volumeData);
        }

        // Track last candle time for polling logic
        lastCandleTime = candleData[candleData.length - 1]?.time || lastCandleTime;
    }
    
    // Update prediction data - note: the structure is data.prediction, not data.prediction.summary
    if (data.prediction) {
        console.log('Prediction data received:', !!data.prediction.mean_path, 'Mean path length:', data.prediction.mean_path?.length);
        const summary = data.prediction; // The prediction IS the summary
        const lastCandle = data.historical ? data.historical[data.historical.length - 1] : null;
        
        let baseDate;
        if (lastCandle) {
            // Use the aligned time from the last candle if available
            // convertCandlestickData converts to unix seconds, so we need to * 1000
            // But we don't have the converted data here easily without re-converting
            // So we just align the timestamp again
            const date = new Date(lastCandle.timestamp);
            const unixTime = Math.floor(date.getTime() / 1000);
            const timeframeSeconds = currentTimeframe * 60;
            const bucketedTime = Math.floor(unixTime / timeframeSeconds) * timeframeSeconds;
            baseDate = new Date(bucketedTime * 1000);
        } else {
            baseDate = new Date();
        }

        const timeOffset = 1; // Start predictions 1 unit (timeframe) after last candle
        
        // Update mean prediction line
        if (summary.mean_path) {
            const predictionData = convertToChartData(summary.mean_path, baseDate, timeOffset);
            console.log('Setting prediction data:', predictionData.slice(0, 5));
            predictionLineSeries.setData(predictionData);

            // Ensure prediction line is visible
            predictionLineSeries.applyOptions({
                visible: showPredictions
            });
            console.log('Prediction line visibility set to:', showPredictions);
        }
        
        // Update confidence bands
        if (summary.percentiles) {
            const p90Data = convertToChartData(summary.percentiles.p90, baseDate, timeOffset);
            const p75Data = convertToChartData(summary.percentiles.p75, baseDate, timeOffset);
            const p25Data = convertToChartData(summary.percentiles.p25, baseDate, timeOffset);
            const p10Data = convertToChartData(summary.percentiles.p10, baseDate, timeOffset);
            
            confidenceBandSeries.p90.setData(p90Data);
            confidenceBandSeries.p75.setData(p75Data);
            confidenceBandSeries.p25.setData(p25Data);
            confidenceBandSeries.p10.setData(p10Data);

            // Ensure confidence bands are visible
            Object.values(confidenceBandSeries).forEach(series => {
                series.applyOptions({ visible: showConfidence });
            });
            console.log('Confidence bands visibility set to:', showConfidence);
        }
        
        // Update indicators (draw as horizontal lines across the prediction period)
        // Skip VWAP for Day/Week timeframes (VWAP is intraday indicator only)
        const isDailyOrWeekly = currentTimeframe >= 1440;

        if (summary.current_vwap && baseDate && !isDailyOrWeekly) {
            const vwapData = [];
            for (let i = 0; i <= 30; i++) {
                const date = addTimeframeOffset(baseDate, i);
                const unixTime = Math.floor(date.getTime() / 1000);
                vwapData.push({
                    time: unixTime,
                    value: summary.current_vwap
                });
            }
            indicatorSeries.vwap.setData(vwapData);
            indicatorSeries.vwap.applyOptions({ visible: showIndicators });
        } else if (isDailyOrWeekly) {
            // Clear VWAP for Day/Week timeframes
            indicatorSeries.vwap.setData([]);
        }

        if (summary.bollinger_bands && baseDate) {
            const bb = summary.bollinger_bands;
            const bbUpperData = [];
            const bbMiddleData = [];
            const bbLowerData = [];

            for (let i = 0; i <= 30; i++) {
                const date = addTimeframeOffset(baseDate, i);
                const unixTime = Math.floor(date.getTime() / 1000);

                bbUpperData.push({ time: unixTime, value: bb.upper });
                bbMiddleData.push({ time: unixTime, value: bb.middle });
                bbLowerData.push({ time: unixTime, value: bb.lower });
            }

            indicatorSeries.bbUpper.setData(bbUpperData);
            indicatorSeries.bbMiddle.setData(bbMiddleData);
            indicatorSeries.bbLower.setData(bbLowerData);

            // Ensure Bollinger Bands visibility is preserved after data update
            ['bbUpper', 'bbMiddle', 'bbLower'].forEach(key => {
                indicatorSeries[key].applyOptions({ visible: showIndicators });
            });
        }

        // Update SMA data - only if we have historical data to work with
        // Don't clear existing SMA data if historical is not provided (e.g., WebSocket updates)
        if (data.historical && data.historical.length > 0 && summary.sma_5_series) {
            console.log('SMA data received:', {
                sma5_length: summary.sma_5_series?.length,
                sma21_length: summary.sma_21_series?.length,
                sma233_length: summary.sma_233_series?.length,
                historical_length: data.historical.length
            });

            const sma5Data = convertSmaToChartData(summary.sma_5_series, data.historical);
            const sma21Data = convertSmaToChartData(summary.sma_21_series, data.historical);
            const sma233Data = convertSmaToChartData(summary.sma_233_series, data.historical);

            console.log('Converted SMA data:', {
                sma5_points: sma5Data.length,
                sma21_points: sma21Data.length,
                sma233_points: sma233Data.length
            });

            indicatorSeries.sma5.setData(sma5Data);
            indicatorSeries.sma21.setData(sma21Data);
            indicatorSeries.sma233.setData(sma233Data);

            // Ensure SMA visibility is preserved after data update
            ['sma5', 'sma21', 'sma233'].forEach(key => {
                indicatorSeries[key].applyOptions({ visible: showSMAs });
            });

            console.log('SMA data set to chart series, visibility:', showSMAs);
        } else if (data.historical && data.historical.length > 0) {
            // Historical data present but no SMA series - clear them
            console.log('Historical data present but no SMA series - clearing');
            indicatorSeries.sma5.setData([]);
            indicatorSeries.sma21.setData([]);
            indicatorSeries.sma233.setData([]);
        } else {
            // No historical data in this update - keep existing SMA data
            console.log('No historical data in update - preserving existing SMA series');
        }

        // Update statistics panel
        updateStatsPanel(summary);

        // Update price overlay with latest bar data
        if (data.historical && data.historical.length > 0) {
            const latestBar = data.historical[data.historical.length - 1];
            updatePriceOverlay(latestBar);
            // Store latest bar data for crosshair fallback
            lastBarData = latestBar;
        }
    }
}

// Update statistics panel
function updateStatsPanel(summary) {
    // Update prices
    document.getElementById('current-price').textContent = 
        summary.current_close ? `$${summary.current_close.toFixed(2)}` : '--';
    
    const predictedPrice = summary.mean_path ? summary.mean_path[summary.mean_path.length - 1] : null;
    document.getElementById('predicted-price').textContent = 
        predictedPrice ? `$${predictedPrice.toFixed(2)}` : '--';
    
    // Update probabilities
    const probUp = summary.p_up_30m;
    const probUpElement = document.getElementById('prob-up');
    const probUpBar = document.getElementById('prob-up-bar');
    
    if (probUp !== undefined) {
        const percentVal = (probUp * 100).toFixed(1);
        probUpElement.textContent = `${percentVal}%`;
        
        if (probUpBar) {
            probUpBar.style.width = `${percentVal}%`;
            // Color gradient based on probability
            if (probUp > 0.6) {
                probUpBar.style.background = 'linear-gradient(90deg, #26a69a 0%, #00e676 100%)';
            } else if (probUp < 0.4) {
                probUpBar.style.background = 'linear-gradient(90deg, #ef5350 0%, #ff1744 100%)';
            } else {
                probUpBar.style.background = 'linear-gradient(90deg, #ffa726 0%, #ff9800 100%)';
            }
        }
    } else {
        probUpElement.textContent = '--';
        if (probUpBar) probUpBar.style.width = '0%';
    }
    
    const expReturn = summary.exp_ret_30m;
    const expReturnElement = document.getElementById('exp-return');
    expReturnElement.textContent = expReturn !== undefined ? `${(expReturn * 100).toFixed(3)}%` : '--';
    expReturnElement.className = expReturn >= 0 ? 'value positive' : 'value negative';
    
    // Update percentiles
    if (summary.percentiles) {
        const lastIdx = summary.percentiles.p90.length - 1;
        document.getElementById('p90').textContent = `$${summary.percentiles.p90[lastIdx].toFixed(2)}`;
        document.getElementById('p75').textContent = `$${summary.percentiles.p75[lastIdx].toFixed(2)}`;
        document.getElementById('p50').textContent = `$${summary.percentiles.p50[lastIdx].toFixed(2)}`;
        document.getElementById('p25').textContent = `$${summary.percentiles.p25[lastIdx].toFixed(2)}`;
        document.getElementById('p10').textContent = `$${summary.percentiles.p10[lastIdx].toFixed(2)}`;
    }
    
    // Update indicators
    document.getElementById('vwap').textContent = 
        summary.current_vwap ? `$${summary.current_vwap.toFixed(2)}` : '--';
    
    if (summary.bollinger_bands) {
        document.getElementById('bb-upper').textContent = `$${summary.bollinger_bands.upper.toFixed(2)}`;
        document.getElementById('bb-middle').textContent = `$${summary.bollinger_bands.middle.toFixed(2)}`;
        document.getElementById('bb-lower').textContent = `$${summary.bollinger_bands.lower.toFixed(2)}`;
    }
    
    // Update RTH status and data info based on asset type
    const assetType = summary.asset_type || 'stock';
    if (assetType === 'crypto') {
        document.getElementById('rth-status').textContent = '24/7 Trading';
        document.getElementById('rth-status').className = 'value positive';
    } else {
        // For stocks
        if (summary.rth_only !== undefined) {
            document.getElementById('rth-status').textContent = 
                summary.rth_only ? 'RTH Only (9:30-4:00 ET)' : 'All Hours (24/7)';
            document.getElementById('rth-status').className = 
                summary.rth_only ? 'value positive' : 'value';
        }
    }
    
    if (summary.data_bars_count !== undefined) {
        document.getElementById('data-bars').textContent = summary.data_bars_count;
    }
    
    if (summary.n_samples !== undefined) {
        document.getElementById('n-samples').textContent = summary.n_samples;
    }

    if (summary.model_name !== undefined) {
        document.getElementById('model-name').textContent = summary.model_name;
    }

    // Update Daily Fundamentals (daily context)
    if (summary.daily_context) {
        const dc = summary.daily_context;

        // Daily SMAs with price position
        const currentPrice = summary.current_close;

        // Daily SMA 5
        if (dc.daily_sma_5) {
            const sma5El = document.getElementById('daily-sma5');
            const above5 = currentPrice > dc.daily_sma_5;
            sma5El.textContent = `$${dc.daily_sma_5.toFixed(2)} ${above5 ? '↑' : '↓'}`;
            sma5El.className = `value ${above5 ? 'positive' : 'negative'}`;
        } else {
            document.getElementById('daily-sma5').textContent = '--';
        }

        // Daily SMA 21
        if (dc.daily_sma_21) {
            const sma21El = document.getElementById('daily-sma21');
            const above21 = currentPrice > dc.daily_sma_21;
            sma21El.textContent = `$${dc.daily_sma_21.toFixed(2)} ${above21 ? '↑' : '↓'}`;
            sma21El.className = `value ${above21 ? 'positive' : 'negative'}`;
        } else {
            document.getElementById('daily-sma21').textContent = '--';
        }

        // Daily SMA 233
        if (dc.daily_sma_233) {
            const sma233El = document.getElementById('daily-sma233');
            const above233 = currentPrice > dc.daily_sma_233;
            sma233El.textContent = `$${dc.daily_sma_233.toFixed(2)} ${above233 ? '↑' : '↓'}`;
            sma233El.className = `value ${above233 ? 'positive' : 'negative'}`;
        } else {
            document.getElementById('daily-sma233').textContent = '--';
        }

        // Daily RSI with signal color
        if (dc.daily_rsi !== null && dc.daily_rsi !== undefined) {
            const rsiEl = document.getElementById('daily-rsi');
            rsiEl.textContent = `${dc.daily_rsi.toFixed(1)} (${dc.rsi_signal})`;
            // Color based on RSI signal
            if (dc.rsi_signal === 'Overbought') {
                rsiEl.className = 'value negative';
            } else if (dc.rsi_signal === 'Oversold') {
                rsiEl.className = 'value positive';
            } else if (dc.rsi_signal === 'Bullish') {
                rsiEl.className = 'value positive';
            } else if (dc.rsi_signal === 'Bearish') {
                rsiEl.className = 'value negative';
            } else {
                rsiEl.className = 'value';
            }
        } else {
            document.getElementById('daily-rsi').textContent = '--';
        }

        // Daily CCI with signal color
        if (dc.daily_cci !== null && dc.daily_cci !== undefined) {
            const cciEl = document.getElementById('daily-cci');
            cciEl.textContent = `${dc.daily_cci.toFixed(1)} (${dc.cci_signal})`;
            // Color based on CCI signal
            if (dc.cci_signal === 'Strong Bullish' || dc.cci_signal === 'Bullish') {
                cciEl.className = 'value positive';
            } else if (dc.cci_signal === 'Strong Bearish' || dc.cci_signal === 'Bearish') {
                cciEl.className = 'value negative';
            } else {
                cciEl.className = 'value';
            }
        } else {
            document.getElementById('daily-cci').textContent = '--';
        }

        // Daily Trend
        if (dc.daily_trend) {
            const trendEl = document.getElementById('daily-trend');
            trendEl.textContent = dc.daily_trend;
            // Color based on trend
            if (dc.daily_trend.includes('Bullish')) {
                trendEl.className = 'value positive';
            } else if (dc.daily_trend.includes('Bearish')) {
                trendEl.className = 'value negative';
            } else {
                trendEl.className = 'value';
            }
        } else {
            document.getElementById('daily-trend').textContent = '--';
        }

        // Daily Price Levels (currentPrice already declared above)

        // Previous Day High
        if (dc.prev_day_high) {
            const el = document.getElementById('prev-day-high');
            el.textContent = `$${dc.prev_day_high.toFixed(2)}`;
            el.className = currentPrice > dc.prev_day_high ? 'value positive' : 'value level-resistance';
        } else {
            document.getElementById('prev-day-high').textContent = '--';
        }

        // Previous Day Low
        if (dc.prev_day_low) {
            const el = document.getElementById('prev-day-low');
            el.textContent = `$${dc.prev_day_low.toFixed(2)}`;
            el.className = currentPrice < dc.prev_day_low ? 'value negative' : 'value level-support';
        } else {
            document.getElementById('prev-day-low').textContent = '--';
        }

        // Previous Day Close
        if (dc.prev_day_close) {
            const el = document.getElementById('prev-day-close');
            el.textContent = `$${dc.prev_day_close.toFixed(2)}`;
            el.className = currentPrice > dc.prev_day_close ? 'value positive' : 'value negative';
        } else {
            document.getElementById('prev-day-close').textContent = '--';
        }

        // 3-Day High
        if (dc.three_day_high) {
            const el = document.getElementById('three-day-high');
            el.textContent = `$${dc.three_day_high.toFixed(2)}`;
            el.className = currentPrice > dc.three_day_high ? 'value positive' : 'value level-resistance';
        } else {
            document.getElementById('three-day-high').textContent = '--';
        }

        // 3-Day Low
        if (dc.three_day_low) {
            const el = document.getElementById('three-day-low');
            el.textContent = `$${dc.three_day_low.toFixed(2)}`;
            el.className = currentPrice < dc.three_day_low ? 'value negative' : 'value level-support';
        } else {
            document.getElementById('three-day-low').textContent = '--';
        }
    }

    // Update last update time - use prediction timestamp if available
    let updateTimeText = 'Last Update: ';
    if (summary.timestamp) {
        // Convert the prediction timestamp to local time
        const predictionTime = new Date(summary.timestamp);
        updateTimeText += predictionTime.toLocaleTimeString('en-US', {
            hour: '2-digit',
            minute: '2-digit',
            second: '2-digit',
            hour12: false
        });
    } else {
        updateTimeText += new Date().toLocaleTimeString('en-US', {
            hour: '2-digit',
            minute: '2-digit',
            second: '2-digit',
            hour12: false
        });
    }
    
    document.getElementById('last-update').textContent = updateTimeText;
}

// Update price overlay with OHLC data
function updatePriceOverlay(barData) {
    if (!barData) return;

    // Format prices to 2 decimal places
    const formatPrice = (price) => {
        return typeof price === 'number' ? price.toFixed(2) : '--';
    };

    // Update OHLC values
    document.getElementById('overlay-open').textContent = formatPrice(barData.open);
    document.getElementById('overlay-high').textContent = formatPrice(barData.high);
    document.getElementById('overlay-low').textContent = formatPrice(barData.low);
    document.getElementById('overlay-close').textContent = formatPrice(barData.close);
}

// Update ticker symbol in overlay
function updateTickerSymbol(symbol) {
    const tickerElement = document.getElementById('ticker-symbol');
    if (tickerElement) {
        tickerElement.textContent = symbol.toUpperCase();
    }
}

// Handle crosshair movement for mouse tracking
let lastBarData = null; // Store the latest bar data for fallback

function handleCrosshairMove(param) {
    // If no valid data point or mouse is outside chart area
    if (!param.time || !param.point || param.point.x < 0 || param.point.y < 0) {
        // Fallback to latest bar data when mouse leaves chart
        if (lastBarData) {
            updatePriceOverlay(lastBarData);
        }
        return;
    }

    // Get the OHLC data for the candlestick series at the crosshair position
    const candleData = param.seriesData?.get(candlestickSeries);

    if (candleData) {
        // Convert the candlestick data to the format expected by updatePriceOverlay
        const barData = {
            open: candleData.open,
            high: candleData.high,
            low: candleData.low,
            close: candleData.close,
            timestamp: new Date(param.time * 1000).toISOString() // Convert Unix timestamp back to ISO string
        };

        // Update the price overlay with the hovered bar's data
        updatePriceOverlay(barData);
    }
}

// Toggle visibility functions
function togglePredictions() {
    showPredictions = !showPredictions;
    predictionLineSeries.applyOptions({
        visible: showPredictions
    });
    document.getElementById('toggle-predictions').classList.toggle('active', showPredictions);
}

function toggleConfidenceBands() {
    showConfidence = !showConfidence;
    Object.values(confidenceBandSeries).forEach(series => {
        series.applyOptions({ visible: showConfidence });
    });
    document.getElementById('toggle-confidence').classList.toggle('active', showConfidence);
}

function toggleIndicators() {
    showIndicators = !showIndicators;
    // Toggle only VWAP and Bollinger Bands, not SMAs
    ['vwap', 'bbUpper', 'bbMiddle', 'bbLower'].forEach(key => {
        if (indicatorSeries[key]) {
            indicatorSeries[key].applyOptions({ visible: showIndicators });
        }
    });
    document.getElementById('toggle-indicators').classList.toggle('active', showIndicators);
}

function toggleSMAs() {
    showSMAs = !showSMAs;
    // Toggle only SMA series
    ['sma5', 'sma21', 'sma233'].forEach(key => {
        if (indicatorSeries[key]) {
            indicatorSeries[key].applyOptions({ visible: showSMAs });
        }
    });
    document.getElementById('toggle-sma').classList.toggle('active', showSMAs);
}

// Initialize WebSocket connection
function initWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws`;
    
    socket = new WebSocket(wsUrl);
    
    socket.onopen = () => {
        console.log('Connected to prediction server');
        document.getElementById('connection-status').textContent = '● Connected';
        document.getElementById('connection-status').className = 'status connected';
    };
    
    socket.onclose = () => {
        console.log('Disconnected from prediction server');
        document.getElementById('connection-status').textContent = '● Disconnected';
        document.getElementById('connection-status').className = 'status disconnected';
        isStreamingActive = false;
        // Streaming only; no REST polling fallback
        
        // Attempt to reconnect after 5 seconds
        setTimeout(() => {
            console.log('Attempting to reconnect...');
            initWebSocket();
        }, 5000);
    };
    
    socket.onmessage = (event) => {
        const data = JSON.parse(event.data);
        const messageType = data.type;
        
        switch (messageType) {
            case 'connected':
                console.log('WebSocket connected:', data.message);
                break;
                
            case 'prediction_update':
                console.log('Received prediction update', data);
                updateChart(data);
                break;
                
            case 'bar_update':
                console.log('Bar update:', data);
                updateCurrentBar(data);
                break;
                
            case 'bar_complete':
                console.log('Bar completed:', data);
                addCompletedBar(data);
                
                // Generate new prediction every N bars
                barCompletedCount++;
                if (barCompletedCount % PREDICTION_UPDATE_INTERVAL === 0) {
                    console.log(`Generating new prediction after ${barCompletedCount} bars`);
                    refreshData();
                }
                break;
                
            case 'stream_status':
                console.log('Stream status:', data);
                isStreamingActive = data.connected;
                updateStreamStatus(data.connected);
                // Streaming only; no REST polling fallback
                break;
                
            case 'stream_started':
                console.log('Stream started:', data);
                isStreamingActive = true;
                updateStreamStatus(true);
                // Streaming only; no REST polling fallback
                break;
                
            case 'stream_stopped':
                console.log('Stream stopped:', data);
                isStreamingActive = false;
                break;
                
            case 'stream_error':
                console.warn('Stream error:', data);
                
                // Show user-friendly message for crypto limitations with fallback info
                if (data.fallback) {
                    // Handle fallback scenarios (timeout, auth errors, etc.)
                    document.getElementById('stream-status').innerHTML = 
                        '⚠️ Real-time crypto unavailable on current data feed - Using historical data only';
                    document.getElementById('stream-status').className = 'stream-status warning';
                    
                    // Show temporary notification
                    showNotification('Crypto streaming issue - switched to historical data mode', 'warning', 5000);
                } else {
                    // Generic stream error
                    document.getElementById('stream-status').innerHTML = 
                        '⚠️ Stream connection issue';
                    document.getElementById('stream-status').className = 'stream-status error';
                }
                break;
                
            case 'stream_info':
                console.info('Stream info:', data);
                if (data.mode === 'historical_primary') {
                    document.getElementById('stream-status').innerHTML = 
                        '📊 Historical Data Mode';
                    document.getElementById('stream-status').className = 'stream-status historical';
                    showNotification(data.info, 'info', 4000);
                }
                break;
                
            case 'error':
                console.error('WebSocket error:', data.message);
                break;
                
            default:
                console.log('Unknown message type:', messageType, data);
        }
    };
    
    socket.onerror = (error) => {
        console.error('WebSocket connection error:', error);
        updateStreamStatus('Connection Error', false);
    };
}

// Update current bar in real-time
function updateCurrentBar(barData) {
    if (!candlestickSeries) return;

    // Convert timestamp to Unix time and bucket to timeframe granularity
    const unixTime = Math.floor(new Date(barData.timestamp).getTime() / 1000);
    const timeframeSeconds = currentTimeframe * 60;
    const bucketedTime = Math.floor(unixTime / timeframeSeconds) * timeframeSeconds;

    let displayBar;
    let volumeValue;

    if (aggregatedBar && aggregatedBar.time === bucketedTime) {
        // Merge current update with existing aggregated data
        displayBar = {
            time: bucketedTime,
            open: aggregatedBar.open,
            high: Math.max(aggregatedBar.high, barData.high),
            low: Math.min(aggregatedBar.low, barData.low),
            close: barData.close
        };
        volumeValue = (aggregatedBar.volume || 0) + (parseFloat(barData.volume) || 0);
    } else {
        // Start of a new bucket (or no aggregation yet)
        displayBar = {
            time: bucketedTime,
            open: barData.open,
            high: barData.high,
            low: barData.low,
            close: barData.close
        };
        volumeValue = parseFloat(barData.volume) || 0;
    }

    // Update the last bar in the series
    candlestickSeries.update(displayBar);

    // Update volume bar
    if (volumeSeries) {
        volumeSeries.update({
            time: bucketedTime,
            value: volumeValue,
            color: displayBar.close >= displayBar.open ? '#26a69a80' : '#ef535080'
        });
    }

    // Update price overlay with current bar data
    updatePriceOverlay(barData);

    // Store latest bar data for crosshair fallback
    lastBarData = barData;

    // Store current bar for reference
    currentBar = displayBar;
    currentBar.volume = volumeValue; // Store volume for aggregation
    lastCandleTime = Math.max(lastCandleTime || 0, displayBar.time);
}

// Track bar boundaries to prevent duplicate updates
let barBoundaries = new Set();

// Add completed bar to chart
function addCompletedBar(barData) {
    if (!candlestickSeries) return;

    // Convert timestamp to Unix time and bucket to timeframe granularity
    const unixTime = Math.floor(new Date(barData.timestamp).getTime() / 1000);
    const timeframeSeconds = currentTimeframe * 60;
    const bucketedTime = Math.floor(unixTime / timeframeSeconds) * timeframeSeconds;

    // Check if this is actually a new bucket
    const isNewBucket = !barBoundaries.has(bucketedTime);
    const volumeValue = parseFloat(barData.volume) || 0;

    if (isNewBucket) {
        // New bucket - start new aggregated bar
        aggregatedBar = {
            time: bucketedTime,
            open: barData.open,
            high: barData.high,
            low: barData.low,
            close: barData.close,
            volume: volumeValue
        };

        candlestickSeries.update(aggregatedBar);
        barBoundaries.add(bucketedTime);
        console.log('Added new bar at:', new Date(bucketedTime * 1000).toISOString());

        // Keep only last 1000 bars in memory for performance
        if (barBoundaries.size > 1000) {
            const oldestTime = Math.min(...barBoundaries);
            barBoundaries.delete(oldestTime);
        }
    } else {
        // Update existing aggregated bar
        if (!aggregatedBar || aggregatedBar.time !== bucketedTime) {
            // Fallback if aggregatedBar state was lost/reset but boundary exists
             aggregatedBar = {
                time: bucketedTime,
                open: barData.open,
                high: barData.high,
                low: barData.low,
                close: barData.close,
                volume: volumeValue
            };
        } else {
            // Aggregate data
            aggregatedBar.high = Math.max(aggregatedBar.high, barData.high);
            aggregatedBar.low = Math.min(aggregatedBar.low, barData.low);
            aggregatedBar.close = barData.close;
            aggregatedBar.volume = (aggregatedBar.volume || 0) + volumeValue;
        }

        candlestickSeries.update(aggregatedBar);
        console.log('Updated existing bar at:', new Date(bucketedTime * 1000).toISOString());
    }

    // Update volume bar
    if (volumeSeries) {
        volumeSeries.update({
            time: bucketedTime,
            value: aggregatedBar.volume,
            color: aggregatedBar.close >= aggregatedBar.open ? '#26a69a80' : '#ef535080'
        });
    }

    // Update price overlay with completed bar data
    updatePriceOverlay(barData);

    // Store latest bar data for crosshair fallback
    lastBarData = barData;

    // Reset current bar tracking
    currentBar = null;
    lastCandleTime = aggregatedBar.time;
}

// Update stream status indicator
function updateStreamStatus(isConnected) {
    const statusElement = document.getElementById('stream-status');
    if (statusElement) {
        if (isConnected) {
            statusElement.textContent = '🔴 LIVE';
            statusElement.className = 'stream-status live';
        } else {
            statusElement.textContent = '';
            statusElement.className = 'stream-status';
        }
    }
}

// No REST polling fallback; WebSocket streaming only

// Show notification to user
function showNotification(message, type = 'info', duration = 3000) {
    // Create notification element
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.textContent = message;
    
    // Style the notification
    Object.assign(notification.style, {
        position: 'fixed',
        top: '20px',
        right: '20px',
        padding: '12px 20px',
        borderRadius: '4px',
        color: 'white',
        fontWeight: 'bold',
        zIndex: '1000',
        boxShadow: '0 4px 6px rgba(0, 0, 0, 0.1)',
        opacity: '0',
        transform: 'translateX(100%)',
        transition: 'all 0.3s ease'
    });
    
    // Set background color based on type
    switch (type) {
        case 'warning':
            notification.style.backgroundColor = '#f59e0b';
            break;
        case 'error':
            notification.style.backgroundColor = '#ef4444';
            break;
        case 'success':
            notification.style.backgroundColor = '#10b981';
            break;
        default:
            notification.style.backgroundColor = '#3b82f6';
    }
    
    // Add to page
    document.body.appendChild(notification);
    
    // Animate in
    setTimeout(() => {
        notification.style.opacity = '1';
        notification.style.transform = 'translateX(0)';
    }, 10);
    
    // Remove after duration
    setTimeout(() => {
        notification.style.opacity = '0';
        notification.style.transform = 'translateX(100%)';
        setTimeout(() => {
            if (notification.parentNode) {
                notification.parentNode.removeChild(notification);
            }
        }, 300);
    }, duration);
}

// Start real-time streaming
async function startStreaming() {
    const ticker = document.getElementById('ticker-select').value;
    const timeframe = document.getElementById('timeframe-select').value;
    
    try {
        const response = await fetch('/api/start_stream', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                symbol: ticker,
                timeframe: timeframe + 'Min'
            })
        });
        
        const result = await response.json();
        console.log('Stream started:', result);
        isStreamingActive = true;
        
    } catch (error) {
        console.error('Failed to start stream:', error);
    }
}

// Stop real-time streaming
async function stopStreaming() {
    try {
        const response = await fetch('/api/stop_stream', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        });
        
        const result = await response.json();
        console.log('Stream stopped:', result);
        isStreamingActive = false;
        updateStreamStatus(false);
        
    } catch (error) {
        console.error('Failed to stop stream:', error);
    }
}

// Load initial data
async function loadInitialData() {
    try {
        const symbol = document.getElementById('ticker-select').value;
        const timeframe = document.getElementById('timeframe-select').value;
        currentTimeframe = parseInt(timeframe);
        
        const response = await fetch(`/api/initial_data?symbol=${symbol}&timeframe=${timeframe}`);
        const data = await response.json();
        
        if (data.error) {
            console.error('Error loading initial data:', data.error);
            return;
        }
        
        updateChart(data);
    } catch (error) {
        console.error('Failed to load initial data:', error);
    }
}

// Refresh data manually
async function refreshData() {
    const btn = document.getElementById('refresh-btn');
    btn.disabled = true;
    btn.textContent = '⏳ Loading...';
    
    try {
        // Use new API endpoint for on-demand prediction
        const response = await fetch('/api/generate_prediction');
        const data = await response.json();
        
        if (data.error) {
            console.error('Error generating prediction:', data.error);
            return;
        }
        
        updateChart(data);
        
        // Also send via WebSocket to update other connected clients
        if (socket && socket.readyState === WebSocket.OPEN) {
            socket.send(JSON.stringify({
                type: 'request_update'
            }));
        }
    } catch (error) {
        console.error('Failed to refresh data:', error);
    } finally {
        btn.disabled = false;
        btn.textContent = '🔄 Refresh';
    }
}

// Handle timeframe changes
async function handleTimeframeChange() {
    const timeframe = document.getElementById('timeframe-select').value;
    console.log('Timeframe changed to:', timeframe);
    currentTimeframe = parseInt(timeframe);

    // Reload data with new timeframe via HTTP API
    await loadInitialData();

    // Also notify server via WebSocket for streaming updates
    if (socket && socket.readyState === WebSocket.OPEN) {
        socket.send(JSON.stringify({
            type: 'settings_changed',
            timeframe: timeframe,
            changeType: 'timeframe'
        }));
    }
}

// Handle ticker changes
async function handleTickerChange() {
    const ticker = document.getElementById('ticker-select').value;
    console.log('Ticker changed to:', ticker);
    
    // Stop current stream if active
    if (isStreamingActive) {
        await stopStreaming();
    }
    
    // Get friendly display name
    const displayName = getFriendlyName(ticker);
    
    // Update the header title
    const headerTitle = document.querySelector('.header h1');
    headerTitle.textContent = `🚀 Kronos ${displayName} Prediction`;

    // Update ticker symbol in price overlay
    updateTickerSymbol(ticker);

    // Send settings change to server
    if (socket && socket.readyState === WebSocket.OPEN) {
        socket.send(JSON.stringify({
            type: 'settings_changed',
            ticker: ticker,
            changeType: 'ticker'
        }));
    }
    
    // Start new stream for new ticker
    setTimeout(() => startStreaming(), 1000);
}

// Get friendly display name for ticker
function getFriendlyName(ticker) {
    const cryptoNames = {
        'BTC/USD': 'Bitcoin',
        'ETH/USD': 'Ethereum', 
        'LTC/USD': 'Litecoin',
        'DOGE/USD': 'Dogecoin'
    };
    
    return cryptoNames[ticker] || ticker;
}

// Handle periodic data checks (can be called by timer if needed)
function checkForNewData() {
    if (socket && socket.readyState === WebSocket.OPEN) {
        socket.send(JSON.stringify({
            type: 'check_for_new_data'
        }));
    }
}

// Resize chart on window resize
let currentVolumeHeight = VOLUME_CHART_HEIGHT; // Track current volume chart height

function resizeChart() {
    if (chart) {
        const container = document.getElementById('chart');
        const volumeContainer = document.getElementById('volume-chart');
        const chartsWrapper = document.querySelector('.charts-wrapper');
        const divider = document.getElementById('chart-divider');
        const dividerHeight = divider ? divider.offsetHeight : 8;

        // Use wrapper height if available, otherwise calculate from window
        const totalHeight = chartsWrapper ? chartsWrapper.clientHeight : Math.max(window.innerHeight - 140, 500);
        const mainChartHeight = totalHeight - currentVolumeHeight - dividerHeight;

        chart.applyOptions({
            width: container.clientWidth,
            height: mainChartHeight
        });
        container.style.height = mainChartHeight + 'px';

        // Resize volume chart
        if (volumeChart && volumeContainer) {
            volumeChart.applyOptions({
                width: volumeContainer.clientWidth,
                height: currentVolumeHeight
            });
            volumeContainer.style.height = currentVolumeHeight + 'px';
        }
    }
}

// ==========================================
// Chart Divider Drag Functionality
// ==========================================

const MIN_MAIN_CHART_HEIGHT = 200;
const MIN_VOLUME_HEIGHT = 60;
const MAX_VOLUME_HEIGHT = 300;

function initChartDivider() {
    const divider = document.getElementById('chart-divider');
    const chartsWrapper = document.querySelector('.charts-wrapper');
    const mainChartContainer = document.getElementById('chart');
    const volumeContainer = document.getElementById('volume-chart');

    if (!divider || !chartsWrapper || !mainChartContainer || !volumeContainer) {
        console.warn('Chart divider elements not found');
        return;
    }

    let isDragging = false;
    let startY = 0;
    let startVolumeHeight = 0;

    // Mouse down - start dragging
    divider.addEventListener('mousedown', (e) => {
        isDragging = true;
        startY = e.clientY;
        startVolumeHeight = volumeContainer.offsetHeight;
        divider.classList.add('dragging');
        document.body.style.cursor = 'row-resize';
        document.body.style.userSelect = 'none';
        e.preventDefault();
    });

    // Mouse move - resize charts
    document.addEventListener('mousemove', (e) => {
        if (!isDragging) return;

        const deltaY = startY - e.clientY; // Positive = dragging up = more volume
        let newVolumeHeight = startVolumeHeight + deltaY;

        // Get available space
        const wrapperHeight = chartsWrapper.clientHeight;
        const dividerHeight = divider.offsetHeight;
        const maxVolumeHeight = Math.min(MAX_VOLUME_HEIGHT, wrapperHeight - MIN_MAIN_CHART_HEIGHT - dividerHeight);

        // Clamp to min/max
        newVolumeHeight = Math.max(MIN_VOLUME_HEIGHT, Math.min(maxVolumeHeight, newVolumeHeight));

        // Calculate main chart height
        const newMainChartHeight = wrapperHeight - newVolumeHeight - dividerHeight;

        // Update current volume height
        currentVolumeHeight = newVolumeHeight;

        // Apply new sizes
        mainChartContainer.style.height = newMainChartHeight + 'px';
        volumeContainer.style.height = newVolumeHeight + 'px';

        // Update chart dimensions
        if (chart) {
            chart.applyOptions({
                width: mainChartContainer.clientWidth,
                height: newMainChartHeight
            });
        }

        if (volumeChart) {
            volumeChart.applyOptions({
                width: volumeContainer.clientWidth,
                height: newVolumeHeight
            });
        }
    });

    // Mouse up - stop dragging
    document.addEventListener('mouseup', () => {
        if (isDragging) {
            isDragging = false;
            divider.classList.remove('dragging');
            document.body.style.cursor = '';
            document.body.style.userSelect = '';

            // Save to localStorage for persistence
            localStorage.setItem('volumeChartHeight', currentVolumeHeight);
        }
    });

    // Load saved height from localStorage
    const savedHeight = localStorage.getItem('volumeChartHeight');
    if (savedHeight) {
        currentVolumeHeight = parseInt(savedHeight, 10);
        if (isNaN(currentVolumeHeight) || currentVolumeHeight < MIN_VOLUME_HEIGHT) {
            currentVolumeHeight = VOLUME_CHART_HEIGHT;
        }
    }
}

// ==========================================
// LLM Analysis Functions
// ==========================================

let llmAvailable = false;
let currentAnalysis = {
    highlights: null,
    technical: null,
    rules: null,
    sentiment: null
};

// Check LLM status on load
async function checkLLMStatus() {
    const statusIndicator = document.getElementById('llm-status-indicator');
    const statusText = document.getElementById('llm-status-text');

    statusIndicator.className = 'status-indicator checking';
    statusText.textContent = 'Checking LLM status...';

    try {
        const response = await fetch('/api/llm_status');
        const data = await response.json();

        llmAvailable = data.available;

        if (llmAvailable) {
            statusIndicator.className = 'status-indicator available';
            statusText.textContent = `LLM Ready (${data.model || 'Gemini'})`;
        } else {
            statusIndicator.className = 'status-indicator unavailable';
            statusText.textContent = data.reason || 'LLM not configured';
        }
    } catch (error) {
        console.error('Failed to check LLM status:', error);
        statusIndicator.className = 'status-indicator unavailable';
        statusText.textContent = 'LLM status check failed';
        llmAvailable = false;
    }
}

// Run analysis
async function runAnalysis() {
    const analyzeBtn = document.getElementById('analyze-btn');
    const analysisType = document.getElementById('analysis-type').value;
    const ticker = document.getElementById('ticker-select').value;

    if (!llmAvailable) {
        showNotification('LLM service not available. Please configure Gemini API key.', 'error');
        return;
    }

    // Update button state
    analyzeBtn.disabled = true;
    analyzeBtn.textContent = '⏳ Analyzing...';
    analyzeBtn.classList.add('loading');

    try {
        const response = await fetch('/api/analyze', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                symbol: ticker,
                analysis_type: analysisType
            })
        });

        const data = await response.json();

        if (data.error) {
            throw new Error(data.error);
        }

        // Update analysis content
        if (data.highlights) {
            currentAnalysis.highlights = data.highlights;
            document.getElementById('highlights-analysis').innerHTML = formatAnalysisText(data.highlights);
            // Auto-switch to highlights tab
            switchAnalysisTab('highlights');
        }

        if (data.technical) {
            currentAnalysis.technical = data.technical;
            document.getElementById('technical-analysis').innerHTML = formatAnalysisText(data.technical);
        }

        if (data.rules) {
            currentAnalysis.rules = data.rules;
            document.getElementById('rules-analysis').innerHTML = formatAnalysisText(data.rules);
        }

        if (data.sentiment) {
            currentAnalysis.sentiment = data.sentiment;
            document.getElementById('sentiment-analysis').innerHTML = formatAnalysisText(data.sentiment);
        }

        // Update timestamp
        const timestamp = new Date().toLocaleString();
        document.getElementById('analysis-timestamp').textContent = `Last analysis: ${timestamp}`;

        showNotification('Analysis complete!', 'success');

    } catch (error) {
        console.error('Analysis failed:', error);
        showNotification(`Analysis failed: ${error.message}`, 'error');
    } finally {
        analyzeBtn.disabled = false;
        analyzeBtn.textContent = '✨ Analyze';
        analyzeBtn.classList.remove('loading');
    }
}

// Format analysis text (markdown-like to HTML)
function formatAnalysisText(text) {
    if (!text) return '<p class="placeholder">No analysis available.</p>';

    // Convert markdown-like syntax to HTML
    let html = text
        // Escape HTML
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        // Headers
        .replace(/^### (.+)$/gm, '<h3>$1</h3>')
        .replace(/^## (.+)$/gm, '<h2>$1</h2>')
        .replace(/^# (.+)$/gm, '<h1>$1</h1>')
        // Bold
        .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
        // Italic
        .replace(/\*(.+?)\*/g, '<em>$1</em>')
        // Bullet points
        .replace(/^[-•] (.+)$/gm, '<li>$1</li>')
        // Numbered lists
        .replace(/^\d+\. (.+)$/gm, '<li>$1</li>')
        // Code
        .replace(/`(.+?)`/g, '<code>$1</code>')
        // Line breaks
        .replace(/\n\n/g, '</p><p>')
        .replace(/\n/g, '<br>');

    // Wrap in paragraphs
    html = '<p>' + html + '</p>';

    // Wrap consecutive list items
    html = html.replace(/(<li>.*?<\/li>(?:<br>)?)+/g, '<ul>$&</ul>');
    html = html.replace(/<br><\/li>/g, '</li>');

    // Highlight signals
    html = html.replace(/\b(Buy|Strong Buy|Bullish|BULLISH)\b/gi, '<span class="signal-buy">$1</span>');
    html = html.replace(/\b(Sell|Strong Sell|Bearish|BEARISH)\b/gi, '<span class="signal-sell">$1</span>');
    html = html.replace(/\b(Hold|Wait|Neutral|NEUTRAL)\b/gi, '<span class="signal-hold">$1</span>');
    html = html.replace(/\b(PASS)\b/g, '<span class="signal-buy">$1</span>');
    html = html.replace(/\b(FAIL)\b/g, '<span class="signal-sell">$1</span>');
    html = html.replace(/\b(WARN)\b/g, '<span class="signal-hold">$1</span>');
    // Day type highlights
    html = html.replace(/\b(TRENDING DAY|Trending Day|Trend Day)\b/gi, '<span class="signal-buy">$1</span>');
    html = html.replace(/\b(RANGE DAY|Range Day|Swing Day|SWING DAY|Choppy Day)\b/gi, '<span class="signal-hold">$1</span>');
    // Key levels
    html = html.replace(/\b(Resistance|RESISTANCE)\b/gi, '<span class="signal-sell">$1</span>');
    html = html.replace(/\b(Support|SUPPORT)\b/gi, '<span class="signal-buy">$1</span>');

    return html;
}

// Handle tab switching
function switchAnalysisTab(tabName) {
    // Update tab buttons
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.remove('active');
        if (btn.dataset.tab === tabName) {
            btn.classList.add('active');
        }
    });

    // Update tab content
    document.querySelectorAll('.tab-content').forEach(content => {
        content.classList.remove('active');
    });
    document.getElementById(`tab-${tabName}`).classList.add('active');
}

// Toggle analysis panel
function toggleAnalysisPanel() {
    const panel = document.getElementById('analysis-panel');
    panel.classList.toggle('collapsed');
}

// Initialize analysis panel event listeners
function initAnalysisPanel() {
    // Check LLM status
    checkLLMStatus();

    // Analyze button
    const analyzeBtn = document.getElementById('analyze-btn');
    if (analyzeBtn) {
        analyzeBtn.addEventListener('click', runAnalysis);
    }

    // Toggle panel button
    const toggleBtn = document.getElementById('toggle-analysis-panel');
    if (toggleBtn) {
        toggleBtn.addEventListener('click', toggleAnalysisPanel);
    }

    // Tab buttons
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            switchAnalysisTab(btn.dataset.tab);
        });
    });
}

// ==========================================
// Chat Functions
// ==========================================

let chatHistory = [];
let latestPredictionData = null; // Store latest prediction data for chat context

// Store prediction data when chart updates
function storePredictionData(data) {
    if (data && data.prediction) {
        latestPredictionData = data;
    }
}

// Send chat message
async function sendChatMessage() {
    const input = document.getElementById('chat-input');
    const sendBtn = document.getElementById('chat-send-btn');
    const messagesContainer = document.getElementById('chat-messages');

    const message = input.value.trim();
    if (!message) return;

    // Disable input while processing
    input.disabled = true;
    sendBtn.disabled = true;

    // Add user message to chat
    addChatMessage('user', message);
    input.value = '';

    // Add loading indicator
    const loadingDiv = document.createElement('div');
    loadingDiv.className = 'chat-message assistant loading';
    loadingDiv.innerHTML = `
        <div class="message-content">
            <div class="typing-indicator">
                <span></span><span></span><span></span>
            </div>
            Thinking...
        </div>
    `;
    messagesContainer.appendChild(loadingDiv);
    scrollChatToBottom();

    try {
        const ticker = document.getElementById('ticker-select').value;

        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                message: message,
                symbol: ticker,
                chat_history: chatHistory.slice(-10) // Send last 10 messages for context
            })
        });

        const data = await response.json();

        // Remove loading indicator
        messagesContainer.removeChild(loadingDiv);

        if (data.error) {
            addChatMessage('assistant', `Error: ${data.error}`);
        } else {
            addChatMessage('assistant', data.response);

            // Store in history
            chatHistory.push({ role: 'user', content: message });
            chatHistory.push({ role: 'assistant', content: data.response });
        }

    } catch (error) {
        console.error('Chat error:', error);
        messagesContainer.removeChild(loadingDiv);
        addChatMessage('assistant', 'Sorry, there was an error processing your message. Please try again.');
    } finally {
        input.disabled = false;
        sendBtn.disabled = false;
        input.focus();
    }
}

// Add message to chat display
function addChatMessage(role, content) {
    const messagesContainer = document.getElementById('chat-messages');

    const messageDiv = document.createElement('div');
    messageDiv.className = `chat-message ${role}`;

    // Format the content (convert markdown-like to HTML for assistant messages)
    const formattedContent = role === 'assistant' ? formatChatResponse(content) : escapeHtml(content);

    const time = new Date().toLocaleTimeString('en-US', {
        hour: '2-digit',
        minute: '2-digit',
        hour12: false
    });

    messageDiv.innerHTML = `
        <div class="message-content">${formattedContent}</div>
        <div class="message-time">${time}</div>
    `;

    messagesContainer.appendChild(messageDiv);
    scrollChatToBottom();
}

// Format chat response (similar to analysis text formatting)
function formatChatResponse(text) {
    if (!text) return '';

    let html = text
        // Escape HTML
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        // Headers
        .replace(/^### (.+)$/gm, '<h3>$1</h3>')
        .replace(/^## (.+)$/gm, '<h2>$1</h2>')
        .replace(/^# (.+)$/gm, '<h1>$1</h1>')
        // Bold
        .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
        // Italic
        .replace(/\*(.+?)\*/g, '<em>$1</em>')
        // Bullet points
        .replace(/^[-•] (.+)$/gm, '<li>$1</li>')
        // Code
        .replace(/`(.+?)`/g, '<code>$1</code>')
        // Line breaks
        .replace(/\n\n/g, '<br><br>')
        .replace(/\n/g, '<br>');

    // Wrap consecutive list items
    html = html.replace(/(<li>.*?<\/li>(<br>)?)+/g, '<ul>$&</ul>');
    html = html.replace(/<br><\/li>/g, '</li>');

    // Highlight trading signals
    html = html.replace(/\b(Buy|Strong Buy|Bullish|BULLISH|Long)\b/gi, '<span class="signal-buy">$1</span>');
    html = html.replace(/\b(Sell|Strong Sell|Bearish|BEARISH|Short)\b/gi, '<span class="signal-sell">$1</span>');
    html = html.replace(/\b(Hold|Wait|Neutral|NEUTRAL)\b/gi, '<span class="signal-hold">$1</span>');

    return html;
}

// Escape HTML for user messages
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Scroll chat to bottom
function scrollChatToBottom() {
    const messagesContainer = document.getElementById('chat-messages');
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

// Initialize chat
function initChat() {
    const input = document.getElementById('chat-input');
    const sendBtn = document.getElementById('chat-send-btn');

    if (input && sendBtn) {
        // Send on button click
        sendBtn.addEventListener('click', sendChatMessage);

        // Send on Enter key
        input.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendChatMessage();
            }
        });
    }
}

// Initialize everything when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    // Initialize chart
    initChart();

    // Initialize chart divider for resizing
    initChartDivider();

    // Initialize WebSocket
    initWebSocket();

    // Load initial data
    loadInitialData();

    // Start streaming after initial load
    setTimeout(() => startStreaming(), 2000);

    // Setup event listeners
    document.getElementById('refresh-btn').addEventListener('click', refreshData);
    document.getElementById('toggle-predictions').addEventListener('click', togglePredictions);
    document.getElementById('toggle-confidence').addEventListener('click', toggleConfidenceBands);
    document.getElementById('toggle-indicators').addEventListener('click', toggleIndicators);
    document.getElementById('toggle-sma').addEventListener('click', toggleSMAs);
    document.getElementById('timeframe-select').addEventListener('change', handleTimeframeChange);
    document.getElementById('ticker-select').addEventListener('change', handleTickerChange);

    // Handle window resize with debounce for performance
    let resizeTimeout;
    window.addEventListener('resize', () => {
        clearTimeout(resizeTimeout);
        resizeTimeout = setTimeout(resizeChart, 100);
    });

    // Initialize timezone display
    const timezone = Intl.DateTimeFormat().resolvedOptions().timeZone;
    const timezoneShort = new Date().toLocaleDateString('en', {timeZoneName:'short'}).split(', ')[1];
    document.getElementById('timezone-info').textContent = `Times shown in ${timezoneShort}`;

    // Initial resize
    resizeChart();

    // Initialize analysis panel
    initAnalysisPanel();

    // Initialize chat
    initChat();
});
