/**
 * QQQ Live Prediction Chart with TradingView Lightweight Charts
 */

// Global variables
let chart = null;
let candlestickSeries = null;
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
const chartOptions = {
    width: 800,
    height: 500,
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
function convertToChartData(data, baseDate = null, timeOffset = 0) {
    if (!data || !Array.isArray(data)) return [];
    
    const startDate = baseDate || new Date();
    
    return data.map((value, index) => {
        const date = new Date(startDate);
        // Use currentTimeframe for spacing points
        date.setMinutes(date.getMinutes() + ((index + timeOffset) * currentTimeframe));
        const unixTime = Math.floor(date.getTime() / 1000);
        
        return {
            time: unixTime,
            value: typeof value === 'number' ? value : parseFloat(value)
        };
    });
}

// Convert candlestick data with proper OHLC aggregation for timeframe buckets
function convertCandlestickData(data) {
    if (!data || !Array.isArray(data)) return [];

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
                close: parseFloat(candle.close)
            });
        } else {
            // Aggregate with existing bar in bucket
            const existing = buckets.get(bucketedTime);
            existing.high = Math.max(existing.high, parseFloat(candle.high));
            existing.low = Math.min(existing.low, parseFloat(candle.low));
            existing.close = parseFloat(candle.close); // Last close wins
        }
    });

    // Debug: log conversion result
    console.log(`Conversion result: ${data.length} bars -> ${buckets.size} unique buckets`);

    // Convert to sorted array (TradingView requires ascending time order)
    return Array.from(buckets.values()).sort((a, b) => a.time - b.time);
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
        const candleData = convertCandlestickData(data.historical);

        // Clear and repopulate barBoundaries with historical data to prevent duplicates
        barBoundaries.clear();
        aggregatedBar = null; // Reset aggregation state
        candleData.forEach(candle => {
            barBoundaries.add(candle.time);
        });

        console.log('Setting candlestick data:', candleData.slice(0, 5)); // Log first 5 for debugging
        console.log('Initialized barBoundaries with', barBoundaries.size, 'historical bars');

        candlestickSeries.setData(candleData);
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
        if (summary.current_vwap && baseDate) {
            const vwapData = [];
            for (let i = 0; i <= 30; i++) {
                const date = new Date(baseDate);
                date.setMinutes(date.getMinutes() + (i * currentTimeframe));
                const unixTime = Math.floor(date.getTime() / 1000);
                vwapData.push({
                    time: unixTime,
                    value: summary.current_vwap
                });
            }
            indicatorSeries.vwap.setData(vwapData);
        }
        
        if (summary.bollinger_bands && baseDate) {
            const bb = summary.bollinger_bands;
            const bbUpperData = [];
            const bbMiddleData = [];
            const bbLowerData = [];
            
            for (let i = 0; i <= 30; i++) {
                const date = new Date(baseDate);
                date.setMinutes(date.getMinutes() + (i * currentTimeframe));
                const unixTime = Math.floor(date.getTime() / 1000);
                
                bbUpperData.push({ time: unixTime, value: bb.upper });
                bbMiddleData.push({ time: unixTime, value: bb.middle });
                bbLowerData.push({ time: unixTime, value: bb.lower });
            }
            
            indicatorSeries.bbUpper.setData(bbUpperData);
            indicatorSeries.bbMiddle.setData(bbMiddleData);
            indicatorSeries.bbLower.setData(bbLowerData);
        }

        // Update SMA data
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

            console.log('SMA data set to chart series');
        } else {
            console.log('SMA data not available - clearing series');
            indicatorSeries.sma5.setData([]);
            indicatorSeries.sma21.setData([]);
            indicatorSeries.sma233.setData([]);
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
    probUpElement.textContent = probUp !== undefined ? `${(probUp * 100).toFixed(1)}%` : '--';
    probUpElement.className = probUp >= 0.5 ? 'value positive' : 'value negative';
    
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
}

function toggleConfidenceBands() {
    showConfidence = !showConfidence;
    Object.values(confidenceBandSeries).forEach(series => {
        series.applyOptions({ visible: showConfidence });
    });
}

function toggleIndicators() {
    showIndicators = !showIndicators;
    // Toggle only VWAP and Bollinger Bands, not SMAs
    ['vwap', 'bbUpper', 'bbMiddle', 'bbLower'].forEach(key => {
        if (indicatorSeries[key]) {
            indicatorSeries[key].applyOptions({ visible: showIndicators });
        }
    });
}

function toggleSMAs() {
    showSMAs = !showSMAs;
    // Toggle only SMA series
    ['sma5', 'sma21', 'sma233'].forEach(key => {
        if (indicatorSeries[key]) {
            indicatorSeries[key].applyOptions({ visible: showSMAs });
        }
    });
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

    if (aggregatedBar && aggregatedBar.time === bucketedTime) {
        // Merge current update with existing aggregated data
        displayBar = {
            time: bucketedTime,
            open: aggregatedBar.open,
            high: Math.max(aggregatedBar.high, barData.high),
            low: Math.min(aggregatedBar.low, barData.low),
            close: barData.close
        };
    } else {
        // Start of a new bucket (or no aggregation yet)
        displayBar = {
            time: bucketedTime,
            open: barData.open,
            high: barData.high,
            low: barData.low,
            close: barData.close
        };
    }

    // Update the last bar in the series
    candlestickSeries.update(displayBar);

    // Update price overlay with current bar data
    updatePriceOverlay(barData);

    // Store latest bar data for crosshair fallback
    lastBarData = barData;

    // Store current bar for reference
    currentBar = displayBar;
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

    if (isNewBucket) {
        // New bucket - start new aggregated bar
        aggregatedBar = {
            time: bucketedTime,
            open: barData.open,
            high: barData.high,
            low: barData.low,
            close: barData.close
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
                close: barData.close
            };
        } else {
            // Aggregate data
            aggregatedBar.high = Math.max(aggregatedBar.high, barData.high);
            aggregatedBar.low = Math.min(aggregatedBar.low, barData.low);
            aggregatedBar.close = barData.close;
        }
        
        candlestickSeries.update(aggregatedBar);
        console.log('Updated existing bar at:', new Date(bucketedTime * 1000).toISOString());
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
function resizeChart() {
    if (chart) {
        const container = document.getElementById('chart');
        chart.applyOptions({
            width: container.clientWidth,
            height: container.clientHeight || 500
        });
    }
}

// Initialize everything when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    // Initialize chart
    initChart();
    
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
    
    // Handle window resize
    window.addEventListener('resize', resizeChart);
    
    // Initialize timezone display
    const timezone = Intl.DateTimeFormat().resolvedOptions().timeZone;
    const timezoneShort = new Date().toLocaleDateString('en', {timeZoneName:'short'}).split(', ')[1];
    document.getElementById('timezone-info').textContent = `Times shown in ${timezoneShort}`;
    
    // Initial resize
    resizeChart();
});
