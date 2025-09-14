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
    
    // Fit content
    chart.timeScale().fitContent();
}

// Convert data for chart
function convertToChartData(data, baseDate = null, timeOffset = 0) {
    if (!data || !Array.isArray(data)) return [];
    
    const startDate = baseDate || new Date();
    
    return data.map((value, index) => {
        const date = new Date(startDate);
        date.setMinutes(date.getMinutes() + (index + timeOffset));
        const unixTime = Math.floor(date.getTime() / 1000);
        
        return {
            time: unixTime,
            value: typeof value === 'number' ? value : parseFloat(value)
        };
    });
}

// Convert candlestick data
function convertCandlestickData(data) {
    if (!data || !Array.isArray(data)) return [];
    
    return data.map(candle => {
        // Convert timestamp to Unix timestamp (seconds since epoch)
        // The timestamp from server is already in the correct timezone after RTH filtering
        const date = new Date(candle.timestamp);
        const unixTime = Math.floor(date.getTime() / 1000);
        
        return {
            time: unixTime,
            open: parseFloat(candle.open),
            high: parseFloat(candle.high),
            low: parseFloat(candle.low),
            close: parseFloat(candle.close)
        };
    });
}

// Update chart with new data
function updateChart(data) {
    if (!data) return;
    
    console.log('Updating chart with data:', data);
    
    // Update historical candlestick data
    if (data.historical && data.historical.length > 0) {
        const candleData = convertCandlestickData(data.historical);
        console.log('Setting candlestick data:', candleData.slice(0, 5)); // Log first 5 for debugging
        candlestickSeries.setData(candleData);
    }
    
    // Update prediction data - note: the structure is data.prediction, not data.prediction.summary
    if (data.prediction) {
        const summary = data.prediction; // The prediction IS the summary
        const lastCandle = data.historical ? data.historical[data.historical.length - 1] : null;
        const baseDate = lastCandle ? new Date(lastCandle.timestamp) : new Date();
        const timeOffset = 1; // Start predictions 1 minute after last candle
        
        // Update mean prediction line
        if (summary.mean_path) {
            const predictionData = convertToChartData(summary.mean_path, baseDate, timeOffset);
            console.log('Setting prediction data:', predictionData.slice(0, 5));
            predictionLineSeries.setData(predictionData);
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
        }
        
        // Update indicators (draw as horizontal lines across the prediction period)
        if (summary.current_vwap && baseDate) {
            const vwapData = [];
            for (let i = 0; i <= 30; i++) {
                const date = new Date(baseDate);
                date.setMinutes(date.getMinutes() + i);
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
                date.setMinutes(date.getMinutes() + i);
                const unixTime = Math.floor(date.getTime() / 1000);
                
                bbUpperData.push({ time: unixTime, value: bb.upper });
                bbMiddleData.push({ time: unixTime, value: bb.middle });
                bbLowerData.push({ time: unixTime, value: bb.lower });
            }
            
            indicatorSeries.bbUpper.setData(bbUpperData);
            indicatorSeries.bbMiddle.setData(bbMiddleData);
            indicatorSeries.bbLower.setData(bbLowerData);
        }
        
        // Update statistics panel
        updateStatsPanel(summary);
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
    
    // Update RTH status and data info
    if (summary.rth_only !== undefined) {
        document.getElementById('rth-status').textContent = 
            summary.rth_only ? 'RTH Only (9:30-4:00 ET)' : 'All Hours (24/7)';
        document.getElementById('rth-status').className = 
            summary.rth_only ? 'value positive' : 'value';
    }
    
    if (summary.data_bars_count !== undefined) {
        document.getElementById('data-bars').textContent = summary.data_bars_count;
    }
    
    if (summary.n_samples !== undefined) {
        document.getElementById('n-samples').textContent = summary.n_samples;
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
    Object.values(indicatorSeries).forEach(series => {
        series.applyOptions({ visible: showIndicators });
    });
}

// Initialize WebSocket connection
function initWebSocket() {
    socket = io();
    
    socket.on('connect', () => {
        console.log('Connected to prediction server');
        document.getElementById('connection-status').textContent = '● Connected';
        document.getElementById('connection-status').className = 'status connected';
    });
    
    socket.on('disconnect', () => {
        console.log('Disconnected from prediction server');
        document.getElementById('connection-status').textContent = '● Disconnected';
        document.getElementById('connection-status').className = 'status disconnected';
    });
    
    socket.on('prediction_update', (data) => {
        console.log('Received prediction update', data);
        updateChart(data);
    });
    
    socket.on('error', (error) => {
        console.error('WebSocket error:', error);
    });
}

// Load initial data
async function loadInitialData() {
    try {
        const response = await fetch('/api/initial_data');
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
        
        // Also emit via WebSocket to update other connected clients
        if (socket && socket.connected) {
            socket.emit('request_update');
        }
    } catch (error) {
        console.error('Failed to refresh data:', error);
    } finally {
        btn.disabled = false;
        btn.textContent = '🔄 Refresh';
    }
}

// Handle timeframe changes
function handleTimeframeChange() {
    const timeframe = document.getElementById('timeframe-select').value;
    console.log('Timeframe changed to:', timeframe);
    
    // Emit settings change to server
    if (socket && socket.connected) {
        socket.emit('settings_changed', {
            timeframe: timeframe,
            type: 'timeframe'
        });
    }
}

// Handle ticker changes
function handleTickerChange() {
    const ticker = document.getElementById('ticker-select').value;
    console.log('Ticker changed to:', ticker);
    
    // Update the header title
    const headerTitle = document.querySelector('.header h1');
    headerTitle.textContent = `🚀 Kronos ${ticker} Prediction`;
    
    // Emit settings change to server
    if (socket && socket.connected) {
        socket.emit('settings_changed', {
            ticker: ticker,
            type: 'ticker'
        });
    }
}

// Handle periodic data checks (can be called by timer if needed)
function checkForNewData() {
    if (socket && socket.connected) {
        socket.emit('check_for_new_data');
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
    
    // Setup event listeners
    document.getElementById('refresh-btn').addEventListener('click', refreshData);
    document.getElementById('toggle-predictions').addEventListener('click', togglePredictions);
    document.getElementById('toggle-confidence').addEventListener('click', toggleConfidenceBands);
    document.getElementById('toggle-indicators').addEventListener('click', toggleIndicators);
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