# Live Chart Prediction System - Technical Documentation

## Table of Contents
- [System Overview](#system-overview)
- [Architecture](#architecture)
- [Data Pipeline](#data-pipeline)
- [Configuration](#configuration)
- [API Reference](#api-reference)
- [Frontend Integration](#frontend-integration)
- [Model Integration](#model-integration)
- [Technical Indicators](#technical-indicators)
- [Real-time Updates](#real-time-updates)
- [Troubleshooting](#troubleshooting)

---

## System Overview

### Purpose
Real-time financial prediction visualization system using the Kronos model with TradingView Lightweight Charts. Provides live market data streaming, AI-powered predictions, and interactive charting for stocks and cryptocurrencies.

### Key Features
- ✅ **Multi-Asset Support**: Stocks (QQQ, SPY, AAPL, TSLA, MSFT, NVDA, META, BABA) and Cryptocurrencies (BTC/USD, ETH/USD, LTC/USD, DOGE/USD)
- ✅ **Dynamic Historical Data**: Automatically fetches 300-400 bars based on timeframe selection
- ✅ **Real-time Streaming**: Live WebSocket data from Alpaca Markets
- ✅ **Interactive Charts**: TradingView Lightweight Charts with candlesticks and prediction overlays
- ✅ **Technical Indicators**: Simple Moving Averages (5, 21, 233), VWAP, Bollinger Bands
- ✅ **OHLC Price Overlay**: Interactive price display with crosshair tracking
- ✅ **Confidence Bands**: Prediction uncertainty visualization (10th, 25th, 75th, 90th percentiles)
- ✅ **Multi-Timeframe**: 1min, 5min, 15min, 30min chart intervals

### Technology Stack
- **Backend**: FastAPI (Python 3.8+)
- **Frontend**: Vanilla JavaScript with TradingView Lightweight Charts
- **ML Model**: Kronos (HuggingFace Transformers)
- **Data Provider**: Alpaca Markets API
- **WebSocket**: FastAPI native WebSockets + Alpaca streaming

---

## Architecture

### System Components

```
┌─────────────────────────────────────────────────────────────────┐
│                         User Browser                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │  index.html  │  │   chart.js   │  │   style.css  │         │
│  └──────┬───────┘  └──────┬───────┘  └──────────────┘         │
│         │                  │                                     │
│         └──────────────────┴─────────────────┐                  │
└────────────────────────────────────────────┬─┴──────────────────┘
                                             │
                                    WebSocket + HTTP
                                             │
┌────────────────────────────────────────────┴───────────────────┐
│                    FastAPI Server (main.py)                     │
│  ┌──────────────────┐  ┌──────────────────┐                   │
│  │  REST Endpoints  │  │  WebSocket /ws   │                   │
│  └────────┬─────────┘  └────────┬─────────┘                   │
│           │                      │                              │
│  ┌────────┴──────────────────────┴─────────┐                  │
│  │      Connection Manager                  │                  │
│  └────────┬──────────────────────┬──────────┘                  │
└───────────┼──────────────────────┼─────────────────────────────┘
            │                      │
    ┌───────┴────────┐    ┌───────┴────────────┐
    │ Prediction     │    │ WebSocket          │
    │ Service        │    │ Manager            │
    └───────┬────────┘    └───────┬────────────┘
            │                      │
    ┌───────┴────────┐    ┌───────┴────────────┐
    │ Kronos Model   │    │ Alpaca Streaming   │
    │ (GPU/CPU)      │    │ (Stock/Crypto)     │
    └────────────────┘    └────────────────────┘
            │                      │
    ┌───────┴────────┐    ┌───────┴────────────┐
    │ HuggingFace    │    │ Alpaca Markets     │
    │ Model Hub      │    │ API                │
    └────────────────┘    └────────────────────┘
```

### File Structure
```
live_chart_prediction/
├── main.py                      # FastAPI server & WebSocket management
├── prediction_service.py        # Kronos model integration & predictions
├── websocket_manager.py         # Real-time data streaming from Alpaca
├── config.yaml                  # Parent directory configuration
├── requirements.txt             # Python dependencies
├── README.md                    # User documentation
├── CLAUDE.md                    # This technical documentation
├── templates/
│   └── index.html              # Main chart interface
├── static/
│   ├── js/
│   │   └── chart.js            # Chart logic & WebSocket client
│   └── css/
│       └── style.css           # Dark theme styling
└── models/                      # Local model checkpoints (optional)
```

---

## Data Pipeline

### 1. Historical Data Acquisition

#### Dynamic Bar Calculation
The system automatically calculates how many days of data to fetch based on the selected timeframe to maintain 300-400 bars:

```python
def _calculate_days_to_fetch(timeframe_minutes, target_bars=350):
    """
    Formula: days = (target_bars / bars_per_day) * 1.5 + 1

    Examples:
    - 1min:  350 bars / 390 bars/day = 0.9 days → 2 days (780 bars)
    - 5min:  350 bars / 78 bars/day  = 4.5 days → 5 days (390 bars)
    - 15min: 350 bars / 26 bars/day  = 13.5 days → 14 days (364 bars)
    - 30min: 350 bars / 13 bars/day  = 27 days → 28 days (364 bars)
    """
    rth_minutes_per_day = 390  # 6.5 hours of RTH
    bars_per_day = rth_minutes_per_day / timeframe_minutes
    days_needed = target_bars / bars_per_day
    days_with_buffer = int(days_needed * 1.5) + 1
    return min(days_with_buffer, 60)  # Cap at 60 days
```

#### RTH Filtering (Stocks Only)
For stocks, the system filters to Regular Trading Hours (9:30 AM - 4:00 PM ET):

```python
# Location: prediction_service.py:202-215
if self.rth_only:
    df_et = df.copy()
    df_et['timestamp'] = df_et['timestamp'].dt.tz_convert(self.timezone)
    df_et = df_et.set_index('timestamp')
    df_rth = df_et.between_time('09:30', '15:59')
    df = df_rth.reset_index()
```

**Data Reduction Examples:**
- **QQQ (1min)**: 1,727 total bars → 780 RTH bars (removed 947 pre/post-market)
- **SPY (5min)**: ~345 total bars → ~156 RTH bars
- **Crypto**: No filtering (trades 24/7)

### 2. Data Processing Flow

```
┌─────────────────────────────────────────────────────────────────┐
│ Step 1: Fetch Raw Data                                          │
│  • Alpaca API call (StockBarsRequest/CryptoBarsRequest)        │
│  • Time range: end_time - calculated_days to end_time          │
│  • Timeframe: 1min/5min/15min/30min                            │
└────────────────────────┬────────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────────┐
│ Step 2: Timezone Conversion                                     │
│  • Convert to US/Eastern timezone                               │
│  • Ensure timezone-aware datetime objects                       │
└────────────────────────┬────────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────────┐
│ Step 3: RTH Filtering (Stocks Only)                            │
│  • Filter between 09:30 - 15:59 ET                             │
│  • Skip for crypto (24/7 trading)                              │
└────────────────────────┬────────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────────┐
│ Step 4: Data Validation                                         │
│  • Check for empty DataFrame                                    │
│  • Validate OHLCV integrity                                     │
│  • Log bar counts                                               │
└────────────────────────┬────────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────────┐
│ Step 5: Technical Indicator Calculation                         │
│  • SMA 5, 21, 233 (entire series)                              │
│  • VWAP (last 20 bars)                                          │
│  • Bollinger Bands (20-period, ±2σ)                            │
└────────────────────────┬────────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────────┐
│ Step 6: Format for Client                                       │
│  • Convert to JSON-serializable format                          │
│  • Include timestamp, OHLCV, indicators                         │
│  • Return to FastAPI endpoint                                   │
└─────────────────────────────────────────────────────────────────┘
```

### 3. Prediction Generation Pipeline

```python
# Location: prediction_service.py:296-412

# Input: Historical bars (last 480 for context)
context_length = 480  # 8 hours of 1-min bars
context_data = self.latest_historical[-context_length:]

# Prepare data for Kronos model
context_df = pd.DataFrame(context_data)
context_df['amount'] = context_df['volume'] * context_df['close']

# Generate timestamps
x_timestamp = pd.Series([pd.Timestamp(bar['timestamp']) for bar in context_data])
last_timestamp = x_timestamp.iloc[-1]
y_timestamp = pd.Series(pd.date_range(
    start=last_timestamp + pd.Timedelta(minutes=1),
    periods=30,  # horizon
    freq='1min'
))

# Kronos prediction (Monte Carlo sampling)
pred_df = self.predictor.predict(
    df=context_df,
    x_timestamp=x_timestamp,
    y_timestamp=y_timestamp,
    pred_len=30,
    sample_count=5,
    T=1.0,        # temperature
    top_p=0.9     # nucleus sampling
)

# Statistical analysis
predictions = np.array(all_predictions)
percentiles = {
    'p10': np.percentile(predictions, 10, axis=0),
    'p25': np.percentile(predictions, 25, axis=0),
    'p50': np.percentile(predictions, 50, axis=0),
    'p75': np.percentile(predictions, 75, axis=0),
    'p90': np.percentile(predictions, 90, axis=0)
}

# Probability calculations
final_prices = predictions[:, -1]
p_up = np.mean(final_prices > current_price)
exp_return = (np.mean(final_prices) - current_price) / current_price
```

### 4. Real-time Data Streaming

```python
# Location: websocket_manager.py

# Stock streaming (IEX or SIP feed based on config)
async def _subscribe_stock_stream(symbol):
    stream = StockDataStream(api_key, secret_key)

    async def bar_handler(bar):
        # Process incoming bar data
        bar_data = {
            'timestamp': bar.timestamp,
            'open': bar.open,
            'high': bar.high,
            'low': bar.low,
            'close': bar.close,
            'volume': bar.volume
        }
        # Broadcast to all connected clients
        await fastapi_callback({'type': 'bar_update', 'data': bar_data})

    stream.subscribe_bars(bar_handler, symbol)
    await stream._run_forever()

# Crypto streaming (24/7 feed)
async def _subscribe_crypto_stream(symbol):
    stream = CryptoDataStream(api_key, secret_key)
    # Similar to stock but no RTH filtering
```

---

## Configuration

### config.yaml Structure

```yaml
# Parent directory: ../config.yaml

# Trading Symbol Configuration
symbol: "QQQ"  # Default symbol on startup

# Data Parameters
data:
  lookback_bars: 480              # Context window for model (8 hours)
  horizon: 30                     # Prediction horizon (30 minutes)
  target_historical_bars: 350     # Target bars to fetch (auto-calculates days)
  days_to_fetch: 10               # Fallback for manual override
  timeframe: "1Min"               # Bar interval
  rth_only: true                  # Regular Trading Hours filtering
  timezone: "UTC"                 # Data processing timezone

# Model Configuration
model:
  checkpoint: "NeoQuasar/Kronos-base"      # HuggingFace model ID or local path
  tokenizer: "NeoQuasar/Kronos-Tokenizer-base"
  device: "cuda:0"                         # GPU device or "cpu"
  max_context: 512                         # Maximum context length

# Sampling Parameters
sampling:
  n_samples: 100                  # Monte Carlo samples for uncertainty
  temperature: 1.0                # Sampling temperature (diversity)
  top_p: 0.9                      # Nucleus sampling threshold

# Mean Reversion Parameters
mean_reversion:
  calculate_vwap: true            # Enable VWAP calculation
  calculate_bollinger: true       # Enable Bollinger Bands
  sma_period: 20                  # SMA period for Bollinger
  bb_std: 2                       # Standard deviations for bands

# Alpaca API Configuration
ALPACA_KEY_ID: "YOUR_KEY"
ALPACA_SECRET_KEY: "YOUR_SECRET"

alpaca:
  paper_account: false            # Use paper trading account
  paper_trading: false            # Paper trading mode
  api_base_url: "https://api.alpaca.markets"
  data_api_url: "https://data.alpaca.markets"

# Logging
logging:
  level: "INFO"                   # DEBUG, INFO, WARNING, ERROR
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
```

### Server Configuration

```python
# Location: main.py:466-473

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting FastAPI server with uvicorn...")
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=5000,              # Default port (changed from 5000 to 5001 if conflicts)
        reload=False,           # Auto-reload on code changes (dev only)
        log_config=LOG_CONFIG,
        log_level="info",
    )
```

---

## API Reference

### REST Endpoints

#### GET `/`
Returns the main chart interface (HTML).

**Response**: HTML page with TradingView chart

---

#### GET `/api/initial_data`
Fetches initial historical data and latest prediction for chart initialization.

**Response:**
```json
{
  "historical": [
    {
      "timestamp": "2025-10-03T09:30:00-04:00",
      "open": 600.50,
      "high": 601.20,
      "low": 600.10,
      "close": 600.80,
      "volume": 1000000
    }
  ],
  "prediction": {
    "current_close": 600.80,
    "mean_path": [600.85, 600.90, 601.00],
    "percentiles": {
      "p10": [600.70, 600.75, 600.80],
      "p25": [600.75, 600.80, 600.85],
      "p50": [600.85, 600.90, 601.00],
      "p75": [600.95, 601.00, 601.15],
      "p90": [601.00, 601.10, 601.25]
    },
    "p_up_30m": 0.65,
    "exp_ret_30m": 0.00125,
    "current_vwap": 600.75,
    "bollinger_bands": {
      "upper": 602.50,
      "middle": 600.80,
      "lower": 599.10
    },
    "sma_5": 600.85,
    "sma_21": 601.20,
    "sma_233": 595.40,
    "sma_5_series": [null, null, null, null, 598.2, 598.5, ...],
    "sma_21_series": [...],
    "sma_233_series": [...],
    "n_samples": 100,
    "model_name": "Kronos-base",
    "symbol": "QQQ",
    "display_name": "QQQ",
    "asset_type": "stock",
    "rth_only": true,
    "data_bars_count": 780,
    "timestamp": "2025-10-03T21:51:38-04:00"
  },
  "timestamp": "2025-10-03T21:51:38-04:00"
}
```

---

#### GET `/api/latest_prediction`
Gets the current cached prediction without re-generating.

**Response:** Same as `prediction` object above

---

#### GET `/api/generate_prediction`
Forces generation of a new prediction with fresh data.

**Process:**
1. Fetches latest historical data (dynamic bars based on timeframe)
2. Generates new prediction using Kronos model
3. Calculates all technical indicators
4. Broadcasts to all connected WebSocket clients
5. Returns new prediction and historical data

**Response:** Same structure as `/api/initial_data`

---

#### POST `/api/start_stream`
Starts real-time WebSocket streaming for a symbol.

**Request Body:**
```json
{
  "symbol": "BTC/USD",
  "timeframe": "1Min"
}
```

**Response:**
```json
{
  "status": "started",
  "symbol": "BTC/USD"
}
```

**Behavior:**
- Stops existing stream first
- Subscribes to new symbol
- Starts background WebSocket connection to Alpaca
- Broadcasts `stream_started` message to clients

---

#### POST `/api/stop_stream`
Stops the current WebSocket stream.

**Response:**
```json
{
  "status": "stopped"
}
```

---

### WebSocket Protocol

#### Connection: `ws://localhost:5000/ws`

#### Client → Server Messages

**1. Request Manual Update**
```json
{
  "type": "request_update"
}
```
Triggers a new prediction generation.

---

**2. Settings Changed**
```json
{
  "type": "settings_changed",
  "ticker": "SPY",
  "timeframe": "5",
  "changeType": "ticker"  // or "timeframe"
}
```
Updates symbol/timeframe and regenerates prediction with new settings.

---

**3. Check for New Data**
```json
{
  "type": "check_for_new_data"
}
```
Checks if new bar data is available and updates prediction if needed.

---

#### Server → Client Messages

**1. Connected**
```json
{
  "type": "connected",
  "message": "Connected to prediction server"
}
```

---

**2. Prediction Update**
```json
{
  "type": "prediction_update",
  "prediction": { /* prediction object */ },
  "historical": [ /* historical bars */ ],
  "timestamp": "2025-10-03T21:51:38-04:00"
}
```

---

**3. Stream Started**
```json
{
  "type": "stream_started",
  "symbol": "BTC/USD",
  "timeframe": "1Min"
}
```

---

**4. Bar Update (Real-time)**
```json
{
  "type": "bar_update",
  "data": {
    "timestamp": "2025-10-03T14:30:00-04:00",
    "open": 600.50,
    "high": 601.20,
    "low": 600.10,
    "close": 600.80,
    "volume": 1000000
  }
}
```

---

**5. Stream Error**
```json
{
  "type": "stream_error",
  "message": "Crypto streaming unavailable",
  "fallback": true,
  "suggestion": "Using historical data mode"
}
```

---

## Frontend Integration

### Chart Initialization

```javascript
// Location: static/js/chart.js:74-93

const chart = LightweightCharts.createChart(chartContainer, {
    width: 800,
    height: 500,
    layout: {
        backgroundColor: '#1e222d',
        textColor: '#d1d4dc',
    },
    grid: {
        vertLines: { color: '#2B2B43' },
        horzLines: { color: '#2B2B43' }
    },
    crosshair: {
        mode: LightweightCharts.CrosshairMode.Normal,
    },
    timeScale: {
        timeVisible: true,
        secondsVisible: false
    }
});

// Create series
candlestickSeries = chart.addCandlestickSeries({
    upColor: '#26a69a',
    downColor: '#ef5350',
    wickUpColor: '#26a69a',
    wickDownColor: '#ef5350',
});

predictionLineSeries = chart.addLineSeries({
    color: '#2962FF',
    lineWidth: 3,
    title: 'Prediction'
});
```

### SMA Series Configuration

```javascript
// Location: static/js/chart.js:157-173

indicatorSeries.sma5 = chart.addLineSeries({
    color: '#FF8C00',    // Orange
    lineWidth: 2,
    title: 'SMA 5',
});

indicatorSeries.sma21 = chart.addLineSeries({
    color: '#FF0000',    // Red
    lineWidth: 2,
    title: 'SMA 21',
});

indicatorSeries.sma233 = chart.addLineSeries({
    color: '#808080',    // Grey
    lineWidth: 3,
    title: 'SMA 233',
});
```

### OHLC Price Overlay with Crosshair

```javascript
// Location: static/js/chart.js:600-620

chart.subscribeCrosshairMove((param) => {
    if (!param || !param.time) {
        return;
    }

    const candleData = param.seriesData?.get(candlestickSeries);
    if (candleData) {
        updatePriceOverlay({
            open: candleData.open,
            high: candleData.high,
            low: candleData.low,
            close: candleData.close
        });
    }
});

function updatePriceOverlay(prices) {
    document.getElementById('overlay-open').textContent = prices.open.toFixed(2);
    document.getElementById('overlay-high').textContent = prices.high.toFixed(2);
    document.getElementById('overlay-low').textContent = prices.low.toFixed(2);
    document.getElementById('overlay-close').textContent = prices.close.toFixed(2);
}
```

### WebSocket Client

```javascript
// Location: static/js/chart.js:850-900

function initWebSocket() {
    const wsUrl = `ws://${window.location.host}/ws`;
    socket = new WebSocket(wsUrl);

    socket.onopen = () => {
        console.log('WebSocket connected');
    };

    socket.onmessage = (event) => {
        const message = JSON.parse(event.data);

        switch(message.type) {
            case 'connected':
                console.log('Connected to server:', message.message);
                break;

            case 'prediction_update':
                handlePredictionUpdate(message);
                break;

            case 'bar_update':
                handleBarUpdate(message.data);
                break;

            case 'stream_started':
                console.log('Stream started:', message.symbol);
                break;

            case 'stream_error':
                console.error('Stream error:', message.message);
                break;
        }
    };

    socket.onerror = (error) => {
        console.error('WebSocket error:', error);
    };

    socket.onclose = () => {
        console.log('WebSocket disconnected');
        // Auto-reconnect after 3 seconds
        setTimeout(initWebSocket, 3000);
    };
}
```

---

## Model Integration

### Kronos Model Loading

```python
# Location: prediction_service.py:105-125

def _init_model(self):
    """Initialize Kronos model and tokenizer"""
    try:
        # Load tokenizer from HuggingFace or local path
        self.tokenizer = KronosTokenizer.from_pretrained(
            self.config['model']['tokenizer']
        )

        # Load model from HuggingFace or local path
        self.model = Kronos.from_pretrained(
            self.config['model']['checkpoint']
        )

        # Create predictor with device and context settings
        self.predictor = KronosPredictor(
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.config['model']['device'],  # "cuda:0" or "cpu"
            max_context=self.config['model']['max_context']  # 512
        )

        logger.info(f"Kronos model loaded: {self.config['model']['checkpoint']}")
    except Exception as e:
        logger.error(f"Failed to initialize model: {e}")
        raise
```

### Model Prediction Flow

```python
# Location: prediction_service.py:244-296

# 1. Prepare context data (last 480 bars)
context_length = 480
context_data = self.latest_historical[-context_length:]

# 2. Create DataFrame with OHLCV + amount
data_for_kronos = []
for bar in context_data:
    data_for_kronos.append({
        'open': bar['open'],
        'high': bar['high'],
        'low': bar['low'],
        'close': bar['close'],
        'volume': bar['volume']
    })

context_df = pd.DataFrame(data_for_kronos)
context_df['amount'] = context_df['volume'] * context_df['close']

# 3. Create timestamps
x_timestamp = pd.Series([pd.Timestamp(bar['timestamp']) for bar in context_data])

# 4. Generate future timestamps (horizon = 30 minutes)
last_timestamp = x_timestamp.iloc[-1]
y_timestamp = pd.Series(pd.date_range(
    start=last_timestamp + pd.Timedelta(minutes=1),
    periods=30,
    freq='1min'
))

# 5. Call Kronos predictor
pred_df = self.predictor.predict(
    df=context_df,
    x_timestamp=x_timestamp,
    y_timestamp=y_timestamp,
    pred_len=30,
    sample_count=5,     # Number of samples for uncertainty
    T=1.0,              # Temperature
    top_p=0.9           # Nucleus sampling
)

# 6. Extract predictions and add noise for Monte Carlo simulation
predictions = pred_df['close'].values
mean_pred = predictions
std_dev = np.std(mean_pred) * 0.1

all_predictions = []
for _ in range(n_samples):
    noise = np.random.normal(0, std_dev, len(mean_pred))
    sample = mean_pred + noise
    all_predictions.append(sample)

predictions = np.array(all_predictions)
```

### Using Local Finetuned Models

To use a local finetuned model instead of HuggingFace:

```yaml
# config.yaml
model:
  checkpoint: "./live_chart_prediction/models/qqq_finetuned"
  tokenizer: "./live_chart_prediction/models/qqq_finetuned/tokenizer"
  device: "cuda:0"
  max_context: 512
```

The system will automatically detect local paths and load from disk instead of downloading from HuggingFace.

---

## Technical Indicators

### Simple Moving Averages (SMA)

#### Current Value Calculation
```python
# Location: prediction_service.py:441-448

def _calculate_sma(prices, period):
    """Calculate Simple Moving Average for current price"""
    if len(prices) < period:
        return None
    return np.mean(prices[-period:])

# Usage
sma_5 = self._calculate_sma(context_df['close'].values, 5)
sma_21 = self._calculate_sma(context_df['close'].values, 21)
sma_233 = self._calculate_sma(context_df['close'].values, 233)
```

#### Series Calculation (for Chart)
```python
# Location: prediction_service.py:450-468

def _calculate_sma_series(df, period):
    """Calculate SMA series for entire dataset"""
    closes = df['close'].values
    sma_values = []

    for i in range(len(closes)):
        if i < period - 1:
            # Not enough data points
            sma_values.append(None)
        else:
            # Calculate SMA for this point
            sma = np.mean(closes[i-period+1:i+1])
            sma_values.append(float(sma))

    return sma_values

# Usage (full historical data)
full_df = pd.DataFrame(self.latest_historical)
sma_5_series = self._calculate_sma_series(full_df, 5)
sma_21_series = self._calculate_sma_series(full_df, 21)
sma_233_series = self._calculate_sma_series(full_df, 233)
```

**Color Coding:**
- SMA 5: Orange (#FF8C00) - Short-term trend
- SMA 21: Red (#FF0000) - Medium-term trend
- SMA 233: Grey (#808080) - Long-term trend

### Volume-Weighted Average Price (VWAP)

```python
# Location: prediction_service.py:413-424

def _calculate_vwap(df):
    """Calculate VWAP from last 20 bars"""
    recent_df = df.tail(20)
    typical_price = (recent_df['high'] + recent_df['low'] + recent_df['close']) / 3
    total_value = (typical_price * recent_df['volume']).sum()
    total_volume = recent_df['volume'].sum()

    return total_value / total_volume if total_volume > 0 else df['close'].iloc[-1]
```

### Bollinger Bands

```python
# Location: prediction_service.py:426-439

def _calculate_bollinger(prices, period=20, std_dev=2):
    """Calculate Bollinger Bands"""
    recent_prices = prices[-period:]
    middle = np.mean(recent_prices)
    std = np.std(recent_prices)

    return {
        'upper': middle + (std_dev * std),
        'middle': middle,
        'lower': middle - (std_dev * std)
    }
```

**Formula:**
- Middle Band = 20-period SMA
- Upper Band = Middle + (2 × Standard Deviation)
- Lower Band = Middle - (2 × Standard Deviation)

---

## Real-time Updates

### Bar Update Flow

```
┌──────────────────┐
│  Alpaca Market   │
│  Data Stream     │
└────────┬─────────┘
         │
         │ New Bar Data
         ▼
┌──────────────────┐
│ WebSocket        │
│ Manager          │
│ (Background      │
│  Thread)         │
└────────┬─────────┘
         │
         │ Process & Format
         ▼
┌──────────────────┐
│ FastAPI Event    │
│ Loop             │
│ (async callback) │
└────────┬─────────┘
         │
         │ Broadcast
         ▼
┌──────────────────┐
│ All Connected    │
│ WebSocket        │
│ Clients          │
└────────┬─────────┘
         │
         │ Update Chart
         ▼
┌──────────────────┐
│ TradingView      │
│ Chart Update     │
│ (Candlestick +   │
│  Indicators)     │
└──────────────────┘
```

### Prediction Update Triggers

1. **Manual Refresh**: User clicks "Refresh" button
   ```javascript
   socket.send(JSON.stringify({type: 'request_update'}));
   ```

2. **Ticker Change**: User selects different symbol
   ```javascript
   socket.send(JSON.stringify({
       type: 'settings_changed',
       ticker: newTicker,
       changeType: 'ticker'
   }));
   ```

3. **Timeframe Change**: User selects different interval
   ```javascript
   socket.send(JSON.stringify({
       type: 'settings_changed',
       timeframe: newTimeframe,
       changeType: 'timeframe'
   }));
   ```

4. **Periodic Update**: Every 5 completed bars
   ```javascript
   if (barCompletedCount % 5 === 0) {
       socket.send(JSON.stringify({type: 'check_for_new_data'}));
   }
   ```

---

## Troubleshooting

### Common Issues

#### 1. No Predictions Showing
**Symptoms**: Empty prediction statistics, no forecast lines on chart

**Diagnosis Steps:**
```bash
# Check server logs for errors
tail -f logs/server.log | grep -i "error\|prediction"

# Test prediction endpoint directly
curl http://localhost:5000/api/generate_prediction

# Verify model loading
grep "Kronos model loaded" logs/server.log
```

**Solutions:**
- Ensure sufficient historical data (min 480 bars for context)
- Check GPU/CPU availability and memory
- Verify model checkpoint path in config.yaml
- Check for SMA calculation errors in logs

---

#### 2. SMA Not Displaying
**Symptoms**: Chart shows candlesticks but no SMA lines

**Diagnosis:**
```javascript
// Browser console
console.log(indicatorSeries.sma5);
console.log(indicatorSeries.sma21);
console.log(indicatorSeries.sma233);

// Check if series have data
indicatorSeries.sma5.data();
```

**Solutions:**
- Click "Toggle SMAs" button to enable display
- Ensure minimum bars: SMA233 needs 233 bars minimum
- Check backend response includes `sma_*_series` fields
- Verify no JavaScript errors in browser console

---

#### 3. Dynamic Bar Fetching Not Working
**Symptoms**: Always fetching same number of bars regardless of timeframe

**Diagnosis:**
```bash
# Check logs for calculation
grep "Auto-calculated" logs/server.log
grep "Timeframe:" logs/server.log
```

**Expected Log Output:**
```
Timeframe: 1min, Target bars: 350, Bars/day: 390.0, Days needed: 0.9, Days to fetch (with buffer): 2
Auto-calculated 2 days to fetch for 1min timeframe
```

**Solutions:**
- Ensure `target_historical_bars` is set in config.yaml
- Verify `fetch_historical_data()` is called without explicit days parameter
- Check timeframe conversion in `update_settings()` method

---

#### 4. WebSocket Disconnections
**Symptoms**: Real-time updates stop, chart freezes

**Diagnosis:**
```javascript
// Browser console
socket.readyState  // Should be 1 (OPEN)

// Check connection
socket.onclose = (event) => {
    console.log('Close code:', event.code);
    console.log('Close reason:', event.reason);
};
```

**Solutions:**
- Check network connectivity
- Verify firewall allows WebSocket connections
- Ensure server is running (check port 5000/5001)
- Browser auto-reconnects after 3 seconds
- Check server logs for WebSocket errors

---

#### 5. Timeframe Selection Not Updating Chart
**Symptoms**: Changing timeframe doesn't update bars

**Diagnosis:**
```bash
# Check if settings_changed message is received
grep "settings_changed" logs/server.log

# Verify timeframe update
grep "Updated timeframe to" logs/server.log
```

**Solutions:**
- Check WebSocket connection is active
- Verify frontend sends correct message format
- Clear cache and refresh browser
- Restart server to clear stale data

---

### Performance Optimization

#### GPU Memory Issues
```yaml
# Reduce these values if running out of GPU memory
sampling:
  n_samples: 50  # Instead of 100

model:
  max_context: 256  # Instead of 512

# Or switch to CPU
model:
  device: "cpu"
```

#### Slow Predictions
```yaml
# Reduce sample count for faster generation
sampling:
  n_samples: 50

# Use smaller model
model:
  checkpoint: "NeoQuasar/Kronos-small"  # Instead of Kronos-base
```

#### High API Usage
```yaml
# Reduce data fetching
data:
  target_historical_bars: 250  # Instead of 350

# Increase cache time
# Location: prediction_service.py:547
def _is_stale(self, max_age_minutes=10):  # Instead of 5
```

---

### Logging Configuration

```yaml
# config.yaml
logging:
  level: "DEBUG"  # For detailed troubleshooting

# Or set in code
# Location: prediction_service.py:27-32
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
```

**Log Locations:**
- Server logs: Console output
- Browser logs: Developer Console (F12)
- Network logs: Network tab in Dev Tools

---

### Health Checks

```bash
# Check server is running
curl http://localhost:5000/

# Check API endpoints
curl http://localhost:5000/api/initial_data

# Test WebSocket connection
wscat -c ws://localhost:5000/ws

# Monitor resource usage
nvidia-smi  # For GPU
top  # For CPU/Memory
```

---

## Development Workflow

### Making Changes

1. **Backend Changes** (Python files):
   ```bash
   # Stop server (Ctrl+C)
   # Make changes to .py files
   # Restart server
   python main.py
   ```

2. **Frontend Changes** (HTML/CSS/JS):
   - Edit files in `templates/` or `static/`
   - Refresh browser (Ctrl+R or Cmd+R)
   - No server restart needed

3. **Configuration Changes** (config.yaml):
   ```bash
   # Stop server
   # Edit ../config.yaml
   # Restart server
   python main.py
   ```

### Testing

```bash
# Test prediction endpoint
curl -X GET http://localhost:5000/api/generate_prediction | jq .

# Test with different symbols
curl -X POST http://localhost:5000/api/start_stream \
  -H "Content-Type: application/json" \
  -d '{"symbol":"SPY","timeframe":"5Min"}'

# Monitor WebSocket messages
wscat -c ws://localhost:5000/ws
```

---

## Production Deployment

### Using Gunicorn

```bash
# Install production dependencies
pip install gunicorn uvloop

# Run with multiple workers
gunicorn main:app \
  -w 4 \
  -k uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:5000 \
  --access-logfile logs/access.log \
  --error-logfile logs/error.log
```

### Environment Variables

```bash
# Set API keys via environment
export ALPACA_KEY_ID="your_key"
export ALPACA_SECRET_KEY="your_secret"
export KRONOS_MODEL_PATH="/path/to/model"
export CUDA_VISIBLE_DEVICES="0"
```

### Docker Deployment (Future)

```dockerfile
FROM python:3.8-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "main.py"]
```

---

## Version History

### v2.1.0 (Current)
- ✅ Dynamic historical bar fetching (300-400 bars based on timeframe)
- ✅ Simple Moving Averages (SMA 5, 21, 233)
- ✅ OHLC price overlay with crosshair tracking
- ✅ Dynamic model name display
- ✅ Enhanced error handling for SMA calculations
- ✅ Added META and BABA to ticker dropdown

### v2.0.0
- ✅ Flask to FastAPI migration
- ✅ Native WebSocket support
- ✅ Multi-asset support (stocks + crypto)
- ✅ Fixed WebSocket hanging issues
- ✅ Improved timezone handling

### v1.0.0
- Initial Flask-SocketIO implementation
- Basic QQQ predictions
- TradingView chart integration

---

## References

### External Documentation
- [Kronos Model Paper](https://arxiv.org/abs/example)
- [TradingView Lightweight Charts](https://tradingview.github.io/lightweight-charts/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Alpaca Markets API](https://alpaca.markets/docs/)

### Internal Documentation
- [README.md](README.md) - User guide and setup
- [requirements.txt](requirements.txt) - Python dependencies
- [../config.yaml](../config.yaml) - Configuration reference

---

## Contact & Support

For issues and questions:
- Check troubleshooting section above
- Review server logs for error details
- Check browser console for frontend errors
- Verify configuration in config.yaml

**Built with ❤️ using FastAPI, TradingView Charts, and the Kronos Model**

Last updated: 2025-10-03
