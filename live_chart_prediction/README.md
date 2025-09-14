# QQQ Live Prediction Chart with TradingView Lightweight Charts

Real-time visualization of Kronos model predictions for QQQ using TradingView's free Lightweight Charts library.

## Features

- **Real-time Updates**: WebSocket connection for live prediction updates
- **Interactive Charts**: TradingView Lightweight Charts with candlesticks and prediction overlays
- **Confidence Bands**: Visualizes prediction uncertainty with percentile bands (10th, 25th, 75th, 90th)
- **Technical Indicators**: VWAP and Bollinger Bands overlay
- **Statistics Panel**: Live metrics including probability up/down, expected return, and confidence intervals
- **Responsive Design**: Works on desktop and mobile devices

## Prerequisites

- Python 3.8+
- CUDA-capable GPU (for Kronos model)
- Alpaca API credentials (already in config.yaml)
- Kronos model checkpoint (already in ../Kronos)

## Installation

1. Navigate to the live_chart_prediction directory:
```bash
cd live_chart_prediction
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

## Configuration

The service uses the parent directory's `config.yaml` file. Key settings:
- `symbol`: Trading symbol (default: QQQ)
- `data.lookback_bars`: Context window for predictions (default: 480)
- `data.horizon`: Prediction horizon in minutes (default: 30)
- `sampling.n_samples`: Number of Monte Carlo samples (default: 100)

## Running the Application

1. Start the Flask server:
```bash
python app.py
```

2. Open your browser and navigate to:
```
http://localhost:5000
```

## Architecture

```
live_chart_prediction/
├── app.py                    # Flask server with WebSocket support
├── prediction_service.py     # Kronos model integration
├── templates/
│   └── index.html           # Main chart interface
├── static/
│   ├── js/
│   │   └── chart.js        # Chart logic and WebSocket client
│   └── css/
│       └── style.css       # Dark theme styling
└── requirements.txt         # Python dependencies
```

## How It Works

1. **Initial Load**: 
   - Fetches 3 days of historical QQQ data from Alpaca
   - Generates initial prediction using Kronos model
   - Displays candlestick chart with prediction overlay

2. **Real-time Updates**:
   - WebSocket connection maintains live link to server
   - Server generates new predictions every 30 seconds
   - Chart updates automatically without page refresh

3. **Prediction Generation**:
   - Uses last 480 1-minute bars as context
   - Generates 100 Monte Carlo samples for next 30 minutes
   - Calculates mean path and confidence intervals
   - Computes probability metrics and technical indicators

## API Endpoints

- `GET /` - Main chart interface
- `GET /api/initial_data` - Get historical data and latest prediction
- `GET /api/latest_prediction` - Get current prediction
- `WebSocket /socket.io/` - Real-time prediction updates

## WebSocket Events

- `connect` - Client connected to server
- `disconnect` - Client disconnected
- `prediction_update` - New prediction data available
- `request_update` - Client requests fresh prediction

## Customization

### Update Frequency
Edit `background_updates()` in `app.py`:
```python
time.sleep(30)  # Change to desired seconds
```

### Prediction Parameters
Edit `generate_prediction()` in `prediction_service.py`:
```python
n_samples=100  # Number of Monte Carlo samples
temperature=1.0  # Sampling temperature
```

### Chart Appearance
Modify `chartOptions` in `static/js/chart.js`:
```javascript
const chartOptions = {
    layout: {
        backgroundColor: '#1e222d',
        textColor: '#d1d4dc',
    },
    // ... more options
};
```

## Troubleshooting

### GPU Memory Issues
- Reduce `n_samples` in prediction generation
- Use smaller model checkpoint (Kronos-mini)

### Connection Issues
- Check Flask server is running on port 5000
- Verify WebSocket connection in browser console
- Check firewall settings for port 5000

### No Data Showing
- Verify Alpaca API credentials in config.yaml
- Check if market is open (9:30 AM - 4:00 PM ET)
- Look for errors in Flask server console

## Performance Tips

- Run on GPU for faster predictions
- Cache predictions to reduce model calls
- Use production WSGI server (Gunicorn) for deployment
- Enable CORS only for trusted domains in production

## Future Enhancements

- [ ] Add more technical indicators
- [ ] Support multiple symbols
- [ ] Historical prediction accuracy tracking
- [ ] Alert system for significant predictions
- [ ] Mobile app with push notifications
- [ ] Integration with trading APIs

## License

This project uses TradingView Lightweight Charts (Apache 2.0 License).