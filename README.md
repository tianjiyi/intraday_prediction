# Kronos QQQ Intraday Prediction System

A probabilistic intraday trading system for QQQ using the Kronos foundation model. This system generates Monte Carlo sampled future paths for the next 30 minutes and computes various probability metrics for trading decisions.

## Features

- **Monte Carlo Forecasting**: Generate N sampled paths (default: 100) for 30-minute horizon
- **Probability Metrics**:
  - P(Close > Current) - Probability of price increase in 30 minutes
  - Expected 30-minute return
  - Percentile bands (10th, 25th, 50th, 75th, 90th)
- **Mean Reversion Analysis**:
  - VWAP reversion probability
  - Bollinger Band middle reversion
  - Maximum drawdown distribution
- **Interactive Visualizations**: Kronos-style charts with confidence bands and predictions
- **Real-time Data**: Integration with Alpaca API for live market data

## System Requirements

- Python 3.10+ (tested with 3.12.6)
- NVIDIA GPU with CUDA support (RTX 5090 compatible)
- CUDA 12.8+ (driver version 13.0 is backward compatible)
- At least 8GB VRAM for Kronos-small model
- Windows/Linux/macOS

## Installation

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd intraday_predication
```

### 2. Set Up Python Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/macOS
source venv/bin/activate
```

### 3. Install PyTorch with CUDA Support

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

### 4. Install Dependencies

The Kronos repository is already cloned. Install all required packages:

```bash
# Install Kronos dependencies
cd Kronos
pip install -r requirements.txt
cd ..

# Install additional packages
pip install alpaca-py plotly kaleido fastapi uvicorn pyyaml
```

### 5. Set Up Alpaca API Credentials

Set your Alpaca API credentials as environment variables:

```bash
# Windows (Command Prompt)
set ALPACA_KEY_ID=your_api_key_id
set ALPACA_SECRET_KEY=your_secret_key

# Windows (PowerShell)
$env:ALPACA_KEY_ID="your_api_key_id"
$env:ALPACA_SECRET_KEY="your_secret_key"

# Linux/macOS
export ALPACA_KEY_ID="your_api_key_id"
export ALPACA_SECRET_KEY="your_secret_key"
```

## Configuration

Edit `config.yaml` to customize settings:

```yaml
symbol: "QQQ"  # Trading symbol

data:
  lookback_bars: 480  # Context window (8 hours)
  horizon: 30  # Prediction horizon (30 minutes)
  days_to_fetch: 3  # Historical data to fetch

model:
  checkpoint: "NeoQuasar/Kronos-small"  # Model size
  device: "cuda:0"  # GPU device

sampling:
  n_samples: 100  # Number of Monte Carlo samples
  temperature: 1.0  # Sampling diversity
  top_p: 0.9  # Nucleus sampling
```

## Usage

### Running Predictions

Execute the main CLI script:

```bash
python cli_kronos_prob_qqq.py
```

This will:
1. Fetch 3 days of 1-minute QQQ data from Alpaca
2. Load the Kronos model
3. Generate 100 Monte Carlo sample paths
4. Calculate probability metrics
5. Save outputs to `./output/` directory

### Output Files

The script generates:
- `pred_summary_QQQ_[timestamp].json` - Summary statistics and probabilities
- `paths_QQQ_[timestamp].csv` - Individual sample paths
- Console output with key metrics

### Visualizing Results

Generate interactive charts:

```bash
# Use latest prediction data
python visualize_predictions.py

# Or specify files
python visualize_predictions.py --summary output/pred_summary_QQQ_*.json --style kronos
```

Chart options:
- `--style kronos` - Full Kronos-style chart with candlesticks and volume
- `--style simple` - Simple line chart with predictions

### Sample Output

```
============================================================
FORECAST SUMMARY
============================================================
Symbol: QQQ
Current Close: $485.32
Samples: 100
Horizon: 30 minutes
----------------------------------------
P(Close > Current in 30min): 52.3%
Expected 30-min Return: 0.15%
P(Touch VWAP in 30min): 34.5%
P(Touch BB Middle in 30min): 28.9%
Max Drawdown P50: -0.45%
============================================================
```

## Model Options

Available Kronos models:

| Model | Parameters | Context | Speed | Accuracy |
|-------|-----------|---------|-------|----------|
| Kronos-mini | 4.1M | 2048 | Fastest | Good |
| Kronos-small | 24.7M | 512 | Fast | Better |
| Kronos-base | 102.3M | 512 | Moderate | Best |

## Troubleshooting

### CUDA Issues

If you encounter CUDA errors:
```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Check CUDA version
nvidia-smi
```

### Alpaca API Issues

- Ensure API keys are set correctly
- Check if using paper trading or live API
- Verify market hours (RTH: 9:30 AM - 4:00 PM ET)

### Memory Issues

If running out of GPU memory:
- Reduce `n_samples` in config.yaml
- Use smaller model (Kronos-mini)
- Set device to "cpu" for CPU inference

## Advanced Usage

### Backtesting

Run rolling walk-forward backtest:
```python
# TODO: Implement backtest harness
python backtest.py --start 2024-01-01 --end 2024-12-31
```

### Real-time Mode

Stream predictions in real-time:
```python
# TODO: Implement streaming mode
python realtime_stream.py --interval 1min
```

### Web Interface

Launch FastAPI web interface:
```bash
# TODO: Implement web UI
uvicorn web_demo:app --reload
```

## Project Structure

```
intraday_predication/
├── Kronos/                  # Kronos model repository
│   ├── model/              # Model implementation
│   ├── examples/           # Example scripts
│   └── requirements.txt    # Kronos dependencies
├── cli_kronos_prob_qqq.py  # Main prediction script
├── visualize_predictions.py # Visualization script
├── config.yaml             # Configuration file
├── output/                 # Output directory
│   ├── pred_summary_*.json
│   ├── paths_*.csv
│   └── chart_*.html
└── README.md              # This file
```

## Notes

- The system uses UTC timezone consistently
- RTH (Regular Trading Hours) filtering is enabled by default
- Model weights are downloaded automatically from Hugging Face on first run
- Predictions are probabilistic - use appropriate risk management

## References

- [Kronos Paper](https://arxiv.org/abs/2508.02739)
- [Kronos GitHub](https://github.com/shiyu-coder/Kronos)
- [Live Demo](https://shiyu-coder.github.io/Kronos-demo/)
- [Alpaca API Documentation](https://alpaca.markets/docs/)

## License

This project uses the Kronos model which is subject to its own license terms. Please refer to the Kronos repository for details.

## Support

For issues or questions:
1. Check the troubleshooting section
2. Review Kronos documentation
3. Open an issue on GitHub