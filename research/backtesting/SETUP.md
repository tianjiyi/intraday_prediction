# Backtesting Framework Setup Guide

## Quick Start (3 steps)

### 1. Install Dependencies

```bash
# Make sure you're in the project root
cd C:\Users\skysn\workspace\intraday_predication

# Install backtesting requirements
pip install -r backtesting/requirements.txt
```

### 2. Verify Installation

```bash
# Test that the framework is working
python -c "from backtesting import BacktestEngine; print('✅ Setup complete!')"
```

### 3. Run Your First Backtest

```bash
cd backtesting

# Quick test (last 5 days)
python run_backtest.py --symbol QQQ --quick
```

## Detailed Setup

### Prerequisites

1. **Python 3.8+** installed
2. **Live chart prediction system** already set up
3. **Alpaca API credentials** configured in `../config.yaml`

### Installation Steps

#### Step 1: Install Base Dependencies

```bash
# Ensure live_chart_prediction dependencies are installed
cd live_chart_prediction
pip install -r requirements.txt
cd ..
```

#### Step 2: Install Backtesting Dependencies

```bash
# Install additional backtesting packages
cd backtesting
pip install -r requirements.txt
```

If you encounter issues, install packages individually:

```bash
pip install scikit-learn==1.0.2
pip install matplotlib==3.5.3
pip install seaborn==0.11.2
```

#### Step 3: Verify Alpaca API Access

```bash
# Test that API credentials work
python -c "
import yaml
config = yaml.safe_load(open('../config.yaml'))
print(f'API Key: {config[\"ALPACA_KEY_ID\"][:10]}...')
print('✅ API credentials found')
"
```

#### Step 4: Run Test Backtest

```bash
# Short test to verify everything works
python run_backtest.py \
  --symbol QQQ \
  --timeframe 1Min \
  --start-date 2025-10-01 \
  --end-date 2025-10-03 \
  --no-plots \
  --verbose
```

Expected output:
```
[2025-10-03 14:30:00] INFO - Starting backtest for QQQ (1Min)
[2025-10-03 14:30:05] INFO - Generated 12 prediction points
[2025-10-03 14:30:15] INFO - Backtest completed: 12 predictions generated
```

## Configuration

### Edit config_backtest.yaml

```bash
# Open in your editor
notepad config_backtest.yaml
# or
code config_backtest.yaml
```

Key settings to customize:

```yaml
backtest:
  symbols: ["QQQ", "SPY", "AAPL"]  # Add your symbols
  start_date: "2025-09-01"         # Your test period
  end_date: "2025-10-03"

  rolling_window:
    step_size_minutes: 30          # How often to predict

output:
  base_dir: "./backtesting/results"  # Where to save results
```

## Common Issues

### Issue 1: ImportError

```
ImportError: No module named 'sklearn'
```

**Solution:**
```bash
pip install scikit-learn
```

### Issue 2: API Authentication Failed

```
Error: API authentication failed
```

**Solution:**
Check that API credentials are set in `../config.yaml`:
```yaml
ALPACA_KEY_ID: "YOUR_KEY"
ALPACA_SECRET_KEY: "YOUR_SECRET"
```

### Issue 3: Insufficient Historical Data

```
WARNING: Insufficient data at 2025-10-01 09:30:00: 250 bars
```

**Solution:**
- Reduce `target_historical_bars` in `../config.yaml`
- Use a later start date with more historical data available

### Issue 4: matplotlib Backend Error (Windows)

```
ImportError: cannot import name 'pyplot' from 'matplotlib'
```

**Solution:**
```bash
pip uninstall matplotlib
pip install matplotlib==3.5.3
```

## Usage Examples

### Example 1: Quick 5-Day Test

```bash
python run_backtest.py --symbol QQQ --quick
```

### Example 2: Full Month Backtest

```bash
python run_backtest.py \
  --symbol QQQ \
  --timeframe 1Min \
  --start-date 2025-09-01 \
  --end-date 2025-10-01 \
  --output-dir ./results/september
```

### Example 3: Multiple Symbols

```bash
python run_backtest.py \
  --symbols QQQ SPY AAPL TSLA \
  --timeframe 5Min \
  --start-date 2025-09-15 \
  --end-date 2025-10-01
```

### Example 4: Use Python API

```bash
python example_usage.py
```

### Example 5: Custom Configuration

```bash
# Edit config first
notepad config_backtest.yaml

# Run with config
python run_backtest.py --config config_backtest.yaml
```

## Directory Structure

After setup, your structure should be:

```
intraday_predication/
├── backtesting/                    # ← You are here
│   ├── __init__.py
│   ├── backtest_engine.py
│   ├── metrics.py
│   ├── visualize.py
│   ├── run_backtest.py
│   ├── example_usage.py
│   ├── config_backtest.yaml
│   ├── requirements.txt
│   ├── README.md
│   ├── SETUP.md                   # ← This file
│   └── results/                   # Created on first run
│       └── QQQ_1Min_20251003_143052/
│           ├── predictions.csv
│           ├── metrics.json
│           ├── summary.txt
│           └── *.png              # Visualizations
├── live_chart_prediction/
│   ├── main.py
│   ├── prediction_service.py
│   └── ...
└── config.yaml                    # Shared config
```

## Next Steps

After setup is complete:

1. **Read the README.md** for detailed usage instructions
2. **Run example_usage.py** to see all features
3. **Customize config_backtest.yaml** for your needs
4. **Run a full backtest** on your symbols of interest
5. **Analyze the results** in the generated dashboard

## Performance Tuning

For faster backtesting:

1. **Use longer step sizes:**
   ```yaml
   rolling_window:
     step_size_minutes: 60  # Instead of 30
   ```

2. **Reduce visualization DPI:**
   ```yaml
   visualization:
     dpi: 150  # Instead of 300
   ```

3. **Skip plots for initial runs:**
   ```bash
   python run_backtest.py --symbol QQQ --no-plots
   ```

4. **Use 5Min or 15Min timeframes** instead of 1Min

## Getting Help

If you encounter issues:

1. Check the **Troubleshooting** section in README.md
2. Verify all dependencies: `pip list | grep -E "sklearn|matplotlib|seaborn"`
3. Check logs in `backtesting/results/backtest.log`
4. Run with `--verbose` flag for detailed logging

## Uninstallation

To remove the backtesting framework:

```bash
# Remove installed packages (optional)
pip uninstall scikit-learn matplotlib seaborn

# Remove backtesting directory
cd ..
rmdir /s backtesting  # Windows
# or
rm -rf backtesting    # Linux/Mac
```

---

✅ **Setup complete! You're ready to evaluate your Kronos model predictions.**

For detailed usage instructions, see [README.md](README.md)
