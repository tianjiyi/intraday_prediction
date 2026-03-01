# Kronos Model Backtesting Framework

Comprehensive backtesting system for evaluating Kronos time series predictions with classification and regression metrics.

## Overview

This backtesting framework allows you to:
- ✅ **Evaluate historical predictions** using rolling window backtesting
- ✅ **Calculate comprehensive metrics** (F1, AUC-ROC, MAE, RMSE, Brier Score, etc.)
- ✅ **Visualize results** with ROC curves, confusion matrices, and more
- ✅ **Test multiple symbols** and timeframes
- ✅ **Compare model performance** across different configurations

## Features

### 🎯 Evaluation Metrics

**Classification (Direction Prediction):**
- F1 Score, Precision, Recall, Accuracy
- AUC-ROC curve
- Confusion Matrix
- Specificity, NPV

**Regression (Price Level Prediction):**
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- MAPE (Mean Absolute Percentage Error)
- R² Score
- Residual Analysis

**Probabilistic (Uncertainty Calibration):**
- Brier Score
- Log Loss
- Calibration Curves
- Expected Calibration Error (ECE)

### 📊 Visualizations

- ROC Curves
- Confusion Matrices
- Prediction vs Actual Scatter Plots
- Price Trajectory Comparisons
- Calibration Curves
- Residual Plots
- Comprehensive Dashboard

## Installation

### Prerequisites

```bash
# Ensure you have the live prediction system installed
cd ../live_chart_prediction
pip install -r requirements.txt

# Install additional backtesting dependencies
pip install scikit-learn matplotlib seaborn
```

### Verify Installation

```bash
python -c "from backtesting import BacktestEngine; print('✅ Backtesting framework installed')"
```

## Quick Start

### 1. Run Your First Backtest

```bash
# Test QQQ predictions from last 5 days
python run_backtest.py --symbol QQQ --quick

# Full backtest with date range
python run_backtest.py --symbol QQQ --timeframe 1Min --start-date 2025-09-01 --end-date 2025-10-01
```

### 2. Use Configuration File

```bash
# Edit config_backtest.yaml to set parameters
python run_backtest.py --config config_backtest.yaml
```

### 3. Backtest Multiple Symbols

```bash
python run_backtest.py --symbols QQQ SPY AAPL --timeframe 5Min --start-date 2025-09-01
```

## Usage Guide

### Command Line Interface

```bash
# Basic usage
python run_backtest.py [OPTIONS]

# Options:
  --symbol, -s          Trading symbol (e.g., QQQ, SPY, AAPL)
  --symbols             Multiple symbols to test
  --timeframe, -t       Bar timeframe (1Min, 5Min, 15Min, 30Min)
  --start-date          Start date (YYYY-MM-DD)
  --end-date            End date (YYYY-MM-DD)
  --quick               Quick test: last 5 trading days
  --config, -c          Path to config YAML file
  --output-dir, -o      Output directory for results
  --step-size           Minutes between predictions (default: 30)
  --no-plots            Skip visualization generation
  --verbose, -v         Verbose logging (DEBUG level)
```

### Python API

```python
import yaml
from backtesting import BacktestEngine, BacktestMetrics, BacktestVisualizer

# Load configuration
with open('config_backtest.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Initialize components
engine = BacktestEngine(config)
metrics_calc = BacktestMetrics()
visualizer = BacktestVisualizer()

# Run backtest
results_df = engine.run_backtest(
    symbol='QQQ',
    timeframe='1Min',
    start_date='2025-09-01',
    end_date='2025-10-01',
    step_size_minutes=30
)

# Calculate metrics
predictions = results_df[['timestamp', 'predicted_price', 'prob_up', 'current_price']]
actuals = results_df[['timestamp', 'actual_price']]

metrics = metrics_calc.calculate_all_metrics(predictions, actuals, threshold=0.0)

# Print summary
print(metrics_calc.get_summary_report())

# Create visualizations
visualizer.plot_roc_curve(metrics, save_path='roc_curve.png')
visualizer.plot_confusion_matrix(metrics, save_path='confusion_matrix.png')
visualizer.plot_prediction_vs_actual(results_df, save_path='pred_vs_actual.png')
visualizer.create_summary_dashboard(results_df, metrics, save_path='dashboard.png')

# Save results
results_df.to_csv('backtest_results.csv', index=False)
```

## Configuration

### config_backtest.yaml Structure

```yaml
# Backtesting Parameters
backtest:
  start_date: "2025-09-01"
  end_date: "2025-10-03"
  symbols: ["QQQ", "SPY", "AAPL"]
  timeframes: ["1Min", "5Min"]
  horizon_minutes: 30

  rolling_window:
    step_size_minutes: 30      # Frequency of predictions
    min_context_bars: 480      # Minimum bars needed

  market_hours:
    start: "09:30"
    end: "16:00"
    rth_only: true             # Regular Trading Hours only

# Evaluation Metrics
metrics:
  classification:
    enabled: true
    threshold: 0.0             # Price change threshold for up/down

  regression:
    enabled: true

  probabilistic:
    enabled: true

# Visualization Settings
visualization:
  save_plots: true
  output_formats: ["png", "html"]
  dpi: 300
  style: "dark_background"

  plots:
    roc_curve: true
    confusion_matrix: true
    prediction_vs_actual: true
    price_trajectory: true
    calibration_curve: true
    residual_plot: true

# Output Configuration
output:
  base_dir: "./backtesting/results"
  save_predictions: true
  save_metrics: true
  save_summary: true
```

## Output Structure

After running a backtest, results are organized as:

```
backtesting/results/
└── QQQ_1Min_20251003_143052/
    ├── predictions.csv           # Raw predictions and actuals
    ├── metrics.json              # All calculated metrics
    ├── summary.txt               # Human-readable summary
    ├── roc_curve.png             # ROC curve plot
    ├── confusion_matrix.png      # Confusion matrix heatmap
    ├── prediction_vs_actual.png  # Scatter plot
    ├── price_trajectory.png      # Time series comparison
    ├── calibration_curve.png     # Probability calibration
    ├── residual_plot.png         # Residual analysis
    └── dashboard.png             # Combined dashboard
```

## Understanding the Metrics

### Classification Metrics (Direction Prediction)

**F1 Score** (0.0 - 1.0, higher is better)
- Harmonic mean of precision and recall
- Good: > 0.6, Excellent: > 0.8

**AUC-ROC** (0.0 - 1.0, higher is better)
- Area Under the ROC Curve
- Random: 0.5, Good: > 0.7, Excellent: > 0.85

**Precision** (0.0 - 1.0)
- Of predicted "ups", how many were correct?
- Important if you want to avoid false positives

**Recall** (0.0 - 1.0)
- Of actual "ups", how many did we catch?
- Important if you want to catch all opportunities

### Regression Metrics (Price Level Prediction)

**MAE** (Mean Absolute Error, lower is better)
- Average absolute difference between predicted and actual prices
- Same units as price (e.g., dollars)

**RMSE** (Root Mean Squared Error, lower is better)
- Square root of average squared errors
- Penalizes large errors more than MAE

**MAPE** (Mean Absolute Percentage Error, %, lower is better)
- Percentage error, easier to interpret
- Good: < 2%, Excellent: < 1%

**R² Score** (-∞ to 1.0, higher is better)
- Proportion of variance explained
- Good: > 0.5, Excellent: > 0.8

### Probabilistic Metrics (Uncertainty Calibration)

**Brier Score** (0.0 - 1.0, lower is better)
- Measures calibration of probabilities
- Perfect: 0.0

**Expected Calibration Error (ECE)** (0.0 - 1.0, lower is better)
- Average gap between predicted probabilities and observed frequencies
- Good: < 0.1

## Examples

### Example 1: Quick Model Check

Test if your model is working correctly:

```bash
python run_backtest.py --symbol QQQ --quick --verbose
```

This runs a 5-day backtest with detailed logging.

### Example 2: Compare Timeframes

```python
from backtesting import BacktestEngine, compare_models
import yaml

config = yaml.safe_load(open('config_backtest.yaml'))

timeframes = ['1Min', '5Min', '15Min']
results = []

for tf in timeframes:
    engine = BacktestEngine(config)
    df = engine.run_backtest('QQQ', tf, '2025-09-01', '2025-10-01')

    metrics_calc = BacktestMetrics()
    metrics = metrics_calc.calculate_all_metrics(df, df, threshold=0.0)
    results.append(metrics)

# Compare
comparison_df = compare_models(results, timeframes)
print(comparison_df)
```

### Example 3: Production Evaluation

Full evaluation for production model:

```bash
# 1. Run comprehensive backtest (1 month)
python run_backtest.py \
  --symbol QQQ \
  --timeframe 1Min \
  --start-date 2025-09-01 \
  --end-date 2025-10-01 \
  --output-dir ./production_eval \
  --verbose

# 2. Review metrics
cat production_eval/QQQ_1Min_*/summary.txt

# 3. Check visualizations
open production_eval/QQQ_1Min_*/dashboard.png
```

## Interpretation Guide

### What Makes a Good Model?

**Minimum Acceptable:**
- F1 Score: > 0.55
- AUC-ROC: > 0.6
- MAPE: < 3%

**Production Ready:**
- F1 Score: > 0.65
- AUC-ROC: > 0.7
- MAPE: < 2%
- Calibration Error: < 0.1

**Excellent:**
- F1 Score: > 0.75
- AUC-ROC: > 0.8
- MAPE: < 1%
- Calibration Error: < 0.05

### Red Flags

🚩 **AUC-ROC close to 0.5**: Model is no better than random guessing
🚩 **Very high F1 but low AUC**: Model may be overfitting
🚩 **Large calibration error**: Probabilities are not trustworthy
🚩 **Residuals show patterns**: Model is missing important features

## Troubleshooting

### Issue: "No predictions generated"

**Causes:**
- Insufficient historical data
- Market was closed (check RTH settings)
- API rate limits

**Solutions:**
- Reduce `target_historical_bars` in config
- Use longer date range
- Check Alpaca API status

### Issue: "Model predictions are random (AUC ≈ 0.5)"

**Causes:**
- Model not trained properly
- Wrong model checkpoint loaded
- Prediction horizon too long

**Solutions:**
- Verify model checkpoint in `config.yaml`
- Try shorter prediction horizon
- Retrain model with more data

### Issue: Visualizations not saving

**Causes:**
- Missing matplotlib/seaborn
- Permission issues
- Invalid output directory

**Solutions:**
```bash
pip install matplotlib seaborn
mkdir -p backtesting/results
```

## Advanced Usage

### Custom Metrics

```python
from backtesting.metrics import BacktestMetrics

class CustomMetrics(BacktestMetrics):
    def calculate_sharpe_ratio(self, df):
        returns = (df['actual_price'] - df['current_price']) / df['current_price']
        return returns.mean() / returns.std() * np.sqrt(252)

# Use custom metrics
custom_calc = CustomMetrics()
```

### Parallel Backtesting

```python
from concurrent.futures import ProcessPoolExecutor

symbols = ['QQQ', 'SPY', 'AAPL', 'TSLA', 'MSFT']

def run_single_backtest(symbol):
    engine = BacktestEngine(config)
    return engine.run_backtest(symbol, '1Min', '2025-09-01', '2025-10-01')

with ProcessPoolExecutor(max_workers=5) as executor:
    results = list(executor.map(run_single_backtest, symbols))
```

## Performance Tips

1. **Use longer step sizes** for faster backtesting:
   ```yaml
   rolling_window:
     step_size_minutes: 60  # Instead of 30
   ```

2. **Cache historical data** to reduce API calls:
   ```yaml
   performance:
     cache_historical: true
   ```

3. **Reduce visualization DPI** for faster rendering:
   ```yaml
   visualization:
     dpi: 150  # Instead of 300
   ```

## FAQs

**Q: How long does a backtest take?**
A: For 1 month of 1Min data with 30-min steps: ~20-30 minutes (depends on API speed).

**Q: Can I backtest crypto?**
A: Yes, but crypto trades 24/7, so RTH filtering won't apply. Set `rth_only: false`.

**Q: What's the difference between F1 and AUC?**
A: F1 measures classification accuracy at a fixed threshold. AUC measures performance across all thresholds.

**Q: Should I trust high accuracy with low AUC?**
A: No. High accuracy with low AUC usually means class imbalance. Focus on F1 and AUC together.

## Contributing

To add new metrics or visualizations:

1. Add metric calculation in `metrics.py`
2. Add visualization in `visualize.py`
3. Update CLI in `run_backtest.py`
4. Update config schema in `config_backtest.yaml`

## License

This backtesting framework is part of the Kronos intraday prediction project.

## Support

For issues or questions:
- Check troubleshooting section above
- Review logs in `backtesting/results/backtest.log`
- Verify configuration in `config_backtest.yaml`

---

**Built for evaluating Kronos time series predictions with comprehensive metrics and visualizations.**

Last updated: 2025-10-03
