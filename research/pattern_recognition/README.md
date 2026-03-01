# Ascending Triangle Pattern Recognizer

ZigZag-based ascending triangle pattern detection with multi-timeframe support, annotated chart generation, and YOLO training data export.

## Features

- **Numba JIT-optimized ZigZag algorithm** - O(N) complexity for fast pivot detection
- **Multi-timeframe support** - 1min, 5min, 15min, 30min, 1hour, daily
- **Automatic deviation scaling** - Timeframe-aware ZigZag deviation presets
- **Annotated chart generation** - Candlestick charts with resistance/support lines
- **YOLO format export** - Training data for YOLOv8 object detection
- **Alpaca API integration** - Real-time and historical market data

## Installation

```bash
# Install dependencies
pip install numba mplfinance alpaca-py pandas numpy pytz pyyaml

# Or add to requirements.txt
numba>=0.57.0
mplfinance>=0.12.10b0
alpaca-py>=0.10.0
```

## Configuration

Create `config.yaml` in the project root with your Alpaca API credentials:

```yaml
ALPACA_KEY_ID: "your_api_key"
ALPACA_SECRET_KEY: "your_secret_key"
```

## CLI Usage

### Basic Pattern Scan

Scan a single ticker for ascending triangle patterns:

```bash
python -m pattern_recognition.cli --tickers QQQ --timeframe 5min
```

### Multiple Tickers

Scan multiple tickers at once:

```bash
python -m pattern_recognition.cli --tickers QQQ SPY AAPL TSLA NVDA --timeframe 5min
```

### With Chart Output

Generate annotated PNG charts showing resistance and support lines:

```bash
python -m pattern_recognition.cli --tickers QQQ SPY --timeframe 5min --chart
```

### With JSON Output

Export pattern data to JSON format:

```bash
python -m pattern_recognition.cli --tickers QQQ SPY --timeframe 5min --json
```

### Custom Lookback Period

Specify number of bars to analyze (default: 200):

```bash
python -m pattern_recognition.cli --tickers QQQ --timeframe 15min --lookback 300
```

### Custom Output Directory

Save outputs to a specific directory:

```bash
python -m pattern_recognition.cli --tickers QQQ SPY --chart --json --output ./my_patterns
```

### YOLO Training Data Export

Generate YOLO-format training data for object detection:

```bash
python -m pattern_recognition.cli --tickers QQQ --yolo --yolo-dir ./yolo_dataset
```

### Full Output (JSON + Charts + YOLO)

Generate all output formats:

```bash
python -m pattern_recognition.cli --tickers QQQ SPY AAPL TSLA NVDA \
    --timeframe 5min \
    --lookback 300 \
    --chart \
    --json \
    --yolo \
    --yolo-dir ./yolo_dataset \
    --output ./pattern_output
```

### Breakout Detection

Analyze post-pattern bars to detect breakout success or failure:

```bash
# Default breakout analysis (20 bars pre/post context)
python -m pattern_recognition.cli --tickers QQQ SPY --chart --json

# Extended lookforward for breakout (40 bars after pattern)
python -m pattern_recognition.cli --tickers QQQ --post-context 40 --chart

# More context before pattern (30 bars)
python -m pattern_recognition.cli --tickers QQQ --pre-context 30 --post-context 40 --chart

# Skip breakout analysis (pattern detection only)
python -m pattern_recognition.cli --tickers QQQ --no-breakout-analysis
```

**Breakout Status Types:**
- `SUCCESS` - Price closed above resistance level
- `FAILURE` - Price closed below support trendline
- `PENDING` - Not enough post-pattern data
- `EXPIRED` - Lookforward period passed without breakout

### Full Chart Range

Show all lookback bars instead of just the pattern area:

```bash
# Full lookback range on chart (all 300 bars)
python -m pattern_recognition.cli --tickers QQQ --timeframe daily --lookback 100 --chart --full-chart

# Compare: default shows only pattern with context
python -m pattern_recognition.cli --tickers QQQ --chart  # Just pattern area
```

### Different Timeframes

```bash
# 1-minute bars (scalping)
python -m pattern_recognition.cli --tickers QQQ --timeframe 1min --lookback 500

# 5-minute bars (intraday)
python -m pattern_recognition.cli --tickers QQQ --timeframe 5min --lookback 300

# 15-minute bars (swing)
python -m pattern_recognition.cli --tickers QQQ --timeframe 15min --lookback 200

# 30-minute bars
python -m pattern_recognition.cli --tickers QQQ --timeframe 30min --lookback 150

# 1-hour bars
python -m pattern_recognition.cli --tickers QQQ --timeframe 1hour --lookback 100

# Daily bars
python -m pattern_recognition.cli --tickers QQQ --timeframe daily --lookback 60
```

### Include Extended Hours

By default, only Regular Trading Hours (9:30 AM - 4:00 PM ET) are included. To include pre/post market:

```bash
python -m pattern_recognition.cli --tickers QQQ --timeframe 5min --no-rth
```

### Custom Detection Parameters

Fine-tune the pattern detection:

```bash
python -m pattern_recognition.cli --tickers QQQ \
    --timeframe 5min \
    --resistance-tolerance 0.02 \
    --min-support-slope 0.00005 \
    --zigzag-deviation 0.01 \
    --min-bars 25
```

### Verbose Output

Enable detailed logging:

```bash
python -m pattern_recognition.cli --tickers QQQ --timeframe 5min --verbose
```

## CLI Options Reference

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--tickers` | `-t` | Required | List of ticker symbols |
| `--timeframe` | `-tf` | `5min` | Bar timeframe (1min, 5min, 15min, 30min, 1hour, daily) |
| `--lookback` | `-l` | `200` | Number of bars to analyze |
| `--output` | `-o` | `./pattern_output` | Output directory |
| `--chart` | | False | Generate annotated PNG charts |
| `--json` | | False | Export results to JSON |
| `--yolo` | | False | Generate YOLO training format |
| `--yolo-dir` | | `./yolo_dataset` | YOLO dataset output directory |
| `--full-chart` | | False | Plot all lookback bars instead of just pattern area |
| `--pre-context` | | `20` | Bars before pattern for context |
| `--post-context` | | `20` | Bars after pattern for breakout analysis |
| `--no-breakout-analysis` | | False | Skip breakout detection |
| `--resistance-tolerance` | | `0.015` | Tolerance for flat resistance (1.5%) |
| `--min-support-slope` | | `0.00001` | Minimum positive slope for support |
| `--zigzag-deviation` | | Auto | ZigZag deviation (auto-calculated by timeframe) |
| `--min-bars` | | Auto | Minimum bars for valid pattern (auto-scaled by timeframe) |
| `--no-rth` | | False | Include extended hours data |
| `--verbose` | `-v` | False | Enable verbose logging |
| `--config` | | Auto | Path to config.yaml |

## Output Formats

### JSON Output

```json
{
  "timestamp": "20251228T112832",
  "timeframe": "5min",
  "lookback_bars": 300,
  "pre_context_bars": 30,
  "post_context_bars": 40,
  "patterns": [
    {
      "ticker": "NVDA",
      "timeframe": "5min",
      "resistance_level": 192.44,
      "support_slope": 0.012,
      "support_intercept": 188.50,
      "confidence": 0.624,
      "breakout_status": "success",
      "breakout_index": 279,
      "breakout_price": 192.46,
      "bars_to_breakout": 1,
      "post_pattern_high": 192.64,
      "post_pattern_low": 188.0,
      "start_index": 150,
      "end_index": 280,
      "peaks": [
        {"index": 180, "price": 192.42},
        {"index": 250, "price": 192.46}
      ],
      "valleys": [
        {"index": 160, "price": 189.20},
        {"index": 220, "price": 190.80}
      ]
    }
  ],
  "summary": {
    "count": 1,
    "avg_confidence": 0.624,
    "max_confidence": 0.624,
    "highest_confidence_ticker": "NVDA"
  }
}
```

### Chart Output

PNG files with:
- Candlestick chart
- Red dashed horizontal line (flat resistance)
- Green dashed ascending line (support trendline)
- Red triangle markers at peaks
- Green triangle markers at valleys
- Legend with price levels

### YOLO Output

Directory structure:
```
yolo_dataset/
├── images/
│   ├── train/
│   │   └── NVDA_20251228T112832.png
│   └── val/
├── labels/
│   ├── train/
│   │   └── NVDA_20251228T112832.txt
│   └── val/
└── data.yaml
```

Annotation format (normalized 0-1):
```
0 0.450000 0.550000 0.300000 0.200000
```
`<class_id> <x_center> <y_center> <width> <height>`

## Python API Usage

```python
from pattern_recognition import (
    AscendingTriangleDetector,
    PatternChartGenerator,
    fetch_bars,
    export_yolo_annotation
)

# Fetch data
data = fetch_bars(["QQQ", "SPY"], lookback_bars=200, timeframe="5min")

# Detect pattern
detector = AscendingTriangleDetector()
pattern = detector.detect(
    highs=data["QQQ"]['high'].values,
    lows=data["QQQ"]['low'].values,
    closes=data["QQQ"]['close'].values,
    ticker="QQQ",
    timeframe="5min"
)

if pattern:
    print(f"Pattern found: resistance=${pattern.resistance_level:.2f}")
    print(f"Confidence: {pattern.confidence:.1%}")

    # Generate chart
    generator = PatternChartGenerator()
    generator.generate_annotated_chart(data["QQQ"], pattern, "qqq_pattern.png")

    # Export YOLO annotation
    export_yolo_annotation(pattern, "qqq_pattern.txt")
```

## Detection Algorithm

### ZigZag Pivot Detection

1. **State Machine**: Tracks trend direction (uptrend/downtrend/looking)
2. **Deviation Threshold**: Minimum price change to confirm pivot
3. **Timeframe Presets** (deviation):
   - 1min: 0.5% deviation
   - 5min: 0.8% deviation
   - 15min: 1.0% deviation
   - 30min: 1.2% deviation
   - 1hour: 1.5% deviation
   - daily: 3.0% deviation
   - weekly: 5.0% deviation

### Minimum Bars Presets

Auto-scaled by timeframe to ensure valid patterns at different time horizons:
- 1min: 30 bars (~30 minutes)
- 5min: 25 bars (~2 hours)
- 15min: 20 bars (~5 hours)
- 30min: 15 bars (~7.5 hours)
- 1hour: 12 bars (~12 hours)
- daily: 10 bars (~2 weeks)
- weekly: 6 bars (~6 weeks)

### Resistance Tolerance Presets

Auto-scaled by timeframe for stricter intraday detection:
- 1min: 0.5% - very tight for scalping
- 5min: 0.7% - tight for intraday
- 15min: 1.0% - moderate for swing
- 30min: 1.2%
- 1hour: 1.5%
- daily: 1.5% - original default
- weekly: 2.0% - looser for weekly patterns

### Ascending Triangle Criteria

1. **Flat Resistance**: Peaks within tolerance (auto-scaled by timeframe)
2. **Rising Support**: Higher lows with positive slope
3. **Minimum Pivots**: At least 2 peaks and 2 valleys
4. **Compression**: Price range narrowing toward apex

### Confidence Calculation

```
confidence = (resistance_score * 0.4) + (support_score * 0.4) + (compression_score * 0.2)
```

- **Resistance Score**: How flat are the peaks (0-1)
- **Support Score**: How consistent is the upward slope (0-1)
- **Compression Score**: How much is price range narrowing (0-1)

## Troubleshooting

### No patterns found

- Increase lookback period: `--lookback 400`
- Adjust resistance tolerance: `--resistance-tolerance 0.02`
- Try different timeframe: `--timeframe 15min`
- Check if market is ranging (triangles form in consolidation)

### API errors

- Verify `config.yaml` has valid Alpaca API keys
- Check internet connection
- Ensure ticker symbols are valid (uppercase)

### Chart generation errors

- Install mplfinance: `pip install mplfinance`
- Ensure output directory is writable

### Slow performance

- Numba JIT compilation is slow on first run (cached after)
- Reduce lookback for faster scans: `--lookback 150`

## License

Part of the Kronos Trading System.
