# Ascending Triangle Pattern Recognizer Implementation Plan

## Overview
Build a ZigZag-based ascending triangle pattern recognizer with multi-timeframe support, CLI interface, and YOLO training data export.

## Requirements
- **Input**: List of tickers + lookback bars + timeframe (1min, 5min, 15min, 30min)
- **Data Source**: Alpaca API
- **Output**: JSON + PNG charts + YOLO annotation format
- **Interface**: CLI script

## Module Structure
```
live_chart_prediction/pattern_recognition/
├── __init__.py              # Module exports
├── models.py                # Dataclasses: Pivot, PivotType, PatternResult, BoundingBox
├── zigzag.py                # Numba-optimized ZigZag algorithm O(N)
├── pivots.py                # Pivot classification (HH, HL, LH, LL)
├── ascending_triangle.py    # Ascending triangle detection logic
├── chart_generator.py       # mplfinance chart generation with trendlines
├── yolo_exporter.py         # YOLO format annotation export
├── data_fetcher.py          # Alpaca data fetching
└── cli.py                   # CLI interface with argparse
```

## Implementation Steps

### Step 1: Create `models.py`
Define dataclasses:
- `PivotType` (enum): PEAK, VALLEY
- `PivotClass` (enum): HH, LH, HL, LL
- `Pivot`: index, price, pivot_type, timestamp, pivot_class
- `TrendLine`: start/end index, start/end price, slope, intercept
- `BoundingBox`: x_center, y_center, width, height (normalized 0-1)
- `AscendingTrianglePattern`: ticker, timeframe, peaks, valleys, resistance_level, support_slope, confidence

### Step 2: Create `zigzag.py`
Numba JIT-compiled ZigZag algorithm:
- `@jit(nopython=True)` for `_zigzag_core()` function
- State machine: trend=0 (looking), trend=1 (uptrend), trend=-1 (downtrend)
- Deviation presets: 0.5% (1min), 0.8% (5min), 1.0% (15min), 1.2% (30min)
- Returns: pivot_indices, pivot_values, pivot_types arrays

### Step 3: Create `pivots.py`
Pivot classification functions:
- `classify_pivots()`: Label HH/LH/HL/LL based on price comparisons
- `get_recent_pivots()`: Get last N peaks and valleys
- `calculate_trendline()`: Linear regression through pivot points

### Step 4: Create `ascending_triangle.py`
Pattern detection logic:
```python
class AscendingTriangleDetector:
    def detect(highs, lows, closes, ticker, timeframe):
        # 1. Compute ZigZag pivots
        # 2. Classify pivots
        # 3. Check flat resistance: |Peak_i - Peak_{i-2}| / Peak_i < 1.5%
        # 4. Check ascending support: Valley_i > Valley_{i-1} AND slope > 0
        # 5. Check compression: price range narrowing
        # 6. Return AscendingTrianglePattern or None
```

### Step 5: Create `chart_generator.py`
Chart generation using mplfinance:
- `generate_annotated_chart()`: Candlesticks + resistance/support lines + pivot markers
- `generate_yolo_training_image()`: 640x640 image, no axes/borders, returns BoundingBox

### Step 6: Create `yolo_exporter.py`
YOLO format export:
- `export_yolo_annotation()`: Write `.txt` file with `<class> <x> <y> <w> <h>`
- `export_dataset()`: Create YOLO directory structure (images/train, labels/train, data.yaml)

### Step 7: Create `data_fetcher.py`
Alpaca API wrapper (reuse patterns from prediction_service.py):
- `fetch_bars()`: Fetch OHLCV for multiple tickers
- RTH filtering (9:30 AM - 4:00 PM ET)
- Timeframe mapping to Alpaca TimeFrame objects

### Step 8: Create `cli.py`
CLI interface:
```bash
python -m pattern_recognition.cli \
    --tickers QQQ SPY AAPL \
    --timeframe 5min \
    --lookback 200 \
    --chart \
    --yolo \
    --output ./output
```

### Step 9: Update `requirements.txt`
Add dependencies:
```
numba>=0.57.0          # JIT compilation for ZigZag
mplfinance>=0.12.10b0  # Candlestick chart generation
```

### Step 10: Create `__init__.py`
Module exports following market_regime pattern.

## Key Detection Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| resistance_tolerance | 1.5% | Max deviation for flat resistance |
| min_support_slope | 0.0001 | Minimum positive slope for support |
| min_peaks | 2 | Minimum peaks required |
| min_valleys | 2 | Minimum valleys required |
| zigzag_deviation | Auto | Based on timeframe (0.5%-1.2%) |

## Critical Files to Reference
- `prediction_service.py:292-408` - Alpaca data fetching patterns
- `market_regime/ofi_divergence.py` - Dataclass patterns
- `market_regime/__init__.py` - Module export structure

## Expected Output

### JSON Format
```json
{
  "ticker": "QQQ",
  "timeframe": "5min",
  "resistance_level": 525.50,
  "support_slope": 0.00123,
  "confidence": 0.85,
  "peaks": [{"index": 150, "price": 525.48}, {"index": 180, "price": 525.52}],
  "valleys": [{"index": 140, "price": 520.10}, {"index": 170, "price": 522.30}]
}
```

### YOLO Annotation Format
```
0 0.450000 0.550000 0.300000 0.200000
```
(class_id x_center y_center width height - all normalized 0-1)

### Chart Output
PNG with:
- Candlestick chart
- Red dashed horizontal line (resistance)
- Green dashed ascending line (support)
- Red triangles at peaks, green triangles at valleys
