"""
Pattern Recognition Module

Provides ascending triangle detection using ZigZag algorithm
with YOLO training data export capabilities.

Features:
- Numba-optimized ZigZag pivot detection (O(N) complexity)
- Ascending triangle pattern recognition
- Multi-timeframe support (1min, 5min, 15min, 30min, daily)
- Annotated chart generation with resistance/support lines
- YOLO format export for ML training

Quick Start:
    from pattern_recognition import AscendingTriangleDetector, fetch_bars

    # Fetch data
    data = fetch_bars(["QQQ"], lookback_bars=200, timeframe="5min")

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

CLI Usage:
    python -m pattern_recognition.cli --tickers QQQ SPY --timeframe 5min --chart

Author: Kronos Trading System
Version: 1.0.0
"""

# Models
from .models import (
    Pivot,
    PivotType,
    PivotClass,
    BreakoutStatus,
    TrendLine,
    BoundingBox,
    AscendingTrianglePattern,
    PATTERN_CLASSES,
    PatternRecognitionError,
    InsufficientDataError,
    NoPivotsFoundError,
    DataFetchError,
)

# ZigZag algorithm
from .zigzag import (
    compute_zigzag,
    compute_zigzag_for_timeframe,
    DEVIATION_PRESETS,
    get_zigzag_segments,
    filter_pivots_by_range,
    get_last_n_pivots,
)

# Pivot classification
from .pivots import (
    classify_pivots,
    get_recent_pivots,
    calculate_trendline,
    create_trendline,
    calculate_average_price,
    calculate_price_deviation,
    check_higher_lows,
    check_lower_highs,
    check_flat_tops,
    check_flat_bottoms,
    get_pivot_sequence,
    get_market_structure,
)

# Pattern detection
from .ascending_triangle import (
    AscendingTriangleDetector,
    scan_for_ascending_triangles,
    get_pattern_summary,
    MIN_BARS_PRESETS,
    RESISTANCE_TOLERANCE_PRESETS,
)

# Chart generation
from .chart_generator import PatternChartGenerator

# YOLO export
from .yolo_exporter import (
    export_yolo_annotation,
    export_multiple_annotations,
    create_yolo_dataset_structure,
    generate_data_yaml,
    export_dataset,
    validate_annotation,
    get_dataset_stats,
)

# Data fetching
from .data_fetcher import (
    fetch_bars,
    fetch_single_ticker,
    load_config,
    get_alpaca_client,
    get_supported_timeframes,
    get_market_status,
    validate_ticker,
)


__all__ = [
    # Models
    'Pivot',
    'PivotType',
    'PivotClass',
    'BreakoutStatus',
    'TrendLine',
    'BoundingBox',
    'AscendingTrianglePattern',
    'PATTERN_CLASSES',
    'PatternRecognitionError',
    'InsufficientDataError',
    'NoPivotsFoundError',
    'DataFetchError',

    # ZigZag
    'compute_zigzag',
    'compute_zigzag_for_timeframe',
    'DEVIATION_PRESETS',
    'get_zigzag_segments',
    'filter_pivots_by_range',
    'get_last_n_pivots',

    # Pivots
    'classify_pivots',
    'get_recent_pivots',
    'calculate_trendline',
    'create_trendline',
    'calculate_average_price',
    'calculate_price_deviation',
    'check_higher_lows',
    'check_lower_highs',
    'check_flat_tops',
    'check_flat_bottoms',
    'get_pivot_sequence',
    'get_market_structure',

    # Detection
    'AscendingTriangleDetector',
    'scan_for_ascending_triangles',
    'get_pattern_summary',
    'MIN_BARS_PRESETS',
    'RESISTANCE_TOLERANCE_PRESETS',

    # Charts
    'PatternChartGenerator',

    # YOLO
    'export_yolo_annotation',
    'export_multiple_annotations',
    'create_yolo_dataset_structure',
    'generate_data_yaml',
    'export_dataset',
    'validate_annotation',
    'get_dataset_stats',

    # Data
    'fetch_bars',
    'fetch_single_ticker',
    'load_config',
    'get_alpaca_client',
    'get_supported_timeframes',
    'get_market_status',
    'validate_ticker',
]

__version__ = '1.0.0'
__author__ = 'Kronos Trading System'
