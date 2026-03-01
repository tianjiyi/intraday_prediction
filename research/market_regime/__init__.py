"""
Market Regime Detection Module

This module implements HMM-based market regime detection using:
- OFI (Order Flow Imbalance) - Cont (2014)
- VPIN (Volume-Synchronized Probability of Informed Trading)
- GaussianHMM with 4 states: Range, Bull Trend, Bear Trend, Stress/Reversal

Data source: Polygon.io for historical tick/quote data

Usage:
    from market_regime import MarketRegimeService, get_regime_service

    # Initialize with API key
    service = MarketRegimeService(polygon_api_key="your_key")

    # Train model
    service.train_model("QQQ", training_days=30)

    # Get current regime
    regime = service.get_current_regime("QQQ")
    print(regime['regime']['state'])  # 'Bull Trend', 'Bear Trend', etc.
"""

from .polygon_fetcher import PolygonDataFetcher
from .volume_bars import resample_to_volume_bars, resample_to_volume_bars_vectorized
from .ofi import calculate_ofi, calculate_ofi_cumulative, calculate_ofi_normalized
from .vpin import calculate_vpin, calculate_vpin_for_volume_bars
from .features import (
    rolling_z_score,
    calculate_log_return,
    calculate_volatility,
    calculate_realized_volatility,
    prepare_hmm_features
)
from .regime_detector import RegimeDetector, REGIME_LABELS
from .regime_service import (
    MarketRegimeService,
    get_regime_service,
    init_regime_service
)

__all__ = [
    # Data fetching
    'PolygonDataFetcher',

    # Volume bar resampling
    'resample_to_volume_bars',
    'resample_to_volume_bars_vectorized',

    # OFI calculation
    'calculate_ofi',
    'calculate_ofi_cumulative',
    'calculate_ofi_normalized',

    # VPIN calculation
    'calculate_vpin',
    'calculate_vpin_for_volume_bars',

    # Feature engineering
    'rolling_z_score',
    'calculate_log_return',
    'calculate_volatility',
    'calculate_realized_volatility',
    'prepare_hmm_features',

    # Regime detection
    'RegimeDetector',
    'REGIME_LABELS',

    # Service layer
    'MarketRegimeService',
    'get_regime_service',
    'init_regime_service',
]

__version__ = '1.0.0'
