"""
Train HMM Model Using Tick-Level Features from TimescaleDB

This script trains the HMM regime detector using true tick-level
OFI and VPIN calculations from the trades/quotes database.

Usage:
    python -m market_regime.scripts.train_hmm_tick [--symbol QQQ] [--days 25]
"""

import argparse
import logging
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from market_regime.feature_pipeline import FeaturePipeline
from market_regime.regime_detector import RegimeDetector
from market_regime.database import test_connection

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s [%(name)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Default model save path
MODEL_DIR = Path(__file__).parent.parent / 'models'
MODEL_DIR.mkdir(exist_ok=True)


def train_hmm_with_tick_features(
    symbol: str = 'QQQ',
    training_days: int = 25,
    timeframe: str = '1min',
    bucket_volume: int = 10000,
    vpin_buckets: int = 50,
    n_states: int = 4,
    save_model: bool = True
) -> RegimeDetector:
    """
    Train HMM using tick-level features from database.

    Args:
        symbol: Trading symbol
        training_days: Number of days to use for training
        timeframe: Bar frequency for features
        bucket_volume: Volume per VPIN bucket
        vpin_buckets: Rolling window for VPIN
        n_states: Number of HMM states
        save_model: Whether to save the trained model

    Returns:
        Trained RegimeDetector
    """
    logger.info("=" * 60)
    logger.info("HMM Training with Tick-Level Features")
    logger.info("=" * 60)

    # Test database connection
    logger.info("Testing database connection...")
    if not test_connection():
        raise RuntimeError("Cannot connect to database. Is TimescaleDB running?")

    # Initialize pipeline
    logger.info(f"Initializing feature pipeline for {symbol}...")
    pipeline = FeaturePipeline(
        symbol=symbol,
        bucket_volume=bucket_volume,
        vpin_buckets=vpin_buckets
    )

    # Get available dates
    available_dates = pipeline.get_available_dates()
    if not available_dates:
        raise ValueError(f"No data available for {symbol} in database")

    logger.info(f"Available dates: {len(available_dates)} ({available_dates[0]} to {available_dates[-1]})")

    # Use most recent N days for training
    train_dates = available_dates[-training_days:]
    start_date = train_dates[0]
    end_date = train_dates[-1]

    logger.info(f"Training period: {start_date} to {end_date} ({len(train_dates)} days)")

    # Compute features
    logger.info("Computing tick-level features from database...")
    raw_features = pipeline.compute_features(
        start_date=start_date,
        end_date=end_date,
        timeframe=timeframe,
        rth_only=True,
        use_cache=True
    )

    if raw_features.empty:
        raise ValueError("No features computed. Check database and date range.")

    logger.info(f"Raw features: {len(raw_features)} bars")
    logger.info(f"Date range: {raw_features.index.min()} to {raw_features.index.max()}")

    # Normalize for HMM
    logger.info("Normalizing features for HMM...")
    hmm_features = pipeline.normalize_for_hmm(raw_features)

    logger.info(f"HMM features: {len(hmm_features)} samples")
    logger.info(f"Feature summary:\n{hmm_features.describe()}")

    # Check for minimum samples
    min_samples = n_states * 100  # 100 samples per state minimum
    if len(hmm_features) < min_samples:
        logger.warning(
            f"Only {len(hmm_features)} samples available. "
            f"Recommended minimum: {min_samples} ({n_states} states x 100)"
        )

    # Train HMM
    logger.info(f"Training HMM with {n_states} states...")
    detector = RegimeDetector(
        n_states=n_states,
        covariance_type='diag',  # More stable than 'full'
        n_iter=100,
        random_state=42
    )

    detector.fit(hmm_features)

    # Print model info
    info = detector.get_model_info()
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)

    logger.info(f"\nState Labels: {info['state_labels']}")

    logger.info("\nState Means:")
    for state_name, means in info['means'].items():
        logger.info(f"  {state_name}:")
        for feat, val in means.items():
            logger.info(f"    {feat}: {val:.4f}")

    logger.info("\nTransition Matrix:")
    trans_mat = np.array(info['transition_matrix'])
    labels = [info['state_labels'].get(i, f"State_{i}") for i in range(n_states)]
    logger.info(f"  {labels}")
    for i, row in enumerate(trans_mat):
        logger.info(f"  {labels[i]}: {[f'{p:.3f}' for p in row]}")

    # Validate on training data
    logger.info("\nValidating on training data...")
    predictions = detector.predict(hmm_features)
    unique, counts = np.unique(predictions, return_counts=True)

    logger.info("State distribution:")
    for state_id, count in zip(unique, counts):
        label = info['state_labels'].get(state_id, f"State_{state_id}")
        pct = count / len(predictions) * 100
        logger.info(f"  {label}: {count} ({pct:.1f}%)")

    # Save model
    if save_model:
        model_path = MODEL_DIR / f'hmm_regime_tick_{symbol}.pkl'
        detector.save(str(model_path))
        logger.info(f"\nModel saved to: {model_path}")

        # Also save as default model
        default_path = MODEL_DIR / 'hmm_regime_model.pkl'
        detector.save(str(default_path))
        logger.info(f"Also saved as default: {default_path}")

    return detector


def main():
    parser = argparse.ArgumentParser(
        description='Train HMM with tick-level features from TimescaleDB'
    )
    parser.add_argument('--symbol', type=str, default='QQQ',
                        help='Trading symbol (default: QQQ)')
    parser.add_argument('--days', type=int, default=25,
                        help='Number of training days (default: 25)')
    parser.add_argument('--timeframe', type=str, default='1min',
                        help='Bar frequency (default: 1min)')
    parser.add_argument('--bucket-volume', type=int, default=10000,
                        help='Volume per VPIN bucket (default: 10000)')
    parser.add_argument('--states', type=int, default=4,
                        help='Number of HMM states (default: 4)')

    args = parser.parse_args()

    try:
        detector = train_hmm_with_tick_features(
            symbol=args.symbol,
            training_days=args.days,
            timeframe=args.timeframe,
            bucket_volume=args.bucket_volume,
            n_states=args.states,
            save_model=True
        )

        logger.info("\nTraining completed successfully!")

    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()
