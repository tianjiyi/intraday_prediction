"""
Market Regime Service

Orchestrates the full pipeline for market regime detection:
1. Fetch tick/bar data from Polygon.io
2. Resample to volume bars (or use time bars)
3. Calculate OFI and VPIN
4. Prepare features
5. Train/load HMM model
6. Real-time regime inference

This service provides a high-level API for the market regime module.
Supports both tick data (paid tier) and aggregated bars (free tier).
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta, date
from typing import Dict, Any, Optional, List, Literal
from pathlib import Path
import logging
import os

from .polygon_fetcher import PolygonDataFetcher
from .volume_bars import resample_to_volume_bars_vectorized
from .ofi import calculate_ofi
from .vpin import calculate_vpin_for_volume_bars
from .features import prepare_hmm_features
from .regime_detector import RegimeDetector
from .database import test_connection
from .feature_pipeline import FeaturePipeline
from .time_mapping import align_regime_history_to_time

logger = logging.getLogger(__name__)


def resample_to_timeframe(history_df: pd.DataFrame, timeframe: str = '1T') -> pd.DataFrame:
    """
    Resample volume bucket data to time-aligned bars.

    Args:
        history_df: DataFrame with timestamp, ofi, vpin, state_id, close columns
        timeframe: Pandas resample string ('1T'=1min, '5T'=5min, '15T'=15min, '30T'=30min)

    Returns:
        DataFrame with columns: timestamp, ofi, vpin, hmm_state, close
        Timestamps are converted to US/Eastern to match Alpaca candlestick data.
    """
    if history_df.empty:
        return history_df

    # Convert timestamp column to datetime if it's string
    # Use format='mixed' to handle timestamps with/without microseconds
    # e.g., "2025-12-24 14:39:41" vs "2025-12-24 14:39:41.123456"
    df = history_df.copy()
    if 'timestamp' in df.columns:
        df['time'] = pd.to_datetime(df['timestamp'], format='mixed', utc=True)
    elif 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'], format='mixed', utc=True)
    else:
        logger.warning("No timestamp column found in history data")
        return history_df

    # Convert to US/Eastern timezone to match Alpaca candlestick data
    # Note: pd.to_datetime with utc=True already makes timestamps timezone-aware
    try:
        df['time'] = df['time'].dt.tz_convert('US/Eastern')
    except Exception as e:
        logger.warning(f"Timezone conversion failed: {e}")

    # Set index for resampling
    df = df.set_index('time')

    # Define aggregation functions
    def safe_mode(x):
        """Get mode of series, return 0 if empty."""
        if len(x) == 0:
            return 0
        modes = x.mode()
        return int(modes.iloc[0]) if len(modes) > 0 else 0

    # Resample with appropriate aggregations
    agg_dict = {
        'ofi': 'sum',          # Sum of OFI for the period
        'vpin': 'max',         # Max VPIN (peak toxicity)
        'state_id': safe_mode, # Mode of HMM state
        'close': 'last'        # Last close price
    }

    # Only include columns that exist
    existing_cols = {col: agg for col, agg in agg_dict.items() if col in df.columns}

    if not existing_cols:
        logger.warning("No aggregatable columns found")
        return history_df

    resampled = df.resample(timeframe).agg(existing_cols).dropna()

    # Rename state_id to hmm_state for clarity in frontend
    if 'state_id' in resampled.columns:
        resampled = resampled.rename(columns={'state_id': 'hmm_state'})

    # Reset index to get timestamp as column
    resampled = resampled.reset_index()
    resampled = resampled.rename(columns={'time': 'timestamp'})

    # Convert timestamp to ISO string format for JSON serialization
    # Format: "2024-12-24T09:30:00-05:00" (with timezone)
    resampled['timestamp'] = resampled['timestamp'].apply(
        lambda x: x.isoformat() if hasattr(x, 'isoformat') else str(x)
    )

    return resampled


class MarketRegimeService:
    """
    High-level service for market regime detection.

    Manages the full pipeline from data fetching to regime prediction.
    """

    # Default model save path (relative to this file's directory)
    DEFAULT_MODEL_PATH = Path(__file__).parent / "models" / "hmm_regime_model.pkl"

    def __init__(
        self,
        polygon_api_key: Optional[str] = None,
        volume_bar_threshold: int = 10000,
        vpin_buckets: int = 50,
        z_score_window: int = 50,
        model_path: Optional[str] = None,
        auto_load: bool = True,
        use_s3: bool = True,
        s3_access_key: Optional[str] = None,
        s3_secret_key: Optional[str] = None,
        s3_endpoint: Optional[str] = None,
        s3_bucket: Optional[str] = None,
        bar_type: Literal['time', 'volume'] = 'time'
    ):
        """
        Initialize the market regime service.

        Args:
            polygon_api_key: Polygon.io API key (or from env POLYGON_API_KEY)
            volume_bar_threshold: Shares per volume bar (default 10,000)
            vpin_buckets: Rolling window for VPIN (default 50)
            z_score_window: Window for z-score normalization (default 50)
            model_path: Path to pre-trained model (optional, uses default if not provided)
            auto_load: Automatically load saved model on startup (default True)
            use_s3: Use S3 flat files for faster downloads (default True)
            s3_access_key: Polygon S3 Access Key (or from env/config)
            s3_secret_key: Polygon S3 Secret Key (or from env/config)
            s3_endpoint: S3 endpoint URL (default: https://files.polygon.io)
            s3_bucket: S3 bucket name (default: flatfiles)
            bar_type: 'time' for time bars, 'volume' for volume bars
        """
        # Get API key from parameter or environment
        self.api_key = polygon_api_key or os.environ.get('POLYGON_API_KEY')

        if not self.api_key:
            logger.warning(
                "No Polygon.io API key provided. "
                "Set POLYGON_API_KEY environment variable or pass to constructor."
            )
            self.data_fetcher = None
        else:
            self.data_fetcher = PolygonDataFetcher(
                self.api_key,
                use_s3=use_s3,
                s3_access_key=s3_access_key,
                s3_secret_key=s3_secret_key,
                s3_endpoint=s3_endpoint,
                s3_bucket=s3_bucket
            )

        self.volume_bar_threshold = volume_bar_threshold
        self.vpin_buckets = vpin_buckets
        self.z_score_window = z_score_window
        self.bar_type = bar_type

        # Set model path (use provided or default)
        self.model_path = Path(model_path) if model_path else self.DEFAULT_MODEL_PATH

        # Ensure models directory exists
        self.model_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize detector
        self.detector = RegimeDetector(n_states=4, random_state=42)

        # Auto-load pre-trained model if exists
        if auto_load and self.model_path.exists():
            try:
                self.detector = RegimeDetector.load(str(self.model_path))
                logger.info(f"Auto-loaded pre-trained model from {self.model_path}")
            except Exception as e:
                logger.warning(f"Failed to load model from {self.model_path}: {e}")
                logger.info("Starting with fresh model")

        # Cache for recent data
        self._tick_cache: Optional[pd.DataFrame] = None
        self._volume_bars_cache: Optional[pd.DataFrame] = None
        self._features_cache: Optional[pd.DataFrame] = None
        self._last_fetch_time: Optional[datetime] = None

        # Database feature pipeline (for tick-based features)
        self._db_available: Optional[bool] = None
        self._feature_pipeline: Optional[FeaturePipeline] = None
        self._db_symbols: List[str] = ['QQQ']  # Symbols with data in TimescaleDB

        logger.info("MarketRegimeService initialized")

    def _check_database_available(self) -> bool:
        """Check if TimescaleDB is available and has data."""
        if self._db_available is None:
            try:
                self._db_available = test_connection()
                if self._db_available:
                    logger.info("TimescaleDB connection available - using tick-based features")
                else:
                    logger.warning("TimescaleDB not available - falling back to OHLC features")
            except Exception as e:
                logger.warning(f"Database check failed: {e}")
                self._db_available = False
        return self._db_available

    def _get_feature_pipeline(self, symbol: str) -> FeaturePipeline:
        """Get or create FeaturePipeline for a symbol with current bar_type."""
        if (self._feature_pipeline is None
            or self._feature_pipeline.symbol != symbol
            or self._feature_pipeline.bar_type != self.bar_type):
            self._feature_pipeline = FeaturePipeline(
                symbol=symbol,
                bucket_volume=self.volume_bar_threshold,
                vpin_buckets=self.vpin_buckets,
                zscore_window=self.z_score_window,
                bar_type=self.bar_type,
                volume_bar_size=self.volume_bar_threshold
            )
        return self._feature_pipeline

    def _compute_tick_features_from_db(
        self,
        ticker: str,
        start_date: date,
        end_date: date,
        timeframe: str = '1min'
    ) -> Optional[pd.DataFrame]:
        """
        Compute tick-level features from TimescaleDB.

        Uses the FeaturePipeline to calculate true OFI and VPIN from
        the trades and quotes tables.

        Args:
            ticker: Stock ticker symbol (e.g., 'QQQ')
            start_date: Start date
            end_date: End date
            timeframe: Bar frequency ('1min', '5min', '15min', '30min')

        Returns:
            DataFrame with features or None if not available
        """
        if ticker not in self._db_symbols:
            logger.debug(f"No database data for {ticker}, using OHLC fallback")
            return None

        if not self._check_database_available():
            return None

        try:
            pipeline = self._get_feature_pipeline(ticker)

            # Check if dates are available
            available_dates = pipeline.get_available_dates()
            if not available_dates:
                logger.warning(f"No dates available in database for {ticker}")
                return None

            # Filter to requested range
            valid_dates = [d for d in available_dates if start_date <= d <= end_date]
            if not valid_dates:
                logger.warning(f"No data in requested date range for {ticker}")
                return None

            logger.info(f"Computing tick-based features for {ticker} from database...")
            logger.info(f"Date range: {valid_dates[0]} to {valid_dates[-1]} ({len(valid_dates)} days)")

            # Compute raw features
            raw_features = pipeline.compute_features(
                start_date=valid_dates[0],
                end_date=valid_dates[-1],
                timeframe=timeframe,
                rth_only=True,
                use_cache=True
            )

            if raw_features.empty:
                logger.warning("No features computed from database")
                return None

            logger.info(f"Computed {len(raw_features)} feature bars from tick data")

            # Normalize for HMM
            hmm_features = pipeline.normalize_for_hmm(raw_features)

            return hmm_features

        except Exception as e:
            logger.error(f"Failed to compute tick features from database: {e}")
            return None

    def fetch_and_process(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        force_refresh: bool = False,
        use_agg_bars: bool = False  # Use tick data for accurate VPIN (requires paid Polygon tier)
    ) -> pd.DataFrame:
        """
        Fetch data and process into features.

        Args:
            ticker: Stock ticker symbol (e.g., 'QQQ')
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            force_refresh: Force re-fetch even if cached
            use_agg_bars: Use aggregated bars instead of tick data (default True for free tier)

        Returns:
            DataFrame with volume bars and features
        """
        if self.data_fetcher is None:
            raise ValueError("Polygon.io API key not configured")

        cache_valid = (
            not force_refresh
            and self._volume_bars_cache is not None
            and self._last_fetch_time is not None
            and (datetime.now() - self._last_fetch_time).seconds < 300  # 5 min cache
        )

        if not cache_valid:
            if use_agg_bars:
                # Use aggregated bars (works with free tier rate limits)
                logger.info(f"Fetching aggregated bars for {ticker} from {start_date} to {end_date}")
                bars_df = self.data_fetcher.fetch_agg_bars(
                    ticker, start_date, end_date, timeframe_minutes=1
                )

                if bars_df.empty:
                    logger.warning(f"No bar data available for {ticker}")
                    return pd.DataFrame()

                # For aggregated bars, we use the bars directly (already time-sampled)
                # Add placeholder quote data for OFI (will be estimated from OHLC)
                bars_df['time'] = bars_df['timestamp']
                bars_df['bid_price'] = bars_df['close'] - 0.01  # Estimate
                bars_df['bid_size'] = bars_df['volume'] / 2
                bars_df['ask_price'] = bars_df['close'] + 0.01  # Estimate
                bars_df['ask_size'] = bars_df['volume'] / 2

                self._volume_bars_cache = bars_df
                self._tick_cache = bars_df  # No tick data in this mode

            else:
                # Use tick data (requires higher tier or slower rate)
                logger.info(f"Fetching tick data for {ticker} from {start_date} to {end_date}")
                self._tick_cache = self.data_fetcher.fetch_tick_data(
                    ticker, start_date, end_date, include_quotes=True
                )

                if self._tick_cache.empty:
                    logger.warning(f"No tick data available for {ticker}")
                    return pd.DataFrame()

                # Resample to volume bars
                logger.info("Resampling to volume bars...")
                self._volume_bars_cache = resample_to_volume_bars_vectorized(
                    self._tick_cache,
                    volume_threshold=self.volume_bar_threshold
                )

            self._last_fetch_time = datetime.now()

        if self._volume_bars_cache.empty:
            logger.warning("No volume bars created")
            return pd.DataFrame()

        # Calculate OFI (will be estimated from price changes if no quote data)
        logger.info("Calculating OFI...")
        try:
            ofi = calculate_ofi(self._volume_bars_cache)
        except Exception as e:
            logger.warning(f"OFI calculation failed, using price-based estimate: {e}")
            # Fallback: use price change as proxy for OFI
            closes = self._volume_bars_cache['close'].values
            ofi = np.concatenate([[0], np.diff(closes) * 100])

        # Calculate VPIN
        logger.info("Calculating VPIN...")
        vpin = calculate_vpin_for_volume_bars(
            self._volume_bars_cache,
            n_buckets=self.vpin_buckets
        )

        # Prepare features
        logger.info("Preparing HMM features...")
        self._features_cache = prepare_hmm_features(
            self._volume_bars_cache.rename(columns={'close': 'close'}),
            ofi,
            vpin,
            z_score_window=self.z_score_window
        )

        # Add OFI and VPIN raw values to volume bars
        result = self._volume_bars_cache.copy()
        result['ofi'] = ofi
        result['vpin'] = vpin

        logger.info(f"Processed {len(result)} bars with features")

        return result

    def train_model(
        self,
        ticker: str,
        training_days: int = 21,
        save_path: Optional[str] = None,
        auto_save: bool = True
    ) -> Dict[str, Any]:
        """
        Train the HMM model on historical data.

        Args:
            ticker: Stock ticker symbol
            training_days: Number of trading days for training (default 30)
            save_path: Optional path to save the trained model (uses default if not provided)
            auto_save: Automatically save model after training (default True)

        Returns:
            Dict with training results and model info
        """
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=training_days * 1.5)  # Buffer for weekends

        # Fetch and process data
        self.fetch_and_process(
            ticker,
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d'),
            force_refresh=True
        )

        if self._features_cache is None or self._features_cache.empty:
            raise ValueError("No training data available")

        # Train model
        logger.info(f"Training HMM on {len(self._features_cache)} samples...")
        self.detector.fit(self._features_cache)

        # Auto-save model
        if auto_save:
            actual_save_path = save_path or str(self.model_path)
            self.detector.save(actual_save_path)
            logger.info(f"Model auto-saved to {actual_save_path}")

        model_info = self.detector.get_model_info()
        model_info['model_saved_path'] = str(self.model_path) if auto_save else None

        return model_info

    def get_current_regime(
        self,
        ticker: str,
        lookback_days: int = 5
    ) -> Dict[str, Any]:
        """
        Get the current market regime for a ticker.

        Uses tick-based features from TimescaleDB when available for consistency
        with the trained HMM model. Falls back to OHLC-based features from
        Polygon API when database is not available.

        Args:
            ticker: Stock ticker symbol
            lookback_days: Days of data for context (default 5)

        Returns:
            Dict with regime information and indicators
        """
        if not self.detector.is_fitted:
            raise ValueError(
                "Model not trained. Call train_model() first or load a pre-trained model."
            )

        # Get recent data
        end_date_dt = datetime.now()
        start_date_dt = end_date_dt - timedelta(days=lookback_days * 1.5)

        # Try tick-based features from database first
        tick_features = self._compute_tick_features_from_db(
            ticker,
            start_date=start_date_dt.date(),
            end_date=end_date_dt.date(),
            timeframe='1min'
        )

        feature_source = 'ohlc'  # Default to OHLC

        if tick_features is not None and not tick_features.empty:
            # Use tick-based features from database
            logger.info(f"Using tick-based features for {ticker} current regime ({len(tick_features)} bars)")
            features_for_hmm = tick_features
            feature_source = 'tick'

            # Get raw features for indicator values
            pipeline = self._get_feature_pipeline(ticker)
            raw_features = pipeline.compute_features(
                start_date=start_date_dt.date(),
                end_date=end_date_dt.date(),
                timeframe='1min',
                rth_only=True,
                use_cache=True
            )
            result_df = raw_features
            self._features_cache = features_for_hmm
        else:
            # Fall back to OHLC-based features from Polygon
            logger.info(f"Using OHLC-based features for {ticker} current regime (database unavailable)")
            result_df = self.fetch_and_process(
                ticker,
                start_date_dt.strftime('%Y-%m-%d'),
                end_date_dt.strftime('%Y-%m-%d')
            )
            features_for_hmm = self._features_cache

        if result_df.empty or features_for_hmm is None or features_for_hmm.empty:
            return {
                'error': 'No data available',
                'symbol': ticker
            }

        # Get current features (last row)
        current_features = features_for_hmm.iloc[-1].values

        # Get recent history for context
        history = features_for_hmm.iloc[-50:].values if len(features_for_hmm) > 1 else None

        # Predict current regime
        prediction = self.detector.predict_live(current_features, history)

        # Get latest indicators (handle both DataFrame structures)
        latest_bar = result_df.iloc[-1]

        def safe_float(val):
            """Convert to float, replacing NaN/inf with 0."""
            try:
                f = float(val)
                if np.isnan(f) or np.isinf(f):
                    return 0.0
                return f
            except (TypeError, ValueError):
                return 0.0

        # Sanitize state_probs for JSON
        state_probs = prediction['state_probs']
        if isinstance(state_probs, dict):
            state_probs = {k: safe_float(v) for k, v in state_probs.items()}

        # Get timestamp (handle both index-based and column-based)
        if hasattr(result_df.index, 'name') and result_df.index.name == 'time':
            timestamp = str(result_df.index[-1])
        elif 'time' in result_df.columns:
            timestamp = str(latest_bar['time'])
        else:
            timestamp = str(result_df.index[-1])

        return {
            'symbol': ticker,
            'regime': {
                'state': prediction['state'],
                'state_id': prediction['state_id'],
                'confidence': safe_float(prediction['confidence']),
                'state_probs': state_probs
            },
            'indicators': {
                'ofi': safe_float(latest_bar.get('ofi', 0) if hasattr(latest_bar, 'get') else latest_bar['ofi']),
                'vpin': safe_float(latest_bar.get('vpin', 0) if hasattr(latest_bar, 'get') else latest_bar['vpin']),
                'log_return': safe_float(features_for_hmm.iloc[-1]['log_return']),
                'volatility': safe_float(features_for_hmm.iloc[-1]['volatility'])
            },
            'price': {
                'open': safe_float(latest_bar.get('open', 0) if hasattr(latest_bar, 'get') else latest_bar['open']),
                'high': safe_float(latest_bar.get('high', 0) if hasattr(latest_bar, 'get') else latest_bar['high']),
                'low': safe_float(latest_bar.get('low', 0) if hasattr(latest_bar, 'get') else latest_bar['low']),
                'close': safe_float(latest_bar.get('close', 0) if hasattr(latest_bar, 'get') else latest_bar['close']),
                'volume': safe_float(latest_bar.get('volume', 0) if hasattr(latest_bar, 'get') else latest_bar['volume'])
            },
            'timestamp': timestamp,
            'data_bars': len(result_df),
            'feature_source': feature_source
        }

    def get_regime_history(
        self,
        ticker: str,
        days: int = 5,
        timeframe_minutes: int = 1
    ) -> List[Dict[str, Any]]:
        """
        Get historical regime predictions for backtesting/visualization.

        Uses tick-based features from TimescaleDB when available for consistency
        with the trained HMM model. Falls back to OHLC-based features from
        Polygon API when database is not available.

        Args:
            ticker: Stock ticker symbol
            days: Number of days of history
            timeframe_minutes: Timeframe for resampling (1, 5, 15, 30, etc.)

        Returns:
            List of regime predictions with timestamps, resampled to timeframe
        """
        if not self.detector.is_fitted:
            raise ValueError("Model not trained")

        # Fetch data
        end_date_dt = datetime.now()
        start_date_dt = end_date_dt - timedelta(days=days * 1.5)

        # Convert timeframe_minutes to string format for feature pipeline
        timeframe_str = f'{timeframe_minutes}min'

        # Try tick-based features from database first
        tick_features = self._compute_tick_features_from_db(
            ticker,
            start_date=start_date_dt.date(),
            end_date=end_date_dt.date(),
            timeframe=timeframe_str
        )

        if tick_features is not None and not tick_features.empty:
            # Use tick-based features from database
            logger.info(f"Using tick-based features for {ticker} regime history ({len(tick_features)} bars)")
            features_for_hmm = tick_features

            # Get raw features for OFI/VPIN values
            pipeline = self._get_feature_pipeline(ticker)
            raw_features = pipeline.compute_features(
                start_date=start_date_dt.date(),
                end_date=end_date_dt.date(),
                timeframe=timeframe_str,
                rth_only=True,
                use_cache=True
            )

            # Use the raw features DataFrame as result_df
            result_df = raw_features.reset_index()
            result_df = result_df.rename(columns={'time': 'time'})
            if 'time' not in result_df.columns and raw_features.index.name == 'time':
                result_df['time'] = raw_features.index
            self._features_cache = features_for_hmm
        else:
            # Fall back to OHLC-based features from Polygon
            logger.info(f"Using OHLC-based features for {ticker} regime history (database unavailable)")
            result_df = self.fetch_and_process(
                ticker,
                start_date_dt.strftime('%Y-%m-%d'),
                end_date_dt.strftime('%Y-%m-%d')
            )
            features_for_hmm = self._features_cache

        if result_df.empty or features_for_hmm is None or features_for_hmm.empty:
            return []

        # Predict all states
        states = self.detector.predict(features_for_hmm)
        probs = self.detector.predict_proba(features_for_hmm)

        def safe_float(val):
            """Convert to float, replacing NaN/inf with 0."""
            f = float(val)
            if np.isnan(f) or np.isinf(f):
                return 0.0
            return f

        # Build raw history from volume bucket data
        raw_history = []
        for i in range(len(result_df)):
            state_id = int(states[i])
            raw_history.append({
                'timestamp': str(result_df.iloc[i]['time']),
                'state_id': state_id,
                'confidence': safe_float(probs[i].max()),
                'ofi': safe_float(result_df.iloc[i]['ofi']),
                'vpin': safe_float(result_df.iloc[i]['vpin']),
                'close': safe_float(result_df.iloc[i]['close'])
            })

        # ALWAYS resample to align to timeframe boundaries
        # Volume buckets are NOT time-aligned, so we must resample even for 1-minute
        if len(raw_history) > 0:
            # Convert to DataFrame for resampling
            history_df = pd.DataFrame(raw_history)

            # Convert to pandas timeframe string
            timeframe_str = f'{timeframe_minutes}T'

            # Resample to align timestamps
            resampled_df = resample_to_timeframe(history_df, timeframe_str)

            # Convert back to list of dicts
            history = []
            if not resampled_df.empty:
                for _, row in resampled_df.iterrows():
                    state_id = int(row.get('hmm_state', 0))
                    state_label = self.detector.state_labels.get(state_id, f"State_{state_id}")
                    history.append({
                        'timestamp': row['timestamp'],  # Already formatted as ISO string
                        'state': state_label,
                        'state_id': state_id,
                        'ofi': safe_float(row.get('ofi', 0)),
                        'vpin': safe_float(row.get('vpin', 0)),
                        'close': safe_float(row.get('close', 0))
                    })
        else:
            history = []

        return history

    def get_model_info(self) -> Dict[str, Any]:
        """Get current model information."""
        return self.detector.get_model_info()

    def get_model_status(self) -> Dict[str, Any]:
        """
        Get model status including saved model info and database availability.

        Returns:
            Dict with model status information
        """
        model_exists = self.model_path.exists()
        model_info = self.detector.get_model_info()

        # Check database availability
        db_available = self._check_database_available()

        # Get list of symbols with database data
        db_symbols = self._db_symbols if db_available else []

        return {
            'is_fitted': model_info.get('is_fitted', False),
            'model_path': str(self.model_path),
            'model_saved': model_exists,
            'n_states': model_info.get('n_states', 4),
            'state_labels': model_info.get('state_labels', {}),
            'ready_for_inference': model_info.get('is_fitted', False),
            'database_available': db_available,
            'database_symbols': db_symbols,
            'feature_source': 'tick' if db_available else 'ohlc',
            'bar_type': self.bar_type,
            'volume_bar_size': self.volume_bar_threshold
        }


# Singleton instance for easy access
_service_instance: Optional[MarketRegimeService] = None


def get_regime_service() -> MarketRegimeService:
    """Get or create the singleton regime service instance."""
    global _service_instance
    if _service_instance is None:
        _service_instance = MarketRegimeService()
    return _service_instance


def init_regime_service(
    polygon_api_key: Optional[str] = None,
    use_s3: bool = True,
    s3_access_key: Optional[str] = None,
    s3_secret_key: Optional[str] = None,
    s3_endpoint: Optional[str] = None,
    s3_bucket: Optional[str] = None,
    bar_type: Literal['time', 'volume'] = 'time',
    **kwargs
) -> MarketRegimeService:
    """Initialize the singleton regime service with configuration."""
    global _service_instance
    _service_instance = MarketRegimeService(
        polygon_api_key=polygon_api_key,
        use_s3=use_s3,
        s3_access_key=s3_access_key,
        s3_secret_key=s3_secret_key,
        s3_endpoint=s3_endpoint,
        s3_bucket=s3_bucket,
        bar_type=bar_type,
        **kwargs
    )
    return _service_instance


if __name__ == "__main__":
    # Example usage
    import logging
    logging.basicConfig(level=logging.INFO)

    print("=== Market Regime Service Test ===\n")

    # Check for API key
    api_key = os.environ.get('POLYGON_API_KEY')
    if not api_key:
        print("POLYGON_API_KEY not set. Running in demo mode with synthetic data.\n")

        # Demo with synthetic data
        from .regime_detector import RegimeDetector

        # Create synthetic features
        np.random.seed(42)
        n = 200

        features_df = pd.DataFrame({
            'ofi': np.random.randn(n),
            'vpin': np.random.uniform(0.2, 0.8, n),
            'log_return': np.random.randn(n) * 0.002,
            'volatility': np.random.uniform(0.01, 0.03, n)
        })

        # Train detector
        detector = RegimeDetector(n_states=4)
        detector.fit(features_df)

        # Test prediction
        current = features_df.iloc[-1].values
        result = detector.predict_live(current)

        print("Demo prediction result:")
        print(f"  State: {result['state']}")
        print(f"  Confidence: {result['confidence']:.2%}")
        print(f"  Probabilities: {result['state_probs']}")

    else:
        print(f"API key found. Testing with live data...\n")

        # Initialize service
        service = MarketRegimeService(polygon_api_key=api_key)

        # Train on recent data
        print("Training model on QQQ data (last 10 days)...")
        model_info = service.train_model('QQQ', training_days=10)
        print(f"Model trained. State labels: {model_info['state_labels']}")

        # Get current regime
        print("\nGetting current regime...")
        regime = service.get_current_regime('QQQ', lookback_days=2)
        print(f"Current regime: {regime['regime']['state']}")
        print(f"Confidence: {regime['regime']['confidence']:.2%}")
        print(f"OFI: {regime['indicators']['ofi']:.2f}")
        print(f"VPIN: {regime['indicators']['vpin']:.4f}")
