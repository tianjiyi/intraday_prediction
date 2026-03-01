"""
Feature Pipeline for HMM Market Regime Detection

This module provides an end-to-end pipeline for computing tick-level
features from TimescaleDB and preparing them for HMM training.

Usage:
    from market_regime.feature_pipeline import FeaturePipeline

    pipeline = FeaturePipeline(symbol='QQQ')
    features = pipeline.compute_features(
        start_date=date(2025, 12, 1),
        end_date=date(2025, 12, 23),
        timeframe='1min'
    )
    hmm_features = pipeline.normalize_for_hmm(features)
"""

import logging
import os
from datetime import datetime, date, timedelta
from typing import Optional, List, Tuple, Literal
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# Alpaca imports for daily data
try:
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False

from .db_queries import (
    get_trades,
    get_quotes,
    get_trades_with_prevailing_quotes,
    get_available_dates,
    get_date_stats
)
from .features_tick import (
    calculate_ofi_from_quotes,
    calculate_vpin_from_trades,
    aggregate_ofi_to_bars,
    aggregate_ofi_to_volume_bars,
    resample_vpin_to_time_bars,
    normalize_features_for_hmm
)
from .volume_bars import resample_to_volume_bars_vectorized

logger = logging.getLogger(__name__)


class FeaturePipeline:
    """
    End-to-end pipeline for computing tick-level features.

    Handles:
    - Fetching data from TimescaleDB
    - Computing OFI and VPIN
    - Aggregating to time bars
    - Normalizing for HMM
    - Caching results
    """

    def __init__(
        self,
        symbol: str,
        bucket_volume: int = 10000,
        vpin_buckets: int = 50,
        zscore_window: int = 50,
        cache_dir: Optional[Path] = None,
        bar_type: Literal['time', 'volume'] = 'time',
        volume_bar_size: int = 10000
    ):
        """
        Initialize the feature pipeline.

        Args:
            symbol: Trading symbol (e.g., 'QQQ')
            bucket_volume: Volume per VPIN bucket
            vpin_buckets: Rolling window for VPIN
            zscore_window: Window for z-score normalization
            cache_dir: Directory for caching computed features
            bar_type: 'time' for time bars, 'volume' for volume bars
            volume_bar_size: Shares per volume bar (only used when bar_type='volume')
        """
        self.symbol = symbol
        self.bucket_volume = bucket_volume
        self.vpin_buckets = vpin_buckets
        self.zscore_window = zscore_window
        self.bar_type = bar_type
        self.volume_bar_size = volume_bar_size

        if cache_dir is None:
            cache_dir = Path(__file__).parent / 'data' / 'features'
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_available_dates(self) -> List[date]:
        """Get list of dates with data in the database."""
        return get_available_dates(self.symbol)

    def compute_features_for_date(
        self,
        trade_date: date,
        timeframe: str = '1min',
        rth_only: bool = True
    ) -> pd.DataFrame:
        """
        Compute all features for a single date.

        Dispatches to time-bar or volume-bar computation based on bar_type.

        Args:
            trade_date: Date to compute features for
            timeframe: Bar frequency ('1min', '5min', etc.) - ignored for volume bars
            rth_only: If True, filter to Regular Trading Hours (9:30-16:00 ET)

        Returns:
            DataFrame with features indexed by time
        """
        if self.bar_type == 'volume':
            return self._compute_volume_bar_features(trade_date, rth_only)
        else:
            return self._compute_time_bar_features(trade_date, timeframe, rth_only)

    def _compute_time_bar_features(
        self,
        trade_date: date,
        timeframe: str = '1min',
        rth_only: bool = True
    ) -> pd.DataFrame:
        """
        Compute features using time bars.

        Args:
            trade_date: Date to compute features for
            timeframe: Bar frequency ('1min', '5min', '15min', '30min')
            rth_only: If True, filter to Regular Trading Hours (9:30-16:00 ET)

        Returns:
            DataFrame with features indexed by time
        """
        logger.info(f"Computing TIME BAR features for {self.symbol} on {trade_date}...")

        # Define time range
        start_time = datetime.combine(trade_date, datetime.min.time())
        end_time = start_time + timedelta(days=1)

        # Fetch data
        logger.info("Fetching quotes...")
        quotes_df = get_quotes(self.symbol, start_time, end_time)

        logger.info("Fetching trades with prevailing quotes...")
        trades_df = get_trades_with_prevailing_quotes(self.symbol, start_time, end_time)

        if quotes_df.empty or trades_df.empty:
            logger.warning(f"No data found for {self.symbol} on {trade_date}")
            return pd.DataFrame()

        # Filter to RTH if requested
        if rth_only:
            quotes_df = self._filter_rth(quotes_df)
            trades_df = self._filter_rth(trades_df)

            if quotes_df.empty or trades_df.empty:
                logger.warning(f"No RTH data for {self.symbol} on {trade_date}")
                return pd.DataFrame()

        # Calculate OFI
        logger.info("Calculating OFI from quote changes...")
        ofi_series = calculate_ofi_from_quotes(quotes_df)
        ofi_bars = aggregate_ofi_to_bars(quotes_df, ofi_series, freq=timeframe)

        # Calculate VPIN
        logger.info("Calculating VPIN from trades...")
        vpin_df = calculate_vpin_from_trades(
            trades_df,
            bucket_volume=self.bucket_volume,
            n_buckets=self.vpin_buckets,
            classification_method='quote_rule'
        )
        vpin_bars = resample_vpin_to_time_bars(vpin_df, freq=timeframe)

        # Aggregate trades to bars
        trades_bars = trades_df.set_index('time').resample(timeframe).agg({
            'price': ['first', 'max', 'min', 'last'],
            'size': 'sum'
        })
        trades_bars.columns = ['open', 'high', 'low', 'close', 'volume']
        trades_bars['return'] = np.log(trades_bars['close'] / trades_bars['close'].shift(1))
        trades_bars['trade_count'] = trades_df.set_index('time').resample(timeframe).size()

        # Aggregate quotes to bars
        quotes_bars = quotes_df.set_index('time').resample(timeframe).agg({
            'bid_price': 'mean',
            'ask_price': 'mean',
            'bid_size': 'mean',
            'ask_size': 'mean'
        })
        quotes_bars['spread'] = quotes_bars['ask_price'] - quotes_bars['bid_price']
        quotes_bars['mid_price'] = (quotes_bars['bid_price'] + quotes_bars['ask_price']) / 2

        # Combine all features
        features = pd.DataFrame(index=trades_bars.index)
        features['ofi'] = ofi_bars['ofi']
        features['vpin'] = vpin_bars['vpin'].ffill()  # Forward fill VPIN
        features['volume'] = trades_bars['volume']
        features['trade_count'] = trades_bars['trade_count']
        features['return'] = trades_bars['return']
        features['spread'] = quotes_bars['spread']
        features['close'] = trades_bars['close']
        features['high'] = trades_bars['high']
        features['low'] = trades_bars['low']
        features['open'] = trades_bars['open']
        features['mid_price'] = quotes_bars['mid_price']

        # Add derived features
        features['volatility'] = features['return'].rolling(window=20, min_periods=5).std()
        features['log_volume'] = np.log1p(features['volume'])

        features = features.dropna()

        logger.info(f"Generated {len(features)} feature bars for {trade_date}")

        return features

    def _compute_volume_bar_features(
        self,
        trade_date: date,
        rth_only: bool = True
    ) -> pd.DataFrame:
        """
        Compute features using volume bars.

        Volume bars normalize for trading activity, making OFI and VPIN
        calculations more stable across different market conditions.

        Args:
            trade_date: Date to compute features for
            rth_only: If True, filter to Regular Trading Hours (9:30-16:00 ET)

        Returns:
            DataFrame with features indexed by time (bar completion time)
        """
        logger.info(f"Computing VOLUME BAR features for {self.symbol} on {trade_date}...")
        logger.info(f"  Volume bar size: {self.volume_bar_size:,} shares")

        # Define time range
        start_time = datetime.combine(trade_date, datetime.min.time())
        end_time = start_time + timedelta(days=1)

        # Fetch data
        logger.info("Fetching quotes...")
        quotes_df = get_quotes(self.symbol, start_time, end_time)

        logger.info("Fetching trades with prevailing quotes...")
        trades_df = get_trades_with_prevailing_quotes(self.symbol, start_time, end_time)

        if quotes_df.empty or trades_df.empty:
            logger.warning(f"No data found for {self.symbol} on {trade_date}")
            return pd.DataFrame()

        # Filter to RTH if requested
        if rth_only:
            quotes_df = self._filter_rth(quotes_df)
            trades_df = self._filter_rth(trades_df)

            if quotes_df.empty or trades_df.empty:
                logger.warning(f"No RTH data for {self.symbol} on {trade_date}")
                return pd.DataFrame()

        logger.info(f"  Trades: {len(trades_df):,}, Quotes: {len(quotes_df):,}")

        # Create volume bars from trades
        logger.info("Creating volume bars...")
        tick_df = pd.DataFrame({
            'timestamp': trades_df['time'],
            'price': trades_df['price'],
            'volume': trades_df['size'],
            'bid_price': trades_df['bid_price'],
            'bid_size': trades_df.get('bid_size', 100),  # Default if not available
            'ask_price': trades_df['ask_price'],
            'ask_size': trades_df.get('ask_size', 100)
        })

        volume_bars = resample_to_volume_bars_vectorized(
            tick_df,
            volume_threshold=self.volume_bar_size
        )

        if volume_bars.empty:
            logger.warning(f"No volume bars created for {self.symbol} on {trade_date}")
            return pd.DataFrame()

        logger.info(f"  Created {len(volume_bars)} volume bars")

        # Calculate OFI and aggregate to volume bars
        logger.info("Calculating OFI from quote changes...")
        ofi_series = calculate_ofi_from_quotes(quotes_df)
        ofi_volume_bars = aggregate_ofi_to_volume_bars(
            quotes_df, trades_df, ofi_series,
            bucket_volume=self.volume_bar_size
        )

        # Calculate VPIN (align buckets with volume bars)
        logger.info("Calculating VPIN from trades...")
        vpin_df = calculate_vpin_from_trades(
            trades_df,
            bucket_volume=self.volume_bar_size,  # Same as volume bar size!
            n_buckets=self.vpin_buckets,
            classification_method='bvc'  # Use continuous BVC
        )

        # Combine features
        # Volume bars have 'time' column, OFI/VPIN are indexed by time
        # Normalize timezone to avoid "Cannot compare dtypes" errors
        volume_bar_times = volume_bars['time'].copy()
        if volume_bar_times.dt.tz is None:
            # Make timezone-naive for consistency
            pass
        else:
            # Convert to UTC then remove timezone info
            volume_bar_times = volume_bar_times.dt.tz_convert('UTC').dt.tz_localize(None)

        features = pd.DataFrame(index=volume_bar_times)
        features.index.name = 'time'

        # OFI - match by volume bar completion time
        if not ofi_volume_bars.empty:
            # Normalize OFI index timezone
            ofi_idx = ofi_volume_bars.index
            if ofi_idx.tz is not None:
                ofi_volume_bars = ofi_volume_bars.copy()
                ofi_volume_bars.index = ofi_idx.tz_convert('UTC').tz_localize(None)

            # Merge OFI values with volume bars using time index
            features['ofi'] = ofi_volume_bars['ofi'].reindex(
                features.index, method='ffill'
            )
        else:
            features['ofi'] = 0

        # VPIN - already volume-bucket aligned
        if not vpin_df.empty:
            # VPIN has 'time' as index (not column)
            vpin_idx = vpin_df.index
            if hasattr(vpin_idx, 'tz') and vpin_idx.tz is not None:
                vpin_idx = vpin_idx.tz_convert('UTC').tz_localize(None)
            vpin_series = pd.Series(vpin_df['vpin'].values, index=vpin_idx)

            # Remove duplicate indices (take last value for each timestamp)
            if vpin_series.index.has_duplicates:
                vpin_series = vpin_series[~vpin_series.index.duplicated(keep='last')]

            features['vpin'] = vpin_series.reindex(features.index, method='ffill')
        else:
            features['vpin'] = 0.5

        # OHLCV from volume bars
        features['open'] = volume_bars['open'].values
        features['high'] = volume_bars['high'].values
        features['low'] = volume_bars['low'].values
        features['close'] = volume_bars['close'].values
        features['volume'] = volume_bars['volume'].values

        # Returns and volatility
        features['return'] = np.log(features['close'] / features['close'].shift(1))
        features['volatility'] = features['return'].rolling(window=20, min_periods=5).std()
        features['log_volume'] = np.log1p(features['volume'])

        # Bar duration - useful additional feature for volume bars
        # How long each bar took to fill (shorter = higher activity)
        time_diffs = features.index.to_series().diff().dt.total_seconds()
        features['bar_duration_seconds'] = time_diffs

        features = features.dropna()

        logger.info(f"Generated {len(features)} VOLUME BAR features for {trade_date}")

        # Log bar duration statistics
        if len(features) > 0 and 'bar_duration_seconds' in features.columns:
            avg_duration = features['bar_duration_seconds'].mean()
            logger.info(f"  Average bar duration: {avg_duration:.1f} seconds")

        return features

    def compute_features(
        self,
        start_date: date,
        end_date: date,
        timeframe: str = '1min',
        rth_only: bool = True,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Compute features for a date range.

        Args:
            start_date: Start date (inclusive)
            end_date: End date (inclusive)
            timeframe: Bar frequency
            rth_only: Filter to RTH only
            use_cache: Use cached results if available

        Returns:
            DataFrame with concatenated features for all dates
        """
        available_dates = self.get_available_dates()

        # Filter to requested range
        dates_to_process = [
            d for d in available_dates
            if start_date <= d <= end_date
        ]

        if not dates_to_process:
            logger.warning(f"No data available for {self.symbol} between {start_date} and {end_date}")
            return pd.DataFrame()

        logger.info(f"Processing {len(dates_to_process)} dates for {self.symbol}")

        all_features = []

        for trade_date in dates_to_process:
            # Check cache
            cache_file = self._get_cache_path(trade_date, timeframe)

            if use_cache and cache_file.exists():
                logger.info(f"Loading cached features for {trade_date}")
                features = pd.read_parquet(cache_file)
            else:
                features = self.compute_features_for_date(
                    trade_date,
                    timeframe=timeframe,
                    rth_only=rth_only
                )

                # Cache the result
                if not features.empty:
                    features.to_parquet(cache_file)
                    logger.info(f"Cached features for {trade_date}")

            if not features.empty:
                all_features.append(features)

        if not all_features:
            return pd.DataFrame()

        combined = pd.concat(all_features)
        combined = combined.sort_index()

        logger.info(f"Total: {len(combined)} feature bars from {start_date} to {end_date}")

        return combined

    def normalize_for_hmm(
        self,
        features_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Normalize features for HMM training.

        Args:
            features_df: Raw features from compute_features()

        Returns:
            DataFrame with normalized features ready for HMM
        """
        return normalize_features_for_hmm(features_df, zscore_window=self.zscore_window)

    def get_hmm_training_data(
        self,
        start_date: date,
        end_date: date,
        timeframe: str = '1min',
        rth_only: bool = True
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Get training data ready for HMM.

        Args:
            start_date: Start date
            end_date: End date
            timeframe: Bar frequency
            rth_only: Filter to RTH

        Returns:
            Tuple of (feature_matrix, raw_features_df)
        """
        # Compute raw features
        raw_features = self.compute_features(
            start_date=start_date,
            end_date=end_date,
            timeframe=timeframe,
            rth_only=rth_only
        )

        if raw_features.empty:
            return np.array([]), pd.DataFrame()

        # Normalize for HMM
        hmm_features = self.normalize_for_hmm(raw_features)

        # Convert to numpy array
        feature_matrix = hmm_features.values

        logger.info(f"HMM training data: {feature_matrix.shape}")

        return feature_matrix, raw_features

    def _filter_rth(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter DataFrame to Regular Trading Hours (9:30-16:00 ET)."""
        if df.empty:
            return df

        # Ensure timezone awareness
        if df['time'].dt.tz is None:
            df = df.copy()
            df['time'] = df['time'].dt.tz_localize('UTC')

        # Convert to Eastern
        df_et = df.copy()
        df_et['time_et'] = df_et['time'].dt.tz_convert('America/New_York')

        # Filter to RTH (9:30-16:00)
        rth_mask = (
            (df_et['time_et'].dt.time >= pd.Timestamp('09:30').time()) &
            (df_et['time_et'].dt.time < pd.Timestamp('16:00').time())
        )

        return df[rth_mask]

    def _get_cache_path(self, trade_date: date, timeframe: str) -> Path:
        """Get cache file path for a date."""
        if self.bar_type == 'volume':
            return self.cache_dir / f"{self.symbol}_{trade_date}_vol{self.volume_bar_size}.parquet"
        else:
            return self.cache_dir / f"{self.symbol}_{trade_date}_{timeframe}.parquet"

    # =========================================================================
    # Daily-Level Features for Income Defender HMM
    # =========================================================================

    def _get_alpaca_client(self) -> Optional[StockHistoricalDataClient]:
        """Get Alpaca client from config."""
        if not ALPACA_AVAILABLE:
            logger.warning("Alpaca SDK not available")
            return None

        # Load config
        config_path = Path(__file__).parent.parent.parent / 'config.yaml'
        if not config_path.exists():
            config_path = Path(__file__).parent.parent / 'config.yaml'

        if config_path.exists():
            with open(config_path) as f:
                config = yaml.safe_load(f)
            api_key = config.get('ALPACA_KEY_ID') or os.getenv('ALPACA_KEY_ID')
            secret_key = config.get('ALPACA_SECRET_KEY') or os.getenv('ALPACA_SECRET_KEY')
        else:
            api_key = os.getenv('ALPACA_KEY_ID')
            secret_key = os.getenv('ALPACA_SECRET_KEY')

        if not api_key or not secret_key:
            logger.warning("Alpaca credentials not found")
            return None

        return StockHistoricalDataClient(api_key, secret_key)

    def get_daily_closes(self, end_date: date, lookback_days: int = 35) -> pd.Series:
        """
        Fetch daily close prices from Alpaca for SMA calculation.

        Args:
            end_date: End date (inclusive)
            lookback_days: Number of calendar days to look back (default 35 for SMA21 + buffer)

        Returns:
            Series with daily close prices indexed by date
        """
        client = self._get_alpaca_client()

        if client is None:
            logger.warning("Cannot fetch daily data - Alpaca client not available")
            return pd.Series(dtype=float)

        start_date = end_date - timedelta(days=lookback_days + 10)  # Buffer for weekends/holidays

        try:
            request = StockBarsRequest(
                symbol_or_symbols=self.symbol,
                timeframe=TimeFrame.Day,
                start=datetime.combine(start_date, datetime.min.time()),
                end=datetime.combine(end_date, datetime.max.time())
            )

            bars = client.get_stock_bars(request)
            df = bars.df

            if df.empty:
                logger.warning(f"No daily bars returned from Alpaca for {self.symbol}")
                return pd.Series(dtype=float)

            # Handle multi-index (symbol, timestamp)
            if isinstance(df.index, pd.MultiIndex):
                df = df.reset_index()
                df = df[df['symbol'] == self.symbol]
                df = df.set_index('timestamp')

            # Extract close prices with date index
            daily_closes = df['close'].copy()
            daily_closes.index = pd.to_datetime(daily_closes.index).date

            logger.info(f"Fetched {len(daily_closes)} daily closes from Alpaca for {self.symbol}")
            logger.debug(f"  Date range: {daily_closes.index.min()} to {daily_closes.index.max()}")

            return daily_closes

        except Exception as e:
            logger.error(f"Error fetching daily data from Alpaca: {e}")
            return pd.Series(dtype=float)

    def compute_dist_sma21(self, trade_date: date) -> float:
        """
        Compute the deviation from 21-day SMA for a given date.

        Formula: (Close - SMA21) / SMA21

        Args:
            trade_date: The date to compute dist_sma21 for

        Returns:
            Float representing the percentage deviation from SMA21
            (e.g., 0.02 = 2% above SMA, -0.01 = 1% below SMA)
        """
        daily_closes = self.get_daily_closes(trade_date, lookback_days=35)

        if len(daily_closes) < 21:
            logger.warning(f"Insufficient daily data for SMA21: {len(daily_closes)} days")
            return 0.0

        # Compute SMA21
        sma_21 = daily_closes.rolling(21).mean()

        if trade_date not in daily_closes.index:
            logger.warning(f"Date {trade_date} not in daily closes")
            return 0.0

        current_close = daily_closes[trade_date]
        current_sma = sma_21[trade_date]

        if pd.isna(current_sma) or current_sma == 0:
            return 0.0

        dist_sma21 = (current_close - current_sma) / current_sma

        logger.debug(f"{trade_date}: Close={current_close:.2f}, SMA21={current_sma:.2f}, Dist={dist_sma21:.4f}")

        return float(dist_sma21)

    def compute_vrp(self, features_df: pd.DataFrame, window: int = 20) -> pd.Series:
        """
        Compute VRP (Volatility Risk Premium) proxy as percentage.

        Uses volatility momentum: VRP = (MA20_Vol - Current_Vol) / MA20_Vol * 100
        - Positive = volatility declining (safe, premium reasonable)
        - Negative = volatility spiking (dangerous)

        Args:
            features_df: DataFrame with 'volatility' column
            window: Rolling window for MA

        Returns:
            Series with VRP proxy values (as percentage)
        """
        if 'volatility' not in features_df.columns:
            logger.warning("No 'volatility' column in features_df")
            return pd.Series(0.0, index=features_df.index)

        vol = features_df['volatility']
        vol_ma = vol.rolling(window, min_periods=5).mean()

        # VRP as percentage of MA
        # Positive = vol below MA (declining), Negative = vol above MA (spiking)
        vrp = (vol_ma - vol) / vol_ma * 100

        # Handle edge cases
        vrp = vrp.fillna(0)
        vrp = vrp.clip(-100, 100)  # Cap extreme values

        return vrp

    def compute_quote_imbalance_batch(
        self,
        features_df: pd.DataFrame,
        start_date: date,
        end_date: date,
        rth_only: bool = True
    ) -> pd.Series:
        """
        Optimized quote imbalance calculation using batch fetch + merge_asof.

        Fixes N+1 query problem by:
        1. Fetching ALL quotes in ONE batch query
        2. Computing imbalance with vectorized operations
        3. Aligning to volume bars using merge_asof (O(N) complexity)

        Quote imbalance = (bid_size - ask_size) / (bid_size + ask_size)
        - Positive = more bids than asks (buying pressure)
        - Negative = more asks than bids (selling pressure, WARNING!)

        This is a LEADING indicator - market makers adjust quotes BEFORE price moves.

        Args:
            features_df: DataFrame with volume bar data (index = bar completion times)
            start_date: Start date for batch query
            end_date: End date for batch query
            rth_only: Filter to RTH hours

        Returns:
            Series with quote imbalance values aligned to volume bars
        """
        import time as time_module
        from .db_queries import get_quotes

        if features_df.empty:
            return pd.Series(dtype=float)

        t0 = time_module.time()

        # 1. Batch fetch ALL quotes for the date range (ONE query!)
        start_time = datetime.combine(start_date, datetime.min.time())
        end_time = datetime.combine(end_date, datetime.max.time())

        logger.info(f"Batch fetching quotes from {start_date} to {end_date}...")
        quotes_df = get_quotes(self.symbol, start_time, end_time)

        if quotes_df.empty:
            logger.warning(f"No quotes data for {self.symbol}")
            return pd.Series(0.0, index=features_df.index, name='quote_imbalance')

        # Filter to RTH if requested
        if rth_only:
            quotes_df = self._filter_rth(quotes_df)
            if quotes_df.empty:
                return pd.Series(0.0, index=features_df.index, name='quote_imbalance')

        fetch_time = time_module.time() - t0
        logger.info(f"Fetched {len(quotes_df):,} quotes in {fetch_time:.2f}s. Computing features...")

        # 2. Vectorized calculation with EMA smoothing at tick level
        quotes_df = quotes_df.copy()
        bid_smooth = quotes_df['bid_size'].ewm(span=20).mean()
        ask_smooth = quotes_df['ask_size'].ewm(span=20).mean()
        total_size = bid_smooth + ask_smooth

        # Compute imbalance with safe division
        quotes_df['qib'] = np.where(
            total_size > 0,
            (bid_smooth - ask_smooth) / total_size,
            0.0
        )

        # Ensure timezone consistency for merge_asof
        if quotes_df['time'].dt.tz is not None:
            quotes_df['time'] = quotes_df['time'].dt.tz_convert('UTC').dt.tz_localize(None)

        quotes_df = quotes_df.sort_values('time')

        # 3. Align to volume bars using merge_asof (O(N) complexity!)
        logger.info("Aligning quotes to volume bars with merge_asof...")

        # Prepare bars DataFrame for merge
        bars_for_merge = pd.DataFrame({'bar_time': features_df.index})
        bars_for_merge = bars_for_merge.sort_values('bar_time')

        # The magic: merge_asof finds the most recent quote for each bar
        merged = pd.merge_asof(
            bars_for_merge,
            quotes_df[['time', 'qib']],
            left_on='bar_time',
            right_on='time',
            direction='backward',  # Find quote just before or at bar time
            tolerance=pd.Timedelta('5m')  # Max 5 min gap tolerance
        )

        # Fill NaN with 0 (no quotes found within tolerance)
        merged['qib'] = merged['qib'].fillna(0.0)
        # Second smoothing pass: EMA on bar-level values to remove barcode noise
        merged['qib'] = merged['qib'].ewm(span=10, min_periods=1).mean()

        # Create result series aligned to original index
        result = pd.Series(
            merged['qib'].values,
            index=features_df.index,
            name='quote_imbalance'
        )

        total_time = time_module.time() - t0
        logger.info(f"Quote imbalance computed in {total_time:.2f}s total")

        return result

    def compute_income_defender_features(
        self,
        start_date: date,
        end_date: date,
        timeframe: str = 'volume',
        rth_only: bool = True,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Compute features for the Income Defender HMM model.

        Features (4-feature model):
        - ofi: Order Flow Imbalance (EMA smoothed)
        - vpin_rank: VPIN percentile rank (0-1)
        - vrp: Volatility Risk Premium proxy
        - quote_imbalance: Quote-level bid/ask imbalance (LEADING indicator)

        Args:
            start_date: Start date
            end_date: End date
            timeframe: Bar timeframe
            rth_only: Filter to RTH
            use_cache: Use cached base features

        Returns:
            DataFrame with Income Defender features
        """
        # Get base features
        features = self.compute_features(
            start_date=start_date,
            end_date=end_date,
            timeframe=timeframe,
            rth_only=rth_only,
            use_cache=use_cache
        )

        if features.empty:
            return features

        # Add date column for daily grouping
        features['date'] = features.index.date

        # Feature A: OFI (EMA smoothed)
        features['ofi_smooth'] = features['ofi'].ewm(span=3, min_periods=1).mean()

        # Feature B: VPIN Rank (0-1 percentile)
        features['vpin_rank'] = features['vpin'].rank(pct=True)

        # Feature C: VRP (volatility momentum)
        features['vrp'] = self.compute_vrp(features, window=20)

        # Feature D: Quote Imbalance (LEADING indicator from order book)
        # Use optimized batch method (ONE query, vectorized, merge_asof)
        logger.info("Computing quote imbalance feature (batch mode)...")
        features['quote_imbalance'] = self.compute_quote_imbalance_batch(
            features_df=features,
            start_date=start_date,
            end_date=end_date,
            rth_only=rth_only
        )

        logger.info(f"Income Defender features computed: {len(features)} bars")
        logger.info(f"  vrp range: [{features['vrp'].min():.4f}, {features['vrp'].max():.4f}]")
        logger.info(f"  quote_imbalance range: [{features['quote_imbalance'].min():.4f}, {features['quote_imbalance'].max():.4f}]")

        return features

    def clear_cache(self) -> int:
        """Clear all cached feature files. Returns number of files deleted."""
        count = 0
        for f in self.cache_dir.glob(f"{self.symbol}_*.parquet"):
            f.unlink()
            count += 1
        logger.info(f"Cleared {count} cached feature files")
        return count


def compare_tick_vs_ohlc_features(
    symbol: str,
    trade_date: date,
    timeframe: str = '1min'
) -> pd.DataFrame:
    """
    Compare tick-based features with OHLC-based features.

    This helps validate that tick-level features are working correctly.

    Args:
        symbol: Trading symbol
        trade_date: Date to compare
        timeframe: Bar frequency

    Returns:
        DataFrame with both feature sets for comparison
    """
    pipeline = FeaturePipeline(symbol)
    tick_features = pipeline.compute_features_for_date(trade_date, timeframe)

    if tick_features.empty:
        logger.warning("No tick features computed")
        return pd.DataFrame()

    # Compare correlation, stats
    comparison = pd.DataFrame({
        'tick_ofi': tick_features['ofi'],
        'tick_vpin': tick_features['vpin'],
        'tick_volume': tick_features['volume'],
        'close': tick_features['close']
    })

    return comparison


# Quick test
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s [%(name)s] %(message)s'
    )

    from datetime import date

    # Test pipeline
    print("Testing FeaturePipeline...")
    pipeline = FeaturePipeline(symbol='QQQ', bucket_volume=10000)

    # Check available dates
    dates = pipeline.get_available_dates()
    print(f"Available dates: {len(dates)}")
    if dates:
        print(f"  First: {dates[0]}, Last: {dates[-1]}")

    # Compute features for one day
    if dates:
        test_date = dates[-1]  # Most recent
        print(f"\nComputing features for {test_date}...")

        features = pipeline.compute_features_for_date(
            test_date,
            timeframe='1min',
            rth_only=True
        )

        if not features.empty:
            print(f"Features shape: {features.shape}")
            print(f"\nFeature summary:")
            print(features.describe())

            # Normalize for HMM
            print(f"\nNormalizing for HMM...")
            hmm_features = pipeline.normalize_for_hmm(features)
            print(f"HMM features shape: {hmm_features.shape}")
            print(hmm_features.describe())
        else:
            print("No features generated")
