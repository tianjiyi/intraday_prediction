#!/usr/bin/env python3
"""
Prediction Service - Connects to Kronos model for QQQ predictions
"""

import os
import sys
import json
import yaml
import logging
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import pytz
import warnings
warnings.filterwarnings('ignore')

# Add parent directory and Kronos to path
sys.path.append("..")
sys.path.append("../Kronos")

from alpaca.data.historical import StockHistoricalDataClient, CryptoHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from model import Kronos, KronosTokenizer, KronosPredictor

# Setup logging with timestamps (module-level)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

class PredictionService:
    """Service for generating QQQ predictions using Kronos model"""
    
    def __init__(self, config_path="../config.yaml"):
        """Initialize the prediction service"""
        self.config = self._load_config(config_path)
        self.config_dir = os.path.dirname(os.path.abspath(config_path))  # Store config directory
        self.symbol = self.config['symbol']
        self.timeframe = TimeFrame.Minute  # Default timeframe
        
        # RTH configuration
        self.rth_only = self.config.get('data', {}).get('rth_only', True)
        self.timezone = pytz.timezone('US/Eastern')  # Market timezone
        
        # Initialize Alpaca clients
        self._init_alpaca()
        
        # Initialize Kronos model
        self._init_model()
        
        # Cache for latest data
        self.latest_historical = None
        self.latest_prediction = None
        self.last_update = None
        
        logger.info(f"RTH Only mode: {'Enabled' if self.rth_only else 'Disabled'}")
    
    def _is_crypto_symbol(self, symbol):
        """Determine if symbol is cryptocurrency based on format"""
        return '/' in symbol  # Crypto symbols contain '/' like 'BTC/USD'
    
    def _get_asset_type(self, symbol):
        """Get asset type string for display"""
        if self._is_crypto_symbol(symbol):
            return "crypto"
        return "stock"
    
    def _get_display_name(self, symbol):
        """Get friendly display name for symbol"""
        crypto_names = {
            'BTC/USD': 'Bitcoin',
            'ETH/USD': 'Ethereum',
            'LTC/USD': 'Litecoin',
            'DOGE/USD': 'Dogecoin'
        }
        
        if symbol in crypto_names:
            return crypto_names[symbol]
        return symbol  # Return symbol as-is for stocks
        
    def _load_config(self, config_path):
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _init_alpaca(self):
        """Initialize Alpaca API clients"""
        # Stock client (requires API keys)
        self.stock_client = StockHistoricalDataClient(
            api_key=self.config['ALPACA_KEY_ID'],
            secret_key=self.config['ALPACA_SECRET_KEY']
        )
        
        # Crypto client (no API keys required, but using them increases rate limits)
        self.crypto_client = CryptoHistoricalDataClient(
            api_key=self.config.get('ALPACA_KEY_ID'),
            secret_key=self.config.get('ALPACA_SECRET_KEY')
        )
        
        logger.info("Alpaca clients initialized (stock and crypto)")
    
    def _init_model(self):
        """Initialize Kronos model and tokenizer"""
        try:
            # Convert relative paths to absolute paths for local models
            # Relative paths are resolved relative to config.yaml location
            tokenizer_path = self.config['model']['tokenizer']
            if tokenizer_path.startswith('./') or tokenizer_path.startswith('../'):
                tokenizer_path = os.path.abspath(os.path.join(self.config_dir, tokenizer_path))

            checkpoint_path = self.config['model']['checkpoint']
            if checkpoint_path.startswith('./') or checkpoint_path.startswith('../'):
                checkpoint_path = os.path.abspath(os.path.join(self.config_dir, checkpoint_path))

            logger.info(f"Loading tokenizer from: {tokenizer_path}")
            logger.info(f"Loading model from: {checkpoint_path}")

            # Load tokenizer from pretrained (HuggingFace or local)
            self.tokenizer = KronosTokenizer.from_pretrained(tokenizer_path)

            # Load model from pretrained (HuggingFace or local)
            self.model = Kronos.from_pretrained(checkpoint_path)
            
            # Create predictor
            self.predictor = KronosPredictor(
                model=self.model,
                tokenizer=self.tokenizer,
                device=self.config['model']['device'],
                max_context=self.config['model']['max_context']
            )
            
            logger.info(f"Kronos model loaded: {self.config['model']['checkpoint']}")
        except Exception as e:
            logger.error(f"Failed to initialize model: {e}")
            raise
    
    def _get_timeframe_minutes(self):
        """Extract timeframe in minutes from self.timeframe"""
        # Check by string representation for Day/Week (Alpaca SDK comparison workaround)
        tf_str = str(self.timeframe).lower()
        if 'day' in tf_str:
            return 1440  # 24 * 60
        elif 'week' in tf_str:
            return 10080  # 7 * 24 * 60
        elif self.timeframe == TimeFrame.Minute:
            return 1
        elif hasattr(self.timeframe, 'amount'):
            return self.timeframe.amount
        else:
            return 1  # Default to 1 minute

    def _calculate_days_to_fetch(self, timeframe_minutes, target_bars=None):
        """
        Calculate how many days to fetch to get target number of bars

        Args:
            timeframe_minutes: Timeframe in minutes (1, 5, 15, 30, 1440=Day, 10080=Week)
            target_bars: Target number of bars to fetch (default: from config or 350)

        Returns:
            Number of days to fetch
        """
        # Get target bars from config or use default
        if target_bars is None:
            target_bars = self.config.get('data', {}).get('target_historical_bars', 350)

        # Handle Day and Week timeframes differently
        if timeframe_minutes == 1440:  # Daily
            # 1 bar per trading day, ~252 trading days per year
            # Need ~350 trading days = ~500 calendar days (~1.4 years)
            days_needed = target_bars  # 1 bar per day
            days_with_buffer = int(days_needed * 1.5) + 1  # Account for weekends/holidays
            days_to_fetch = min(days_with_buffer, 730)  # Cap at 2 years
            logger.info(f"Timeframe: Day, Target bars: {target_bars}, "
                       f"Days to fetch: {days_to_fetch}")
            return days_to_fetch

        elif timeframe_minutes == 10080:  # Weekly
            # 1 bar per week, 52 weeks per year
            # Need ~350 weeks = ~2450 calendar days (~7 years)
            weeks_needed = target_bars
            days_to_fetch = min(weeks_needed * 7, 2555)  # Cap at 7 years
            logger.info(f"Timeframe: Week, Target bars: {target_bars}, "
                       f"Days to fetch: {days_to_fetch}")
            return days_to_fetch

        # Intraday timeframes
        # RTH trading hours: 6.5 hours = 390 minutes per day
        rth_minutes_per_day = 390
        bars_per_day = rth_minutes_per_day / timeframe_minutes
        days_needed = target_bars / bars_per_day

        # Round up and add buffer for weekends/holidays (1.5x multiplier)
        days_with_buffer = int(days_needed * 1.5) + 1

        # Cap at reasonable maximum (60 days to avoid excessive API calls)
        days_to_fetch = min(days_with_buffer, 60)

        logger.info(f"Timeframe: {timeframe_minutes}min, Target bars: {target_bars}, "
                   f"Bars/day: {bars_per_day:.1f}, Days needed: {days_needed:.1f}, "
                   f"Days to fetch (with buffer): {days_to_fetch}")

        return days_to_fetch

    def _get_market_aware_end_time(self, timezone):
        """
        Get appropriate end time for data fetching.
        - During market hours: use current time
        - Outside market hours: use last market close (4:00 PM ET)
        - Weekends: use previous Friday 4:00 PM ET

        Args:
            timezone: pytz timezone object (US/Eastern)

        Returns:
            Timezone-aware datetime representing the appropriate end time
        """
        now = datetime.now(timezone)

        # Market hours: Mon-Fri 9:30 AM - 4:00 PM ET
        is_weekend = now.weekday() >= 5  # Sat=5, Sun=6

        if is_weekend:
            # Go back to Friday
            days_back = now.weekday() - 4  # Sat=1 day, Sun=2 days
            last_friday = now - timedelta(days=days_back)
            end_time = last_friday.replace(hour=16, minute=0, second=0, microsecond=0)
            logger.info(f"Weekend detected - using last Friday market close: {end_time.strftime('%Y-%m-%d %H:%M %Z')}")
            return end_time

        # Weekday - check if market is open
        market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)

        if now < market_open:
            # Before market opens - use previous day's close
            prev_day = now - timedelta(days=1)
            # Check if previous day was weekend
            if prev_day.weekday() >= 5:
                days_back = prev_day.weekday() - 4
                prev_day = prev_day - timedelta(days=days_back)
            end_time = prev_day.replace(hour=16, minute=0, second=0, microsecond=0)
            logger.info(f"Before market open - using previous close: {end_time.strftime('%Y-%m-%d %H:%M %Z')}")
            return end_time
        elif now > market_close:
            # After market closes - use today's close
            logger.info(f"After market close - using today's close: {market_close.strftime('%Y-%m-%d %H:%M %Z')}")
            return market_close
        else:
            # During market hours - use current time
            logger.info(f"Market open - using current time: {now.strftime('%Y-%m-%d %H:%M %Z')}")
            return now

    def update_settings(self, symbol=None, timeframe_minutes=None):
        """Update service settings"""
        if symbol:
            self.symbol = symbol
            logger.info(f"Updated symbol to: {symbol}")

        if timeframe_minutes:
            # Convert minutes to TimeFrame
            timeframe_map = {
                1: TimeFrame.Minute,
                5: TimeFrame(5, TimeFrameUnit.Minute),
                15: TimeFrame(15, TimeFrameUnit.Minute),
                30: TimeFrame(30, TimeFrameUnit.Minute),
                1440: TimeFrame.Day,      # 1 day = 1440 minutes
                10080: TimeFrame.Week     # 1 week = 10080 minutes
            }
            self.timeframe = timeframe_map.get(int(timeframe_minutes), TimeFrame.Minute)

            # Log appropriate message based on timeframe
            tf_minutes = int(timeframe_minutes)
            if tf_minutes == 1440:
                logger.info("Updated timeframe to: Day")
            elif tf_minutes == 10080:
                logger.info("Updated timeframe to: Week")
            else:
                logger.info(f"Updated timeframe to: {timeframe_minutes} minutes")

        # Clear cached data when settings change
        self.latest_historical = None
        self.latest_prediction = None
        self.last_update = None
    
    def fetch_historical_data(self, days=None):
        """Fetch historical data from Alpaca

        Args:
            days: Number of days to fetch. If None, automatically calculates based on timeframe
                  to achieve target bar count (300-400 bars)
        """
        try:
            # If days not specified, calculate based on timeframe
            if days is None:
                timeframe_minutes = self._get_timeframe_minutes()
                days = self._calculate_days_to_fetch(timeframe_minutes)
                # Log with appropriate timeframe name
                if timeframe_minutes == 1440:
                    tf_name = "Day"
                elif timeframe_minutes == 10080:
                    tf_name = "Week"
                else:
                    tf_name = f"{timeframe_minutes}min"
                logger.info(f"Auto-calculated {days} days to fetch for {tf_name} timeframe")

            # Use timezone-aware datetime for proper market data fetching
            # Alpaca expects UTC or market timezone (Eastern)
            eastern = pytz.timezone('US/Eastern')
            end_time = self._get_market_aware_end_time(eastern)
            start_time = end_time - timedelta(days=days)

            # Log the date range being fetched
            logger.info(f"Fetching data from {start_time.strftime('%Y-%m-%d %H:%M %Z')} to {end_time.strftime('%Y-%m-%d %H:%M %Z')}")
            
            # Determine if this is a crypto or stock symbol
            is_crypto = self._is_crypto_symbol(self.symbol)
            
            if is_crypto:
                # Use crypto client and request
                request = CryptoBarsRequest(
                    symbol_or_symbols=self.symbol,
                    timeframe=self.timeframe,
                    start=start_time,
                    end=end_time
                )
                bars = self.crypto_client.get_crypto_bars(request)
                logger.info(f"Fetching crypto data for {self.symbol}")
            else:
                # Use stock client and request
                request = StockBarsRequest(
                    symbol_or_symbols=self.symbol,
                    timeframe=self.timeframe,
                    start=start_time,
                    end=end_time
                )
                bars = self.stock_client.get_stock_bars(request)
                logger.info(f"Fetching stock data for {self.symbol}")
            df = bars.df
            
            if df.empty:
                logger.warning("No data received from Alpaca")
                return None
            
            # Reset index and format
            df = df.reset_index()
            
            # Log initial data count
            initial_count = len(df)
            
            if is_crypto:
                logger.info(f"Fetched {initial_count} bars from Alpaca (crypto trades 24/7)")
            else:
                logger.info(f"Fetched {initial_count} bars from Alpaca (includes pre/post market)")

                # Skip RTH filtering for Day/Week timeframes (already aggregated)
                timeframe_minutes = self._get_timeframe_minutes()
                is_daily_or_weekly = timeframe_minutes >= 1440

                # Apply RTH filtering if enabled (only for intraday stocks, not daily/weekly)
                if self.rth_only and not is_daily_or_weekly:
                    # Convert timezone to Eastern Time for RTH filtering
                    df_et = df.copy()
                    df_et['timestamp'] = df_et['timestamp'].dt.tz_convert(self.timezone)
                    df_et = df_et.set_index('timestamp')

                    # Filter to RTH only (9:30 AM - 4:00 PM ET)
                    df_rth = df_et.between_time('09:30', '15:59')

                    # Convert back to original format
                    df = df_rth.reset_index()

                    rth_count = len(df)
                    logger.info(f"RTH filtering: {rth_count} bars remaining (removed {initial_count - rth_count} pre/post-market bars)")
                elif is_daily_or_weekly:
                    logger.info("RTH filtering skipped for Day/Week timeframe (already aggregated)")
                else:
                    logger.info("RTH filtering disabled - using all trading hours")
            
            if df.empty:
                logger.warning("No data remaining after RTH filtering")
                return None
            
            # Convert to format expected by chart
            historical = []
            for _, row in df.iterrows():
                historical.append({
                    'timestamp': row['timestamp'].isoformat(),
                    'open': float(row['open']),
                    'high': float(row['high']),
                    'low': float(row['low']),
                    'close': float(row['close']),
                    'volume': int(row['volume'])
                })
            
            self.latest_historical = historical
            logger.info(f"Final dataset: {len(historical)} bars for model and chart")
            return historical
            
        except Exception as e:
            logger.error(f"Error fetching historical data: {e}")
            return None

    
    def _get_pandas_freq(self):
        """Get pandas frequency string from current timeframe"""
        minutes = self._get_timeframe_minutes()
        if minutes == 1440:
            return 'D'  # Daily
        elif minutes == 10080:
            return 'W'  # Weekly
        return f'{minutes}min'

    def generate_prediction(self, n_samples=25):
        """Generate prediction using Kronos model with sequential batch Monte Carlo sampling"""
        try:
            logger.info(f"Starting prediction generation for {self.symbol} with {n_samples} samples")
            if not self.latest_historical:
                self.fetch_historical_data()  # Use dynamic calculation based on timeframe
            
            if not self.latest_historical:
                return None
            
            # Prepare context data (last 480 bars)
            context_length = self.config['data']['lookback_bars']
            context_data = self.latest_historical[-context_length:]
            
            # Create DataFrame for Kronos
            data_for_kronos = []
            for bar in context_data:
                data_for_kronos.append({
                    'open': bar['open'],
                    'high': bar['high'],
                    'low': bar['low'],
                    'close': bar['close'],
                    'volume': bar['volume']
                })
            
            context_df = pd.DataFrame(data_for_kronos)
            context_df['amount'] = context_df['volume'] * context_df['close']  # Kronos expects amount column
            
            # Create timestamps - normalize to UTC and make timezone-naive for Kronos
            # Kronos's calc_time_stamps() uses .dt accessor which requires datetime64 dtype
            # Using utc=True ensures consistent handling of DST transitions across long time ranges
            raw_timestamps = [bar['timestamp'] for bar in context_data]
            x_timestamp = pd.to_datetime(raw_timestamps, utc=True).tz_localize(None)
            x_timestamp = pd.Series(x_timestamp)

            # Generate future timestamps for prediction
            last_timestamp = x_timestamp.iloc[-1]
            freq = self._get_pandas_freq()
            y_timestamp = pd.Series(pd.date_range(
                start=last_timestamp + pd.Timedelta(minutes=self._get_timeframe_minutes()),
                periods=self.config['data']['horizon'],
                freq=freq
            ))
            
            # Generate predictions with single Kronos call (all samples in parallel)
            logger.info(f"Generating {n_samples} Monte Carlo samples...")

            # Single GPU call with all samples
            pred_df = self.predictor.predict(
                df=context_df,
                x_timestamp=x_timestamp,
                y_timestamp=y_timestamp,
                pred_len=self.config['data']['horizon'],
                sample_count=n_samples,  # All samples in one call (e.g., 25)
                T=self.config['sampling']['temperature'],
                top_p=self.config['sampling']['top_p'],
                return_samples=True
            )

            # Extract all predictions from single response
            # Handle different return formats from Kronos
            if isinstance(pred_df, dict):
                # If dict with 'samples' key
                all_predictions = [sample['close'].values for sample in pred_df.get('samples', [pred_df])]
            elif 'close' in pred_df.columns:
                # Fallback: single averaged prediction (shouldn't occur when return_samples=True)
                all_predictions = [pred_df['close'].values] * n_samples
            else:
                # Fallback
                all_predictions = [pred_df['close'].values] * n_samples

            predictions = np.array(all_predictions)
            logger.info(f"Generated predictions with shape: {predictions.shape}")
            
            # Calculate statistics
            current_price = context_df['close'].iloc[-1]
            mean_path = np.mean(predictions, axis=0)
            
            # Calculate percentiles
            percentiles = {
                'p10': np.percentile(predictions, 10, axis=0).tolist(),
                'p25': np.percentile(predictions, 25, axis=0).tolist(),
                'p50': np.percentile(predictions, 50, axis=0).tolist(),
                'p75': np.percentile(predictions, 75, axis=0).tolist(),
                'p90': np.percentile(predictions, 90, axis=0).tolist()
            }
            
            # Calculate probability of price going up
            final_prices = predictions[:, -1]
            p_up = np.mean(final_prices > current_price)
            
            # Calculate expected return
            exp_return = (np.mean(final_prices) - current_price) / current_price
            
            # Calculate technical indicators
            # Skip VWAP for Day/Week timeframes (VWAP is intraday indicator only)
            timeframe_minutes = self._get_timeframe_minutes()
            if timeframe_minutes >= 1440:
                vwap = None
                logger.info("VWAP calculation skipped for Day/Week timeframe")
            else:
                vwap = self._calculate_vwap(context_df)
            bollinger = self._calculate_bollinger(context_df['close'].values)

            # Calculate Simple Moving Averages for current price
            sma_5 = self._calculate_sma(context_df['close'].values, 5)
            sma_21 = self._calculate_sma(context_df['close'].values, 21)
            sma_233 = self._calculate_sma(context_df['close'].values, 233)

            # Calculate SMA series for chart display using full historical data
            # Create DataFrame from the latest_historical data for SMA calculation
            try:
                if self.latest_historical and len(self.latest_historical) > 0:
                    # Ensure we have proper DataFrame structure
                    full_df = pd.DataFrame(self.latest_historical)

                    # Check if we have the required 'close' column
                    if 'close' in full_df.columns:
                        sma_5_series = self._calculate_sma_series(full_df, 5)
                        sma_21_series = self._calculate_sma_series(full_df, 21)
                        sma_233_series = self._calculate_sma_series(full_df, 233)
                        logger.info(f"SMA series lengths: SMA5={len(sma_5_series)}, SMA21={len(sma_21_series)}, SMA233={len(sma_233_series)}")
                    else:
                        logger.warning(f"No 'close' column in historical data. Columns: {list(full_df.columns)}")
                        sma_5_series = []
                        sma_21_series = []
                        sma_233_series = []
                else:
                    logger.warning("No historical data available for SMA calculation")
                    sma_5_series = []
                    sma_21_series = []
                    sma_233_series = []

                # Debug logging
                sma_5_str = f"{sma_5:.2f}" if sma_5 is not None else "None"
                sma_21_str = f"{sma_21:.2f}" if sma_21 is not None else "None"
                sma_233_str = f"{sma_233:.2f}" if sma_233 is not None else "None"
                logger.info(f"SMA calculations: SMA5={sma_5_str}, SMA21={sma_21_str}, SMA233={sma_233_str}")

            except Exception as e:
                logger.error(f"Error in SMA calculation: {e}")
                sma_5_series = []
                sma_21_series = []
                sma_233_series = []

            # Get asset type info
            asset_type = self._get_asset_type(self.symbol)
            display_name = self._get_display_name(self.symbol)

            # Extract model name from checkpoint path (e.g., "NeoQuasar/Kronos-base" -> "Kronos-base")
            model_checkpoint = self.config['model']['checkpoint']
            model_name = model_checkpoint.split('/')[-1] if '/' in model_checkpoint else model_checkpoint

            prediction_summary = {
                'current_close': current_price,
                'mean_path': mean_path.tolist(),
                'percentiles': percentiles,
                'p_up_30m': float(p_up),
                'exp_ret_30m': float(exp_return),
                'current_vwap': vwap,
                'bollinger_bands': bollinger,
                'sma_5': sma_5,
                'sma_21': sma_21,
                'sma_233': sma_233,
                'sma_5_series': sma_5_series,
                'sma_21_series': sma_21_series,
                'sma_233_series': sma_233_series,
                'n_samples': n_samples,
                'rth_only': self.rth_only if asset_type == 'stock' else False,  # RTH only applies to stocks
                'asset_type': asset_type,
                'display_name': display_name,
                'symbol': self.symbol,
                'timeframe_minutes': self._get_timeframe_minutes(),
                'model_name': model_name,
                'data_bars_count': len(context_data),
                'timestamp': datetime.now(pytz.timezone('US/Eastern')).isoformat()
            }
            
            self.latest_prediction = prediction_summary
            self.last_update = datetime.now(pytz.timezone('US/Eastern'))
            
            logger.info(f"Prediction generated: P(up)={p_up:.2%}, E[r]={exp_return:.3%}")
            return prediction_summary
            
        except Exception as e:
            logger.error(f"Error generating prediction: {e}")
            return None
    
    def _calculate_vwap(self, df):
        """Calculate VWAP from DataFrame"""
        try:
            # Use last 20 bars for VWAP calculation
            recent_df = df.tail(20)
            typical_price = (recent_df['high'] + recent_df['low'] + recent_df['close']) / 3
            total_value = (typical_price * recent_df['volume']).sum()
            total_volume = recent_df['volume'].sum()
            
            return total_value / total_volume if total_volume > 0 else df['close'].iloc[-1]
        except:
            return df['close'].iloc[-1] if not df.empty else None
    
    def _calculate_bollinger(self, prices, period=20, std_dev=2):
        """Calculate Bollinger Bands"""
        try:
            recent_prices = prices[-period:]
            middle = np.mean(recent_prices)
            std = np.std(recent_prices)
            
            return {
                'upper': middle + (std_dev * std),
                'middle': middle,
                'lower': middle - (std_dev * std)
            }
        except:
            return None

    def _calculate_sma(self, prices, period):
        """Calculate Simple Moving Average"""
        try:
            if len(prices) < period:
                return None
            return np.mean(prices[-period:])
        except:
            return None

    def _calculate_sma_series(self, df, period):
        """Calculate SMA series for entire dataset"""
        try:
            closes = df['close'].values
            sma_values = []

            for i in range(len(closes)):
                if i < period - 1:
                    # Not enough data points, use NaN or skip
                    sma_values.append(None)
                else:
                    # Calculate SMA for this point
                    sma = np.mean(closes[i-period+1:i+1])
                    sma_values.append(float(sma))

            return sma_values
        except Exception as e:
            logger.warning(f"Error calculating SMA series: {e}")
            return []

    def get_historical_data(self):
        """Get cached historical data or fetch new"""
        if not self.latest_historical or self._is_stale():
            self.fetch_historical_data()  # Use dynamic calculation based on timeframe
        return self.latest_historical
    
    def get_latest_prediction(self):
        """Get cached prediction or generate new"""
        if not self.latest_prediction or self._is_stale():
            self.generate_prediction()
        return self.latest_prediction
    
    def generate_new_prediction(self):
        """Force generation of new prediction"""
        self.fetch_historical_data()  # Use dynamic calculation based on timeframe
        return self.generate_prediction()
    
    def _is_stale(self, max_age_minutes=5):
        """Check if cached data is stale"""
        if not self.last_update:
            return True
        age = datetime.now(pytz.timezone('US/Eastern')) - self.last_update
        return age.total_seconds() > (max_age_minutes * 60)

# Test the service
if __name__ == "__main__":
    service = PredictionService()
    
    # Test fetching historical data
    historical = service.fetch_historical_data()
    if historical:
        print(f"Fetched {len(historical)} bars")
        print(f"Latest bar: {historical[-1]}")
    
    # Test generating prediction
    prediction = service.generate_prediction()
    if prediction:
        print(f"\nPrediction Summary:")
        print(f"Current Price: ${prediction['current_close']:.2f}")
        print(f"P(Up in 30m): {prediction['p_up_30m']:.2%}")
        print(f"Expected Return: {prediction['exp_ret_30m']:.3%}")
        print(f"VWAP: ${prediction['current_vwap']:.2f}")
