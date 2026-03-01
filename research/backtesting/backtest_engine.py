"""
Backtesting Engine for Kronos Model

Implements rolling window backtesting for time series predictions.
Generates predictions at historical points and compares with actual outcomes.
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import logging
from pathlib import Path

# Add parent directory to path to import from live_chart_prediction
sys.path.insert(0, str(Path(__file__).parent.parent))

from live_chart_prediction.prediction_service import PredictionService

logger = logging.getLogger(__name__)


class BacktestEngine:
    """
    Rolling window backtesting engine for Kronos model.

    Generates predictions at multiple historical points and compares
    with actual outcomes after the prediction horizon.
    """

    def __init__(self, config: Dict):
        """
        Initialize backtesting engine.

        Args:
            config: Backtesting configuration dictionary
        """
        self.config = config
        self.prediction_service = None
        self.results = []

        # Initialize prediction service
        self._init_prediction_service()

    def _init_prediction_service(self):
        """Initialize the PredictionService with backtesting config"""
        try:
            # Load parent config
            parent_config_path = Path(__file__).parent.parent / "config.yaml"

            # Create prediction service
            self.prediction_service = PredictionService(str(parent_config_path))

            logger.info("Prediction service initialized for backtesting")

        except Exception as e:
            logger.error(f"Failed to initialize prediction service: {e}")
            raise

    def _convert_timeframe_format(self, timeframe: str) -> str:
        """
        Convert timeframe from '1Min' format to '1' format for prediction service.

        Args:
            timeframe: Timeframe like "1Min", "5Min", "15Min", "30Min"

        Returns:
            Timeframe as number string like "1", "5", "15", "30"
        """
        # Remove 'Min' suffix if present
        if timeframe.endswith('Min'):
            return timeframe[:-3]
        return timeframe

    def run_backtest(
        self,
        symbol: str,
        timeframe: str,
        start_date: str,
        end_date: str,
        step_size_minutes: int = 30,
    ) -> pd.DataFrame:
        """
        Run rolling window backtest for a symbol.

        Args:
            symbol: Trading symbol (e.g., "QQQ", "SPY")
            timeframe: Bar timeframe (e.g., "1Min", "5Min")
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            step_size_minutes: Minutes between prediction points

        Returns:
            DataFrame with predictions and actuals
        """
        logger.info(f"Starting backtest for {symbol} ({timeframe})")
        logger.info(f"Date range: {start_date} to {end_date}")

        # Convert timeframe format for prediction service (expects "1", "5", etc.)
        timeframe_number = self._convert_timeframe_format(timeframe)

        # Update prediction service settings
        self.prediction_service.update_settings(symbol, timeframe_number)

        # Convert dates to datetime
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)

        # Generate prediction timestamps (RTH only)
        prediction_timestamps = self._generate_prediction_timestamps(
            start_dt, end_dt, step_size_minutes
        )

        logger.info(f"Generated {len(prediction_timestamps)} prediction points")

        # Run predictions for each timestamp
        predictions = []
        for i, pred_time in enumerate(prediction_timestamps):
            try:
                logger.info(
                    f"Processing {i+1}/{len(prediction_timestamps)}: {pred_time}"
                )

                # Generate prediction at this point in time
                pred_result = self._generate_prediction_at_time(
                    symbol, timeframe, pred_time
                )

                if pred_result is not None:
                    predictions.append(pred_result)

            except Exception as e:
                logger.warning(f"Failed to generate prediction at {pred_time}: {e}")
                continue

        # Convert to DataFrame
        if predictions:
            df = pd.DataFrame(predictions)
            logger.info(f"Backtest completed: {len(df)} predictions generated")
            return df
        else:
            logger.warning("No predictions generated")
            return pd.DataFrame()

    def _generate_prediction_timestamps(
        self, start_dt: datetime, end_dt: datetime, step_size_minutes: int
    ) -> List[datetime]:
        """
        Generate timestamps for predictions (RTH only).

        Creates timestamps spaced by step_size_minutes during market hours.
        All timestamps are timezone-aware (US/Eastern).
        """
        import pytz

        timestamps = []
        eastern = pytz.timezone('US/Eastern')

        current_date = start_dt.date()
        end_date = end_dt.date()

        while current_date <= end_date:
            # Market hours: 9:30 AM - 4:00 PM ET
            market_open = datetime.combine(current_date, datetime.min.time()).replace(
                hour=9, minute=30
            )
            market_close = datetime.combine(current_date, datetime.min.time()).replace(
                hour=16, minute=0
            )

            # Make timezone-aware
            market_open = eastern.localize(market_open)
            market_close = eastern.localize(market_close)

            # Generate timestamps for this day
            current_time = market_open

            while current_time <= market_close - timedelta(
                minutes=30
            ):  # Leave room for 30-min horizon
                timestamps.append(current_time)
                current_time += timedelta(minutes=step_size_minutes)

            # Move to next trading day (skip weekends)
            current_date += timedelta(days=1)
            while current_date.weekday() >= 5:  # Skip Saturday (5) and Sunday (6)
                current_date += timedelta(days=1)

        return timestamps

    def _generate_prediction_at_time(
        self, symbol: str, timeframe: str, pred_time: datetime
    ) -> Optional[Dict]:
        """
        Generate a prediction as if we were at pred_time.

        Args:
            symbol: Trading symbol
            timeframe: Bar timeframe
            pred_time: The historical point in time to make prediction

        Returns:
            Dictionary with prediction and actual outcome, or None if failed
        """
        horizon_minutes = self.config["backtest"].get("horizon_minutes", 30)

        # Calculate actual outcome time (30 minutes after prediction)
        outcome_time = pred_time + timedelta(minutes=horizon_minutes)

        # Fetch historical data UP TO pred_time (simulate real-time)
        context_data = self._fetch_data_until(symbol, timeframe, pred_time)

        if context_data is None or len(context_data) < 480:
            logger.warning(
                f"Insufficient data at {pred_time}: {len(context_data) if context_data else 0} bars"
            )
            return None

        # Get current price (last bar before prediction)
        current_price = context_data[-1]["close"]

        # Generate prediction using the prediction service
        # We need to temporarily set the historical data
        original_historical = self.prediction_service.latest_historical
        self.prediction_service.latest_historical = context_data

        try:
            prediction_summary = self.prediction_service.generate_prediction()

            # Extract prediction data
            # The prediction service returns 'mean_path' (list) and percentiles
            mean_path = prediction_summary.get("mean_path")
            percentiles = prediction_summary.get("percentiles", {})
            prob_up = prediction_summary.get("p_up_30m")

            # Get the final predicted price (last value in mean path or p50)
            if mean_path and len(mean_path) > 0:
                final_predicted_price = mean_path[-1]  # Last value is 30-min prediction
            elif percentiles and 'p50' in percentiles:
                p50_array = percentiles['p50']
                final_predicted_price = p50_array[-1] if isinstance(p50_array, list) and len(p50_array) > 0 else None
            else:
                final_predicted_price = None

            # Validate prediction data
            if final_predicted_price is None:
                logger.warning(f"No valid prediction generated at {pred_time}")
                return None

            # Fetch actual outcome
            actual_price = self._fetch_actual_price(symbol, timeframe, outcome_time)

            if actual_price is None:
                logger.warning(f"No actual price found at {outcome_time}")
                return None

            # Validate all required fields before creating result
            if prob_up is None:
                prob_up = 0.5  # Default to 50% if not available

            # Create result - all fields are guaranteed to be valid
            result = {
                "timestamp": pred_time,
                "symbol": symbol,
                "timeframe": timeframe,
                "current_price": current_price,
                "predicted_price": final_predicted_price,
                "actual_price": actual_price,
                "prob_up": prob_up,
                "prediction_horizon_minutes": horizon_minutes,
                "outcome_timestamp": outcome_time,
            }

            logger.debug(f"Valid prediction created: predicted={final_predicted_price:.2f}, actual={actual_price:.2f}")

            return result

        except Exception as e:
            logger.error(f"Error generating prediction: {e}")
            return None

        finally:
            # Restore original historical data
            self.prediction_service.latest_historical = original_historical

    def _fetch_data_until(
        self, symbol: str, timeframe: str, until_time: datetime
    ) -> Optional[List[Dict]]:
        """
        Fetch historical data up to a specific time (for backtesting).

        Simulates fetching data as if we were at that point in time.
        """
        try:
            # Calculate how many days back we need
            # Use same logic as live system
            timeframe_minutes = self._parse_timeframe(timeframe)
            days_to_fetch = self._calculate_days_to_fetch(timeframe_minutes)

            # Fetch data from Alpaca
            from alpaca.data.historical import StockHistoricalDataClient
            from alpaca.data.requests import StockBarsRequest
            from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
            import yaml
            import pytz

            # Load API credentials
            config_path = Path(__file__).parent.parent / "config.yaml"
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)

            api_key = config["ALPACA_KEY_ID"]
            secret_key = config["ALPACA_SECRET_KEY"]

            client = StockHistoricalDataClient(api_key, secret_key)

            # Create timeframe object based on parsed minutes
            if timeframe_minutes == 1:
                tf = TimeFrame(1, TimeFrameUnit.Minute)
            elif timeframe_minutes == 5:
                tf = TimeFrame(5, TimeFrameUnit.Minute)
            elif timeframe_minutes == 15:
                tf = TimeFrame(15, TimeFrameUnit.Minute)
            elif timeframe_minutes == 30:
                tf = TimeFrame(30, TimeFrameUnit.Minute)
            else:
                tf = TimeFrame(timeframe_minutes, TimeFrameUnit.Minute)

            # Ensure timezone-aware datetimes
            eastern = pytz.timezone('US/Eastern')
            if until_time.tzinfo is None:
                until_time = eastern.localize(until_time)

            start_time = until_time - timedelta(days=days_to_fetch)

            logger.debug(f"Fetching data: {start_time} to {until_time} ({days_to_fetch} days)")

            request = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=tf,
                start=start_time,
                end=until_time,
            )

            bars = client.get_stock_bars(request)

            # Convert to list of dicts
            data = []
            if symbol in bars.data:
                for bar in bars.data[symbol]:
                    data.append(
                        {
                            "timestamp": bar.timestamp.isoformat(),
                            "open": float(bar.open),
                            "high": float(bar.high),
                            "low": float(bar.low),
                            "close": float(bar.close),
                            "volume": int(bar.volume),
                        }
                    )

            logger.debug(f"Fetched {len(data)} bars before RTH filtering")

            # Filter RTH only
            if self.config["backtest"].get("market_hours", {}).get("rth_only", True):
                data = self._filter_rth(data)
                logger.debug(f"After RTH filtering: {len(data)} bars")

            return data

        except Exception as e:
            logger.error(f"Failed to fetch data until {until_time}: {e}", exc_info=True)
            return None

    def _fetch_actual_price(
        self, symbol: str, timeframe: str, outcome_time: datetime
    ) -> Optional[float]:
        """
        Fetch the actual price at outcome_time.

        Returns:
            Actual close price at outcome_time, or None if not found
        """
        try:
            from alpaca.data.historical import StockHistoricalDataClient
            from alpaca.data.requests import StockBarsRequest
            from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
            import yaml
            import pytz

            # Load API credentials
            config_path = Path(__file__).parent.parent / "config.yaml"
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)

            api_key = config["ALPACA_KEY_ID"]
            secret_key = config["ALPACA_SECRET_KEY"]

            client = StockHistoricalDataClient(api_key, secret_key)

            # Parse timeframe to get minutes
            timeframe_minutes = self._parse_timeframe(timeframe)

            # Create timeframe object based on parsed minutes
            if timeframe_minutes == 1:
                tf = TimeFrame(1, TimeFrameUnit.Minute)
            elif timeframe_minutes == 5:
                tf = TimeFrame(5, TimeFrameUnit.Minute)
            elif timeframe_minutes == 15:
                tf = TimeFrame(15, TimeFrameUnit.Minute)
            elif timeframe_minutes == 30:
                tf = TimeFrame(30, TimeFrameUnit.Minute)
            else:
                tf = TimeFrame(timeframe_minutes, TimeFrameUnit.Minute)

            # Ensure timezone-aware datetimes
            eastern = pytz.timezone('US/Eastern')
            if outcome_time.tzinfo is None:
                outcome_time = eastern.localize(outcome_time)

            # Fetch single bar at outcome_time
            request = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=tf,
                start=outcome_time - timedelta(minutes=5),
                end=outcome_time + timedelta(minutes=5),
            )

            bars = client.get_stock_bars(request)

            # Find closest bar to outcome_time
            if symbol in bars.data and len(bars.data[symbol]) > 0:
                # Get the bar closest to outcome_time
                closest_bar = min(
                    bars.data[symbol],
                    key=lambda b: abs((b.timestamp - outcome_time).total_seconds()),
                )
                return float(closest_bar.close)

            return None

        except Exception as e:
            logger.error(f"Failed to fetch actual price at {outcome_time}: {e}", exc_info=True)
            return None

    def _filter_rth(self, data: List[Dict]) -> List[Dict]:
        """Filter data to Regular Trading Hours (9:30 AM - 4:00 PM ET)"""
        filtered = []

        for bar in data:
            ts = pd.to_datetime(bar["timestamp"])
            # Assume data is already in ET timezone
            hour = ts.hour
            minute = ts.minute

            # RTH: 9:30 AM - 3:59 PM
            if (hour == 9 and minute >= 30) or (10 <= hour <= 14) or (hour == 15):
                filtered.append(bar)

        return filtered

    def _parse_timeframe(self, timeframe: str) -> int:
        """Parse timeframe string to minutes"""
        if timeframe == "1Min":
            return 1
        elif timeframe == "5Min":
            return 5
        elif timeframe == "15Min":
            return 15
        elif timeframe == "30Min":
            return 30
        else:
            return 1

    def _calculate_days_to_fetch(self, timeframe_minutes: int) -> int:
        """Calculate days to fetch based on timeframe"""
        target_bars = 350
        rth_minutes_per_day = 390
        bars_per_day = rth_minutes_per_day / timeframe_minutes
        days_needed = target_bars / bars_per_day
        days_with_buffer = int(days_needed * 1.5) + 1
        return min(days_with_buffer, 60)

    def save_results(self, df: pd.DataFrame, output_path: str):
        """
        Save backtest results to CSV.

        Args:
            df: Results DataFrame
            output_path: Path to save CSV file
        """
        try:
            # Create directory if needed
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)

            # Save to CSV
            df.to_csv(output_path, index=False)
            logger.info(f"Results saved to {output_path}")

        except Exception as e:
            logger.error(f"Failed to save results: {e}")


def load_backtest_results(csv_path: str) -> pd.DataFrame:
    """
    Load saved backtest results from CSV.

    Args:
        csv_path: Path to CSV file

    Returns:
        DataFrame with results
    """
    try:
        df = pd.read_csv(csv_path)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["outcome_timestamp"] = pd.to_datetime(df["outcome_timestamp"])
        return df
    except Exception as e:
        logger.error(f"Failed to load results: {e}")
        return pd.DataFrame()
