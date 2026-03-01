"""
Alpaca Data Fetcher for Kronos Training
Fetches 1-minute bars for QQQ over the last 12 months
Filters to 10:00 AM - 4:00 PM ET (skip first 30 minutes)
"""
import os
import pickle
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import pytz
from tqdm import tqdm

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

from config import TrainingConfig


class AlpacaDataFetcher:
    """
    Fetches and processes historical 1-minute bar data from Alpaca for training.
    """

    def __init__(self, config: TrainingConfig):
        """
        Initialize data fetcher with configuration.

        Args:
            config: TrainingConfig instance
        """
        self.config = config
        self.client = StockHistoricalDataClient(
            config.alpaca_api_key,
            config.alpaca_secret_key
        )
        self.eastern = pytz.timezone('US/Eastern')

    def fetch_historical_bars(self) -> pd.DataFrame:
        """
        Fetch 1-minute bars from Alpaca for the configured time period.

        Returns:
            DataFrame with OHLCV data
        """
        print(f"\nFetching {self.config.symbol} 1-minute bars...")
        print(f"Date Range: {self.config.dataset_begin_time} to {self.config.dataset_end_time}")

        # Parse dates
        start_date = pd.to_datetime(self.config.dataset_begin_time)
        end_date = pd.to_datetime(self.config.dataset_end_time)

        # Make timezone-aware
        if start_date.tzinfo is None:
            start_date = self.eastern.localize(start_date)
        if end_date.tzinfo is None:
            end_date = self.eastern.localize(end_date)

        # Alpaca API has limits, so fetch in chunks (1 month at a time)
        all_bars = []
        current_start = start_date

        while current_start < end_date:
            # Fetch 1 month at a time
            current_end = min(current_start + timedelta(days=30), end_date)

            print(f"  Fetching: {current_start.date()} to {current_end.date()}...")

            try:
                request = StockBarsRequest(
                    symbol_or_symbols=self.config.symbol,
                    timeframe=TimeFrame(1, TimeFrameUnit.Minute),
                    start=current_start,
                    end=current_end,
                )

                bars = self.client.get_stock_bars(request)

                if self.config.symbol in bars.data:
                    for bar in bars.data[self.config.symbol]:
                        all_bars.append({
                            'timestamp': bar.timestamp,
                            'open': float(bar.open),
                            'high': float(bar.high),
                            'low': float(bar.low),
                            'close': float(bar.close),
                            'volume': int(bar.volume),
                        })

            except Exception as e:
                print(f"  Warning: Failed to fetch data for {current_start.date()}: {e}")

            current_start = current_end

        print(f"Total bars fetched: {len(all_bars)}")

        if len(all_bars) == 0:
            raise ValueError("No data fetched from Alpaca!")

        # Convert to DataFrame
        df = pd.DataFrame(all_bars)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)

        return df

    def filter_market_hours(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter data to Regular Trading Hours (9:30 AM - 4:00 PM ET).

        IMPORTANT: We include 9:30-10:00 AM bars for context, even though
        we won't trade during this period. This ensures we have proper
        context when making predictions at 10:00 AM and later.

        Args:
            df: DataFrame with timestamp column

        Returns:
            Filtered DataFrame
        """
        print("\nFiltering to RTH (9:30 AM - 4:00 PM ET)...")
        print("Note: Including 9:30-10:00 AM for context (not for trading)")

        # Ensure timezone-aware
        if df['timestamp'].dt.tz is None:
            df['timestamp'] = df['timestamp'].dt.tz_localize('UTC').dt.tz_convert('US/Eastern')
        else:
            df['timestamp'] = df['timestamp'].dt.tz_convert('US/Eastern')

        # Extract time components
        df['hour'] = df['timestamp'].dt.hour
        df['minute'] = df['timestamp'].dt.minute

        # Filter to 9:30 AM - 3:59 PM (RTH)
        mask = (
            ((df['hour'] == 9) & (df['minute'] >= 30)) |   # 9:30 AM onwards
            (df['hour'].isin([10, 11, 12, 13, 14, 15]))    # 10 AM - 3:59 PM
        )

        filtered_df = df[mask].copy()
        filtered_df = filtered_df.drop(columns=['hour', 'minute'])

        print(f"Bars before filtering: {len(df)}")
        print(f"Bars after filtering: {len(filtered_df)}")
        print(f"Removed: {len(df) - len(filtered_df)} bars (pre/post market)")

        return filtered_df

    def validate_data(self, df: pd.DataFrame) -> bool:
        """
        Validate data quality.

        Args:
            df: DataFrame to validate

        Returns:
            True if data is valid
        """
        print("\nValidating data quality...")

        # Check for missing values
        missing = df.isnull().sum()
        if missing.any():
            print(f"Warning: Missing values detected:")
            print(missing[missing > 0])
            return False

        # Check for negative prices
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            if (df[col] <= 0).any():
                print(f"Error: Negative or zero prices in {col}")
                return False

        # Check OHLC relationships
        if ((df['high'] < df['low']) |
            (df['high'] < df['open']) |
            (df['high'] < df['close']) |
            (df['low'] > df['open']) |
            (df['low'] > df['close'])).any():
            print("Error: Invalid OHLC relationships detected")
            return False

        # Check for duplicates
        duplicates = df['timestamp'].duplicated().sum()
        if duplicates > 0:
            print(f"Warning: {duplicates} duplicate timestamps found")
            return False

        print("[OK] Data validation passed")
        return True

    def save_raw_data(self, df: pd.DataFrame):
        """
        Save raw data to pickle file.

        Args:
            df: DataFrame to save
        """
        os.makedirs(self.config.raw_data_path, exist_ok=True)

        output_path = os.path.join(
            self.config.raw_data_path,
            f"{self.config.symbol}_1min_raw.pkl"
        )

        with open(output_path, 'wb') as f:
            pickle.dump(df, f)

        print(f"\n[OK] Raw data saved to: {output_path}")
        print(f"  Total bars: {len(df)}")
        print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"  File size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")

    def run(self):
        """
        Execute complete data fetching pipeline.
        """
        print("=" * 80)
        print("ALPACA DATA FETCHER - QQQ 1-MINUTE BARS")
        print("=" * 80)

        # Step 1: Fetch data
        df = self.fetch_historical_bars()

        # Step 2: Filter to market hours (10 AM - 4 PM)
        df = self.filter_market_hours(df)

        # Step 3: Validate data
        if not self.validate_data(df):
            raise ValueError("Data validation failed!")

        # Step 4: Save raw data
        self.save_raw_data(df)

        print("\n" + "=" * 80)
        print("DATA FETCHING COMPLETE")
        print("=" * 80)

        return df


if __name__ == '__main__':
    # Load configuration
    config = TrainingConfig()
    config.print_summary()

    # Fetch data
    fetcher = AlpacaDataFetcher(config)
    df = fetcher.run()

    # Print sample
    print("\nFirst 5 bars:")
    print(df.head())
    print("\nLast 5 bars:")
    print(df.tail())
