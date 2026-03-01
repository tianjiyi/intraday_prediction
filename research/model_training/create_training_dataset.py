"""
Create Training Datasets from Raw Alpaca Data
Generates sliding windows and splits into train/val/test sets
"""
import os
import pickle
from datetime import datetime
import pandas as pd
import numpy as np
from tqdm import tqdm

from config import TrainingConfig


class DatasetCreator:
    """
    Creates training, validation, and test datasets from raw Alpaca data.
    Generates sliding windows for time series prediction.
    """

    def __init__(self, config: TrainingConfig):
        """
        Initialize dataset creator.

        Args:
            config: TrainingConfig instance
        """
        self.config = config

    def load_raw_data(self) -> pd.DataFrame:
        """
        Load raw data from pickle file.

        Returns:
            DataFrame with raw OHLCV data
        """
        raw_file = os.path.join(
            self.config.raw_data_path,
            f"{self.config.symbol}_1min_raw.pkl"
        )

        print(f"Loading raw data from: {raw_file}")

        with open(raw_file, 'rb') as f:
            df = pickle.load(f)

        print(f"  Loaded {len(df)} bars")
        print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

        return df

    def add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add time-based features and calculate additional features.

        Args:
            df: DataFrame with timestamp column

        Returns:
            DataFrame with added time and calculated features
        """
        print("\nAdding time features and calculating 'amount'...")

        df = df.copy()

        # Calculate 'amount' (volume * close price)
        # This is expected by Kronos as the 6th feature
        df['amount'] = df['volume'] * df['close']

        # Ensure timestamp is timezone-aware
        if df['timestamp'].dt.tz is None:
            df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize('UTC').dt.tz_convert('US/Eastern')
        else:
            df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_convert('US/Eastern')

        # Extract time features
        df['minute'] = df['timestamp'].dt.minute
        df['hour'] = df['timestamp'].dt.hour
        df['weekday'] = df['timestamp'].dt.weekday  # Monday=0, Sunday=6
        df['day'] = df['timestamp'].dt.day
        df['month'] = df['timestamp'].dt.month

        return df

    def split_by_time(self, df: pd.DataFrame) -> tuple:
        """
        Split data into train, validation, and test sets by time.

        Args:
            df: Full DataFrame

        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        print("\nSplitting data by time...")

        # Convert split dates to datetime
        train_start = pd.to_datetime(self.config.train_time_range[0])
        train_end = pd.to_datetime(self.config.train_time_range[1])
        val_start = pd.to_datetime(self.config.val_time_range[0])
        val_end = pd.to_datetime(self.config.val_time_range[1])
        test_start = pd.to_datetime(self.config.test_time_range[0])
        test_end = pd.to_datetime(self.config.test_time_range[1])

        # Make timezone-aware if needed
        tz = df['timestamp'].dt.tz
        if tz is not None:
            train_start = train_start.tz_localize(tz)
            train_end = train_end.tz_localize(tz)
            val_start = val_start.tz_localize(tz)
            val_end = val_end.tz_localize(tz)
            test_start = test_start.tz_localize(tz)
            test_end = test_end.tz_localize(tz)

        # Create masks
        train_mask = (df['timestamp'] >= train_start) & (df['timestamp'] <= train_end)
        val_mask = (df['timestamp'] >= val_start) & (df['timestamp'] <= val_end)
        test_mask = (df['timestamp'] >= test_start) & (df['timestamp'] <= test_end)

        # Split dataframes
        train_df = df[train_mask].copy()
        val_df = df[val_mask].copy()
        test_df = df[test_mask].copy()

        print(f"  Train: {len(train_df)} bars ({train_start.date()} to {train_end.date()})")
        print(f"  Val:   {len(val_df)} bars ({val_start.date()} to {val_end.date()})")
        print(f"  Test:  {len(test_df)} bars ({test_start.date()} to {test_end.date()})")

        # Validate minimum data requirements
        min_window = self.config.lookback_window + self.config.predict_window + 1
        if len(train_df) < min_window:
            raise ValueError(f"Insufficient training data: {len(train_df)} bars < {min_window} required")
        if len(val_df) < min_window:
            raise ValueError(f"Insufficient validation data: {len(val_df)} bars < {min_window} required")
        if len(test_df) < min_window:
            raise ValueError(f"Insufficient test data: {len(test_df)} bars < {min_window} required")

        return train_df, val_df, test_df

    def prepare_dataset_for_kronos(self, df: pd.DataFrame) -> dict:
        """
        Prepare dataset in the format expected by Kronos training.

        The format matches Qlib's structure:
        {
            'SYMBOL': pd.DataFrame(
                index='datetime',
                columns=['open', 'high', 'low', 'close', 'volume', 'minute', 'hour', 'weekday', 'day', 'month']
            )
        }

        Args:
            df: DataFrame with features and time columns

        Returns:
            Dictionary with symbol as key
        """
        # Set timestamp as index and name it 'datetime'
        df = df.set_index('timestamp')
        df.index.name = 'datetime'

        # Select only the required columns
        columns_to_keep = self.config.feature_list + self.config.time_feature_list
        df = df[columns_to_keep]

        # Package in dictionary format
        data_dict = {self.config.symbol: df}

        return data_dict

    def save_datasets(self, train_data: dict, val_data: dict, test_data: dict):
        """
        Save datasets to pickle files.

        Args:
            train_data: Training dataset dictionary
            val_data: Validation dataset dictionary
            test_data: Test dataset dictionary
        """
        os.makedirs(self.config.dataset_path, exist_ok=True)

        # Save files
        files = {
            'train_data.pkl': train_data,
            'val_data.pkl': val_data,
            'test_data.pkl': test_data
        }

        print("\nSaving datasets...")
        for filename, data in files.items():
            filepath = os.path.join(self.config.dataset_path, filename)
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)

            # Get dataset info
            df = data[self.config.symbol]
            filesize = os.path.getsize(filepath) / 1024 / 1024

            print(f"  [OK] {filename}")
            print(f"      Bars: {len(df)}")
            print(f"      Date range: {df.index.min()} to {df.index.max()}")
            print(f"      File size: {filesize:.2f} MB")

    def print_dataset_stats(self, train_data: dict, val_data: dict, test_data: dict):
        """
        Print statistics about the datasets.

        Args:
            train_data: Training dataset dictionary
            val_data: Validation dataset dictionary
            test_data: Test dataset dictionary
        """
        print("\n" + "=" * 80)
        print("DATASET STATISTICS")
        print("=" * 80)

        for name, data in [("TRAIN", train_data), ("VALIDATION", val_data), ("TEST", test_data)]:
            df = data[self.config.symbol]

            # Calculate potential windows
            window_size = self.config.lookback_window + self.config.predict_window
            max_windows = max(0, len(df) - window_size + 1)

            print(f"\n{name} SET:")
            print(f"  Total bars: {len(df)}")
            print(f"  Max possible windows: {max_windows:,}")
            print(f"  Date range: {df.index.min()} to {df.index.max()}")
            print(f"  Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
            print(f"  Avg volume: {df['volume'].mean():,.0f}")

    def run(self):
        """
        Execute complete dataset creation pipeline.
        """
        print("=" * 80)
        print("DATASET CREATION - KRONOS TRAINING")
        print("=" * 80)

        # Step 1: Load raw data
        df = self.load_raw_data()

        # Step 2: Add time features
        df = self.add_time_features(df)

        # Step 3: Split by time
        train_df, val_df, test_df = self.split_by_time(df)

        # Step 4: Prepare in Kronos format
        train_data = self.prepare_dataset_for_kronos(train_df)
        val_data = self.prepare_dataset_for_kronos(val_df)
        test_data = self.prepare_dataset_for_kronos(test_df)

        # Step 5: Print statistics
        self.print_dataset_stats(train_data, val_data, test_data)

        # Step 6: Save datasets
        self.save_datasets(train_data, val_data, test_data)

        print("\n" + "=" * 80)
        print("DATASET CREATION COMPLETE")
        print("=" * 80)

        return train_data, val_data, test_data


if __name__ == '__main__':
    # Load configuration
    config = TrainingConfig()

    # Create datasets
    creator = DatasetCreator(config)
    train_data, val_data, test_data = creator.run()

    # Show sample
    print("\nSample from training data:")
    print(train_data[config.symbol].head(10))
