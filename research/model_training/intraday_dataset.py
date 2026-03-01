"""
PyTorch Dataset for Intraday Training
Generates sliding windows for Kronos model training
Adapted from Kronos/finetune/dataset.py
"""
import pickle
import random
import numpy as np
import torch
from torch.utils.data import Dataset

from config import TrainingConfig


class IntradayDataset(Dataset):
    """
    A PyTorch Dataset for handling intraday financial time series data.

    This dataset pre-computes all possible start indices for sliding windows
    and then randomly samples from them during training/validation.

    Args:
        data_type (str): The type of dataset to load, either 'train' or 'val'.
    """

    def __init__(self, data_type: str = 'train'):
        self.config = TrainingConfig()

        if data_type not in ['train', 'val']:
            raise ValueError("data_type must be 'train' or 'val'")

        self.data_type = data_type

        # Dedicated random number generator for reproducibility
        self.py_rng = random.Random(self.config.seed)

        # Set paths and number of samples
        if data_type == 'train':
            self.data_path = f"{self.config.dataset_path}/train_data.pkl"
            self.n_samples = self.config.n_train_iter
        else:
            self.data_path = f"{self.config.dataset_path}/val_data.pkl"
            self.n_samples = self.config.n_val_iter

        # Load data
        with open(self.data_path, 'rb') as f:
            self.data = pickle.load(f)

        self.window = self.config.lookback_window + self.config.predict_window + 1
        self.symbols = list(self.data.keys())
        self.feature_list = self.config.feature_list
        self.time_feature_list = self.config.time_feature_list

        # Pre-compute all possible (symbol, start_index) pairs
        # IMPORTANT: Only include windows where prediction target (last 30 bars) is >= 10:00 AM
        # We keep 9:30-10:00 AM bars for context, but don't predict during that period
        self.indices = []
        print(f"[{data_type.upper()}] Pre-computing sample indices...")
        print(f"[{data_type.upper()}] Sliding interval: {self.config.sliding_interval_minutes} minutes")
        print(f"[{data_type.upper()}] Filtering: prediction targets must be >= 10:00 AM")

        for symbol in self.symbols:
            df = self.data[symbol].reset_index()
            series_len = len(df)
            num_samples = series_len - self.window + 1

            if num_samples > 0:
                # Time features are already in the dataframe from preprocessing
                # Just verify they exist
                for time_feat in self.time_feature_list:
                    if time_feat not in df.columns:
                        raise ValueError(f"Time feature '{time_feat}' not found in data")

                # Keep only necessary columns to save memory
                self.data[symbol] = df[['datetime'] + self.feature_list + self.time_feature_list]

                # Add valid starting indices where prediction window is >= 10:00 AM
                # Window structure: [lookback: 480 bars][predict: 30 bars][+1]
                # We check if the FIRST bar of the prediction window (index: lookback_window) is >= 10:00 AM
                for i in range(num_samples):
                    # Index of first prediction bar (after 480 lookback bars)
                    prediction_start_idx = i + self.config.lookback_window

                    if prediction_start_idx < len(df):
                        prediction_start_time = df.iloc[prediction_start_idx]['datetime']

                        # Only include if prediction window starts at or after 10:00 AM
                        # AND the minute aligns with sliding_interval_minutes (e.g., :00, :15, :30, :45 for 15-min interval)
                        if (prediction_start_time.hour >= 10 and
                            prediction_start_time.minute % self.config.sliding_interval_minutes == 0):
                            self.indices.append((symbol, i))

        # Effective dataset size
        self.n_samples = min(self.n_samples, len(self.indices))

        print(f"[{data_type.upper()}] Found {len(self.indices)} possible samples. "
              f"Using {self.n_samples} per epoch.")

    def set_epoch_seed(self, epoch: int):
        """
        Sets a new seed for the random sampler for each epoch.
        Crucial for reproducibility in distributed training.

        Args:
            epoch (int): The current epoch number.
        """
        epoch_seed = self.config.seed + epoch
        self.py_rng.seed(epoch_seed)

    def __len__(self) -> int:
        """Returns the number of samples per epoch."""
        return self.n_samples

    def __getitem__(self, idx: int) -> tuple:
        """
        Retrieves a random sample from the dataset.

        Note: The `idx` argument is ignored. Instead, a random index is drawn
        from the pre-computed `self.indices` list using `self.py_rng`.

        Args:
            idx (int): Ignored.

        Returns:
            tuple: (x_tensor, x_stamp_tensor)
                - x_tensor: Normalized feature tensor [window_size, num_features]
                - x_stamp_tensor: Time feature tensor [window_size, num_time_features]
        """
        # Select a random sample from the entire pool of indices
        random_idx = self.py_rng.randint(0, len(self.indices) - 1)
        symbol, start_idx = self.indices[random_idx]

        # Extract the sliding window
        df = self.data[symbol]
        end_idx = start_idx + self.window
        win_df = df.iloc[start_idx:end_idx]

        # Separate main features and time features
        x = win_df[self.feature_list].values.astype(np.float32)
        x_stamp = win_df[self.time_feature_list].values.astype(np.float32)

        # Perform instance-level normalization
        x_mean, x_std = np.mean(x, axis=0), np.std(x, axis=0)
        x = (x - x_mean) / (x_std + 1e-5)
        x = np.clip(x, -self.config.clip, self.config.clip)

        # Convert to PyTorch tensors
        x_tensor = torch.from_numpy(x)
        x_stamp_tensor = torch.from_numpy(x_stamp)

        return x_tensor, x_stamp_tensor


if __name__ == '__main__':
    # Test the dataset
    print("Creating training dataset instance...")
    train_dataset = IntradayDataset(data_type='train')

    print(f"Dataset length: {len(train_dataset)}")

    if len(train_dataset) > 0:
        # Get a sample
        try_x, try_x_stamp = train_dataset[0]
        print(f"Sample feature shape: {try_x.shape}")
        print(f"Sample time feature shape: {try_x_stamp.shape}")
        print(f"\nFirst 5 timesteps of features:")
        print(try_x[:5])
        print(f"\nFirst 5 timesteps of time features:")
        print(try_x_stamp[:5])
    else:
        print("Dataset is empty.")
