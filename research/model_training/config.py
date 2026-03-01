"""
Training Configuration for Kronos Finetuning
Intraday 30-minute prediction for QQQ
"""
import os
from datetime import datetime, timedelta


class TrainingConfig:
    """
    Configuration class for Kronos model finetuning on QQQ intraday data.
    """

    def __init__(self):
        # =================================================================
        # Data & Feature Parameters
        # =================================================================

        # Symbol configuration
        self.symbol = 'QQQ'

        # Time period (last 3 years)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=1095)  # 3 years = 365 * 3

        self.dataset_begin_time = start_date.strftime("%Y-%m-%d")
        self.dataset_end_time = end_date.strftime("%Y-%m-%d")

        # Market hours for PREDICTION (when we trade)
        # Note: Data includes 9:30-10:00 AM for context, but we only
        # make predictions/trade from 10:00 AM onwards
        self.trading_start_hour = 10
        self.trading_start_minute = 0
        self.trading_end_hour = 16
        self.trading_end_minute = 0

        # Data includes full RTH for context (9:30 AM - 4:00 PM)
        self.data_start_hour = 9
        self.data_start_minute = 30

        # Sliding window parameters
        self.lookback_window = 480  # 8 hours of 1-min bars (context)
        self.predict_window = 30    # 30 minutes ahead (prediction target)
        self.max_context = 512      # Kronos maximum context length

        # Sliding interval (how many minutes between consecutive training samples)
        # Lower values = more samples but higher overlap (more overfitting risk)
        # Higher values = fewer samples but lower overlap (better generalization)
        # Examples: 1 (every minute), 5 (every 5 mins), 15 (every 15 mins), 30 (every 30 mins)
        self.sliding_interval_minutes = 30  # Create training sample every N minutes

        # Features to be used from raw data
        # Note: Kronos expects 6 features. We calculate 'amount' from volume * close
        self.feature_list = ['open', 'high', 'low', 'close', 'volume', 'amount']

        # Time-based features to be generated
        self.time_feature_list = ['minute', 'hour', 'weekday', 'day', 'month']

        # =================================================================
        # Dataset Splitting & Paths
        # =================================================================

        # Calculate split dates (27 months train, 4.5 months val, 4.5 months test)
        train_end = start_date + timedelta(days=810)  # ~27 months (75%)
        val_end = train_end + timedelta(days=135)      # ~4.5 months (12.5%)

        self.train_time_range = [
            start_date.strftime("%Y-%m-%d"),
            train_end.strftime("%Y-%m-%d")
        ]
        self.val_time_range = [
            (train_end + timedelta(days=1)).strftime("%Y-%m-%d"),
            val_end.strftime("%Y-%m-%d")
        ]
        self.test_time_range = [
            (val_end + timedelta(days=1)).strftime("%Y-%m-%d"),
            end_date.strftime("%Y-%m-%d")
        ]

        # Dataset paths
        self.dataset_path = "./model_training/data"
        self.raw_data_path = "./model_training/data/raw"

        # =================================================================
        # Alpaca API Configuration
        # =================================================================

        # Load from parent config.yaml
        import yaml
        from pathlib import Path
        config_path = Path(__file__).parent.parent / "config.yaml"
        with open(config_path, 'r') as f:
            parent_config = yaml.safe_load(f)

        self.alpaca_api_key = parent_config['ALPACA_KEY_ID']
        self.alpaca_secret_key = parent_config['ALPACA_SECRET_KEY']

        # =================================================================
        # Training Hyperparameters
        # =================================================================

        # Data normalization
        self.clip = 5.0  # Clipping value for normalized data

        # Training epochs
        self.epochs = 15  # Increased for 6-month data
        self.log_interval = 100  # Log every N batches

        # Batch size per GPU
        self.batch_size = 40

        # Number of samples per epoch (for large datasets)
        # Set high to use ALL available samples (will be capped by dataset size)
        self.n_train_iter = 100000  # Will use all ~24-26K available training samples
        self.n_val_iter = 20000     # Will use all ~4-4.5K available validation samples

        # Learning rates
        self.tokenizer_learning_rate = 2e-4
        self.predictor_learning_rate = 4e-5

        # Gradient accumulation
        self.accumulation_steps = 1

        # AdamW optimizer parameters
        self.adam_beta1 = 0.9
        self.adam_beta2 = 0.95
        self.adam_weight_decay = 0.1

        # Random seed for reproducibility
        self.seed = 100

        # Number of dataloader workers
        self.num_workers = 2

        # =================================================================
        # Checkpoint Resumption
        # =================================================================

        # Path to checkpoint file to resume training from
        # Set to None for fresh training, or path to resume
        # Example: "./model_training/outputs/predictor_qqq_intraday/checkpoints/checkpoint_epoch_1.pt"
        self.resume_from_checkpoint = None

        # Mid-epoch checkpoint frequency (save every N steps to prevent data loss from crashes)
        # Set to None to disable mid-epoch checkpointing, or number of steps (e.g., 200)
        self.checkpoint_every_n_steps = 200  # Save every 200 steps (~every 15 mins for predictor)

        # =================================================================
        # Experiment Logging & Saving
        # =================================================================

        # Disable Comet ML for now
        self.use_comet = False

        # Model checkpoint paths
        self.save_path = "./model_training/outputs"
        self.tokenizer_save_folder_name = 'tokenizer_qqq_intraday'
        self.predictor_save_folder_name = 'predictor_qqq_intraday'

        # =================================================================
        # Model & Checkpoint Paths
        # =================================================================

        # Pretrained models from HuggingFace
        # Note: Kronos-small uses the same tokenizer as Kronos-base
        self.pretrained_tokenizer_path = "NeoQuasar/Kronos-Tokenizer-base"
        self.pretrained_predictor_path = "NeoQuasar/Kronos-small"

        # Paths to finetuned models (will be created during training)
        self.finetuned_tokenizer_path = f"{self.save_path}/{self.tokenizer_save_folder_name}/checkpoints/best_model"
        self.finetuned_predictor_path = f"{self.save_path}/{self.predictor_save_folder_name}/checkpoints/best_model"

        # =================================================================
        # Inference & Evaluation Parameters
        # =================================================================

        self.inference_T = 1.0           # Temperature for sampling
        self.inference_top_p = 0.9       # Nucleus sampling
        self.inference_top_k = 0         # Top-k sampling (0 = disabled)
        self.inference_sample_count = 30  # Number of samples for Monte Carlo
        self.eval_batch_size = 100       # Batch size for evaluation

    def print_summary(self):
        """Print configuration summary"""
        print("=" * 80)
        print("TRAINING CONFIGURATION SUMMARY")
        print("=" * 80)
        print(f"\nSymbol: {self.symbol}")
        print(f"Time Period: {self.dataset_begin_time} to {self.dataset_end_time}")
        print(f"Data Hours: {self.data_start_hour:02d}:{self.data_start_minute:02d} - {self.trading_end_hour:02d}:{self.trading_end_minute:02d} ET (full RTH)")
        print(f"Trading Hours: {self.trading_start_hour:02d}:{self.trading_start_minute:02d} - {self.trading_end_hour:02d}:{self.trading_end_minute:02d} ET (predictions only)")
        print(f"\nLookback Window: {self.lookback_window} bars")
        print(f"Prediction Window: {self.predict_window} bars")
        print(f"Sliding Interval: {self.sliding_interval_minutes} minutes")
        print(f"Max Context: {self.max_context} bars")
        print(f"\nTrain Period: {self.train_time_range[0]} to {self.train_time_range[1]}")
        print(f"Val Period: {self.val_time_range[0]} to {self.val_time_range[1]}")
        print(f"Test Period: {self.test_time_range[0]} to {self.test_time_range[1]}")
        print(f"\nBatch Size: {self.batch_size}")
        print(f"Epochs: {self.epochs}")
        print(f"Tokenizer LR: {self.tokenizer_learning_rate}")
        print(f"Predictor LR: {self.predictor_learning_rate}")
        print(f"\nPretrained Tokenizer: {self.pretrained_tokenizer_path}")
        print(f"Pretrained Predictor: {self.pretrained_predictor_path}")
        print(f"\nOutput Path: {self.save_path}")
        print("=" * 80)


if __name__ == '__main__':
    # Test configuration
    config = TrainingConfig()
    config.print_summary()
