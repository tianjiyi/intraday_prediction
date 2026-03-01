# Kronos Model Training for QQQ Intraday Predictions

This directory contains the complete pipeline for finetuning Kronos-base model on QQQ 1-minute intraday bars to improve 30-minute price predictions.

## Directory Structure

```
model_training/
├── README.md                      # This file
├── TRAINING_SPEC.md              # Detailed training specification
├── config.py                     # Training configuration
├── alpaca_data_fetch.py          # Fetch historical data from Alpaca
├── create_training_dataset.py    # Process and split data
├── intraday_dataset.py           # PyTorch Dataset class
├── train_tokenizer.py            # Finetune tokenizer
├── train_predictor.py            # Finetune predictor
├── utils/                        # Training utilities
│   ├── __init__.py
│   └── training_utils.py
├── data/                         # Processed datasets (created by scripts)
│   ├── raw/                      # Raw Alpaca data
│   ├── train_data.pkl
│   ├── val_data.pkl
│   └── test_data.pkl
└── outputs/                      # Model checkpoints (created by training)
    ├── tokenizer_qqq_intraday/
    │   └── checkpoints/
    │       └── best_model/
    └── predictor_qqq_intraday/
        └── checkpoints/
            └── best_model/
```

## Training Pipeline

### Step 1: Fetch Historical Data

Fetch 12 months of 1-minute bars for QQQ from Alpaca, filtered to 10:00 AM - 4:00 PM ET:

```bash
python model_training/alpaca_data_fetch.py
```

**Output**: `data/raw/QQQ_1min_raw.pkl` (~5 MB)

### Step 2: Create Training Datasets

Process raw data and split into train/val/test sets:

```bash
python model_training/create_training_dataset.py
```

**Output**:
- `data/train_data.pkl` (~4.6 MB)
- `data/val_data.pkl` (~0.8 MB)
- `data/test_data.pkl` (~0.9 MB)

### Step 3: Finetune Tokenizer

Finetune the Kronos tokenizer on QQQ distribution (2-4 hours on RTX 4090):

**Windows (single GPU)**:
```bash
python model_training/train_tokenizer.py
```

**Linux / Multi-GPU**:
```bash
torchrun --standalone --nproc_per_node=1 model_training/train_tokenizer.py
```

**Output**: `outputs/tokenizer_qqq_intraday/checkpoints/best_model/`

### Step 4: Finetune Predictor

Finetune the Kronos predictor using the finetuned tokenizer (6-10 hours on RTX 4090):

**Windows (single GPU)**:
```bash
python model_training/train_predictor.py
```

**Linux / Multi-GPU**:
```bash
torchrun --standalone --nproc_per_node=1 model_training/train_predictor.py
```

**Output**: `outputs/predictor_qqq_intraday/checkpoints/best_model/`

## Configuration

All training parameters are defined in `config.py`:

- **Symbol**: QQQ
- **Time Period**: Last 6 months (configurable)
- **Data Hours**: 9:30 AM - 4:00 PM ET (full RTH for context)
- **Trading Hours**: 10:00 AM - 4:00 PM ET (predictions only)
- **Lookback Window**: 480 bars (8 hours)
- **Prediction Horizon**: 30 bars (30 minutes)
- **Sliding Interval**: 15 minutes (configurable - see below)
- **Batch Size**: 40
- **Epochs**: 15 (configurable - see Training Modes below)
- **Learning Rates**:
  - Tokenizer: 2e-4
  - Predictor: 4e-5
- **Checkpoint Resumption**: `resume_from_checkpoint` (None for fresh training)

## Sliding Interval Configuration

The `sliding_interval_minutes` parameter controls how frequently training samples are created from the time series data. This has a **major impact on dataset size and overfitting**.

### How It Works

Each training sample uses:
- **480 bars of lookback** (8 hours of context)
- **30 bars of prediction target** (30 minutes to predict)

The sliding interval determines **how many minutes to move forward** before creating the next training sample.

### Example: 15-Minute Sliding Interval

```python
# In config.py:
self.sliding_interval_minutes = 15
```

**Training samples created**:
```
Sample 1: Lookback ends at 9:59 AM → Predict 10:00-10:30 AM
Sample 2: Lookback ends at 10:14 AM → Predict 10:15-10:45 AM  (15 mins later)
Sample 3: Lookback ends at 10:29 AM → Predict 10:30-11:00 AM  (15 mins later)
...
```

### Impact on Dataset Size and Overfitting

| Interval | Samples (6mo) | Overlap | Overfitting Risk | Recommendation |
|----------|---------------|---------|------------------|----------------|
| **1 min** | ~35,735 | 99.8% | Very High ❌ | Not recommended - severe overfitting |
| **5 min** | ~7,147 | 99.0% | High ⚠️ | Use with caution - still high overlap |
| **15 min** | ~2,382 | 97.1% | Moderate ✓ | **Recommended** - good balance |
| **30 min** | ~1,191 | 94.1% | Low ✓✓ | Good for small datasets or large models |

### Why Higher Intervals Reduce Overfitting

**Problem with 1-minute sliding**:
```
Window 1: [Bars 1-510]   → Predict bars 481-510
Window 2: [Bars 2-511]   → Predict bars 482-511
          ↑ 99.8% identical input, only 1 bar different!
```
Model memorizes these highly overlapping patterns instead of learning general features.

**Solution with 15-minute sliding**:
```
Window 1: [Bars 1-510]   → Predict bars 481-510
Window 2: [Bars 16-525]  → Predict bars 496-525
          ↑ Only 97.1% overlap, 15 bars different
```
Much harder to memorize, forces the model to learn robust patterns.

### Choosing the Right Interval

**Use 1-5 min if**:
- You have 12+ months of data
- Using strong regularization (high dropout, weight decay)
- Want maximum data utilization
- Have computational resources for long training

**Use 15 min if** (Recommended):
- Using 6 months of data
- Want to prevent overfitting
- Balance between data size and generalization
- Training the base Kronos model (102M parameters)

**Use 30 min if**:
- Have very limited data (3-6 months)
- Using a very large model
- Experiencing severe overfitting even with 15-min interval

### How to Change

Simply edit `config.py`:
```python
# For 5-minute sliding:
self.sliding_interval_minutes = 5

# For 15-minute sliding (recommended):
self.sliding_interval_minutes = 15

# For 30-minute sliding:
self.sliding_interval_minutes = 30
```

Then re-run the dataset creation and training:
```bash
python model_training/create_training_dataset.py  # NOT needed - interval is applied during training
python model_training/train_predictor.py
```

**Note**: You do NOT need to re-fetch or re-create datasets when changing the sliding interval. The interval is applied dynamically when the PyTorch dataset loads the data during training.

## Training Modes

Choose one of three training modes based on your needs. Edit `config.py` to adjust these parameters:

### Option 1: Quick Smoke Test (2-3 minutes)

**Use case**: Verify pipeline works correctly before full training

```python
# In config.py, modify:
self.epochs = 1
self.n_train_iter = 500    # 10 batches
self.n_val_iter = 100      # 2 batches
```

**What happens**:
- Processes 500 training samples (10 batches)
- Processes 100 validation samples (2 batches)
- Completes in ~2-3 minutes
- Good for testing that everything runs without errors

### Option 2: Mini Training Run (30-60 minutes)

**Use case**: Quick iteration to test hyperparameters or validate convergence

```python
# In config.py, modify:
self.epochs = 3
self.n_train_iter = 5000   # 100 batches per epoch
self.n_val_iter = 500      # 10 batches
```

**What happens**:
- Runs 3 epochs with 5,000 samples each
- Total: 300 batches (100 per epoch × 3 epochs)
- Completes in ~30-60 minutes
- Shows if loss is decreasing as expected

### Option 3: Full Production Training (8-15 hours)

**Use case**: Final model training for deployment

```python
# In config.py, modify:
self.epochs = 30
self.n_train_iter = 100000  # Uses all ~65k samples
self.n_val_iter = 20000     # Uses all ~10k samples
```

**What happens**:
- Runs 30 full epochs through entire dataset
- Each epoch: ~1,304 batches (65,217 samples / 50 batch_size)
- Total: ~39,000 iterations
- Tokenizer: ~2-4 hours
- Predictor: ~6-10 hours
- Produces production-ready finetuned model

### Understanding Epochs vs Iterations

**Important**: 1 epoch ≠ 1 iteration!

| Term | Meaning |
|------|---------|
| **Iteration** | 1 batch processed (50 samples) |
| **Epoch** | Full pass through dataset |
| **1 epoch** | ~1,304 iterations (for full dataset) |
| **30 epochs** | ~39,000 iterations total |

**Current configuration** (as modified): Option 1 (Quick Smoke Test)

## Hardware Requirements

- **GPU**: RTX 3090+ (24GB+) or RTX 4090/A6000 (32GB+ recommended)
- **Storage**: ~1 GB total (data: ~6 MB, checkpoints: ~400 MB each)
- **Training Time**: ~8-15 hours total on RTX 4090

## Expected Results

### Current Base Model Performance
- F1 Score: 0.48
- AUC-ROC: 0.63
- Precision: 0.87
- Recall: 0.33

### Target After Finetuning
- F1 Score: >0.60 (+25%)
- AUC-ROC: >0.75 (+20%)
- Better probability calibration

## Integration with Live System

After training, update `config.yaml` to use the finetuned model:

```yaml
model:
  checkpoint: "./model_training/outputs/predictor_qqq_intraday/checkpoints/best_model"
  tokenizer: "./model_training/outputs/tokenizer_qqq_intraday/checkpoints/best_model"
```

## Resuming Training from Checkpoint

If training is interrupted (crashes, power loss, etc.), you can resume from the last saved checkpoint.

### Checkpoint Types

**End-of-Epoch Checkpoints** (saved after each full epoch):
```
./model_training/outputs/predictor_qqq_intraday/checkpoints/checkpoint_epoch_1.pt
./model_training/outputs/predictor_qqq_intraday/checkpoints/checkpoint_epoch_2.pt
```

**Mid-Epoch Checkpoints** (saved every 200 steps, ~every 15 minutes):
```
./model_training/outputs/predictor_qqq_intraday/checkpoints/checkpoint_epoch_1_step_200.pt
./model_training/outputs/predictor_qqq_intraday/checkpoints/checkpoint_epoch_1_step_400.pt
./model_training/outputs/predictor_qqq_intraday/checkpoints/checkpoint_epoch_1_step_600.pt
```

Mid-epoch checkpoints protect against system crashes during long training epochs.

### Step 1: Find the latest checkpoint

After a crash, list checkpoints by timestamp:
```bash
ls -lt model_training/outputs/predictor_qqq_intraday/checkpoints/*.pt
```

Use the most recent `.pt` file.

### Step 2: Update config.py

```python
# In config.py, set:
self.resume_from_checkpoint = "./model_training/outputs/predictor_qqq_intraday/checkpoints/checkpoint_epoch_1_step_600.pt"
```

### Step 3: Run training normally

**Windows**:
```bash
python model_training/train_predictor.py
```

**Linux/Multi-GPU**:
```bash
torchrun --standalone --nproc_per_node=1 model_training/train_predictor.py
```

### What gets restored:
- ✅ Model weights from the checkpoint
- ✅ Optimizer state (Adam momentum)
- ✅ Learning rate scheduler state
- ✅ Best validation loss
- ✅ Current epoch number
- ✅ Current step number (for mid-epoch checkpoints)

### Examples:

**Example 1: Resume from mid-epoch checkpoint**
- Crash happened at step 750 of epoch 1
- Latest checkpoint: `checkpoint_epoch_1_step_600.pt`
- Training resumes from step 600, skips steps 1-600, continues 600-1305

**Example 2: Resume from end-of-epoch checkpoint**
- Completed epoch 2, crash during epoch 3
- Latest checkpoint: `checkpoint_epoch_2.pt`
- Training resumes from epoch 3 start

## Troubleshooting

### Windows: `torchrun` fails with libuv error
**Error**: `use_libuv was requested but PyTorch was built without libuv support`

**Solution**: Use standalone Python execution instead:
```bash
python model_training/train_tokenizer.py
python model_training/train_predictor.py
```

The training scripts auto-detect whether they're running under `torchrun` (DDP mode) or standalone (single-GPU mode).

### Out of Memory
- Reduce `batch_size` in `config.py` (try 25 or 20)
- Reduce `n_train_iter` and `n_val_iter`

### Insufficient Data
- Adjust date range in `config.py`
- Check Alpaca API limits and retry

### Training Too Slow
- Ensure CUDA is available: `python -c "import torch; print(torch.cuda.is_available())"`
- Check GPU utilization: `nvidia-smi`
- Linux/Multi-GPU: Use `torchrun` for distributed training

## References

- [TRAINING_SPEC.md](TRAINING_SPEC.md) - Complete technical specification
- [Kronos Paper](https://arxiv.org/abs/2508.02739)
- [Kronos GitHub](https://github.com/shiyu-coder/Kronos)
