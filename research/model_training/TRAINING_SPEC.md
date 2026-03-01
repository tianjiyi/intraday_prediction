# Kronos Model Training Specification
## Intraday 30-Minute Prediction for QQQ

---

## Overview
This document specifies the training pipeline for finetuning Kronos-base model on QQQ 1-minute intraday bars to improve 30-minute price predictions.

## Training Objective
Finetune the pretrained `Kronos-base` (102M parameters) model to predict QQQ price movements 30 minutes ahead using historical 1-minute bar data.

---

## Data Specification

### Symbol
- **Primary Symbol**: QQQ (Nasdaq-100 ETF)
- **Rationale**: High liquidity, consistent volume, clean intraday patterns

### Time Period
- **Total Data**: Last 12 months from current date (October 2024 - October 2025)
- **Train Split**: First 9 months (~70%)
- **Validation Split**: Next 1.5 months (~12.5%)
- **Test Split**: Last 1.5 months (~12.5%)

### Bar Timeframe
- **Base Data**: 1-minute bars
- **Context Window**: 480 bars (8 hours of trading)
- **Prediction Horizon**: 30 bars (30 minutes ahead)

### Features (OHLCV)
```python
feature_list = ['open', 'high', 'low', 'close', 'volume']
```

### Time Features
```python
time_feature_list = ['minute', 'hour', 'weekday', 'day', 'month']
```
- Extracted from timestamp
- Used by Kronos model for temporal pattern recognition

### Market Hours
- **Data Includes**: 9:30 AM - 4:00 PM ET (Full Regular Trading Hours)
- **Predictions/Trading**: 10:00 AM - 4:00 PM ET only
- **Key Design Decision**:
  - ✅ **Include 9:30-10:00 AM bars for CONTEXT** (provides opening price action)
  - ❌ **Don't predict/trade during 9:30-10:00 AM** (too volatile)
  - This ensures we have proper context when making 10:00 AM predictions
  - Avoids training/inference mismatch (live system also uses full RTH for context)

---

## Data Preparation Pipeline

### Step 1: Fetch Historical Data
**Script**: `alpaca_data_fetch.py`

**Process**:
1. Use Alpaca API to fetch 1-minute bars for QQQ
2. Date range: Last 12 months
3. Filter to 9:30 AM - 4:00 PM ET (full RTH for context)
4. Validate data quality (no gaps, correct OHLCV format)

**Expected Output**:
- ~97,000 bars per symbol (12 months × 250 trading days × 390 minutes)
- 390 minutes per day: 9:30 AM to 4:00 PM (6.5 hours RTH)

### Step 2: Create Sliding Windows
**Script**: `create_training_dataset.py`

**Process**:
1. Generate sliding windows:
   - Input: 480 bars (lookback)
   - Output: 30 bars (prediction target)
   - Total window: 510 bars
2. Slide window forward by 30 minutes (step_size = 30)
3. Generate time features for each window
4. Instance-level normalization:
   ```python
   x_mean = np.mean(x, axis=0)
   x_std = np.std(x, axis=0)
   x_normalized = (x - x_mean) / (x_std + 1e-5)
   x_clipped = np.clip(x_normalized, -5.0, 5.0)
   ```

**Expected Samples** (after filtering to predictions >= 10:00 AM):
- Total windows: ~65,000 (after removing predictions into 9:30-10:00 AM)
- Train: ~52,000 samples (9 months)
- Validation: ~7,000 samples (1.5 months)
- Test: ~8,000 samples (1.5 months)

### Step 3: Data Splitting
**Time-based split** (NO random shuffle - preserve temporal order):

```python
# Approximate dates (adjust based on actual 12-month period)
train_range = ["2024-10-04", "2025-07-04"]      # 9 months
val_range = ["2025-07-05", "2025-08-20"]        # 1.5 months
test_range = ["2025-08-21", "2025-10-04"]       # 1.5 months
```

### Step 4: Save Processed Data
**Format**: Pickle files for efficient loading

```
model_training/data/
├── train_data.pkl       # Training samples
├── val_data.pkl         # Validation samples
└── test_data.pkl        # Test samples
```

**Data structure**:
```python
{
    'QQQ': pd.DataFrame(
        index=datetime,
        columns=['open', 'high', 'low', 'close', 'volume']
    )
}
```

---

## Training Configuration

### Model Architecture
- **Base Model**: `NeoQuasar/Kronos-base` (102M params)
- **Tokenizer**: `NeoQuasar/Kronos-Tokenizer-base`
- **Max Context**: 512 (model limit)
- **Lookback Window**: 480 bars
- **Predict Window**: 30 bars

### Training Hyperparameters

#### Tokenizer Finetuning
```python
{
    'epochs': 30,
    'batch_size': 50,           # Per GPU
    'learning_rate': 2e-4,
    'optimizer': 'AdamW',
    'adam_beta1': 0.9,
    'adam_beta2': 0.95,
    'weight_decay': 0.1,
    'scheduler': 'OneCycleLR',
    'warmup_pct': 0.03,
    'gradient_clip': 3.0,
    'seed': 100
}
```

#### Predictor Finetuning
```python
{
    'epochs': 30,
    'batch_size': 50,           # Per GPU
    'learning_rate': 4e-5,      # Lower than tokenizer
    'optimizer': 'AdamW',
    'adam_beta1': 0.9,
    'adam_beta2': 0.95,
    'weight_decay': 0.1,
    'scheduler': 'OneCycleLR',
    'warmup_pct': 0.03,
    'gradient_clip': 3.0,
    'seed': 100
}
```

### Training Strategy
1. **Two-stage finetuning** (Kronos official approach):
   - Stage 1: Finetune tokenizer on QQQ distribution
   - Stage 2: Finetune predictor using finetuned tokenizer

2. **Validation-based early stopping**:
   - Save best model based on validation loss
   - Monitor overfitting

3. **Gradient accumulation** (if memory limited):
   - Accumulation steps: 1 (can increase to 2-4 if needed)
   - Effective batch size = batch_size × accumulation_steps

---

## Training Scripts

### Directory Structure
```
model_training/
├── TRAINING_SPEC.md              # This file
├── config.py                     # Training configuration
├── alpaca_data_fetch.py          # Fetch data from Alpaca
├── create_training_dataset.py    # Process and split data
├── intraday_dataset.py           # PyTorch Dataset class
├── train_tokenizer.py            # Finetune tokenizer
├── train_predictor.py            # Finetune predictor
├── evaluate_model.py             # Test set evaluation
├── data/                         # Processed datasets
│   ├── raw/                      # Raw Alpaca data
│   ├── train_data.pkl
│   ├── val_data.pkl
│   └── test_data.pkl
└── outputs/
    ├── tokenizer/                # Tokenizer checkpoints
    │   └── checkpoints/
    │       └── best_model/
    └── predictor/                # Predictor checkpoints
        └── checkpoints/
            └── best_model/
```

### Execution Order
```bash
# Step 1: Fetch raw data from Alpaca
python model_training/alpaca_data_fetch.py

# Step 2: Process and create training datasets
python model_training/create_training_dataset.py

# Step 3: Finetune tokenizer (single GPU)
torchrun --standalone --nproc_per_node=1 model_training/train_tokenizer.py

# Step 4: Finetune predictor (single GPU)
torchrun --standalone --nproc_per_node=1 model_training/train_predictor.py

# Step 5: Evaluate on test set
python model_training/evaluate_model.py
```

---

## Hardware Requirements

### GPU
- **Minimum**: RTX 3090 (24GB VRAM)
- **Recommended**: RTX 4090 / A6000 (32GB+ VRAM)
- **Current Setup**: 32GB VRAM ✓

### Memory Usage Estimates
- Model weights (Kronos-base): ~8GB
- Batch processing (batch_size=50): ~10-12GB
- Peak usage: ~20GB VRAM
- **Safe margin**: 32GB is sufficient

### Storage
- Raw data: ~500MB (1 year of 1-min bars)
- Processed datasets: ~2-3GB
- Model checkpoints: ~1GB per checkpoint
- Total: ~10GB

### Training Time Estimates (Single RTX 4090)
- Data preparation: ~15-30 minutes
- Tokenizer finetuning: ~2-4 hours (30 epochs)
- Predictor finetuning: ~6-10 hours (30 epochs)
- **Total**: ~8-15 hours

---

## Evaluation Metrics

### During Training
- **Loss**: Cross-entropy loss (tokenized sequences)
- **Validation Loss**: Monitor for overfitting
- **Learning Rate**: Track scheduler progression

### Post-Training (Test Set)
Use existing `backtesting/metrics.py`:

#### Classification Metrics (Direction)
- Accuracy
- Precision / Recall / F1 Score
- AUC-ROC
- Confusion Matrix

#### Regression Metrics (Price Level)
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- MAPE (Mean Absolute Percentage Error)
- R² Score

#### Probabilistic Metrics
- Brier Score
- Log Loss
- Calibration Error

### Success Criteria
**Current Base Model Performance**:
- F1 Score: 0.4819
- AUC-ROC: 0.6325
- MAE: ~(to be measured)

**Target After Finetuning**:
- F1 Score: >0.60 (+25% improvement)
- AUC-ROC: >0.75 (+20% improvement)
- Better probability calibration (smoother distributions)

---

## Output & Integration

### Finetuned Model Artifacts
```
outputs/
├── tokenizer/
│   └── checkpoints/
│       └── best_model/
│           ├── config.json
│           ├── model.safetensors
│           └── tokenizer.json
└── predictor/
    └── checkpoints/
        └── best_model/
            ├── config.json
            ├── model.safetensors
            └── generation_config.json
```

### Integration with Live System
Update `config.yaml`:
```yaml
model:
  # Option 1: Use base model
  checkpoint: "NeoQuasar/Kronos-base"
  tokenizer: "NeoQuasar/Kronos-Tokenizer-base"

  # Option 2: Use finetuned model
  checkpoint: "./model_training/outputs/predictor/checkpoints/best_model"
  tokenizer: "./model_training/outputs/tokenizer/checkpoints/best_model"
```

---

## Risk Mitigation

### Overfitting Prevention
- Validation-based early stopping
- Weight decay (L2 regularization)
- Gradient clipping
- Monitor train vs validation loss divergence

### Data Quality
- Validate no missing bars in RTH
- Check for outliers / data errors
- Ensure timezone consistency (US/Eastern)

### Reproducibility
- Fixed random seed: 100
- Deterministic data splitting
- Save all hyperparameters in config

---

## Next Steps

### Immediate Actions
1. ✅ Create `model_training/` directory
2. ✅ Write this specification
3. ⏳ Implement `config.py`
4. ⏳ Implement `alpaca_data_fetch.py`
5. ⏳ Implement `create_training_dataset.py`
6. ⏳ Adapt `intraday_dataset.py` from Kronos
7. ⏳ Adapt `train_tokenizer.py` from Kronos
8. ⏳ Adapt `train_predictor.py` from Kronos
9. ⏳ Run training pipeline
10. ⏳ Evaluate and compare results

---

## References
- Kronos Paper: https://arxiv.org/abs/2508.02739
- Kronos GitHub: https://github.com/shiyu-coder/Kronos
- Kronos Models: https://huggingface.co/NeoQuasar
- Official Finetuning Example: `Kronos/finetune/`

---

**Document Version**: 1.0
**Created**: 2025-10-04
**Author**: Claude Code
**Status**: Ready for Implementation
