"""
Test the scratch-trained Kronos model on W_Bottom pattern prediction.

This script:
1. Loads a sample W_Bottom pattern
2. Runs prediction with base model vs scratch-trained model
3. Compares predicted breakout with actual outcome

Usage:
    python -m pattern_recognition.test_kronos_scratch
"""

import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "Kronos"))

from model.kronos import KronosTokenizer, Kronos, KronosPredictor

# Paths
DATA_DIR = Path(__file__).parent.parent / "kronos_training_data"
SCRATCH_DIR = Path(__file__).parent.parent / "kronos_w_bottom_scratch" / "best_model"
OUTPUT_DIR = Path(__file__).parent.parent / "kronos_scratch_test_output"


def load_val_samples(n_samples: int = 5):
    """Load validation samples for testing."""
    with open(DATA_DIR / "val_samples.json", 'r') as f:
        samples = json.load(f)
    return samples[:n_samples]


def prepare_input(sample: dict, max_len: int = 300):
    """Prepare input tensor from sample."""
    input_data = np.array(sample['input'])  # Shape: (seq_len, 5)

    # Add amount column
    amount = input_data[:, 4] * input_data[:, 3]
    input_data = np.column_stack([input_data, amount])

    # Pad/truncate
    if len(input_data) > max_len:
        input_data = input_data[-max_len:]
    elif len(input_data) < max_len:
        pad_len = max_len - len(input_data)
        input_data = np.vstack([np.zeros((pad_len, 6)), input_data])

    return input_data


def calculate_atr_for_input(input_data: np.ndarray, period: int = 14) -> float:
    """
    Calculate Average True Range for input data.

    Args:
        input_data: numpy array with columns [open, high, low, close, volume, amount]
        period: ATR period (default 14)

    Returns:
        ATR value
    """
    if len(input_data) < 2:
        return 0.0

    high = input_data[:, 1]
    low = input_data[:, 2]
    close = input_data[:, 3]

    tr_list = []
    for i in range(1, len(input_data)):
        tr = max(
            high[i] - low[i],
            abs(high[i] - close[i-1]),
            abs(low[i] - close[i-1])
        )
        tr_list.append(tr)

    if len(tr_list) < period:
        return np.mean(tr_list) if tr_list else 0.0

    return np.mean(tr_list[-period:])


def run_prediction(predictor, input_data, pred_len: int = 60, n_samples: int = 10, use_pct_norm: bool = False):
    """Run prediction and return mean prediction path."""
    # Prepare DataFrame
    df = pd.DataFrame(input_data, columns=['open', 'high', 'low', 'close', 'volume', 'amount'])

    # Create timestamps
    x_timestamp = pd.Series(pd.date_range('2025-01-01 09:30', periods=len(df), freq='1min'))
    y_timestamp = pd.Series(pd.date_range(
        x_timestamp.iloc[-1] + pd.Timedelta(minutes=1),
        periods=pred_len,
        freq='1min'
    ))

    # Run prediction with lower temperature for more stable predictions
    pred_result = predictor.predict(
        df=df,
        x_timestamp=x_timestamp,
        y_timestamp=y_timestamp,
        pred_len=pred_len,
        sample_count=n_samples,
        T=0.2,         # Lowered from 0.8 to reduce sampling randomness
        top_p=0.95,    # Slightly higher for some diversity
        return_samples=True
    )

    return pred_result


def run_prediction_pct_norm(model, tokenizer, input_data, device, pred_len: int = 60, n_samples: int = 10):
    """
    Run prediction with PERCENTAGE-BASED normalization.

    This matches the training normalization for the scratch model,
    making predictions comparable across different price scales.
    """
    import torch
    from model.kronos import auto_regressive_inference, calc_time_stamps

    # Use percentage-based normalization (matches training)
    x_mean = np.mean(input_data, axis=0)
    price_mean = np.mean(input_data[:, 3])  # Mean close price as reference
    x_std_abs = np.std(input_data, axis=0)  # Absolute std in $
    x_std = x_std_abs / price_mean * 100    # Convert to % std

    # Normalize
    x_normalized = (input_data - x_mean) / (x_std + 1e-5)
    x_normalized = np.clip(x_normalized, -5.0, 5.0)

    # Create timestamps
    x_timestamp = pd.Series(pd.date_range('2025-01-01 09:30', periods=len(input_data), freq='1min'))
    y_timestamp = pd.Series(pd.date_range(
        x_timestamp.iloc[-1] + pd.Timedelta(minutes=1),
        periods=pred_len,
        freq='1min'
    ))

    x_stamp = calc_time_stamps(x_timestamp).values.astype(np.float32)
    y_stamp = calc_time_stamps(y_timestamp).values.astype(np.float32)

    # Convert to tensors
    x_tensor = torch.from_numpy(x_normalized.astype(np.float32)).unsqueeze(0).to(device)
    x_stamp_tensor = torch.from_numpy(x_stamp).unsqueeze(0).to(device)
    y_stamp_tensor = torch.from_numpy(y_stamp).unsqueeze(0).to(device)

    # Run inference
    with torch.no_grad():
        preds = auto_regressive_inference(
            tokenizer, model, x_tensor, x_stamp_tensor, y_stamp_tensor,
            max_context=512, pred_len=pred_len, clip=5,
            T=0.2, top_k=0, top_p=0.95, sample_count=n_samples,
            verbose=True, return_raw=True
        )

    # preds shape: [1, n_samples, seq_len, 6]
    preds = preds[0, :, -pred_len:, :]  # [n_samples, pred_len, 6]

    # Denormalize with percentage-based stats
    preds = preds * (x_std + 1e-5) + x_mean

    # Build result dict similar to KronosPredictor.predict()
    samples = []
    for i in range(preds.shape[0]):
        df_i = pd.DataFrame(
            preds[i],
            columns=['open', 'high', 'low', 'close', 'volume', 'amount'],
            index=y_timestamp,
        )
        samples.append(df_i)

    mean_vals = preds.mean(axis=0)
    mean_df = pd.DataFrame(
        mean_vals,
        columns=['open', 'high', 'low', 'close', 'volume', 'amount'],
        index=y_timestamp,
    )

    return {"samples": samples, "mean": mean_df}


def run_prediction_atr_norm(model, tokenizer, input_data, device, pred_len: int = 60, n_samples: int = 10):
    """
    Run prediction with ATR-BASED normalization.

    This matches the ATR training normalization for volatility-aware predictions.
    """
    import torch
    from model.kronos import auto_regressive_inference, calc_time_stamps

    # Calculate ATR for this input
    atr = calculate_atr_for_input(input_data)
    if atr < 1e-6:
        atr = 1.0  # Fallback

    # Calculate mean for centering
    x_mean = np.mean(input_data, axis=0)

    # Use ATR for price columns, std for volume/amount
    x_std = np.zeros(6)
    x_std[:4] = atr  # Price columns normalized by ATR
    x_std[4:] = np.std(input_data[:, 4:], axis=0)  # Volume/amount by std

    # Normalize
    x_normalized = (input_data - x_mean) / (x_std + 1e-5)
    x_normalized = np.clip(x_normalized, -5.0, 5.0)

    # Create timestamps
    x_timestamp = pd.Series(pd.date_range('2025-01-01 09:30', periods=len(input_data), freq='1min'))
    y_timestamp = pd.Series(pd.date_range(
        x_timestamp.iloc[-1] + pd.Timedelta(minutes=1),
        periods=pred_len,
        freq='1min'
    ))

    x_stamp = calc_time_stamps(x_timestamp).values.astype(np.float32)
    y_stamp = calc_time_stamps(y_timestamp).values.astype(np.float32)

    # Convert to tensors
    x_tensor = torch.from_numpy(x_normalized.astype(np.float32)).unsqueeze(0).to(device)
    x_stamp_tensor = torch.from_numpy(x_stamp).unsqueeze(0).to(device)
    y_stamp_tensor = torch.from_numpy(y_stamp).unsqueeze(0).to(device)

    # Run inference
    with torch.no_grad():
        preds = auto_regressive_inference(
            tokenizer, model, x_tensor, x_stamp_tensor, y_stamp_tensor,
            max_context=512, pred_len=pred_len, clip=5,
            T=0.2, top_k=0, top_p=0.95, sample_count=n_samples,
            verbose=True, return_raw=True
        )

    # preds shape: [1, n_samples, seq_len, 6]
    preds = preds[0, :, -pred_len:, :]  # [n_samples, pred_len, 6]

    # Denormalize with ATR-based stats
    preds = preds * (x_std + 1e-5) + x_mean

    # Build result dict
    samples = []
    for i in range(preds.shape[0]):
        df_i = pd.DataFrame(
            preds[i],
            columns=['open', 'high', 'low', 'close', 'volume', 'amount'],
            index=y_timestamp,
        )
        samples.append(df_i)

    mean_vals = preds.mean(axis=0)
    mean_df = pd.DataFrame(
        mean_vals,
        columns=['open', 'high', 'low', 'close', 'volume', 'amount'],
        index=y_timestamp,
    )

    return {"samples": samples, "mean": mean_df}


def anchor_to_last_close(prediction_close, last_input_close):
    """
    Shift entire prediction curve to start from last input close.

    This eliminates the gap caused by tokenizer quantization and sampling
    while preserving the predicted shape and direction.
    """
    offset = prediction_close[0] - last_input_close
    return prediction_close - offset


def load_scratch_model(model_path: Path, device):
    """Load the scratch-trained model."""
    checkpoint = torch.load(model_path / "model.pt", map_location=device)
    config = checkpoint['config']

    model = Kronos(
        s1_bits=config['s1_bits'],
        s2_bits=config['s2_bits'],
        n_layers=config['n_layers'],
        d_model=config['d_model'],
        n_heads=config['n_heads'],
        ff_dim=config['ff_dim'],
        ffn_dropout_p=0.0,
        attn_dropout_p=0.0,
        resid_dropout_p=0.0,
        token_dropout_p=0.0,  # No dropout during inference
        learn_te=False
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    return model


def compare_predictions(sample: dict, base_predictor, scratch_model, tokenizer, device, use_atr: bool = False):
    """Compare base vs scratch-trained model predictions."""
    # Prepare input - RAW data (Kronos predictor does its own normalization)
    input_data = prepare_input(sample)
    target_data = np.array(sample['target'])

    # Get last close for return calculation
    last_input_close = input_data[-1, 3]
    actual_close = target_data[:, 3]

    # Run predictions
    print(f"\n  Running base model prediction...")
    base_pred = run_prediction(base_predictor, input_data)

    if use_atr:
        print(f"  Running scratch-trained model prediction (with ATR normalization)...")
        scratch_pred = run_prediction_atr_norm(scratch_model, tokenizer, input_data, device)
    else:
        print(f"  Running scratch-trained model prediction (with % normalization)...")
        scratch_pred = run_prediction_pct_norm(scratch_model, tokenizer, input_data, device)

    # Get predicted close prices
    base_close = base_pred['mean']['close'].values
    scratch_close = scratch_pred['mean']['close'].values

    # Anchor predictions to start from last input close
    # This eliminates gap caused by tokenizer quantization + sampling
    base_close = anchor_to_last_close(base_close, last_input_close)
    scratch_close = anchor_to_last_close(scratch_close, last_input_close)

    # Calculate returns
    actual_return = (actual_close[-1] - last_input_close) / last_input_close * 100
    base_return = (base_close[-1] - last_input_close) / last_input_close * 100
    scratch_return = (scratch_close[-1] - last_input_close) / last_input_close * 100

    # Debug output
    print(f"    Last input close: ${last_input_close:.2f}")
    print(f"    Actual final close: ${actual_close[-1]:.2f} ({actual_return:+.2f}%)")
    print(f"    Base prediction: ${base_close[-1]:.2f} ({base_return:+.2f}%)")
    print(f"    Scratch prediction: ${scratch_close[-1]:.2f} ({scratch_return:+.2f}%)")

    return {
        'date': sample['date'],
        'input_close': input_data[-60:, 3],
        'actual_close': actual_close,
        'base_pred_close': base_close,
        'scratch_pred_close': scratch_close,
        'actual_return': actual_return,
        'base_return': base_return,
        'scratch_return': scratch_return,
        'base_error': abs(base_return - actual_return),
        'scratch_error': abs(scratch_return - actual_return)
    }


def plot_comparison(result: dict, output_path: Path):
    """Plot comparison of predictions vs actual."""
    fig, ax = plt.subplots(figsize=(14, 6))

    # Input data (last 60 bars)
    input_x = np.arange(len(result['input_close']))
    ax.plot(input_x, result['input_close'], 'b-', label='Input (historical)', linewidth=1.5)

    # Predictions start after input
    pred_start = len(result['input_close'])
    pred_x = np.arange(pred_start, pred_start + len(result['actual_close']))

    ax.plot(pred_x, result['actual_close'], 'g-', label='Actual', linewidth=2)
    ax.plot(pred_x, result['base_pred_close'], 'r--', label='Base Model', linewidth=1.5, alpha=0.7)
    ax.plot(pred_x, result['scratch_pred_close'], 'c-', label='Scratch-trained', linewidth=2)

    ax.axvline(x=pred_start, color='gray', linestyle=':', alpha=0.5, label='Pattern End')

    ax.set_title(f"W_Bottom Prediction - {result['date']}\n"
                 f"Actual: {result['actual_return']:+.2f}%, Base: {result['base_return']:+.2f}%, "
                 f"Scratch: {result['scratch_return']:+.2f}%")
    ax.set_xlabel('Bar')
    ax.set_ylabel('Close Price')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=100)
    plt.close()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Test scratch-trained Kronos model")
    parser.add_argument('--n-samples', type=int, default=50, help='Number of samples to test')
    parser.add_argument('--use-atr', action='store_true', help='Use ATR normalization')
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Kronos Scratch-Trained Model Test")
    print("=" * 60)
    print(f"Testing with {args.n_samples} samples")

    # Check device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load tokenizer
    print("\n1. Loading tokenizer...")
    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    tokenizer.eval().to(device)

    # Load base model
    print("\n2. Loading base model...")
    base_model = Kronos.from_pretrained("NeoQuasar/Kronos-base")
    base_model.to(device)
    base_predictor = KronosPredictor(
        model=base_model,
        tokenizer=tokenizer,
        device=str(device),
        max_context=512
    )

    # Load scratch-trained model (we'll use raw model + tokenizer for % normalization)
    print("\n3. Loading scratch-trained model...")
    scratch_model = load_scratch_model(SCRATCH_DIR, device)
    # Note: Not creating KronosPredictor for scratch - we use run_prediction_pct_norm directly

    # Load validation samples
    print("\n4. Loading validation samples...")
    samples = load_val_samples(n_samples=args.n_samples)
    print(f"   Testing on {len(samples)} samples")
    if args.use_atr:
        print("   Using ATR normalization")

    # Run comparisons
    print("\n5. Running predictions...")
    results = []

    for i, sample in enumerate(samples):
        print(f"\nSample {i+1}/{len(samples)}: {sample['date']}")
        result = compare_predictions(sample, base_predictor, scratch_model, tokenizer, device, use_atr=args.use_atr)
        results.append(result)

        # Only plot first 20 samples to save time
        if i < 20:
            plot_path = OUTPUT_DIR / f"comparison_{sample['date']}.png"
            plot_comparison(result, plot_path)
        print(f"  Actual: {result['actual_return']:+.2f}%")
        print(f"  Base: {result['base_return']:+.2f}% (error: {result['base_error']:.2f}%)")
        print(f"  Scratch: {result['scratch_return']:+.2f}% (error: {result['scratch_error']:.2f}%)")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    avg_base_error = np.mean([r['base_error'] for r in results])
    avg_scratch_error = np.mean([r['scratch_error'] for r in results])

    # Direction accuracy
    base_correct_dir = sum(1 for r in results
                          if (r['actual_return'] > 0) == (r['base_return'] > 0))
    scratch_correct_dir = sum(1 for r in results
                              if (r['actual_return'] > 0) == (r['scratch_return'] > 0))

    print(f"Average Return Error:")
    print(f"  Base model: {avg_base_error:.2f}%")
    print(f"  Scratch-trained: {avg_scratch_error:.2f}%")
    print(f"\nDirection Accuracy:")
    print(f"  Base model: {base_correct_dir}/{len(results)} ({base_correct_dir/len(results)*100:.0f}%)")
    print(f"  Scratch-trained: {scratch_correct_dir}/{len(results)} ({scratch_correct_dir/len(results)*100:.0f}%)")
    print(f"\nPlots saved to: {OUTPUT_DIR}")

    # Save results
    results_summary = {
        'samples': len(results),
        'avg_base_error': avg_base_error,
        'avg_scratch_error': avg_scratch_error,
        'base_direction_accuracy': base_correct_dir / len(results),
        'scratch_direction_accuracy': scratch_correct_dir / len(results),
        'details': [
            {
                'date': r['date'],
                'actual_return': r['actual_return'],
                'base_return': r['base_return'],
                'scratch_return': r['scratch_return']
            }
            for r in results
        ]
    }

    with open(OUTPUT_DIR / "test_results.json", 'w') as f:
        json.dump(results_summary, f, indent=2)


if __name__ == "__main__":
    main()
