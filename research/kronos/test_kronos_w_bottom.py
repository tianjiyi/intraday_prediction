"""
Test the fine-tuned Kronos model on W_Bottom pattern prediction.

This script:
1. Loads a sample W_Bottom pattern
2. Runs prediction with base model vs fine-tuned model
3. Compares predicted breakout with actual outcome

Usage:
    python -m pattern_recognition.test_kronos_w_bottom
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
FINETUNED_DIR = Path(__file__).parent.parent / "kronos_w_bottom_finetuned" / "best_model"
OUTPUT_DIR = Path(__file__).parent.parent / "kronos_test_output"


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


def normalize_data(data):
    """Normalize data for model input."""
    x_mean = np.mean(data, axis=0)
    x_std = np.std(data, axis=0)
    normalized = (data - x_mean) / (x_std + 1e-5)
    normalized = np.clip(normalized, -5.0, 5.0)
    return normalized, x_mean, x_std


def denormalize_data(normalized, x_mean, x_std):
    """Denormalize predictions back to original scale."""
    return normalized * (x_std + 1e-5) + x_mean


def run_prediction(predictor, input_data, pred_len: int = 60, n_samples: int = 10):
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

    # Run prediction
    pred_result = predictor.predict(
        df=df,
        x_timestamp=x_timestamp,
        y_timestamp=y_timestamp,
        pred_len=pred_len,
        sample_count=n_samples,
        T=0.8,
        top_p=0.9,
        return_samples=True
    )

    return pred_result


def compare_predictions(sample: dict, base_predictor, finetuned_predictor, device):
    """Compare base vs fine-tuned model predictions."""
    # Prepare input - RAW data (Kronos predictor does its own normalization)
    input_data = prepare_input(sample)
    target_data = np.array(sample['target'])

    # Get last close for return calculation
    last_input_close = input_data[-1, 3]
    actual_close = target_data[:, 3]

    # Run predictions with RAW input data
    # KronosPredictor handles normalization internally
    print(f"\n  Running base model prediction...")
    base_pred = run_prediction(base_predictor, input_data)

    print(f"  Running fine-tuned model prediction...")
    finetuned_pred = run_prediction(finetuned_predictor, input_data)

    # Get predicted close prices (already in real price scale from predictor)
    base_close = base_pred['mean']['close'].values
    finetuned_close = finetuned_pred['mean']['close'].values

    # Calculate returns
    actual_return = (actual_close[-1] - last_input_close) / last_input_close * 100
    base_return = (base_close[-1] - last_input_close) / last_input_close * 100
    finetuned_return = (finetuned_close[-1] - last_input_close) / last_input_close * 100

    # Debug: print to understand model behavior
    print(f"    Last input close: ${last_input_close:.2f}")
    print(f"    Actual final close: ${actual_close[-1]:.2f} ({actual_return:+.2f}%)")
    print(f"    Base prediction: ${base_close[-1]:.2f} ({base_return:+.2f}%)")
    print(f"    Fine-tuned prediction: ${finetuned_close[-1]:.2f} ({finetuned_return:+.2f}%)")

    return {
        'date': sample['date'],
        'input_close': input_data[-60:, 3],  # Last 60 bars of input
        'actual_close': actual_close,
        'base_pred_close': base_close,
        'finetuned_pred_close': finetuned_close,
        'actual_return': actual_return,
        'base_return': base_return,
        'finetuned_return': finetuned_return,
        'base_error': abs(base_return - actual_return),
        'finetuned_error': abs(finetuned_return - actual_return)
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
    ax.plot(pred_x, result['finetuned_pred_close'], 'm-', label='Fine-tuned', linewidth=2)

    # Mark breakout direction
    ax.axvline(x=pred_start, color='gray', linestyle=':', alpha=0.5, label='Pattern End')

    ax.set_title(f"W_Bottom Prediction - {result['date']}\n"
                 f"Actual: {result['actual_return']:+.2f}%, Base: {result['base_return']:+.2f}%, "
                 f"Fine-tuned: {result['finetuned_return']:+.2f}%")
    ax.set_xlabel('Bar')
    ax.set_ylabel('Close Price')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=100)
    plt.close()


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Kronos W_Bottom Prediction Test")
    print("=" * 60)

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

    # Load fine-tuned model
    print("\n3. Loading fine-tuned model...")
    finetuned_model = Kronos.from_pretrained(str(FINETUNED_DIR))
    finetuned_model.to(device)
    finetuned_predictor = KronosPredictor(
        model=finetuned_model,
        tokenizer=tokenizer,
        device=str(device),
        max_context=512
    )

    # Load validation samples
    print("\n4. Loading validation samples...")
    samples = load_val_samples(n_samples=10)
    print(f"   Testing on {len(samples)} samples")

    # Run comparisons
    print("\n5. Running predictions...")
    results = []

    for i, sample in enumerate(samples):
        print(f"\nSample {i+1}/{len(samples)}: {sample['date']}")
        result = compare_predictions(sample, base_predictor, finetuned_predictor, device)
        results.append(result)

        # Plot comparison
        plot_path = OUTPUT_DIR / f"comparison_{sample['date']}.png"
        plot_comparison(result, plot_path)
        print(f"  Actual: {result['actual_return']:+.2f}%")
        print(f"  Base: {result['base_return']:+.2f}% (error: {result['base_error']:.2f}%)")
        print(f"  Fine-tuned: {result['finetuned_return']:+.2f}% (error: {result['finetuned_error']:.2f}%)")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    avg_base_error = np.mean([r['base_error'] for r in results])
    avg_finetuned_error = np.mean([r['finetuned_error'] for r in results])

    # Direction accuracy
    base_correct_dir = sum(1 for r in results
                          if (r['actual_return'] > 0) == (r['base_return'] > 0))
    finetuned_correct_dir = sum(1 for r in results
                                if (r['actual_return'] > 0) == (r['finetuned_return'] > 0))

    print(f"Average Return Error:")
    print(f"  Base model: {avg_base_error:.2f}%")
    print(f"  Fine-tuned: {avg_finetuned_error:.2f}%")
    print(f"\nDirection Accuracy:")
    print(f"  Base model: {base_correct_dir}/{len(results)} ({base_correct_dir/len(results)*100:.0f}%)")
    print(f"  Fine-tuned: {finetuned_correct_dir}/{len(results)} ({finetuned_correct_dir/len(results)*100:.0f}%)")
    print(f"\nPlots saved to: {OUTPUT_DIR}")

    # Save results
    results_summary = {
        'samples': len(results),
        'avg_base_error': avg_base_error,
        'avg_finetuned_error': avg_finetuned_error,
        'base_direction_accuracy': base_correct_dir / len(results),
        'finetuned_direction_accuracy': finetuned_correct_dir / len(results),
        'details': [
            {
                'date': r['date'],
                'actual_return': r['actual_return'],
                'base_return': r['base_return'],
                'finetuned_return': r['finetuned_return']
            }
            for r in results
        ]
    }

    with open(OUTPUT_DIR / "test_results.json", 'w') as f:
        json.dump(results_summary, f, indent=2)


if __name__ == "__main__":
    main()
