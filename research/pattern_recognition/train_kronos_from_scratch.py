"""
Train Kronos predictor from scratch on W_Bottom pattern data.

This script trains a fresh Kronos predictor (random init) while using
the pre-trained tokenizer. This removes mean-reversion bias from the
base model.

Usage:
    python -m pattern_recognition.train_kronos_from_scratch
    python -m pattern_recognition.train_kronos_from_scratch --epochs 50 --batch-size 8 --lr 1e-4
"""

import argparse
import json
import sys
import time
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# Force unbuffered output for background execution
sys.stdout.reconfigure(line_buffering=True)

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "Kronos"))

from model.kronos import KronosTokenizer, Kronos

# Configuration
DATA_DIR = Path(__file__).parent.parent / "kronos_training_data"
OUTPUT_DIR = Path(__file__).parent.parent / "kronos_w_bottom_scratch"


class WBottomDataset(Dataset):
    """
    Dataset for W_Bottom pattern training.

    Each sample has:
    - input: Variable length context + pattern (OHLCV)
    - target: Fixed length breakout continuation (OHLCV)
    """

    def __init__(self, data_type: str = 'train', max_input_len: int = 300,
                 use_atr_normalization: bool = False):
        self.data_type = data_type
        self.max_input_len = max_input_len
        self.use_atr_normalization = use_atr_normalization

        # Load data
        data_dir = DATA_DIR
        inputs_file = data_dir / f"{data_type}_inputs.npy"
        targets_file = data_dir / f"{data_type}_targets.npy"
        samples_file = data_dir / f"{data_type}_samples.json"

        # Load with allow_pickle for variable-length inputs
        self.inputs = np.load(inputs_file, allow_pickle=True)
        self.targets = np.load(targets_file)

        # Load sample metadata for ATR values
        self.samples_meta = None
        if use_atr_normalization and samples_file.exists():
            with open(samples_file, 'r') as f:
                self.samples_meta = json.load(f)
            print(f"[{data_type.upper()}] ATR normalization enabled (loaded metadata)")

        print(f"[{data_type.upper()}] Loaded {len(self.inputs)} samples")
        print(f"[{data_type.upper()}] Target shape: {self.targets.shape}")

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        """
        Returns normalized input and target tensors.

        Input is padded/truncated to max_input_len.
        Adds 'amount' column (volume * close) to match Kronos format.
        """
        input_data = self.inputs[idx]  # Shape: (seq_len, 5) - OHLCV
        target_data = self.targets[idx]  # Shape: (60, 5) - OHLCV

        # Add 'amount' column (volume * close) to match Kronos format
        # Kronos expects: open, high, low, close, volume, amount
        input_amount = input_data[:, 4] * input_data[:, 3]  # volume * close
        input_data = np.column_stack([input_data, input_amount])

        target_amount = target_data[:, 4] * target_data[:, 3]
        target_data = np.column_stack([target_data, target_amount])

        # Pad or truncate input to max_input_len
        if len(input_data) > self.max_input_len:
            # Take the most recent bars
            input_data = input_data[-self.max_input_len:]
        elif len(input_data) < self.max_input_len:
            # Pad with zeros at the beginning
            pad_len = self.max_input_len - len(input_data)
            input_data = np.vstack([np.zeros((pad_len, 6)), input_data])

        # Calculate mean for centering (always input-only)
        x_mean = np.mean(input_data, axis=0)

        if self.use_atr_normalization and self.samples_meta:
            # ATR-BASED NORMALIZATION
            # Uses ATR for price columns, making predictions scale correctly across volatility regimes
            sample_meta = self.samples_meta[idx]
            atr = sample_meta.get('atr_14', 1.0)
            if atr < 1e-6:
                atr = 1.0  # Fallback for zero ATR

            # Use ATR for price columns (OHLC), std for volume/amount
            x_std = np.zeros(6)
            x_std[:4] = atr  # Price columns normalized by ATR
            x_std[4:] = np.std(input_data[:, 4:], axis=0)  # Volume/amount by std
        else:
            # PERCENTAGE-BASED NORMALIZATION (original approach)
            # Use std as % of price for scale-invariant normalization
            price_mean = np.mean(input_data[:, 3])  # Mean close price as reference
            x_std_abs = np.std(input_data, axis=0)  # Absolute std in $
            x_std = x_std_abs / price_mean * 100    # Convert to % std

        # Apply same normalization stats to both input and target
        input_normalized = (input_data - x_mean) / (x_std + 1e-5)
        target_normalized = (target_data - x_mean) / (x_std + 1e-5)

        # Concatenate and clip
        full_seq = np.vstack([input_normalized, target_normalized])
        full_seq = np.clip(full_seq, -5.0, 5.0)

        # Use realistic intraday time features
        # Market opens at 9:30 AM (570 minutes from midnight)
        seq_len = len(full_seq)
        time_features = np.zeros((seq_len, 5), dtype=np.float32)

        base_minute = 9 * 60 + 30  # 9:30 AM = 570 minutes from midnight
        for i in range(seq_len):
            total_minutes = base_minute + i
            # Wrap around if we exceed market hours (for long sequences)
            if total_minutes >= 16 * 60:  # Past 4 PM
                total_minutes = base_minute + (i % 390)  # 390 = 6.5 hours of trading
            time_features[i, 0] = total_minutes % 60  # minute (0-59)
            time_features[i, 1] = total_minutes // 60  # hour (9-15)
            time_features[i, 2] = 2  # weekday: Wednesday (mid-week, typical trading)
            time_features[i, 3] = 15  # day: mid-month
            time_features[i, 4] = 6  # month: June (mid-year)

        x_tensor = torch.from_numpy(full_seq.astype(np.float32))
        x_stamp_tensor = torch.from_numpy(time_features)

        return x_tensor, x_stamp_tensor, self.max_input_len


def train_epoch(model, tokenizer, dataloader, optimizer, scheduler, device, epoch, log_interval=10):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    n_batches = 0

    for batch_idx, (batch_x, batch_x_stamp, input_lens) in enumerate(dataloader):
        batch_x = batch_x.to(device)
        batch_x_stamp = batch_x_stamp.to(device)

        # Tokenize input data (tokenizer is frozen)
        with torch.no_grad():
            token_seq_0, token_seq_1 = tokenizer.encode(batch_x, half=True)

        # Prepare inputs and targets for next-token prediction
        token_in = [token_seq_0[:, :-1], token_seq_1[:, :-1]]
        token_out = [token_seq_0[:, 1:], token_seq_1[:, 1:]]

        # Forward pass
        logits = model(token_in[0], token_in[1], batch_x_stamp[:, :-1, :])

        # Compute loss
        loss, s1_loss, s2_loss = model.head.compute_loss(
            logits[0], logits[1], token_out[0], token_out[1]
        )

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        n_batches += 1

        if (batch_idx + 1) % log_interval == 0:
            avg_loss = total_loss / n_batches
            lr = optimizer.param_groups[0]['lr']
            print(f"  Epoch {epoch}, Batch {batch_idx+1}/{len(dataloader)}, Loss: {avg_loss:.4f}, LR: {lr:.6f}")

    return total_loss / n_batches


def validate(model, tokenizer, dataloader, device):
    """Validate the model."""
    model.eval()
    total_loss = 0
    n_batches = 0

    with torch.no_grad():
        for batch_x, batch_x_stamp, input_lens in dataloader:
            batch_x = batch_x.to(device)
            batch_x_stamp = batch_x_stamp.to(device)

            token_seq_0, token_seq_1 = tokenizer.encode(batch_x, half=True)
            token_in = [token_seq_0[:, :-1], token_seq_1[:, :-1]]
            token_out = [token_seq_0[:, 1:], token_seq_1[:, 1:]]

            logits = model(token_in[0], token_in[1], batch_x_stamp[:, :-1, :])
            loss, _, _ = model.head.compute_loss(
                logits[0], logits[1], token_out[0], token_out[1]
            )

            total_loss += loss.item()
            n_batches += 1

    return total_loss / n_batches


def main():
    parser = argparse.ArgumentParser(description="Train Kronos from scratch on W_Bottom patterns")
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--max-input-len', type=int, default=300, help='Max input sequence length')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device')
    parser.add_argument('--tokenizer', type=str, default='NeoQuasar/Kronos-Tokenizer-base', help='Tokenizer path')
    parser.add_argument('--output', type=str, default=str(OUTPUT_DIR), help='Output directory')
    # Model architecture (larger for more data)
    parser.add_argument('--n-layers', type=int, default=18, help='Number of transformer layers')
    parser.add_argument('--d-model', type=int, default=384, help='Model dimension')
    parser.add_argument('--ff-dim', type=int, default=1536, help='Feedforward dimension')
    parser.add_argument('--n-heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--atr-normalization', action='store_true', help='Use ATR-based normalization (requires ATR in training data)')
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Kronos Training FROM SCRATCH on W_Bottom Patterns")
    print("=" * 60)

    # Check device
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'
    device = torch.device(args.device)
    print(f"Device: {device}")

    # Load datasets
    print("\n1. Loading datasets...")
    print(f"   ATR normalization: {'ENABLED' if args.atr_normalization else 'DISABLED'}")
    train_dataset = WBottomDataset('train', max_input_len=args.max_input_len,
                                   use_atr_normalization=args.atr_normalization)
    val_dataset = WBottomDataset('val', max_input_len=args.max_input_len,
                                 use_atr_normalization=args.atr_normalization)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )

    # Load tokenizer (FROZEN - no training)
    print("\n2. Loading tokenizer (frozen)...")
    print(f"   Tokenizer: {args.tokenizer}")
    tokenizer = KronosTokenizer.from_pretrained(args.tokenizer)
    tokenizer.eval().to(device)

    # Freeze tokenizer parameters
    for param in tokenizer.parameters():
        param.requires_grad = False

    # Get tokenizer config for model creation
    tokenizer_s1_bits = tokenizer.s1_bits
    tokenizer_s2_bits = tokenizer.s2_bits
    print(f"   Tokenizer s1_bits: {tokenizer_s1_bits}, s2_bits: {tokenizer_s2_bits}")

    # Create fresh model (RANDOM INITIALIZATION)
    print("\n3. Creating fresh Kronos model (random init)...")
    print(f"   Architecture: n_layers={args.n_layers}, d_model={args.d_model}, ff_dim={args.ff_dim}")

    model = Kronos(
        s1_bits=tokenizer_s1_bits,
        s2_bits=tokenizer_s2_bits,
        n_layers=args.n_layers,
        d_model=args.d_model,
        n_heads=args.n_heads,
        ff_dim=args.ff_dim,
        ffn_dropout_p=0.0,
        attn_dropout_p=0.0,
        resid_dropout_p=0.0,
        token_dropout_p=0.1,
        learn_te=False
    )
    model.to(device)

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total params: {n_params:,}")
    print(f"   Trainable params: {n_trainable:,}")

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=0.1
    )

    total_steps = len(train_loader) * args.epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        total_steps=total_steps,
        pct_start=0.05,  # 5% warmup
        div_factor=10
    )

    # Training loop
    print("\n4. Starting training...")
    print(f"   Epochs: {args.epochs}")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Learning rate: {args.lr}")
    print(f"   Total steps: {total_steps}")

    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': []}

    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        # Train
        train_loss = train_epoch(
            model, tokenizer, train_loader, optimizer, scheduler,
            device, epoch, log_interval=20
        )

        # Validate
        val_loss = validate(model, tokenizer, val_loader, device)

        # Record history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        epoch_time = time.time() - epoch_start

        print(f"\nEpoch {epoch}/{args.epochs}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Time: {epoch_time:.1f}s")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = output_dir / "best_model"
            save_path.mkdir(parents=True, exist_ok=True)

            # Save model state dict and config
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': {
                    's1_bits': tokenizer_s1_bits,
                    's2_bits': tokenizer_s2_bits,
                    'n_layers': args.n_layers,
                    'd_model': args.d_model,
                    'n_heads': args.n_heads,
                    'ff_dim': args.ff_dim,
                }
            }, save_path / "model.pt")
            print(f"  -> Best model saved (val_loss: {best_val_loss:.4f})")

    total_time = time.time() - start_time

    # Save final model
    final_path = output_dir / "final_model"
    final_path.mkdir(parents=True, exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': {
            's1_bits': tokenizer_s1_bits,
            's2_bits': tokenizer_s2_bits,
            'n_layers': args.n_layers,
            'd_model': args.d_model,
            'n_heads': args.n_heads,
            'ff_dim': args.ff_dim,
        }
    }, final_path / "model.pt")

    # Save training history
    with open(output_dir / "training_history.json", 'w') as f:
        json.dump({
            'history': history,
            'best_val_loss': best_val_loss,
            'total_time_seconds': total_time,
            'args': vars(args),
            'atr_normalization': args.atr_normalization,
            'timestamp': datetime.now().isoformat()
        }, f, indent=2)

    # Summary
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Total time: {total_time / 60:.1f} minutes")
    print(f"Output directory: {output_dir}")
    print(f"  - best_model/")
    print(f"  - final_model/")
    print(f"  - training_history.json")


if __name__ == "__main__":
    main()
