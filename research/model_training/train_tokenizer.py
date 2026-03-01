"""
Train Kronos Tokenizer on QQQ Intraday Data
Adapted from Kronos/finetune/train_tokenizer.py
"""
import os
import sys
import json
import time
from time import gmtime, strftime
import torch.distributed as dist
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

# Add project root and Kronos to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'Kronos'))

from model_training.config import TrainingConfig
from model_training.intraday_dataset import IntradayDataset
from model.kronos import KronosTokenizer
from model_training.utils.training_utils import (
    setup_ddp,
    cleanup_ddp,
    set_seed,
    get_model_size,
    format_time
)


def create_dataloaders(config: dict, rank: int, world_size: int, use_ddp: bool):
    """
    Creates and returns dataloaders for training and validation.
    Supports both DDP and standalone modes.
    """
    print(f"[Rank {rank}] Creating dataloaders...")
    train_dataset = IntradayDataset('train')
    valid_dataset = IntradayDataset('val')
    print(f"[Rank {rank}] Train dataset size: {len(train_dataset)}, Validation dataset size: {len(valid_dataset)}")

    if use_ddp:
        # Distributed sampling
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        val_sampler = DistributedSampler(valid_dataset, num_replicas=world_size, rank=rank, shuffle=False)
        shuffle_train = False
    else:
        # Standalone mode - no sampler needed
        train_sampler = None
        val_sampler = None
        shuffle_train = True

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        sampler=train_sampler,
        shuffle=shuffle_train,
        num_workers=config.get('num_workers', 2),
        pin_memory=True,
        drop_last=True
    )
    val_loader = DataLoader(
        valid_dataset,
        batch_size=config['batch_size'],
        sampler=val_sampler,
        shuffle=False,
        num_workers=config.get('num_workers', 2),
        pin_memory=True,
        drop_last=False
    )
    print(f"[Rank {rank}] Dataloaders created. Train steps/epoch: {len(train_loader)}, Val steps: {len(val_loader)}")
    return train_loader, val_loader, train_dataset, valid_dataset


def train_model(model, device, config, save_dir, rank, world_size, use_ddp, start_epoch=0, resume_state=None):
    """
    The main training and validation loop for the tokenizer.

    Args:
        start_epoch: Epoch to start from (0 for fresh training, >0 for resume)
        resume_state: Dictionary with optimizer/scheduler/best_val_loss states
    """
    start_time = time.time()
    if rank == 0:
        effective_bs = config['batch_size'] * world_size * config['accumulation_steps']
        print(f"[Rank {rank}] BATCHSIZE (per GPU): {config['batch_size']}")
        print(f"[Rank {rank}] Effective total batch size: {effective_bs}")
        if start_epoch > 0:
            print(f"Resuming from epoch {start_epoch}...")

    train_loader, val_loader, train_dataset, valid_dataset = create_dataloaders(config, rank, world_size, use_ddp)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['tokenizer_learning_rate'],
        weight_decay=config['adam_weight_decay']
    )

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer=optimizer,
        max_lr=config['tokenizer_learning_rate'],
        steps_per_epoch=len(train_loader),
        epochs=config['epochs'],
        pct_start=0.03,
        div_factor=10,
        last_epoch=start_epoch * len(train_loader) - 1 if start_epoch > 0 else -1
    )

    # Load resumed states if provided
    best_val_loss = float('inf')
    if resume_state is not None:
        best_val_loss = resume_state.get('best_val_loss', float('inf'))
        if 'optimizer_state_dict' in resume_state:
            optimizer.load_state_dict(resume_state['optimizer_state_dict'])
            print(f"[Rank {rank}] Loaded optimizer state")
        if 'scheduler_state_dict' in resume_state:
            scheduler.load_state_dict(resume_state['scheduler_state_dict'])
            print(f"[Rank {rank}] Loaded scheduler state")
    dt_result = {}
    batch_idx_global_train = start_epoch * len(train_loader)

    for epoch_idx in range(start_epoch, config['epochs']):
        epoch_start_time = time.time()
        model.train()

        # Set epoch for distributed sampler (only if using DDP)
        if use_ddp and hasattr(train_loader, 'sampler') and train_loader.sampler is not None:
            train_loader.sampler.set_epoch(epoch_idx)

        # Set dataset seeds for reproducible sampling
        train_dataset.set_epoch_seed(epoch_idx * 10000 + rank)
        valid_dataset.set_epoch_seed(0)

        for i, (ori_batch_x, _) in enumerate(train_loader):
            ori_batch_x = ori_batch_x.squeeze(0).to(device, non_blocking=True)

            # Gradient Accumulation Loop
            current_batch_total_loss = 0.0
            for j in range(config['accumulation_steps']):
                start_idx = j * (ori_batch_x.shape[0] // config['accumulation_steps'])
                end_idx = (j + 1) * (ori_batch_x.shape[0] // config['accumulation_steps'])
                batch_x = ori_batch_x[start_idx:end_idx]

                # Forward pass
                zs, bsq_loss, _, _ = model(batch_x)
                z_pre, z = zs

                # Loss calculation
                recon_loss_pre = F.mse_loss(z_pre, batch_x)
                recon_loss_all = F.mse_loss(z, batch_x)
                recon_loss = recon_loss_pre + recon_loss_all
                loss = (recon_loss + bsq_loss) / 2

                loss_scaled = loss / config['accumulation_steps']
                current_batch_total_loss += loss.item()
                loss_scaled.backward()

            # Optimizer Step after Accumulation
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            # Logging (Master Process Only)
            if rank == 0 and (batch_idx_global_train + 1) % config['log_interval'] == 0:
                avg_loss = current_batch_total_loss / config['accumulation_steps']
                print(
                    f"[Rank {rank}, Epoch {epoch_idx + 1}/{config['epochs']}, Step {i + 1}/{len(train_loader)}] "
                    f"LR {optimizer.param_groups[0]['lr']:.6f}, Loss: {avg_loss:.4f}"
                )

            batch_idx_global_train += 1

        # Validation Loop
        model.eval()
        tot_val_loss_sum_rank = 0.0
        val_sample_count_rank = 0
        with torch.no_grad():
            for ori_batch_x, _ in val_loader:
                ori_batch_x = ori_batch_x.squeeze(0).to(device, non_blocking=True)
                zs, _, _, _ = model(ori_batch_x)
                _, z = zs
                val_loss_item = F.mse_loss(z, ori_batch_x)

                tot_val_loss_sum_rank += val_loss_item.item() * ori_batch_x.size(0)
                val_sample_count_rank += ori_batch_x.size(0)

        # Reduce validation losses (only in DDP mode)
        if use_ddp:
            val_loss_sum_tensor = torch.tensor(tot_val_loss_sum_rank, device=device)
            val_count_tensor = torch.tensor(val_sample_count_rank, device=device)
            dist.all_reduce(val_loss_sum_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(val_count_tensor, op=dist.ReduceOp.SUM)
            avg_val_loss = val_loss_sum_tensor.item() / val_count_tensor.item() if val_count_tensor.item() > 0 else 0
        else:
            # Standalone mode - no reduction needed
            avg_val_loss = tot_val_loss_sum_rank / val_sample_count_rank if val_sample_count_rank > 0 else 0

        # End of Epoch Summary & Checkpointing
        if rank == 0:
            print(f"\n--- Epoch {epoch_idx + 1}/{config['epochs']} Summary ---")
            print(f"Validation Loss: {avg_val_loss:.4f}")
            print(f"Time This Epoch: {format_time(time.time() - epoch_start_time)}")
            print(f"Total Time Elapsed: {format_time(time.time() - start_time)}\n")

            # Save best model (HuggingFace format)
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                save_path = f"{save_dir}/checkpoints/best_model"
                # Save model (access .module only if using DDP)
                model_to_save = model.module if use_ddp else model
                model_to_save.save_pretrained(save_path)
                print(f"[OK] Best model saved to {save_path} (Val Loss: {best_val_loss:.4f})")

            # Save resumable checkpoint (PyTorch format with optimizer/scheduler states)
            checkpoint_path = f"{save_dir}/checkpoints/checkpoint_epoch_{epoch_idx + 1}.pt"
            model_to_save = model.module if use_ddp else model
            checkpoint = {
                'epoch': epoch_idx + 1,
                'model_state_dict': model_to_save.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'config': config
            }
            torch.save(checkpoint, checkpoint_path)
            print(f"[OK] Checkpoint saved to {checkpoint_path}")

        # Synchronize processes (only in DDP mode)
        if use_ddp:
            dist.barrier()

    dt_result['best_val_loss'] = best_val_loss
    return model, dt_result


def main(config: dict):
    """Main function to orchestrate the training process (DDP or standalone)."""
    rank, world_size, local_rank, use_ddp = setup_ddp()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    set_seed(config['seed'], rank)

    save_dir = os.path.join(config['save_path'], config['tokenizer_save_folder_name'])

    # Setup (master process only)
    master_summary = {}
    if rank == 0:
        os.makedirs(os.path.join(save_dir, 'checkpoints'), exist_ok=True)
        master_summary = {
            'start_time': strftime("%Y-%m-%dT%H-%M-%S", gmtime()),
            'save_directory': save_dir,
            'world_size': world_size,
            'use_ddp': use_ddp,
        }
        print("=" * 80)
        print("TOKENIZER TRAINING - QQQ INTRADAY")
        print("=" * 80)

    # Synchronize processes (only in DDP mode)
    if use_ddp:
        dist.barrier()

    # Model Initialization
    start_epoch = 0
    resume_state = None

    if config.get('resume_from_checkpoint') and os.path.exists(config['resume_from_checkpoint']):
        checkpoint_path = config['resume_from_checkpoint']
        print(f"[Rank {rank}] Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Load model from checkpoint
        print(f"[Rank {rank}] Loading model weights from checkpoint...")
        model = KronosTokenizer.from_pretrained(config['pretrained_tokenizer_path'])
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)

        start_epoch = checkpoint['epoch']
        resume_state = {
            'best_val_loss': checkpoint.get('best_val_loss', float('inf')),
            'optimizer_state_dict': checkpoint.get('optimizer_state_dict'),
            'scheduler_state_dict': checkpoint.get('scheduler_state_dict')
        }

        print(f"[Rank {rank}] Resuming from epoch {start_epoch} (best val loss: {resume_state['best_val_loss']:.4f})")
    else:
        # Fresh training
        print(f"[Rank {rank}] Loading pretrained tokenizer from {config['pretrained_tokenizer_path']}...")
        model = KronosTokenizer.from_pretrained(config['pretrained_tokenizer_path'])
        model.to(device)

    # Wrap in DDP only if using distributed training
    if use_ddp:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)
        model_for_size = model.module
    else:
        model_for_size = model

    if rank == 0:
        print(f"Model Size: {get_model_size(model_for_size)}")

    # Start Training
    _, dt_result = train_model(
        model, device, config, save_dir, rank, world_size, use_ddp,
        start_epoch=start_epoch, resume_state=resume_state
    )

    # Finalize and save summary
    if rank == 0:
        master_summary['final_result'] = dt_result
        with open(os.path.join(save_dir, 'summary.json'), 'w') as f:
            json.dump(master_summary, f, indent=4)
        print('=' * 80)
        print('TOKENIZER TRAINING COMPLETE')
        print('=' * 80)
        print(f'Best validation loss: {dt_result["best_val_loss"]:.4f}')
        print(f'Model saved to: {save_dir}/checkpoints/best_model')

    cleanup_ddp()


if __name__ == '__main__':
    # Usage:
    # - Windows (standalone): python model_training/train_tokenizer.py
    # - Linux/Multi-GPU: torchrun --standalone --nproc_per_node=1 model_training/train_tokenizer.py

    config_instance = TrainingConfig()

    # Print config summary (only on rank 0 or standalone)
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        config_instance.print_summary()

    main(config_instance.__dict__)
