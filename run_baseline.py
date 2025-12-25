#!/usr/bin/env python
"""
Run S0 Baseline with same settings as Phase A.
Settings: epochs=200, batch_size=64, lr=0.05, wd=1e-3

Usage:
    python run_baseline.py
    python run_baseline.py --epochs 200 --min_epochs 100 --early_stop_patience 30

Output:
    outputs/baseline_result.csv
"""
import argparse
import csv
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import CIFAR100Subsampled
from src.augmentations import get_baseline_transform, get_val_transform
from src.models import create_model
from src.utils import (
    set_seed_deterministic,
    get_device,
    train_one_epoch,
    evaluate,
    get_optimizer_and_scheduler,
    EarlyStopping,
)


def run_baseline(
    epochs: int = 200,
    fold_idx: int = 0,
    batch_size: int = 128,
    num_workers: int = 8,
    early_stop_patience: int = 80,
    min_epochs: int = 80,
    seed: int = 42,
    deterministic: bool = True,
):
    """Run S0 Baseline with same settings as Phase A."""
    
    # Set seed with deterministic mode for reproducibility
    set_seed_deterministic(seed, deterministic=deterministic)
    device = get_device()
    
    print("=" * 70)
    print("Baseline: S0 Baseline Training")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Fold: {fold_idx}")
    print(f"Seed: {seed}")
    print(f"Deterministic: {deterministic}")
    print(f"LR: 0.1, WD: 5e-3, Momentum: 0.9, Warmup: 5 epochs, Label Smoothing: 0.1")
    print(f"Min epochs: {min_epochs}")
    print(f"Early stop patience: {early_stop_patience}")
    print(f"Output dir: outputs")
    print("=" * 70)
    
    # Build transforms - S0 Baseline only (RandomCrop + HorizontalFlip)
    train_transform = get_baseline_transform(include_normalize=False)
    val_transform = get_val_transform(include_normalize=False)
    
    # Create datasets
    train_dataset = CIFAR100Subsampled(
        root="./data",
        train=True,
        fold_idx=fold_idx,
        transform=train_transform,
        download=True,
    )
    
    val_dataset = CIFAR100Subsampled(
        root="./data",
        train=False,
        fold_idx=fold_idx,
        transform=val_transform,
        download=True,
    )
    
    # Create data loaders with optimized settings for large batch
    use_cuda = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_cuda,
        drop_last=False,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=4 if num_workers > 0 else None,  # Increased from 2 to 4
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_cuda,
        drop_last=False,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=4 if num_workers > 0 else None,  # Increased from 2 to 4
    )
    
    # Create model with channels_last memory format for better GPU performance
    model = create_model(num_classes=100, pretrained=False)
    model = model.to(device)
    if use_cuda:
        model = model.to(memory_format=torch.channels_last)
    
    # Loss function (with label smoothing for regularization)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # Optimizer and scheduler
    optimizer, scheduler = get_optimizer_and_scheduler(
        model=model,
        total_epochs=epochs,
        lr=0.1,
        weight_decay=5e-3,
        momentum=0.9,
        warmup_epochs=5,
    )
    
    # AMP scaler (only for CUDA)
    scaler = None
    if device.type == "cuda":
        scaler = torch.amp.GradScaler()
    
    # Early stopping (v5.1 strategy: monitor val_acc, mode="max")
    early_stopper = EarlyStopping(
        patience=early_stop_patience,
        mode="max",  # Monitor val_acc
        min_epochs=min_epochs,
        min_delta=0.2,
    )
    
    # Training loop - track all metrics for CSV
    best_val_acc = 0.0
    best_val_loss = float("inf")
    best_top5_acc = 0.0
    best_train_acc = 0.0
    best_train_loss = 0.0
    best_epoch = 0
    epochs_run = 0
    early_stopped = False
    best_model_state = None  # Store best model state for checkpoint
    
    # Prepare checkpoint directory
    checkpoint_dir = Path("outputs/checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    print("\nTraining S0 Baseline...")
    print("-" * 50)
    
    start_time = time.time()
    
    for epoch in range(epochs):
        epochs_run = epoch + 1
        
        # Train one epoch
        train_loss, train_acc = train_one_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
        )
        
        # Evaluate
        val_loss, val_acc, top5_acc = evaluate(
            model=model,
            val_loader=val_loader,
            criterion=criterion,
            device=device,
        )
        
        # Update scheduler
        scheduler.step()
        
        # Track best (by val_acc) and save model state
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_top5_acc = top5_acc
            best_val_loss = val_loss
            best_train_acc = train_acc
            best_train_loss = train_loss
            best_epoch = epoch + 1
            # Save best model state (deep copy to avoid reference issues)
            best_model_state = {
                "model_state_dict": {k: v.cpu().clone() for k, v in model.state_dict().items()},
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
            }
        
        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d}: train_acc={train_acc:.1f}%, val_acc={val_acc:.1f}%, top5={top5_acc:.1f}%")
        
        # Early stopping check (monitor val_acc)
        if early_stopper(val_acc, epoch + 1):
            print(f"\nEarly stopping at epoch {epoch + 1}")
            early_stopped = True
            break
    
    runtime_sec = time.time() - start_time
    timestamp = datetime.now().isoformat(timespec='seconds')
    
    print("-" * 70)
    print("\n" + "=" * 70)
    print("Baseline Complete")
    print("=" * 70)
    print(f"Successful: 1")
    print(f"Failed: 0")
    print(f"Total runtime: {runtime_sec:.1f}s ({runtime_sec/60:.1f}min)")
    print("-" * 70)
    print(f"Best Epoch: {best_epoch}/{epochs_run}")
    print(f"Val Acc: {best_val_acc:.2f}%  |  Top-5: {best_top5_acc:.2f}%")
    print(f"Train Acc: {best_train_acc:.2f}%  |  Val Loss: {best_val_loss:.4f}")
    print(f"Early Stopped: {early_stopped}")
    print("=" * 70)
    
    # Save to CSV with unified format
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    csv_path = output_dir / "baseline_result.csv"
    
    # Unified CSV fieldnames (same for all phases)
    fieldnames = [
        "phase", "op_name", "magnitude", "probability", "seed", "fold_idx",
        "val_acc", "val_loss", "top5_acc", "train_acc", "train_loss",
        "epochs_run", "best_epoch", "early_stopped", "runtime_sec",
        "timestamp", "error"
    ]
    
    row = {
        "phase": "Baseline",
        "op_name": "Baseline",
        "magnitude": "0.0",
        "probability": "1.0",
        "seed": seed,
        "fold_idx": fold_idx,
        "val_acc": round(best_val_acc, 4),
        "val_loss": round(best_val_loss, 6),
        "top5_acc": round(best_top5_acc, 4),
        "train_acc": round(best_train_acc, 4),
        "train_loss": round(best_train_loss, 6),
        "epochs_run": epochs_run,
        "best_epoch": best_epoch,
        "early_stopped": early_stopped,
        "runtime_sec": round(runtime_sec, 2),
        "timestamp": timestamp,
        "error": "",
    }
    
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)
    
    print(f"\nResults saved to: {csv_path}")
    
    # Save best model checkpoint
    if best_model_state is not None:
        checkpoint_path = checkpoint_dir / "baseline_best.pth"
        checkpoint = {
            "model_state_dict": best_model_state["model_state_dict"],
            "optimizer_state_dict": best_model_state["optimizer_state_dict"],
            "scheduler_state_dict": best_model_state["scheduler_state_dict"],
            "epoch": best_epoch,
            "val_acc": best_val_acc,
            "top5_acc": best_top5_acc,
            "val_loss": best_val_loss,
            "config": {
                "seed": seed,
                "fold_idx": fold_idx,
                "epochs": epochs,
                "batch_size": batch_size,
                "lr": 0.1,
                "weight_decay": 5e-3,
                "momentum": 0.9,
            }
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved to: {checkpoint_path}")
    
    return {
        "val_acc": best_val_acc,
        "top5_acc": best_top5_acc,
        "val_loss": best_val_loss,
        "train_acc": best_train_acc,
        "train_loss": best_train_loss,
        "epochs_run": epochs_run,
        "best_epoch": best_epoch,
        "early_stopped": early_stopped,
        "runtime_sec": runtime_sec,
    }


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run S0 Baseline training")
    parser.add_argument("--epochs", type=int, default=200, help="Number of epochs")
    parser.add_argument("--fold_idx", type=int, default=0, help="Fold index")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=8, help="Data loader workers")
    parser.add_argument("--early_stop_patience", type=int, default=80, help="Early stopping patience")
    parser.add_argument("--min_epochs", type=int, default=80, help="Minimum epochs before early stopping")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no_deterministic", action="store_true", help="Disable deterministic mode")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = run_baseline(
        epochs=args.epochs,
        fold_idx=args.fold_idx,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        early_stop_patience=args.early_stop_patience,
        min_epochs=args.min_epochs,
        seed=args.seed,
        deterministic=not args.no_deterministic,
    )
