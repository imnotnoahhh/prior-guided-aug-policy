#!/usr/bin/env python
"""
Run S0 Baseline with EXACTLY the same settings as Phase A.
Settings: epochs=200, batch_size=64, lr=0.05, wd=1e-3, early_stop_patience=5

Usage:
    python run_baseline.py
"""
import sys
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
    set_seed,
    get_device,
    train_one_epoch,
    evaluate,
    get_optimizer_and_scheduler,
    EarlyStopping,
)


def run_baseline(
    epochs: int = 200,
    fold_idx: int = 0,
    batch_size: int = 64,
    num_workers: int = 6,
    early_stop_patience: int = 5,
    seed: int = 42,
):
    """Run S0 Baseline with same settings as Phase A."""
    
    set_seed(seed)
    device = get_device()
    
    print("=" * 70)
    print("S0 Baseline Test (Same settings as Phase A)")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"LR: 0.05, WD: 1e-3, Momentum: 0.9")
    print(f"Early stop patience: {early_stop_patience}")
    print(f"Fold: {fold_idx}")
    print(f"Seed: {seed}")
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
    
    # Create data loaders (same settings as Phase A)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if device.type == "cuda" else False,
        drop_last=False,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if device.type == "cuda" else False,
        drop_last=False,
    )
    
    # Create model
    model = create_model(num_classes=100, pretrained=False)
    model = model.to(device)
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer and scheduler (EXACTLY same as Phase A)
    optimizer, scheduler = get_optimizer_and_scheduler(
        model=model,
        total_epochs=epochs,
        lr=0.05,
        weight_decay=1e-3,
        momentum=0.9,
    )
    
    # AMP scaler (only for CUDA)
    scaler = None
    if device.type == "cuda":
        scaler = torch.amp.GradScaler()
    
    # Early stopping (same as Phase A)
    early_stopper = EarlyStopping(patience=early_stop_patience, mode="min")
    
    # Training loop
    best_val_acc = 0.0
    best_val_loss = float("inf")
    best_top5_acc = 0.0
    epochs_run = 0
    
    print("\nTraining S0 Baseline...")
    print("-" * 50)
    
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
        
        # Track best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_top5_acc = top5_acc
            best_val_loss = val_loss
        
        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d}: train_acc={train_acc:.1f}%, val_acc={val_acc:.1f}%, top5={top5_acc:.1f}%")
        
        # Early stopping check
        if early_stopper(val_loss):
            print(f"\nEarly stopping at epoch {epoch + 1}")
            break
    
    print("-" * 50)
    print("\n" + "=" * 70)
    print("S0 BASELINE RESULT")
    print("=" * 70)
    print(f"Epochs run: {epochs_run}")
    print(f"Best Val Acc: {best_val_acc:.1f}%")
    print(f"Best Top-5 Acc: {best_top5_acc:.1f}%")
    print(f"Best Val Loss: {best_val_loss:.4f}")
    print("=" * 70)
    
    return {
        "val_acc": best_val_acc,
        "top5_acc": best_top5_acc,
        "val_loss": best_val_loss,
        "epochs_run": epochs_run,
    }


if __name__ == "__main__":
    result = run_baseline()
