#!/usr/bin/env python3
"""
Phase 0: Hyperparameter Calibration (v5.5)

Purpose: Find optimal (weight_decay, label_smoothing) combination before main experiments.
This ensures the "fixed hyperparameters" are data-driven and defensible in paper.

Configuration:
- Dataset: CIFAR-100 Subsampled (Fold-0)
- Augmentation: S0 Baseline only
- Epochs: 100 (sufficient to see convergence trends)
- Seeds: 3 (42, 123, 456)
- Grid: weight_decay × label_smoothing

Selection Criteria:
1. Not underfitting (train_acc keeps rising, loss decreases)
2. High mean val_acc with low std (stable)

Output: outputs/phase0_calibration.csv
"""

import argparse
import csv
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.augmentations import get_baseline_transform, get_val_transform
from src.dataset import CIFAR100Subsampled
from src.models import create_model
from src.utils import (
    set_seed_deterministic,
    get_device,
    train_one_epoch,
    evaluate,
    ensure_dir,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Phase 0: Hyperparameter Calibration")
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Training epochs (default: 100)"
    )
    
    parser.add_argument(
        "--seeds",
        type=str,
        default="42,123,456",
        help="Comma-separated seeds (default: 42,123,456)"
    )
    
    parser.add_argument(
        "--fold_idx",
        type=int,
        default=0,
        help="Fold index (default: 0)"
    )
    
    parser.add_argument(
        "--weight_decays",
        type=str,
        default="5e-4,1e-3,5e-3,1e-2",
        help="Comma-separated weight decay values (default: 5e-4,1e-3,5e-3,1e-2)"
    )
    
    parser.add_argument(
        "--label_smoothings",
        type=str,
        default="0.0,0.05,0.1",
        help="Comma-separated label smoothing values (default: 0.0,0.05,0.1)"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size (default: 128)"
    )
    
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of data loading workers (default: 8)"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Output directory (default: outputs)"
    )
    
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Run minimal epochs for testing"
    )
    
    return parser.parse_args()


def train_single_config(
    weight_decay: float,
    label_smoothing: float,
    seed: int,
    epochs: int,
    fold_idx: int,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> dict:
    """Train a single configuration and return results."""
    
    set_seed_deterministic(seed, deterministic=True)
    
    # Create model
    model = create_model(num_classes=100)
    model = model.to(device)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    
    # Create datasets
    train_transform = get_baseline_transform(include_normalize=False)
    val_transform = get_val_transform(include_normalize=False)
    
    train_dataset = CIFAR100Subsampled(
        root="data",
        train=True,
        fold_idx=fold_idx,
        transform=train_transform,
        download=True,
    )
    
    val_dataset = CIFAR100Subsampled(
        root="data",
        train=False,
        fold_idx=fold_idx,
        transform=val_transform,
        download=True,
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=4 if num_workers > 0 else None,
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=4 if num_workers > 0 else None,
    )
    
    # Create optimizer and scheduler
    from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
    
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.1,
        momentum=0.9,
        weight_decay=weight_decay,
    )
    
    warmup_epochs = 5
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=1.0 / warmup_epochs,
        end_factor=1.0,
        total_iters=warmup_epochs,
    )
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=epochs - warmup_epochs,
        eta_min=1e-6,
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs],
    )
    
    # Loss function
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    
    # Training loop
    best_val_acc = 0.0
    best_epoch = 0
    
    train_accs = []
    val_accs = []
    
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc, top5_acc = evaluate(model, val_loader, criterion, device)
        
        scheduler.step()
        
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
    
    runtime = time.time() - start_time
    
    # Analyze convergence
    final_train_acc = train_accs[-1]
    final_val_acc = val_accs[-1]
    train_val_gap = final_train_acc - final_val_acc
    
    # Check if underfitting (train_acc not improving in last 20 epochs)
    if len(train_accs) >= 20:
        early_train = np.mean(train_accs[-40:-20]) if len(train_accs) >= 40 else np.mean(train_accs[:20])
        late_train = np.mean(train_accs[-20:])
        is_underfitting = late_train < early_train + 1.0  # Should improve at least 1%
    else:
        is_underfitting = False
    
    return {
        "weight_decay": weight_decay,
        "label_smoothing": label_smoothing,
        "seed": seed,
        "fold_idx": fold_idx,
        "final_train_acc": round(final_train_acc, 2),
        "final_val_acc": round(final_val_acc, 2),
        "best_val_acc": round(best_val_acc, 2),
        "best_epoch": best_epoch,
        "train_val_gap": round(train_val_gap, 2),
        "is_underfitting": is_underfitting,
        "runtime_sec": round(runtime, 1),
        "timestamp": datetime.now().isoformat(timespec='seconds'),
    }


def run_calibration(args):
    """Run full calibration grid."""
    
    device = get_device()
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)
    
    # Parse configurations
    seeds = [int(s) for s in args.seeds.split(",")]
    weight_decays = [float(wd) for wd in args.weight_decays.split(",")]
    label_smoothings = [float(ls) for ls in args.label_smoothings.split(",")]
    
    epochs = args.epochs if not args.dry_run else 5
    
    print("=" * 70)
    print("Phase 0: Hyperparameter Calibration (v5.5)")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Epochs: {epochs}")
    print(f"Seeds: {seeds}")
    print(f"Weight Decays: {weight_decays}")
    print(f"Label Smoothings: {label_smoothings}")
    print(f"Total configs: {len(weight_decays)} × {len(label_smoothings)} × {len(seeds)} = {len(weight_decays) * len(label_smoothings) * len(seeds)}")
    print("=" * 70)
    
    # Output file
    csv_path = output_dir / "phase0_calibration.csv"
    
    fieldnames = [
        "weight_decay", "label_smoothing", "seed", "fold_idx",
        "final_train_acc", "final_val_acc", "best_val_acc", "best_epoch",
        "train_val_gap", "is_underfitting", "runtime_sec", "timestamp"
    ]
    
    # Check if file exists
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    
    results = []
    config_count = 0
    total_configs = len(weight_decays) * len(label_smoothings) * len(seeds)
    
    for wd in weight_decays:
        for ls in label_smoothings:
            for seed in seeds:
                config_count += 1
                print(f"\n[{config_count}/{total_configs}] wd={wd}, ls={ls}, seed={seed}")
                print("-" * 40)
                
                result = train_single_config(
                    weight_decay=wd,
                    label_smoothing=ls,
                    seed=seed,
                    epochs=epochs,
                    fold_idx=args.fold_idx,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    device=device,
                )
                
                results.append(result)
                
                # Write to CSV immediately
                with open(csv_path, mode="a", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    if write_header:
                        writer.writeheader()
                        write_header = False
                    writer.writerow(result)
                
                print(f"Result: train={result['final_train_acc']:.1f}%, val={result['best_val_acc']:.1f}%, gap={result['train_val_gap']:.1f}%")
    
    # Aggregate results
    print("\n" + "=" * 70)
    print("Phase 0 Complete - Aggregating Results")
    print("=" * 70)
    
    import pandas as pd
    df = pd.DataFrame(results)
    
    # Group by (wd, ls) and compute mean/std
    summary = df.groupby(['weight_decay', 'label_smoothing']).agg(
        mean_val_acc=('best_val_acc', 'mean'),
        std_val_acc=('best_val_acc', 'std'),
        mean_train_acc=('final_train_acc', 'mean'),
        mean_gap=('train_val_gap', 'mean'),
        any_underfitting=('is_underfitting', 'any'),
    ).reset_index()
    
    # Sort before rounding to preserve true ordering
    summary = summary.sort_values('mean_val_acc', ascending=False)
    
    # Save full precision summary
    summary_path = output_dir / "phase0_summary.csv"
    summary.to_csv(summary_path, index=False, float_format="%.6f")
    
    # Pretty display: keep wd/ls with higher precision, metrics rounded
    display_summary = summary.copy()
    display_summary["weight_decay"] = display_summary["weight_decay"].map(lambda x: f"{x:.6f}")
    display_summary["label_smoothing"] = display_summary["label_smoothing"].map(lambda x: f"{x:.4f}")
    for col in ["mean_val_acc", "std_val_acc", "mean_train_acc", "mean_gap"]:
        display_summary[col] = display_summary[col].round(2)
    
    print("\nResults (sorted by mean_val_acc):")
    print(display_summary.to_string(index=False))
    
    # Recommend best config (prefer non-underfitting if available)
    valid = summary[~summary['any_underfitting']]
    best = valid.iloc[0] if len(valid) > 0 else summary.iloc[0]
    if len(valid) == 0:
        print("\nWARNING: All configs show underfitting. Consider reducing regularization.")
    
    print(f"\n{'='*70}")
    print(f"RECOMMENDED CONFIG:")
    print(f"  weight_decay = {best['weight_decay']}")
    print(f"  label_smoothing = {best['label_smoothing']}")
    print(f"  mean_val_acc = {best['mean_val_acc']:.2f}% ± {best['std_val_acc']:.2f}%")
    print(f"  train_val_gap = {best['mean_gap']:.1f}%")
    print(f"{'='*70}")
    
    print(f"\nRaw results: {csv_path}")
    print(f"Summary: {summary_path}")
    
    return summary


if __name__ == "__main__":
    args = parse_args()
    run_calibration(args)

