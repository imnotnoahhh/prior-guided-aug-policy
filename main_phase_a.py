# Phase A: Screening - Single operation evaluation with Sobol sampling
"""
Phase A: Augmentation Screening Script.

Evaluates each candidate augmentation operation with Sobol-sampled magnitudes.
Logs results to CSV for subsequent analysis.

Reference: docs/research_plan_v4.md Section 3 (Phase A)

Usage:
    # Full run (200 epochs, 32 samples per op)
    python main_phase_a.py

    # Dry run (1 epoch, 2 samples total)
    python main_phase_a.py --epochs 1 --n_samples 2
"""

import argparse
import csv
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.augmentations import AugmentationSpace, build_transform_with_op, get_val_transform
from src.dataset import CIFAR100Subsampled
from src.models import create_model
from src.utils import (
    EarlyStopping,
    evaluate,
    get_device,
    get_optimizer_and_scheduler,
    set_seed,
    train_one_epoch,
    ensure_dir,
)


# =============================================================================
# Sobol Sampling
# =============================================================================

def generate_sobol_samples(n_samples: int, seed: int = 42) -> np.ndarray:
    """Generate Sobol sequence samples in [0, 1].
    
    Sobol sequences provide better coverage than random sampling
    for low-dimensional parameter spaces.
    
    Args:
        n_samples: Number of samples to generate.
        seed: Random seed for scrambling.
        
    Returns:
        Array of shape (n_samples,) with values in [0, 1].
    """
    from scipy.stats.qmc import Sobol
    
    # Sobol sampler for 1D (magnitude only)
    sampler = Sobol(d=1, scramble=True, seed=seed)
    samples = sampler.random(n_samples).flatten()
    
    return samples


# =============================================================================
# Single Configuration Training
# =============================================================================

def train_single_config(
    op_name: str,
    magnitude: float,
    epochs: int,
    device: torch.device,
    fold_idx: int = 0,
    batch_size: int = 64,
    num_workers: int = 6,
    early_stop_patience: int = 5,
    seed: int = 42,
) -> Dict:
    """Train one configuration and return metrics.
    
    Args:
        op_name: Name of the augmentation operation.
        magnitude: Magnitude value in [0, 1].
        epochs: Number of training epochs.
        device: Device to train on.
        fold_idx: Which fold to use (default 0 for search).
        batch_size: Batch size (64 for low-data regime).
        num_workers: Number of data loading workers.
        early_stop_patience: Epochs to wait before early stopping.
        seed: Random seed for reproducibility.
        
    Returns:
        Dict with keys: op_name, magnitude, val_acc, val_loss, top5_acc,
                        epochs_run, train_time, error
    """
    set_seed(seed)
    
    result = {
        "op_name": op_name,
        "magnitude": magnitude,
        "val_acc": -1.0,
        "val_loss": -1.0,
        "top5_acc": -1.0,
        "epochs_run": 0,
        "error": "",
    }
    
    # Build transforms
    train_transform = build_transform_with_op(
        op_name=op_name,
        magnitude=magnitude,
        include_baseline=True,
        include_normalize=False,  # Keep in [0, 1] for augmentations
    )
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
    
    # Create data loaders
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
    
    # Optimizer and scheduler (optimized for low-data regime)
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
    
    # Early stopping
    early_stopper = EarlyStopping(patience=early_stop_patience, mode="min")
    
    # Training loop
    best_val_acc = 0.0
    best_val_loss = float("inf")
    best_top5_acc = 0.0
    
    for epoch in range(epochs):
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
        
        # Update best metrics
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_top5_acc = top5_acc
        
        # Step scheduler
        scheduler.step()
        
        # Early stopping check
        if early_stopper(val_loss):
            result["epochs_run"] = epoch + 1
            break
        
        result["epochs_run"] = epoch + 1
    
    # Final results
    result["val_acc"] = best_val_acc
    result["val_loss"] = best_val_loss
    result["top5_acc"] = best_top5_acc
    
    return result


# =============================================================================
# CSV Writing with Safety
# =============================================================================

def write_csv_row(
    path: Path,
    row: Dict,
    write_header: bool,
) -> None:
    """Append one row to CSV with immediate flush.
    
    Args:
        path: Path to CSV file.
        row: Dict representing one row.
        write_header: If True, write header before row.
    """
    fieldnames = ["op_name", "magnitude", "val_acc", "val_loss", "top5_acc", "epochs_run", "error"]
    
    # Ensure parent directory exists
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Open in append mode with line buffering
    with open(path, mode="a", buffering=1, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        if write_header:
            writer.writeheader()
        
        writer.writerow(row)
        f.flush()
        
        # Extra safety: sync to disk
        try:
            os.fsync(f.fileno())
        except OSError:
            pass  # fsync may not be available on all systems


def check_csv_needs_header(path: Path) -> bool:
    """Check if CSV file needs a header.
    
    Returns True if file doesn't exist or is empty.
    """
    if not path.exists():
        return True
    
    if path.stat().st_size == 0:
        return True
    
    return False


# =============================================================================
# Main Entry Point
# =============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Phase A: Augmentation Screening with Sobol Sampling"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
        help="Number of training epochs per config (default: 200)"
    )
    
    parser.add_argument(
        "--n_samples",
        type=int,
        default=32,
        help="Number of Sobol samples per operation (default: 32)"
    )
    
    parser.add_argument(
        "--fold_idx",
        type=int,
        default=0,
        help="Which fold to use for search (default: 0)"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Output directory for results (default: outputs)"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    
    parser.add_argument(
        "--num_workers",
        type=int,
        default=6,
        help="Number of data loading workers (default: 6)"
    )
    
    parser.add_argument(
        "--early_stop_patience",
        type=int,
        default=5,
        help="Early stopping patience (default: 5)"
    )
    
    return parser.parse_args()


def main() -> int:
    """Main entry point.
    
    Returns:
        Exit code (0 for success).
    """
    args = parse_args()
    
    print("=" * 70)
    print("Phase A: Augmentation Screening")
    print("=" * 70)
    print(f"Epochs per config: {args.epochs}")
    print(f"Sobol samples: {args.n_samples}")
    print(f"Fold: {args.fold_idx}")
    print(f"Output dir: {args.output_dir}")
    print(f"Seed: {args.seed}")
    print("=" * 70)
    
    # Setup
    set_seed(args.seed)
    device = get_device()
    print(f"Device: {device}")
    
    # Output path
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)
    csv_path = output_dir / "phase_a_results.csv"
    
    # Check if we need to write header
    write_header = check_csv_needs_header(csv_path)
    
    # Get available operations
    ops = AugmentationSpace.get_available_ops()
    print(f"Operations to evaluate: {len(ops)}")
    print(f"Operations: {ops}")
    
    # Generate Sobol samples for magnitudes
    magnitudes = generate_sobol_samples(args.n_samples, seed=args.seed)
    print(f"Magnitude samples: {magnitudes[:5]}... (total: {len(magnitudes)})")
    
    # Build list of all configurations
    # For dry run (n_samples=2), just pick first 2 configs total
    configs = []
    for op_name in ops:
        for mag in magnitudes:
            configs.append((op_name, float(mag)))
    
    # Limit configs for dry run
    if args.n_samples <= 2:
        configs = configs[:args.n_samples]
        print(f"Dry run mode: only running {len(configs)} configs")
    
    print(f"\nTotal configurations: {len(configs)}")
    print("-" * 70)
    
    # Main loop with tqdm on outermost level
    success_count = 0
    error_count = 0
    
    for op_name, magnitude in tqdm(configs, desc="Screening", unit="config"):
        try:
            # Train and evaluate this configuration
            result = train_single_config(
                op_name=op_name,
                magnitude=magnitude,
                epochs=args.epochs,
                device=device,
                fold_idx=args.fold_idx,
                num_workers=args.num_workers,
                early_stop_patience=args.early_stop_patience,
                seed=args.seed,
            )
            success_count += 1
            
        except Exception as e:
            # Log error but continue
            error_msg = f"{type(e).__name__}: {str(e)}"
            print(f"\nERROR in {op_name} (m={magnitude:.4f}): {error_msg}")
            traceback.print_exc()
            
            result = {
                "op_name": op_name,
                "magnitude": magnitude,
                "val_acc": -1.0,
                "val_loss": -1.0,
                "top5_acc": -1.0,
                "epochs_run": 0,
                "error": error_msg,
            }
            error_count += 1
        
        # Write result to CSV immediately
        write_csv_row(csv_path, result, write_header)
        write_header = False  # Only write header once
    
    # Summary
    print("\n" + "=" * 70)
    print("Phase A Complete")
    print("=" * 70)
    print(f"Successful configs: {success_count}")
    print(f"Failed configs: {error_count}")
    print(f"Results saved to: {csv_path}")
    print("=" * 70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())



