# Phase A: Screening - Single operation evaluation with 2D Sobol sampling
"""
Phase A: Augmentation Screening Script.

v5 CHANGED: 2D Sobol sampling in (magnitude, probability) space.

Evaluates each candidate augmentation operation with Sobol-sampled (m, p) pairs.
Each operation uses customized search ranges from OP_SEARCH_SPACE.
Logs results to CSV for subsequent analysis.

Reference: docs/research_plan_v5.md Section 3 (Phase A)

Changelog (v4 → v5):
- [CHANGED] Sobol sampling: 1D (magnitude) → 2D (magnitude, probability)
- [NEW] Per-operation search space from OP_SEARCH_SPACE
- [CHANGED] CSV output includes actual probability values

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
import time
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

from src.augmentations import (
    AugmentationSpace,
    OP_SEARCH_SPACE,
    build_transform_with_op,
    get_val_transform,
)
from src.dataset import CIFAR100Subsampled
from src.models import create_model
from src.utils import (
    EarlyStopping,
    evaluate,
    get_device,
    get_optimizer_and_scheduler,
    set_seed_deterministic,
    train_one_epoch,
    ensure_dir,
)


# =============================================================================
# 2D Sobol Sampling (v5)
# =============================================================================

def generate_sobol_samples_2d(
    n_samples: int,
    m_range: Tuple[float, float],
    p_range: Tuple[float, float],
    seed: int = 42,
) -> np.ndarray:
    """Generate 2D Sobol sequence samples for (magnitude, probability).
    
    v5 NEW: 2D sampling in (m, p) space with per-operation ranges.
    
    Sobol sequences provide better coverage than random sampling
    for low-dimensional parameter spaces.
    
    Args:
        n_samples: Number of samples to generate.
        m_range: (min, max) for magnitude.
        p_range: (min, max) for probability.
        seed: Random seed for scrambling.
        
    Returns:
        Array of shape (n_samples, 2) where [:, 0] is magnitude, [:, 1] is probability.
    """
    from scipy.stats.qmc import Sobol
    
    # Sobol sampler for 2D (magnitude, probability)
    sampler = Sobol(d=2, scramble=True, seed=seed)
    samples = sampler.random(n_samples)  # Shape: (n_samples, 2) in [0, 1]
    
    # Scale to actual ranges
    m_min, m_max = m_range
    p_min, p_max = p_range
    
    samples[:, 0] = m_min + samples[:, 0] * (m_max - m_min)  # magnitude
    samples[:, 1] = p_min + samples[:, 1] * (p_max - p_min)  # probability
    
    return samples


# Legacy function for backward compatibility
def generate_sobol_samples(n_samples: int, seed: int = 42) -> np.ndarray:
    """Generate 1D Sobol sequence samples in [0, 1] (legacy).
    
    DEPRECATED: Use generate_sobol_samples_2d for v5 experiments.
    """
    from scipy.stats.qmc import Sobol
    sampler = Sobol(d=1, scramble=True, seed=seed)
    return sampler.random(n_samples).flatten()


# =============================================================================
# Single Configuration Training
# =============================================================================

def train_single_config(
    op_name: str,
    magnitude: float,
    probability: float,
    epochs: int,
    device: torch.device,
    fold_idx: int = 0,
    batch_size: int = 64,
    num_workers: int = 6,
    early_stop_patience: int = 5,
    seed: int = 42,
    deterministic: bool = True,
) -> Dict:
    """Train one configuration and return metrics.
    
    v5 CHANGED: Added probability parameter for stochastic application.
    
    Args:
        op_name: Name of the augmentation operation.
        magnitude: Magnitude value in [0, 1].
        probability: Probability of applying the augmentation.
        epochs: Number of training epochs.
        device: Device to train on.
        fold_idx: Which fold to use (default 0 for search).
        batch_size: Batch size (64 for low-data regime).
        num_workers: Number of data loading workers.
        early_stop_patience: Epochs to wait before early stopping.
        seed: Random seed for reproducibility.
        deterministic: If True, enable deterministic CUDA mode.
        
    Returns:
        Dict with unified CSV format fields.
    """
    start_time = time.time()
    set_seed_deterministic(seed, deterministic=deterministic)
    
    # Initialize result with unified format (v5: actual probability value)
    result = {
        "phase": "PhaseA",
        "op_name": op_name,
        "magnitude": str(round(magnitude, 4)),
        "probability": str(round(probability, 4)),
        "seed": seed,
        "fold_idx": fold_idx,
        "val_acc": -1.0,
        "val_loss": -1.0,
        "top5_acc": -1.0,
        "train_acc": -1.0,
        "train_loss": -1.0,
        "epochs_run": 0,
        "best_epoch": 0,
        "early_stopped": False,
        "runtime_sec": 0.0,
        "timestamp": "",
        "error": "",
    }
    
    # Build transforms (v5: with probability)
    train_transform = build_transform_with_op(
        op_name=op_name,
        magnitude=magnitude,
        probability=probability,
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
    
    # Create data loaders with optimized settings
    use_cuda = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_cuda,
        drop_last=False,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=4 if num_workers > 0 else None,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_cuda,
        drop_last=False,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=4 if num_workers > 0 else None,
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
    
    # Early stopping with grace period (ASHA-style, per research plan)
    # Grace period = 40 means no early stopping before epoch 40
    early_stopper = EarlyStopping(patience=early_stop_patience, mode="min", grace_period=40)
    
    # Training loop - track all metrics
    best_val_acc = 0.0
    best_val_loss = float("inf")
    best_top5_acc = 0.0
    best_train_acc = 0.0
    best_train_loss = 0.0
    best_epoch = 0
    early_stopped = False
    
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
        
        # Update best metrics (by val_acc)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_top5_acc = top5_acc
            best_train_acc = train_acc
            best_train_loss = train_loss
            best_epoch = epoch + 1
        
        # Step scheduler
        scheduler.step()
        
        # Early stopping check (with grace period)
        if early_stopper(val_loss, epoch):
            result["epochs_run"] = epoch + 1
            early_stopped = True
            break
        
        result["epochs_run"] = epoch + 1
    
    # Final results with unified format
    result["val_acc"] = round(best_val_acc, 4)
    result["val_loss"] = round(best_val_loss, 6)
    result["top5_acc"] = round(best_top5_acc, 4)
    result["train_acc"] = round(best_train_acc, 4)
    result["train_loss"] = round(best_train_loss, 6)
    result["best_epoch"] = best_epoch
    result["early_stopped"] = early_stopped
    result["runtime_sec"] = round(time.time() - start_time, 2)
    result["timestamp"] = datetime.now().isoformat(timespec='seconds')
    
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
    
    Uses unified CSV format for all phases.
    
    Args:
        path: Path to CSV file.
        row: Dict representing one row.
        write_header: If True, write header before row.
    """
    # Unified CSV fieldnames (same for all phases)
    fieldnames = [
        "phase", "op_name", "magnitude", "probability", "seed", "fold_idx",
        "val_acc", "val_loss", "top5_acc", "train_acc", "train_loss",
        "epochs_run", "best_epoch", "early_stopped", "runtime_sec",
        "timestamp", "error"
    ]
    
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
    
    parser.add_argument(
        "--ops",
        type=str,
        default=None,
        help="Comma-separated list of ops to evaluate (default: all ops). "
             "Example: --ops RandomRotation,ColorJitter"
    )
    
    return parser.parse_args()


def main() -> int:
    """Main entry point.
    
    Returns:
        Exit code (0 for success).
    """
    args = parse_args()
    
    # Setup
    set_seed_deterministic(args.seed, deterministic=True)
    device = get_device()
    
    print("=" * 70)
    print("Phase A: Augmentation Screening")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: 64")
    print(f"Fold: {args.fold_idx}")
    print(f"Seed: {args.seed}")
    print(f"Deterministic: True")
    print(f"LR: 0.05, WD: 1e-3, Momentum: 0.9")
    print(f"Early stop patience: {args.early_stop_patience}")
    print(f"Output dir: {args.output_dir}")
    print("-" * 70)
    print(f"v5: 2D Sobol samples per op: {args.n_samples}")
    print(f"Search space: (magnitude, probability) from OP_SEARCH_SPACE")
    print("=" * 70)
    
    # Output path
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)
    csv_path = output_dir / "phase_a_results.csv"
    
    # Check if we need to write header
    write_header = check_csv_needs_header(csv_path)
    
    # Get available operations
    all_ops = AugmentationSpace.get_available_ops()
    
    # Filter ops if --ops is specified
    if args.ops:
        ops = [op.strip() for op in args.ops.split(",")]
        # Validate ops
        invalid_ops = [op for op in ops if op not in all_ops]
        if invalid_ops:
            print(f"ERROR: Invalid ops: {invalid_ops}")
            print(f"Available ops: {all_ops}")
            return 1
    else:
        ops = all_ops
    
    print(f"Operations to evaluate: {len(ops)}")
    print(f"Operations: {ops}")
    
    # v5: Generate 2D Sobol samples for each operation with customized ranges
    configs = []
    print(f"\nGenerating {args.n_samples} Sobol samples per operation...")
    
    for op_name in ops:
        # Get per-operation search space
        space = OP_SEARCH_SPACE[op_name]
        m_range = tuple(space["m"])
        p_range = tuple(space["p"])
        
        # Generate 2D Sobol samples for this operation
        samples = generate_sobol_samples_2d(
            n_samples=args.n_samples,
            m_range=m_range,
            p_range=p_range,
            seed=args.seed,
        )
        
        # Add to configs: (op_name, magnitude, probability)
        for i in range(len(samples)):
            mag, prob = samples[i, 0], samples[i, 1]
            configs.append((op_name, float(mag), float(prob)))
        
        print(f"  {op_name}: m∈{m_range}, p∈{p_range} → {len(samples)} samples")
    
    # Limit configs for dry run
    if args.n_samples <= 2:
        configs = configs[:args.n_samples]
        print(f"Dry run mode: only running {len(configs)} configs")
    
    print(f"\nTotal configurations: {len(configs)}")
    print("-" * 70)
    
    # Main loop with tqdm on outermost level
    success_count = 0
    error_count = 0
    total_start_time = time.time()
    
    # v5: configs are now (op_name, magnitude, probability) tuples
    for op_name, magnitude, probability in tqdm(configs, desc="Phase A Screening", unit="config"):
        try:
            # Train and evaluate this configuration
            result = train_single_config(
                op_name=op_name,
                magnitude=magnitude,
                probability=probability,
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
            print(f"\nERROR in {op_name} (m={magnitude:.4f}, p={probability:.4f}): {error_msg}")
            traceback.print_exc()
            
            result = {
                "phase": "PhaseA",
                "op_name": op_name,
                "magnitude": str(round(magnitude, 4)),
                "probability": str(round(probability, 4)),
                "seed": args.seed,
                "fold_idx": args.fold_idx,
                "val_acc": -1.0,
                "val_loss": -1.0,
                "top5_acc": -1.0,
                "train_acc": -1.0,
                "train_loss": -1.0,
                "epochs_run": 0,
                "best_epoch": 0,
                "early_stopped": False,
                "runtime_sec": 0.0,
                "timestamp": datetime.now().isoformat(timespec='seconds'),
                "error": error_msg,
            }
            error_count += 1
        
        # Write result to CSV immediately
        write_csv_row(csv_path, result, write_header)
        write_header = False  # Only write header once
    
    # Summary
    total_runtime = time.time() - total_start_time
    
    print("\n" + "=" * 70)
    print("Phase A Complete")
    print("=" * 70)
    print(f"Successful: {success_count}")
    print(f"Failed: {error_count}")
    print(f"Total runtime: {total_runtime:.1f}s ({total_runtime/60:.1f}min)")
    print(f"Estimated per-config time: {total_runtime/max(success_count+error_count, 1):.1f}s")
    print("-" * 70)
    print(f"Results saved to: {csv_path}")
    print("=" * 70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())



