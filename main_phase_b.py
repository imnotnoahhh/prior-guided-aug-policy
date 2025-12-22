# Phase B: Tuning - Grid search around Top-4 configurations with 3-seed robustness
"""
Phase B: Augmentation Tuning Script.

Performs grid search around Top-4 configurations from Phase A for each promoted op.
Runs 3 random seeds per parameter point for robustness.
Outputs results sorted by Mean Validation Accuracy.

Reference: docs/research_plan_v4.md Section 3 (Phase B)

Usage:
    # Full run (200 epochs, 3 seeds)
    python main_phase_b.py

    # Dry run (2 epochs, 1 seed, specific ops)
    python main_phase_b.py --epochs 2 --seeds 42 --ops ColorJitter --dry_run
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
import pandas as pd
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
    set_seed_deterministic,
    train_one_epoch,
    ensure_dir,
)


# =============================================================================
# Phase A Results Loading
# =============================================================================

def load_phase_a_results(csv_path: Path) -> pd.DataFrame:
    """Load Phase A screening results from CSV.
    
    Args:
        csv_path: Path to phase_a_results.csv
        
    Returns:
        DataFrame with columns: op_name, magnitude, val_acc, val_loss, top5_acc, epochs_run, error
        
    Raises:
        FileNotFoundError: If CSV file doesn't exist.
        ValueError: If CSV is empty or has unexpected format.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Phase A results not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    if df.empty:
        raise ValueError(f"Phase A results CSV is empty: {csv_path}")
    
    required_cols = ["op_name", "magnitude", "val_acc", "top5_acc"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in Phase A CSV: {missing}")
    
    # Filter out error rows
    df = df[df["error"].isna() | (df["error"] == "")]
    
    return df


def load_baseline_result(csv_path: Path) -> Tuple[float, float]:
    """Load baseline result for promotion threshold calculation.
    
    Args:
        csv_path: Path to baseline_result.csv
        
    Returns:
        Tuple of (baseline_val_acc, baseline_top5_acc)
        
    Raises:
        FileNotFoundError: If CSV file doesn't exist.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Baseline results not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    if df.empty:
        raise ValueError(f"Baseline results CSV is empty: {csv_path}")
    
    baseline_acc = df["val_acc"].iloc[0]
    baseline_top5 = df["top5_acc"].iloc[0]
    
    return baseline_acc, baseline_top5


# =============================================================================
# Promoted Ops Selection
# =============================================================================

def get_promoted_ops(
    phase_a_df: pd.DataFrame,
    baseline_acc: float,
    baseline_top5: float,
    delta_threshold: float = 0.5,
) -> List[str]:
    """Determine which ops are promoted from Phase A to Phase B.
    
    Promotion criteria (from research_plan_v4.md):
    1. Top-1 Acc: Δ ≥ -0.5% (max_acc >= baseline_acc - 0.5)
    2. Top-5 Acc: Δ > 0% (max_top5 > baseline_top5)
    3. Loss Analysis: Not implemented (no loss slope in CSV)
    
    An op is promoted if ANY of the above conditions is met.
    
    Args:
        phase_a_df: DataFrame with Phase A results.
        baseline_acc: Baseline Top-1 accuracy.
        baseline_top5: Baseline Top-5 accuracy.
        delta_threshold: Allowed drop in Top-1 accuracy. Default 0.5%.
        
    Returns:
        List of promoted operation names.
    """
    acc_threshold = baseline_acc - delta_threshold
    
    promoted = []
    for op_name in phase_a_df["op_name"].unique():
        op_data = phase_a_df[phase_a_df["op_name"] == op_name]
        max_acc = op_data["val_acc"].max()
        max_top5 = op_data["top5_acc"].max()
        
        # Promotion condition: Top-1 >= threshold OR Top-5 > baseline
        if max_acc >= acc_threshold or max_top5 > baseline_top5:
            promoted.append(op_name)
    
    return promoted


# =============================================================================
# Top-K Configuration Selection
# =============================================================================

def get_top_k_configs(
    phase_a_df: pd.DataFrame,
    op_name: str,
    k: int = 4,
) -> List[float]:
    """Get top-K magnitude values for an operation from Phase A.
    
    Selects the K configurations with highest val_acc for the given op.
    
    Args:
        phase_a_df: DataFrame with Phase A results.
        op_name: Name of the operation.
        k: Number of top configurations to return. Default 4.
        
    Returns:
        List of magnitude values (centers for grid search).
    """
    op_data = phase_a_df[phase_a_df["op_name"] == op_name].copy()
    op_data = op_data.sort_values("val_acc", ascending=False)
    
    top_k = op_data.head(k)["magnitude"].tolist()
    
    return top_k


# =============================================================================
# Local Grid Construction
# =============================================================================

def build_local_grid(
    centers: List[float],
    step: float = 0.05,
    n_steps: int = 2,
) -> List[float]:
    """Build local grid around center magnitudes.
    
    For each center, generates points at ±step intervals up to n_steps away.
    Merges all points, removes duplicates, and clamps to [0, 1].
    
    Args:
        centers: List of center magnitude values.
        step: Step size for grid. Default 0.05.
        n_steps: Number of steps in each direction. Default 2.
        
    Returns:
        Sorted list of unique magnitude values in [0, 1].
        
    Example:
        centers=[0.5], step=0.05, n_steps=2
        -> [0.4, 0.45, 0.5, 0.55, 0.6]
    """
    points = set()
    
    for center in centers:
        for i in range(-n_steps, n_steps + 1):
            point = center + i * step
            # Clamp to [0, 1]
            point = max(0.0, min(1.0, point))
            points.add(round(point, 4))  # Round to avoid float precision issues
    
    return sorted(list(points))


# =============================================================================
# Single Configuration Training
# =============================================================================

def train_single_config(
    op_name: str,
    magnitude: float,
    seed: int,
    epochs: int,
    device: torch.device,
    fold_idx: int = 0,
    batch_size: int = 64,
    num_workers: int = 6,
    early_stop_patience: int = 5,
    deterministic: bool = True,
) -> Dict:
    """Train one configuration and return metrics.
    
    Args:
        op_name: Name of the augmentation operation.
        magnitude: Magnitude value in [0, 1].
        seed: Random seed for this run.
        epochs: Number of training epochs.
        device: Device to train on.
        fold_idx: Which fold to use (default 0 for search).
        batch_size: Batch size (64 for low-data regime).
        num_workers: Number of data loading workers.
        early_stop_patience: Epochs to wait before early stopping.
        deterministic: If True, enable deterministic CUDA mode.
        
    Returns:
        Dict with keys: op_name, magnitude, seed, val_acc, val_loss, top5_acc,
                        epochs_run, runtime_sec, error
    """
    start_time = time.time()
    
    # Set seed with deterministic mode
    set_seed_deterministic(seed, deterministic=deterministic)
    
    result = {
        "op_name": op_name,
        "magnitude": magnitude,
        "seed": seed,
        "val_acc": -1.0,
        "val_loss": -1.0,
        "top5_acc": -1.0,
        "epochs_run": 0,
        "runtime_sec": 0.0,
        "error": "",
    }
    
    try:
        # Build transforms
        train_transform = build_transform_with_op(
            op_name=op_name,
            magnitude=magnitude,
            include_baseline=True,
            include_normalize=False,
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
        
        # Optimizer and scheduler (fixed hyperparameters per No-NAS)
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
        
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {str(e)}"
        traceback.print_exc()
    
    result["runtime_sec"] = time.time() - start_time
    
    return result


# =============================================================================
# CSV Writing
# =============================================================================

def write_raw_csv_row(
    path: Path,
    row: Dict,
    write_header: bool,
) -> None:
    """Append one row to raw CSV with immediate flush.
    
    Args:
        path: Path to CSV file.
        row: Dict representing one row.
        write_header: If True, write header before row.
    """
    fieldnames = [
        "op_name", "magnitude", "seed", "val_acc", "val_loss",
        "top5_acc", "epochs_run", "runtime_sec", "error"
    ]
    
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, mode="a", buffering=1, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        if write_header:
            writer.writeheader()
        
        writer.writerow(row)
        f.flush()
        
        try:
            os.fsync(f.fileno())
        except OSError:
            pass


def check_csv_needs_header(path: Path) -> bool:
    """Check if CSV file needs a header."""
    if not path.exists():
        return True
    if path.stat().st_size == 0:
        return True
    return False


# =============================================================================
# Results Aggregation
# =============================================================================

def aggregate_results(raw_csv_path: Path, summary_csv_path: Path) -> pd.DataFrame:
    """Aggregate raw results into summary with mean/std per (op, magnitude).
    
    Groups by (op_name, magnitude) and computes:
    - mean_val_acc, std_val_acc
    - mean_top5_acc, std_top5_acc
    - n_seeds (should be 3 for valid runs)
    
    Sorts by mean_val_acc descending.
    
    Args:
        raw_csv_path: Path to raw results CSV.
        summary_csv_path: Path to write summary CSV.
        
    Returns:
        Summary DataFrame.
    """
    df = pd.read_csv(raw_csv_path)
    
    # Filter out error rows
    df = df[(df["error"].isna()) | (df["error"] == "")]
    
    # Group by (op_name, magnitude)
    summary = df.groupby(["op_name", "magnitude"]).agg(
        mean_val_acc=("val_acc", "mean"),
        std_val_acc=("val_acc", "std"),
        mean_top5_acc=("top5_acc", "mean"),
        std_top5_acc=("top5_acc", "std"),
        n_seeds=("seed", "count"),
    ).reset_index()
    
    # Fill NaN std (happens when n=1)
    summary["std_val_acc"] = summary["std_val_acc"].fillna(0.0)
    summary["std_top5_acc"] = summary["std_top5_acc"].fillna(0.0)
    
    # Sort by mean_val_acc descending
    summary = summary.sort_values("mean_val_acc", ascending=False)
    
    # Save to CSV
    summary.to_csv(summary_csv_path, index=False)
    
    return summary


# =============================================================================
# Main Grid Search
# =============================================================================

def run_phase_b_grid_search(
    phase_a_csv: Path,
    baseline_csv: Path,
    output_dir: Path,
    epochs: int = 200,
    seeds: List[int] = [42, 123, 456],
    fold_idx: int = 0,
    batch_size: int = 64,
    num_workers: int = 6,
    early_stop_patience: int = 5,
    deterministic: bool = True,
    top_k: int = 4,
    grid_step: float = 0.05,
    grid_n_steps: int = 2,
    ops_filter: Optional[List[str]] = None,
    max_grid_points: Optional[int] = None,
    dry_run: bool = False,
) -> Path:
    """Run Phase B grid search with multi-seed robustness.
    
    Args:
        phase_a_csv: Path to Phase A results CSV.
        baseline_csv: Path to baseline results CSV.
        output_dir: Directory for output files.
        epochs: Training epochs per config.
        seeds: List of random seeds for robustness.
        fold_idx: Which fold to use (0 for search).
        batch_size: Training batch size.
        num_workers: Data loading workers.
        early_stop_patience: Early stopping patience.
        deterministic: Enable deterministic CUDA mode.
        top_k: Number of top configs per op as grid centers.
        grid_step: Step size for local grid.
        grid_n_steps: Number of steps in each direction.
        ops_filter: If provided, only tune these ops.
        max_grid_points: If provided, limit grid points per op (for testing).
        dry_run: If True, only run minimal configs for testing.
        
    Returns:
        Path to summary CSV.
    """
    # Setup
    device = get_device()
    ensure_dir(output_dir)
    
    raw_csv_path = output_dir / "phase_b_tuning_raw.csv"
    summary_csv_path = output_dir / "phase_b_tuning_summary.csv"
    
    # Load Phase A results and baseline
    print("Loading Phase A results...")
    phase_a_df = load_phase_a_results(phase_a_csv)
    baseline_acc, baseline_top5 = load_baseline_result(baseline_csv)
    
    print(f"Baseline: Top-1={baseline_acc:.1f}%, Top-5={baseline_top5:.1f}%")
    
    # Determine promoted ops
    promoted_ops = get_promoted_ops(phase_a_df, baseline_acc, baseline_top5)
    print(f"Promoted ops ({len(promoted_ops)}): {promoted_ops}")
    
    # Filter ops if specified
    if ops_filter:
        promoted_ops = [op for op in promoted_ops if op in ops_filter]
        print(f"Filtered to: {promoted_ops}")
    
    if not promoted_ops:
        print("WARNING: No promoted ops found!")
        return summary_csv_path
    
    # Build configurations
    all_configs = []
    for op_name in promoted_ops:
        centers = get_top_k_configs(phase_a_df, op_name, k=top_k)
        grid_points = build_local_grid(centers, step=grid_step, n_steps=grid_n_steps)
        
        if max_grid_points:
            grid_points = grid_points[:max_grid_points]
        
        for mag in grid_points:
            for seed in seeds:
                all_configs.append((op_name, mag, seed))
        
        print(f"  {op_name}: {len(centers)} centers -> {len(grid_points)} grid points")
    
    total_runs = len(all_configs)
    print(f"\nTotal configurations: {total_runs}")
    print(f"  = {len(promoted_ops)} ops × ~grid_points × {len(seeds)} seeds")
    
    if dry_run:
        # Limit to first few configs for testing
        all_configs = all_configs[:min(len(all_configs), len(seeds) * 2)]
        print(f"DRY RUN: Limited to {len(all_configs)} configs")
    
    # Check if we need to write header
    write_header = check_csv_needs_header(raw_csv_path)
    
    # Main training loop
    print(f"\nStarting Phase B grid search...")
    print(f"Device: {device}")
    print(f"Epochs: {epochs}")
    print(f"Seeds: {seeds}")
    print(f"Deterministic: {deterministic}")
    print("-" * 70)
    
    success_count = 0
    error_count = 0
    
    for op_name, magnitude, seed in tqdm(all_configs, desc="Phase B Tuning", unit="run"):
        try:
            result = train_single_config(
                op_name=op_name,
                magnitude=magnitude,
                seed=seed,
                epochs=epochs,
                device=device,
                fold_idx=fold_idx,
                batch_size=batch_size,
                num_workers=num_workers,
                early_stop_patience=early_stop_patience,
                deterministic=deterministic,
            )
            
            if result["error"]:
                error_count += 1
            else:
                success_count += 1
                
        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            print(f"\nERROR in {op_name} (m={magnitude:.4f}, seed={seed}): {error_msg}")
            traceback.print_exc()
            
            result = {
                "op_name": op_name,
                "magnitude": magnitude,
                "seed": seed,
                "val_acc": -1.0,
                "val_loss": -1.0,
                "top5_acc": -1.0,
                "epochs_run": 0,
                "runtime_sec": 0.0,
                "error": error_msg,
            }
            error_count += 1
        
        # Write result immediately
        write_raw_csv_row(raw_csv_path, result, write_header)
        write_header = False
    
    # Aggregate results
    print("\n" + "-" * 70)
    print("Aggregating results...")
    summary_df = aggregate_results(raw_csv_path, summary_csv_path)
    
    # Print summary
    print("\n" + "=" * 70)
    print("Phase B Complete")
    print("=" * 70)
    print(f"Successful runs: {success_count}")
    print(f"Failed runs: {error_count}")
    print(f"Raw results: {raw_csv_path}")
    print(f"Summary: {summary_csv_path}")
    print("\nTop 10 configurations by mean_val_acc:")
    print(summary_df.head(10).to_string(index=False))
    print("=" * 70)
    
    return summary_csv_path


# =============================================================================
# Argument Parsing
# =============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Phase B: Augmentation Tuning with Grid Search and Multi-Seed Robustness"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
        help="Number of training epochs per config (default: 200)"
    )
    
    parser.add_argument(
        "--seeds",
        type=str,
        default="42,123,456",
        help="Comma-separated list of random seeds (default: 42,123,456)"
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
        "--phase_a_csv",
        type=str,
        default="outputs/phase_a_results.csv",
        help="Path to Phase A results CSV"
    )
    
    parser.add_argument(
        "--baseline_csv",
        type=str,
        default="outputs/baseline_result.csv",
        help="Path to baseline results CSV"
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
        "--deterministic",
        action="store_true",
        help="Enable deterministic CUDA mode for reproducibility"
    )
    
    parser.add_argument(
        "--top_k",
        type=int,
        default=4,
        help="Number of top configs per op as grid centers (default: 4)"
    )
    
    parser.add_argument(
        "--grid_step",
        type=float,
        default=0.05,
        help="Step size for local grid (default: 0.05)"
    )
    
    parser.add_argument(
        "--grid_n_steps",
        type=int,
        default=2,
        help="Number of steps in each direction for grid (default: 2)"
    )
    
    parser.add_argument(
        "--ops",
        type=str,
        default=None,
        help="Comma-separated list of ops to tune (default: all promoted ops)"
    )
    
    parser.add_argument(
        "--grid_points",
        type=int,
        default=None,
        help="Limit grid points per op (for testing)"
    )
    
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Run minimal configs for testing"
    )
    
    return parser.parse_args()


# =============================================================================
# Main Entry Point
# =============================================================================

def main() -> int:
    """Main entry point.
    
    Returns:
        Exit code (0 for success).
    """
    args = parse_args()
    
    # Parse seeds
    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    
    # Parse ops filter
    ops_filter = None
    if args.ops:
        ops_filter = [op.strip() for op in args.ops.split(",")]
    
    print("=" * 70)
    print("Phase B: Augmentation Tuning")
    print("=" * 70)
    print(f"Epochs per config: {args.epochs}")
    print(f"Seeds: {seeds}")
    print(f"Fold: {args.fold_idx}")
    print(f"Output dir: {args.output_dir}")
    print(f"Phase A CSV: {args.phase_a_csv}")
    print(f"Baseline CSV: {args.baseline_csv}")
    print(f"Top-K centers: {args.top_k}")
    print(f"Grid step: {args.grid_step}")
    print(f"Grid n_steps: {args.grid_n_steps}")
    print(f"Deterministic: {args.deterministic}")
    if ops_filter:
        print(f"Ops filter: {ops_filter}")
    if args.dry_run:
        print("MODE: DRY RUN")
    print("=" * 70)
    
    try:
        run_phase_b_grid_search(
            phase_a_csv=Path(args.phase_a_csv),
            baseline_csv=Path(args.baseline_csv),
            output_dir=Path(args.output_dir),
            epochs=args.epochs,
            seeds=seeds,
            fold_idx=args.fold_idx,
            num_workers=args.num_workers,
            early_stop_patience=args.early_stop_patience,
            deterministic=args.deterministic,
            top_k=args.top_k,
            grid_step=args.grid_step,
            grid_n_steps=args.grid_n_steps,
            ops_filter=ops_filter,
            max_grid_points=args.grid_points,
            dry_run=args.dry_run,
        )
        return 0
        
    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
