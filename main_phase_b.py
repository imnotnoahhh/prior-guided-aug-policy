# Phase B: Tuning - ASHA Early-Stopping Tournament
"""
Phase B: Augmentation Tuning Script with ASHA Scheduler.

v5.3 CHANGED: ASHA (Asynchronous Successive Halving) replaces Grid Search.
- Sobol sampling instead of grid search (more exploration, less bias)
- Multi-fidelity early stopping: 30ep → 80ep → 200ep
- Each rung keeps top 1/3, eliminates weak configs early
- ~10x faster than full grid search with same or better results

Reference: docs/research_plan_v5.md Section 3 (Phase B)

Changelog:
- v5.3: ASHA scheduler with Sobol sampling
- v5.2: channels_last, prefetch_factor=4
- v5.1: Early stopping monitors val_acc
- v5.0: 2D Grid Search in (m, p) space

Usage:
    # Full ASHA run (~2-4 hours instead of ~28 hours)
    python main_phase_b.py
    
    # Dry run
    python main_phase_b.py --n_samples 5 --dry_run
    
    # Custom rungs
    python main_phase_b.py --rungs 30,80,200 --reduction_factor 3
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
# Phase A Results Loading
# =============================================================================

def load_phase_a_results(csv_path: Path) -> pd.DataFrame:
    """Load Phase A screening results from CSV."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Phase A results not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    if df.empty:
        raise ValueError(f"Phase A results CSV is empty: {csv_path}")
    
    required_cols = ["op_name", "magnitude", "val_acc", "top5_acc"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in Phase A CSV: {missing}")
    
    df = df[df["error"].isna() | (df["error"] == "")]
    return df


def load_baseline_result(csv_path: Path) -> Tuple[float, float, float]:
    """Load baseline result for promotion threshold calculation."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Baseline results not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"Baseline results CSV is empty: {csv_path}")
    
    return df["val_acc"].iloc[0], df["top5_acc"].iloc[0], df["train_loss"].iloc[0]


# =============================================================================
# Promoted Ops Selection
# =============================================================================

def get_promoted_ops(
    phase_a_df: pd.DataFrame,
    baseline_acc: float,
    baseline_top5: float,
    baseline_train_loss: float,
    delta_threshold: float = 0.5,
) -> List[str]:
    """Determine which ops are promoted from Phase A to Phase B."""
    acc_threshold = baseline_acc - delta_threshold
    
    promoted = []
    for op_name in phase_a_df["op_name"].unique():
        op_data = phase_a_df[phase_a_df["op_name"] == op_name]
        max_acc = op_data["val_acc"].max()
        max_top5 = op_data["top5_acc"].max()
        min_train_loss = op_data["train_loss"].min()
        
        if (max_acc >= acc_threshold or 
            max_top5 > baseline_top5 or 
            min_train_loss <= baseline_train_loss):
            promoted.append(op_name)
    
    return promoted


# =============================================================================
# Sobol Sampling for (m, p) Space (v5.3 NEW)
# =============================================================================

def sobol_sample_configs(
    op_name: str,
    n_samples: int = 30,
    seed: int = 42,
) -> List[Tuple[float, float]]:
    """Sample (magnitude, probability) pairs using Sobol sequence.
    
    Uses operation-specific bounds from OP_SEARCH_SPACE.
    
    Args:
        op_name: Name of the operation.
        n_samples: Number of samples to generate.
        seed: Random seed for Sobol sequence.
        
    Returns:
        List of (magnitude, probability) tuples.
    """
    try:
        from scipy.stats.qmc import Sobol
        sampler = Sobol(d=2, scramble=True, seed=seed)
        samples = sampler.random(n_samples)
    except ImportError:
        # Fallback to numpy random if scipy not available
        np.random.seed(seed)
        samples = np.random.rand(n_samples, 2)
    
    # Get operation-specific bounds
    space = OP_SEARCH_SPACE[op_name]
    m_min, m_max = space["m"]
    p_min, p_max = space["p"]
    
    configs = []
    for m_unit, p_unit in samples:
        m = m_min + m_unit * (m_max - m_min)
        p = p_min + p_unit * (p_max - p_min)
        configs.append((round(m, 4), round(p, 4)))
    
    return configs


# =============================================================================
# ASHA Training with Checkpoint Support (v5.3 NEW)
# =============================================================================

def train_to_epoch(
    op_name: str,
    magnitude: float,
    probability: float,
    target_epochs: int,
    device: torch.device,
    checkpoint: Optional[Dict] = None,
    fold_idx: int = 0,
    batch_size: int = 512,
    num_workers: int = 8,
    seed: int = 42,
    deterministic: bool = True,
) -> Tuple[Dict, Dict]:
    """Train a configuration to a target epoch, optionally resuming from checkpoint.
    
    Args:
        op_name: Name of the augmentation operation.
        magnitude: Magnitude value.
        probability: Probability value.
        target_epochs: Train until this epoch.
        device: Device to train on.
        checkpoint: Optional checkpoint dict to resume from.
        fold_idx: Which fold to use.
        batch_size: Batch size.
        num_workers: Number of data loading workers.
        seed: Random seed.
        deterministic: Enable deterministic mode.
        
    Returns:
        Tuple of (result_dict, checkpoint_dict for continuation)
    """
    start_time = time.time()
    set_seed_deterministic(seed, deterministic=deterministic)
    
    # Determine starting epoch
    start_epoch = 0
    if checkpoint is not None:
        start_epoch = checkpoint.get("epoch", 0)
    
    # Build transforms
    train_transform = build_transform_with_op(
        op_name=op_name,
        magnitude=magnitude,
        probability=probability,
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
    
    # Create model with channels_last
    model = create_model(num_classes=100, pretrained=False)
    model = model.to(device)
    if use_cuda:
        model = model.to(memory_format=torch.channels_last)
    
    criterion = nn.CrossEntropyLoss()
    
    # Create optimizer and scheduler for FULL training duration
    # (scheduler needs to know total epochs for proper cosine annealing)
    optimizer, scheduler = get_optimizer_and_scheduler(
        model=model,
        total_epochs=200,  # Fixed max epochs for consistent LR schedule
        lr=0.4,
        weight_decay=1e-3,
        momentum=0.9,
        warmup_epochs=5,
    )
    
    # AMP scaler
    scaler = None
    if use_cuda:
        scaler = torch.amp.GradScaler()
    
    # Restore from checkpoint if provided
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if scaler is not None and "scaler_state_dict" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
    
    # Training loop
    best_val_acc = checkpoint.get("best_val_acc", 0.0) if checkpoint else 0.0
    best_val_loss = checkpoint.get("best_val_loss", float("inf")) if checkpoint else float("inf")
    best_top5_acc = checkpoint.get("best_top5_acc", 0.0) if checkpoint else 0.0
    best_train_acc = checkpoint.get("best_train_acc", 0.0) if checkpoint else 0.0
    best_train_loss = checkpoint.get("best_train_loss", 0.0) if checkpoint else 0.0
    best_epoch = checkpoint.get("best_epoch", 0) if checkpoint else 0
    
    for epoch in range(start_epoch, target_epochs):
        train_loss, train_acc = train_one_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
        )
        
        val_loss, val_acc, top5_acc = evaluate(
            model=model,
            val_loader=val_loader,
            criterion=criterion,
            device=device,
        )
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_top5_acc = top5_acc
            best_train_acc = train_acc
            best_train_loss = train_loss
            best_epoch = epoch + 1
        
        scheduler.step()
    
    # Build result
    result = {
        "phase": "PhaseB",
        "op_name": op_name,
        "magnitude": str(round(magnitude, 4)),
        "probability": str(round(probability, 4)),
        "seed": seed,
        "fold_idx": fold_idx,
        "val_acc": round(best_val_acc, 4),
        "val_loss": round(best_val_loss, 6),
        "top5_acc": round(best_top5_acc, 4),
        "train_acc": round(best_train_acc, 4),
        "train_loss": round(best_train_loss, 6),
        "epochs_run": target_epochs,
        "best_epoch": best_epoch,
        "early_stopped": False,
        "runtime_sec": round(time.time() - start_time, 2),
        "timestamp": datetime.now().isoformat(timespec='seconds'),
        "error": "",
    }
    
    # Build checkpoint for continuation
    new_checkpoint = {
        "epoch": target_epochs,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_val_acc": best_val_acc,
        "best_val_loss": best_val_loss,
        "best_top5_acc": best_top5_acc,
        "best_train_acc": best_train_acc,
        "best_train_loss": best_train_loss,
        "best_epoch": best_epoch,
    }
    if scaler is not None:
        new_checkpoint["scaler_state_dict"] = scaler.state_dict()
    
    return result, new_checkpoint


# =============================================================================
# CSV Writing
# =============================================================================

def write_raw_csv_row(path: Path, row: Dict, write_header: bool) -> None:
    """Append one row to raw CSV with immediate flush."""
    fieldnames = [
        "phase", "op_name", "magnitude", "probability", "seed", "fold_idx",
        "val_acc", "val_loss", "top5_acc", "train_acc", "train_loss",
        "epochs_run", "best_epoch", "early_stopped", "runtime_sec",
        "timestamp", "error"
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
    """Aggregate raw results into summary with mean/std per (op, m, p)."""
    if not raw_csv_path.exists():
        print(f"WARNING: Raw CSV not found: {raw_csv_path}")
        return pd.DataFrame()
    
    df = pd.read_csv(raw_csv_path)
    
    if df.empty:
        print("WARNING: Raw CSV is empty")
        return pd.DataFrame()
    
    # Filter out errors
    df = df[df["error"].isna() | (df["error"] == "")]
    
    if df.empty:
        print("WARNING: No successful runs in raw CSV")
        return pd.DataFrame()
    
    # Group by (op_name, magnitude, probability)
    summary = df.groupby(["op_name", "magnitude", "probability"]).agg(
        mean_val_acc=("val_acc", "mean"),
        std_val_acc=("val_acc", "std"),
        mean_top5_acc=("top5_acc", "mean"),
        std_top5_acc=("top5_acc", "std"),
        mean_train_acc=("train_acc", "mean"),
        std_train_acc=("train_acc", "std"),
        mean_runtime_sec=("runtime_sec", "mean"),
        n_seeds=("seed", "count"),
    ).reset_index()
    
    summary["std_val_acc"] = summary["std_val_acc"].fillna(0)
    summary["std_top5_acc"] = summary["std_top5_acc"].fillna(0)
    summary["std_train_acc"] = summary["std_train_acc"].fillna(0)
    
    summary = summary.sort_values("mean_val_acc", ascending=False)
    summary.to_csv(summary_csv_path, index=False)
    
    return summary


# =============================================================================
# ASHA Scheduler (v5.3 NEW)
# =============================================================================

def run_phase_b_asha(
    phase_a_csv: Path,
    baseline_csv: Path,
    output_dir: Path,
    rungs: List[int] = [30, 80, 200],
    reduction_factor: int = 3,
    n_samples: int = 30,
    seed: int = 42,
    fold_idx: int = 0,
    batch_size: int = 512,
    num_workers: int = 8,
    deterministic: bool = True,
    ops_filter: Optional[List[str]] = None,
    dry_run: bool = False,
) -> Path:
    """Run Phase B with ASHA scheduler.
    
    ASHA (Asynchronous Successive Halving Algorithm):
    1. Sample n_samples (m, p) points per op using Sobol
    2. Train all to first rung (e.g., 30 epochs)
    3. Keep top 1/reduction_factor, continue to next rung
    4. Repeat until final rung
    
    Args:
        phase_a_csv: Path to Phase A results CSV.
        baseline_csv: Path to baseline results CSV.
        output_dir: Output directory.
        rungs: List of epoch checkpoints [30, 80, 200].
        reduction_factor: Keep 1/r configs each rung. Default 3.
        n_samples: Number of Sobol samples per op. Default 30.
        seed: Random seed for Sobol sampling.
        fold_idx: Which fold to use.
        batch_size: Training batch size.
        num_workers: Data loading workers.
        deterministic: Enable deterministic mode.
        ops_filter: If provided, only tune these ops.
        dry_run: If True, run minimal configs.
        
    Returns:
        Path to summary CSV.
    """
    device = get_device()
    ensure_dir(output_dir)
    
    raw_csv_path = output_dir / "phase_b_tuning_raw.csv"
    summary_csv_path = output_dir / "phase_b_tuning_summary.csv"
    
    # Load Phase A results
    print("Loading Phase A results...")
    phase_a_df = load_phase_a_results(phase_a_csv)
    baseline_acc, baseline_top5, baseline_train_loss = load_baseline_result(baseline_csv)
    
    print(f"Baseline: Top-1={baseline_acc:.1f}%, Top-5={baseline_top5:.1f}%")
    
    # Determine promoted ops
    promoted_ops = get_promoted_ops(phase_a_df, baseline_acc, baseline_top5, baseline_train_loss)
    print(f"Promoted ops ({len(promoted_ops)}): {promoted_ops}")
    
    if ops_filter:
        promoted_ops = [op for op in promoted_ops if op in ops_filter]
        print(f"Filtered to: {promoted_ops}")
    
    if not promoted_ops:
        print("WARNING: No promoted ops found!")
        return summary_csv_path
    
    # Generate Sobol samples for each op
    all_configs = []  # (op_name, magnitude, probability)
    for op_name in promoted_ops:
        configs = sobol_sample_configs(op_name, n_samples=n_samples, seed=seed)
        for m, p in configs:
            all_configs.append((op_name, m, p))
        print(f"  {op_name}: {len(configs)} Sobol samples")
    
    if dry_run:
        all_configs = all_configs[:min(len(all_configs), 5)]
        rungs = [5, 10]  # Quick test
        print(f"DRY RUN: Limited to {len(all_configs)} configs, rungs={rungs}")
    
    print(f"\nTotal initial configs: {len(all_configs)}")
    print(f"ASHA rungs: {rungs}")
    print(f"Reduction factor: 1/{reduction_factor}")
    
    # Estimate time savings
    full_epochs = len(all_configs) * rungs[-1]
    asha_epochs = 0
    remaining = len(all_configs)
    prev_rung = 0
    for rung in rungs:
        asha_epochs += remaining * (rung - prev_rung)
        remaining = max(1, remaining // reduction_factor)
        prev_rung = rung
    
    print(f"Estimated epoch-equivalents: {asha_epochs} (vs {full_epochs} full = {100*asha_epochs/full_epochs:.1f}%)")
    
    # Check CSV header
    write_header = check_csv_needs_header(raw_csv_path)
    
    # ASHA main loop
    print(f"\n{'='*70}")
    print("Starting ASHA Phase B...")
    print("="*70)
    
    # Track configs with their checkpoints
    # Use index as key to avoid issues with op names containing underscores
    # config_idx -> (op_name, m, p, checkpoint_dict)
    active_configs = {i: (op, m, p, None) for i, (op, m, p) in enumerate(all_configs)}
    
    total_start = time.time()
    
    for rung_idx, target_epochs in enumerate(rungs):
        print(f"\n{'='*70}")
        print(f"RUNG {rung_idx + 1}/{len(rungs)}: Training to {target_epochs} epochs")
        print(f"Active configs: {len(active_configs)}")
        print("="*70)
        
        rung_results = []
        
        for key, (op_name, m, p, checkpoint) in tqdm(
            active_configs.items(), 
            desc=f"Rung {rung_idx+1} ({target_epochs}ep)",
            unit="cfg"
        ):
            try:
                result, new_checkpoint = train_to_epoch(
                    op_name=op_name,
                    magnitude=m,
                    probability=p,
                    target_epochs=target_epochs,
                    device=device,
                    checkpoint=checkpoint,
                    fold_idx=fold_idx,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    seed=seed,
                    deterministic=deterministic,
                )
                
                # Store result
                rung_results.append((key, result["val_acc"], new_checkpoint))
                
                # Write to CSV (only at final rung for cleaner output)
                if rung_idx == len(rungs) - 1:
                    write_raw_csv_row(raw_csv_path, result, write_header)
                    write_header = False
                    
            except Exception as e:
                print(f"\nERROR in {op_name} (m={m}, p={p}): {e}")
                traceback.print_exc()
                # Remove failed config
                rung_results.append((key, -1.0, None))
        
        # Select top configs for next rung
        if rung_idx < len(rungs) - 1:
            # Sort by val_acc descending
            rung_results.sort(key=lambda x: x[1], reverse=True)
            
            # Keep top 1/reduction_factor
            n_keep = max(1, len(rung_results) // reduction_factor)
            survivors = rung_results[:n_keep]
            
            print(f"\nSurvivors: {n_keep}/{len(rung_results)} (top 1/{reduction_factor})")
            print(f"  Best: {survivors[0][1]:.2f}%, Cutoff: {survivors[-1][1]:.2f}%")
            
            # Update active_configs - get original config info from current active_configs
            new_active = {}
            for idx, val_acc, checkpoint in survivors:
                if checkpoint is not None and idx in active_configs:
                    op, m, p, _ = active_configs[idx]
                    new_active[idx] = (op, m, p, checkpoint)
            active_configs = new_active
    
    total_time = time.time() - total_start
    print(f"\n{'='*70}")
    print(f"ASHA Phase B Complete!")
    print(f"Total time: {total_time/3600:.2f} hours")
    print("="*70)
    
    # Aggregate results
    print("\nAggregating results...")
    summary = aggregate_results(raw_csv_path, summary_csv_path)
    
    if not summary.empty:
        print("\nTop 10 configurations by mean_val_acc:")
        print(summary.head(10).to_string(index=False))
    
    print(f"\nRaw results: {raw_csv_path}")
    print(f"Summary: {summary_csv_path}")
    
    return summary_csv_path


# =============================================================================
# Argument Parsing
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Phase B: ASHA Augmentation Tuning (v5.3)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Full ASHA run (~2-4 hours)
    python main_phase_b.py
    
    # Quick test
    python main_phase_b.py --n_samples 5 --dry_run
    
    # Custom rungs (more aggressive)
    python main_phase_b.py --rungs 20,60,200 --reduction_factor 4
    
    # Specific ops only
    python main_phase_b.py --ops ColorJitter,GaussianBlur
"""
    )
    
    parser.add_argument(
        "--output_dir", type=str, default="outputs",
        help="Output directory (default: outputs)"
    )
    parser.add_argument(
        "--phase_a_csv", type=str, default="outputs/phase_a_results.csv",
        help="Path to Phase A results CSV"
    )
    parser.add_argument(
        "--baseline_csv", type=str, default="outputs/baseline_result.csv",
        help="Path to baseline results CSV"
    )
    
    # ASHA parameters
    parser.add_argument(
        "--rungs", type=str, default="30,80,200",
        help="Comma-separated epoch checkpoints (default: 30,80,200)"
    )
    parser.add_argument(
        "--reduction_factor", type=int, default=3,
        help="Keep 1/r configs each rung (default: 3)"
    )
    parser.add_argument(
        "--n_samples", type=int, default=30,
        help="Number of Sobol samples per op (default: 30)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for Sobol sampling (default: 42)"
    )
    
    # Training parameters
    parser.add_argument(
        "--fold_idx", type=int, default=0,
        help="Fold index for training (default: 0)"
    )
    parser.add_argument(
        "--num_workers", type=int, default=8,
        help="Number of data loading workers (default: 8)"
    )
    parser.add_argument(
        "--no_deterministic", action="store_true",
        help="Disable deterministic mode"
    )
    
    # Filtering
    parser.add_argument(
        "--ops", type=str, default=None,
        help="Comma-separated list of ops to tune (default: all promoted)"
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Run minimal configs for testing"
    )
    
    return parser.parse_args()


# =============================================================================
# Main Entry Point
# =============================================================================

def main() -> int:
    args = parse_args()
    
    # Parse rungs
    rungs = [int(r.strip()) for r in args.rungs.split(",")]
    
    # Parse ops filter
    ops_filter = None
    if args.ops:
        ops_filter = [op.strip() for op in args.ops.split(",")]
    
    deterministic = not args.no_deterministic
    device = get_device()
    
    print("=" * 70)
    print("Phase B: ASHA Augmentation Tuning (v5.3)")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"ASHA Rungs: {rungs}")
    print(f"Reduction factor: 1/{args.reduction_factor}")
    print(f"Sobol samples per op: {args.n_samples}")
    print(f"Seed: {args.seed}")
    print(f"Batch size: 512")
    print(f"Fold: {args.fold_idx}")
    print(f"Deterministic: {deterministic}")
    print("-" * 70)
    print(f"Phase A CSV: {args.phase_a_csv}")
    print(f"Baseline CSV: {args.baseline_csv}")
    if ops_filter:
        print(f"Ops filter: {ops_filter}")
    if args.dry_run:
        print("MODE: DRY RUN")
    print("=" * 70)
    
    try:
        run_phase_b_asha(
            phase_a_csv=Path(args.phase_a_csv),
            baseline_csv=Path(args.baseline_csv),
            output_dir=Path(args.output_dir),
            rungs=rungs,
            reduction_factor=args.reduction_factor,
            n_samples=args.n_samples,
            seed=args.seed,
            fold_idx=args.fold_idx,
            num_workers=args.num_workers,
            deterministic=deterministic,
            ops_filter=ops_filter,
            dry_run=args.dry_run,
        )
        return 0
        
    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
