# Phase C: Prior-Guided Greedy Ensemble
"""
Phase C: Prior-Guided Greedy Ensemble Script.

v5.4 CHANGED: Multi-start greedy search using both Phase A and Phase B top configs.

Constructs the final augmentation policy by greedily adding operations
from Phase B results, using 3-seed validation for robustness.

Reference: docs/research_plan_v5.md Section 3 (Phase C)

Algorithm (v5.4 Multi-Start):
1. Load Phase A results (best per op) and Phase B summary
2. Create K starting points from top performers of Phase A + Phase B
3. For each starting point:
    a. Initialize P with the starting op
    b. Greedy add remaining ops if improvement > threshold
4. Select best final policy among all paths
5. Output final policy P_final

Usage:
    # Full run (200 epochs, 3 seeds)
    python main_phase_c.py
    
    # Dry run for testing
    python main_phase_c.py --epochs 2 --dry_run
    
"""

import argparse
import csv
import json
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
    build_transform_with_ops,
    get_baseline_transform,
    get_val_transform,
    get_compatible_ops,
    check_mutual_exclusion,
    adjust_probabilities_for_combination,
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
    load_phase0_best_config,
)


# =============================================================================
# Phase A Results Loading (v5.4 NEW)
# =============================================================================

def load_phase_a_best_per_op(csv_path: Path) -> Dict[str, Tuple[float, float, float]]:
    """Load Phase A results and get best config per operation.
    
    Args:
        csv_path: Path to phase_a_results.csv
        
    Returns:
        Dict mapping op_name to (magnitude, probability, val_acc)
    """
    if not csv_path.exists():
        print(f"WARNING: Phase A results not found: {csv_path}")
        return {}
    
    df = pd.read_csv(csv_path)
    
    if df.empty:
        return {}
    
    # Filter out errors
    df = df[df["error"].isna() | (df["error"] == "")]
    
    if df.empty:
        return {}
    
    best_configs = {}
    for op_name in df["op_name"].unique():
        op_data = df[df["op_name"] == op_name]
        best_row = op_data.loc[op_data["val_acc"].idxmax()]
        best_configs[op_name] = (
            float(best_row["magnitude"]),
            float(best_row["probability"]),
            float(best_row["val_acc"]),
        )
    
    return best_configs


# =============================================================================
# Phase B Results Loading
# =============================================================================

def load_phase_b_summary(csv_path: Path) -> pd.DataFrame:
    """Load Phase B summary results, sorted by mean_val_acc descending.
    
    Args:
        csv_path: Path to phase_b_tuning_summary.csv
        
    Returns:
        DataFrame with columns: op_name, magnitude, probability, mean_val_acc, std_val_acc, ...
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Phase B summary not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    if df.empty:
        raise ValueError(f"Phase B summary is empty: {csv_path}")
    
    required_cols = ["op_name", "magnitude", "probability", "mean_val_acc"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    # Sort by mean_val_acc descending
    df = df.sort_values("mean_val_acc", ascending=False).reset_index(drop=True)
    
    return df


def get_best_config_per_op(phase_b_df: pd.DataFrame) -> Dict[str, Tuple[float, float, float]]:
    """Get best (magnitude, probability, mean_val_acc) for each operation.
    
    Args:
        phase_b_df: DataFrame from load_phase_b_summary
        
    Returns:
        Dict mapping op_name to (magnitude, probability, mean_val_acc)
    """
    best_configs = {}
    
    for op_name in phase_b_df["op_name"].unique():
        op_data = phase_b_df[phase_b_df["op_name"] == op_name]
        best_row = op_data.loc[op_data["mean_val_acc"].idxmax()]
        best_configs[op_name] = (
            float(best_row["magnitude"]),
            float(best_row["probability"]),
            float(best_row["mean_val_acc"]),
        )
    
    return best_configs


# =============================================================================
# Training Functions
# =============================================================================

def train_single_config(
    ops: List[Tuple[str, float, float]],
    seed: int,
    epochs: int,
    device: torch.device,
    fold_idx: int = 0,
    batch_size: int = 128,
    num_workers: int = 8,
    early_stop_patience: int = 99999,
    deterministic: bool = True,
    save_checkpoint: bool = False,
    checkpoint_dir: Optional[Path] = None,
    weight_decay: float = 1e-2,
    label_smoothing: float = 0.1,
) -> Dict:
    """Train one configuration (policy) and return metrics.
    
    v5.1: Added checkpoint saving for Phase C.
    
    Args:
        ops: List of (op_name, magnitude, probability) tuples defining the policy.
        seed: Random seed for this run.
        epochs: Number of training epochs.
        device: Device to train on.
        fold_idx: Which fold to use (default 0).
        batch_size: Batch size.
        num_workers: Number of data loading workers.
        early_stop_patience: Early stopping patience.
        deterministic: If True, enable deterministic CUDA mode.
        save_checkpoint: If True, save best model checkpoint.
        checkpoint_dir: Directory for checkpoints.
        
    Returns:
        Dict with training results.
    """
    start_time = time.time()
    set_seed_deterministic(seed, deterministic=deterministic)
    
    # Build ops string for logging
    if not ops:
        ops_str = "Baseline"
        mag_str = "0.0"
        prob_str = "1.0"
    else:
        ops_str = "+".join([op[0] for op in ops])
        mag_str = "+".join([str(round(op[1], 4)) for op in ops])
        prob_str = "+".join([str(round(op[2], 4)) for op in ops])
    
    result = {
        "phase": "PhaseC",
        "op_name": ops_str,
        "magnitude": mag_str,
        "probability": prob_str,
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
    
    try:
        # Build transforms
        if not ops:
            train_transform = get_baseline_transform(include_normalize=False)
        else:
            train_transform = build_transform_with_ops(
                ops=ops,
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
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        
        # Optimizer and scheduler
        optimizer, scheduler = get_optimizer_and_scheduler(
            model=model,
            total_epochs=epochs,
            lr=0.1,
            weight_decay=weight_decay,
            momentum=0.9,
            warmup_epochs=5,
        )
        
        # AMP scaler
        scaler = None
        if device.type == "cuda":
            scaler = torch.amp.GradScaler()
        
        # Early stopping (v5.1: disabled for Phase C to ensure full training)
        # Phase C uses same early stopping as Phase A/B for fair comparison
        early_stopper = EarlyStopping(
            patience=early_stop_patience,  # Default effectively disables early stop
            mode="max",  # Monitor val_acc (higher is better)
            min_epochs=60,  # At least 60 epochs before considering stop
            min_delta=0.2,
        )
        
        # Training loop
        best_val_acc = 0.0
        best_val_loss = float("inf")
        best_top5_acc = 0.0
        best_train_acc = 0.0
        best_train_loss = 0.0
        best_epoch = 0
        early_stopped = False
        best_model_state = None  # For checkpoint saving
        
        for epoch in range(epochs):
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
                # Save best model state for checkpoint
                if save_checkpoint:
                    best_model_state = {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                    }
            
            scheduler.step()
            
            if early_stopper(val_acc, epoch):
                result["epochs_run"] = epoch + 1
                early_stopped = True
                break
            
            result["epochs_run"] = epoch + 1
        
        # Save checkpoint if requested
        if save_checkpoint and best_model_state is not None and checkpoint_dir is not None:
            ops_str = "+".join([op[0] for op in ops]) if ops else "baseline"
            checkpoint_path = checkpoint_dir / f"phase_c_{ops_str}_seed{seed}_best.pth"
            checkpoint = {
                **best_model_state,
                "epoch": best_epoch,
                "val_acc": best_val_acc,
                "top5_acc": best_top5_acc,
                "val_loss": best_val_loss,
                "config": {
                    "ops": [(op[0], op[1], op[2]) for op in ops],
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
            print(f"Checkpoint saved: {checkpoint_path}")
        
        result["val_acc"] = round(best_val_acc, 4)
        result["val_loss"] = round(best_val_loss, 6)
        result["top5_acc"] = round(best_top5_acc, 4)
        result["train_acc"] = round(best_train_acc, 4)
        result["train_loss"] = round(best_train_loss, 6)
        result["best_epoch"] = best_epoch
        result["early_stopped"] = early_stopped
        
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {str(e)}"
        traceback.print_exc()
    
    result["runtime_sec"] = round(time.time() - start_time, 2)
    result["timestamp"] = datetime.now().isoformat(timespec='seconds')
    
    return result


def train_with_multi_seed(
    ops: List[Tuple[str, float, float]],
    seeds: List[int],
    epochs: int,
    device: torch.device,
    fold_idx: int = 0,
    weight_decay: float = 1e-2,
    label_smoothing: float = 0.1,
    **kwargs,
) -> Tuple[float, float, List[Dict]]:
    """Train configuration with multiple seeds and return mean/std accuracy.
    
    Args:
        ops: List of (op_name, magnitude, probability) tuples.
        seeds: List of random seeds.
        epochs: Number of training epochs.
        device: Device to train on.
        fold_idx: Which fold to use.
        **kwargs: Additional arguments passed to train_single_config.
        
    Returns:
        Tuple of (mean_val_acc, std_val_acc, list_of_results)
    """
    results = []
    val_accs = []
    
    for seed in seeds:
        result = train_single_config(
            ops=ops,
            seed=seed,
            epochs=epochs,
            device=device,
            fold_idx=fold_idx,
            weight_decay=weight_decay,
            label_smoothing=label_smoothing,
            **kwargs,
        )
        results.append(result)
        
        if not result["error"]:
            val_accs.append(result["val_acc"])
    
    if val_accs:
        mean_acc = np.mean(val_accs)
        std_acc = np.std(val_accs) if len(val_accs) > 1 else 0.0
    else:
        mean_acc = -1.0
        std_acc = 0.0
    
    return mean_acc, std_acc, results


# =============================================================================
# CSV Writing
# =============================================================================

def write_csv_row(path: Path, row: Dict, write_header: bool) -> None:
    """Append one row to CSV with immediate flush."""
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
# Policy Serialization
# =============================================================================

def save_policy(
    policy: List[Tuple[str, float, float]],
    baseline_acc: float,
    final_acc: float,
    output_path: Path,
    p_any_target: float = 0.5,
) -> None:
    """Save final policy to JSON file.
    
    v5.4: Now includes adjusted probabilities for combination.
    
    Args:
        policy: List of (op_name, magnitude, probability) tuples.
        baseline_acc: Baseline accuracy for reference.
        final_acc: Final policy accuracy.
        output_path: Path to save JSON file.
        p_any_target: Target probability for combination adjustment.
    """
    # Compute adjusted probabilities for combination
    if len(policy) > 1:
        adjusted_policy = adjust_probabilities_for_combination(policy, p_any_target)
    else:
        adjusted_policy = policy
    
    policy_dict = {
        "version": "v5.4",
        "phase": "PhaseC",
        "baseline_acc": baseline_acc,
        "final_acc": final_acc,
        "improvement": round(final_acc - baseline_acc, 4),
        "n_ops": len(policy),
        "p_any_target": p_any_target,
        "ops": [
            {
                "name": policy[i][0],
                "magnitude": round(policy[i][1], 4),
                "probability_original": round(policy[i][2], 4),
                "probability_adjusted": round(adjusted_policy[i][2], 4),
            }
            for i in range(len(policy))
        ],
        "timestamp": datetime.now().isoformat(timespec='seconds'),
    }
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(policy_dict, f, indent=2)
    
    print(f"Policy saved to: {output_path}")


def load_policy(json_path: Path) -> List[Tuple[str, float, float]]:
    """Load policy from JSON file.
    
    Args:
        json_path: Path to policy JSON file.
        
    Returns:
        List of (op_name, magnitude, probability) tuples.
    """
    with open(json_path, "r") as f:
        policy_dict = json.load(f)
    
    policy = []
    for op in policy_dict["ops"]:
        policy.append((op["name"], op["magnitude"], op["probability"]))
    
    return policy


# =============================================================================
# Single-Path Greedy Search Helper
# =============================================================================

def _greedy_search_from_start(
    starting_op: Optional[Tuple[str, float, float]],
    candidates: Dict[str, Tuple[float, float, float]],
    baseline_acc: float,
    epochs: int,
    seeds: List[int],
    device: torch.device,
    fold_idx: int,
    max_ops: int,
    improvement_threshold: float,
    num_workers: int,
    early_stop_patience: int,
    deterministic: bool,
    dry_run: bool,
    save_checkpoints: bool,
    checkpoint_dir: Path,
    history_csv_path: Path,
    write_header_ref: List[bool],
    path_name: str = "Path",
    weight_decay: float = 1e-2,
    label_smoothing: float = 0.1,
) -> Tuple[List[Tuple[str, float, float]], float]:
    """Run greedy search from a single starting point.
    
    Args:
        starting_op: Starting operation tuple (name, mag, prob) or None for baseline start.
        candidates: Dict of op_name -> (magnitude, probability, acc) for all candidates.
        baseline_acc: Baseline accuracy.
        epochs: Training epochs.
        seeds: Random seeds.
        device: Device to train on.
        fold_idx: Fold index.
        max_ops: Maximum operations.
        improvement_threshold: Minimum improvement threshold.
        num_workers: Data loader workers.
        early_stop_patience: Early stopping patience.
        deterministic: Deterministic mode.
        dry_run: Dry run mode.
        save_checkpoints: Save checkpoints.
        checkpoint_dir: Checkpoint directory.
        history_csv_path: Path to history CSV.
        write_header_ref: Mutable reference for write_header flag.
        path_name: Name for logging.
        
    Returns:
        Tuple of (final_policy, final_accuracy).
    """
    print(f"\n{'#'*70}")
    print(f"# {path_name}")
    print(f"{'#'*70}")
    
    # Initialize policy
    if starting_op is not None:
        current_policy = [starting_op]
        # Evaluate starting op with multi-seed
        if dry_run:
            train_epochs = min(2, epochs)
            train_seeds = seeds[:1]
        else:
            train_epochs = epochs
            train_seeds = seeds
        
        print(f"Evaluating starting op: {starting_op[0]}")
        mean_acc, std_acc, results = train_with_multi_seed(
            ops=current_policy,
            seeds=train_seeds,
            epochs=train_epochs,
            device=device,
            fold_idx=fold_idx,
            num_workers=num_workers,
            early_stop_patience=early_stop_patience,
            deterministic=deterministic,
            save_checkpoint=save_checkpoints and not dry_run,
            checkpoint_dir=checkpoint_dir,
            weight_decay=weight_decay,
            label_smoothing=label_smoothing,
        )
        
        # Write results
        for result in results:
            write_csv_row(history_csv_path, result, write_header_ref[0])
            write_header_ref[0] = False
        
        current_acc = mean_acc
        print(f"Starting point: {starting_op[0]} = {current_acc:.2f}%")
    else:
        current_policy = []
        current_acc = baseline_acc
        print(f"Starting from baseline: {current_acc:.2f}%")
    
    # Sort remaining candidates
    remaining_candidates = sorted(
        [(name, cfg) for name, cfg in candidates.items() 
         if starting_op is None or name != starting_op[0]],
        key=lambda x: -x[1][2]
    )
    
    # Greedy loop
    for op_name, (magnitude, probability, phase_b_acc) in remaining_candidates:
        if len(current_policy) >= max_ops:
            print(f"\nReached max_ops limit ({max_ops}). Stopping.")
            break
        
        # Check mutual exclusion
        current_op_names = [op[0] for op in current_policy]
        if not get_compatible_ops(current_op_names, op_name):
            print(f"\nSkipping {op_name}: conflicts with {current_op_names}")
            continue
        
        # Propose new policy
        proposed_policy = current_policy + [(op_name, magnitude, probability)]
        
        print(f"\n{'='*70}")
        print(f"[{path_name}] Trying to add: {op_name} (m={magnitude:.4f}, p={probability:.4f})")
        print(f"Current policy: {current_op_names if current_op_names else 'S0'}")
        print("-" * 70)
        
        # Train with multi-seed
        if dry_run:
            train_epochs = min(2, epochs)
            train_seeds = seeds[:1]
        else:
            train_epochs = epochs
            train_seeds = seeds
        
        mean_acc, std_acc, results = train_with_multi_seed(
            ops=proposed_policy,
            seeds=train_seeds,
            epochs=train_epochs,
            device=device,
            fold_idx=fold_idx,
            num_workers=num_workers,
            early_stop_patience=early_stop_patience,
            deterministic=deterministic,
            save_checkpoint=save_checkpoints and not dry_run,
            checkpoint_dir=checkpoint_dir,
            weight_decay=weight_decay,
            label_smoothing=label_smoothing,
        )
        
        # Write results
        for result in results:
            write_csv_row(history_csv_path, result, write_header_ref[0])
            write_header_ref[0] = False
        
        # Decision (v5.5: stricter acceptance with majority rule)
        improvement = mean_acc - current_acc
        
        # v5.5: Count how many seeds show improvement
        individual_accs = [r["val_acc"] for r in results if r.get("val_acc", -1) > 0]
        n_improved = sum(1 for acc in individual_accs if acc > current_acc)
        n_total = len(individual_accs)
        majority_improved = n_improved >= (n_total + 1) // 2  # At least half (2/3 for 3 seeds)
        
        print(f"Results: mean_acc={mean_acc:.2f}% ± {std_acc:.2f}%")
        print(f"Improvement: {improvement:+.2f}% (threshold: +{improvement_threshold}%)")
        print(f"Seeds improved: {n_improved}/{n_total} (majority rule: {'✓' if majority_improved else '✗'})")
        
        # v5.5: Accept only if BOTH conditions are met:
        # 1. Mean improvement >= threshold
        # 2. Majority of seeds show improvement
        if improvement >= improvement_threshold and majority_improved:
            print(f"✓ ACCEPTED: {op_name}")
            current_policy = proposed_policy
            current_acc = mean_acc
        else:
            if improvement >= improvement_threshold and not majority_improved:
                print(f"✗ REJECTED: {op_name} (mean OK but minority of seeds improved)")
            else:
                print(f"✗ REJECTED: {op_name}")
    
    print(f"\n[{path_name}] Final: {[op[0] for op in current_policy]} = {current_acc:.2f}%")
    return current_policy, current_acc


# =============================================================================
# Main Greedy Algorithm (v5.4 Multi-Start)
# =============================================================================

def run_phase_c(
    phase_b_csv: Path,
    baseline_acc: float,
    output_dir: Path,
    epochs: int = 200,
    seeds: List[int] = [42, 123, 456],
    fold_idx: int = 0,
    max_ops: int = 3,
    improvement_threshold: float = 0.2,  # v5.5: increased from 0.1
    num_workers: int = 8,
    early_stop_patience: int = 99999,
    deterministic: bool = True,
    dry_run: bool = False,
    save_checkpoints: bool = True,
    phase_a_csv: Optional[Path] = None,
    n_start_points: int = 3,
    p_any_target: float = 0.5,
    weight_decay: float = 1e-2,
    label_smoothing: float = 0.1,
) -> List[Tuple[str, float, float]]:
    """Run Phase C greedy ensemble algorithm with multi-start search.
    
    v5.4: Multi-start greedy search using Phase A and Phase B top configs.
    v5.4: Added probability adjustment for combination (p_any_target).
    v5.1: Added checkpoint saving for accepted policies.
    
    Args:
        phase_b_csv: Path to Phase B summary CSV.
        baseline_acc: Baseline accuracy at 200 epochs.
        output_dir: Directory for output files.
        epochs: Training epochs per configuration.
        seeds: List of random seeds for multi-seed validation.
        fold_idx: Which fold to use.
        max_ops: Maximum number of operations to add.
        improvement_threshold: Minimum improvement to accept an op (%).
        num_workers: Data loading workers.
        early_stop_patience: Early stopping patience.
        deterministic: Enable deterministic CUDA mode.
        dry_run: If True, run minimal epochs for testing.
        save_checkpoints: If True, save checkpoints for accepted policies.
        phase_a_csv: Path to Phase A results CSV for multi-start search.
        n_start_points: Number of starting points to try (default 3).
        p_any_target: Target probability for at least one augmentation (default 0.5).
        
    Returns:
        Final policy as list of (op_name, magnitude, probability) tuples.
    """
    device = get_device()
    
    # Load Phase 0 hyperparameters if available
    phase0_cfg = load_phase0_best_config()
    wd = phase0_cfg[0] if phase0_cfg else 1e-2
    ls = phase0_cfg[1] if phase0_cfg else 0.1
    if phase0_cfg:
        print(f"[Phase0] Using recommended hyperparams: wd={wd}, ls={ls}")
    else:
        print("[Phase0] phase0_summary.csv not found; fallback to defaults wd=1e-2, ls=0.1")
    ensure_dir(output_dir)
    
    # Create checkpoint directory
    checkpoint_dir = output_dir / "checkpoints"
    ensure_dir(checkpoint_dir)
    
    history_csv_path = output_dir / "phase_c_history.csv"
    policy_json_path = output_dir / "phase_c_final_policy.json"
    
    # Load Phase B results
    print("Loading Phase B results...")
    phase_b_df = load_phase_b_summary(phase_b_csv)
    phase_b_best = get_best_config_per_op(phase_b_df)
    
    print(f"Found {len(phase_b_best)} operations from Phase B:")
    for op_name, (mag, prob, acc) in sorted(phase_b_best.items(), key=lambda x: -x[1][2]):
        print(f"  {op_name}: m={mag:.4f}, p={prob:.4f}, acc={acc:.2f}%")
    
    # Load Phase A results (v5.4)
    phase_a_best = {}
    if phase_a_csv is not None and phase_a_csv.exists():
        print("\nLoading Phase A results...")
        phase_a_best = load_phase_a_best_per_op(phase_a_csv)
        print(f"Found {len(phase_a_best)} operations from Phase A:")
        for op_name, (mag, prob, acc) in sorted(phase_a_best.items(), key=lambda x: -x[1][2])[:5]:
            print(f"  {op_name}: m={mag:.4f}, p={prob:.4f}, acc={acc:.2f}%")
    
    # Create unified candidate pool (prefer Phase B configs but include Phase A top performers)
    all_candidates = dict(phase_b_best)  # Start with Phase B
    for op_name, (mag, prob, acc) in phase_a_best.items():
        if op_name not in all_candidates or acc > all_candidates[op_name][2]:
            all_candidates[op_name] = (mag, prob, acc)
    
    print(f"\nUnified candidate pool: {len(all_candidates)} operations")
    
    # Determine starting points (v5.4 Multi-Start)
    # Combine Phase A and Phase B top performers
    phase_a_sorted = sorted(phase_a_best.items(), key=lambda x: -x[1][2])
    phase_b_sorted = sorted(phase_b_best.items(), key=lambda x: -x[1][2])
    
    start_points: List[Optional[Tuple[str, float, float]]] = []
    used_ops = set()
    
    # Interleave Phase A and Phase B top performers
    a_idx, b_idx = 0, 0
    while len(start_points) < n_start_points and (a_idx < len(phase_a_sorted) or b_idx < len(phase_b_sorted)):
        # Add from Phase A
        while a_idx < len(phase_a_sorted) and phase_a_sorted[a_idx][0] in used_ops:
            a_idx += 1
        if a_idx < len(phase_a_sorted) and len(start_points) < n_start_points:
            op_name, (mag, prob, acc) = phase_a_sorted[a_idx]
            start_points.append((op_name, mag, prob))
            used_ops.add(op_name)
            a_idx += 1
        
        # Add from Phase B
        while b_idx < len(phase_b_sorted) and phase_b_sorted[b_idx][0] in used_ops:
            b_idx += 1
        if b_idx < len(phase_b_sorted) and len(start_points) < n_start_points:
            op_name, (mag, prob, acc) = phase_b_sorted[b_idx]
            start_points.append((op_name, mag, prob))
            used_ops.add(op_name)
            b_idx += 1
    
    print(f"\nMulti-start search with {len(start_points)} starting points:")
    for i, sp in enumerate(start_points):
        if sp is not None:
            print(f"  Path {i+1}: Start with {sp[0]} (m={sp[1]:.4f}, p={sp[2]:.4f})")
    
    print(f"\nBaseline accuracy: {baseline_acc:.2f}%")
    print(f"Improvement threshold: +{improvement_threshold}%")
    print(f"Max operations: {max_ops}")
    print(f"Seeds: {seeds}")
    print("-" * 70)
    
    # Check if CSV needs header
    write_header_ref = [check_csv_needs_header(history_csv_path)]
    
    # Run greedy search from each starting point
    all_paths: List[Tuple[List[Tuple[str, float, float]], float, str]] = []
    
    for i, start_op in enumerate(start_points):
        path_name = f"Path {i+1} ({start_op[0] if start_op else 'Baseline'})"
        policy, acc = _greedy_search_from_start(
            starting_op=start_op,
            candidates=all_candidates,
            baseline_acc=baseline_acc,
            epochs=epochs,
            seeds=seeds,
            device=device,
            fold_idx=fold_idx,
            max_ops=max_ops,
            improvement_threshold=improvement_threshold,
            num_workers=num_workers,
            early_stop_patience=early_stop_patience,
            deterministic=deterministic,
            dry_run=dry_run,
            save_checkpoints=save_checkpoints,
            checkpoint_dir=checkpoint_dir,
            history_csv_path=history_csv_path,
            write_header_ref=write_header_ref,
            path_name=path_name,
            weight_decay=weight_decay,
            label_smoothing=label_smoothing,
        )
        all_paths.append((policy, acc, path_name))
    
    # Select best path
    best_policy, best_acc, best_path_name = max(all_paths, key=lambda x: x[1])
    
    # Summary
    print("\n" + "=" * 70)
    print("Phase C Complete (Multi-Start Search)")
    print("=" * 70)
    print("\nAll paths:")
    for policy, acc, name in all_paths:
        marker = " ★ BEST" if acc == best_acc else ""
        print(f"  {name}: {[op[0] for op in policy] if policy else 'S0'} = {acc:.2f}%{marker}")
    
    print(f"\nSelected: {best_path_name}")
    print(f"Final policy: {[op[0] for op in best_policy] if best_policy else 'S0'}")
    print(f"Final accuracy: {best_acc:.2f}%")
    print(f"Improvement over baseline: {best_acc - baseline_acc:+.2f}%")
    print("-" * 70)
    print(f"History saved to: {history_csv_path}")
    
    # Save final policy (includes original and adjusted probabilities)
    save_policy(
        policy=best_policy,
        baseline_acc=baseline_acc,
        final_acc=best_acc,
        output_path=policy_json_path,
        p_any_target=p_any_target,
    )
    
    return best_policy


def run_baseline_for_phase_c(
    output_dir: Path,
    epochs: int = 200,
    seed: int = 42,
    fold_idx: int = 0,
    num_workers: int = 8,
    early_stop_patience: int = 60,
    deterministic: bool = True,
    weight_decay: float = 1e-2,
    label_smoothing: float = 0.1,
) -> float:
    """Run baseline for Phase C comparison (same epochs as A/B/D).
    
    Args:
        output_dir: Directory for output files.
        epochs: Number of epochs (default 200).
        seed: Random seed.
        fold_idx: Which fold to use.
        num_workers: Data loading workers.
        early_stop_patience: Early stopping patience.
        deterministic: Enable deterministic CUDA mode.
        
    Returns:
        Best validation accuracy.
    """
    device = get_device()
    
    print("=" * 70)
    print(f"Running {epochs}-epoch Baseline for Phase C/D")
    print("=" * 70)
    
    result = train_single_config(
        ops=[],  # Empty = baseline
        seed=seed,
        epochs=epochs,
        device=device,
        fold_idx=fold_idx,
        num_workers=num_workers,
        early_stop_patience=early_stop_patience,
        deterministic=deterministic,
        weight_decay=weight_decay,
        label_smoothing=label_smoothing,
    )
    
    if result["error"]:
        raise RuntimeError(f"Baseline training failed: {result['error']}")
    
    baseline_acc = result["val_acc"]
    
    # Save result
    csv_path = output_dir / "baseline_result.csv"
    write_csv_row(csv_path, result, write_header=True)
    
    print(f"\nBaseline {epochs}-epoch result: {baseline_acc:.2f}%")
    print(f"Saved to: {csv_path}")
    
    return baseline_acc


# =============================================================================
# Argument Parsing
# =============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Phase C: Prior-Guided Greedy Ensemble"
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
        help="Which fold to use (default: 0)"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Output directory for history/policy/checkpoints (default: outputs)"
    )
    
    parser.add_argument(
        "--phase_b_csv",
        type=str,
        default="outputs/phase_b_tuning_summary.csv",
        help="Path to Phase B summary CSV (required input)"
    )
    
    parser.add_argument(
        "--baseline_acc",
        type=float,
        default=None,
        help="Baseline accuracy for comparison (if not provided, will load outputs/baseline_result.csv or run baseline)"
    )
    
    parser.add_argument(
        "--max_ops",
        type=int,
        default=3,
        help="Maximum number of operations to add (default: 3)"
    )
    
    parser.add_argument(
        "--improvement_threshold",
        type=float,
        default=0.2,
        help="Minimum improvement threshold in %% (default: 0.2, v5.5)"
    )
    
    parser.add_argument(
        "--p_any_target",
        type=float,
        default=0.5,
        help="Target probability for at least one augmentation (default: 0.5)"
    )
    
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of data loading workers (default: 8)"
    )
    
    parser.add_argument(
        "--early_stop_patience",
        type=int,
        default=60,
        help="Early stopping patience (default: 60 for Phase C)"
    )
    
    parser.add_argument(
        "--no_deterministic",
        action="store_true",
        help="Disable deterministic CUDA mode"
    )
    
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Run minimal epochs for testing"
    )
    
    parser.add_argument(
        "--phase_a_csv",
        type=str,
        default="outputs/phase_a_results.csv",
        help="Path to Phase A results CSV for multi-start search (optional, improves diversity)"
    )
    
    parser.add_argument(
        "--n_start_points",
        type=int,
        default=3,
        help="Number of starting points for multi-start greedy search (default: 3)"
    )
    
    return parser.parse_args()


# =============================================================================
# Main Entry Point
# =============================================================================

def main() -> int:
    """Main entry point."""
    args = parse_args()
    
    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    deterministic = not args.no_deterministic
    output_dir = Path(args.output_dir)
    device = get_device()
    
    print("=" * 70)
    print("Phase C: Prior-Guided Greedy Ensemble (v5.4 Multi-Start)")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Epochs: {args.epochs}")
    print(f"Seeds: {seeds}")
    print(f"Fold: {args.fold_idx}")
    print(f"Max ops: {args.max_ops}")
    print(f"Improvement threshold: {args.improvement_threshold}%")
    print(f"Start points: {args.n_start_points}")
    print(f"Phase A CSV: {args.phase_a_csv}")
    print(f"Deterministic: {deterministic}")
    print(f"LR: 0.1, WD: {wd}, Momentum: 0.9, Warmup: 5 epochs, Label Smoothing: {ls}")
    if args.dry_run:
        print("MODE: DRY RUN")
    print("=" * 70)
    
    # Get baseline accuracy
    baseline_acc = args.baseline_acc
    if baseline_acc is None:
        # Try to load from baseline result (200ep, same as Phase A/B)
        baseline_path = output_dir / "baseline_result.csv"
        if baseline_path.exists():
            df = pd.read_csv(baseline_path)
            baseline_acc = df["val_acc"].iloc[0]
            print(f"Loaded baseline accuracy from {baseline_path}: {baseline_acc:.2f}%")
        else:
            # Fallback: run baseline first
            print("No baseline found. Running baseline first...")
            baseline_acc = run_baseline_for_phase_c(
                output_dir=output_dir,
                epochs=args.epochs if not args.dry_run else 2,
                seed=seeds[0],
                fold_idx=args.fold_idx,
                num_workers=args.num_workers,
                early_stop_patience=args.early_stop_patience,
                deterministic=deterministic,
                weight_decay=wd,
                label_smoothing=ls,
            )
    
    print(f"Using baseline accuracy: {baseline_acc:.2f}%")
    
    try:
        run_phase_c(
            phase_b_csv=Path(args.phase_b_csv),
            baseline_acc=baseline_acc,
            output_dir=output_dir,
            epochs=args.epochs,
            seeds=seeds,
            fold_idx=args.fold_idx,
            max_ops=args.max_ops,
            improvement_threshold=args.improvement_threshold,
            num_workers=args.num_workers,
            early_stop_patience=args.early_stop_patience,
            deterministic=deterministic,
            dry_run=args.dry_run,
            phase_a_csv=Path(args.phase_a_csv),
            n_start_points=args.n_start_points,
            p_any_target=args.p_any_target,
            weight_decay=wd,
            label_smoothing=ls,
        )
        return 0
        
    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
