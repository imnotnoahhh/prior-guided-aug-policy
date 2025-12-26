# Phase D: SOTA Benchmark Comparison (5-Fold Validation)
"""
Phase D: SOTA Benchmark Comparison Script.

Runs 5-fold cross-validation comparing our method against SOTA baselines.

Reference: docs/research_plan_v5.md Section 3 (Phase D)

Methods compared:
1. Baseline: S0 only (RandomCrop + HorizontalFlip)
2. RandAugment: N=2, M=9 (standard settings)
3. Cutout: n_holes=1, length=16
4. Ours_p1: Phase C policy with all probabilities set to 1.0 (ablation)
5. Ours_optimal: Phase C final policy

Each method runs on 5 folds × 200 epochs × 1 seed per fold.
Final results are Mean ± Std across folds.

Usage:
    # Full run (200 epochs, 5 folds)
    python main_phase_d.py
    
    # Run specific methods only
    python main_phase_d.py --methods Baseline,RandAugment
    
    # Run specific folds only
    python main_phase_d.py --folds 0,1,2
    
    # Dry run for testing
    python main_phase_d.py --epochs 2 --dry_run
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
from typing import Callable, Dict, List, Optional, Tuple

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
    build_transform_with_op,  # v5.5: for Best_SingleOp
    build_ours_p1_transform,
    get_baseline_transform,
    get_randaugment_transform,
    get_cutout_transform,
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
    load_phase0_best_config,
)


# =============================================================================
# Method Definitions
# =============================================================================

# Available methods for comparison
# v5.4: Added "Baseline-NoAug" for ablation (no augmentation at all)
# v5.5: Added "Best_SingleOp" for ablation (best single operation from Phase B)
AVAILABLE_METHODS = ["Baseline", "Baseline-NoAug", "RandAugment", "Cutout", "Best_SingleOp", "Ours_p1", "Ours_optimal"]

# v5.5: Global variable to store best single op config (loaded from Phase B)
BEST_SINGLE_OP_CONFIG = None  # Will be set to (op_name, magnitude, probability)


def get_method_transform(
    method_name: str,
    policy: Optional[List[Tuple[str, float, float]]] = None,
) -> Callable:
    """Get transform for a given method.
    
    v5.4: Ours_optimal now uses probability_adjusted from policy JSON.
    
    Args:
        method_name: One of AVAILABLE_METHODS.
        policy: Phase C policy (required for Ours_p1 and Ours_optimal).
        
    Returns:
        Transform callable.
        
    Raises:
        ValueError: If method_name is unknown or policy is missing.
    """
    if method_name == "Baseline":
        return get_baseline_transform(include_normalize=False)
    
    elif method_name == "Baseline-NoAug":
        # v5.4: No augmentation at all - just ToTensor without any augmentation
        # Note: still need to resize to 32x32 for CIFAR
        from torchvision import transforms
        return transforms.Compose([
            transforms.ToTensor(),
        ])
    
    elif method_name == "RandAugment":
        return get_randaugment_transform(n=2, m=9, include_baseline=True, include_normalize=False)
    
    elif method_name == "Cutout":
        return get_cutout_transform(n_holes=1, length=16, include_baseline=True, include_normalize=False)
    
    elif method_name == "Best_SingleOp":
        # v5.5: Use the best single operation from Phase B
        global BEST_SINGLE_OP_CONFIG
        if BEST_SINGLE_OP_CONFIG is None:
            # Fallback to baseline if not configured
            return get_baseline_transform(include_normalize=False)
        op_name, m, p = BEST_SINGLE_OP_CONFIG
        return build_transform_with_op(op_name, m, p, include_baseline=True, include_normalize=False)
    
    elif method_name == "Ours_p1":
        if policy is None or len(policy) == 0:
            # No policy = just baseline
            return get_baseline_transform(include_normalize=False)
        return build_ours_p1_transform(policy, include_baseline=True, include_normalize=False)
    
    elif method_name == "Ours_optimal":
        if policy is None or len(policy) == 0:
            # No policy = just baseline
            return get_baseline_transform(include_normalize=False)
        return build_transform_with_ops(policy, include_baseline=True, include_normalize=False)
    
    else:
        raise ValueError(f"Unknown method: {method_name}. Available: {AVAILABLE_METHODS}")


def get_method_description(method_name: str) -> str:
    """Get human-readable description of a method."""
    global BEST_SINGLE_OP_CONFIG
    descriptions = {
        "Baseline": "S0 (RandomCrop + HorizontalFlip)",
        "Baseline-NoAug": "No augmentation (ablation)",
        "RandAugment": "RandAugment (N=2, M=9)",
        "Cutout": "Cutout (length=16)",
        "Best_SingleOp": f"Best single op from Phase B",
        "Ours_p1": "Ours (p=1.0 ablation)",
        "Ours_optimal": "Ours (optimal p, adjusted)",
    }
    # v5.5: Add specific op info for Best_SingleOp
    if method_name == "Best_SingleOp" and BEST_SINGLE_OP_CONFIG:
        op_name, m, p = BEST_SINGLE_OP_CONFIG
        return f"Best single: {op_name}(m={m:.2f}, p={p:.2f})"
    return descriptions.get(method_name, method_name)


# =============================================================================
# Policy Loading
# =============================================================================

def load_policy(json_path: Path, use_adjusted: bool = True) -> List[Tuple[str, float, float]]:
    """Load policy from Phase C JSON file.
    
    v5.4: Supports loading adjusted probabilities for combination.
    
    Args:
        json_path: Path to phase_c_final_policy.json
        use_adjusted: If True, use probability_adjusted (for Ours_optimal).
                      If False, use probability_original (for Ours_p1).
                      Falls back to "probability" for backward compatibility.
        
    Returns:
        List of (op_name, magnitude, probability) tuples.
    """
    if not json_path.exists():
        raise FileNotFoundError(f"Policy file not found: {json_path}")
    
    with open(json_path, "r") as f:
        policy_dict = json.load(f)
    
    policy = []
    for op in policy_dict.get("ops", []):
        name = op["name"]
        magnitude = op["magnitude"]
        
        # v5.4: Support adjusted probabilities
        if use_adjusted and "probability_adjusted" in op:
            probability = op["probability_adjusted"]
        elif "probability_original" in op:
            probability = op["probability_original"]
        else:
            # Backward compatibility: use "probability" key
            probability = op.get("probability", 1.0)
        
        policy.append((name, magnitude, probability))
    
    return policy


# =============================================================================
# Training Functions
# =============================================================================

def train_single_config(
    method_name: str,
    transform: Callable,
    seed: int,
    epochs: int,
    device: torch.device,
    fold_idx: int,
    policy: Optional[List[Tuple[str, float, float]]] = None,
    batch_size: int = 128,
    num_workers: int = 8,
    early_stop_patience: int = 99999,
    deterministic: bool = True,
    save_checkpoint: bool = False,
    checkpoint_dir: Optional[Path] = None,
    weight_decay: float = 1e-2,
    label_smoothing: float = 0.1,
) -> Dict:
    """Train one configuration and return metrics.
    
    Args:
        method_name: Name of the method being evaluated.
        transform: Transform to apply to training data.
        seed: Random seed.
        epochs: Number of training epochs.
        device: Device to train on.
        fold_idx: Which fold to use.
        policy: Optional policy for Ours methods (for CSV output).
        batch_size: Batch size.
        num_workers: Number of data loading workers.
        early_stop_patience: Early stopping patience.
        deterministic: Enable deterministic CUDA mode.
        save_checkpoint: If True, save best model checkpoint.
        checkpoint_dir: Directory for checkpoints.
        
    Returns:
        Dict with training results.
    """
    start_time = time.time()
    set_seed_deterministic(seed, deterministic=deterministic)
    
    # Build magnitude and probability strings for CSV
    if method_name in ["Ours_p1", "Ours_optimal"] and policy and len(policy) > 0:
        if method_name == "Ours_p1":
            # Ablation: all p = 1.0
            mag_str = "+".join([str(round(op[1], 4)) for op in policy])
            prob_str = "+".join(["1.0" for _ in policy])
        else:
            # Optimal: use actual p values
            mag_str = "+".join([str(round(op[1], 4)) for op in policy])
            prob_str = "+".join([str(round(op[2], 4)) for op in policy])
    else:
        mag_str = "N/A"
        prob_str = "N/A"
    
    result = {
        "phase": "PhaseD",
        "op_name": method_name,
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
        val_transform = get_val_transform(include_normalize=False)
        
        # Create datasets
        train_dataset = CIFAR100Subsampled(
            root="./data",
            train=True,
            fold_idx=fold_idx,
            transform=transform,
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
        
        # Early stopping (v5.4: same settings as Phase A/B for consistency)
        early_stopper = EarlyStopping(
            patience=early_stop_patience,  # Default 80 for Phase D
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
        best_model_state = None
        
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
                        "model_state_dict": {k: v.cpu().clone() for k, v in model.state_dict().items()},
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                    }
            
            scheduler.step()
            
            if early_stopper(val_acc, epoch):
                result["epochs_run"] = epoch + 1
                early_stopped = True
                break
            
            result["epochs_run"] = epoch + 1
        
        result["val_acc"] = round(best_val_acc, 4)
        result["val_loss"] = round(best_val_loss, 6)
        result["top5_acc"] = round(best_top5_acc, 4)
        result["train_acc"] = round(best_train_acc, 4)
        result["train_loss"] = round(best_train_loss, 6)
        result["best_epoch"] = best_epoch
        result["early_stopped"] = early_stopped
        
        # Save checkpoint for Ours_optimal (final model for paper)
        if save_checkpoint and best_model_state is not None and checkpoint_dir is not None:
            checkpoint_path = checkpoint_dir / f"phase_d_fold{fold_idx}_best.pth"
            checkpoint = {
                "model_state_dict": best_model_state["model_state_dict"],
                "optimizer_state_dict": best_model_state["optimizer_state_dict"],
                "scheduler_state_dict": best_model_state["scheduler_state_dict"],
                "epoch": best_epoch,
                "val_acc": best_val_acc,
                "top5_acc": best_top5_acc,
                "val_loss": best_val_loss,
                "config": {
                    "method": method_name,
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
        
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {str(e)}"
        traceback.print_exc()
    
    result["runtime_sec"] = round(time.time() - start_time, 2)
    result["timestamp"] = datetime.now().isoformat(timespec='seconds')
    
    return result


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


def aggregate_results(raw_csv_path: Path, summary_csv_path: Path) -> pd.DataFrame:
    """Aggregate results by method, computing mean and std across folds.
    
    Args:
        raw_csv_path: Path to phase_d_results.csv
        summary_csv_path: Path to write summary CSV.
        
    Returns:
        Summary DataFrame.
    """
    df = pd.read_csv(raw_csv_path)
    
    # Filter out error rows
    df = df[(df["error"].isna()) | (df["error"] == "")]
    
    # Group by method (op_name)
    summary = df.groupby("op_name").agg(
        mean_val_acc=("val_acc", "mean"),
        std_val_acc=("val_acc", "std"),
        mean_top5_acc=("top5_acc", "mean"),
        std_top5_acc=("top5_acc", "std"),
        mean_train_acc=("train_acc", "mean"),
        std_train_acc=("train_acc", "std"),
        mean_runtime_sec=("runtime_sec", "mean"),
        n_folds=("fold_idx", "count"),
    ).reset_index()
    
    # Rename op_name to method
    summary = summary.rename(columns={"op_name": "method"})
    
    # Fill NaN std
    for col in ["std_val_acc", "std_top5_acc", "std_train_acc"]:
        summary[col] = summary[col].fillna(0.0)
    
    # Round values
    for col in ["mean_val_acc", "std_val_acc", "mean_top5_acc", "std_top5_acc",
                "mean_train_acc", "std_train_acc", "mean_runtime_sec"]:
        summary[col] = summary[col].round(4)
    
    # Sort by mean_val_acc descending
    summary = summary.sort_values("mean_val_acc", ascending=False)
    
    # Save to CSV
    summary.to_csv(summary_csv_path, index=False)
    
    return summary


# =============================================================================
# Main Benchmark
# =============================================================================

def run_phase_d(
    output_dir: Path,
    policy_json: Optional[Path] = None,
    phase_b_csv: Optional[Path] = None,  # v5.5: for Best_SingleOp
    methods: Optional[List[str]] = None,
    folds: Optional[List[int]] = None,
    epochs: int = 200,
    seed: int = 42,
    num_workers: int = 8,
    early_stop_patience: int = 60,
    deterministic: bool = True,
    dry_run: bool = False,
    weight_decay: float = 1e-2,
    label_smoothing: float = 0.1,
) -> pd.DataFrame:
    """Run Phase D benchmark comparison.
    
    v5.5: Added phase_b_csv parameter for Best_SingleOp method.
    
    Args:
        output_dir: Directory for output files.
        policy_json: Path to Phase C policy JSON (for Ours methods).
        phase_b_csv: Path to Phase B summary CSV (for Best_SingleOp).
        methods: List of methods to run. Default: all.
        folds: List of fold indices to run. Default: [0,1,2,3,4].
        epochs: Training epochs per configuration.
        seed: Random seed (same for all folds for reproducibility).
        num_workers: Data loading workers.
        early_stop_patience: Early stopping patience.
        deterministic: Enable deterministic CUDA mode.
        dry_run: If True, run minimal epochs for testing.
        
    Returns:
        Summary DataFrame.
    """
    device = get_device()
    ensure_dir(output_dir)
    
    raw_csv_path = output_dir / "phase_d_results.csv"
    summary_csv_path = output_dir / "phase_d_summary.csv"
    checkpoint_dir = output_dir / "checkpoints"
    ensure_dir(checkpoint_dir)
    
    # Default methods and folds
    if methods is None:
        methods = AVAILABLE_METHODS
    if folds is None:
        folds = [0, 1, 2, 3, 4]
    
    # Load policy if needed
    policy = None
    if any(m in methods for m in ["Ours_p1", "Ours_optimal"]):
        if policy_json is None:
            # Try main outputs directory first, then output_dir
            main_policy_path = Path("outputs/phase_c_final_policy.json")
            local_policy_path = output_dir / "phase_c_final_policy.json"
            
            if main_policy_path.exists():
                policy_json = main_policy_path
            elif local_policy_path.exists():
                policy_json = local_policy_path
            else:
                policy_json = main_policy_path  # For error message
        
        if policy_json.exists():
            # v5.4: Load both original and adjusted policies
            policy_original = load_policy(policy_json, use_adjusted=False)
            policy_adjusted = load_policy(policy_json, use_adjusted=True)
            policy = policy_adjusted  # Default for get_method_transform
            
            print(f"Loaded policy from {policy_json}")
            print(f"Policy (original): {[(op[0], f'm={op[1]:.2f}', f'p={op[2]:.2f}') for op in policy_original]}")
            print(f"Policy (adjusted): {[(op[0], f'm={op[1]:.2f}', f'p={op[2]:.2f}') for op in policy_adjusted]}")
        else:
            print(f"WARNING: Policy file not found: {policy_json}")
            print("Ours_p1 and Ours_optimal will use baseline only.")
            policy_original = None
            policy_adjusted = None
    
    # v5.5: Load Best_SingleOp configuration from Phase B summary
    global BEST_SINGLE_OP_CONFIG
    if "Best_SingleOp" in methods:
        if phase_b_csv is None:
            # Try default locations
            main_b_path = Path("outputs/phase_b_tuning_summary.csv")
            local_b_path = output_dir / "phase_b_tuning_summary.csv"
            
            if main_b_path.exists():
                phase_b_csv = main_b_path
            elif local_b_path.exists():
                phase_b_csv = local_b_path
        
        if phase_b_csv and phase_b_csv.exists():
            import pandas as pd
            b_df = pd.read_csv(phase_b_csv)
            if len(b_df) > 0:
                # Get the row with highest mean_val_acc
                best_row = b_df.loc[b_df['mean_val_acc'].idxmax()]
                BEST_SINGLE_OP_CONFIG = (
                    best_row['op_name'],
                    float(best_row['magnitude']),
                    float(best_row['probability']),
                )
                print(f"Best_SingleOp: {BEST_SINGLE_OP_CONFIG[0]} (m={BEST_SINGLE_OP_CONFIG[1]:.4f}, p={BEST_SINGLE_OP_CONFIG[2]:.4f})")
            else:
                print("WARNING: Phase B summary is empty. Best_SingleOp will use baseline.")
                BEST_SINGLE_OP_CONFIG = None
        else:
            print(f"WARNING: Phase B summary not found. Best_SingleOp will use baseline.")
    
    print("=" * 70)
    print("Phase D: SOTA Benchmark Comparison")
    print("=" * 70)
    print(f"Methods: {methods}")
    print(f"Folds: {folds}")
    print(f"Epochs: {epochs}")
    print(f"Seed: {seed}")
    print(f"Deterministic: {deterministic}")
    if dry_run:
        print("MODE: DRY RUN")
    print("=" * 70)
    
    # Check if CSV needs header
    write_header = check_csv_needs_header(raw_csv_path)
    
    # Total configurations
    total_configs = len(methods) * len(folds)
    
    print(f"\nTotal configurations: {total_configs}")
    print("-" * 70)
    
    # Main loop
    config_count = 0
    for method_name in methods:
        print(f"\n{'='*70}")
        print(f"Method: {method_name} - {get_method_description(method_name)}")
        print("=" * 70)
        
        try:
            # v5.4: Use appropriate policy version for each method
            if method_name == "Ours_p1":
                method_policy = policy_original
            elif method_name == "Ours_optimal":
                method_policy = policy_adjusted
            else:
                method_policy = policy
            
            transform = get_method_transform(method_name, method_policy)
        except Exception as e:
            print(f"ERROR creating transform for {method_name}: {e}")
            continue
        
        for fold_idx in folds:
            config_count += 1
            
            print(f"\n[{config_count}/{total_configs}] {method_name} - Fold {fold_idx}")
            print("-" * 40)
            
            train_epochs = epochs if not dry_run else 2
            
            # Save checkpoint only for Ours_optimal (final model)
            save_ckpt = (method_name == "Ours_optimal")
            
            result = train_single_config(
                method_name=method_name,
                transform=transform,
                seed=seed,
                epochs=train_epochs,
                device=device,
                fold_idx=fold_idx,
                policy=method_policy,  # v5.4: Use method-specific policy
                num_workers=num_workers,
                early_stop_patience=early_stop_patience,
                deterministic=deterministic,
                save_checkpoint=save_ckpt,
                checkpoint_dir=checkpoint_dir,
                weight_decay=weight_decay,
                label_smoothing=label_smoothing,
            )
            
            # Write result
            write_csv_row(raw_csv_path, result, write_header)
            write_header = False
            
            if result["error"]:
                print(f"ERROR: {result['error']}")
            else:
                print(f"Result: val_acc={result['val_acc']:.2f}%, "
                      f"top5={result['top5_acc']:.2f}%, "
                      f"runtime={result['runtime_sec']:.1f}s")
    
    # Aggregate results
    print("\n" + "-" * 70)
    print("Aggregating results...")
    summary_df = aggregate_results(raw_csv_path, summary_csv_path)
    
    # Print summary
    print("\n" + "=" * 70)
    print("Phase D Complete")
    print("=" * 70)
    print(f"Raw results: {raw_csv_path}")
    print(f"Summary: {summary_csv_path}")
    print("-" * 70)
    print("\nFinal Results (Mean ± Std across folds):")
    print(summary_df.to_string(index=False))
    print("=" * 70)
    
    return summary_df


# =============================================================================
# Argument Parsing
# =============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Phase D: SOTA Benchmark Comparison (5-Fold)"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
        help="Number of training epochs per config (default: 200)"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Output directory for raw/summary CSV and checkpoints (default: outputs)"
    )
    
    parser.add_argument(
        "--policy_json",
        type=str,
        default=None,
        help="Path to Phase C policy JSON (default: outputs/phase_c_final_policy.json; required for Ours_* methods)"
    )
    
    parser.add_argument(
        "--phase_b_csv",
        type=str,
        default=None,
        help="Path to Phase B summary CSV for Best_SingleOp (default: outputs/phase_b_tuning_summary.csv; required if Best_SingleOp enabled)"
    )
    
    parser.add_argument(
        "--methods",
        type=str,
        default=None,
        help=f"Comma-separated list of methods (default: all). Available: {AVAILABLE_METHODS}"
    )
    
    parser.add_argument(
        "--folds",
        type=str,
        default=None,
        help="Comma-separated list of fold indices (default: 0,1,2,3,4)"
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
        help="Early stopping patience (default: 60 for Phase D)"
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
    
    return parser.parse_args()


# =============================================================================
# Main Entry Point
# =============================================================================

def main() -> int:
    """Main entry point."""
    args = parse_args()
    
    output_dir = Path(args.output_dir)
    deterministic = not args.no_deterministic
    
    # Load Phase 0 hyperparameters if available
    phase0_cfg = load_phase0_best_config()
    wd = phase0_cfg[0] if phase0_cfg else 1e-2
    ls = phase0_cfg[1] if phase0_cfg else 0.1
    if phase0_cfg:
        print(f"[Phase0] Using recommended hyperparams: wd={wd}, ls={ls}")
    else:
        print("[Phase0] phase0_summary.csv not found; fallback to defaults wd=1e-2, ls=0.1")
    
    # Parse methods
    methods = None
    if args.methods:
        methods = [m.strip() for m in args.methods.split(",")]
        invalid = [m for m in methods if m not in AVAILABLE_METHODS]
        if invalid:
            print(f"ERROR: Invalid methods: {invalid}")
            print(f"Available: {AVAILABLE_METHODS}")
            return 1
    
    # Parse folds
    folds = None
    if args.folds:
        folds = [int(f.strip()) for f in args.folds.split(",")]
        invalid = [f for f in folds if f < 0 or f > 4]
        if invalid:
            print(f"ERROR: Invalid folds: {invalid}. Must be in [0, 4].")
            return 1
    
    # Parse policy path
    policy_json = None
    if args.policy_json:
        policy_json = Path(args.policy_json)
    
    device = get_device()
    
    print("=" * 70)
    print("Phase D: SOTA Benchmark Comparison")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Epochs: {args.epochs}")
    print(f"Seed: {args.seed}")
    print(f"Deterministic: {deterministic}")
    print(f"LR: 0.1, WD: {wd}, Momentum: 0.9, Warmup: 5 epochs, Label Smoothing: {ls}")
    if args.dry_run:
        print("MODE: DRY RUN")
    print("=" * 70)
    
    # v5.5: Parse phase_b_csv
    phase_b_csv = Path(args.phase_b_csv) if args.phase_b_csv else None
    
    try:
        run_phase_d(
            output_dir=output_dir,
            policy_json=policy_json,
            phase_b_csv=phase_b_csv,  # v5.5: for Best_SingleOp
            methods=methods,
            folds=folds,
            epochs=args.epochs,
            seed=args.seed,
            num_workers=args.num_workers,
            early_stop_patience=args.early_stop_patience,
            deterministic=deterministic,
            dry_run=args.dry_run,
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
