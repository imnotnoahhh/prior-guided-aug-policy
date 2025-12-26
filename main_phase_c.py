# Phase C: Greedy Combination Search with Validation
"""
Phase C: Greedy Combination Search with Validation.

Algorithm:
1. Load Phase B results → get best (m, p) for each operation.
2. Start with the single best operation.
3. Greedy search: try adding complementary operations.
4. Validate each combination: only accept if acc > current + min_improvement.
5. Use t-test to verify statistical significance.
6. Stop when no improvement found or max_ops reached.
7. Save policy with full (name, magnitude, probability) tuples.

Key Parameters:
- max_ops: Maximum number of operations to combine (default 3)
- min_improvement: Minimum accuracy gain to accept new op (default 0.1%)
- p_any_target: Target probability for combination adjustment (default 0.7)

Usage:
    # Full run
    python main_phase_c.py --max_ops 3 --min_improvement 0.1
    
    # Dry run
    python main_phase_c.py --dry_run
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
    build_dynamic_transform,  # v6: Dynamic
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

def train_dynamic_policy(
    ops: List[Tuple[str, float]],
    n_ops: int,
    seed: int,
    epochs: int,
    device: torch.device,
    fold_idx: int = 0,
    batch_size: int = 128,
    num_workers: int = 8,
    early_stop_patience: int = 200, # v6: Disable early stopping essentially
    deterministic: bool = True,
    save_checkpoint: bool = False,
    checkpoint_dir: Optional[Path] = None,
    weight_decay: float = 1e-2,
    label_smoothing: float = 0.1,
) -> Dict:
    """Train Dynamic Policy and return metrics.
    
    Args:
        ops: List of (op_name, magnitude) tuples for the Elite Pool.
        n_ops: Number of ops to select per image.
        seed: Random seed.
        epochs: Training epochs.
        ...
        
    Returns:
        Dict with training results.
    """
    start_time = time.time()
    set_seed_deterministic(seed, deterministic=deterministic)
    
    # Build string for logging
    ops_str = f"Dynamic(Top-{len(ops)}, N={n_ops})"
    
    result = {
        "phase": "PhaseC",
        "op_name": ops_str,
        "magnitude": "N/A",  # Varied
        "probability": "dynamic",
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
        # Build dynamic transform
        if not ops:
            train_transform = get_baseline_transform(include_normalize=False)
        else:
            train_transform = build_dynamic_transform(
                ops=ops,
                n=n_ops,
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
        
        # Model
        model = create_model(num_classes=100, pretrained=False)
        model = model.to(device)
        if use_cuda:
            model = model.to(memory_format=torch.channels_last)
        
        # Optimization
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        optimizer, scheduler = get_optimizer_and_scheduler(
            model=model,
            total_epochs=epochs,
            lr=0.1,
            weight_decay=weight_decay,
            momentum=0.9,
            warmup_epochs=5,
        )
        
        scaler = torch.amp.GradScaler() if device.type == "cuda" else None
        
        # Early stopping
        early_stopper = EarlyStopping(
            patience=early_stop_patience,
            mode="max",
            min_epochs=60,
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
        
        # Save checkpoint
        if save_checkpoint and best_model_state is not None and checkpoint_dir is not None:
            checkpoint_path = checkpoint_dir / f"phase_c_dynamic_seed{seed}_best.pth"
            checkpoint = {
                **best_model_state,
                "epoch": best_epoch,
                "val_acc": best_val_acc,
                "config": {"ops": ops, "n_ops": n_ops}
            }
            torch.save(checkpoint, checkpoint_path)
        
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
    ops: List[Tuple[str, float]],
    n_ops: int,
    seeds: List[int],
    epochs: int,
    device: torch.device,
    fold_idx: int = 0,
    weight_decay: float = 1e-2,
    label_smoothing: float = 0.1,
    **kwargs,
) -> Tuple[float, float, List[Dict]]:
    """Train configuration with multiple seeds."""
    results = []
    val_accs = []
    
    for seed in seeds:
        result = train_dynamic_policy(
            ops=ops,
            n_ops=n_ops,
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
    ops: List[Tuple[str, float, float]],  # v7: now includes probability
    baseline_acc: float,
    final_acc: float,
    output_path: Path,
    strategy: str = "greedy_validated",  # v7: new strategy type
    best_single_acc: float = None,
) -> None:
    """Save final policy to JSON file (v7 format).
    
    v7: Saves full (name, magnitude, probability) for each operation.
    """
    policy_dict = {
        "version": "v7.0",
        "phase": "PhaseC",
        "strategy": strategy,
        "baseline_acc": baseline_acc,
        "best_single_acc": best_single_acc,
        "final_acc": final_acc,
        "improvement_over_baseline": round(final_acc - baseline_acc, 4),
        "improvement_over_single": round(final_acc - best_single_acc, 4) if best_single_acc else None,
        "n_ops": len(ops),
        "ops": [
            {
                "name": op[0],
                "magnitude": round(op[1], 4),
                "probability": round(op[2], 4),
            }
            for op in ops
        ],
        "timestamp": datetime.now().isoformat(timespec='seconds'),
    }
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(policy_dict, f, indent=2)
    
    print(f"Policy saved to: {output_path}")


def load_policy_for_dynamic(json_path: Path) -> Tuple[List[Tuple[str, float]], int]:
    """Load dynamic policy from JSON."""
    with open(json_path, "r") as f:
        d = json.load(f)
    
    if d.get("strategy") != "dynamic_prior":
        print(f"WARNING: Policy strategy is {d.get('strategy')}, expected dynamic_prior")
        
    ops = []
    for op in d["pool"]:
        ops.append((op["name"], op["magnitude"]))
    
    return ops, d["n_ops"]



# =============================================================================
# v7: Greedy Combination Search with Validation (New Main Logic)
# =============================================================================

def train_static_policy(
    ops: List[Tuple[str, float, float]],  # (name, magnitude, probability)
    seed: int,
    epochs: int,
    device: torch.device,
    fold_idx: int = 0,
    batch_size: int = 128,
    num_workers: int = 8,
    weight_decay: float = 1e-2,
    label_smoothing: float = 0.1,
    p_any_target: float = 0.7,
) -> Dict:
    """Train a static combination policy and return metrics.
    
    v7: Uses build_transform_with_ops with adjusted probabilities.
    """
    from src.augmentations import build_transform_with_ops
    
    start_time = time.time()
    set_seed_deterministic(seed, deterministic=True)
    
    # Adjust probabilities for combination
    if len(ops) > 1:
        adjusted_ops = adjust_probabilities_for_combination(ops, p_any_target=p_any_target)
    else:
        adjusted_ops = ops
    
    # Build string for logging
    ops_str = "+".join([op[0] for op in adjusted_ops])
    mag_str = "+".join([f"{op[1]:.3f}" for op in adjusted_ops])
    prob_str = "+".join([f"{op[2]:.3f}" for op in adjusted_ops])
    
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
        train_transform = build_transform_with_ops(adjusted_ops, include_baseline=True, include_normalize=False)
        val_transform = get_val_transform(include_normalize=False)
        
        # Create datasets
        train_dataset = CIFAR100Subsampled(
            root="./data", train=True, fold_idx=fold_idx,
            transform=train_transform, download=True,
        )
        val_dataset = CIFAR100Subsampled(
            root="./data", train=False, fold_idx=fold_idx,
            transform=val_transform, download=True,
        )
        
        # Create data loaders
        use_cuda = device.type == "cuda"
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=use_cuda, drop_last=False,
            persistent_workers=True if num_workers > 0 else False,
            prefetch_factor=4 if num_workers > 0 else None,
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=use_cuda, drop_last=False,
            persistent_workers=True if num_workers > 0 else False,
            prefetch_factor=4 if num_workers > 0 else None,
        )
        
        # Model
        model = create_model(num_classes=100, pretrained=False)
        model = model.to(device)
        if use_cuda:
            model = model.to(memory_format=torch.channels_last)
        
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        optimizer, scheduler = get_optimizer_and_scheduler(
            model=model, total_epochs=epochs, lr=0.1,
            weight_decay=weight_decay, momentum=0.9, warmup_epochs=5,
        )
        scaler = torch.amp.GradScaler() if device.type == "cuda" else None
        
        early_stopper = EarlyStopping(patience=60, mode="max", min_epochs=60, min_delta=0.2)
        
        best_val_acc = 0.0
        best_val_loss = float("inf")
        best_top5_acc = 0.0
        best_train_acc = 0.0
        best_train_loss = 0.0
        best_epoch = 0
        
        for epoch in range(epochs):
            train_loss, train_acc = train_one_epoch(
                model=model, train_loader=train_loader, criterion=criterion,
                optimizer=optimizer, device=device, scaler=scaler,
            )
            val_loss, val_acc, top5_acc = evaluate(
                model=model, val_loader=val_loader, criterion=criterion, device=device,
            )
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_val_loss = val_loss
                best_top5_acc = top5_acc
                best_train_acc = train_acc
                best_train_loss = train_loss
                best_epoch = epoch + 1
            
            scheduler.step()
            
            if early_stopper(val_acc, epoch):
                result["epochs_run"] = epoch + 1
                result["early_stopped"] = True
                break
            result["epochs_run"] = epoch + 1
        
        result["val_acc"] = round(best_val_acc, 4)
        result["val_loss"] = round(best_val_loss, 6)
        result["top5_acc"] = round(best_top5_acc, 4)
        result["train_acc"] = round(best_train_acc, 4)
        result["train_loss"] = round(best_train_loss, 6)
        result["best_epoch"] = best_epoch
        
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {str(e)}"
        traceback.print_exc()
    
    result["runtime_sec"] = round(time.time() - start_time, 2)
    result["timestamp"] = datetime.now().isoformat(timespec='seconds')
    
    return result


def train_static_multi_seed(
    ops: List[Tuple[str, float, float]],
    seeds: List[int],
    epochs: int,
    device: torch.device,
    fold_idx: int = 0,
    p_any_target: float = 0.7,
    weight_decay: float = 1e-2,
    label_smoothing: float = 0.1,
    **kwargs,
) -> Tuple[float, float, List[Dict], List[float]]:
    """Train static policy with multiple seeds, return mean/std and raw accuracies."""
    results = []
    val_accs = []
    
    for seed in seeds:
        result = train_static_policy(
            ops=ops, seed=seed, epochs=epochs, device=device, fold_idx=fold_idx,
            p_any_target=p_any_target, weight_decay=weight_decay,
            label_smoothing=label_smoothing, **kwargs,
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
    
    return mean_acc, std_acc, results, val_accs


def run_greedy_combination_search(
    phase_b_csv: Path,
    baseline_acc: float,
    output_dir: Path,
    max_ops: int = 3,
    min_improvement: float = 0.1,
    p_any_target: float = 0.7,
    epochs: int = 200,
    seeds: List[int] = [42, 123, 456],
    fold_idx: int = 0,
    num_workers: int = 8,
    dry_run: bool = False,
    weight_decay: float = 1e-2,
    label_smoothing: float = 0.1,
) -> float:
    """Run Phase C Greedy Combination Search with Validation (v7).
    
    Algorithm:
    1. Start with best single operation from Phase B.
    2. Try adding each candidate operation.
    3. Keep addition only if it improves accuracy by min_improvement.
    4. Repeat until no improvement or max_ops reached.
    """
    from scipy import stats
    
    ensure_dir(output_dir)
    ensure_dir(output_dir / "checkpoints")
    history_csv_path = output_dir / "phase_c_history.csv"
    final_policy_json = output_dir / "phase_c_final_policy.json"
    device = get_device()
    
    # Load Phase B results
    print(f"Loading Phase B results from: {phase_b_csv}")
    phase_b_df = load_phase_b_summary(phase_b_csv)
    best_configs = get_best_config_per_op(phase_b_df)
    
    # Sort by accuracy
    sorted_ops = sorted(best_configs.items(), key=lambda x: x[1][2], reverse=True)
    
    print(f"\n{'='*70}")
    print(f"Phase C: Greedy Combination Search with Validation (v7)")
    print(f"{'='*70}")
    print(f"Parameters: max_ops={max_ops}, min_improvement={min_improvement}%, p_any_target={p_any_target}")
    print(f"\nAvailable operations (sorted by Phase B accuracy):")
    for i, (name, (mag, prob, acc)) in enumerate(sorted_ops):
        print(f"  {i+1}. {name}: m={mag:.4f}, p={prob:.4f}, acc={acc:.2f}%")
    
    if dry_run:
        epochs = 2
        seeds = seeds[:1]
        print("\n*** DRY RUN MODE ***")
    
    write_header = check_csv_needs_header(history_csv_path)
    
    # Step 1: Start with best single operation
    best_op_name, (best_m, best_p, best_phase_b_acc) = sorted_ops[0]
    current_policy = [(best_op_name, best_m, best_p)]
    
    print(f"\n--- Step 1: Evaluate Best Single Operation ---")
    print(f"Testing: {best_op_name} (m={best_m:.4f}, p={best_p:.4f})")
    
    mean_acc, std_acc, results, raw_accs = train_static_multi_seed(
        ops=current_policy, seeds=seeds, epochs=epochs, device=device,
        fold_idx=fold_idx, p_any_target=p_any_target,
        weight_decay=weight_decay, label_smoothing=label_smoothing,
    )
    
    for res in results:
        write_csv_row(history_csv_path, res, write_header)
        write_header = False
    
    current_acc = mean_acc
    current_raw_accs = raw_accs
    best_single_acc = mean_acc
    
    print(f"Result: {mean_acc:.2f}% ± {std_acc:.2f}%")
    
    # Step 2: Greedy search for additional operations
    for step in range(2, max_ops + 1):
        print(f"\n--- Step {step}: Search for Complementary Operation ---")
        
        # Get candidate operations (exclude those already in policy)
        used_names = {op[0] for op in current_policy}
        candidates = [(name, cfg) for name, cfg in sorted_ops if name not in used_names]
        
        if not candidates:
            print("No more candidate operations available.")
            break
        
        best_candidate = None
        best_candidate_acc = current_acc
        best_candidate_raw = None
        
        for cand_name, (cand_m, cand_p, _) in candidates:
            # Check mutual exclusion
            is_excluded = False
            for existing_op in current_policy:
                if check_mutual_exclusion(existing_op[0], cand_name):
                    print(f"  Skipping {cand_name}: mutually exclusive with {existing_op[0]}")
                    is_excluded = True
                    break
            if is_excluded:
                continue
            
            test_policy = current_policy + [(cand_name, cand_m, cand_p)]
            print(f"  Testing: {'+'.join([op[0] for op in test_policy])}")
            
            mean_acc, std_acc, results, raw_accs = train_static_multi_seed(
                ops=test_policy, seeds=seeds, epochs=epochs, device=device,
                fold_idx=fold_idx, p_any_target=p_any_target,
                weight_decay=weight_decay, label_smoothing=label_smoothing,
            )
            
            for res in results:
                write_csv_row(history_csv_path, res, write_header)
                write_header = False
            
            print(f"    Result: {mean_acc:.2f}% ± {std_acc:.2f}%")
            
            if mean_acc > best_candidate_acc:
                best_candidate = (cand_name, cand_m, cand_p)
                best_candidate_acc = mean_acc
                best_candidate_raw = raw_accs
        
        # Check if best candidate improves enough
        if best_candidate is not None:
            improvement = best_candidate_acc - current_acc
            
            # T-test for significance (if we have enough samples)
            if len(current_raw_accs) >= 2 and len(best_candidate_raw) >= 2:
                _, p_value = stats.ttest_ind(best_candidate_raw, current_raw_accs)
            else:
                p_value = 1.0
            
            print(f"\n  Best candidate: {best_candidate[0]}")
            print(f"  Improvement: {improvement:+.2f}% (p-value: {p_value:.4f})")
            
            if improvement >= min_improvement and p_value < 0.2:
                print(f"  ✓ Accepted! Adding {best_candidate[0]} to policy.")
                current_policy.append(best_candidate)
                current_acc = best_candidate_acc
                current_raw_accs = best_candidate_raw
            else:
                print(f"  ✗ Rejected. Improvement too small or not significant.")
                break
        else:
            print(f"\n  No improvement found. Stopping search.")
            break
    
    # Final summary
    print(f"\n{'='*70}")
    print(f"Phase C Complete")
    print(f"{'='*70}")
    print(f"Final Policy: {'+'.join([op[0] for op in current_policy])}")
    print(f"Final Accuracy: {current_acc:.2f}%")
    print(f"Improvement over Baseline: {current_acc - baseline_acc:+.2f}%")
    print(f"Improvement over Single Op: {current_acc - best_single_acc:+.2f}%")
    
    # Save policy
    save_policy(
        ops=current_policy,
        baseline_acc=baseline_acc,
        final_acc=current_acc,
        output_path=final_policy_json,
        strategy="greedy_validated",
        best_single_acc=best_single_acc,
    )
    
    return current_acc

def parse_args() -> argparse.Namespace:
    """Parse command line arguments (v7)."""
    parser = argparse.ArgumentParser(
        description="Phase C: Greedy Combination Search with Validation (v7)"
    )
    
    parser.add_argument(
        "--phase_b_csv",
        type=str,
        default="outputs/phase_b_tuning_summary.csv",
        help="Path to Phase B summary CSV (default: outputs/phase_b_tuning_summary.csv)"
    )
    
    parser.add_argument(
        "--baseline_acc",
        type=float,
        default=39.5,
        help="Baseline accuracy for reference (default: 39.5)"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Output directory (default: outputs)"
    )
    
    parser.add_argument(
        "--max_ops",
        type=int,
        default=3,
        help="Maximum number of operations to combine (default: 3)"
    )
    
    parser.add_argument(
        "--min_improvement",
        type=float,
        default=0.1,
        help="Minimum accuracy improvement to accept new op (default: 0.1%%)"
    )
    
    parser.add_argument(
        "--p_any_target",
        type=float,
        default=0.7,
        help="Target probability for combination adjustment (default: 0.7)"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
        help="Number of training epochs per evaluation (default: 200)"
    )
    
    parser.add_argument(
        "--seeds",
        type=str,
        default="42,123,456",
        help="Comma-separated random seeds (default: 42,123,456)"
    )
    
    parser.add_argument(
        "--fold_idx",
        type=int,
        default=0,
        help="Fold index to use (default: 0)"
    )
    
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Run fast dry-run for testing"
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    seeds = [int(s) for s in args.seeds.split(",")]
    phase_b_csv = Path(args.phase_b_csv)
    output_dir = Path(args.output_dir)
    
    # Load Phase 0 recommended hyperparameters
    phase0_cfg = load_phase0_best_config()
    wd = phase0_cfg[0] if phase0_cfg else 1e-2
    ls = phase0_cfg[1] if phase0_cfg else 0.1
    if phase0_cfg:
        print(f"[Phase0] Using recommended hyperparams: wd={wd}, ls={ls}")
    else:
        print("[Phase0] phase0_summary.csv not found; fallback to defaults wd=1e-2, ls=0.1")
    
    print(f"Starting Phase C (Greedy Combination Search v7)...")
    try:
        run_greedy_combination_search(
            phase_b_csv=phase_b_csv,
            baseline_acc=args.baseline_acc,
            output_dir=output_dir,
            max_ops=args.max_ops,
            min_improvement=args.min_improvement,
            p_any_target=args.p_any_target,
            epochs=args.epochs,
            seeds=seeds,
            fold_idx=args.fold_idx,
            dry_run=args.dry_run,
            weight_decay=wd,
            label_smoothing=ls,
        )
    except Exception as e:
        print(f"FATAL ERROR: {e}")
        traceback.print_exc()
        sys.exit(1)
