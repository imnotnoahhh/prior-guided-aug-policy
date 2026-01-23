#!/usr/bin/env python3
"""
Shot Sweep Experiment: Accuracy/Variance vs Sample Size

This script evaluates how different augmentation strategies perform across
varying sample sizes (shots per class). It demonstrates that SAS maintains
lower variance as sample size decreases, while RandAugment's variance increases.

Strategy:
1. First run fold0 for all shots to get trend signal quickly
2. After confirming trend, run remaining folds (1-4)

Usage:
    # Quick trend check (fold0 only, ~3-4 hours)
    python scripts/run_shot_sweep.py --folds 0

    # Full run (all folds, ~12+ hours)
    python scripts/run_shot_sweep.py

    # Dry run
    python scripts/run_shot_sweep.py --dry_run --epochs 2
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
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.augmentations import (
    build_transform_with_op,
    get_baseline_transform,
    get_randaugment_transform,
    get_val_transform,
)
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

import torchvision
from sklearn.model_selection import StratifiedKFold, train_test_split


# =============================================================================
# Shot-Based Dataset
# =============================================================================

class CIFAR100ShotBased(Dataset):
    """CIFAR-100 dataset with configurable samples per class (shot).
    
    Args:
        root: Root directory for dataset storage.
        train: If True, returns training split.
        fold_idx: Which fold to use (0-4).
        samples_per_class: Number of samples per class (shot).
        transform: Transform to apply.
        val_ratio: Fraction for validation split.
        random_state: Random seed.
    """
    
    def __init__(
        self,
        root: str,
        train: bool = True,
        fold_idx: int = 0,
        samples_per_class: int = 100,
        transform: Optional[Callable] = None,
        val_ratio: float = 0.1,
        random_state: int = 42,
    ):
        self.root = Path(root)
        self.train = train
        self.fold_idx = fold_idx
        self.samples_per_class = samples_per_class
        self.transform = transform
        self.val_ratio = val_ratio
        self.random_state = random_state
        
        # Load full CIFAR-100
        self.full_dataset = torchvision.datasets.CIFAR100(
            root=str(self.root),
            train=True,
            download=True,
        )
        
        targets = np.array(self.full_dataset.targets)
        n_classes = 100
        
        # Set random seed for reproducibility
        np.random.seed(random_state + fold_idx)  # Different seed per fold
        
        # Sample `samples_per_class` from each class
        selected_indices = []
        for class_id in range(n_classes):
            class_indices = np.where(targets == class_id)[0]
            
            # Shuffle and select
            np.random.shuffle(class_indices)
            selected = class_indices[:samples_per_class]
            selected_indices.extend(selected.tolist())
        
        selected_indices = np.array(selected_indices)
        selected_targets = targets[selected_indices]
        
        # Split into train/val
        n_val = int(len(selected_indices) * val_ratio)
        
        # Use stratified split
        train_idx, val_idx = train_test_split(
            np.arange(len(selected_indices)),
            test_size=val_ratio,
            random_state=random_state,
            stratify=selected_targets,
        )
        
        if train:
            self.indices = selected_indices[train_idx]
        else:
            self.indices = selected_indices[val_idx]
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        image, label = self.full_dataset[real_idx]
        
        if self.transform is not None:
            image = self.transform(image)
        
        return image, label


# =============================================================================
# Training Function
# =============================================================================

def train_single_config(
    method_name: str,
    transform: Callable,
    train_seed: int,
    epochs: int,
    device: torch.device,
    fold_idx: int,
    samples_per_class: int,
    data_seed: int = 42,
    batch_size: int = 128,
    num_workers: int = 8,
    weight_decay: float = 1e-2,
    label_smoothing: float = 0.1,
    early_stop_patience: int = 60,
) -> Dict:
    """Train one configuration and return metrics with timing.
    
    Args:
        train_seed: Seed for training randomness (weight init, dropout, etc.)
        data_seed: Seed for data sampling (which images are selected). 
                   Keep fixed to isolate training variance from data variance.
    """
    
    start_time = time.time()
    # Use train_seed for training randomness
    set_seed_deterministic(train_seed, deterministic=True)
    
    result = {
        "shot": samples_per_class,
        "method": method_name,
        "fold_idx": fold_idx,
        "seed": train_seed,
        "data_seed": data_seed,
        "val_acc": -1.0,
        "val_loss": -1.0,
        "top5_acc": -1.0,
        "train_acc": -1.0,
        "train_loss": -1.0,
        "epochs_run": 0,
        "best_epoch": 0,
        "runtime_sec": 0.0,
        "epoch_time_avg": 0.0,
        "timestamp": "",
        "error": "",
    }
    
    try:
        val_transform = get_val_transform(include_normalize=False)
        
        # Create datasets with shot-based sampling
        # Use data_seed (fixed) for data sampling, not train_seed
        train_dataset = CIFAR100ShotBased(
            root="./data",
            train=True,
            fold_idx=fold_idx,
            samples_per_class=samples_per_class,
            transform=transform,
            random_state=data_seed,  # Fixed data seed
        )
        
        val_dataset = CIFAR100ShotBased(
            root="./data",
            train=False,
            fold_idx=fold_idx,
            samples_per_class=samples_per_class,
            transform=val_transform,
            random_state=data_seed,  # Fixed data seed
        )
        
        print(f"    Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
        
        # Data loaders
        use_cuda = device.type == "cuda"
        train_loader = DataLoader(
            train_dataset,
            batch_size=min(batch_size, len(train_dataset)),
            shuffle=True,
            num_workers=num_workers,
            pin_memory=use_cuda,
            drop_last=False,
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=min(batch_size, len(val_dataset)),
            shuffle=False,
            num_workers=num_workers,
            pin_memory=use_cuda,
            drop_last=False,
        )
        
        # Create model
        model = create_model(num_classes=100, pretrained=False)
        model = model.to(device)
        if use_cuda:
            model = model.to(memory_format=torch.channels_last)
        
        # Loss, optimizer, scheduler
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        optimizer, scheduler = get_optimizer_and_scheduler(
            model=model,
            total_epochs=epochs,
            lr=0.1,
            weight_decay=weight_decay,
            momentum=0.9,
            warmup_epochs=5,
        )
        
        # AMP scaler
        scaler = torch.amp.GradScaler() if use_cuda else None
        
        # Early stopping
        early_stopper = EarlyStopping(
            patience=early_stop_patience,
            mode="max",
            min_epochs=min(60, epochs // 2),
            min_delta=0.2,
        )
        
        # Training loop with timing
        best_val_acc = 0.0
        best_val_loss = float("inf")
        best_top5_acc = 0.0
        best_train_acc = 0.0
        best_train_loss = 0.0
        best_epoch = 0
        epoch_times = []
        
        for epoch in range(epochs):
            epoch_start = time.time()
            
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
            
            epoch_time = time.time() - epoch_start
            epoch_times.append(epoch_time)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_val_loss = val_loss
                best_top5_acc = top5_acc
                best_train_acc = train_acc
                best_train_loss = train_loss
                best_epoch = epoch + 1
            
            scheduler.step()
            
            # Early stopping check
            if early_stopper(val_acc, epoch):
                print(f"    Early stopped at epoch {epoch + 1}")
                break
        
        runtime = time.time() - start_time
        
        result.update({
            "val_acc": best_val_acc,
            "val_loss": best_val_loss,
            "top5_acc": best_top5_acc,
            "train_acc": best_train_acc,
            "train_loss": best_train_loss,
            "epochs_run": epoch + 1,
            "best_epoch": best_epoch,
            "runtime_sec": runtime,
            "epoch_time_avg": np.mean(epoch_times),
            "timestamp": datetime.now().isoformat(),
        })
        
    except Exception as e:
        result["error"] = str(e)
        result["runtime_sec"] = time.time() - start_time
        result["timestamp"] = datetime.now().isoformat()
        traceback.print_exc()
    
    return result


# =============================================================================
# Method Transforms
# =============================================================================

def get_method_transform(method_name: str, sas_config: Optional[Tuple] = None) -> Callable:
    """Get transform for a method."""
    if method_name == "Baseline":
        return get_baseline_transform(include_normalize=False)
    
    elif method_name == "RandAugment":
        return get_randaugment_transform(n=2, m=9, include_baseline=True, include_normalize=False)
    
    elif method_name == "SAS":
        if sas_config is None:
            # Default: ColorJitter with optimal params from Phase B
            op_name, magnitude, probability = "ColorJitter", 0.35, 0.8
        else:
            op_name, magnitude, probability = sas_config
        return build_transform_with_op(
            op_name, magnitude, probability,
            include_baseline=True, include_normalize=False
        )
    
    else:
        raise ValueError(f"Unknown method: {method_name}")


# =============================================================================
# Main
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Shot Sweep Experiment")
    parser.add_argument("--shots", type=str, default="20,50,100,200",
                        help="Comma-separated shot values")
    parser.add_argument("--methods", type=str, default="Baseline,RandAugment,SAS",
                        help="Comma-separated method names")
    parser.add_argument("--folds", type=str, default="0,1,2,3,4",
                        help="Comma-separated fold indices")
    parser.add_argument("--epochs", type=int, default=200,
                        help="Training epochs")
    parser.add_argument("--seed", type=int, default=42,
                        help="Training random seed (weight init, dropout, etc.)")
    parser.add_argument("--data_seed", type=int, default=42,
                        help="Data sampling seed (which images are selected). Keep fixed to isolate training variance.")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=8,
                        help="Number of data loading workers. Use 0 for Mac/debugging.")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--reuse_100shot", action="store_true", default=True,
                        help="Reuse 100-shot results from phase_d_results.csv")
    parser.add_argument("--sas_config", type=str, default=None,
                        help="SAS config as 'op_name,magnitude,probability'")
    return parser.parse_args()


def load_existing_100shot_results(output_dir: Path) -> pd.DataFrame:
    """Load existing 100-shot results from phase_d_results.csv."""
    phase_d_path = output_dir / "phase_d_results.csv"
    if not phase_d_path.exists():
        return pd.DataFrame()
    
    df = pd.read_csv(phase_d_path)
    
    # Map method names
    method_map = {
        "Baseline": "Baseline",
        "RandAugment": "RandAugment",
        "Ours_optimal": "SAS",
    }
    
    results = []
    for _, row in df.iterrows():
        if row["op_name"] in method_map:
            results.append({
                "shot": 100,
                "method": method_map[row["op_name"]],
                "fold_idx": row["fold_idx"],
                "seed": row["seed"],
                "val_acc": row["val_acc"],
                "val_loss": row["val_loss"],
                "top5_acc": row["top5_acc"],
                "train_acc": row["train_acc"],
                "train_loss": row["train_loss"],
                "epochs_run": row["epochs_run"],
                "best_epoch": row["best_epoch"],
                "runtime_sec": row["runtime_sec"],
                "epoch_time_avg": row["runtime_sec"] / row["epochs_run"] if row["epochs_run"] > 0 else 0,
                "timestamp": row["timestamp"],
                "error": row.get("error", ""),
            })
    
    return pd.DataFrame(results)


def main():
    args = parse_args()
    
    # Parse arguments
    shots = [int(s) for s in args.shots.split(",")]
    methods = [m.strip() for m in args.methods.split(",")]
    folds = [int(f) for f in args.folds.split(",")]
    
    # Parse SAS config
    sas_config = None
    if args.sas_config:
        parts = args.sas_config.split(",")
        sas_config = (parts[0], float(parts[1]), float(parts[2]))
    else:
        # Try to load from phase_b_tuning_summary.csv
        phase_b_path = Path(args.output_dir) / "phase_b_tuning_summary.csv"
        if phase_b_path.exists():
            df = pd.read_csv(phase_b_path)
            # Get best ColorJitter config
            cj = df[df["op_name"] == "ColorJitter"]
            if len(cj) > 0:
                best = cj.loc[cj["mean_val_acc"].idxmax()]
                sas_config = ("ColorJitter", best["magnitude"], best["probability"])
                print(f"Loaded SAS config from Phase B: {sas_config}")
    
    if sas_config is None:
        sas_config = ("ColorJitter", 0.35, 0.8)
        print(f"Using default SAS config: {sas_config}")
    else:
        print(f"Using SAS config: {sas_config}")
    
    # Setup
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)
    
    device = get_device()
    print(f"Using device: {device}")
    
    # Load Phase 0 config
    try:
        weight_decay, label_smoothing = load_phase0_best_config(output_dir)
    except:
        weight_decay, label_smoothing = 1e-2, 0.1
    print(f"Using weight_decay={weight_decay}, label_smoothing={label_smoothing}")
    
    # Output file
    output_csv = output_dir / "shot_sweep_results.csv"
    
    # Load existing results
    existing_results = []
    if output_csv.exists():
        existing_df = pd.read_csv(output_csv)
        existing_results = existing_df.to_dict("records")
        print(f"Loaded {len(existing_results)} existing results")
    
    # Check for 100-shot reuse
    reused_100shot = pd.DataFrame()
    if args.reuse_100shot and 100 in shots:
        reused_100shot = load_existing_100shot_results(output_dir)
        if len(reused_100shot) > 0:
            print(f"Reusing {len(reused_100shot)} results from phase_d_results.csv for 100-shot")
    
    # Build experiment queue
    experiments = []
    for shot in shots:
        for method in methods:
            for fold_idx in folds:
                # Check if already done in existing results
                done = any(
                    r["shot"] == shot and r["method"] == method and r["fold_idx"] == fold_idx
                    for r in existing_results
                )
                
                # Check if can reuse 100-shot (but also check if already in existing_results)
                if shot == 100 and len(reused_100shot) > 0:
                    reused = reused_100shot[
                        (reused_100shot["method"] == method) & 
                        (reused_100shot["fold_idx"] == fold_idx)
                    ]
                    if len(reused) > 0:
                        continue  # Skip, will add reused results later
                
                if not done:
                    experiments.append((shot, method, fold_idx))
    
    # Sort experiments for optimal trend discovery:
    # Priority: fold0 first, then 50-shot → 20-shot → 100-shot → 200-shot
    shot_priority = {50: 0, 20: 1, 100: 2, 200: 3}
    method_priority = {"Baseline": 0, "RandAugment": 1, "SAS": 2}
    
    experiments.sort(key=lambda x: (
        x[2],  # fold_idx first (fold0 comes first)
        shot_priority.get(x[0], 99),  # then shot priority
        method_priority.get(x[1], 99),  # then method priority
    ))
    
    print(f"\nExperiment queue: {len(experiments)} runs")
    print(f"Shots: {shots}")
    print(f"Methods: {methods}")
    print(f"Folds: {folds}")
    print(f"Epochs: {args.epochs}")
    print(f"Data seed: {args.data_seed} (fixed for all experiments)")
    
    if args.dry_run:
        print("\n[DRY RUN MODE]")
        args.epochs = 2
    
    # Run experiments
    all_results = existing_results.copy()
    
    # Add reused 100-shot results (with deduplication check)
    if len(reused_100shot) > 0:
        for _, row in reused_100shot.iterrows():
            if row["method"] in methods and row["fold_idx"] in folds:
                # Check if this combination already exists in all_results
                already_exists = any(
                    r["shot"] == 100 and 
                    r["method"] == row["method"] and 
                    r["fold_idx"] == row["fold_idx"]
                    for r in all_results
                )
                if not already_exists:
                    all_results.append(row.to_dict())
    
    for i, (shot, method, fold_idx) in enumerate(experiments):
        print(f"\n[{i+1}/{len(experiments)}] Shot={shot}, Method={method}, Fold={fold_idx}")
        
        # Get transform
        transform = get_method_transform(method, sas_config if method == "SAS" else None)
        
        # Train
        result = train_single_config(
            method_name=method,
            transform=transform,
            train_seed=args.seed,
            epochs=args.epochs,
            device=device,
            fold_idx=fold_idx,
            samples_per_class=shot,
            data_seed=args.data_seed,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            weight_decay=weight_decay,
            label_smoothing=label_smoothing,
        )
        
        all_results.append(result)
        
        # Save incrementally
        pd.DataFrame(all_results).to_csv(output_csv, index=False)
        
        print(f"    Val Acc: {result['val_acc']:.2f}%, "
              f"Epoch Time: {result['epoch_time_avg']:.2f}s")
    
    # Generate summary
    print("\n" + "=" * 60)
    print("Shot Sweep Summary")
    print("=" * 60)
    
    df = pd.DataFrame(all_results)
    
    summary_rows = []
    for shot in shots:
        for method in methods:
            subset = df[(df["shot"] == shot) & (df["method"] == method)]
            if len(subset) > 0:
                mean_acc = subset["val_acc"].mean()
                std_acc = subset["val_acc"].std()
                min_acc = subset["val_acc"].min()
                lower_bound = mean_acc - std_acc
                mean_epoch_time = subset["epoch_time_avg"].mean()
                n_folds = len(subset)
                
                summary_rows.append({
                    "shot": shot,
                    "method": method,
                    "mean_acc": mean_acc,
                    "std_acc": std_acc,
                    "min_acc": min_acc,
                    "lower_bound": lower_bound,
                    "epoch_time_avg": mean_epoch_time,
                    "n_folds": n_folds,
                })
                
                print(f"Shot={shot:3d}, {method:12s}: {mean_acc:.2f} ± {std_acc:.2f} "
                      f"(min={min_acc:.2f}, n={n_folds})")
    
    # Save summary
    summary_df = pd.DataFrame(summary_rows)
    summary_csv = output_dir / "shot_sweep_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"\nSummary saved to: {summary_csv}")
    print(f"Raw results saved to: {output_csv}")


if __name__ == "__main__":
    main()
