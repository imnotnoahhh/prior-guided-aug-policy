#!/usr/bin/env python3
"""
Search Ablation: Evaluate Phase A only configuration on 5-fold.

Phase A best config: RandomPerspective (m=0.014, p=0.1133)
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import csv
from datetime import datetime

import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

from src.augmentations import build_transform_with_op, get_val_transform
from src.dataset import CIFAR100Subsampled
from src.models import create_model
from src.utils import get_device, set_seed_deterministic, train_one_epoch, evaluate, get_optimizer_and_scheduler


def run_phase_a_only_evaluation(
    epochs: int = 200,
    folds: list = None,
    output_dir: str = "outputs",
):
    """Evaluate Phase A only configuration on 5-fold."""
    
    device = get_device()
    print(f"Using device: {device}")
    
    # Phase A best config (from Sobol screening, no ASHA tuning)
    OP_NAME = "RandomPerspective"
    MAGNITUDE = 0.014
    PROBABILITY = 0.1133
    
    if folds is None:
        folds = [0, 1, 2, 3, 4]
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    results_file = output_path / "search_ablation_results.csv"
    
    # Write header if file doesn't exist
    write_header = not results_file.exists()
    
    all_results = []
    data_root = PROJECT_ROOT / "data"
    
    for fold_idx in folds:
        print(f"\n{'='*60}")
        print(f"Fold {fold_idx + 1}/5 - Phase A Only ({OP_NAME} m={MAGNITUDE}, p={PROBABILITY})")
        print(f"{'='*60}")
        
        set_seed_deterministic(42)
        
        # Build transforms
        train_transform = build_transform_with_op(
            OP_NAME, MAGNITUDE, PROBABILITY,
            include_baseline=True,
            include_normalize=False,
        )
        val_transform = get_val_transform(include_normalize=False)
        
        # Create datasets
        train_dataset = CIFAR100Subsampled(
            root=data_root,
            train=True,
            fold_idx=fold_idx,
            transform=train_transform,
        )
        val_dataset = CIFAR100Subsampled(
            root=data_root,
            train=False,
            fold_idx=fold_idx,
            transform=val_transform,
        )
        
        # Create loaders
        train_loader = DataLoader(
            train_dataset, batch_size=128, shuffle=True,
            num_workers=0, pin_memory=True,
        )
        val_loader = DataLoader(
            val_dataset, batch_size=128, shuffle=False,
            num_workers=0, pin_memory=True,
        )
        
        # Create model
        model = create_model(num_classes=100, pretrained=False)
        model = model.to(device)
        
        # Optimizer and scheduler (same as main_phase_d.py)
        criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
        optimizer, scheduler = get_optimizer_and_scheduler(
            model=model,
            total_epochs=epochs,
            lr=0.1,
            weight_decay=1e-2,  # Same as Phase D
            momentum=0.9,
            warmup_epochs=5,
        )
        
        best_val_acc = 0.0
        best_epoch = 0
        
        # Training loop with progress bar
        pbar = tqdm(range(epochs), desc=f"Fold {fold_idx}", unit="epoch")
        for epoch in pbar:
            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, device
            )
            val_loss, val_acc, top5_acc = evaluate(model, val_loader, criterion, device)
            scheduler.step()
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch + 1
            
            pbar.set_postfix({
                "train": f"{train_acc:.1f}%",
                "val": f"{val_acc:.1f}%",
                "best": f"{best_val_acc:.1f}%",
            })
        
        result = {
            "method": "PhaseA_Only",
            "op_name": OP_NAME,
            "magnitude": MAGNITUDE,
            "probability": PROBABILITY,
            "fold_idx": fold_idx,
            "val_acc": best_val_acc,
            "best_epoch": best_epoch,
            "timestamp": datetime.now().isoformat(),
        }
        all_results.append(result)
        
        # Write to CSV
        with open(results_file, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=result.keys())
            if write_header:
                writer.writeheader()
                write_header = False
            writer.writerow(result)
        
        print(f"  Fold {fold_idx}: Best Val Acc = {best_val_acc:.2f}% (epoch {best_epoch})")
    
    # Summary
    accs = [r["val_acc"] for r in all_results]
    print(f"\n{'='*60}")
    print(f"Phase A Only Summary:")
    print(f"  Mean: {np.mean(accs):.2f}% ± {np.std(accs):.2f}%")
    print(f"  Min:  {np.min(accs):.2f}%")
    print(f"  Results saved to: {results_file}")
    print(f"{'='*60}")
    
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--folds", type=str, default=None, 
                        help="Comma-separated fold indices, e.g., '0,1,2'")
    parser.add_argument("--output_dir", type=str, default="outputs")
    args = parser.parse_args()
    
    folds = None
    if args.folds:
        folds = [int(f) for f in args.folds.split(",")]
    
    run_phase_a_only_evaluation(
        epochs=args.epochs,
        folds=folds,
        output_dir=args.output_dir,
    )
