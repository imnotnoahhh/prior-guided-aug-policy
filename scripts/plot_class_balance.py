#!/usr/bin/env python
"""
Class balance check for CIFAR-100 splits (Polished Version).
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import seaborn as sns
from src.dataset import CIFAR100Subsampled

# Style
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

def main() -> int:
    parser = argparse.ArgumentParser(description="Plot class balance")
    parser.add_argument("--data_root", type=Path, default=Path("./data"), help="Root path")
    parser.add_argument("--fold_idx", type=int, default=0, help="Fold index")
    parser.add_argument("--out_dir", type=Path, default=Path("outputs/figures"), help="Output directory")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading datasets...")
    train_ds = CIFAR100Subsampled(root=args.data_root, train=True, fold_idx=args.fold_idx, download=True)
    val_ds = CIFAR100Subsampled(root=args.data_root, train=False, fold_idx=args.fold_idx, download=True)

    def get_counts(ds):
        dist = ds.get_class_distribution()
        return [dist[i] for i in sorted(dist.keys())]

    train_counts = get_counts(train_ds)
    val_counts = get_counts(val_ds)

    # Plot Train
    plt.figure(figsize=(10, 5))
    sns.barplot(x=list(range(100)), y=train_counts, color="#4C72B0", edgecolor="none")
    plt.axhline(y=90, color='#C44E52', linestyle='--', linewidth=2, label='Target (90 samples)')
    
    plt.title(f"Train Set Class Distribution (Fold {args.fold_idx})", fontsize=14, fontweight='bold')
    plt.xlabel("Class ID (0-99)", fontsize=12)
    plt.ylabel("Number of Samples", fontsize=12)
    plt.legend(loc='upper right', frameon=True)
    
    # Clean up x-axis ticks (too many for 100 classes)
    plt.xticks(range(0, 101, 5), fontsize=10) # Show every 5th label
    
    plt.tight_layout()
    out_train = args.out_dir / f"class_balance_train_fold{args.fold_idx}.png"
    plt.savefig(out_train, dpi=300)
    plt.close()

    # Plot Val
    plt.figure(figsize=(10, 5))
    sns.barplot(x=list(range(100)), y=val_counts, color="#DD8452", edgecolor="none")
    plt.axhline(y=10, color='#C44E52', linestyle='--', linewidth=2, label='Target (10 samples)')
    
    plt.title(f"Validation Set Class Distribution (Fold {args.fold_idx})", fontsize=14, fontweight='bold')
    plt.xlabel("Class ID (0-99)", fontsize=12)
    plt.ylabel("Number of Samples", fontsize=12)
    plt.legend(loc='upper right', frameon=True)
    
    plt.xticks(range(0, 101, 5), fontsize=10)
    
    plt.tight_layout()
    out_val = args.out_dir / f"class_balance_val_fold{args.fold_idx}.png"
    plt.savefig(out_val, dpi=300)
    plt.close()

    print(f"Saved {out_train}")
    print(f"Saved {out_val}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
