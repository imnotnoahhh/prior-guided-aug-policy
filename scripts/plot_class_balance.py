#!/usr/bin/env python
"""
Class balance check for CIFAR-100 splits.

Uses CIFAR100Subsampled to compute train/val class counts and plots histograms.

Usage:
    python scripts/plot_class_balance.py --data_root ./data --fold_idx 0 --out_dir outputs/figures
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

from src.dataset import CIFAR100Subsampled


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot class balance for CIFAR100Subsampled")
    parser.add_argument("--data_root", type=Path, default=Path("./data"), help="Root path for CIFAR-100 data")
    parser.add_argument("--fold_idx", type=int, default=0, help="Fold index")
    parser.add_argument("--out_dir", type=Path, default=Path("outputs/figures"), help="Output directory")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    train_ds = CIFAR100Subsampled(root=args.data_root, train=True, fold_idx=args.fold_idx, download=True)
    val_ds = CIFAR100Subsampled(root=args.data_root, train=False, fold_idx=args.fold_idx, download=True)

    def get_counts(ds):
        dist = ds.get_class_distribution()
        return [dist[i] for i in sorted(dist.keys())]

    train_counts = get_counts(train_ds)
    val_counts = get_counts(val_ds)

    plt.figure(figsize=(9, 4))
    sns.barplot(x=list(range(100)), y=train_counts, color="#4C72B0")
    plt.title(f"Train class counts (fold {args.fold_idx})")
    plt.xlabel("Class id")
    plt.ylabel("Count")
    plt.tight_layout()
    out_train = args.out_dir / f"class_balance_train_fold{args.fold_idx}.png"
    plt.savefig(out_train, dpi=200)
    plt.close()

    plt.figure(figsize=(9, 4))
    sns.barplot(x=list(range(100)), y=val_counts, color="#DD8452")
    plt.title(f"Val class counts (fold {args.fold_idx})")
    plt.xlabel("Class id")
    plt.ylabel("Count")
    plt.tight_layout()
    out_val = args.out_dir / f"class_balance_val_fold{args.fold_idx}.png"
    plt.savefig(out_val, dpi=200)
    plt.close()

    print(f"Saved {out_train}")
    print(f"Saved {out_val}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
