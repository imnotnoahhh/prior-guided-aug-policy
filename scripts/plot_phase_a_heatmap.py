#!/usr/bin/env python
"""
Phase A (m, p) heatmap visualization.

Reads outputs/phase_a_results.csv and plots magnitude vs probability heatmap
using stable_score if available, otherwise val_acc.

Usage:
    python scripts/plot_phase_a_heatmap.py --csv outputs/phase_a_results.csv --out_dir outputs/figures
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot Phase A (m,p) heatmap")
    parser.add_argument("--csv", type=Path, default=Path("outputs/phase_a_results.csv"), help="Path to phase_a_results.csv")
    parser.add_argument("--metric", type=str, default=None, help="Metric to plot (stable_score or val_acc). Default: stable_score if present.")
    parser.add_argument("--out_dir", type=Path, default=Path("outputs/figures"), help="Output directory for figures")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    metric = args.metric or ("stable_score" if "stable_score" in df.columns else "val_acc")
    if metric not in df.columns:
        raise ValueError(f"Metric {metric} not found in CSV columns: {df.columns.tolist()}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Plot per-op heatmaps
    for op_name, sub in df.groupby("op_name"):
        pivot = sub.pivot_table(index="magnitude", columns="probability", values=metric, aggfunc="mean")
        if pivot.empty:
            continue
        plt.figure(figsize=(8, 6))
        sns.heatmap(pivot.sort_index(ascending=True), cmap="viridis", annot=False, cbar_kws={"label": metric})
        plt.title(f"{op_name} {metric}")
        plt.xlabel("probability")
        plt.ylabel("magnitude")
        out_path = args.out_dir / f"phase_a_heatmap_{op_name}.png"
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        plt.close()
        print(f"Saved {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
