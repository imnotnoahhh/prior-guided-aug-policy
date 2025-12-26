#!/usr/bin/env python
"""
Phase D performance and ablation plots.

Reads outputs/phase_d_summary.csv and produces:
- Bar plot with mean_val_acc (error bars = std_val_acc)
- Markdown table saved to out_dir/phase_d_table.md

Usage:
    python scripts/plot_phase_d_ablation.py --csv outputs/phase_d_summary.csv --out_dir outputs/figures
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot Phase D performance/ablation")
    parser.add_argument("--csv", type=Path, default=Path("outputs/phase_d_summary.csv"), help="Path to phase_d_summary.csv")
    parser.add_argument("--out_dir", type=Path, default=Path("outputs/figures"), help="Directory to save figures and table")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    # Filter to key methods for ablation ordering
    order = ["Baseline", "Baseline-NoAug", "RandAugment", "Cutout", "Best_SingleOp", "Ours_p1", "Ours_optimal"]
    df["method"] = pd.Categorical(df["method"], categories=order, ordered=True)
    df = df.sort_values("method")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Bar plot
    plt.figure(figsize=(9, 5))
    plt.bar(df["method"], df["mean_val_acc"], yerr=df["std_val_acc"], capsize=4, color="#4C72B0")
    plt.ylabel("Top-1 Mean (%)")
    plt.title("Phase D Performance")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    out_fig = args.out_dir / "phase_d_ablation.png"
    plt.savefig(out_fig, dpi=200)
    plt.close()
    print(f"Saved {out_fig}")

    # Markdown table
    table_path = args.out_dir / "phase_d_table.md"
    with open(table_path, "w") as f:
        f.write("| Method | Top-1 Mean | Top-1 Std | Top-5 Mean | Top-5 Std |\n")
        f.write("|--------|-----------:|----------:|-----------:|----------:|\n")
        for _, row in df.iterrows():
            f.write(
                f"| {row['method']} | {row['mean_val_acc']:.2f} | {row['std_val_acc']:.2f} | "
                f"{row['mean_top5_acc']:.2f} | {row['std_top5_acc']:.2f} |\n"
            )
    print(f"Saved {table_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
