#!/usr/bin/env python
"""
Phase D performance and ablation plots (Polished Version).
"""

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

def main() -> int:
    parser = argparse.ArgumentParser(description="Plot Phase D")
    parser.add_argument("--csv", type=Path, default=Path("outputs/phase_d_summary.csv"), help="Path to csv")
    parser.add_argument("--out_dir", type=Path, default=Path("outputs/figures"), help="Output directory")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    # Order
    order = ["Baseline", "Baseline-NoAug", "RandAugment", "Cutout", "Best_SingleOp", "Ours_p1", "Ours_optimal"]
    # Filter to only existing
    existing = set(df["method"].unique())
    valid_order = [o for o in order if o in existing]
    
    df["method"] = pd.Categorical(df["method"], categories=valid_order, ordered=True)
    df = df.sort_values("method")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))
    
    # Use a distinct palette
    colors = sns.color_palette("muted", n_colors=len(valid_order))
    
    # Bar plot with error bars
    bar = plt.bar(df["method"], df["mean_val_acc"], yerr=df["std_val_acc"], 
            capsize=6, color=colors, alpha=0.9, ecolor='black')

    plt.ylabel("Top-1 Accuracy (%)", fontsize=12)
    plt.title("Method Comparison (Test Set)", fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha="right", fontsize=11)
    
    # Add value labels on top
    for rect in bar:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width()/2.0, height + 0.5, 
                 f'{height:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.ylim(bottom=30) # Zoom in a bit since acc is usually > 35
    plt.grid(axis='x', visible=False) # Only y grid needed
    
    plt.tight_layout()
    out_fig = args.out_dir / "phase_d_ablation.png"
    plt.savefig(out_fig, dpi=300)
    plt.close()
    print(f"Saved {out_fig}")

    # Generate Markdown Table (Preserve functionality)
    table_path = args.out_dir / "phase_d_table.md"
    with open(table_path, "w") as f:
        f.write("### Phase D Quantitative Results\n\n")
        f.write("| Method | Mean Acc | Std Dev | Top-5 Mean | Top-5 Std |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: |\n")
        for _, row in df.iterrows():
            f.write(
                f"| **{row['method']}** | {row['mean_val_acc']:.2f}% | {row['std_val_acc']:.2f} | "
                f"{row['mean_top5_acc']:.2f}% | {row['std_top5_acc']:.2f} |\n"
            )
    print(f"Saved {table_path}")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
