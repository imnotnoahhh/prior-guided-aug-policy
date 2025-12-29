#!/usr/bin/env python
"""
Phase A (m, p) heatmap visualization (Polished Version).
"""

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="white", context="paper", font_scale=1.2)

def main() -> int:
    parser = argparse.ArgumentParser(description="Plot Phase A heatmaps")
    parser.add_argument("--csv", type=Path, default=Path("outputs/phase_a_results.csv"), help="Path to csv")
    parser.add_argument("--metric", type=str, default=None, help="Metric to plot")
    parser.add_argument("--out_dir", type=Path, default=Path("outputs/figures"), help="Output directory")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    metric = args.metric or ("stable_score" if "stable_score" in df.columns else "val_acc")
    
    # Check if metric exists
    if metric not in df.columns:
        print(f"Warning: Metric {metric} missing. Skipping Phase A plots.")
        return 0

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Use divergent colormap for scores centered on 0 (if stable_score)
    # or sequential for val_acc
    cmap = "RdBu_r" if metric == "stable_score" else "viridis"
    
    # Plot per-op heatmaps
    # Plot per-op heatmaps
    for op_name, sub in df.groupby("op_name"):
        
        # Conditional Layout Logic
        if op_name == "RandomGrayscale":
            # Horizontal Layout (Original): X=Probability, Y=Magnitude
            # Better for single-row data (1 magnitude, many probs)
            pivot = sub.pivot_table(index="magnitude", columns="probability", values=metric, aggfunc="mean")
            xlabel_txt, ylabel_txt = "Probability", "Magnitude"
            
            # Sort Descending so larger value is at Top (Seaborn defaults 0 to top)
            pivot = pivot.sort_index(ascending=False) 
            
            rows, cols = pivot.shape
            # Wide, Short figure
            fig_width = max(8, cols * 1.2)
            fig_height = max(4, rows * 1.5)
            
            x_rot, y_rot = 0, 0
            
        else:
            # Vertical Layout (New): X=Magnitude, Y=Probability
            # Better for standard matrices
            pivot = sub.pivot_table(index="probability", columns="magnitude", values=metric, aggfunc="mean")
            xlabel_txt, ylabel_txt = "Magnitude", "Probability"
            
            pivot = pivot.sort_index(ascending=False)
            
            rows, cols = pivot.shape
            # Tall, Narrower figure
            fig_width = max(5, cols * 1.5)
            fig_height = max(5, rows * 1.0)
            
            x_rot, y_rot = 45, 0

        if pivot.empty:
            continue
            
        plt.figure(figsize=(fig_width, fig_height))
        
        ax = sns.heatmap(pivot, cmap=cmap, 
                         annot=True, fmt=".2g", 
                         annot_kws={"size": 14, "weight": "bold"},
                         linewidths=.5, square=True, 
                         cbar_kws={"label": "Impact Score" if metric == "stable_score" else "Accuracy (%)"})
        
        # Adjust colorbar font
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=14)
        cbar.set_label(cbar.ax.get_ylabel(), fontsize=16)

        plt.title(f"{op_name}", fontsize=20, fontweight='bold', pad=20)
        plt.xlabel(xlabel_txt, fontsize=18)
        plt.ylabel(ylabel_txt, fontsize=18)
        
        plt.xticks(fontsize=14, rotation=x_rot)
        plt.yticks(fontsize=14, rotation=y_rot)
        
        out_path = args.out_dir / f"phase_a_heatmap_{op_name}.png"
        plt.savefig(out_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
        plt.close()
        print(f"Saved {out_path}")


    return 0

if __name__ == "__main__":
    raise SystemExit(main())
