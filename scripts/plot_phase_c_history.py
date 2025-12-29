#!/usr/bin/env python
"""
Plot Phase C greedy search history (Polished Version).
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
    parser = argparse.ArgumentParser(description="Plot Phase C history")
    parser.add_argument("--csv", type=Path, default=Path("outputs/phase_c_history.csv"), help="Path to csv")
    parser.add_argument("--out_dir", type=Path, default=Path("outputs/figures"), help="Output directory")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    
    # Map 'val_acc'. If 'mean_acc' missing, use 'val_acc'
    if "mean_acc" not in df.columns and "val_acc" in df.columns:
        df["mean_acc"] = df["val_acc"]

    # Check for required columns
    required = {"op_name", "mean_acc"}
    if not required.issubset(df.columns):
        # Graceful fallback if CSV is empty or malformed
        print(f"Warning: {args.csv} missing key columns. Skipping Phase C plot.")
        return 0

    args.out_dir.mkdir(parents=True, exist_ok=True)
    
    # We only care about the main history plot usually, but let's handle potential paths
    path_col = "path" if "path" in df.columns else ("path_name" if "path_name" in df.columns else None)
    
    if path_col:
        groups = df.groupby(path_col)
    else:
        # Treat entire df as one group
        groups = [("Combined", df)]

    for name, sub in groups:
        plt.figure(figsize=(10, 6))
        
        # Line plot + Markers to show trajectory
        # X-axis will be categorical op names, so we just use range(len) for line plotting
        x_indices = range(len(sub))
        
        plt.plot(x_indices, sub["mean_acc"], color="#55A868", marker="o", 
                 linewidth=2, markersize=8, label="Search Trajectory")
        
        # Highlight max point
        max_idx = sub["mean_acc"].argmax()
        max_acc = sub["mean_acc"].iloc[max_idx]
        
        plt.scatter([max_idx], [max_acc], color="#C44E52", s=150, zorder=10, 
                    label=f"Peak: {max_acc:.2f}%")

        plt.xticks(x_indices, sub["op_name"], rotation=45, ha="right", fontsize=11)
        
        plt.xlabel("Operation Added Used", fontsize=12)
        plt.ylabel("Validation Accuracy (%)", fontsize=12)
        plt.title(f"Greedy Search Trajectory (Phase C)", fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.5)
        
        plt.tight_layout()
        
        safe_name = str(name).replace(" ", "_").replace("/", "_")
        out_path = args.out_dir / "phase_c_history.png" # Just overwrite main one for now
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"Saved {out_path}")
        
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
