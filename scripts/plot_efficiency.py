#!/usr/bin/env python
"""
Efficiency/cost plot (Polished Version).
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
    parser = argparse.ArgumentParser(description="Plot efficiency")
    parser.add_argument("--csv", type=Path, default=Path("outputs/phase_b_tuning_raw.csv"), help="Path to csv")
    parser.add_argument("--out_dir", type=Path, default=Path("outputs/figures"), help="Output directory")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    df = df[df["error"].isna() | (df["error"] == "")]

    # Sort by runtime and compute best-so-far curve
    df_sorted = df.sort_values("runtime_sec").reset_index(drop=True)
    df_sorted["cumulative_runtime"] = df_sorted["runtime_sec"].cumsum() / 3600.0  # hours
    df_sorted["best_val_acc"] = df_sorted["val_acc"].cummax()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 6))
    
    # Plot the raw points first (scatter) to show individual trials
    plt.scatter(df_sorted["cumulative_runtime"], df_sorted["val_acc"], 
                alpha=0.3, color="gray", s=20, label="Individual Trials")
    
    # Plot the best-so-far curve prominently
    plt.plot(df_sorted["cumulative_runtime"], df_sorted["best_val_acc"], 
             color="#C44E52", linewidth=2.5, marker="o", markersize=4, label="Best-so-far Acc")

    plt.xlabel("Cumulative Search Time (Hours)", fontsize=12)
    plt.ylabel("Validation Accuracy (%)", fontsize=12)
    plt.title("Search Efficiency Curve (Phase B)", fontsize=14, fontweight='bold')
    
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(frameon=True)
    
    plt.tight_layout()
    out_path = args.out_dir / "efficiency_curve.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    
    print(f"Saved {out_path}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
