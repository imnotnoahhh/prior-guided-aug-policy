#!/usr/bin/env python
"""
Phase B stability plot (Polished Version).
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
    parser = argparse.ArgumentParser(description="Plot Phase B stability")
    parser.add_argument("--csv", type=Path, default=Path("outputs/phase_b_tuning_raw.csv"), help="Path to csv")
    parser.add_argument("--out_dir", type=Path, default=Path("outputs/figures"), help="Output directory")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    df = df[df["error"].isna() | (df["error"] == "")]
    
    # Filter to final rung only if data exists
    final_rung = df[df["epochs_run"] >= 200]
    if final_rung.empty:
        # Fallback to whatever is max epochs if incomplete run
        max_epochs = df["epochs_run"].max()
        final_rung = df[df["epochs_run"] == max_epochs]
        print(f"Warning: No full 200-epoch runs found. Plotting max found epoch ({max_epochs}).")

    if final_rung.empty:
        print("Error: No valid Phase B data to plot.")
        return 0

    args.out_dir.mkdir(parents=True, exist_ok=True)
    
    # Sort by median acc for cleaner look
    order = final_rung.groupby("op_name")["val_acc"].median().sort_values(ascending=False).index

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=final_rung, x="op_name", y="val_acc", order=order, 
                palette="viridis", hue="op_name", legend=False)
    sns.stripplot(data=final_rung, x="op_name", y="val_acc", order=order, 
                  color="black", alpha=0.4, jitter=0.2)
    
    plt.xticks(rotation=45, ha="right", fontsize=11)
    plt.ylabel("Validation Accuracy (%)", fontsize=12)
    plt.xlabel("Operation Name", fontsize=12)
    plt.title("Stability Analysis: Single-Op Performance Distribution", fontsize=14, fontweight='bold')
    
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    out_path = args.out_dir / "phase_b_stability.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved {out_path}")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
