#!/usr/bin/env python
"""
Comprehensive Phase D analysis plots (Polished Version).
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
    parser = argparse.ArgumentParser(description="Comprehensive Phase D Plots")
    parser.add_argument("--csv", type=Path, default=Path("outputs/phase_d_results.csv"), help="Path to csv")
    parser.add_argument("--out_dir", type=Path, default=Path("outputs/figures"), help="Output directory")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    
    name_map = {
        "Baseline": "ResNet18\n(Baseline)",
        "Baseline-NoAug": "No Aug",
        "RandAugment": "RandAugment\n(SOTA)",
        "Cutout": "Cutout",
        "Best_SingleOp": "Best Single\n(Simple)",
        "Ours_optimal": "Ours\n(Prior-Guided)"
    }

    df = pd.read_csv(args.csv)
    df["display_name"] = df["op_name"].map(lambda x: name_map.get(x, x))
    
    # Priority Order for plots
    order = ["No Aug", "ResNet18\n(Baseline)", "Cutout", "RandAugment\n(SOTA)", "Best Single\n(Simple)", "Ours\n(Prior-Guided)"]
    df = df[df["display_name"].isin(order)]

    # 1. Robustness (Box Plot)
    plt.figure(figsize=(11, 7))
    
    # Boxplot
    sns.boxplot(data=df, x="display_name", y="val_acc", order=order, 
                hue="display_name", palette="Set2", legend=False, width=0.6)
    
    # Swarmplot overlay
    sns.swarmplot(data=df, x="display_name", y="val_acc", order=order, 
                  color=".25", size=5, alpha=0.8)
    
    plt.title("Robustness Analysis: Validation Accuracy Distribution (5 Folds)", fontsize=16, fontweight='bold', pad=20)
    plt.ylabel("Validation Accuracy (%)", fontsize=13)
    plt.xlabel("")
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    out_robust = args.out_dir / "phase_d_robustness.png"
    plt.savefig(out_robust, dpi=300)
    plt.close()
    print(f"Saved {out_robust}")

    # 2. Overfitting Analysis (Generalization Gap)
    df["gen_gap"] = df["train_acc"] - df["val_acc"]
    
    plt.figure(figsize=(11, 7))
    sns.barplot(data=df, x="display_name", y="gen_gap", order=order, 
                hue="display_name", palette="Set2", legend=False, 
                errorbar="sd", capsize=.15)
    
    plt.title("Generalization Gap (Training Acc - Validation Acc)", fontsize=16, fontweight='bold', pad=20)
    plt.ylabel("Gap Percentage (%) (Lower is Better)", fontsize=13)
    plt.xlabel("")
    plt.axhline(y=0, color='black', linewidth=1)
    
    # Add annotation for 'Ours'
    # plt.text(...) # Optional

    plt.tight_layout()
    out_gap = args.out_dir / "phase_d_overfitting.png"
    plt.savefig(out_gap, dpi=300)
    plt.close()
    print(f"Saved {out_gap}")
    
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
