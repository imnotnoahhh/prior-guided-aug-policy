#!/usr/bin/env python
"""
Plot Phase C greedy history.

Expects phase_c_history.csv with at least columns:
 - op_name (candidate op)
 - mean_acc (after adding op)
 - path (optional) or path_name
If columns differ, the script will report and exit.

Usage:
    python scripts/plot_phase_c_history.py --csv outputs/phase_c_history.csv --out_dir outputs/figures
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot Phase C greedy history")
    parser.add_argument("--csv", type=Path, default=Path("outputs/phase_c_history.csv"), help="Path to phase_c_history.csv")
    parser.add_argument("--out_dir", type=Path, default=Path("outputs/figures"), help="Output directory")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    required = {"op_name", "mean_acc"}
    if not required.issubset(df.columns):
        raise ValueError(f"phase_c_history.csv missing required columns {required}, found {df.columns.tolist()}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    path_col = "path" if "path" in df.columns else ("path_name" if "path_name" in df.columns else None)
    if path_col:
        for name, sub in df.groupby(path_col):
            plt.figure(figsize=(8, 4))
            sns.barplot(x=sub["op_name"], y=sub["mean_acc"], color="#4C72B0")
            plt.xticks(rotation=30, ha="right")
            plt.ylabel("Mean Acc (%)")
            plt.title(f"Phase C path: {name}")
            plt.tight_layout()
            out_path = args.out_dir / f"phase_c_history_{name}.png"
            plt.savefig(out_path, dpi=200)
            plt.close()
            print(f"Saved {out_path}")
    else:
        plt.figure(figsize=(8, 4))
        sns.barplot(x=df["op_name"], y=df["mean_acc"], color="#4C72B0")
        plt.xticks(rotation=30, ha="right")
        plt.ylabel("Mean Acc (%)")
        plt.title("Phase C history")
        plt.tight_layout()
        out_path = args.out_dir / "phase_c_history.png"
        plt.savefig(out_path, dpi=200)
        plt.close()
        print(f"Saved {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
