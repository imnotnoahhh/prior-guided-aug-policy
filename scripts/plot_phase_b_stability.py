#!/usr/bin/env python
"""
Phase B stability plot (final rung multi-seed).

Reads outputs/phase_b_tuning_raw.csv, filters to epochs_run >= 200 (final rung),
and plots val_acc distributions per op_name.

Usage:
    python scripts/plot_phase_b_stability.py --csv outputs/phase_b_tuning_raw.csv --out_dir outputs/figures
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot Phase B stability (multi-seed distributions)")
    parser.add_argument("--csv", type=Path, default=Path("outputs/phase_b_tuning_raw.csv"), help="Path to phase_b_tuning_raw.csv")
    parser.add_argument("--out_dir", type=Path, default=Path("outputs/figures"), help="Output directory for figures")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    df = df[df["error"].isna() | (df["error"] == "")]
    df = df[df["epochs_run"] >= 200]  # final rung
    if df.empty:
        raise ValueError("No final rung entries (epochs_run >= 200) found in CSV.")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(9, 5))
    sns.boxplot(data=df, x="op_name", y="val_acc")
    sns.stripplot(data=df, x="op_name", y="val_acc", color="black", alpha=0.4, jitter=0.2)
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Val Acc (%)")
    plt.title("Phase B Final Rung Stability (multi-seed)")
    plt.tight_layout()
    out_path = args.out_dir / "phase_b_stability.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
