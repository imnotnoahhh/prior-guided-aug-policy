#!/usr/bin/env python
"""
Efficiency/cost plot using runtime vs val_acc.

This script reads Phase B raw results as a proxy for search cost and accuracy.
It plots best-so-far val_acc against cumulative runtime_sec (sorted by runtime)
to illustrate efficiency gains.

Usage:
    python scripts/plot_efficiency.py --csv outputs/phase_b_tuning_raw.csv --out_dir outputs/figures
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot efficiency (runtime vs best val_acc)")
    parser.add_argument("--csv", type=Path, default=Path("outputs/phase_b_tuning_raw.csv"), help="Path to phase_b_tuning_raw.csv")
    parser.add_argument("--out_dir", type=Path, default=Path("outputs/figures"), help="Output directory")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    df = df[df["error"].isna() | (df["error"] == "")]

    # Sort by runtime and compute best-so-far curve
    df_sorted = df.sort_values("runtime_sec").reset_index(drop=True)
    df_sorted["cumulative_runtime"] = df_sorted["runtime_sec"].cumsum() / 3600.0  # hours
    df_sorted["best_val_acc"] = df_sorted["val_acc"].cummax()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(df_sorted["cumulative_runtime"], df_sorted["best_val_acc"], marker="o", linewidth=1.5)
    plt.xlabel("Cumulative runtime (hours)")
    plt.ylabel("Best-so-far val_acc (%)")
    plt.title("Efficiency Curve (Phase B search)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = args.out_dir / "efficiency_curve.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
