#!/usr/bin/env python
"""
Parse training logs to plot val_acc vs epoch (convergence).

Expects log lines containing "Epoch" and "val_acc", e.g.:
    Epoch  10: train_acc=..., val_acc=...

Usage:
    python scripts/plot_convergence_from_logs.py --logs logs/baseline.log logs/phase_c.log --labels Baseline Ours --out_dir outputs/figures
"""

import argparse
import re
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt

epoch_re = re.compile(r"Epoch\\s+(\\d+).*?val_acc=([0-9.]+)")


def parse_log(path: Path) -> List[tuple]:
    points = []
    with open(path, "r") as f:
        for line in f:
            m = epoch_re.search(line)
            if m:
                epoch = int(m.group(1))
                acc = float(m.group(2))
                points.append((epoch, acc))
    return points


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot convergence curves from logs")
    parser.add_argument("--logs", type=Path, nargs="+", required=True, help="Log files to parse")
    parser.add_argument("--labels", type=str, nargs="+", help="Labels for each log (same order as logs)")
    parser.add_argument("--out_dir", type=Path, default=Path("outputs/figures"), help="Output directory")
    args = parser.parse_args()

    if args.labels and len(args.labels) != len(args.logs):
        raise ValueError("labels length must match logs length")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    for idx, log_path in enumerate(args.logs):
        pts = parse_log(log_path)
        if not pts:
            print(f"Warning: no epoch/val_acc lines found in {log_path}")
            continue
        pts = sorted(pts, key=lambda x: x[0])
        epochs, accs = zip(*pts)
        label = args.labels[idx] if args.labels else log_path.stem
        plt.plot(epochs, accs, marker="o", label=label)

    plt.xlabel("Epoch")
    plt.ylabel("Val Acc (%)")
    plt.title("Convergence")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out_path = args.out_dir / "convergence.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
