#!/usr/bin/env python
"""
Parse training logs to plot val_acc vs epoch (convergence).
(Polished Version)
"""

import argparse
import re
import sys
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

epoch_re = re.compile(r"Epoch\s+(\d+).*?val_acc=([0-9.]+)")

def parse_log(path: Path) -> List[tuple]:
    points = []
    if not path.exists():
        return []
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                m = epoch_re.search(line)
                if m:
                    epoch = int(m.group(1))
                    acc = float(m.group(2))
                    points.append((epoch, acc))
    except Exception as e:
        print(f"Error reading {path}: {e}")
    return points

def main() -> int:
    parser = argparse.ArgumentParser(description="Plot convergence curves from logs")
    parser.add_argument("--logs", type=Path, nargs="+", help="Log files to parse")
    parser.add_argument("--labels", type=str, nargs="+", help="Labels for each log")
    parser.add_argument("--out_dir", type=Path, default=Path("outputs/figures"), help="Output directory")
    args = parser.parse_args()

    # If no logs provided, create a dummy placeholder note instead of crashing
    if not args.logs:
        print("Note: No logs provided to plot_convergence. Skipping.")
        with open(args.out_dir / "convergence_note.txt", "w") as f:
            f.write("Convergence plot skipped: Detailed logs not available in this run.")
        return 0

    args.out_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(9, 6))
    
    valid_plots = 0
    for idx, log_path in enumerate(args.logs):
        pts = parse_log(log_path)
        if not pts:
            print(f"Warning: no data or file not found for {log_path}")
            continue
        
        valid_plots += 1
        pts = sorted(pts, key=lambda x: x[0])
        epochs, accs = zip(*pts)
        label = args.labels[idx] if (args.labels and idx < len(args.labels)) else log_path.stem
        
        plt.plot(epochs, accs, linewidth=2, label=label)

    if valid_plots == 0:
        print("No valid data found in any log files. Skipping plot.")
        return 0

    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Validation Accuracy (%)", fontsize=12)
    plt.title("Convergence Analysis", fontsize=14, fontweight='bold')
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    
    out_path = args.out_dir / "convergence.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved {out_path}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
