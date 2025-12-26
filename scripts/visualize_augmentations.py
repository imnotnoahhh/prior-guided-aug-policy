#!/usr/bin/env python
"""
Visualize augmentation effects using the final policy.

Loads phase_c_final_policy.json and applies adjusted probabilities to a few CIFAR-100 samples,
saving before/after grids.

Usage:
    python scripts/visualize_augmentations.py --policy outputs/phase_c_final_policy.json --out_dir outputs/figures/augment_examples
"""

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import torch
from torchvision import transforms

from src.augmentations import build_transform_with_ops, get_val_transform
from src.dataset import CIFAR100Subsampled


def load_policy(path: Path) -> List[Tuple[str, float, float]]:
    with open(path, "r") as f:
        data = json.load(f)
    ops = data.get("ops", [])
    # Prefer adjusted probabilities if present
    result = []
    for op in ops:
        p = op.get("probability_adjusted", op.get("probability_original", op.get("probability", 1.0)))
        result.append((op["name"], float(op["magnitude"]), float(p)))
    return result


def make_grid(images):
    # images: list of CHW tensors in [0,1]
    n = len(images)
    rows = 1
    cols = n
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3))
    if cols == 1:
        axes = [axes]
    for ax, img in zip(axes, images):
        img = img.permute(1, 2, 0).clamp(0, 1).cpu()
        ax.imshow(img)
        ax.axis("off")
    fig.tight_layout()
    return fig


def main() -> int:
    parser = argparse.ArgumentParser(description="Visualize augmentations from final policy")
    parser.add_argument("--policy", type=Path, default=Path("outputs/phase_c_final_policy.json"), help="Path to policy JSON")
    parser.add_argument("--out_dir", type=Path, default=Path("outputs/figures/augment_examples"), help="Directory to save figures")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of samples to visualize")
    parser.add_argument("--data_root", type=Path, default=Path("./data"), help="CIFAR-100 data root")
    parser.add_argument("--fold_idx", type=int, default=0, help="Fold index for sampling")
    args = parser.parse_args()

    ops = load_policy(args.policy)
    if not ops:
        raise ValueError(f"No ops found in policy: {args.policy}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Use raw PIL images; we'll apply transforms manually
    ds = CIFAR100Subsampled(root=args.data_root, train=False, fold_idx=args.fold_idx, transform=None, download=True)
    aug_transform = build_transform_with_ops(ops, include_baseline=True, include_normalize=False)
    base_to_tensor = get_val_transform(include_normalize=False)

    for idx in range(args.num_samples):
        pil_img, label = ds[idx]
        orig_tensor = base_to_tensor(pil_img)
        aug_tensor = aug_transform(pil_img)
        fig = make_grid([orig_tensor, aug_tensor])
        out_path = args.out_dir / f"sample_{idx}_class{label}.png"
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        print(f"Saved {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
