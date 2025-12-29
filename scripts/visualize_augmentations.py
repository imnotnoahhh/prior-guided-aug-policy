#!/usr/bin/env python
"""
Visualize ALL available augmentation effects (Individual & Grid).

Generates:
1. Individual images for each augmentation (Max Mag, Prob=1.0).
2. A summary grid image.
3. Visualization of the Base Augmentation (Crop+Flip).

Usage:
    python scripts/visualize_augmentations.py --sample_idx 123
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import torch
from torchvision import transforms

from src.augmentations import AugmentationSpace, build_transform_with_op
from src.dataset import CIFAR100Subsampled

# Style settings
sns.set_theme(style="white", context="paper")

def save_single_image(img_tensor, title, out_path):
    """Save a single tensor image to path."""
    # Convert tensor to PIL
    if isinstance(img_tensor, torch.Tensor):
        img = img_tensor.permute(1, 2, 0).clamp(0, 1).numpy()
    else:
        img = img_tensor
        
    plt.figure(figsize=(3, 3))
    plt.imshow(img)
    plt.axis("off")
    # plt.title(title, fontsize=12, fontweight='bold') # Disable title on individual images per requestle on individual images per request
    plt.tight_layout(pad=0)
    plt.savefig(out_path, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()

def main() -> int:
    parser = argparse.ArgumentParser(description="Visualize all augmentations")
    parser.add_argument("--out_dir", type=Path, default=Path("outputs/figures/augment_examples"), help="Directory to save figures")
    parser.add_argument("--data_root", type=Path, default=Path("./data"), help="CIFAR-100 data root")
    parser.add_argument("--fold_idx", type=int, default=0, help="Fold index for sampling")
    parser.add_argument("--sample_idx", type=int, default=999, help="Index of the image to use (pick one that looks good)")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading dataset (Fold {args.fold_idx})...")
    ds = CIFAR100Subsampled(root=args.data_root, train=False, fold_idx=args.fold_idx, download=True)
    
    # Get raw PIL image
    pil_img, label_idx = ds[args.sample_idx]
    
    # Try to get class name
    class_name = str(label_idx)
    if hasattr(ds, "full_dataset") and hasattr(ds.full_dataset, "classes"):
        class_name = ds.full_dataset.classes[label_idx]

    print(f"Sample Index: {args.sample_idx}, Class: {class_name}")

    # 1. Save Original
    save_single_image(transforms.ToTensor()(pil_img), f"Original\n({class_name})", args.out_dir / "00_Original.png")
    print("Saved 00_Original.png")
    
    # 2. Base Augmentation (RandomCrop + HorizontalFlip)
    # We simulate this by applying the pytorch transform manually
    base_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=1.0), # Force flip for visibility
        transforms.ToTensor()
    ])
    base_img = base_tf(pil_img)
    save_single_image(base_img, "Base Augmentation", args.out_dir / "01_BaseAugment.png")
    print("Saved 01_BaseAugment.png")

    # 3. Policy Operations
    available_ops = AugmentationSpace.get_available_ops()
    op_images = []
    
    for i, op_name in enumerate(sorted(available_ops)):
        # Build transform with max effect
        transform = build_transform_with_op(op_name, magnitude=1.0, probability=1.0)
        
        try:
            # Apply transform
            # Note: build_transform_with_op returns a pipeline that expects PIL and returns Tensor
            aug_img_tensor = transform(pil_img)
            
            # Save individual
            safe_name = op_name.replace(" ", "_")
            out_name = f"02_Op_{safe_name}.png"
            save_single_image(aug_img_tensor, op_name, args.out_dir / out_name)
            
            op_images.append((op_name, aug_img_tensor))
            print(f"Saved {out_name}")
            
        except Exception as e:
            print(f"Skipping {op_name}: {e}")

    # 4. Generate Summary Grid
    print("Generating summary grid...")
    n = len(op_images) + 2 # Orig + Base
    cols = 6
    rows = (n + cols - 1) // cols
    
    fig = plt.figure(figsize=(2.5 * cols, 2.8 * rows))
    
    # Original
    plt.subplot(rows, cols, 1)
    plt.imshow(transforms.ToTensor()(pil_img).permute(1, 2, 0))
    plt.title("Original", fontsize=11, fontweight='bold')
    plt.axis("off")
    
    # Base
    plt.subplot(rows, cols, 2)
    plt.imshow(base_img.permute(1, 2, 0))
    plt.title("Base (Crop+Flip)", fontsize=11, fontweight='bold')
    plt.axis("off")
    
    # Ops
    for i, (name, img_tensor) in enumerate(op_images):
        plt.subplot(rows, cols, i + 3)
        plt.imshow(img_tensor.permute(1, 2, 0))
        plt.title(name, fontsize=10)
        plt.axis("off")
        
    # Add Class Name / Sample ID to bottom right (Footer)
    footer_text = f"Sample #{args.sample_idx} ({class_name})"
    fig.text(0.98, 0.01, footer_text, 
             horizontalalignment='right', 
             verticalalignment='bottom',
             fontsize=12, color='gray', style='italic')

    plt.tight_layout()
    grid_path = args.out_dir / "all_augmentations_grid.png"
    plt.savefig(grid_path, dpi=200)
    plt.close()
    print(f"Saved Grid: {grid_path}")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
