#!/usr/bin/env python
"""
Visualize Failure Cases: Compare RandAugment vs SAS augmentation effects.

Protocol (fixed to avoid cherry-picking):
- Randomly sample N=10 images from validation set, seed=42
- For each image: Original → RandAugment (2 samples) → SAS (1 sample)
- Annotate: prediction, confidence, SSIM value
- Uses baseline_best.pth model for predictions

Usage:
    python scripts/visualize_failure_cases.py
    python scripts/visualize_failure_cases.py --n_samples 5
"""

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from torchvision import transforms
from skimage.metrics import structural_similarity as ssim

from src.augmentations import (
    get_baseline_transform,
    get_randaugment_transform,
    build_transform_with_op,
)
from src.dataset import CIFAR100Subsampled
from src.models import create_model

# Style
sns.set_theme(style="white", context="paper")

# CIFAR-100 class names (subset for display)
CIFAR100_CLASSES = None  # Will be loaded from dataset


def compute_ssim(img1: torch.Tensor, img2: torch.Tensor) -> float:
    """Compute SSIM between two tensor images."""
    # Convert to numpy, shape (H, W, C)
    img1_np = img1.permute(1, 2, 0).numpy()
    img2_np = img2.permute(1, 2, 0).numpy()
    
    # SSIM expects values in [0, 1]
    img1_np = np.clip(img1_np, 0, 1)
    img2_np = np.clip(img2_np, 0, 1)
    
    # Compute SSIM (multichannel for RGB)
    return ssim(img1_np, img2_np, channel_axis=2, data_range=1.0)


def load_model(checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    """Load a trained model from checkpoint."""
    model = create_model(num_classes=100, pretrained=False)
    
    if checkpoint_path.exists():
        state_dict = torch.load(checkpoint_path, map_location=device)
        # Handle different checkpoint formats
        if "model_state_dict" in state_dict:
            model.load_state_dict(state_dict["model_state_dict"])
        else:
            model.load_state_dict(state_dict)
        print(f"Loaded model from {checkpoint_path}")
    else:
        print(f"Warning: Checkpoint not found at {checkpoint_path}, using random weights")
    
    model = model.to(device)
    model.eval()
    return model


def predict(model: torch.nn.Module, img_tensor: torch.Tensor, device: torch.device):
    """Get prediction and confidence for an image."""
    with torch.no_grad():
        img = img_tensor.unsqueeze(0).to(device)
        logits = model(img)
        probs = F.softmax(logits, dim=1)
        confidence, pred_class = probs.max(dim=1)
        return pred_class.item(), confidence.item()


def create_augmented_samples(pil_img, sas_config):
    """Create augmented samples for visualization."""
    # ToTensor for original
    to_tensor = transforms.ToTensor()
    original = to_tensor(pil_img)
    
    # RandAugment transform (n=2, m=9 is standard)
    ra_transform = get_randaugment_transform(n=2, m=9, include_baseline=True, include_normalize=False)
    
    # SAS transform
    op_name, magnitude, probability = sas_config
    sas_transform = build_transform_with_op(
        op_name, magnitude, probability,
        include_baseline=True, include_normalize=False
    )
    
    # Generate samples
    ra_sample1 = ra_transform(pil_img)
    ra_sample2 = ra_transform(pil_img)
    sas_sample = sas_transform(pil_img)
    
    return {
        "original": original,
        "ra1": ra_sample1,
        "ra2": ra_sample2,
        "sas": sas_sample,
    }


def visualize_single_image(
    samples: dict,
    true_label: int,
    class_names: list,
    model=None,
    device=None,
    ax_row=None,
):
    """Visualize a single image row with annotations."""
    keys = ["original", "ra1", "ra2", "sas"]
    titles = ["Original", "RA Sample 1", "RA Sample 2", "SAS"]
    
    original = samples["original"]
    
    for i, (key, title) in enumerate(zip(keys, titles)):
        ax = ax_row[i]
        img = samples[key]
        
        # Display image
        ax.imshow(img.permute(1, 2, 0).clamp(0, 1).numpy())
        ax.axis("off")
        
        # Compute SSIM (skip for original)
        if key == "original":
            ssim_val = 1.0
        else:
            ssim_val = compute_ssim(original, img)
        
        # Get prediction if model is available
        if model is not None and device is not None:
            pred_class, confidence = predict(model, img, device)
            pred_name = class_names[pred_class] if class_names else str(pred_class)
            true_name = class_names[true_label] if class_names else str(true_label)
            
            # Color: green if correct, red if wrong
            color = "green" if pred_class == true_label else "red"
            
            if key == "original":
                label = f"{title}\n{true_name}"
            else:
                label = f"{title}\nSSIM: {ssim_val:.2f}\n{pred_name} ({confidence:.0%})"
                ax.set_title(label, fontsize=8, color=color)
                continue
        else:
            if key == "original":
                true_name = class_names[true_label] if class_names else str(true_label)
                label = f"{title}\n{true_name}"
            else:
                label = f"{title}\nSSIM: {ssim_val:.2f}"
        
        ax.set_title(label, fontsize=8)


def main():
    parser = argparse.ArgumentParser(description="Visualize Failure Cases")
    parser.add_argument("--n_samples", type=int, default=10, help="Number of samples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    parser.add_argument("--checkpoint", type=str, default="outputs/checkpoints/baseline_best.pth",
                        help="Path to model checkpoint")
    # Phase B optimal SAS parameters as defaults
    parser.add_argument("--sas_magnitude", type=float, default=0.2575, help="SAS magnitude")
    parser.add_argument("--sas_probability", type=float, default=0.4239, help="SAS probability")
    parser.add_argument("--output_dir", type=str, default="outputs/figures")
    args = parser.parse_args()
    
    # Set seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # SAS config
    sas_config = ("ColorJitter", args.sas_magnitude, args.sas_probability)
    print(f"SAS config: {sas_config}")
    
    # Load dataset (validation split)
    print("Loading validation dataset...")
    ds = CIFAR100Subsampled(root="./data", train=False, fold_idx=0, download=True)
    
    # Get class names
    global CIFAR100_CLASSES
    if hasattr(ds, "full_dataset") and hasattr(ds.full_dataset, "classes"):
        CIFAR100_CLASSES = ds.full_dataset.classes
    else:
        CIFAR100_CLASSES = [str(i) for i in range(100)]
    
    # Random sample indices
    n_total = len(ds)
    sample_indices = np.random.choice(n_total, size=args.n_samples, replace=False)
    print(f"Selected {args.n_samples} samples: {sample_indices.tolist()}")
    
    # Load model (always use model for predictions)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(Path(args.checkpoint), device)
    
    # Create figure
    n_cols = 4  # Original, RA1, RA2, SAS
    n_rows = args.n_samples
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.5, n_rows * 2.5))
    
    if n_rows == 1:
        axes = [axes]
    
    # Process each sample
    for row_idx, sample_idx in enumerate(sample_indices):
        pil_img, true_label = ds[sample_idx]
        
        # Create augmented samples
        samples = create_augmented_samples(pil_img, sas_config)
        
        # Visualize
        visualize_single_image(
            samples=samples,
            true_label=true_label,
            class_names=CIFAR100_CLASSES,
            model=model,
            device=device,
            ax_row=axes[row_idx],
        )
    
    # Add column headers
    col_titles = ["Original", "RandAugment #1", "RandAugment #2", "SAS (Ours)"]
    for col_idx, title in enumerate(col_titles):
        axes[0][col_idx].set_title(title, fontsize=10, fontweight="bold", pad=10)
    
    plt.tight_layout()
    
    # Save
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "fig_failure_cases.png"
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")
    
    # Also save a smaller teaser version (first 3 rows)
    if args.n_samples >= 3:
        fig, axes = plt.subplots(3, n_cols, figsize=(n_cols * 2, 3 * 2))
        
        for row_idx, sample_idx in enumerate(sample_indices[:3]):
            pil_img, true_label = ds[sample_idx]
            samples = create_augmented_samples(pil_img, sas_config)
            visualize_single_image(
                samples=samples,
                true_label=true_label,
                class_names=CIFAR100_CLASSES,
                model=model,
                device=device,
                ax_row=axes[row_idx],
            )
        
        plt.tight_layout()
        teaser_path = output_dir / "fig_failure_cases_teaser.png"
        plt.savefig(teaser_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"Saved teaser: {teaser_path}")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
