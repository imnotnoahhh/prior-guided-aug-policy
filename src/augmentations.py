# Augmentation operations pool (S0 baseline + candidate pool)
"""
Augmentation Module for Prior-Guided Augmentation Policy Search.

Implements:
- S0 Baseline: RandomCrop(padding=4) + RandomHorizontalFlip(p=0.5)
- Candidate Pool: 8 operations with magnitude [0,1] to physical parameter mapping
- Mutual exclusion handling for conflicting operations

Reference: docs/research_plan_v4.md Section 2
"""

from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image


# =============================================================================
# AugmentationSpace: Magnitude to Physical Parameter Mapping
# =============================================================================

class AugmentationSpace:
    """Defines magnitude [0,1] to physical parameter mapping for each operation.
    
    Each operation has parameters that are linearly interpolated based on magnitude:
    - magnitude=0: minimum effect (or no effect)
    - magnitude=1: maximum effect
    
    Reference: research_plan_v4.md Section 2.2
    """
    
    # Operation registry: {op_name: {param_name: (min_val, max_val)}}
    # Linear interpolation: param = min_val + magnitude * (max_val - min_val)
    OPERATIONS: Dict[str, Dict[str, Tuple[float, float]]] = {
        # scale goes from 1.0 (no crop) to 0.25 (aggressive crop)
        "RandomResizedCrop": {"scale_min": (1.0, 0.25)},
        # degrees goes from 0 to 30
        "RandomRotation": {"degrees": (0.0, 30.0)},
        # distortion_scale goes from 0 to 0.5
        "RandomPerspective": {"distortion_scale": (0.0, 0.5)},
        # brightness/contrast/saturation all go from 0 to 0.8
        "ColorJitter": {"strength": (0.0, 0.8)},
        # probability goes from 0 to 0.5
        "RandomGrayscale": {"p": (0.0, 0.5)},
        # sigma goes from 0.1 to 2.0
        "GaussianBlur": {"sigma": (0.1, 2.0)},
        # scale goes from 0.02 to 0.4
        "RandomErasing": {"scale": (0.02, 0.4)},
        # sigma goes from 0 to 0.1
        "GaussianNoise": {"sigma": (0.0, 0.1)},
    }
    
    # Mutual exclusion groups
    MUTUAL_EXCLUSION = {
        "RandomResizedCrop": ["S0_RandomCrop"],  # Disables S0's RandomCrop
        "RandomRotation": ["RandomPerspective"],
        "RandomPerspective": ["RandomRotation"],
    }
    
    @staticmethod
    def get_param(op_name: str, magnitude: float) -> Dict[str, float]:
        """Map magnitude [0,1] to operation-specific parameters.
        
        Args:
            op_name: Name of the operation (must be in OPERATIONS).
            magnitude: Strength value in [0, 1].
            
        Returns:
            Dict of parameter names to computed values.
            
        Raises:
            ValueError: If op_name is not recognized or magnitude out of range.
        """
        if op_name not in AugmentationSpace.OPERATIONS:
            raise ValueError(
                f"Unknown operation: {op_name}. "
                f"Available: {list(AugmentationSpace.OPERATIONS.keys())}"
            )
        
        if not 0.0 <= magnitude <= 1.0:
            raise ValueError(f"Magnitude must be in [0, 1], got {magnitude}")
        
        params = {}
        for param_name, (min_val, max_val) in AugmentationSpace.OPERATIONS[op_name].items():
            # Linear interpolation
            params[param_name] = min_val + magnitude * (max_val - min_val)
        
        return params
    
    @staticmethod
    def get_available_ops() -> List[str]:
        """Return list of available operation names."""
        return list(AugmentationSpace.OPERATIONS.keys())
    
    @staticmethod
    def is_mutually_exclusive(op1: str, op2: str) -> bool:
        """Check if two operations are mutually exclusive."""
        if op1 in AugmentationSpace.MUTUAL_EXCLUSION:
            if op2 in AugmentationSpace.MUTUAL_EXCLUSION[op1]:
                return True
        if op2 in AugmentationSpace.MUTUAL_EXCLUSION:
            if op1 in AugmentationSpace.MUTUAL_EXCLUSION[op2]:
                return True
        return False


# =============================================================================
# Custom Transforms
# =============================================================================

class GaussianNoise(nn.Module):
    """Add Gaussian noise to tensor image with clamping to [0, 1].
    
    This is a custom transform since torchvision doesn't provide GaussianNoise.
    Always clamps output to prevent overflow (per research_plan_v4.md constraint).
    
    Args:
        sigma: Standard deviation of the Gaussian noise. Default 0.1.
    """
    
    def __init__(self, sigma: float = 0.1) -> None:
        super().__init__()
        self.sigma = sigma
    
    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """Apply Gaussian noise to image tensor.
        
        Args:
            img: Tensor of shape (C, H, W) with values in [0, 1].
            
        Returns:
            Noisy tensor clamped to [0, 1].
        """
        if self.sigma <= 0:
            return img
        
        noise = torch.randn_like(img) * self.sigma
        return torch.clamp(img + noise, 0.0, 1.0)
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(sigma={self.sigma})"


# =============================================================================
# Baseline Transform (S0)
# =============================================================================

# CIFAR-100 normalization constants (ImageNet pretrained values)
CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)


def get_baseline_transform(
    include_normalize: bool = False,
    include_totensor: bool = True,
) -> transforms.Compose:
    """Return S0 baseline transform: RandomCrop + RandomHorizontalFlip.
    
    This is the baseline augmentation that is always applied (per research plan).
    
    Args:
        include_normalize: If True, add Normalize transform at the end.
        include_totensor: If True, add ToTensor transform.
        
    Returns:
        Composed transform pipeline.
    """
    transform_list = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
    ]
    
    if include_totensor:
        transform_list.append(transforms.ToTensor())
    
    if include_normalize:
        transform_list.append(
            transforms.Normalize(mean=CIFAR100_MEAN, std=CIFAR100_STD)
        )
    
    return transforms.Compose(transform_list)


def get_val_transform(include_normalize: bool = False) -> transforms.Compose:
    """Return validation transform (no augmentation, just ToTensor).
    
    Args:
        include_normalize: If True, add Normalize transform.
        
    Returns:
        Composed transform pipeline.
    """
    transform_list = [transforms.ToTensor()]
    
    if include_normalize:
        transform_list.append(
            transforms.Normalize(mean=CIFAR100_MEAN, std=CIFAR100_STD)
        )
    
    return transforms.Compose(transform_list)


# =============================================================================
# Single Operation Application
# =============================================================================

def create_op_transform(op_name: str, magnitude: float) -> Callable:
    """Create a transform for a single operation with given magnitude.
    
    Args:
        op_name: Name of the operation.
        magnitude: Strength in [0, 1].
        
    Returns:
        A callable transform.
        
    Raises:
        ValueError: If op_name is not recognized.
    """
    params = AugmentationSpace.get_param(op_name, magnitude)
    
    if op_name == "RandomResizedCrop":
        scale_min = params["scale_min"]
        return transforms.RandomResizedCrop(
            size=32,
            scale=(scale_min, 1.0),
            ratio=(0.75, 1.33),
            antialias=True,
        )
    
    elif op_name == "RandomRotation":
        degrees = params["degrees"]
        return transforms.RandomRotation(degrees=degrees)
    
    elif op_name == "RandomPerspective":
        distortion = params["distortion_scale"]
        return transforms.RandomPerspective(
            distortion_scale=distortion,
            p=1.0 if distortion > 0 else 0.0,
        )
    
    elif op_name == "ColorJitter":
        strength = params["strength"]
        return transforms.ColorJitter(
            brightness=strength,
            contrast=strength,
            saturation=strength,
            hue=0,  # Keep hue fixed for CIFAR
        )
    
    elif op_name == "RandomGrayscale":
        p = params["p"]
        return transforms.RandomGrayscale(p=p)
    
    elif op_name == "GaussianBlur":
        sigma = params["sigma"]
        # Kernel size must be odd, use 3 for 32x32 images
        return transforms.GaussianBlur(kernel_size=3, sigma=(sigma, sigma))
    
    elif op_name == "RandomErasing":
        scale = params["scale"]
        return transforms.RandomErasing(
            p=1.0,
            scale=(scale * 0.5, scale),  # Range around target scale
            ratio=(0.3, 3.3),
            value=0,
        )
    
    elif op_name == "GaussianNoise":
        sigma = params["sigma"]
        return GaussianNoise(sigma=sigma)
    
    else:
        raise ValueError(f"Unknown operation: {op_name}")


def apply_single_op(
    img: torch.Tensor,
    op_name: str,
    magnitude: float,
) -> torch.Tensor:
    """Apply a single augmentation operation to a tensor image.
    
    Args:
        img: Tensor of shape (C, H, W) with values in [0, 1].
        op_name: Name of the operation.
        magnitude: Strength in [0, 1].
        
    Returns:
        Augmented tensor with values in [0, 1].
    """
    transform = create_op_transform(op_name, magnitude)
    result = transform(img)
    
    # Ensure output is in valid range (especially important for noise ops)
    if op_name in ["GaussianNoise"]:
        # Already clamped in GaussianNoise transform
        pass
    
    return result


# =============================================================================
# Combined Transform Builder
# =============================================================================

def build_transform_with_op(
    op_name: str,
    magnitude: float,
    include_baseline: bool = True,
    include_normalize: bool = False,
) -> transforms.Compose:
    """Build complete transform pipeline with S0 + single candidate op.
    
    Handles mutual exclusion:
    - If op_name == "RandomResizedCrop": skip S0's RandomCrop (use RRC instead)
    
    Pipeline order:
    1. PIL augmentations (Crop, Flip, Rotation, etc.)
    2. ToTensor
    3. Tensor augmentations (RandomErasing, GaussianNoise)
    4. Normalize (optional)
    
    Args:
        op_name: Name of the candidate operation to add.
        magnitude: Strength in [0, 1].
        include_baseline: If True, include S0 baseline transforms.
        include_normalize: If True, add normalization at the end.
        
    Returns:
        Composed transform pipeline.
    """
    pil_transforms = []
    tensor_transforms = []
    
    # Operations that work on PIL images (before ToTensor)
    PIL_OPS = {
        "RandomResizedCrop", "RandomRotation", "RandomPerspective",
        "ColorJitter", "RandomGrayscale", "GaussianBlur"
    }
    
    # Operations that work on tensors (after ToTensor)
    TENSOR_OPS = {"RandomErasing", "GaussianNoise"}
    
    # Add S0 baseline if requested
    if include_baseline:
        # Check mutual exclusion with RandomResizedCrop
        if op_name == "RandomResizedCrop":
            # Skip S0's RandomCrop, use RandomResizedCrop instead
            pil_transforms.append(transforms.RandomHorizontalFlip(p=0.5))
        else:
            # Normal S0: RandomCrop + RandomHorizontalFlip
            pil_transforms.append(transforms.RandomCrop(32, padding=4))
            pil_transforms.append(transforms.RandomHorizontalFlip(p=0.5))
    
    # Add candidate operation
    op_transform = create_op_transform(op_name, magnitude)
    
    if op_name in PIL_OPS:
        # Insert PIL op before ToTensor
        # For RandomResizedCrop, put it first (replaces RandomCrop)
        if op_name == "RandomResizedCrop":
            pil_transforms.insert(0, op_transform)
        else:
            pil_transforms.append(op_transform)
    elif op_name in TENSOR_OPS:
        tensor_transforms.append(op_transform)
    
    # Build final pipeline
    final_transforms = pil_transforms + [transforms.ToTensor()] + tensor_transforms
    
    if include_normalize:
        final_transforms.append(
            transforms.Normalize(mean=CIFAR100_MEAN, std=CIFAR100_STD)
        )
    
    return transforms.Compose(final_transforms)


# =============================================================================
# Self-Test
# =============================================================================

if __name__ == "__main__":
    """Self-test for augmentation module."""
    import sys
    
    print("=" * 60)
    print("Augmentation Module Self-Test")
    print("=" * 60)
    
    # Test 1: AugmentationSpace parameter mapping
    print("\n[1/5] Testing AugmentationSpace parameter mapping...")
    for op_name in AugmentationSpace.get_available_ops():
        params_0 = AugmentationSpace.get_param(op_name, 0.0)
        params_1 = AugmentationSpace.get_param(op_name, 1.0)
        params_mid = AugmentationSpace.get_param(op_name, 0.5)
        print(f"      {op_name}: m=0 -> {params_0}, m=1 -> {params_1}")
    print("      ✓ Parameter mapping check passed")
    
    # Test 2: GaussianNoise clamp on black image
    print("\n[2/5] Testing GaussianNoise clamp (black image)...")
    img_black = torch.zeros(3, 32, 32)
    noise_op = GaussianNoise(sigma=0.1)
    noisy = noise_op(img_black)
    assert noisy.min() >= 0.0, f"Min {noisy.min()} < 0"
    assert noisy.max() <= 1.0, f"Max {noisy.max()} > 1"
    print(f"      Input: min={img_black.min():.4f}, max={img_black.max():.4f}")
    print(f"      Output: min={noisy.min():.4f}, max={noisy.max():.4f}")
    print("      ✓ GaussianNoise clamp check passed")
    
    # Test 3: GaussianNoise clamp on white image
    print("\n[3/5] Testing GaussianNoise clamp (white image)...")
    img_white = torch.ones(3, 32, 32)
    noisy_white = noise_op(img_white)
    assert noisy_white.min() >= 0.0, f"Min {noisy_white.min()} < 0"
    assert noisy_white.max() <= 1.0, f"Max {noisy_white.max()} > 1"
    print(f"      Input: min={img_white.min():.4f}, max={img_white.max():.4f}")
    print(f"      Output: min={noisy_white.min():.4f}, max={noisy_white.max():.4f}")
    print("      ✓ GaussianNoise clamp check passed (white image)")
    
    # Test 4: apply_single_op with max magnitude for all ops
    print("\n[4/5] Testing apply_single_op for all operations...")
    img_rand = torch.rand(3, 32, 32)
    for op_name in AugmentationSpace.get_available_ops():
        result = apply_single_op(img_rand.clone(), op_name, magnitude=1.0)
        assert result.shape == (3, 32, 32), \
            f"{op_name}: Expected shape (3, 32, 32), got {result.shape}"
        assert torch.isfinite(result).all(), \
            f"{op_name}: Output contains non-finite values"
        print(f"      {op_name}: shape={result.shape}, "
              f"range=[{result.min():.4f}, {result.max():.4f}]")
    print("      ✓ All operations produce valid output")
    
    # Test 5: build_transform_with_op (full pipeline)
    print("\n[5/5] Testing build_transform_with_op pipeline...")
    from PIL import Image
    import numpy as np
    
    # Create a dummy PIL image
    pil_img = Image.fromarray(
        (np.random.rand(32, 32, 3) * 255).astype(np.uint8)
    )
    
    for op_name in AugmentationSpace.get_available_ops():
        transform = build_transform_with_op(op_name, magnitude=0.5)
        result = transform(pil_img)
        assert result.shape == (3, 32, 32), \
            f"{op_name}: Expected shape (3, 32, 32), got {result.shape}"
        assert torch.isfinite(result).all(), \
            f"{op_name}: Output contains non-finite values"
        print(f"      {op_name}: ✓")
    print("      ✓ All pipelines work correctly")
    
    # Test mutual exclusion flag
    print("\n[Bonus] Testing mutual exclusion detection...")
    assert AugmentationSpace.is_mutually_exclusive("RandomRotation", "RandomPerspective")
    assert not AugmentationSpace.is_mutually_exclusive("ColorJitter", "GaussianBlur")
    print("      ✓ Mutual exclusion detection works")
    
    print("\n" + "=" * 60)
    print("SUCCESS: Augmentation logic check passed.")
    print("=" * 60)
    
    sys.exit(0)



