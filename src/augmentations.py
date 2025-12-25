# Augmentation operations pool (S0 baseline + candidate pool)
"""
Augmentation Module for Prior-Guided Augmentation Policy Search.

Implements:
- S0 Baseline: RandomCrop(padding=4) + RandomHorizontalFlip(p=0.5)
- Candidate Pool: 8 operations with magnitude [0,1] to physical parameter mapping
- Probabilistic application: each operation can be applied with probability p
- Mutual exclusion handling for conflicting operations

Reference: docs/research_plan_v5.md Section 2

Changelog (v4 → v5):
- [NEW] ProbabilisticTransform: wrapper for stochastic application
- [NEW] OP_SEARCH_SPACE: per-operation (m, p) search ranges
- [CHANGED] build_transform_with_op: now accepts probability parameter
"""

import random
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image


# =============================================================================
# v5 NEW: Per-Operation Search Space Configuration
# =============================================================================

# Per-operation search space configuration
# Each operation has customized (magnitude, probability) search ranges.
# This embodies prior knowledge about which operations are "destructive" vs "mild".
# Format: {"op_name": {"m": [min, max], "p": [min, max]}}
# Reference: research_plan_v5.md Section 2.2
#
# NOTE: RandomGrayscale is binary (full grayscale or none), so magnitude has no effect.
#       We set m to a fixed value [0.5, 0.5] to avoid wasting search budget.

OP_SEARCH_SPACE: Dict[str, Dict[str, List[float]]] = {
    # Mild operations - can use higher probability
    "ColorJitter":       {"m": [0.1, 0.8], "p": [0.2, 0.8]},
    "RandomGrayscale":   {"m": [0.5, 0.5], "p": [0.1, 0.6]},  # m fixed, only search p
    "GaussianNoise":     {"m": [0.05, 0.5], "p": [0.2, 0.8]},
    
    # Medium operations
    "RandomResizedCrop": {"m": [0.3, 0.9], "p": [0.3, 0.8]},
    "RandomRotation":    {"m": [0.0, 0.4], "p": [0.2, 0.6]},
    "GaussianBlur":      {"m": [0.0, 0.3], "p": [0.2, 0.6]},
    
    # Destructive operations - need lower probability
    "RandomErasing":     {"m": [0.05, 0.3], "p": [0.1, 0.4]},
    "RandomPerspective": {"m": [0.0, 0.3], "p": [0.1, 0.5]},
}


# =============================================================================
# v5 NEW: Probabilistic Transform Wrapper
# =============================================================================

class ProbabilisticTransform(nn.Module):
    """Wrapper that applies a transform with a given probability.
    
    This enables stochastic application of augmentations, which is crucial
    for low-data regimes where 100% application may be too aggressive.
    
    Args:
        transform: The underlying transform to apply.
        p: Probability of applying the transform. Default 1.0 (always apply).
    
    Example:
        >>> jitter = transforms.ColorJitter(brightness=0.5)
        >>> prob_jitter = ProbabilisticTransform(jitter, p=0.5)
        >>> # Now jitter is applied only 50% of the time
    """
    
    def __init__(self, transform: Callable, p: float = 1.0) -> None:
        super().__init__()
        self.transform = transform
        self.p = p
        
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"Probability must be in [0, 1], got {p}")
    
    def forward(self, img):
        """Apply transform with probability p.
        
        Args:
            img: Input image (PIL or Tensor, depending on wrapped transform).
            
        Returns:
            Transformed image if random < p, else original image.
        """
        if self.p >= 1.0:
            return self.transform(img)
        if self.p <= 0.0:
            return img
        # Use random.random() to avoid torch tensor creation and GPU-CPU sync
        if random.random() < self.p:
            return self.transform(img)
        return img
    
    def __repr__(self) -> str:
        return f"ProbabilisticTransform(p={self.p}, transform={self.transform})"


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
    def get_search_space(op_name: str) -> Dict[str, List[float]]:
        """Get the (m, p) search space for a given operation.
        
        v5 NEW: Returns customized search ranges per operation.
        
        Args:
            op_name: Name of the operation.
            
        Returns:
            Dict with 'm' and 'p' keys, each containing [min, max].
            
        Raises:
            ValueError: If op_name is not recognized.
        """
        if op_name not in OP_SEARCH_SPACE:
            raise ValueError(
                f"Unknown operation: {op_name}. "
                f"Available: {list(OP_SEARCH_SPACE.keys())}"
            )
        return OP_SEARCH_SPACE[op_name]
    
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
        # v5 FIX: RandomGrayscale's internal p is now always 1.0
        # The application probability is controlled by ProbabilisticTransform wrapper
        # magnitude has no effect on this op (grayscale is binary: full or none)
        return transforms.RandomGrayscale(p=1.0)
    
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
    probability: float = 1.0,
    include_baseline: bool = True,
    include_normalize: bool = False,
) -> transforms.Compose:
    """Build complete transform pipeline with S0 + single candidate op.
    
    v5 CHANGED: Added probability parameter for stochastic application.
    
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
        probability: Probability of applying the op (default 1.0 = always).
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
    
    # Add candidate operation with probability wrapper
    op_transform = create_op_transform(op_name, magnitude)
    
    # v5: Wrap with ProbabilisticTransform if probability < 1.0
    if probability < 1.0:
        op_transform = ProbabilisticTransform(op_transform, p=probability)
    
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
# Phase C/D: RandAugment and Cutout (SOTA Comparison)
# =============================================================================

def get_randaugment_transform(
    n: int = 2,
    m: int = 9,
    include_baseline: bool = True,
    include_normalize: bool = False,
) -> transforms.Compose:
    """Create RandAugment transform pipeline for SOTA comparison.
    
    Uses torchvision.transforms.RandAugment with standard settings.
    
    Args:
        n: Number of augmentation transformations to apply sequentially. Default 2.
        m: Magnitude for all transformations (0-30 scale). Default 9.
        include_baseline: If True, include S0 baseline transforms. Default True.
        include_normalize: If True, add normalization at the end. Default False.
        
    Returns:
        Composed transform pipeline.
        
    Reference:
        RandAugment: Practical automated data augmentation (Cubuk et al., 2020)
        Standard settings: N=2, M=9
    """
    pil_transforms = []
    
    # Add S0 baseline if requested
    if include_baseline:
        pil_transforms.append(transforms.RandomCrop(32, padding=4))
        pil_transforms.append(transforms.RandomHorizontalFlip(p=0.5))
    
    # Add RandAugment
    pil_transforms.append(transforms.RandAugment(num_ops=n, magnitude=m))
    
    # Build final pipeline
    final_transforms = pil_transforms + [transforms.ToTensor()]
    
    if include_normalize:
        final_transforms.append(
            transforms.Normalize(mean=CIFAR100_MEAN, std=CIFAR100_STD)
        )
    
    return transforms.Compose(final_transforms)


def get_cutout_transform(
    n_holes: int = 1,
    length: int = 16,
    include_baseline: bool = True,
    include_normalize: bool = False,
) -> transforms.Compose:
    """Create Cutout transform pipeline for SOTA comparison.
    
    Implements Cutout using RandomErasing with square holes.
    
    Args:
        n_holes: Number of holes to cut out (only 1 supported via RandomErasing). Default 1.
        length: Side length of the square hole in pixels. Default 16.
        include_baseline: If True, include S0 baseline transforms. Default True.
        include_normalize: If True, add normalization at the end. Default False.
        
    Returns:
        Composed transform pipeline.
        
    Reference:
        Cutout: Improved Regularization of Convolutional Networks (DeVries & Taylor, 2017)
        Standard settings for CIFAR: n_holes=1, length=16
        
    Note:
        For CIFAR-100 (32x32), length=16 means scale ≈ (16/32)² = 0.25
    """
    pil_transforms = []
    
    # Add S0 baseline if requested
    if include_baseline:
        pil_transforms.append(transforms.RandomCrop(32, padding=4))
        pil_transforms.append(transforms.RandomHorizontalFlip(p=0.5))
    
    # Calculate scale for RandomErasing to match Cutout behavior
    # Cutout: fixed size square hole
    # RandomErasing: scale is fraction of image area
    # For 32x32 image, length=16 means area = 16*16 = 256, scale = 256/1024 = 0.25
    img_size = 32
    hole_area = (length / img_size) ** 2
    
    # Use RandomErasing with square ratio to simulate Cutout
    cutout = transforms.RandomErasing(
        p=1.0,
        scale=(hole_area * 0.9, hole_area * 1.1),  # Slight variation around target
        ratio=(0.9, 1.1),  # Nearly square
        value=0,  # Black fill (original Cutout uses 0)
    )
    
    # Build final pipeline
    final_transforms = pil_transforms + [transforms.ToTensor(), cutout]
    
    if include_normalize:
        final_transforms.append(
            transforms.Normalize(mean=CIFAR100_MEAN, std=CIFAR100_STD)
        )
    
    return transforms.Compose(final_transforms)


# =============================================================================
# Phase C: Multi-Operation Combination Builder
# =============================================================================

def check_mutual_exclusion(ops: List[str]) -> List[Tuple[str, str]]:
    """Check for mutual exclusion conflicts in a list of operations.
    
    Args:
        ops: List of operation names to check.
        
    Returns:
        List of conflicting (op1, op2) pairs. Empty if no conflicts.
        
    Example:
        >>> check_mutual_exclusion(["RandomRotation", "RandomPerspective"])
        [("RandomRotation", "RandomPerspective")]
        >>> check_mutual_exclusion(["ColorJitter", "GaussianBlur"])
        []
    """
    conflicts = []
    for i, op1 in enumerate(ops):
        for op2 in ops[i+1:]:
            if AugmentationSpace.is_mutually_exclusive(op1, op2):
                conflicts.append((op1, op2))
    return conflicts


def get_compatible_ops(current_ops: List[str], candidate_op: str) -> bool:
    """Check if a candidate operation is compatible with current operations.
    
    Args:
        current_ops: List of already selected operation names.
        candidate_op: The candidate operation to add.
        
    Returns:
        True if candidate is compatible (no mutual exclusion), False otherwise.
    """
    for existing_op in current_ops:
        if AugmentationSpace.is_mutually_exclusive(existing_op, candidate_op):
            return False
    return True


def build_transform_with_ops(
    ops: List[Tuple[str, float, float]],
    include_baseline: bool = True,
    include_normalize: bool = False,
) -> transforms.Compose:
    """Build complete transform pipeline with S0 + multiple candidate ops.
    
    Phase C NEW: Supports combining multiple augmentation operations.
    
    Handles mutual exclusion:
    - If any op is "RandomResizedCrop": skip S0's RandomCrop
    - Operations are applied in the order provided
    
    Pipeline order:
    1. S0 baseline (if include_baseline and no RandomResizedCrop)
    2. PIL augmentations in order
    3. ToTensor
    4. Tensor augmentations in order
    5. Normalize (optional)
    
    Args:
        ops: List of (op_name, magnitude, probability) tuples.
        include_baseline: If True, include S0 baseline transforms. Default True.
        include_normalize: If True, add normalization at the end. Default False.
        
    Returns:
        Composed transform pipeline.
        
    Raises:
        ValueError: If ops contain mutually exclusive operations.
        
    Example:
        >>> ops = [("ColorJitter", 0.5, 0.6), ("GaussianBlur", 0.3, 0.4)]
        >>> transform = build_transform_with_ops(ops)
    """
    if not ops:
        # No extra ops, just return baseline
        return get_baseline_transform(
            include_normalize=include_normalize,
            include_totensor=True,
        )
    
    # Check for mutual exclusion conflicts
    op_names = [op[0] for op in ops]
    conflicts = check_mutual_exclusion(op_names)
    if conflicts:
        raise ValueError(
            f"Mutual exclusion conflicts detected: {conflicts}. "
            "Cannot combine these operations."
        )
    
    # Operations that work on PIL images (before ToTensor)
    PIL_OPS = {
        "RandomResizedCrop", "RandomRotation", "RandomPerspective",
        "ColorJitter", "RandomGrayscale", "GaussianBlur"
    }
    
    # Operations that work on tensors (after ToTensor)
    TENSOR_OPS = {"RandomErasing", "GaussianNoise"}
    
    pil_transforms = []
    tensor_transforms = []
    
    # Check if any op is RandomResizedCrop (replaces S0's RandomCrop)
    has_rrc = any(op[0] == "RandomResizedCrop" for op in ops)
    
    # Add S0 baseline if requested
    if include_baseline:
        if has_rrc:
            # Skip S0's RandomCrop, only add HorizontalFlip
            pil_transforms.append(transforms.RandomHorizontalFlip(p=0.5))
        else:
            # Normal S0: RandomCrop + RandomHorizontalFlip
            pil_transforms.append(transforms.RandomCrop(32, padding=4))
            pil_transforms.append(transforms.RandomHorizontalFlip(p=0.5))
    
    # Add each operation in order
    for op_name, magnitude, probability in ops:
        op_transform = create_op_transform(op_name, magnitude)
        
        # Wrap with ProbabilisticTransform if probability < 1.0
        if probability < 1.0:
            op_transform = ProbabilisticTransform(op_transform, p=probability)
        
        if op_name in PIL_OPS:
            # For RandomResizedCrop, put it first (before HorizontalFlip)
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


def build_ours_p1_transform(
    ops: List[Tuple[str, float, float]],
    include_baseline: bool = True,
    include_normalize: bool = False,
) -> transforms.Compose:
    """Build transform with ops but force all probabilities to 1.0 (ablation).
    
    Phase D: Used for p=1.0 ablation comparison.
    
    Args:
        ops: List of (op_name, magnitude, probability) tuples.
             The probability values are ignored and set to 1.0.
        include_baseline: If True, include S0 baseline transforms.
        include_normalize: If True, add normalization at the end.
        
    Returns:
        Composed transform pipeline with all ops applied at p=1.0.
    """
    # Override all probabilities to 1.0
    ops_p1 = [(op_name, magnitude, 1.0) for op_name, magnitude, _ in ops]
    return build_transform_with_ops(ops_p1, include_baseline, include_normalize)


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
    
    # v5: Test ProbabilisticTransform
    print("\n[v5-1] Testing ProbabilisticTransform wrapper...")
    # Test p=0: should never apply
    dummy_transform = transforms.Lambda(lambda x: x * 2)
    prob_never = ProbabilisticTransform(dummy_transform, p=0.0)
    test_tensor = torch.ones(3, 32, 32)
    result = prob_never(test_tensor)
    assert torch.equal(result, test_tensor), "p=0 should never apply transform"
    print("      ✓ p=0.0 works (never applies)")
    
    # Test p=1: should always apply
    prob_always = ProbabilisticTransform(dummy_transform, p=1.0)
    result = prob_always(test_tensor)
    assert torch.equal(result, test_tensor * 2), "p=1 should always apply transform"
    print("      ✓ p=1.0 works (always applies)")
    
    # Test p=0.5: statistical test
    prob_half = ProbabilisticTransform(dummy_transform, p=0.5)
    applied_count = 0
    n_trials = 1000
    for _ in range(n_trials):
        result = prob_half(test_tensor)
        if torch.equal(result, test_tensor * 2):
            applied_count += 1
    ratio = applied_count / n_trials
    assert 0.4 < ratio < 0.6, f"p=0.5 should apply ~50%, got {ratio:.2%}"
    print(f"      ✓ p=0.5 works (applied {ratio:.1%} of {n_trials} trials)")
    
    # v5: Test OP_SEARCH_SPACE
    print("\n[v5-2] Testing OP_SEARCH_SPACE configuration...")
    for op_name in AugmentationSpace.get_available_ops():
        space = AugmentationSpace.get_search_space(op_name)
        assert "m" in space, f"{op_name} missing 'm' range"
        assert "p" in space, f"{op_name} missing 'p' range"
        m_min, m_max = space["m"]
        p_min, p_max = space["p"]
        assert 0 <= m_min <= m_max <= 1, f"{op_name} invalid m range: {space['m']}"
        assert 0 <= p_min <= p_max <= 1, f"{op_name} invalid p range: {space['p']}"
        print(f"      {op_name}: m=[{m_min:.2f}, {m_max:.2f}], p=[{p_min:.2f}, {p_max:.2f}]")
    print("      ✓ All operations have valid search spaces")
    
    # v5: Test build_transform_with_op with probability
    print("\n[v5-3] Testing build_transform_with_op with probability...")
    for op_name in AugmentationSpace.get_available_ops():
        transform = build_transform_with_op(op_name, magnitude=0.5, probability=0.5)
        result = transform(pil_img)
        assert result.shape == (3, 32, 32), \
            f"{op_name}: Expected shape (3, 32, 32), got {result.shape}"
        print(f"      {op_name}: ✓ (with p=0.5)")
    print("      ✓ All pipelines work with probability parameter")
    
    # Phase C/D: Test RandAugment
    print("\n[Phase C/D-1] Testing get_randaugment_transform...")
    ra_transform = get_randaugment_transform(n=2, m=9)
    result = ra_transform(pil_img)
    assert result.shape == (3, 32, 32), f"RandAugment: Expected (3, 32, 32), got {result.shape}"
    print("      ✓ RandAugment transform works")
    
    # Phase C/D: Test Cutout
    print("\n[Phase C/D-2] Testing get_cutout_transform...")
    cutout_transform = get_cutout_transform(n_holes=1, length=16)
    result = cutout_transform(pil_img)
    assert result.shape == (3, 32, 32), f"Cutout: Expected (3, 32, 32), got {result.shape}"
    print("      ✓ Cutout transform works")
    
    # Phase C/D: Test mutual exclusion checker
    print("\n[Phase C/D-3] Testing check_mutual_exclusion...")
    conflicts = check_mutual_exclusion(["RandomRotation", "RandomPerspective"])
    assert len(conflicts) == 1, f"Expected 1 conflict, got {len(conflicts)}"
    conflicts = check_mutual_exclusion(["ColorJitter", "GaussianBlur"])
    assert len(conflicts) == 0, f"Expected 0 conflicts, got {len(conflicts)}"
    print("      ✓ Mutual exclusion checker works")
    
    # Phase C/D: Test get_compatible_ops
    print("\n[Phase C/D-4] Testing get_compatible_ops...")
    assert get_compatible_ops(["ColorJitter"], "GaussianBlur") == True
    assert get_compatible_ops(["RandomRotation"], "RandomPerspective") == False
    print("      ✓ get_compatible_ops works")
    
    # Phase C/D: Test build_transform_with_ops
    print("\n[Phase C/D-5] Testing build_transform_with_ops...")
    ops = [("ColorJitter", 0.5, 0.6), ("GaussianBlur", 0.3, 0.4)]
    combo_transform = build_transform_with_ops(ops)
    result = combo_transform(pil_img)
    assert result.shape == (3, 32, 32), f"Combo: Expected (3, 32, 32), got {result.shape}"
    print("      ✓ Multi-op combination works")
    
    # Test with RandomResizedCrop (mutual exclusion with S0's RandomCrop)
    ops_rrc = [("RandomResizedCrop", 0.5, 0.7), ("ColorJitter", 0.3, 0.5)]
    rrc_transform = build_transform_with_ops(ops_rrc)
    result = rrc_transform(pil_img)
    assert result.shape == (3, 32, 32), f"RRC combo: Expected (3, 32, 32), got {result.shape}"
    print("      ✓ RandomResizedCrop combination works (replaces S0 RandomCrop)")
    
    # Test that conflicting ops raise error
    try:
        bad_ops = [("RandomRotation", 0.5, 0.5), ("RandomPerspective", 0.3, 0.3)]
        build_transform_with_ops(bad_ops)
        assert False, "Should have raised ValueError for conflicting ops"
    except ValueError as e:
        print(f"      ✓ Conflicting ops correctly rejected: {e}")
    
    # Phase C/D: Test build_ours_p1_transform
    print("\n[Phase C/D-6] Testing build_ours_p1_transform...")
    ops = [("ColorJitter", 0.5, 0.3), ("GaussianBlur", 0.2, 0.4)]
    p1_transform = build_ours_p1_transform(ops)
    result = p1_transform(pil_img)
    assert result.shape == (3, 32, 32), f"Ours p=1: Expected (3, 32, 32), got {result.shape}"
    print("      ✓ Ours p=1.0 ablation transform works")
    
    print("\n" + "=" * 60)
    print("SUCCESS: Augmentation logic check passed (v5 + Phase C/D features).")
    print("=" * 60)
    
    sys.exit(0)



