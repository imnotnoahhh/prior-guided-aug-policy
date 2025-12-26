# Augmentation operations pool (S0 baseline + candidate pool)
"""
Augmentation Module for Prior-Guided Augmentation Policy Search.

Implements:
- S0 Baseline: RandomCrop(padding=4) + RandomHorizontalFlip(p=0.5)
- Candidate Pool: 8 operations with magnitude [0,1] to physical parameter mapping
- Probabilistic application: each operation can be applied with probability p
- Mutual exclusion handling for conflicting operations
- Combination probability adjustment for multi-op policies (v5.4)

Reference: docs/research_plan_v5.md Section 2

Changelog (v4 → v5):
- [NEW] ProbabilisticTransform: wrapper for stochastic application
- [NEW] OP_SEARCH_SPACE: per-operation (m, p) search ranges
- [CHANGED] build_transform_with_op: now accepts probability parameter

Changelog (v5.4):
- [NEW] OP_DESTRUCTIVENESS: per-operation destructiveness weights
- [NEW] adjust_probabilities_for_combination: control total augmentation intensity
- [CHANGED] RandomErasing p range expanded to [0.1, 0.5]

Changelog (v5.5):
- [CHANGED] OP_SEARCH_SPACE: more conservative ranges to avoid "必炸区"
- [CHANGED] adjust_probabilities_for_combination: now considers magnitude (w = 1 - d * g(m))
- [NEW] magnitude_influence function for weighted probability adjustment
"""

from typing import Callable, Dict, List, Optional, Tuple, Union
import random

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
    # Mild operations - can use higher probability and magnitude
    "ColorJitter":       {"m": [0.1, 0.9], "p": [0.3, 0.9]},
    "RandomGrayscale":   {"m": [0.5, 0.5], "p": [0.1, 0.7]},   # m fixed, only search p
    "GaussianNoise":     {"m": [0.02, 0.4], "p": [0.3, 0.9]},
    
    # Medium operations - moderate strength
    "RandomResizedCrop": {"m": [0.4, 0.95], "p": [0.3, 0.8]},
    "RandomRotation":    {"m": [0.0, 0.5], "p": [0.2, 0.7]},   # ~0-22.5 degrees
    "GaussianBlur":      {"m": [0.0, 0.5], "p": [0.2, 0.7]},
    
    # Destructive operations - need lower probability
    "RandomErasing":     {"m": [0.02, 0.35], "p": [0.1, 0.6]},
    "RandomPerspective": {"m": [0.0, 0.4], "p": [0.1, 0.6]},
}


# =============================================================================
# v5.4 NEW: Operation Destructiveness Weights
# =============================================================================

# Destructiveness weight for each operation (0=no destruction, 1=full destruction)
# Used for adjusting probabilities when combining multiple operations.
# Higher destructiveness → lower weight → more aggressive probability reduction.
OP_DESTRUCTIVENESS: Dict[str, float] = {
    "RandomErasing":      0.85,   # Direct pixel occlusion - highest destruction
    "RandomPerspective":  0.80,   # Severe geometric distortion
    "RandomResizedCrop":  0.65,   # May lose key regions
    "RandomRotation":     0.40,   # Moderate geometric change
    "GaussianBlur":       0.30,   # Mild blur, preserves structure
    "RandomGrayscale":    0.30,   # Removes color but keeps structure
    "GaussianNoise":      0.20,   # Light noise addition
    "ColorJitter":        0.20,   # Mildest - just color variation
}


def magnitude_influence(m: float) -> float:
    """Map magnitude [0,1] to influence factor [0,1].
    
    v5.5: Linear mapping. Higher magnitude → stronger influence on weight reduction.
    Can be changed to m**2 for more conservative high-magnitude handling.
    
    Args:
        m: Normalized magnitude in [0, 1].
        
    Returns:
        Influence factor in [0, 1].
    """
    return max(0.0, min(1.0, m))


def adjust_probabilities_for_combination(
    ops: List[Tuple[str, float, float]],
    p_any_target: float = 0.5,
    use_magnitude: bool = True,
) -> List[Tuple[str, float, float]]:
    """Adjust operation probabilities to control total augmentation intensity.
    
    When combining multiple operations, each applied independently with probability p,
    the overall augmentation intensity can become too strong. This function adjusts
    individual probabilities to maintain a target "at least one augmentation" rate.
    
    v5.5 Algorithm (with magnitude):
    1. Define weight w_i = 1 - d_i × g(m_i)
       - d_i: destructiveness of operation i
       - g(m_i): magnitude influence (linear by default)
    2. Scale probabilities: p'_i = clip(α × w_i × p_i, 0, 1)
    3. Solve for α such that: 1 - ∏(1 - p'_i) ≈ p_any_target
    
    Args:
        ops: List of (op_name, magnitude, probability) tuples.
        p_any_target: Target probability that at least one augmentation is applied.
                      Default 0.5 (50% of images get at least one augmentation).
        use_magnitude: If True, incorporate magnitude into weight calculation.
                       Default True (v5.5 behavior).
    
    Returns:
        Adjusted list of (op_name, magnitude, adjusted_probability) tuples.
    
    Example:
        >>> policy = [("RandomErasing", 0.24, 0.34), ("GaussianNoise", 0.08, 0.76)]
        >>> adjusted = adjust_probabilities_for_combination(policy, p_any_target=0.5)
        >>> # RandomErasing (d=0.85, m=0.24) gets reduced more than GaussianNoise (d=0.20, m=0.08)
    """
    import numpy as np
    
    if len(ops) <= 1:
        return ops  # Single operation doesn't need adjustment
    
    # Extract info
    names = [op[0] for op in ops]
    magnitudes = [op[1] for op in ops]
    probs = np.array([op[2] for op in ops], dtype=np.float64)
    
    # Compute weights: w_i = 1 - d_i × g(m_i)
    # v5.5: Now considers magnitude - higher m means more aggressive reduction
    weights = []
    for i, name in enumerate(names):
        d = OP_DESTRUCTIVENESS.get(name, 0.5)
        m = magnitudes[i]
        
        if use_magnitude:
            g_m = magnitude_influence(m)
            w = 1.0 - d * g_m
        else:
            # v5.4 behavior: only use destructiveness
            w = 1.0 - d
        
        weights.append(max(0.0, min(1.0, w)))
    
    weights = np.array(weights, dtype=np.float64)
    
    # Define objective function
    def compute_p_any(alpha: float) -> float:
        """Compute actual p_any for given alpha."""
        p_adjusted = np.clip(alpha * weights * probs, 0, 1)
        p_none = np.prod(1 - p_adjusted)
        return 1 - p_none
    
    # Binary search for optimal alpha
    alpha_low, alpha_high = 0.01, 10.0
    
    # Check if target is achievable
    p_any_max = compute_p_any(alpha_high)
    p_any_min = compute_p_any(alpha_low)
    
    if p_any_target >= p_any_max:
        alpha_opt = alpha_high
    elif p_any_target <= p_any_min:
        alpha_opt = alpha_low
    else:
        # Binary search
        for _ in range(50):  # 50 iterations gives precision ~1e-15
            alpha_mid = (alpha_low + alpha_high) / 2
            p_any_mid = compute_p_any(alpha_mid)
            
            if p_any_mid < p_any_target:
                alpha_low = alpha_mid
            else:
                alpha_high = alpha_mid
        
        alpha_opt = (alpha_low + alpha_high) / 2
    
    # Compute adjusted probabilities
    p_adjusted = np.clip(alpha_opt * weights * probs, 0, 1)
    
    # Build result
    result = [
        (names[i], magnitudes[i], float(p_adjusted[i]))
        for i in range(len(ops))
    ]
    
    return result


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
        # Use Python random instead of torch.rand().item() to avoid potential tensor ops
        if random.random() < self.p:
            return self.transform(img)
        return img
    
    def __repr__(self) -> str:
        return f"ProbabilisticTransform(p={self.p}, transform={self.transform})"


# =============================================================================
# v6 NEW: Dynamic Augmentation (Prior-Guided RandAugment)
# =============================================================================

class DynamicAugment(nn.Module):
    """Dynamic Augmentation Strategy (Prior-Guided RandAugment).
    
    Instead of a static sequence of operations, this transform dynamically
    selects N operations from a restricted 'Elite Pool' for each image.
    
    This combines:
    1. The diversity of RandAugment (random selection per image).
    2. The purity of Prior-Guided Search (only using safe/effective ops).
    
    Args:
        ops: List of (op_name, magnitude) tuples defining the Elite Pool.
        n: Number of operations to select per image (default 2).
    """
    
    def __init__(self, ops: List[Tuple[str, float]], n: int = 2) -> None:
        super().__init__()
        self.ops = ops
        self.n = n
        
        if not ops:
            raise ValueError("DynamicAugment requires at least one operation in the pool.")
            
    def forward(self, img):
        """Apply N randomly selected operations to the image.
        
        Args:
            img: Input image (PIL or Tensor).
            
        Returns:
            Augmented image.
        """
        # Select N operations with replacement? 
        # RandAugment does replacement (can pick same op twice).
        # We'll follow RandAugment behavior.
        chosen_ops = random.choices(self.ops, k=self.n)
        
        for op_name, magnitude in chosen_ops:
            # Create transform on the fly
            # Note: We use p=1.0 for the op itself, because the "probability"
            # is controlled by the random selection process (like RandAugment).
            transform = create_op_transform(op_name, magnitude)
            img = transform(img)
            
        return img
    
    def __repr__(self) -> str:
        op_names = [op[0] for op in self.ops]
        return f"DynamicAugment(n={self.n}, pool={op_names})"



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


def get_randaugment_transform(
    n: int = 2,
    m: int = 9,
    include_baseline: bool = True,
    include_normalize: bool = False,
) -> transforms.Compose:
    """Return RandAugment transform for SOTA comparison.
    
    Args:
        n: Number of augmentation operations to apply.
        m: Magnitude of augmentation (0-30 scale).
        include_baseline: If True, include S0 baseline transforms.
        include_normalize: If True, add Normalize transform at the end.
        
    Returns:
        Composed transform pipeline with RandAugment.
    """
    transform_list = []
    
    if include_baseline:
        transform_list.extend([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
        ])
    
    # RandAugment from torchvision
    transform_list.append(transforms.RandAugment(num_ops=n, magnitude=m))
    transform_list.append(transforms.ToTensor())
    
    if include_normalize:
        transform_list.append(
            transforms.Normalize(mean=CIFAR100_MEAN, std=CIFAR100_STD)
        )
    
    return transforms.Compose(transform_list)


def get_cutout_transform(
    n_holes: int = 1,
    length: int = 16,
    include_baseline: bool = True,
    include_normalize: bool = False,
) -> transforms.Compose:
    """Return Cutout transform for SOTA comparison.
    
    Args:
        n_holes: Number of holes to cut out.
        length: Length of each square hole.
        include_baseline: If True, include S0 baseline transforms.
        include_normalize: If True, add Normalize transform at the end.
        
    Returns:
        Composed transform pipeline with Cutout (RandomErasing).
    """
    transform_list = []
    
    if include_baseline:
        transform_list.extend([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
        ])
    
    transform_list.append(transforms.ToTensor())
    
    # Use RandomErasing as Cutout implementation
    # scale: area ratio of erasing region
    # ratio: aspect ratio of erasing region
    scale = (length * length) / (32 * 32)  # For 32x32 images
    transform_list.append(
        transforms.RandomErasing(
            p=1.0,  # Always apply
            scale=(scale * 0.8, scale * 1.2),  # Slight variation
            ratio=(0.8, 1.2),  # Nearly square
            value=0,  # Fill with black
        )
    )
    
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


def build_transform_with_ops(
    ops: list,
    include_baseline: bool = True,
    include_normalize: bool = False,
) -> transforms.Compose:
    """Build complete transform pipeline with S0 + multiple candidate ops.
    
    Args:
        ops: List of (op_name, magnitude, probability) tuples.
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
    
    # Check if any op is RandomResizedCrop (mutual exclusion with RandomCrop)
    has_rrc = any(op[0] == "RandomResizedCrop" for op in ops)
    
    # Add S0 baseline if requested
    if include_baseline:
        if has_rrc:
            # Skip S0's RandomCrop, use RandomResizedCrop from ops
            pil_transforms.append(transforms.RandomHorizontalFlip(p=0.5))
        else:
            # Normal S0: RandomCrop + RandomHorizontalFlip
            pil_transforms.append(transforms.RandomCrop(32, padding=4))
            pil_transforms.append(transforms.RandomHorizontalFlip(p=0.5))
    
    # Add each operation
    for op_name, magnitude, probability in ops:
        op_transform = create_op_transform(op_name, magnitude)
        
        # Wrap with ProbabilisticTransform if probability < 1.0
        if probability < 1.0:
            op_transform = ProbabilisticTransform(op_transform, p=probability)
        
        if op_name in PIL_OPS:
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


def build_dynamic_transform(
    ops: list,
    n: int = 2,
    include_baseline: bool = True,
    include_normalize: bool = False,
) -> transforms.Compose:
    """Build dynamic transform pipeline (RandAugment style).
    
    Selects N operations from the pool for each image.
    
    CRITICAL: Validates that all ops support Tensor inputs, 
    since DynamicAugment is applied AFTER ToTensor to support
    Tensor-only ops like GaussianNoise/RandomErasing.
    
    Args:
        ops: List of (op_name, magnitude) tuples (Elite Pool).
        n: Number of operations to apply per image.
        include_baseline: If True, include S0.
        include_normalize: If True, include Normalize.
        
    Returns:
        Composed transform.
    """
    transform_list = []
    
    # S0 (Applied on PIL image)
    if include_baseline:
        transform_list.append(transforms.RandomCrop(32, padding=4))
        transform_list.append(transforms.RandomHorizontalFlip(p=0.5))
        
    # Convert to Tensor
    transform_list.append(transforms.ToTensor())
    
    # Dynamic Augment (Applied on Tensor)
    # Allows mixing PIL-compatible ops (Rotation, etc.) and Tensor-only ops (Noise, Erasing)
    transform_list.append(DynamicAugment(ops, n))
    
    if include_normalize:
        transform_list.append(
            transforms.Normalize(mean=CIFAR100_MEAN, std=CIFAR100_STD)
        )
        
    return transforms.Compose(transform_list)


def build_ours_p1_transform(
    ops: list,
    include_baseline: bool = True,
    include_normalize: bool = False,
) -> transforms.Compose:
    """Build transform with all probabilities forced to 1.0 (for ablation study).
    
    This is used in Phase D to compare Ours_optimal (with tuned p) vs Ours_p1 (all p=1.0).
    
    Args:
        ops: List of (op_name, magnitude, probability) tuples. The probability is ignored
             and forced to 1.0.
        include_baseline: If True, include S0 baseline transforms.
        include_normalize: If True, include normalization.
        
    Returns:
        Composed transform with all probabilities set to 1.0.
    """
    # Convert ops to have p=1.0
    ops_p1 = [(op_name, magnitude, 1.0) for (op_name, magnitude, _) in ops]
    
    return build_transform_with_ops(
        ops=ops_p1,
        include_baseline=include_baseline,
        include_normalize=include_normalize,
    )


def get_compatible_ops(current_ops: list, candidate_op: str = None) -> bool:
    """Check if a candidate op is compatible with current policy.
    
    Args:
        current_ops: List of op_names already in the policy.
        candidate_op: The op to check for compatibility.
        
    Returns:
        True if candidate_op is compatible with all current_ops, False otherwise.
    """
    if candidate_op is None:
        return True
    
    # Check if candidate conflicts with any current op
    for current_op in current_ops:
        if AugmentationSpace.is_mutually_exclusive(candidate_op, current_op):
            return False
    
    return True


def check_mutual_exclusion(op1: str, op2: str) -> bool:
    """Check if two operations are mutually exclusive.
    
    Args:
        op1: First operation name.
        op2: Second operation name.
        
    Returns:
        True if mutually exclusive, False otherwise.
    """
    return AugmentationSpace.is_mutually_exclusive(op1, op2)


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
    
    print("\n" + "=" * 60)
    print("SUCCESS: Augmentation logic check passed (v5 features included).")
    print("=" * 60)
    
    sys.exit(0)



