# ResNet-18 fixed architecture (No-NAS constraint)
"""
Model Module for Prior-Guided Augmentation Policy Search.

Provides ResNet-18 model for CIFAR-100 classification.
Architecture is fixed per No-NAS constraint (research_plan_v4.md Section 5).

Note: We do NOT modify the architecture. Only the data augmentation changes.
"""

from typing import Optional

import torch
import torch.nn as nn
import torchvision.models as models


def create_model(
    num_classes: int = 100,
    pretrained: bool = False,
) -> nn.Module:
    """Create ResNet-18 model for CIFAR-100.
    
    Uses torchvision.models.resnet18 without architectural modification
    to comply with No-NAS constraint.
    
    Args:
        num_classes: Number of output classes. Default 100 for CIFAR-100.
        pretrained: If True, load ImageNet pretrained weights. Default False.
        
    Returns:
        ResNet-18 model instance.
        
    Note:
        The standard ResNet-18 is designed for ImageNet (224x224 images).
        For CIFAR-100 (32x32), the model still works but the first conv layer
        and max pooling may be suboptimal. However, we keep it unchanged
        per No-NAS constraint.
    """
    # Use weights parameter (new API) instead of deprecated pretrained
    weights = "IMAGENET1K_V1" if pretrained else None
    
    model = models.resnet18(weights=weights)
    
    # Replace the final fully connected layer for num_classes
    # Original: (512, 1000) for ImageNet
    # New: (512, num_classes) for CIFAR-100
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    
    return model


def get_model_info(model: nn.Module) -> dict:
    """Get model information for logging.
    
    Args:
        model: The model to inspect.
        
    Returns:
        Dict with model info (name, params, trainable_params).
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "name": model.__class__.__name__,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "total_params_M": total_params / 1e6,
    }


if __name__ == "__main__":
    """Self-test for model module."""
    import sys
    
    print("=" * 60)
    print("Model Module Self-Test")
    print("=" * 60)
    
    # Create model
    print("\n[1/3] Creating ResNet-18 model...")
    model = create_model(num_classes=100, pretrained=False)
    info = get_model_info(model)
    print(f"      Model: {info['name']}")
    print(f"      Total params: {info['total_params']:,} ({info['total_params_M']:.2f}M)")
    print(f"      Trainable params: {info['trainable_params']:,}")
    
    # Forward pass test
    print("\n[2/3] Testing forward pass...")
    x = torch.randn(2, 3, 32, 32)
    model.eval()
    with torch.no_grad():
        y = model(x)
    print(f"      Input shape: {x.shape}")
    print(f"      Output shape: {y.shape}")
    
    assert y.shape == (2, 100), f"Expected (2, 100), got {y.shape}"
    print("      ✓ Forward pass check passed")
    
    # Check output is valid
    print("\n[3/3] Checking output validity...")
    assert torch.isfinite(y).all(), "Output contains non-finite values"
    print(f"      Output range: [{y.min():.4f}, {y.max():.4f}]")
    print("      ✓ Output validity check passed")
    
    print("\n" + "=" * 60)
    print("SUCCESS: Model forward pass check passed.")
    print("=" * 60)
    
    sys.exit(0)



