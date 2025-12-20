# Architecture test script
"""
Architecture Test for Prior-Guided Augmentation Policy Search.

Verifies that:
1. ResNet-18 model can be created
2. Forward pass works with CIFAR-100 sized inputs (32x32)
3. Output shape is correct (batch_size, 100)
"""

import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch


def main() -> int:
    """Run architecture tests.
    
    Returns:
        Exit code (0 for success, 1 for failure).
    """
    print("=" * 60)
    print("Architecture Test")
    print("=" * 60)
    
    try:
        # Import model creation function
        print("\n[1/4] Importing model module...")
        from src.models import create_model, get_model_info
        print("      ✓ Import successful")
        
        # Create model
        print("\n[2/4] Creating ResNet-18 model (num_classes=100)...")
        model = create_model(num_classes=100, pretrained=False)
        info = get_model_info(model)
        print(f"      Model: {info['name']}")
        print(f"      Parameters: {info['total_params']:,} ({info['total_params_M']:.2f}M)")
        print("      ✓ Model creation successful")
        
        # Forward pass test
        print("\n[3/4] Testing forward pass...")
        model.eval()
        x = torch.randn(2, 3, 32, 32)  # Batch of 2, CIFAR-100 size
        
        with torch.no_grad():
            y = model(x)
        
        print(f"      Input shape: {x.shape}")
        print(f"      Output shape: {y.shape}")
        
        # Assert output shape
        assert y.shape == (2, 100), f"Expected (2, 100), got {y.shape}"
        print("      ✓ Output shape check passed")
        
        # Check output validity
        print("\n[4/4] Checking output validity...")
        assert torch.isfinite(y).all(), "Output contains non-finite values"
        print(f"      Output range: [{y.min():.4f}, {y.max():.4f}]")
        print("      ✓ Output validity check passed")
        
        print("\n" + "=" * 60)
        print("SUCCESS: Model forward pass check passed.")
        print("=" * 60)
        
        return 0
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())



