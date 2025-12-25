# Utilities: seed setting, path management, logging
"""
Utility Module for Prior-Guided Augmentation Policy Search.

Provides:
- set_seed: Reproducibility setup
- train_one_epoch: Training loop with AMP, NaN guard, OOM handling
- evaluate: Validation loop with Top-1 and Top-5 accuracy
- EarlyStopping: Early stopping based on validation loss
- get_optimizer_and_scheduler: Fixed hyperparameters per No-NAS constraint

Reference: docs/research_plan_v4.md Section 5
"""

import random
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


# =============================================================================
# Reproducibility
# =============================================================================

def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility.
    
    Sets seeds for:
    - Python's random module
    - NumPy
    - PyTorch (CPU and CUDA)
    
    Args:
        seed: Random seed value. Default 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # For deterministic behavior (may impact performance)
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False


def set_seed_deterministic(seed: int, deterministic: bool = True) -> None:
    """Set random seeds with optional deterministic CUDA behavior.
    
    Enhanced version of set_seed() that includes deterministic toggle
    for Phase B robustness testing with multiple seeds.
    
    Sets seeds for:
    - Python's random module
    - NumPy
    - PyTorch (CPU and CUDA)
    - Optionally enables CUDA deterministic mode
    
    Args:
        seed: Random seed value.
        deterministic: If True, enables torch.backends.cudnn.deterministic
                      and disables benchmark mode. May impact performance
                      but ensures reproducibility. Default True.
    
    Note:
        When deterministic=True, CUDA operations will be deterministic
        but may be slower. This is important for Phase B where we need
        to compare results across different seeds reliably.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            # Enable deterministic mode for reproducibility
            # Note: This may reduce performance
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            # Allow cuDNN to auto-tune for performance
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True


# =============================================================================
# Device Setup
# =============================================================================

def get_device() -> torch.device:
    """Get the best available device.
    
    Returns:
        torch.device: CUDA if available, else MPS (Apple Silicon), else CPU.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


# =============================================================================
# Training Loop
# =============================================================================

def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: Optional[torch.amp.GradScaler] = None,
    channels_last: bool = True,
) -> Tuple[float, float]:
    """Train for one epoch with AMP support and channels_last optimization.
    
    Features:
    - AMP: torch.autocast + GradScaler when CUDA available and scaler provided
    - channels_last: NHWC memory format for better GPU performance (default: True)
    - NaN guard: raises ValueError if loss becomes NaN
    - OOM handling: catches CUDA OOM, clears cache, skips batch
    
    Args:
        model: The model to train.
        train_loader: Training data loader.
        criterion: Loss function.
        optimizer: Optimizer instance.
        device: Device to train on.
        scaler: GradScaler for AMP. If None, uses FP32 training.
        channels_last: Use NHWC memory format for images. Default True.
        
    Returns:
        Tuple of (average_loss, accuracy_percent).
        
    Raises:
        ValueError: If loss becomes NaN.
    """
    model.train()
    
    # Use tensors to accumulate, avoid GPU-CPU sync per batch
    total_loss = torch.tensor(0.0, device=device)
    correct = torch.tensor(0, device=device)
    total = 0
    num_batches = 0
    skipped_batches = 0
    
    # Determine if we should use AMP
    use_amp = scaler is not None and device.type == "cuda"
    use_channels_last = channels_last and device.type == "cuda"
    
    for images, labels in train_loader:
        try:
            # H2D transfer with channels_last format to avoid nchwToNhwcKernel conversion
            if use_channels_last:
                images = images.to(device, non_blocking=True, memory_format=torch.channels_last)
            else:
                images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            # Forward pass with optional autocast
            with torch.autocast(device_type="cuda", enabled=use_amp):
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            # NaN guard - check before backward
            if torch.isnan(loss):
                raise ValueError(
                    f"Loss is NaN! Batch {num_batches}, "
                    f"outputs range: [{outputs.min():.4f}, {outputs.max():.4f}]"
                )
            
            # Backward pass
            optimizer.zero_grad(set_to_none=True)
            
            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            
            # Statistics - accumulate as tensors to avoid GPU-CPU sync per batch
            total_loss += loss.detach()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum()
            num_batches += 1
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                # Handle CUDA OOM gracefully
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                skipped_batches += 1
                print(f"WARNING: OOM at batch {num_batches}, skipping...")
                continue
            else:
                # Re-raise other RuntimeErrors
                raise
    
    if num_batches == 0:
        raise RuntimeError("All batches were skipped due to OOM!")
    
    # Single GPU-CPU sync at end of epoch
    avg_loss = (total_loss / num_batches).item()
    accuracy = 100.0 * correct.item() / total if total > 0 else 0.0
    
    if skipped_batches > 0:
        print(f"WARNING: Skipped {skipped_batches} batches due to OOM")
    
    return avg_loss, accuracy


# =============================================================================
# Evaluation Loop
# =============================================================================

def evaluate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    channels_last: bool = True,
) -> Tuple[float, float, float]:
    """Evaluate model on validation set.
    
    Computes loss, Top-1 accuracy, and Top-5 accuracy.
    
    Args:
        model: The model to evaluate.
        val_loader: Validation data loader.
        criterion: Loss function.
        device: Device to evaluate on.
        channels_last: Use NHWC memory format for images. Default True.
        
    Returns:
        Tuple of (average_loss, top1_accuracy_percent, top5_accuracy_percent).
    """
    model.eval()
    
    # Use tensors to accumulate, avoid GPU-CPU sync per batch
    total_loss = torch.tensor(0.0, device=device)
    correct_top1 = torch.tensor(0, device=device)
    correct_top5 = torch.tensor(0, device=device)
    total = 0
    num_batches = 0
    use_channels_last = channels_last and device.type == "cuda"
    
    with torch.no_grad():
        for images, labels in val_loader:
            # H2D transfer with channels_last format
            if use_channels_last:
                images = images.to(device, non_blocking=True, memory_format=torch.channels_last)
            else:
                images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Accumulate as tensors to avoid GPU-CPU sync per batch
            total_loss += loss.detach()
            
            # Top-1 accuracy
            _, predicted = outputs.max(1)
            correct_top1 += predicted.eq(labels).sum()
            
            # Top-5 accuracy
            _, top5_pred = outputs.topk(5, dim=1, largest=True, sorted=True)
            correct_top5 += top5_pred.eq(labels.view(-1, 1).expand_as(top5_pred)).any(dim=1).sum()
            
            total += labels.size(0)
            num_batches += 1
    
    # Single GPU-CPU sync at end of evaluation
    avg_loss = (total_loss / num_batches).item() if num_batches > 0 else 0.0
    top1_acc = 100.0 * correct_top1.item() / total if total > 0 else 0.0
    top5_acc = 100.0 * correct_top5.item() / total if total > 0 else 0.0
    
    return avg_loss, top1_acc, top5_acc


# =============================================================================
# Early Stopping
# =============================================================================

class EarlyStopping:
    """Early stopping based on validation metric with minimum epochs requirement.
    
    Stops training if validation metric doesn't improve for `patience` epochs,
    but only starts checking after `min_epochs` have passed.
    
    v5.1 Update: 
    - Renamed grace_period to min_epochs (clearer semantics)
    - Recommended to use mode="max" with val_acc (not val_loss)
    - Added min_delta filtering for noisy metrics
    
    Args:
        patience: Number of epochs to wait before stopping. Default 30.
        min_delta: Minimum improvement to reset counter (in percentage points for acc). Default 0.2.
        mode: 'min' for loss (lower is better), 'max' for accuracy. Default 'max'.
        min_epochs: Minimum epochs before early stopping is considered. Default 100.
        
    Recommended settings:
        - Phase A/B (200 epochs): min_epochs=100, patience=30, min_delta=0.2, mode="max"
        - Phase C/D (800 epochs): Disable early stopping (patience=99999) or min_epochs=500, patience=100
        
    Example:
        # For Phase A/B (monitor val_acc, mode="max")
        early_stopper = EarlyStopping(patience=30, min_epochs=100, mode="max", min_delta=0.2)
        for epoch in range(max_epochs):
            val_acc = evaluate(...)
            if early_stopper(val_acc, epoch):
                print("Early stopping triggered!")
                break
                
        # For Phase C/D (disable early stopping)
        early_stopper = EarlyStopping(patience=99999)  # Effectively disabled
    """
    
    def __init__(
        self,
        patience: int = 30,
        min_delta: float = 0.2,
        mode: str = "max",
        min_epochs: int = 100,
        # Backward compatibility alias
        grace_period: int = None,
    ) -> None:
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got {mode}")
        
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        # Support both min_epochs and legacy grace_period
        self.min_epochs = grace_period if grace_period is not None else min_epochs
        self.counter = 0
        self.best_value = float("inf") if mode == "min" else float("-inf")
        self.early_stop = False
    
    # Backward compatibility property
    @property
    def grace_period(self) -> int:
        return self.min_epochs
    
    def __call__(self, value: float, epoch: int = None) -> bool:
        """Check if training should stop.
        
        Args:
            value: Current validation metric (loss or accuracy).
            epoch: Current epoch (0-indexed). If provided, min_epochs is enforced.
            
        Returns:
            True if training should stop, False otherwise.
        """
        # Always track best value
        if self.mode == "min":
            improved = value < (self.best_value - self.min_delta)
        else:
            improved = value > (self.best_value + self.min_delta)
        
        if improved:
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1
        
        # Only consider stopping after min_epochs
        if epoch is not None and epoch < self.min_epochs:
            return False
        
        if self.counter >= self.patience:
            self.early_stop = True
        
        return self.early_stop
    
    def reset(self) -> None:
        """Reset the early stopping state."""
        self.counter = 0
        self.best_value = float("inf") if self.mode == "min" else float("-inf")
        self.early_stop = False


# =============================================================================
# Optimizer and Scheduler (Fixed Hyperparameters)
# =============================================================================

def get_optimizer_and_scheduler(
    model: nn.Module,
    total_epochs: int,
    lr: float = 0.1,
    weight_decay: float = 5e-3,
    momentum: float = 0.9,
    warmup_epochs: int = 5,
) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
    """Create SGD optimizer with warmup + CosineAnnealingLR scheduler.
    
    Hyperparameters per research_plan_v5.md Section 5.
    
    Args:
        model: The model to optimize.
        total_epochs: Total number of training epochs.
        lr: Learning rate. Default 0.1.
        weight_decay: Weight decay. Default 5e-3 (increased for regularization).
        momentum: SGD momentum. Default 0.9.
        warmup_epochs: Number of warmup epochs. Default 5.
            Linear warmup from lr/warmup_epochs to lr.
        
    Returns:
        Tuple of (optimizer, scheduler).
        
    Note:
        Scheduler structure: LinearLR (warmup) → CosineAnnealingLR
        Combined using SequentialLR with milestone at warmup_epochs.
    """
    from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
    
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
    )
    
    if warmup_epochs > 0 and total_epochs > warmup_epochs:
        # Warmup: linear from lr/warmup_epochs to lr
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=1.0 / warmup_epochs,  # Start at lr/warmup_epochs
            end_factor=1.0,                     # End at lr
            total_iters=warmup_epochs,
        )
        
        # Cosine annealing after warmup
        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=total_epochs - warmup_epochs,
        )
        
        # Combine: warmup → cosine
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs],
        )
    else:
        # No warmup, just cosine
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=total_epochs,
        )
    
    return optimizer, scheduler


# =============================================================================
# Path Management
# =============================================================================

def ensure_dir(path: Path) -> Path:
    """Ensure directory exists, create if not.
    
    Args:
        path: Directory path.
        
    Returns:
        The same path (for chaining).
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


# =============================================================================
# Self-Test
# =============================================================================

if __name__ == "__main__":
    """Self-test for utils module."""
    import sys
    
    print("=" * 60)
    print("Utils Module Self-Test")
    print("=" * 60)
    
    # Test 1: set_seed
    print("\n[1/5] Testing set_seed...")
    set_seed(42)
    val1 = torch.randn(3)
    set_seed(42)
    val2 = torch.randn(3)
    assert torch.allclose(val1, val2), "Seed not working correctly"
    print("      ✓ set_seed reproducibility check passed")
    
    # Test 2: get_device
    print("\n[2/5] Testing get_device...")
    device = get_device()
    print(f"      Device: {device}")
    print("      ✓ get_device check passed")
    
    # Test 3: EarlyStopping
    print("\n[3/5] Testing EarlyStopping...")
    es = EarlyStopping(patience=3, mode="min")
    
    # Simulate improving then stagnating loss
    assert not es(1.0), "Should not stop"
    assert not es(0.9), "Should not stop (improved)"
    assert not es(0.9), "Should not stop (counter=1)"
    assert not es(0.9), "Should not stop (counter=2)"
    assert es(0.9), "Should stop (counter=3)"
    print("      ✓ EarlyStopping logic check passed")
    
    # Test 4: get_optimizer_and_scheduler
    print("\n[4/5] Testing get_optimizer_and_scheduler...")
    from src.models import create_model
    model = create_model(num_classes=100)
    optimizer, scheduler = get_optimizer_and_scheduler(model, total_epochs=200, warmup_epochs=5)
    
    assert isinstance(optimizer, torch.optim.SGD), "Expected SGD optimizer"
    assert optimizer.defaults["lr"] == 0.1, f"Expected lr=0.1, got {optimizer.defaults['lr']}"
    assert optimizer.defaults["momentum"] == 0.9, "Expected momentum=0.9"
    assert optimizer.defaults["weight_decay"] == 5e-3, "Expected weight_decay=5e-3"
    print(f"      Optimizer: {optimizer.__class__.__name__}")
    print(f"      Scheduler: {scheduler.__class__.__name__} (with warmup)")
    print("      ✓ Optimizer and scheduler check passed")
    
    # Test 5: ensure_dir
    print("\n[5/5] Testing ensure_dir...")
    test_dir = Path("./outputs/test_utils_dir")
    ensure_dir(test_dir)
    assert test_dir.exists(), "Directory was not created"
    test_dir.rmdir()  # Clean up
    print("      ✓ ensure_dir check passed")
    
    print("\n" + "=" * 60)
    print("SUCCESS: Utils module check passed.")
    print("=" * 60)
    
    sys.exit(0)
