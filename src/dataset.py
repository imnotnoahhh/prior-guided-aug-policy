# CIFAR-100 dataset loading with stratified 5-Fold split
"""
CIFAR-100 Subsampled Dataset Module.

Implements stratified 5-Fold splitting for low-data augmentation policy search.
Each fold contains 10,000 samples (20% of full dataset), further split into
Train (9,000) / Val (1,000) for hyperparameter search.

Reference: docs/research_plan_v4.md Section 1.2
"""

from pathlib import Path
from typing import Callable, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision import transforms
from sklearn.model_selection import StratifiedKFold, train_test_split


class CIFAR100Subsampled(Dataset):
    """CIFAR-100 dataset with stratified subsampling for low-data experiments.
    
    This dataset implements the data validation protocol from the research plan:
    - 5-Fold stratified split of the full 50,000 training images
    - Each fold contains exactly 10,000 images (20% per class)
    - Within each fold, further split into Train (90%) / Val (10%)
    
    Args:
        root: Root directory for dataset storage. Created if not exists.
        train: If True, returns training split; if False, returns validation split.
        fold_idx: Which fold to use (0-4). Default 0 for search phase.
        transform: Optional transform to apply to images.
        download: If True, downloads CIFAR-100 if not present.
        n_splits: Number of folds for StratifiedKFold. Default 5.
        random_state: Random seed for reproducibility. Default 42.
        val_size: Fraction of fold to use for validation. Default 0.1 (10%).
    
    Raises:
        ValueError: If fold_idx is out of range [0, n_splits-1].
        RuntimeError: If dataset download fails.
    """
    
    def __init__(
        self,
        root: Union[str, Path],
        train: bool = True,
        fold_idx: int = 0,
        transform: Optional[Callable] = None,
        download: bool = True,
        n_splits: int = 5,
        random_state: int = 42,
        val_size: float = 0.1,
    ) -> None:
        # Validate fold_idx
        if not 0 <= fold_idx < n_splits:
            raise ValueError(
                f"fold_idx must be in [0, {n_splits-1}], got {fold_idx}"
            )
        
        self.root = Path(root)
        self.train = train
        self.fold_idx = fold_idx
        self.transform = transform
        self.n_splits = n_splits
        self.random_state = random_state
        self.val_size = val_size
        
        # Create root directory if not exists
        self.root.mkdir(parents=True, exist_ok=True)
        
        # Load full CIFAR-100 training set
        try:
            self.full_dataset = torchvision.datasets.CIFAR100(
                root=str(self.root),
                train=True,
                download=download,
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load/download CIFAR-100 to {self.root}: {e}"
            )
        
        # Get targets as numpy array for sklearn
        targets = np.array(self.full_dataset.targets)
        
        # Generate stratified fold indices
        skf = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=random_state,
        )
        
        # Get all fold splits - use test indices as the subset for each fold
        # This gives us 5 mutually exclusive folds of 10,000 samples each
        all_indices = np.arange(len(self.full_dataset))
        folds = list(skf.split(all_indices, targets))
        
        # Select the specified fold's test indices as our subset
        # (In K-Fold, each sample appears in test exactly once)
        fold_indices = folds[fold_idx][1]  # [1] is test indices
        
        # Get targets for this fold's samples
        fold_targets = targets[fold_indices]
        
        # Further split into train/val within this fold
        fold_local_indices = np.arange(len(fold_indices))
        train_local_idx, val_local_idx = train_test_split(
            fold_local_indices,
            test_size=val_size,
            random_state=random_state,
            stratify=fold_targets,
        )
        
        # Map back to full dataset indices
        if train:
            self.indices = fold_indices[train_local_idx]
        else:
            self.indices = fold_indices[val_local_idx]
        
        # Store for debugging/verification
        self._fold_size = len(fold_indices)
        self._train_size = len(train_local_idx)
        self._val_size = len(val_local_idx)
    
    def __len__(self) -> int:
        """Return number of samples in this split."""
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get a sample by index.
        
        Args:
            idx: Index into this split (0 to len-1).
            
        Returns:
            Tuple of (image, label) where image is transformed if transform is set.
        """
        if idx < 0 or idx >= len(self.indices):
            raise IndexError(
                f"Index {idx} out of range for dataset of size {len(self.indices)}"
            )
        
        # Map to full dataset index
        real_idx = self.indices[idx]
        
        # Get image and label from full dataset
        image, label = self.full_dataset[real_idx]
        
        # Apply transform if provided
        if self.transform is not None:
            image = self.transform(image)
        
        return image, label
    
    def get_class_distribution(self) -> dict:
        """Return class distribution for verification.
        
        Returns:
            Dict mapping class_id to count in this split.
        """
        targets = np.array(self.full_dataset.targets)[self.indices]
        unique, counts = np.unique(targets, return_counts=True)
        return dict(zip(unique.tolist(), counts.tolist()))


def get_cifar100_loaders(
    root: Union[str, Path],
    fold_idx: int = 0,
    batch_size: int = 128,
    train_transform: Optional[Callable] = None,
    val_transform: Optional[Callable] = None,
    num_workers: int = 6,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation DataLoaders for CIFAR-100 subsampled dataset.
    
    Args:
        root: Root directory for dataset storage.
        fold_idx: Which fold to use (0-4).
        batch_size: Batch size for both loaders. Default 128 (per research plan).
        train_transform: Transform for training data. If None, uses ToTensor().
        val_transform: Transform for validation data. If None, uses ToTensor().
        num_workers: Number of worker processes for data loading.
        pin_memory: If True, pin memory for faster GPU transfer.
    
    Returns:
        Tuple of (train_loader, val_loader).
    """
    # Default transforms if not provided
    if train_transform is None:
        train_transform = transforms.ToTensor()
    if val_transform is None:
        val_transform = transforms.ToTensor()
    
    # Create datasets
    train_dataset = CIFAR100Subsampled(
        root=root,
        train=True,
        fold_idx=fold_idx,
        transform=train_transform,
    )
    
    val_dataset = CIFAR100Subsampled(
        root=root,
        train=False,
        fold_idx=fold_idx,
        transform=val_transform,
    )
    
    # Create loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    
    return train_loader, val_loader


if __name__ == "__main__":
    """Self-test for dataset module."""
    import sys
    
    print("=" * 60)
    print("CIFAR-100 Subsampled Dataset Self-Test")
    print("=" * 60)
    
    # Use only ToTensor for self-test (no normalization)
    transform = transforms.ToTensor()
    
    # Create datasets for fold 0
    print("\n[1/4] Creating train dataset (fold_idx=0, train=True)...")
    train_ds = CIFAR100Subsampled(
        root="./data",
        train=True,
        fold_idx=0,
        transform=transform,
        download=True,
    )
    print(f"      Train dataset size: {len(train_ds)}")
    
    print("\n[2/4] Creating val dataset (fold_idx=0, train=False)...")
    val_ds = CIFAR100Subsampled(
        root="./data",
        train=False,
        fold_idx=0,
        transform=transform,
        download=True,
    )
    print(f"      Val dataset size: {len(val_ds)}")
    
    # Create a DataLoader and get one batch
    print("\n[3/4] Loading one batch from train dataset...")
    loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=6)
    images, labels = next(iter(loader))
    print(f"      Batch shape: {images.shape}")
    print(f"      Labels shape: {labels.shape}")
    print(f"      Image dtype: {images.dtype}")
    print(f"      Image min: {images.min().item():.4f}")
    print(f"      Image max: {images.max().item():.4f}")
    
    # Run assertions
    print("\n[4/4] Running assertions...")
    
    # Assert RGB channels
    assert images.shape[1] == 3, f"Expected 3 channels, got {images.shape[1]}"
    print("      ✓ RGB channels check passed")
    
    # Assert value range (ToTensor scales to [0, 1])
    assert images.min() >= 0.0, f"Min value {images.min()} < 0"
    assert images.max() <= 1.0, f"Max value {images.max()} > 1"
    print("      ✓ Value range [0, 1] check passed")
    
    # Assert correct split sizes
    assert len(train_ds) == 9000, f"Expected 9000 train samples, got {len(train_ds)}"
    assert len(val_ds) == 1000, f"Expected 1000 val samples, got {len(val_ds)}"
    print("      ✓ Split size check passed (train=9000, val=1000)")
    
    # Assert image dimensions
    assert images.shape[2] == 32 and images.shape[3] == 32, \
        f"Expected 32x32 images, got {images.shape[2]}x{images.shape[3]}"
    print("      ✓ Image dimension check passed (32x32)")
    
    # Verify class distribution is roughly uniform
    train_dist = train_ds.get_class_distribution()
    assert len(train_dist) == 100, f"Expected 100 classes, got {len(train_dist)}"
    print("      ✓ Class count check passed (100 classes)")
    
    # Each class should have ~90 samples in train (9000/100)
    min_per_class = min(train_dist.values())
    max_per_class = max(train_dist.values())
    print(f"      Class distribution: min={min_per_class}, max={max_per_class}")
    
    print("\n" + "=" * 60)
    print(f"SUCCESS: Dataset shape check passed: {images.shape}")
    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")
    print("=" * 60)
    
    sys.exit(0)



