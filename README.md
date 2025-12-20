# prior-guided-aug-policy
Official PyTorch implementation of "Prior-Guided Augmentation Policy Search in Low-Data Regimes". A data-efficient pipeline for finding robust augmentation policies on CIFAR-100.

## Quick Start

### Environment Setup

```bash
# Create conda environment
conda env create -f environment.yml

# Activate environment
conda activate pga
```

### Running Smoke Tests

One-click full system validation:

```bash
bash scripts/smoke_test.sh
```

**Expected output on success:**

```
[Step 1/7] Dependency import check...
IMPORT_OK
[Step 2/7] Module import check...
SMOKE_OK
...
[Step 7/7] Validating artifacts...
CSV exists: outputs/phase_a_smoke_YYYYMMDD_HHMMSS.csv
Data rows: 2

========================================
ALL TESTS PASSED
========================================
```

**If conda activation fails:**

1. Ensure conda is installed and initialized: `conda init bash`
2. Verify the `pga` environment exists: `conda env list`
3. If missing, create it: `conda env create -f environment.yml`

### Running Phase A (Full Search)

```bash
# Full search: 200 epochs, 32 Sobol samples per operation
python main_phase_a.py

# Dry run for testing
python main_phase_a.py --epochs 1 --n_samples 2
```

## Project Structure

```
├── src/
│   ├── dataset.py        # CIFAR-100 with 5-Fold stratified split
│   ├── augmentations.py  # Augmentation operations pool
│   ├── models.py         # ResNet-18 (fixed architecture)
│   └── utils.py          # Training utilities
├── scripts/
│   └── smoke_test.sh     # One-click validation script
├── main_phase_a.py       # Phase A screening script
├── outputs/              # Results directory
└── docs/
    └── research_plan_v4.md
```

## License

See [LICENSE](LICENSE) for details.
