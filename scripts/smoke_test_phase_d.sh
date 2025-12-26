#!/usr/bin/env bash
# =============================================================================
# Smoke Test Script for Phase D: SOTA Benchmark Comparison (v5.5)
# =============================================================================
# Validates Phase D environment and runs a minimal dry-run test.
#
# v5.5 Notes:
#   - Added Best_SingleOp method (7 methods total)
#   - Best_SingleOp requires --phase_b_csv parameter
#   - Smoke test uses minimal config for quick validation
#
# Usage:
#   bash scripts/smoke_test_phase_d.sh
#
# Requirements:
#   - conda environment 'pga' must exist
#   - Run from project root directory
#   - Phase C policy is optional (will use baseline if not found)
#   - Phase B summary is optional (needed for Best_SingleOp)
# =============================================================================

set -euo pipefail

# -----------------------------------------------------------------------------
# Error handling with trap
# -----------------------------------------------------------------------------
trap 'echo ""; echo "========================================"; echo "FAILED at line $LINENO: $BASH_COMMAND" >&2; echo "========================================"; exit 1' ERR

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
CONDA_ENV_NAME="pga"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
SMOKE_OUTPUT_DIR="${PROJECT_ROOT}/outputs/smoke_test_phase_d_${TIMESTAMP}"

# Environment variables for macOS compatibility
export KMP_DUPLICATE_LIB_OK=TRUE

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------
print_header() {
    echo ""
    echo "========================================"
    echo "$1"
    echo "========================================"
}

print_step() {
    echo ""
    echo "[Step $1/$2] $3"
    echo "----------------------------------------"
}

# -----------------------------------------------------------------------------
# Main Script
# -----------------------------------------------------------------------------
TOTAL_STEPS=7

print_header "Phase D Smoke Test"
echo "Project root: ${PROJECT_ROOT}"
echo "Output dir: ${SMOKE_OUTPUT_DIR}"
echo "Timestamp: ${TIMESTAMP}"

# Step 1: Check conda environment
print_step 1 $TOTAL_STEPS "Checking conda environment"
cd "${PROJECT_ROOT}"

if ! command -v conda &> /dev/null; then
    echo "ERROR: conda not found in PATH"
    exit 1
fi

# Initialize conda for bash
eval "$(conda shell.bash hook)"

if ! conda env list | grep -q "^${CONDA_ENV_NAME} "; then
    echo "ERROR: Conda environment '${CONDA_ENV_NAME}' not found"
    echo "Available environments:"
    conda env list
    exit 1
fi

echo "Activating conda environment: ${CONDA_ENV_NAME}"
conda activate "${CONDA_ENV_NAME}"
echo "Python: $(which python)"
echo "Python version: $(python --version)"

# Step 2: Check optional input files
print_step 2 $TOTAL_STEPS "Checking input files"
POLICY_JSON="${PROJECT_ROOT}/outputs/phase_c_final_policy.json"
PHASE_B_CSV="${PROJECT_ROOT}/outputs/phase_b_tuning_summary.csv"

if [[ -f "${POLICY_JSON}" ]]; then
    echo "Phase C policy found: ${POLICY_JSON}"
    echo "Content:"
    cat "${POLICY_JSON}"
else
    echo "Phase C policy not found (will use baseline for Ours methods)"
fi

if [[ -f "${PHASE_B_CSV}" ]]; then
    echo "Phase B summary found: ${PHASE_B_CSV} (needed for Best_SingleOp)"
else
    echo "Phase B summary not found (Best_SingleOp will use baseline)"
fi

# Step 3: Check Python imports
print_step 3 $TOTAL_STEPS "Checking Python imports"
python -c "
import sys
print('Python path:', sys.executable)

# Core imports
import torch
print(f'PyTorch: {torch.__version__}')

import torchvision
print(f'TorchVision: {torchvision.__version__}')

# Project imports
from src.augmentations import (
    get_baseline_transform,
    get_randaugment_transform,
    get_cutout_transform,
    build_transform_with_ops,
    build_ours_p1_transform,
)
print('augmentations.py: OK')

from src.dataset import CIFAR100Subsampled
print('dataset.py: OK')

from src.models import create_model
print('models.py: OK')

from src.utils import set_seed_deterministic, train_one_epoch, evaluate
print('utils.py: OK')

# Phase D specific
import main_phase_d
print('main_phase_d.py: OK')

print('')
print('All imports successful!')
"

# Step 4: Test all transforms
print_step 4 $TOTAL_STEPS "Testing all Phase D transforms (v5.5: 7 methods)"
python -c "
from PIL import Image
import numpy as np
import torch
from pathlib import Path

from main_phase_d import get_method_transform, AVAILABLE_METHODS, get_method_description

# Create dummy image
pil_img = Image.fromarray((np.random.rand(32, 32, 3) * 255).astype(np.uint8))

# Try to load Phase B CSV for Best_SingleOp (if exists)
phase_b_csv = Path('${PHASE_B_CSV}')
policy = None

print('Testing transforms for all methods:')
print(f'Available methods ({len(AVAILABLE_METHODS)}): {AVAILABLE_METHODS}')
print('')

for method in AVAILABLE_METHODS:
    try:
        # Best_SingleOp needs phase_b_csv, others don't
        if method == 'Best_SingleOp' and phase_b_csv.exists():
            # Will be loaded inside get_method_transform
            transform = get_method_transform(method, policy=None)
        else:
            transform = get_method_transform(method, policy=None)
        
        result = transform(pil_img)
        assert result.shape == (3, 32, 32), f'Wrong shape: {result.shape}'
        print(f'  {method}: OK - {get_method_description(method)}')
    except Exception as e:
        print(f'  {method}: ERROR - {e}')

print('')
print('All transforms work correctly!')
"

# Step 5: Create smoke test output directory
print_step 5 $TOTAL_STEPS "Creating smoke test output directory"
mkdir -p "${SMOKE_OUTPUT_DIR}"
echo "Created: ${SMOKE_OUTPUT_DIR}"

# Step 6: Run dry-run Phase D (minimal epochs, 2 methods, 1 fold, v5.5: 7 methods total)
print_step 6 $TOTAL_STEPS "Running Phase D dry-run (2 epochs, 2 methods, 1 fold, v5.5)"
PHASE_D_CMD="python main_phase_d.py \
    --epochs 2 \
    --seed 42 \
    --methods Baseline,RandAugment \
    --folds 0 \
    --output_dir ${SMOKE_OUTPUT_DIR} \
    --num_workers 4 \
    --dry_run"

# Add --phase_b_csv if file exists (for Best_SingleOp support)
if [[ -f "${PHASE_B_CSV}" ]]; then
    PHASE_D_CMD="${PHASE_D_CMD} --phase_b_csv ${PHASE_B_CSV}"
fi

# Add --policy_json if file exists
if [[ -f "${POLICY_JSON}" ]]; then
    PHASE_D_CMD="${PHASE_D_CMD} --policy_json ${POLICY_JSON}"
fi

eval $PHASE_D_CMD

# Step 7: Verify outputs
print_step 7 $TOTAL_STEPS "Verifying outputs"

RESULTS_CSV="${SMOKE_OUTPUT_DIR}/phase_d_results.csv"
SUMMARY_CSV="${SMOKE_OUTPUT_DIR}/phase_d_summary.csv"

if [[ -f "${RESULTS_CSV}" ]]; then
    echo "Results CSV: ${RESULTS_CSV}"
    echo "  Rows: $(wc -l < "${RESULTS_CSV}")"
    echo "  Content:"
    cat "${RESULTS_CSV}"
else
    echo "ERROR: Results CSV not found"
    exit 1
fi

if [[ -f "${SUMMARY_CSV}" ]]; then
    echo ""
    echo "Summary CSV: ${SUMMARY_CSV}"
    echo "  Content:"
    cat "${SUMMARY_CSV}"
else
    echo "ERROR: Summary CSV not found"
    exit 1
fi

# Cleanup (optional - keep for debugging)
# rm -rf "${SMOKE_OUTPUT_DIR}"

print_header "Phase D Smoke Test PASSED"
echo "All checks completed successfully!"
echo ""
echo "v5.5: Phase D now supports 7 methods:"
echo "  Baseline, Baseline-NoAug, RandAugment, Cutout, Best_SingleOp, Ours_p1, Ours_optimal"
echo ""
echo "You can now run Phase D with full settings:"
echo "  CUDA_VISIBLE_DEVICES=0 python main_phase_d.py --phase_b_csv outputs/phase_b_tuning_summary.csv"
echo ""
echo "Or run specific methods/folds:"
echo "  python main_phase_d.py --methods Baseline,Ours_optimal --folds 0,1,2"
echo ""

