#!/usr/bin/env bash
# =============================================================================
# Smoke Test Script for Phase C: Prior-Guided Greedy Ensemble (v5.5)
# =============================================================================
# Validates Phase C environment and runs a minimal dry-run test.
#
# v5.5 Notes:
#   - Improvement threshold increased to 0.2% (from 0.1%)
#   - Majority rule: requires ≥2/3 seeds to show improvement
#   - Smoke test uses dry_run mode for quick validation
#
# Usage:
#   bash scripts/smoke_test_phase_c.sh
#
# Requirements:
#   - conda environment 'pga' must exist
#   - Phase B summary must exist (outputs/phase_b_tuning_summary.csv)
#   - Run from project root directory
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
SMOKE_OUTPUT_DIR="${PROJECT_ROOT}/outputs/smoke_test_phase_c_${TIMESTAMP}"

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

print_header "Phase C Smoke Test"
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

# Step 2: Check required input files
print_step 2 $TOTAL_STEPS "Checking required input files"
PHASE_B_CSV="${PROJECT_ROOT}/outputs/phase_b_tuning_summary.csv"
PHASE_A_CSV="${PROJECT_ROOT}/outputs/phase_a_results.csv"

if [[ ! -f "${PHASE_B_CSV}" ]]; then
    echo "ERROR: Phase B summary not found: ${PHASE_B_CSV}"
    echo "Please run Phase B first."
    exit 1
fi

echo "Phase B summary: ${PHASE_B_CSV}"
echo "Phase B summary rows: $(wc -l < "${PHASE_B_CSV}")"

# Phase A CSV is optional but recommended for Top-K starting points (v5.5)
if [[ -f "${PHASE_A_CSV}" ]]; then
    echo "Phase A results found: ${PHASE_A_CSV} (will be used for Top-K starting points)"
else
    echo "Phase A results not found (Top-K starting points will be disabled)"
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
    build_transform_with_ops,
    get_compatible_ops,
    check_mutual_exclusion,
)
print('augmentations.py: OK')

from src.dataset import CIFAR100Subsampled
print('dataset.py: OK')

from src.models import create_model
print('models.py: OK')

from src.utils import set_seed_deterministic, train_one_epoch, evaluate
print('utils.py: OK')

# Phase C specific
import main_phase_c
print('main_phase_c.py: OK')

print('')
print('All imports successful!')
"

# Step 4: Create smoke test output directory
print_step 4 $TOTAL_STEPS "Creating smoke test output directory"
mkdir -p "${SMOKE_OUTPUT_DIR}"
echo "Created: ${SMOKE_OUTPUT_DIR}"

# Step 5: Test Phase C functions
print_step 5 $TOTAL_STEPS "Testing Phase C helper functions"
python -c "
import pandas as pd
from pathlib import Path
from main_phase_c import load_phase_b_summary, get_best_config_per_op

# Load Phase B summary
phase_b_csv = Path('${PHASE_B_CSV}')
df = load_phase_b_summary(phase_b_csv)
print(f'Loaded Phase B summary: {len(df)} rows')

# Get best configs
best_configs = get_best_config_per_op(df)
print(f'Found {len(best_configs)} operations with best configs:')
for op, (m, p, acc) in sorted(best_configs.items(), key=lambda x: -x[1][2]):
    print(f'  {op}: m={m:.4f}, p={p:.4f}, acc={acc:.2f}%')

print('')
print('Phase C helper functions: OK')
"

# Step 6: Run dry-run Phase C (minimal epochs, v5.5: threshold 0.2%, majority rule)
print_step 6 $TOTAL_STEPS "Running Phase C dry-run (2 epochs, 1 seed, 1 op max, v5.5)"
PHASE_C_CMD="python main_phase_c.py \
    --epochs 2 \
    --seeds 42 \
    --max_ops 1 \
    --output_dir ${SMOKE_OUTPUT_DIR} \
    --phase_b_csv ${PHASE_B_CSV} \
    --baseline_acc 28.5 \
    --num_workers 4 \
    --dry_run"

# Add --phase_a_csv if file exists (for Top-K starting points, v5.5)
if [[ -f "${PHASE_A_CSV}" ]]; then
    PHASE_C_CMD="${PHASE_C_CMD} --phase_a_csv ${PHASE_A_CSV}"
fi

eval $PHASE_C_CMD

# Step 7: Verify outputs
print_step 7 $TOTAL_STEPS "Verifying outputs"

HISTORY_CSV="${SMOKE_OUTPUT_DIR}/phase_c_history.csv"
POLICY_JSON="${SMOKE_OUTPUT_DIR}/phase_c_final_policy.json"

if [[ -f "${HISTORY_CSV}" ]]; then
    echo "History CSV: ${HISTORY_CSV}"
    echo "  Rows: $(wc -l < "${HISTORY_CSV}")"
    echo "  Columns: $(head -1 "${HISTORY_CSV}")"
else
    echo "WARNING: History CSV not found (may be expected if no ops were tried)"
fi

if [[ -f "${POLICY_JSON}" ]]; then
    echo "Policy JSON: ${POLICY_JSON}"
    echo "  Content:"
    cat "${POLICY_JSON}"
else
    echo "WARNING: Policy JSON not found"
fi

# Cleanup (optional - keep for debugging)
# rm -rf "${SMOKE_OUTPUT_DIR}"

print_header "Phase C Smoke Test PASSED"
echo "All checks completed successfully!"
echo ""
echo "v5.5: Phase C now uses:"
echo "  - Improvement threshold: 0.2% (increased from 0.1%)"
echo "  - Majority rule: requires ≥2/3 seeds to show improvement"
echo "  - Top-K starting points: uses Phase A results if available"
echo ""
echo "You can now run Phase C with full settings:"
echo "  CUDA_VISIBLE_DEVICES=0 python main_phase_c.py --phase_a_csv outputs/phase_a_results.csv"
echo ""

