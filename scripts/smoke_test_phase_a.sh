#!/usr/bin/env bash
# =============================================================================
# Smoke Test Script for Prior-Guided Augmentation Policy Search
# =============================================================================
# One-click full system validation with fail-fast behavior.
#
# Usage:
#   bash scripts/smoke_test.sh
#
# Requirements:
#   - conda environment 'pga' must exist
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
# SMOKE_CSV is set dynamically in Step 6

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
# Main script
# -----------------------------------------------------------------------------
print_header "Smoke Test: Prior-Guided Augmentation Policy Search"
echo "Timestamp: $TIMESTAMP"
echo "Project root: $PROJECT_ROOT"

# Change to project root
cd "$PROJECT_ROOT"

# -----------------------------------------------------------------------------
# Conda activation
# -----------------------------------------------------------------------------
print_step "0" "7" "Activating conda environment '$CONDA_ENV_NAME'..."

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "ERROR: conda is not available in PATH"
    echo "Please ensure conda is installed and initialized (run: conda init bash)"
    exit 1
fi

# Source conda for script environment
# Try common conda locations
if [ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniforge3/etc/profile.d/conda.sh"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
elif [ -f "/opt/homebrew/Caskroom/miniforge/base/etc/profile.d/conda.sh" ]; then
    source "/opt/homebrew/Caskroom/miniforge/base/etc/profile.d/conda.sh"
else
    # Try to find conda.sh dynamically
    CONDA_SH=$(conda info --base)/etc/profile.d/conda.sh
    if [ -f "$CONDA_SH" ]; then
        source "$CONDA_SH"
    else
        echo "ERROR: Cannot find conda.sh for activation"
        exit 1
    fi
fi

# Activate the environment
conda activate "$CONDA_ENV_NAME"

# Verify Python executable
echo "Python executable: $(which python)"
python -c "import sys; print(f'sys.executable: {sys.executable}'); print(f'sys.prefix: {sys.prefix}')"

# Verify we're in the correct environment
if [[ "$(which python)" != *"$CONDA_ENV_NAME"* ]]; then
    echo "WARNING: Python path does not contain '$CONDA_ENV_NAME'"
    echo "Current Python: $(which python)"
fi

# Set PYTHONPATH to include project root
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

# Create outputs directory if missing
mkdir -p "$PROJECT_ROOT/outputs"

# -----------------------------------------------------------------------------
# Step 1: Dependency import check
# -----------------------------------------------------------------------------
print_step "1" "7" "Dependency import check..."
python -c "import torch, torchvision, numpy, sklearn, tqdm; print('IMPORT_OK')"

# -----------------------------------------------------------------------------
# Step 2: Module import check
# -----------------------------------------------------------------------------
print_step "2" "7" "Module import check..."
python -c "import src; import src.dataset, src.augmentations, src.models, src.utils; print('SMOKE_OK')"

# -----------------------------------------------------------------------------
# Step 3: Dataset self-test
# -----------------------------------------------------------------------------
print_step "3" "7" "Dataset module self-test..."
python src/dataset.py

# -----------------------------------------------------------------------------
# Step 4: Augmentations self-test
# -----------------------------------------------------------------------------
print_step "4" "7" "Augmentations module self-test..."
python src/augmentations.py

# -----------------------------------------------------------------------------
# Step 5: Model architecture self-test
# -----------------------------------------------------------------------------
print_step "5" "7" "Model architecture self-test..."
python src/test_arch.py

# -----------------------------------------------------------------------------
# Step 6: Phase A dry run
# -----------------------------------------------------------------------------
print_step "6" "7" "Phase A dry run (epochs=1, n_samples=2)..."

# Create a temporary output directory for smoke test to avoid polluting user data
SMOKE_OUTPUT_DIR="outputs/smoke_${TIMESTAMP}"
mkdir -p "$SMOKE_OUTPUT_DIR"

# Run Phase A with isolated output directory
python main_phase_a.py --epochs 1 --n_samples 2 --num_workers 6 --output_dir "$SMOKE_OUTPUT_DIR"

# Set the CSV path for validation
SMOKE_CSV="${SMOKE_OUTPUT_DIR}/phase_a_results.csv"

# -----------------------------------------------------------------------------
# Step 7: Artifact validation
# -----------------------------------------------------------------------------
print_step "7" "7" "Validating artifacts..."

# Check CSV exists
if [ ! -f "$SMOKE_CSV" ]; then
    echo "ERROR: CSV file not found: $SMOKE_CSV"
    exit 1
fi
echo "CSV exists: $SMOKE_CSV"

# Count data rows (excluding header)
DATA_ROWS=$(tail -n +2 "$SMOKE_CSV" | wc -l | tr -d ' ')
echo "Data rows: $DATA_ROWS"

# Validate row count
if [ "$DATA_ROWS" -ne 2 ]; then
    echo "ERROR: Expected 2 data rows, got $DATA_ROWS"
    exit 1
fi
echo "Row count validation: PASSED"

# Show CSV content
echo ""
echo "CSV content:"
cat "$SMOKE_CSV"

# -----------------------------------------------------------------------------
# Success
# -----------------------------------------------------------------------------
print_header "ALL TESTS PASSED"
echo "Smoke test completed successfully at $(date)"
echo "Results saved to: $SMOKE_CSV"
echo ""

exit 0

