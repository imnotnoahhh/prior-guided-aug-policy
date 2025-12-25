#!/usr/bin/env bash
# =============================================================================
# Smoke Test Script for Phase B: Augmentation Tuning
# =============================================================================
# Validates Phase B environment and runs a minimal dry-run test.
#
# Usage:
#   bash scripts/smoke_test_phase_b.sh
#
# Requirements:
#   - conda environment 'pga' must exist
#   - Phase A results must exist (outputs/phase_a_results.csv)
#   - Baseline results must exist (outputs/baseline_result.csv)
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
print_header "Smoke Test: Phase B Augmentation Tuning"
echo "Timestamp: $TIMESTAMP"
echo "Project root: $PROJECT_ROOT"

# Change to project root
cd "$PROJECT_ROOT"

# -----------------------------------------------------------------------------
# Conda activation
# -----------------------------------------------------------------------------
print_step "0" "6" "Activating conda environment '$CONDA_ENV_NAME'..."

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "ERROR: conda is not available in PATH"
    echo "Please ensure conda is installed and initialized (run: conda init bash)"
    exit 1
fi

# Source conda for script environment
if [ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniforge3/etc/profile.d/conda.sh"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
elif [ -f "/opt/homebrew/Caskroom/miniforge/base/etc/profile.d/conda.sh" ]; then
    source "/opt/homebrew/Caskroom/miniforge/base/etc/profile.d/conda.sh"
else
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

# Set PYTHONPATH
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

# -----------------------------------------------------------------------------
# Step 1: Check Phase A results exist
# -----------------------------------------------------------------------------
print_step "1" "6" "Checking Phase A results exist..."

PHASE_A_CSV="$PROJECT_ROOT/outputs/phase_a_results.csv"
if [ ! -f "$PHASE_A_CSV" ]; then
    echo "ERROR: Phase A results not found: $PHASE_A_CSV"
    echo "Please run Phase A first: python main_phase_a.py"
    exit 1
fi
echo "Phase A results found: $PHASE_A_CSV"

PHASE_A_ROWS=$(tail -n +2 "$PHASE_A_CSV" | wc -l | tr -d ' ')
echo "Phase A data rows: $PHASE_A_ROWS"

# -----------------------------------------------------------------------------
# Step 2: Check baseline results exist
# -----------------------------------------------------------------------------
print_step "2" "6" "Checking baseline results exist..."

BASELINE_CSV="$PROJECT_ROOT/outputs/baseline_result.csv"
if [ ! -f "$BASELINE_CSV" ]; then
    echo "ERROR: Baseline results not found: $BASELINE_CSV"
    echo "Please run baseline first: python run_baseline.py"
    exit 1
fi
echo "Baseline results found: $BASELINE_CSV"

# Show baseline content
echo "Baseline content:"
cat "$BASELINE_CSV"

# -----------------------------------------------------------------------------
# Step 3: Verify Phase B script imports
# -----------------------------------------------------------------------------
print_step "3" "6" "Verifying Phase B script imports..."

python -c "
import sys
sys.path.insert(0, '$PROJECT_ROOT')
from main_phase_b import (
    load_phase_a_results,
    load_baseline_result,
    get_promoted_ops,
    sobol_sample_configs,
    aggregate_results,
)
from src.utils import set_seed_deterministic
print('All Phase B imports successful')
"

# -----------------------------------------------------------------------------
# Step 4: Test promoted ops detection
# -----------------------------------------------------------------------------
print_step "4" "6" "Testing promoted ops detection..."

python -c "
import sys
sys.path.insert(0, '$PROJECT_ROOT')
from pathlib import Path
from main_phase_b import load_phase_a_results, load_baseline_result, get_promoted_ops

phase_a_df = load_phase_a_results(Path('$PHASE_A_CSV'))
baseline_acc, baseline_top5, baseline_train_loss = load_baseline_result(Path('$BASELINE_CSV'))
promoted = get_promoted_ops(phase_a_df, baseline_acc, baseline_top5, baseline_train_loss)

print(f'Baseline: Top-1={baseline_acc:.1f}%, Top-5={baseline_top5:.1f}%, TrainLoss={baseline_train_loss:.4f}')
print(f'Promoted ops ({len(promoted)}): {promoted}')

if len(promoted) == 0:
    print('WARNING: No ops promoted!')
    sys.exit(1)
print('Promoted ops detection: PASSED')
"

# -----------------------------------------------------------------------------
# Step 5: Phase B dry run
# -----------------------------------------------------------------------------
print_step "5" "6" "Phase B ASHA dry run (n_samples=3, rungs=5,10)..."

# Create a temporary output directory for smoke test
SMOKE_OUTPUT_DIR="outputs/smoke_phase_b_${TIMESTAMP}"
mkdir -p "$SMOKE_OUTPUT_DIR"

# Run Phase B ASHA with minimal config
python main_phase_b.py \
    --n_samples 3 \
    --ops ColorJitter \
    --output_dir "$SMOKE_OUTPUT_DIR" \
    --dry_run

# -----------------------------------------------------------------------------
# Step 6: Validate output artifacts
# -----------------------------------------------------------------------------
print_step "6" "6" "Validating output artifacts..."

RAW_CSV="${SMOKE_OUTPUT_DIR}/phase_b_tuning_raw.csv"
SUMMARY_CSV="${SMOKE_OUTPUT_DIR}/phase_b_tuning_summary.csv"

# Check raw CSV exists
if [ ! -f "$RAW_CSV" ]; then
    echo "ERROR: Raw CSV not found: $RAW_CSV"
    exit 1
fi
echo "Raw CSV exists: $RAW_CSV"

# Check summary CSV exists
if [ ! -f "$SUMMARY_CSV" ]; then
    echo "ERROR: Summary CSV not found: $SUMMARY_CSV"
    exit 1
fi
echo "Summary CSV exists: $SUMMARY_CSV"

# Count rows
RAW_ROWS=$(tail -n +2 "$RAW_CSV" | wc -l | tr -d ' ')
SUMMARY_ROWS=$(tail -n +2 "$SUMMARY_CSV" | wc -l | tr -d ' ')
echo "Raw CSV rows: $RAW_ROWS"
echo "Summary CSV rows: $SUMMARY_ROWS"

# Validate row count (should have at least 1 row)
if [ "$RAW_ROWS" -lt 1 ]; then
    echo "ERROR: Raw CSV has no data rows"
    exit 1
fi

if [ "$SUMMARY_ROWS" -lt 1 ]; then
    echo "ERROR: Summary CSV has no data rows"
    exit 1
fi

# Show CSV contents
echo ""
echo "Raw CSV content:"
cat "$RAW_CSV"
echo ""
echo "Summary CSV content:"
cat "$SUMMARY_CSV"

# Validate summary is sorted by mean_val_acc descending
python -c "
import pandas as pd
df = pd.read_csv('$SUMMARY_CSV')
if len(df) > 1:
    sorted_check = df['mean_val_acc'].is_monotonic_decreasing
    if not sorted_check:
        print('WARNING: Summary not sorted by mean_val_acc descending')
    else:
        print('Summary sorting: CORRECT (mean_val_acc descending)')
else:
    print('Summary has only 1 row, sorting check skipped')
print('CSV validation: PASSED')
"

# -----------------------------------------------------------------------------
# Success
# -----------------------------------------------------------------------------
print_header "ALL PHASE B SMOKE TESTS PASSED"
echo "Smoke test completed successfully at $(date)"
echo "Raw results: $RAW_CSV"
echo "Summary results: $SUMMARY_CSV"
echo ""

exit 0
