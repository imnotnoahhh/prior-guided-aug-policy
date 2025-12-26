#!/usr/bin/env bash
# =============================================================================
# 干运行测试脚本 - 快速验证代码无语法错误和逻辑问题
# =============================================================================
# 本脚本用于在本地快速测试所有阶段的代码是否能正常运行
# 每个阶段只运行 2 epochs，仅验证代码逻辑
#
# 用法:
#   bash scripts/test_dry_run.sh
#
# 预计时间: ~10-15 分钟 (取决于硬件)
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
OUTPUT_DIR="${PROJECT_ROOT}/outputs_dryrun"

# Environment variables
export KMP_DUPLICATE_LIB_OK=TRUE
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# Activate conda
eval "$(conda shell.bash hook)"
conda activate pga

cd "${PROJECT_ROOT}"

# Create output directory
rm -rf "${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}"

echo "========================================================================"
echo "干运行测试脚本"
echo "========================================================================"
echo "输出目录: ${OUTPUT_DIR}"
echo ""

# -----------------------------------------------------------------------------
# Phase 0: 超参校准 (dry run)
# -----------------------------------------------------------------------------
echo ""
echo "[1/6] Testing Phase 0 超参校准..."
python run_phase0_calibration.py \
    --epochs 2 \
    --seeds 42 \
    --output_dir "${OUTPUT_DIR}" \
    --dry_run

if [ -f "${OUTPUT_DIR}/phase0_summary.csv" ]; then
    echo "✅ Phase 0 passed"
else
    echo "❌ Phase 0 failed"
    exit 1
fi

# -----------------------------------------------------------------------------
# Baseline
# -----------------------------------------------------------------------------
echo ""
echo "[2/6] Testing Baseline..."
# Baseline writes to outputs/, copy to our dry run dir
python run_baseline.py --epochs 2

if [ -f "outputs/baseline_result.csv" ]; then
    cp "outputs/baseline_result.csv" "${OUTPUT_DIR}/"
    echo "✅ Baseline passed"
else
    echo "❌ Baseline failed"
    exit 1
fi

# -----------------------------------------------------------------------------
# Phase A (单个 op)
# -----------------------------------------------------------------------------
echo ""
echo "[3/6] Testing Phase A..."
python main_phase_a.py \
    --epochs 2 \
    --n_samples 2 \
    --ops ColorJitter \
    --output_dir "${OUTPUT_DIR}"

if [ -f "${OUTPUT_DIR}/phase_a_results.csv" ]; then
    echo "✅ Phase A passed"
else
    echo "❌ Phase A failed"
    exit 1
fi

# -----------------------------------------------------------------------------
# Phase B (dry run)
# -----------------------------------------------------------------------------
echo ""
echo "[4/6] Testing Phase B..."
python main_phase_b.py \
    --output_dir "${OUTPUT_DIR}" \
    --phase_a_csv "${OUTPUT_DIR}/phase_a_results.csv" \
    --baseline_csv "${OUTPUT_DIR}/baseline_result.csv" \
    --dry_run

if [ -f "${OUTPUT_DIR}/phase_b_tuning_summary.csv" ]; then
    echo "✅ Phase B passed"
else
    echo "❌ Phase B failed"
    exit 1
fi

# -----------------------------------------------------------------------------
# Phase C (dry run)
# -----------------------------------------------------------------------------
echo ""
echo "[5/6] Testing Phase C..."
python main_phase_c.py \
    --output_dir "${OUTPUT_DIR}" \
    --phase_b_csv "${OUTPUT_DIR}/phase_b_tuning_summary.csv" \
    --dry_run

if [ -f "${OUTPUT_DIR}/phase_c_final_policy.json" ]; then
    echo "✅ Phase C passed"
else
    echo "❌ Phase C failed"
    exit 1
fi

# -----------------------------------------------------------------------------
# Phase D (dry run, 仅 Baseline 方法, 单 fold)
# -----------------------------------------------------------------------------
echo ""
echo "[6/6] Testing Phase D..."
python main_phase_d.py \
    --output_dir "${OUTPUT_DIR}" \
    --policy_json "${OUTPUT_DIR}/phase_c_final_policy.json" \
    --phase_b_csv "${OUTPUT_DIR}/phase_b_tuning_summary.csv" \
    --methods Baseline \
    --folds 0 \
    --dry_run

if [ -f "${OUTPUT_DIR}/phase_d_summary.csv" ]; then
    echo "✅ Phase D passed"
else
    echo "❌ Phase D failed"
    exit 1
fi

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
echo ""
echo "========================================================================"
echo "✅ 所有测试通过!"
echo "========================================================================"
echo ""
echo "生成的文件:"
ls -la "${OUTPUT_DIR}"/*.csv "${OUTPUT_DIR}"/*.json 2>/dev/null || true
echo ""
echo "提示: 干运行输出保存在 ${OUTPUT_DIR}/"
echo "      正式运行将使用 outputs/ 目录"

exit 0

