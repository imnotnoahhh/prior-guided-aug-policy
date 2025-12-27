#!/usr/bin/env bash
# =============================================================================
# 运行 Phase B/C/D，自动补跑失败的配置
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "${PROJECT_ROOT}"

# Activate conda
eval "$(conda shell.bash hook)"
conda activate pga

OUTPUT_DIR="${PROJECT_ROOT}/outputs"
LOG_DIR="${PROJECT_ROOT}/logs"
mkdir -p "${LOG_DIR}"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/phase_bcd_retry_${TIMESTAMP}.log"

echo "========================================================================"
echo "Phase B/C/D with Auto-Retry"
echo "========================================================================"
echo "日志文件: ${LOG_FILE}"
echo "========================================================================"

# -----------------------------------------------------------------------------
# Phase B
# -----------------------------------------------------------------------------
echo ""
echo "[1/4] Running Phase B..."
python main_phase_b.py --output_dir "${OUTPUT_DIR}" 2>&1 | tee -a "${LOG_FILE}"

if [ ! -f "${OUTPUT_DIR}/phase_b_tuning_summary.csv" ]; then
    echo "❌ Phase B failed!"
    exit 1
fi

# -----------------------------------------------------------------------------
# 检查并补跑失败的配置
# -----------------------------------------------------------------------------
echo ""
echo "[2/4] Checking for failed configurations..."

# 运行补跑脚本（会自动从 CSV 中读取失败的配置）
python scripts/retry_failed_configs.py 2>&1 | tee -a "${LOG_FILE}"

# 检查是否有失败的配置被补跑
RETRY_EXIT_CODE=${PIPESTATUS[0]}
if [ ${RETRY_EXIT_CODE} -eq 0 ]; then
    # 如果有失败的配置被补跑，重新生成 summary
    if grep -q "失败的配置数:" "${LOG_FILE}" && ! grep -q "没有发现失败的配置" "${LOG_FILE}"; then
        echo "Regenerating summary with retried configs..."
        python -c "
from main_phase_b import aggregate_phase_b_results
from pathlib import Path
aggregate_phase_b_results(Path('${OUTPUT_DIR}'))
" 2>&1 | tee -a "${LOG_FILE}"
    fi
fi

# -----------------------------------------------------------------------------
# Phase C
# -----------------------------------------------------------------------------
echo ""
echo "[3/4] Running Phase C..."
python main_phase_c.py \
    --output_dir "${OUTPUT_DIR}" \
    --phase_b_csv "${OUTPUT_DIR}/phase_b_tuning_summary.csv" \
    2>&1 | tee -a "${LOG_FILE}"

if [ ! -f "${OUTPUT_DIR}/phase_c_final_policy.json" ]; then
    echo "❌ Phase C failed!"
    exit 1
fi

# -----------------------------------------------------------------------------
# Phase D
# -----------------------------------------------------------------------------
echo ""
echo "[4/4] Running Phase D..."
python main_phase_d.py \
    --output_dir "${OUTPUT_DIR}" \
    --policy_json "${OUTPUT_DIR}/phase_c_final_policy.json" \
    --phase_b_csv "${OUTPUT_DIR}/phase_b_tuning_summary.csv" \
    2>&1 | tee -a "${LOG_FILE}"

if [ ! -f "${OUTPUT_DIR}/phase_d_summary.csv" ]; then
    echo "❌ Phase D failed!"
    exit 1
fi

echo ""
echo "========================================================================"
echo "✅ All phases completed!"
echo "========================================================================"
echo "日志: ${LOG_FILE}"

exit 0

