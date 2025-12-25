#!/usr/bin/env bash
# =============================================================================
# 完整训练脚本 - 单 GPU 版本
# =============================================================================
# 顺序运行 Baseline → Phase A → Phase B → Phase C → Phase D
# 全部使用单个 GPU (默认 GPU 0)
#
# 用法:
#   bash scripts/train_single_gpu.sh
#   CUDA_VISIBLE_DEVICES=1 bash scripts/train_single_gpu.sh  # 使用 GPU 1
#
# 预计时间 (A10 GPU):
#   Baseline: ~15 min
#   Phase A:  ~4-5 hours (256 configs × 200 epochs, 允许早停)
#   Phase B:  ~2-4 hours (ASHA 早停淘汰, rungs=[30,80,200])
#   Phase C:  ~6 hours (Greedy × 3 seeds × 800 epochs, 禁用早停)
#   Phase D:  ~6 hours (5 methods × 5 folds × 800 epochs, 禁用早停)
#   总计:     ~22-24 hours
# =============================================================================

set -euo pipefail

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="${PROJECT_ROOT}/logs"
OUTPUT_DIR="${PROJECT_ROOT}/outputs"

# 默认使用 GPU 0 (可通过 CUDA_VISIBLE_DEVICES 覆盖)
GPU_ID="${CUDA_VISIBLE_DEVICES:-0}"

# Environment variables
export KMP_DUPLICATE_LIB_OK=TRUE
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------
print_header() {
    echo ""
    echo "========================================================================"
    echo "$1"
    echo "========================================================================"
}

check_success() {
    local csv_file="$1"
    local phase_name="$2"
    if [ -f "$csv_file" ]; then
        local rows=$(tail -n +2 "$csv_file" | wc -l | tr -d ' ')
        echo "✅ ${phase_name} 完成: ${csv_file} (${rows} rows)"
        return 0
    else
        echo "❌ ${phase_name} 失败: ${csv_file} 未找到"
        return 1
    fi
}

# -----------------------------------------------------------------------------
# Main Script
# -----------------------------------------------------------------------------
cd "${PROJECT_ROOT}"

# Activate conda
eval "$(conda shell.bash hook)"
conda activate pga

mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}" "${OUTPUT_DIR}/checkpoints"

print_header "完整训练脚本 - 单 GPU 版本"
echo "开始时间: $(date)"
echo "GPU: ${GPU_ID}"
echo "日志目录: ${LOG_DIR}"
echo "输出目录: ${OUTPUT_DIR}"
echo ""
echo "早停策略 (v5.3):"
echo "  Phase A: min_epochs=100, patience=30"
echo "  Phase B: ASHA 多轮淘汰 (rungs=30,80,200, keep top 1/3)"
echo "  Phase C: min_epochs=500, patience=99999 (禁用)"
echo "  Phase D: min_epochs=500, patience=99999 (禁用)"

# -----------------------------------------------------------------------------
# Baseline
# -----------------------------------------------------------------------------
print_header "[1/5] Baseline 训练"
echo "配置: 200 epochs, min_epochs=100, patience=30"
START_TIME=$(date +%s)

CUDA_VISIBLE_DEVICES=${GPU_ID} python run_baseline.py \
    --epochs 200 \
    --min_epochs 100 \
    --early_stop_patience 30 \
    2>&1 | tee "${LOG_DIR}/baseline_${TIMESTAMP}.log"

END_TIME=$(date +%s)
echo "Baseline 耗时: $(( (END_TIME - START_TIME) / 60 )) 分钟"
check_success "${OUTPUT_DIR}/baseline_result.csv" "Baseline"

# -----------------------------------------------------------------------------
# Phase A
# -----------------------------------------------------------------------------
print_header "[2/5] Phase A 筛选"
echo "配置: 8 ops × 32 samples × 200 epochs, min_epochs=100, patience=30"
START_TIME=$(date +%s)

CUDA_VISIBLE_DEVICES=${GPU_ID} python main_phase_a.py \
    --epochs 200 \
    --n_samples 32 \
    --min_epochs 100 \
    --early_stop_patience 30 \
    --output_dir "${OUTPUT_DIR}" \
    --num_workers 6 \
    2>&1 | tee "${LOG_DIR}/phase_a_${TIMESTAMP}.log"

END_TIME=$(date +%s)
echo "Phase A 耗时: $(( (END_TIME - START_TIME) / 60 )) 分钟"
check_success "${OUTPUT_DIR}/phase_a_results.csv" "Phase A"

# -----------------------------------------------------------------------------
# Phase B (ASHA v5.3)
# -----------------------------------------------------------------------------
print_header "[3/5] Phase B ASHA 微调"
echo "配置: ASHA 早停淘汰赛, rungs=[30,80,200], Sobol 30 samples/op"
START_TIME=$(date +%s)

CUDA_VISIBLE_DEVICES=${GPU_ID} python main_phase_b.py \
    --rungs 30,80,200 \
    --n_samples 30 \
    --reduction_factor 3 \
    --phase_a_csv "${OUTPUT_DIR}/phase_a_results.csv" \
    --baseline_csv "${OUTPUT_DIR}/baseline_result.csv" \
    --output_dir "${OUTPUT_DIR}" \
    --num_workers 6 \
    2>&1 | tee "${LOG_DIR}/phase_b_${TIMESTAMP}.log"

END_TIME=$(date +%s)
echo "Phase B 耗时: $(( (END_TIME - START_TIME) / 60 )) 分钟"
check_success "${OUTPUT_DIR}/phase_b_tuning_summary.csv" "Phase B"

# -----------------------------------------------------------------------------
# Phase C
# -----------------------------------------------------------------------------
print_header "[4/5] Phase C 贪心组合"
echo "配置: Greedy × 3 seeds × 800 epochs, patience=99999 (禁用早停)"
START_TIME=$(date +%s)

CUDA_VISIBLE_DEVICES=${GPU_ID} python main_phase_c.py \
    --epochs 800 \
    --seeds 42,123,456 \
    --max_ops 3 \
    --min_epochs 500 \
    --early_stop_patience 99999 \
    --phase_b_csv "${OUTPUT_DIR}/phase_b_tuning_summary.csv" \
    --output_dir "${OUTPUT_DIR}" \
    --num_workers 6 \
    2>&1 | tee "${LOG_DIR}/phase_c_${TIMESTAMP}.log"

END_TIME=$(date +%s)
echo "Phase C 耗时: $(( (END_TIME - START_TIME) / 60 )) 分钟"
check_success "${OUTPUT_DIR}/phase_c_final_policy.json" "Phase C"

# -----------------------------------------------------------------------------
# Phase D
# -----------------------------------------------------------------------------
print_header "[5/5] Phase D SOTA 对比"
echo "配置: 5 methods × 5 folds × 800 epochs, patience=99999 (禁用早停)"
START_TIME=$(date +%s)

CUDA_VISIBLE_DEVICES=${GPU_ID} python main_phase_d.py \
    --epochs 800 \
    --seed 42 \
    --methods Baseline,RandAugment,Cutout,Ours_p1,Ours_optimal \
    --folds 0,1,2,3,4 \
    --min_epochs 500 \
    --early_stop_patience 99999 \
    --policy_json "${OUTPUT_DIR}/phase_c_final_policy.json" \
    --output_dir "${OUTPUT_DIR}" \
    --num_workers 6 \
    2>&1 | tee "${LOG_DIR}/phase_d_${TIMESTAMP}.log"

END_TIME=$(date +%s)
echo "Phase D 耗时: $(( (END_TIME - START_TIME) / 60 )) 分钟"
check_success "${OUTPUT_DIR}/phase_d_summary.csv" "Phase D"

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
print_header "训练完成"
echo "结束时间: $(date)"
echo ""
echo "输出文件:"
ls -la "${OUTPUT_DIR}"/*.csv "${OUTPUT_DIR}"/*.json 2>/dev/null || true
echo ""
echo "Checkpoints:"
ls -la "${OUTPUT_DIR}/checkpoints/" 2>/dev/null || true
echo ""
echo "最终结果:"
if [ -f "${OUTPUT_DIR}/phase_d_summary.csv" ]; then
    cat "${OUTPUT_DIR}/phase_d_summary.csv"
fi

exit 0

