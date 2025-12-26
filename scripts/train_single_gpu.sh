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
#   Phase 0:  ~1 hour (可选, 仅首次运行)
#   Baseline: ~15 min
#   Phase A:  ~1 hour (256 configs × 40 epochs)
#   Phase B:  ~2-4 hours (ASHA 早停淘汰)
#   Phase C:  ~2-3 hours (贪心搜索 × 3 seeds × 200 epochs)
#   Phase D:  ~2 hours (6 methods × 5 folds × 200 epochs)
#   总计:     ~8-12 hours (含 Phase 0)
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
echo "训练策略:"
echo "  Phase A: 40ep 低保真筛选"
echo "  Phase B: ASHA 多轮淘汰 (rungs=40,100,200)"
echo "  Phase C: 贪心组合搜索, min_improvement=0.1%, t-test (p<0.2)"
echo "  Phase D: 200ep, 5-fold 交叉验证"

# -----------------------------------------------------------------------------
# Phase 0: 超参校准 (可选，如已完成可跳过)
# -----------------------------------------------------------------------------
if [ ! -f "${OUTPUT_DIR}/phase0_summary.csv" ]; then
    print_header "[0/5] Phase 0 超参校准"
    echo "配置: 100 epochs × 3 seeds × 12 configs (4 wd × 3 ls)"
    echo "目的: 确定最优 weight_decay 和 label_smoothing"
    START_TIME=$(date +%s)
    
    CUDA_VISIBLE_DEVICES=${GPU_ID} python run_phase0_calibration.py \
        --output_dir "${OUTPUT_DIR}" \
        2>&1 | tee "${LOG_DIR}/phase0_${TIMESTAMP}.log"
    
    END_TIME=$(date +%s)
    echo "Phase 0 耗时: $(( (END_TIME - START_TIME) / 60 )) 分钟"
    check_success "${OUTPUT_DIR}/phase0_summary.csv" "Phase 0"
else
    echo ""
    echo "跳过 Phase 0: ${OUTPUT_DIR}/phase0_summary.csv 已存在"
    echo "如需重新校准，请删除该文件后重新运行"
fi

# -----------------------------------------------------------------------------
# Baseline
# -----------------------------------------------------------------------------
print_header "[1/5] Baseline 训练"
echo "配置: 200 epochs, min_epochs=60, patience=60"
START_TIME=$(date +%s)

CUDA_VISIBLE_DEVICES=${GPU_ID} python run_baseline.py \
    2>&1 | tee "${LOG_DIR}/baseline_${TIMESTAMP}.log"

END_TIME=$(date +%s)
echo "Baseline 耗时: $(( (END_TIME - START_TIME) / 60 )) 分钟"
check_success "${OUTPUT_DIR}/baseline_result.csv" "Baseline"

# -----------------------------------------------------------------------------
# Phase A (v5.5: 40ep 低保真筛选)
# -----------------------------------------------------------------------------
print_header "[2/5] Phase A 筛选"
echo "配置: 8 ops × 32 samples × 40 epochs (v5.5 低保真)"
START_TIME=$(date +%s)

CUDA_VISIBLE_DEVICES=${GPU_ID} python main_phase_a.py \
    --output_dir "${OUTPUT_DIR}" \
    2>&1 | tee "${LOG_DIR}/phase_a_${TIMESTAMP}.log"

END_TIME=$(date +%s)
echo "Phase A 耗时: $(( (END_TIME - START_TIME) / 60 )) 分钟"
check_success "${OUTPUT_DIR}/phase_a_results.csv" "Phase A"

# -----------------------------------------------------------------------------
# Phase B (v5.5: rungs [40,100,200])
# -----------------------------------------------------------------------------
print_header "[3/5] Phase B ASHA 微调"
echo "配置: ASHA 早停淘汰赛, rungs=[40,100,200], Sobol 30 samples/op (v5.5)"
START_TIME=$(date +%s)

CUDA_VISIBLE_DEVICES=${GPU_ID} python main_phase_b.py \
    --output_dir "${OUTPUT_DIR}" \
    2>&1 | tee "${LOG_DIR}/phase_b_${TIMESTAMP}.log"

END_TIME=$(date +%s)
echo "Phase B 耗时: $(( (END_TIME - START_TIME) / 60 )) 分钟"
check_success "${OUTPUT_DIR}/phase_b_tuning_summary.csv" "Phase B"

# -----------------------------------------------------------------------------
# Phase C (贪心组合搜索 + 验证)
# -----------------------------------------------------------------------------
print_header "[4/5] Phase C 贪心组合"
echo "配置: Greedy × 3 seeds × 200 epochs, min_improvement=0.1%, t-test"
START_TIME=$(date +%s)

CUDA_VISIBLE_DEVICES=${GPU_ID} python main_phase_c.py \
    --output_dir "${OUTPUT_DIR}" \
    --phase_b_csv "${OUTPUT_DIR}/phase_b_tuning_summary.csv" \
    --max_ops 3 \
    --min_improvement 0.1 \
    --p_any_target 0.7 \
    2>&1 | tee "${LOG_DIR}/phase_c_${TIMESTAMP}.log"

END_TIME=$(date +%s)
echo "Phase C 耗时: $(( (END_TIME - START_TIME) / 60 )) 分钟"
check_success "${OUTPUT_DIR}/phase_c_final_policy.json" "Phase C"

# -----------------------------------------------------------------------------
# Phase D (SOTA 对比, 含 Best_SingleOp)
# -----------------------------------------------------------------------------
print_header "[5/5] Phase D SOTA 对比"
echo "配置: 6 methods × 5 folds × 200 epochs"
START_TIME=$(date +%s)

CUDA_VISIBLE_DEVICES=${GPU_ID} python main_phase_d.py \
    --policy_json "${OUTPUT_DIR}/phase_c_final_policy.json" \
    --phase_b_csv "${OUTPUT_DIR}/phase_b_tuning_summary.csv" \
    --output_dir "${OUTPUT_DIR}" \
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

