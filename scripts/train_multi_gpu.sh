#!/usr/bin/env bash
# =============================================================================
# 完整训练脚本 - 混合 GPU 版本
# =============================================================================
# Baseline 和 Phase C 使用单 GPU (串行)
# Phase A, B, D 使用 4 GPU 并行
#
# 用法:
#   bash scripts/train_multi_gpu.sh
#
# 预计时间 (4 × A10 GPU):
#   Baseline: ~15 min (单GPU)
#   Phase A:  ~1-1.5 hours (4 GPU 并行)
#   Phase B:  ~0.5-1 hour (ASHA, 4 GPU 并行)
#   Phase C:  ~6 hours (单GPU, 贪心算法串行)
#   Phase D:  ~1.5 hours (4 GPU 并行)
#   总计:     ~10-11 hours
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

# GPU 分配
SINGLE_GPU=0  # Baseline 和 Phase C 使用的 GPU
MULTI_GPUS=(0 1 2 3)  # Phase A, B, D 并行使用的 GPU

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

wait_for_jobs() {
    echo "等待所有后台任务完成..."
    wait
    echo "所有任务完成!"
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

merge_csv() {
    local output_file="$1"
    local pattern="$2"
    shift 2
    local sources=("$@")
    
    # Get header from first file
    head -1 "${sources[0]}" > "$output_file"
    
    # Append data from all files
    for src in "${sources[@]}"; do
        if [ -f "$src" ]; then
            tail -n +2 "$src" >> "$output_file"
        fi
    done
    
    local rows=$(tail -n +2 "$output_file" | wc -l | tr -d ' ')
    echo "合并完成: $output_file ($rows rows)"
}

# -----------------------------------------------------------------------------
# Main Script
# -----------------------------------------------------------------------------
cd "${PROJECT_ROOT}"

# Activate conda
eval "$(conda shell.bash hook)"
conda activate pga

mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}" "${OUTPUT_DIR}/checkpoints"
mkdir -p "${OUTPUT_DIR}/gpu0" "${OUTPUT_DIR}/gpu1" "${OUTPUT_DIR}/gpu2" "${OUTPUT_DIR}/gpu3"

print_header "完整训练脚本 - 混合 GPU 版本"
echo "开始时间: $(date)"
echo "单 GPU: ${SINGLE_GPU} (用于 Baseline, Phase C)"
echo "多 GPU: ${MULTI_GPUS[*]} (用于 Phase A, B, D)"
echo "日志目录: ${LOG_DIR}"
echo "输出目录: ${OUTPUT_DIR}"
echo ""
echo "早停策略 (v5.3):"
echo "  Phase A: min_epochs=100, patience=30"
echo "  Phase B: ASHA 多轮淘汰 (rungs=30,80,200, keep top 1/3)"
echo "  Phase C: min_epochs=500, patience=99999 (禁用)"
echo "  Phase D: min_epochs=500, patience=99999 (禁用)"

# -----------------------------------------------------------------------------
# Baseline (单 GPU)
# -----------------------------------------------------------------------------
print_header "[1/5] Baseline 训练 (单 GPU)"
echo "配置: 200 epochs, min_epochs=100, patience=30"
START_TIME=$(date +%s)

CUDA_VISIBLE_DEVICES=${SINGLE_GPU} python run_baseline.py \
    --epochs 200 \
    --min_epochs 100 \
    --early_stop_patience 30 \
    2>&1 | tee "${LOG_DIR}/baseline_${TIMESTAMP}.log"

END_TIME=$(date +%s)
echo "Baseline 耗时: $(( (END_TIME - START_TIME) / 60 )) 分钟"
check_success "${OUTPUT_DIR}/baseline_result.csv" "Baseline"

# -----------------------------------------------------------------------------
# Phase A (4 GPU 并行)
# -----------------------------------------------------------------------------
print_header "[2/5] Phase A 筛选 (4 GPU 并行)"
echo "配置: 8 ops 分配到 4 GPU, 每 GPU 2 ops × 32 samples × 200 epochs"
START_TIME=$(date +%s)

# 8 ops 分配到 4 GPU (每个 2 ops)
OPS_GPU0="RandomResizedCrop,RandomRotation"
OPS_GPU1="RandomPerspective,ColorJitter"
OPS_GPU2="RandomGrayscale,GaussianBlur"
OPS_GPU3="RandomErasing,GaussianNoise"

CUDA_VISIBLE_DEVICES=0 nohup python -u main_phase_a.py \
    --epochs 200 \
    --n_samples 32 \
    --min_epochs 100 \
    --early_stop_patience 30 \
    --ops ${OPS_GPU0} \
    --output_dir "${OUTPUT_DIR}/gpu0" \
    --num_workers 6 \
    > "${LOG_DIR}/phase_a_gpu0_${TIMESTAMP}.log" 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python -u main_phase_a.py \
    --epochs 200 \
    --n_samples 32 \
    --min_epochs 100 \
    --early_stop_patience 30 \
    --ops ${OPS_GPU1} \
    --output_dir "${OUTPUT_DIR}/gpu1" \
    --num_workers 6 \
    > "${LOG_DIR}/phase_a_gpu1_${TIMESTAMP}.log" 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup python -u main_phase_a.py \
    --epochs 200 \
    --n_samples 32 \
    --min_epochs 100 \
    --early_stop_patience 30 \
    --ops ${OPS_GPU2} \
    --output_dir "${OUTPUT_DIR}/gpu2" \
    --num_workers 6 \
    > "${LOG_DIR}/phase_a_gpu2_${TIMESTAMP}.log" 2>&1 &

CUDA_VISIBLE_DEVICES=3 nohup python -u main_phase_a.py \
    --epochs 200 \
    --n_samples 32 \
    --min_epochs 100 \
    --early_stop_patience 30 \
    --ops ${OPS_GPU3} \
    --output_dir "${OUTPUT_DIR}/gpu3" \
    --num_workers 6 \
    > "${LOG_DIR}/phase_a_gpu3_${TIMESTAMP}.log" 2>&1 &

wait_for_jobs

END_TIME=$(date +%s)
echo "Phase A 耗时: $(( (END_TIME - START_TIME) / 60 )) 分钟"

# 合并 Phase A 结果
merge_csv "${OUTPUT_DIR}/phase_a_results.csv" "phase_a" \
    "${OUTPUT_DIR}/gpu0/phase_a_results.csv" \
    "${OUTPUT_DIR}/gpu1/phase_a_results.csv" \
    "${OUTPUT_DIR}/gpu2/phase_a_results.csv" \
    "${OUTPUT_DIR}/gpu3/phase_a_results.csv"

check_success "${OUTPUT_DIR}/phase_a_results.csv" "Phase A"

# -----------------------------------------------------------------------------
# Phase B ASHA (4 GPU 并行)
# -----------------------------------------------------------------------------
print_header "[3/5] Phase B ASHA 微调 (4 GPU 并行)"
echo "配置: ASHA 早停淘汰赛, rungs=[30,80,200], 每 GPU 处理 2 ops"
START_TIME=$(date +%s)

CUDA_VISIBLE_DEVICES=0 nohup python -u main_phase_b.py \
    --rungs 30,80,200 \
    --n_samples 30 \
    --reduction_factor 3 \
    --ops ${OPS_GPU0} \
    --phase_a_csv "${OUTPUT_DIR}/phase_a_results.csv" \
    --baseline_csv "${OUTPUT_DIR}/baseline_result.csv" \
    --output_dir "${OUTPUT_DIR}/gpu0" \
    --num_workers 6 \
    > "${LOG_DIR}/phase_b_gpu0_${TIMESTAMP}.log" 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python -u main_phase_b.py \
    --rungs 30,80,200 \
    --n_samples 30 \
    --reduction_factor 3 \
    --ops ${OPS_GPU1} \
    --phase_a_csv "${OUTPUT_DIR}/phase_a_results.csv" \
    --baseline_csv "${OUTPUT_DIR}/baseline_result.csv" \
    --output_dir "${OUTPUT_DIR}/gpu1" \
    --num_workers 6 \
    > "${LOG_DIR}/phase_b_gpu1_${TIMESTAMP}.log" 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup python -u main_phase_b.py \
    --rungs 30,80,200 \
    --n_samples 30 \
    --reduction_factor 3 \
    --ops ${OPS_GPU2} \
    --phase_a_csv "${OUTPUT_DIR}/phase_a_results.csv" \
    --baseline_csv "${OUTPUT_DIR}/baseline_result.csv" \
    --output_dir "${OUTPUT_DIR}/gpu2" \
    --num_workers 6 \
    > "${LOG_DIR}/phase_b_gpu2_${TIMESTAMP}.log" 2>&1 &

CUDA_VISIBLE_DEVICES=3 nohup python -u main_phase_b.py \
    --rungs 30,80,200 \
    --n_samples 30 \
    --reduction_factor 3 \
    --ops ${OPS_GPU3} \
    --phase_a_csv "${OUTPUT_DIR}/phase_a_results.csv" \
    --baseline_csv "${OUTPUT_DIR}/baseline_result.csv" \
    --output_dir "${OUTPUT_DIR}/gpu3" \
    --num_workers 6 \
    > "${LOG_DIR}/phase_b_gpu3_${TIMESTAMP}.log" 2>&1 &

wait_for_jobs

END_TIME=$(date +%s)
echo "Phase B 耗时: $(( (END_TIME - START_TIME) / 60 )) 分钟"

# 合并 Phase B 结果
merge_csv "${OUTPUT_DIR}/phase_b_tuning_raw.csv" "phase_b_raw" \
    "${OUTPUT_DIR}/gpu0/phase_b_tuning_raw.csv" \
    "${OUTPUT_DIR}/gpu1/phase_b_tuning_raw.csv" \
    "${OUTPUT_DIR}/gpu2/phase_b_tuning_raw.csv" \
    "${OUTPUT_DIR}/gpu3/phase_b_tuning_raw.csv"

# 重新生成 summary
python -c "
import pandas as pd
df = pd.read_csv('${OUTPUT_DIR}/phase_b_tuning_raw.csv')
df = df[(df['error'].isna()) | (df['error'] == '')]
summary = df.groupby(['op_name', 'magnitude', 'probability']).agg(
    mean_val_acc=('val_acc', 'mean'),
    std_val_acc=('val_acc', 'std'),
    mean_top5_acc=('top5_acc', 'mean'),
    std_top5_acc=('top5_acc', 'std'),
    n_seeds=('seed', 'count'),
).reset_index()
summary = summary.fillna(0).round(4).sort_values('mean_val_acc', ascending=False)
summary.to_csv('${OUTPUT_DIR}/phase_b_tuning_summary.csv', index=False)
print('Phase B Summary 生成完成')
print(summary.head(10).to_string(index=False))
"

check_success "${OUTPUT_DIR}/phase_b_tuning_summary.csv" "Phase B"

# -----------------------------------------------------------------------------
# Phase C (单 GPU - 贪心算法必须串行)
# -----------------------------------------------------------------------------
print_header "[4/5] Phase C 贪心组合 (单 GPU)"
echo "配置: Greedy × 3 seeds × 800 epochs, patience=99999 (禁用早停)"
echo "注意: Phase C 使用贪心算法，无法并行化"
START_TIME=$(date +%s)

CUDA_VISIBLE_DEVICES=${SINGLE_GPU} python main_phase_c.py \
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
# Phase D (4 GPU 并行 - 按 fold 分配)
# -----------------------------------------------------------------------------
print_header "[5/5] Phase D SOTA 对比 (4 GPU 并行)"
echo "配置: 5 methods × 5 folds × 800 epochs, 按 fold 分配到 4 GPU"
START_TIME=$(date +%s)

# 5 folds 分配到 4 GPU: GPU0=fold0,1, GPU1=fold2, GPU2=fold3, GPU3=fold4
CUDA_VISIBLE_DEVICES=0 nohup python -u main_phase_d.py \
    --epochs 800 \
    --seed 42 \
    --methods Baseline,RandAugment,Cutout,Ours_p1,Ours_optimal \
    --folds 0,1 \
    --min_epochs 500 \
    --early_stop_patience 99999 \
    --policy_json "${OUTPUT_DIR}/phase_c_final_policy.json" \
    --output_dir "${OUTPUT_DIR}/gpu0" \
    --num_workers 6 \
    > "${LOG_DIR}/phase_d_gpu0_${TIMESTAMP}.log" 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python -u main_phase_d.py \
    --epochs 800 \
    --seed 42 \
    --methods Baseline,RandAugment,Cutout,Ours_p1,Ours_optimal \
    --folds 2 \
    --min_epochs 500 \
    --early_stop_patience 99999 \
    --policy_json "${OUTPUT_DIR}/phase_c_final_policy.json" \
    --output_dir "${OUTPUT_DIR}/gpu1" \
    --num_workers 6 \
    > "${LOG_DIR}/phase_d_gpu1_${TIMESTAMP}.log" 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup python -u main_phase_d.py \
    --epochs 800 \
    --seed 42 \
    --methods Baseline,RandAugment,Cutout,Ours_p1,Ours_optimal \
    --folds 3 \
    --min_epochs 500 \
    --early_stop_patience 99999 \
    --policy_json "${OUTPUT_DIR}/phase_c_final_policy.json" \
    --output_dir "${OUTPUT_DIR}/gpu2" \
    --num_workers 6 \
    > "${LOG_DIR}/phase_d_gpu2_${TIMESTAMP}.log" 2>&1 &

CUDA_VISIBLE_DEVICES=3 nohup python -u main_phase_d.py \
    --epochs 800 \
    --seed 42 \
    --methods Baseline,RandAugment,Cutout,Ours_p1,Ours_optimal \
    --folds 4 \
    --min_epochs 500 \
    --early_stop_patience 99999 \
    --policy_json "${OUTPUT_DIR}/phase_c_final_policy.json" \
    --output_dir "${OUTPUT_DIR}/gpu3" \
    --num_workers 6 \
    > "${LOG_DIR}/phase_d_gpu3_${TIMESTAMP}.log" 2>&1 &

wait_for_jobs

END_TIME=$(date +%s)
echo "Phase D 耗时: $(( (END_TIME - START_TIME) / 60 )) 分钟"

# 合并 Phase D 结果
merge_csv "${OUTPUT_DIR}/phase_d_results.csv" "phase_d" \
    "${OUTPUT_DIR}/gpu0/phase_d_results.csv" \
    "${OUTPUT_DIR}/gpu1/phase_d_results.csv" \
    "${OUTPUT_DIR}/gpu2/phase_d_results.csv" \
    "${OUTPUT_DIR}/gpu3/phase_d_results.csv"

# 重新生成 summary
python -c "
import pandas as pd
df = pd.read_csv('${OUTPUT_DIR}/phase_d_results.csv')
df = df[(df['error'].isna()) | (df['error'] == '')]
summary = df.groupby('op_name').agg(
    mean_val_acc=('val_acc', 'mean'),
    std_val_acc=('val_acc', 'std'),
    mean_top5_acc=('top5_acc', 'mean'),
    std_top5_acc=('top5_acc', 'std'),
    n_folds=('fold_idx', 'count'),
).reset_index()
summary = summary.rename(columns={'op_name': 'method'})
summary = summary.fillna(0).round(4).sort_values('mean_val_acc', ascending=False)
summary.to_csv('${OUTPUT_DIR}/phase_d_summary.csv', index=False)
print('Phase D Summary 生成完成')
print(summary.to_string(index=False))
"

# 合并 checkpoints
echo "合并 checkpoints..."
cp "${OUTPUT_DIR}"/gpu*/checkpoints/phase_d_fold*_best.pth "${OUTPUT_DIR}/checkpoints/" 2>/dev/null || true
cp "${OUTPUT_DIR}"/checkpoints/phase_c_*.pth "${OUTPUT_DIR}/checkpoints/" 2>/dev/null || true

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
echo "最终结果 (Phase D Summary):"
if [ -f "${OUTPUT_DIR}/phase_d_summary.csv" ]; then
    cat "${OUTPUT_DIR}/phase_d_summary.csv"
fi

exit 0

