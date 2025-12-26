#!/bin/bash
set -e
set -o pipefail

# ==============================================================================
# Configuration
# ==============================================================================
PHASE_B_CSV="outputs/phase_b_tuning_summary.csv"
PHASE_C_ARGS="--phase_b_csv $PHASE_B_CSV --top_k 6 --n_ops 2 --epochs 200 --seeds 42,123,456"
METHODS=("Baseline" "RandAugment" "Cutout" "Best_SingleOp" "Ours_optimal")
FOLDS=(0 1 2 3 4)
GPU_COUNT=4

mkdir -p outputs/logs
mkdir -p outputs/temp_results

# ==============================================================================
# Step 1: Cleanup
# ==============================================================================
echo "--------------------------------------------------------"
echo "STEP 1: Cleanup"
echo "--------------------------------------------------------"
rm -f outputs/phase_c_history.csv outputs/phase_c_final_policy.json
rm -f outputs/phase_d_results.csv outputs/phase_d_summary.csv
rm -rf outputs/checkpoints/phase_d_* outputs/checkpoints/phase_c_*
rm -rf outputs/temp_results/*
echo "Cleanup done."

# ==============================================================================
# Step 2: Phase C (Policy Construction)
# ==============================================================================
echo "--------------------------------------------------------"
echo "STEP 2: Phase C (Dynamic Policy Construction) - GPU 0"
echo "--------------------------------------------------------"
# Phase C is fast (<30 mins) and produces the policy needed for Phase D.
# Using tee to show progress in terminal while saving to log
CUDA_VISIBLE_DEVICES=0 python main_phase_c.py $PHASE_C_ARGS 2>&1 | tee outputs/logs/phase_c.log
echo "Phase C complete. Policy saved to outputs/phase_c_final_policy.json"

# ==============================================================================
# Step 3: Phase D (Evaluation) - Parallel
# ==============================================================================
echo "--------------------------------------------------------"
echo "STEP 3: Phase D (Evaluation) - 4 GPU Parallel"
echo "--------------------------------------------------------"

# 1. Generate Task List
# 5 Methods * 5 Folds = 25 Tasks
commands=()
for method in "${METHODS[@]}"; do
    for fold in "${FOLDS[@]}"; do
        # Use separate output dir for each task to avoid CSV race conditions
        out_dir="outputs/temp_results/${method}_fold${fold}"
        cmd="python main_phase_d.py --methods $method --folds $fold --epochs 200 --policy_json outputs/phase_c_final_policy.json --phase_b_csv $PHASE_B_CSV --output_dir $out_dir"
        commands+=("$cmd")
    done
done

# 2. Dispatch Tasks (Round Robin)
echo "Dispatching ${#commands[@]} tasks to $GPU_COUNT GPUs..."

pids=()
for gpu_id in {0..3}; do
    (
        # Execute every 4th task starting from gpu_id
        for i in "${!commands[@]}"; do
            if (( i % GPU_COUNT == gpu_id )); then
                cmd="${commands[$i]}"
                echo "[GPU $gpu_id] Running task $i: Method=${cmd#*--methods } (Fold=${cmd#*--folds })" 
                CUDA_VISIBLE_DEVICES=$gpu_id $cmd > "outputs/logs/task_d_${i}_gpu${gpu_id}.log" 2>&1
            fi
        done
        echo "[GPU $gpu_id] All assigned tasks completed."
    ) &
    pids+=($!)
done

# 3. Wait for Completion
for pid in "${pids[@]}"; do
    wait $pid
done

# ==============================================================================
# Step 4: Result Aggregation
# ==============================================================================
echo "--------------------------------------------------------"
echo "STEP 4: Aggregating Results"
echo "--------------------------------------------------------"

# Merge CSVs
header_file=$(find outputs/temp_results -name "phase_d_results.csv" | head -n 1)
if [ -z "$header_file" ]; then
    echo "ERROR: No results found! Check logs in outputs/logs/"
    exit 1
fi

# Write header
head -n 1 "$header_file" > outputs/phase_d_results.csv
# Write all bodies
find outputs/temp_results -name "phase_d_results.csv" -exec tail -n +2 -q {} + >> outputs/phase_d_results.csv
echo "Merged raw results to outputs/phase_d_results.csv"

# Generate Summary CSV using Python
python -c "
import pandas as pd
try:
    df = pd.read_csv('outputs/phase_d_results.csv')
    summary = df.groupby('op_name')[['val_acc', 'top5_acc']].agg(['mean', 'std'])
    summary.columns = ['_'.join(col) for col in summary.columns]
    summary['n_seeds'] = df.groupby('op_name')['seed'].count()
    summary.to_csv('outputs/phase_d_summary.csv')
    print('\nFinal Summary:')
    print(summary)
except Exception as e:
    print(f'Error generating summary: {e}')
"

echo "--------------------------------------------------------"
echo "FULL EXPERIMENT COMPLETED SUCCESSFULLY."
echo "--------------------------------------------------------"
