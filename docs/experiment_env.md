# 实验环境记录 (Experiment Environment Log)

## 1. 服务器配置

### Phase A/B/C/D + Baseline

| 项目 | 配置 |
|------|------|
| **云服务商** | 阿里云 |
| **实例规格** | ecs.gn7i-4x.8xlarge |
| **GPU** | 4 × NVIDIA A10 (4 × 24 GB) |
| **vCPU** | 32 vCPU |
| **内存** | 128 GiB |

### 系统环境

| 项目 | 版本 |
|------|------|
| **操作系统** | Ubuntu 24.04 64-bit |
| **CUDA** | 12.8 |
| **NVIDIA Driver** | 570.113.20 |
| **Python** | 3.14.2 (conda: pga) |

---

## 2. Baseline 运行

> **GPU 说明**: Baseline 只训练 1 次，默认使用 GPU 0。无需并行。

### 准备工作

```bash
# 首次运行前，创建日志目录
mkdir -p logs outputs
```

### 运行命令

```bash
# 运行 Baseline (默认使用 GPU 0)
python run_baseline.py | tee logs/baseline.log

# 指定 GPU (可选)
CUDA_VISIBLE_DEVICES=0 python run_baseline.py | tee logs/baseline.log

# 后台运行
CUDA_VISIBLE_DEVICES=0 nohup python run_baseline.py > logs/baseline.log 2>&1 &
```

### 输出文件

| 文件 | 说明 |
|------|------|
| `outputs/baseline_result.csv` | Baseline 结果 |
| `outputs/checkpoints/baseline_best.pth` | 最佳模型 checkpoint |
| `logs/baseline.log` | 运行日志 |

### 运行记录

| 日期 | 耗时 | 状态 |
|------|------|------|
| - | - | 待运行 |

---

## 3. Phase A 筛选

> **GPU 说明**: Phase A 顺序执行 256 个配置 (v5: 2D Sobol 采样 m×p)，默认使用 GPU 0。
> 支持 4-GPU 并行（见第 6 节）。

### 单 GPU 运行

```bash
# 冒烟测试
bash scripts/smoke_test_phase_a.sh

# 后台运行 (使用 GPU 0)
CUDA_VISIBLE_DEVICES=0 nohup python main_phase_a.py > logs/phase_a.log 2>&1 &

# 前台运行
CUDA_VISIBLE_DEVICES=0 python main_phase_a.py | tee logs/phase_a.log
```

### 可选参数

```bash
python main_phase_a.py \
    --epochs 200 \
    --n_samples 32 \
    --fold_idx 0 \
    --output_dir outputs \
    --seed 42 \
    --num_workers 6 \
    --early_stop_patience 5 \
    --ops RandomRotation,ColorJitter  # 可选，仅评估指定的 ops
```

### 输出文件

| 文件 | 说明 |
|------|------|
| `outputs/phase_a_results.csv` | Phase A 筛选结果 (256 configs, v5: 含 m 和 p) |
| `logs/phase_a.log` | 运行日志 |

### 运行记录

| 日期 | 耗时 | 状态 |
|------|------|------|
| - | - | 待运行 |

---

## 4. Phase B 深度微调

> **GPU 说明**: Phase B 默认使用 GPU 0 (v5: 2D Grid 搜索 m×p)，支持 4-GPU 并行（见第 7 节）。

### 单 GPU 运行

```bash
# 冒烟测试
bash scripts/smoke_test_phase_b.sh

# 后台运行 (使用 GPU 0，默认开启 deterministic)
CUDA_VISIBLE_DEVICES=0 nohup python main_phase_b.py > logs/phase_b.log 2>&1 &

# 前台运行
CUDA_VISIBLE_DEVICES=0 python main_phase_b.py | tee logs/phase_b.log

# 关闭 deterministic 以提高速度 (可选)
CUDA_VISIBLE_DEVICES=0 python main_phase_b.py --no_deterministic | tee logs/phase_b.log
```

### 可选参数

```bash
python main_phase_b.py \
    --epochs 200 \
    --seeds 42,123,456 \
    --output_dir outputs \
    --phase_a_csv outputs/phase_a_results.csv \
    --baseline_csv outputs/baseline_result.csv \
    --fold_idx 0 \
    --num_workers 6 \
    --early_stop_patience 5 \
    --top_k 4 \
    --grid_step 0.1 \
    --grid_n_steps 2 \
    --no_deterministic  # 可选，关闭确定性模式以提高速度
```

### 调试参数

```bash
# 仅调优指定的 ops
python main_phase_b.py --ops ColorJitter,GaussianBlur

# 快速测试模式
python main_phase_b.py --dry_run --epochs 2

# 限制每个 op 的网格点数 (用于测试)
python main_phase_b.py --grid_points 5
```

### 输入文件

| 文件 | 说明 |
|------|------|
| `outputs/phase_a_results.csv` | Phase A 筛选结果 (自动读取) |
| `outputs/baseline_result.csv` | Baseline 结果 (用于晋级判定) |

### 输出文件

| 文件 | 说明 |
|------|------|
| `outputs/phase_b_tuning_raw.csv` | 每个 (op, m, p, seed) 的原始结果 (v5: 2D Grid) |
| `outputs/phase_b_tuning_summary.csv` | 聚合结果，按 mean_val_acc 降序排列 |
| `logs/phase_b.log` | 运行日志 |

### 运行记录

| 日期 | 耗时 | 状态 |
|------|------|------|
| - | - | 待运行 |

---

## 5. Phase C/D (待实现)

> Phase C 和 Phase D 的代码尚未实现，后续添加。

---

## 6. 4-GPU 并行运行 Phase A (可选)

如需加速 Phase A，可使用 4 个 GPU 并行（8 ops 分配到 4 GPU）：

> **输出说明**：并行运行时，每个 GPU 的结果会保存到独立的子目录 (`outputs/gpu0/`, `outputs/gpu1/` 等)。
> 这是**正常现象**，避免多进程同时写入同一文件导致数据损坏。运行完成后需手动合并。
>
> **Phase A/B 共用目录不会冲突**：两阶段输出文件名不同（`phase_a_results.csv` vs `phase_b_tuning_raw.csv`）。

```bash
# 创建输出目录
mkdir -p outputs/gpu{0,1,2,3} logs

# 每个 GPU 分配 2 个 ops
CUDA_VISIBLE_DEVICES=0 python main_phase_a.py --ops RandomResizedCrop,RandomRotation --output_dir outputs/gpu0 > logs/phase_a_gpu0.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python main_phase_a.py --ops RandomPerspective,ColorJitter --output_dir outputs/gpu1 > logs/phase_a_gpu1.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 python main_phase_a.py --ops RandomGrayscale,GaussianBlur --output_dir outputs/gpu2 > logs/phase_a_gpu2.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 python main_phase_a.py --ops RandomErasing,GaussianNoise --output_dir outputs/gpu3 > logs/phase_a_gpu3.log 2>&1 &

wait
echo "All Phase A GPUs finished!"
```

**合并结果（必须执行）**：

```bash
# 合并所有 GPU 的 CSV 结果到主输出目录
head -1 outputs/gpu0/phase_a_results.csv > outputs/phase_a_results.csv
tail -n +2 -q outputs/gpu*/phase_a_results.csv >> outputs/phase_a_results.csv

# 验证合并结果
echo "Total rows: $(wc -l < outputs/phase_a_results.csv)"
# 预期: 257 行 (1 header + 8 ops × 32 samples = 257)
```

---

## 7. 4-GPU 并行运行 Phase B (可选)

如需加速 Phase B，可使用 4 个 GPU 并行：

> **输出说明**：与 Phase A 相同，每个 GPU 的结果会保存到独立的子目录。
> 这是**正常现象**，运行完成后需手动合并并重新生成 summary。
>
> **Phase A/B 共用目录不会冲突**：文件名不同（`phase_a_results.csv` vs `phase_b_tuning_raw.csv`），可安全共存。

```bash
# 创建输出目录 (如果 Phase A 已创建则可跳过)
mkdir -p outputs/gpu{0,1,2,3} logs

# 每个 GPU 分配 2 个 ops (根据 Phase A 晋级结果调整)
CUDA_VISIBLE_DEVICES=0 python main_phase_b.py --ops RandomResizedCrop,RandomRotation --output_dir outputs/gpu0 > logs/phase_b_gpu0.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python main_phase_b.py --ops RandomPerspective,ColorJitter --output_dir outputs/gpu1 > logs/phase_b_gpu1.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 python main_phase_b.py --ops RandomGrayscale,GaussianBlur --output_dir outputs/gpu2 > logs/phase_b_gpu2.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 python main_phase_b.py --ops RandomErasing,GaussianNoise --output_dir outputs/gpu3 > logs/phase_b_gpu3.log 2>&1 &

wait
echo "All Phase B GPUs finished!"
```

**合并结果（必须执行）**：

```bash
# 合并所有 GPU 的 raw CSV 结果
head -1 outputs/gpu0/phase_b_tuning_raw.csv > outputs/phase_b_tuning_raw.csv
tail -n +2 -q outputs/gpu*/phase_b_tuning_raw.csv >> outputs/phase_b_tuning_raw.csv

# 验证合并结果
echo "Total rows: $(wc -l < outputs/phase_b_tuning_raw.csv)"
```

**重新生成 summary（必须执行）**：

```bash
# 使用 Python 重新聚合生成 summary
python -c "
import pandas as pd
df = pd.read_csv('outputs/phase_b_tuning_raw.csv')
df = df[(df['error'].isna()) | (df['error'] == '')]
summary = df.groupby(['op_name', 'magnitude', 'probability']).agg(
    mean_val_acc=('val_acc', 'mean'),
    std_val_acc=('val_acc', 'std'),
    mean_top5_acc=('top5_acc', 'mean'),
    std_top5_acc=('top5_acc', 'std'),
    n_seeds=('seed', 'count'),
).reset_index()
summary = summary.fillna(0).round(4).sort_values('mean_val_acc', ascending=False)
summary.to_csv('outputs/phase_b_tuning_summary.csv', index=False)
print(f'Summary saved with {len(summary)} unique (op, m, p) configs')
"
```

> **注意**：Phase B 的 `--ops` 参数应根据 Phase A 的晋级结果调整，不一定是全部 8 个 ops。
