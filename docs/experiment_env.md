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

# 实时跟踪日志输出
tail -f logs/baseline.log
```

### 可选参数

```bash
python run_baseline.py \
    --epochs 200 \
    --min_epochs 100 \
    --early_stop_patience 30 \
    --batch_size 64 \
    --seed 42 \
    --num_workers 6
```

> **v5.1 早停策略**: Baseline 使用 `min_epochs=100, patience=30, monitor=val_acc`。

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

### 3.1 单 GPU 运行

```bash
# 冒烟测试
bash scripts/smoke_test_phase_a.sh

# 后台运行 (使用 GPU 0)
CUDA_VISIBLE_DEVICES=0 nohup python main_phase_a.py > logs/phase_a.log 2>&1 &

# 前台运行
CUDA_VISIBLE_DEVICES=0 python main_phase_a.py | tee logs/phase_a.log
```

### 3.2 4-GPU 并行运行 (推荐)

如需加速 Phase A，可使用 4 个 GPU 并行（8 ops 分配到 4 GPU）：

> **输出说明**：并行运行时，每个 GPU 的结果会保存到独立的子目录 (`outputs/gpu0/`, `outputs/gpu1/` 等)。
> 这是**正常现象**，避免多进程同时写入同一文件导致数据损坏。运行完成后需手动合并。

```bash
# 创建输出目录
mkdir -p outputs/gpu{0,1,2,3} logs

# 每个 GPU 分配 2 个 ops
CUDA_VISIBLE_DEVICES=0 nohup python -u main_phase_a.py --ops RandomResizedCrop,RandomRotation --output_dir outputs/gpu0 > logs/phase_a_gpu0.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python -u main_phase_a.py --ops RandomPerspective,ColorJitter --output_dir outputs/gpu1 > logs/phase_a_gpu1.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python -u main_phase_a.py --ops RandomGrayscale,GaussianBlur --output_dir outputs/gpu2 > logs/phase_a_gpu2.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup python -u main_phase_a.py --ops RandomErasing,GaussianNoise --output_dir outputs/gpu3 > logs/phase_a_gpu3.log 2>&1 &

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

### 可选参数

```bash
python main_phase_a.py \
    --epochs 200 \
    --n_samples 32 \
    --fold_idx 0 \
    --output_dir outputs \
    --seed 42 \
    --num_workers 6 \
    --min_epochs 100 \
    --early_stop_patience 30 \
    --ops RandomRotation,ColorJitter  # 可选，仅评估指定的 ops
```

> **v5.1 早停策略**: Phase A 使用 `min_epochs=100, patience=30, monitor=val_acc`。

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

## 4. Phase B ASHA 深度微调 (v5.3)

> **v5.3 更新**: Grid Search → ASHA 早停淘汰赛，速度提升 ~10 倍！

### 4.1 单 GPU 运行

```bash
# 冒烟测试
bash scripts/smoke_test_phase_b.sh

# 完整 ASHA 运行 (~2-4 小时)
CUDA_VISIBLE_DEVICES=0 nohup python main_phase_b.py > logs/phase_b.log 2>&1 &

# 前台运行
CUDA_VISIBLE_DEVICES=0 python main_phase_b.py | tee logs/phase_b.log

# 快速测试
CUDA_VISIBLE_DEVICES=0 python main_phase_b.py --n_samples 5 --dry_run
```

### 4.2 4-GPU 并行运行 (推荐)

```bash
mkdir -p outputs/gpu{0,1,2,3} logs

# 每个 GPU 分配 2 个 ops
CUDA_VISIBLE_DEVICES=0 nohup python -u main_phase_b.py --ops RandomResizedCrop,RandomRotation --output_dir outputs/gpu0 > logs/phase_b_gpu0.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python -u main_phase_b.py --ops RandomPerspective,ColorJitter --output_dir outputs/gpu1 > logs/phase_b_gpu1.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python -u main_phase_b.py --ops RandomGrayscale,GaussianBlur --output_dir outputs/gpu2 > logs/phase_b_gpu2.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup python -u main_phase_b.py --ops RandomErasing,GaussianNoise --output_dir outputs/gpu3 > logs/phase_b_gpu3.log 2>&1 &

wait
echo "All Phase B GPUs finished!"
```

**合并结果（必须执行）**：

```bash
head -1 outputs/gpu0/phase_b_tuning_raw.csv > outputs/phase_b_tuning_raw.csv
tail -n +2 -q outputs/gpu*/phase_b_tuning_raw.csv >> outputs/phase_b_tuning_raw.csv
echo "Total rows: $(wc -l < outputs/phase_b_tuning_raw.csv)"

# 重新生成 summary
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
print(summary.head(15).to_string(index=False))
"
```

### ASHA 参数说明

```bash
python main_phase_b.py \
    --rungs 30,80,200 \           # 检查点 epochs (每轮淘汰)
    --n_samples 30 \              # 每个 op 的 Sobol 采样数
    --reduction_factor 3 \        # 每轮保留 top 1/3
    --seed 42 \                   # Sobol 采样种子
    --output_dir outputs \
    --phase_a_csv outputs/phase_a_results.csv \
    --baseline_csv outputs/baseline_result.csv \
    --num_workers 8 \
    --ops ColorJitter,GaussianBlur  # 可选，仅调优指定 ops
```

> **ASHA 工作原理**:
> 1. Sobol 采样 30 个 (m, p) 点/op
> 2. Rung 1: 全部训练到 30 epochs，保留 top 1/3
> 3. Rung 2: 存活者续训到 80 epochs，再保留 top 1/3
> 4. Rung 3: 最终存活者训到 200 epochs

### 输入文件

| 文件 | 说明 |
|------|------|
| `outputs/phase_a_results.csv` | Phase A 筛选结果 |
| `outputs/baseline_result.csv` | Baseline 结果 |

### 输出文件

| 文件 | 说明 |
|------|------|
| `outputs/phase_b_tuning_raw.csv` | ASHA 最终存活配置结果 |
| `outputs/phase_b_tuning_summary.csv` | 按 mean_val_acc 降序排列 |
| `logs/phase_b.log` | 运行日志 |

### 运行记录

| 日期 | 耗时 | 状态 |
|------|------|------|
| - | - | 待运行 |

---

## 5. Phase C 先验贪心组合

> **GPU 说明**: Phase C 使用贪心算法逐步添加增强操作，每次候选需要训练 3 seeds × 800 epochs。

### 5.1 前置条件

- Phase B 完成，`outputs/phase_b_tuning_summary.csv` 存在
- 800 epochs 的 Baseline 结果（脚本会自动生成，或手动指定）

### 5.2 运行命令

```bash
# 冒烟测试
bash scripts/smoke_test_phase_c.sh

# 先运行 800-epoch Baseline（如果尚未运行）
CUDA_VISIBLE_DEVICES=0 python main_phase_c.py --run_baseline_only | tee logs/phase_c_baseline.log

# 运行完整 Phase C（800 epochs, 3 seeds）
CUDA_VISIBLE_DEVICES=0 nohup python main_phase_c.py > logs/phase_c.log 2>&1 &

# 前台运行
CUDA_VISIBLE_DEVICES=0 python main_phase_c.py | tee logs/phase_c.log

# 冒烟测试（2 epochs, 1 seed）
CUDA_VISIBLE_DEVICES=0 python main_phase_c.py --epochs 2 --dry_run | tee logs/phase_c_test.log
```

### 5.3 可选参数

```bash
python main_phase_c.py \
    --epochs 800 \
    --seeds 42,123,456 \
    --fold_idx 0 \
    --output_dir outputs \
    --phase_b_csv outputs/phase_b_tuning_summary.csv \
    --baseline_acc 35.5  # 可选，手动指定 800ep Baseline 准确率 \
    --max_ops 3 \
    --improvement_threshold 0.1 \
    --num_workers 6 \
    --early_stop_patience 99999  # 禁用早停
```

> **v5.1 早停策略**: Phase C 禁用早停 (`patience=99999`)，确保跑满 800 epochs。

### 5.4 输入文件

| 文件 | 说明 |
|------|------|
| `outputs/phase_b_tuning_summary.csv` | Phase B 最佳配置汇总 (自动读取) |
| `outputs/baseline_800ep_result.csv` | 800-epoch Baseline 结果 (可选，不存在则自动生成) |

### 5.5 输出文件

| 文件 | 说明 |
|------|------|
| `outputs/phase_c_history.csv` | 每次尝试添加操作的记录 |
| `outputs/phase_c_final_policy.json` | 最终策略定义 (供 Phase D 使用) |
| `outputs/baseline_800ep_result.csv` | 800-epoch Baseline 结果 |
| `outputs/checkpoints/phase_c_*.pth` | **v5.1 新增**: 各策略的 best checkpoint |
| `logs/phase_c.log` | 运行日志 |

### 5.6 算法说明

Phase C 使用贪心算法构建最终策略：

1. 初始化策略 P = S0，Acc(P) = Baseline_800ep_acc
2. 按 Phase B 的 mean_val_acc 排名，逐个尝试添加 Op(m*, p*)
3. 对每个候选 Op：
   - 检查互斥约束（如 RandomRotation 和 RandomPerspective 互斥）
   - 训练 P + Op × 3 seeds × 800 epochs
   - 如果 mean_acc > Acc(P) + 0.1%，则接受该 Op
4. 最多添加 3 个额外操作
5. 输出最终策略 P_final

### 5.7 运行记录

| 日期 | 耗时 | 状态 |
|------|------|------|
| - | - | 待运行 |

---

## 6. Phase D SOTA 对比实验

> **GPU 说明**: Phase D 在 5 个 Folds 上运行 5 种方法，共 25 次训练，每次 800 epochs。

### 6.1 前置条件

- Phase C 完成，`outputs/phase_c_final_policy.json` 存在
- 如果策略文件不存在，Ours 方法将退化为 Baseline

### 6.2 运行命令

```bash
# 冒烟测试
bash scripts/smoke_test_phase_d.sh

# 完整运行（5 方法 × 5 folds × 800 epochs）
CUDA_VISIBLE_DEVICES=0 nohup python main_phase_d.py > logs/phase_d.log 2>&1 &

# 前台运行
CUDA_VISIBLE_DEVICES=0 python main_phase_d.py | tee logs/phase_d.log

# 冒烟测试
CUDA_VISIBLE_DEVICES=0 python main_phase_d.py --epochs 2 --dry_run | tee logs/phase_d_test.log

# 只运行特定方法
CUDA_VISIBLE_DEVICES=0 python main_phase_d.py --methods Baseline,Ours_optimal

# 只运行特定 folds
CUDA_VISIBLE_DEVICES=0 python main_phase_d.py --folds 0,1,2
```

### 6.3 4-GPU 并行运行 (推荐)

由于 Phase D 涉及 5 个 folds，可将不同 fold 分配到不同 GPU：

```bash
# 创建日志目录
mkdir -p logs

# 分配 folds 到不同 GPU (注意：每个 GPU 会依次运行所有方法)
CUDA_VISIBLE_DEVICES=0 nohup python -u main_phase_d.py --folds 0,1 --output_dir outputs/phase_d_gpu0 > logs/phase_d_gpu0.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python -u main_phase_d.py --folds 2 --output_dir outputs/phase_d_gpu1 > logs/phase_d_gpu1.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python -u main_phase_d.py --folds 3 --output_dir outputs/phase_d_gpu2 > logs/phase_d_gpu2.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup python -u main_phase_d.py --folds 4 --output_dir outputs/phase_d_gpu3 > logs/phase_d_gpu3.log 2>&1 &

wait
echo "All Phase D GPUs finished!"
```

**合并结果（必须执行）**：

```bash
# 合并所有 GPU 的 CSV 结果
head -1 outputs/phase_d_gpu0/phase_d_results.csv > outputs/phase_d_results.csv
tail -n +2 -q outputs/phase_d_gpu*/phase_d_results.csv >> outputs/phase_d_results.csv

# 重新生成 summary
python -c "
import pandas as pd
df = pd.read_csv('outputs/phase_d_results.csv')
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
summary.to_csv('outputs/phase_d_summary.csv', index=False)
print('=== Final Results ===')
print(summary.to_string(index=False))
"

# 合并 checkpoints (只有 Ours_optimal 的)
mkdir -p outputs/checkpoints
cp outputs/phase_d_gpu*/checkpoints/phase_d_fold*_best.pth outputs/checkpoints/ 2>/dev/null || true
```

### 6.4 可选参数

```bash
python main_phase_d.py \
    --epochs 800 \
    --seed 42 \
    --output_dir outputs \
    --policy_json outputs/phase_c_final_policy.json \
    --methods Baseline,RandAugment,Cutout,Ours_p1,Ours_optimal \
    --folds 0,1,2,3,4 \
    --num_workers 6 \
    --early_stop_patience 99999  # 禁用早停
```

> **v5.1 早停策略**: Phase D 禁用早停 (`patience=99999`)，确保所有方法公平对比（相同训练 epochs）。

### 6.5 对比方法说明

| 方法 | 说明 | 参数 |
|------|------|------|
| **Baseline** | S0 基础增强 | RandomCrop(32, padding=4) + HorizontalFlip(p=0.5) |
| **RandAugment** | 自动增强 SOTA | N=2, M=9 (标准设置) |
| **Cutout** | 遮挡增强 SOTA | n_holes=1, length=16 |
| **Ours_p1** | 消融对照 | Phase C 策略，所有 p 强制为 1.0 |
| **Ours_optimal** | 最终方法 | Phase C 策略，使用优化的 (m, p) |

### 6.6 输入文件

| 文件 | 说明 |
|------|------|
| `outputs/phase_c_final_policy.json` | Phase C 最终策略 (必需，否则 Ours 退化为 Baseline) |

### 6.7 输出文件

| 文件 | 说明 |
|------|------|
| `outputs/phase_d_results.csv` | 每个 (method, fold) 的原始结果 |
| `outputs/phase_d_summary.csv` | Mean ± Std 汇总 (用于论文表格) |
| `outputs/checkpoints/phase_d_fold{0-4}_best.pth` | 最终模型 checkpoint (仅 Ours_optimal) |
| `logs/phase_d.log` | 运行日志 |

### 6.8 运行记录

| 日期 | 耗时 | 状态 |
|------|------|------|
| - | - | 待运行 |

---

## 7. 一键运行脚本

### 7.1 综合冒烟测试

验证整个实验流程的正确性（约 5-10 分钟）：

```bash
bash scripts/smoke_test_all.sh
```

### 7.2 完整训练 - 单 GPU

顺序运行 Baseline → A → B → C → D，约 22-24 小时：

```bash
bash scripts/train_single_gpu.sh

# 或指定 GPU
CUDA_VISIBLE_DEVICES=1 bash scripts/train_single_gpu.sh
```

### 7.3 完整训练 - 多 GPU (推荐)

Baseline 和 Phase C 单 GPU，Phase A/B/D 使用 4 GPU 并行，约 10-11 小时：

```bash
bash scripts/train_multi_gpu.sh
```

### 脚本说明

| 脚本 | 说明 | 预计时间 |
|------|------|----------|
| `scripts/smoke_test_all.sh` | 综合冒烟测试 (Baseline + ABCD) | ~5-10 min |
| `scripts/train_single_gpu.sh` | 单 GPU 完整训练 | ~22-24h |
| `scripts/train_multi_gpu.sh` | 混合 GPU 完整训练 | ~10-11h |

---

## 8. 计算量估计 (v5.1 更新)

| 阶段 | 配置数 | 早停策略 | 预计时间 (单 GPU) | 预计时间 (4 GPU) |
|------|--------|----------|------------------|-----------------|
| Baseline | 1 × 200 ep | - | ~15 min | ~15 min |
| Phase A | 8 ops × 32 点 × 200 ep | 允许早停 | ~4-5h | ~1-1.2h |
| Phase B (ASHA) | ~8 ops × 30 samples, rungs=[30,80,200] | ASHA 淘汰 | ~2-4h | ~0.5-1h |
| Phase C | ~8 ops × 3 seeds × 800 ep | **禁用** | ~6h | ~6h (单GPU) |
| Phase D | 5 methods × 5 folds × 800 ep | **禁用** | ~6h | ~1.5h |

> **v5.1 注意**: 
> - Phase A/B: 使用 min_epochs + patience 早停，可能提前结束
> - Phase C/D: **禁用早停**，确保跑满 800 epochs 以保证公平对比
> - Phase C 因贪心算法串行执行，无法多 GPU 并行；Phase D 可按 fold 分配到 4 GPU

---

## 9. 输出文件汇总

```
outputs/
├── baseline_result.csv           # Baseline 200ep 结果
├── baseline_800ep_result.csv     # Baseline 800ep 结果 (Phase C 生成)
├── phase_a_results.csv           # Phase A 筛选结果
├── phase_b_tuning_raw.csv        # Phase B 原始结果
├── phase_b_tuning_summary.csv    # Phase B 汇总结果
├── phase_c_history.csv           # Phase C 组合历史
├── phase_c_final_policy.json     # Phase C 最终策略
├── phase_d_results.csv           # Phase D 原始结果
├── phase_d_summary.csv           # Phase D 汇总结果 (论文用)
└── checkpoints/
    ├── baseline_best.pth           # Baseline 最佳模型
    ├── phase_c_*.pth               # Phase C 各策略最佳模型
    └── phase_d_fold{0-4}_best.pth  # Phase D 最终模型 (5-fold)
