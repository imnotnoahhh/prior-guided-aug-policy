# 实验环境记录 (Experiment Environment Log)

## 1. 服务器配置

> 快捷运行：推荐使用 `bash scripts/train_single_gpu.sh`（单卡串行全流程）。下文保留逐阶段命令与多卡并行/合并说明。

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
| **CUDA** | 12.8.1 |
| **NVIDIA Driver** | 570.195.03 |
| **Python** | 3.14.2 (conda: pga) |
| **cuDNN** | 9.8.0.87 |

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
    --min_epochs 60 \
    --early_stop_patience 60 \
    --batch_size 128 \
    --seed 42 \
    --num_workers 8
```

> **v5.1 早停策略**: Baseline 使用 `min_epochs=60, patience=60, monitor=val_acc`。

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

> **v5.5 更新**: 40ep 低保真筛选，节省 80% 计算量！
> **GPU 说明**: Phase A 顺序执行 256 个配置 (v5: 2D Sobol 采样 m×p)，默认使用 GPU 0。

### 3.1 单 GPU 运行

```bash
# 冒烟测试（1ep）
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
    --epochs 40 \      # v5.5: 低保真筛选
    --n_samples 32 \
    --n_promote 8 \    # v5.5: 每 op 晋级 8 个配置
    --fold_idx 0 \
    --output_dir outputs \
    --seed 42 \
    --num_workers 8 \
    --min_epochs 20 \  # v5.5: 适配 40ep
    --early_stop_patience 15 \
    --ops RandomRotation,ColorJitter  # 可选，仅评估指定的 ops
```

> **v5.5 变更**: 
> - 40ep 低保真筛选，与 200ep 最终性能高度相关 (ρ > 0.8)
> - 新增 stable_score 评分: `mean(top3(val_acc[30:40]))`

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

## 4. Phase B ASHA 深度微调 (v5.5)

> **v5.5 更新**: Rungs 调整为 [40,100,200]，与 Phase A 对齐！
> **v5.4 更新**: 最终 Rung 使用多 seed (3 seeds) 评估，提高排名稳定性！
> **v5.3 更新**: Grid Search → ASHA 早停淘汰赛，速度提升 ~10 倍！

### 4.1 单 GPU 运行

```bash
# 冒烟测试（1ep）
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
    --rungs 40,100,200 \         # 检查点 epochs (每轮淘汰)
    --n_samples 30 \              # 每个 op 的 Sobol 采样数
    --reduction_factor 2 \        # 每轮保留 top 1/2
    --seed 42 \                   # Sobol 采样种子
    --output_dir outputs \
    --phase_a_csv outputs/phase_a_results.csv \
    --baseline_csv outputs/baseline_result.csv \
    --num_workers 8 \
    --ops ColorJitter,GaussianBlur  # 可选，仅调优指定 ops
```

> **ASHA 工作原理**:
> 1. Sobol 采样 30 个 (m, p) 点/op
> 2. Rung 1: 全部训练到 40 epochs，保留 top 1/2
> 3. Rung 2: 存活者续训到 100 epochs，再保留 top 1/2
> 4. Rung 3: 最终存活者训到 200 epochs（final rung 多 seed）

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

## 5. Phase C 先验贪心组合 (v5.4 Multi-Start)

> **v5.4 更新**: Multi-Start 搜索！同时从 Phase A 和 Phase B 的 top 配置出发，选择最优路径。
> **GPU 说明**: Phase C 使用贪心算法逐步添加增强操作，每次候选需要训练 3 seeds × 200 epochs（与 A/B 一致）。

### 5.1 前置条件

- Phase A 完成，`outputs/phase_a_results.csv` 存在（用于多起点搜索）
- Phase B 完成，`outputs/phase_b_tuning_summary.csv` 存在
- Baseline 结果存在（`outputs/baseline_result.csv`，与 Phase A/B 共用）

### 5.2 运行命令

```bash
# 冒烟测试
bash scripts/smoke_test_phase_c.sh

# 运行完整 Phase C（200 epochs, 3 seeds）
CUDA_VISIBLE_DEVICES=0 nohup python main_phase_c.py > logs/phase_c.log 2>&1 &

# 前台运行
CUDA_VISIBLE_DEVICES=0 python main_phase_c.py | tee logs/phase_c.log

# 冒烟测试（2 epochs, 1 seed）
CUDA_VISIBLE_DEVICES=0 python main_phase_c.py --epochs 2 --dry_run | tee logs/phase_c_test.log
```

### 5.3 可选参数

```bash
python main_phase_c.py \
    --epochs 200 \
    --seeds 42,123,456 \
    --fold_idx 0 \
    --output_dir outputs \
    --phase_b_csv outputs/phase_b_tuning_summary.csv \
    --baseline_acc 37.0  # 可选，手动指定 Baseline 准确率 \
    --max_ops 3 \
    --improvement_threshold 0.2 \
    --p_any_target 0.5 \
    --num_workers 8
```

> **v5.4 变更**:
> - Phase C 统一使用 200 epochs，与 Phase A/B 保持一致的训练预算，确保公平对比。
> - 新增 `--p_any_target` 参数，控制组合策略的总体增广强度（默认 0.5 = 50% 样本被增广）。
> - 接受阈值默认 +0.2% 且需要多数 seeds 提升。

### 5.4 输入文件

| 文件 | 说明 |
|------|------|
| `outputs/phase_b_tuning_summary.csv` | Phase B 最佳配置汇总 (自动读取) |
| `outputs/baseline_result.csv` | Baseline 结果 (与 Phase A/B 共用) |

### 5.5 输出文件

| 文件 | 说明 |
|------|------|
| `outputs/phase_c_history.csv` | 每次尝试添加操作的记录 |
| `outputs/phase_c_final_policy.json` | 最终策略定义 (供 Phase D 使用) |
| `outputs/checkpoints/phase_c_*.pth` | 各策略的 best checkpoint |
| `logs/phase_c.log` | 运行日志 |

### 5.6 算法说明

Phase C 使用贪心算法构建最终策略：

1. 初始化策略 P = S0，Acc(P) = Baseline_200ep_acc（与 A/B 一致）
2. 按 Phase B 的 mean_val_acc 排名，逐个尝试添加 Op(m*, p*)
3. 对每个候选 Op：
   - 检查互斥约束（如 RandomRotation 和 RandomPerspective 互斥）
   - 训练 P + Op × 3 seeds × 200 epochs
   - 如果 mean_acc > Acc(P) + 0.2%，且多数 seeds 提升，则接受该 Op
4. 最多添加 3 个额外操作
5. 输出最终策略 P_final

### 5.7 概率调整机制 (v5.4)

当组合多个操作时，每个操作独立应用会导致总体增广强度过高。Phase C 使用破坏性权重自动调整概率：

| 操作 | 破坏性 d | 权重 w=1-d |
|------|----------|------------|
| RandomErasing | 0.85 | 0.15 |
| RandomPerspective | 0.80 | 0.20 |
| RandomResizedCrop | 0.65 | 0.35 |
| RandomRotation | 0.40 | 0.60 |
| GaussianBlur | 0.30 | 0.70 |
| RandomGrayscale | 0.30 | 0.70 |
| GaussianNoise | 0.20 | 0.80 |
| ColorJitter | 0.20 | 0.80 |

**调整公式**: `p'_i = α × w_i × p_i`，其中 α 由二分搜索求解使得 `P(至少一个增广) = p_any_target`。

**策略 JSON 格式** (v5.4):
```json
{
  "version": "v5.4",
  "p_any_target": 0.5,
  "ops": [
    {
      "name": "GaussianNoise",
      "magnitude": 0.34,
      "probability_original": 0.34,
      "probability_adjusted": 0.27
    }
  ]
}
```

### 5.7 运行记录

| 日期 | 耗时 | 状态 |
|------|------|------|
| - | - | 待运行 |

---

## 6. Phase D SOTA 对比实验

> **GPU 说明**: Phase D 在 5 个 Folds 上运行 7 种方法，共 35 次训练，每次 200 epochs（与 A/B/C 一致）。

### 6.1 前置条件

- Phase C 完成，`outputs/phase_c_final_policy.json` 存在
- 如果策略文件不存在，Ours 方法将退化为 Baseline

### 6.2 运行命令

```bash
# 冒烟测试
bash scripts/smoke_test_phase_d.sh

# 完整运行（5 方法 × 5 folds × 200 epochs）
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
    --epochs 200 \
    --seed 42 \
    --output_dir outputs \
    --policy_json outputs/phase_c_final_policy.json \
    --methods Baseline,RandAugment,Cutout,Ours_p1,Ours_optimal \
    --folds 0,1,2,3,4 \
    --num_workers 6
```

> **v5.4 变更**:
> - Phase D 统一使用 200 epochs，与 Phase A/B/C 保持一致的训练预算，确保公平对比。
> - 新增 Baseline-NoAug 消融实验，验证基础增强的作用。
> - Ours_optimal 现在使用 probability_adjusted（调整后概率），控制总体增广强度。

### 6.5 对比方法说明

| 方法 | 说明 | 参数 |
|------|------|------|
| **Baseline** | S0 基础增强 | RandomCrop(32, padding=4) + HorizontalFlip(p=0.5) |
| **Baseline-NoAug** | 无增强消融 | 仅 ToTensor，验证基础增强的作用 |
| **RandAugment** | 自动增强 SOTA | N=2, M=9 (标准设置) |
| **Cutout** | 遮挡增强 SOTA | n_holes=1, length=16 |
| **Ours_p1** | 消融对照 | Phase C 策略，所有 p 强制为 1.0（使用 probability_original） |
| **Ours_optimal** | 最终方法 | Phase C 策略，使用 probability_adjusted（经破坏性加权调整） |

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

## 8. 计算量估计 (v5.4 更新 - 统一 200ep)

| 阶段 | 配置数 | 早停策略 | 预计时间 (单 GPU) | 预计时间 (4 GPU) |
|------|--------|----------|------------------|-----------------|
| Baseline | 1 × 200 ep | 允许早停 | ~15 min | ~15 min |
| Phase A | 8 ops × 32 点 × **40 ep** | 允许早停 | ~1h | ~20-30min |
| Phase B (ASHA) | ~8 ops × 30 samples, rungs=[**40,100,200**] | ASHA 淘汰 | ~2-4h | ~0.5-1h |
| Phase C | ~8 ops × 3 seeds × 200 ep | 阈值 0.2% + 多数规则 | ~1h | ~1h (单GPU) |
| Phase D | **7 methods** × 5 folds × 200 ep | 允许早停 | ~2h | ~0.5-1h |

> **v5.5 变更**: 
> - Phase A 降为 40ep 低保真筛选，节省 80% 计算量
> - Phase B rungs 调整为 [40,100,200]
> - Phase C 接受条件改为阈值 0.2% + 多数规则
> - Phase D 新增 Best_SingleOp 方法 (共 7 methods)
> - 这确保了公平对比：相同 baseline 基准（~37%），验证"多 Op 组合 > 单 Op > 无增强"
> - Phase C 因贪心算法串行执行，无法多 GPU 并行；Phase D 可按 fold 分配到 4 GPU

---

## 9. 输出文件汇总

```
outputs/
├── baseline_result.csv           # Baseline 200ep 结果 (所有阶段共用)
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
