# 实验环境记录 (Experiment Environment Log)

## 1. 服务器配置

> 快捷运行：`bash scripts/train_single_gpu.sh`

### 系统配置

| 项目 | 配置 |
|------|------|
| **GPU** | NVIDIA A10 |
| **CUDA** | 12.8 |
| **Python** | 3.14.2 (conda: rethinking_aug) |

---

## 2. Phase 0 超参校准

### 前置条件

首次运行前，需要确定最优 (weight_decay, label_smoothing) 组合。

### 运行命令

```bash
# 运行 Phase 0 校准 (~1 小时)
CUDA_VISIBLE_DEVICES=0 python run_phase0_calibration.py | tee logs/phase0.log
```

### 可选参数

```bash
python run_phase0_calibration.py \
    --epochs 100 \
    --seeds 42,123,456 \
    --fold_idx 0 \
    --output_dir outputs
```

### 输出文件

| 文件 | 说明 |
|------|------|
| `outputs/phase0_summary.csv` | 各参数组合的平均性能 |

---

## 3. Baseline 运行

### 运行命令

```bash
# 运行 Baseline (默认使用 GPU 0)
python run_baseline.py | tee logs/baseline.log

# 指定 GPU (可选)
CUDA_VISIBLE_DEVICES=0 python run_baseline.py | tee logs/baseline.log
```

### 输出文件

| 文件 | 说明 |
|------|------|
| `outputs/baseline_result.csv` | Baseline 结果 |
| `outputs/checkpoints/baseline_best.pth` | 最佳模型 checkpoint |

---

## 4. Phase A 筛选

### 运行命令

```bash
# 后台运行
CUDA_VISIBLE_DEVICES=0 nohup python main_phase_a.py > logs/phase_a.log 2>&1 &

# 前台运行
CUDA_VISIBLE_DEVICES=0 python main_phase_a.py | tee logs/phase_a.log
```

### 可选参数

```bash
python main_phase_a.py \
    --epochs 40 \
    --n_samples 32 \
    --n_promote 8 \
    --fold_idx 0 \
    --output_dir outputs \
    --seed 42 \
    --num_workers 8  # Mac 用户请设为 0
```

### 输出文件

| 文件 | 说明 |
|------|------|
| `outputs/phase_a_results.csv` | Phase A 筛选结果 (256 configs) |

---

## 5. Phase B ASHA 深度微调

### 运行命令

```bash
# 完整 ASHA 运行 (~2-4 小时)
CUDA_VISIBLE_DEVICES=0 nohup python main_phase_b.py > logs/phase_b.log 2>&1 &

# 快速测试
CUDA_VISIBLE_DEVICES=0 python main_phase_b.py --n_samples 5 --dry_run
```

### ASHA 参数说明

```bash
python main_phase_b.py \
    --rungs 40,100,200 \
    --n_samples 30 \
    --reduction_factor 2 \
    --seed 42 \
    --output_dir outputs \
    --phase_a_csv outputs/phase_a_results.csv \
    --baseline_csv outputs/baseline_result.csv \
    --num_workers 8
```

### 输出文件

| 文件 | 说明 |
|------|------|
| `outputs/phase_b_tuning_raw.csv` | ASHA 最终存活配置结果 |
| `outputs/phase_b_tuning_summary.csv` | 按 mean_val_acc 降序排列 |

---

## 6. Phase C 贪心组合

### 前置条件

- Phase B 完成，`outputs/phase_b_tuning_summary.csv` 存在
- Baseline 结果存在（`outputs/baseline_result.csv`）

### 运行命令

```bash
# 运行完整 Phase C
CUDA_VISIBLE_DEVICES=0 nohup python main_phase_c.py > logs/phase_c.log 2>&1 &

# 前台运行
CUDA_VISIBLE_DEVICES=0 python main_phase_c.py | tee logs/phase_c.log

# 快速测试
CUDA_VISIBLE_DEVICES=0 python main_phase_c.py --dry_run
```

### 可选参数

```bash
python main_phase_c.py \
    --epochs 200 \
    --seeds 42,123,456 \
    --fold_idx 0 \
    --output_dir outputs \
    --phase_b_csv outputs/phase_b_tuning_summary.csv \
    --max_ops 3 \
    --min_improvement 0.1 \
    --p_any_target 0.7 \
    --num_workers 8
```

### 输出文件

| 文件 | 说明 |
|------|------|
| `outputs/phase_c_history.csv` | 每次尝试添加操作的记录 |
| `outputs/phase_c_final_policy.json` | 最终策略定义 |

---

## 7. Phase D SOTA 对比实验

### 前置条件

- Phase C 完成，`outputs/phase_c_final_policy.json` 存在

### 运行命令

```bash
# 完整运行（7 methods × 5 folds × 200 epochs）
CUDA_VISIBLE_DEVICES=0 nohup python main_phase_d.py > logs/phase_d.log 2>&1 &

# 前台运行
CUDA_VISIBLE_DEVICES=0 python main_phase_d.py | tee logs/phase_d.log

# 快速测试
CUDA_VISIBLE_DEVICES=0 python main_phase_d.py --dry_run

# 只运行特定方法
CUDA_VISIBLE_DEVICES=0 python main_phase_d.py --methods Baseline,SAS

# 只运行特定 folds
CUDA_VISIBLE_DEVICES=0 python main_phase_d.py --folds 0,1,2
```

### 对比方法说明

| 方法 | 说明 | 参数 |
|------|------|------|
| **Baseline** | S0 基础增强 | RandomCrop(32, padding=4) + HorizontalFlip(p=0.5) |
| **Baseline-NoAug** | 无增强消融 | 仅 ToTensor |
| **RandAugment** | 自动增强 SOTA | N=2, M=9 |
| **Cutout** | 遮挡增强 SOTA | n_holes=1, length=16 |
| **SAS** | 最终方法 (单操作最优) | Phase B 最佳单操作 ColorJitter |
| **SAS_p1** | 消融对照 | 固定 p=1.0 |
| **Best_SingleOp** | SAS 别名 | 与 SAS 相同 |

### 输出文件

| 文件 | 说明 |
|------|------|
| `outputs/phase_d_results.csv` | 每个 (method, fold) 的原始结果 |
| `outputs/phase_d_summary.csv` | Mean ± Std 汇总 (用于论文表格) |

---

## 8. 补充实验 (Supplementary Experiments)

### 8.1 破坏性分析 (Destructiveness)

```bash
python scripts/calculate_destructiveness.py
```
**输出**: `outputs/destructiveness_metrics.csv`

### 8.2 稳定性验证 (Stability Check)

```bash
python scripts/run_stability_check.py
```
**输出**: `outputs/stability_seeds_results.csv`

### 8.3 公平性对比 (Tuned RandAugment)

```bash
# 1. 搜索最佳参数
python scripts/run_tuned_randaugment.py

# 2. 全量验证最佳参数
python scripts/run_final_tuned_ra.py
```
**输出**: 
- `outputs/tuned_randaugment_results.csv`
- 控制台日志 (Final Acc)

---

## 9. 一键运行脚本

```bash
# 完整训练流程
bash scripts/train_single_gpu.sh

# 或指定 GPU
CUDA_VISIBLE_DEVICES=1 bash scripts/train_single_gpu.sh

# 更稳定：
# 创建日志目录
mkdir -p logs

# 后台运行，日志保存到文件
nohup bash scripts/train_single_gpu.sh > logs/train_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# 查看进程
jobs -l

# 实时查看日志
tail -f logs/train_*.log
```

---

## 10. 计算量估计

| 阶段 | 配置 | 预计时间 |
|------|------|----------|
| Phase 0 | 12 configs × 3 seeds × 100 ep | ~1h |
| Baseline | 1 × 200 ep | ~15 min |
| Phase A | 8 ops × 32 点 × 40 ep | ~1h |
| Phase B (ASHA) | 30 samples/op, rungs=[40,100,200] | ~2-4h |
| Phase C | ~8 ops × 3 seeds × 200 ep | ~1-2h |
| Phase D | 7 methods × 5 folds × 200 ep | ~2h |
| **总计** | | **~8-12h** |

---

## 11. 输出文件汇总

```
outputs/
├── phase0_summary.csv            # Phase 0 超参校准结果
├── baseline_result.csv           # Baseline 结果
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
├── destructiveness_metrics.csv   # 破坏性分析
├── stability_seeds_results.csv   # 稳定性验证
└── tuned_randaugment_results.csv # 公平性对比
```
