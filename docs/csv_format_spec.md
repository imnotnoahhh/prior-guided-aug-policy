# CSV 输出格式规范

本文档定义了所有实验阶段（Baseline、Phase A/B/C/D）的统一 CSV 输出格式。

## 统一字段定义

```
phase, op_name, magnitude, probability, seed, fold_idx, val_acc, val_loss, top5_acc, train_acc, train_loss, epochs_run, best_epoch, early_stopped, runtime_sec, timestamp, error, stable_score
```

## 字段说明

| 字段 | 类型 | 说明 | 示例 |
|------|------|------|------|
| `phase` | str | 阶段标识 | Baseline, PhaseA, PhaseB, PhaseC, PhaseD |
| `op_name` | str | 操作名称（组合用+连接） | ColorJitter, ColorJitter+GaussianBlur |
| `magnitude` | str | 强度值（组合用+连接） | 0.35, 0.35+0.25 |
| `probability` | str | 应用概率（组合用+连接） | 1.0, 0.8+0.6 |
| `seed` | int | 随机种子 | 42, 123, 456 |
| `fold_idx` | int | Fold 索引 (0-4) | 0 |
| `val_acc` | float | 验证集 Top-1 准确率 (%) | 45.2 |
| `val_loss` | float | 验证集损失 | 2.3456 |
| `top5_acc` | float | 验证集 Top-5 准确率 (%) | 72.8 |
| `train_acc` | float | 训练集 Top-1 准确率 (%) | 52.1 |
| `train_loss` | float | 训练集损失 | 1.8765 |
| `epochs_run` | int | 实际运行的 epoch 数 | 200 |
| `best_epoch` | int | 达到最佳 val_acc 的 epoch | 185 |
| `early_stopped` | bool | 是否触发早停 | True, False |
| `runtime_sec` | float | 运行时间（秒） | 3600.5 |
| `timestamp` | str | ISO 8601 时间戳 | 2024-12-22T10:30:00 |
| `error` | str | 错误信息（无错误为空） | OOM, ValueError |
| `stable_score` | float | Phase A 稳定性评分（其余阶段填 -1） | 45.3 |

## 各阶段字段使用

| 字段 | Baseline | Phase A | Phase B | Phase C | Phase D |
|------|----------|---------|---------|---------|---------|
| phase | "Baseline" | "PhaseA" | "PhaseB" | "PhaseC" | "PhaseD" |
| op_name | "Baseline" | 单个 op | 单个 op | 单个或组合 "op1+op2" | 方法名* |
| magnitude | "0.0" | Sobol采样值 | Sobol采样值 | 单个或组合 "m1+m2" | 见下表 |
| probability | "1.0" | Sobol采样值 | Sobol采样值 | 单个或组合 "p1+p2" | 见下表 |
| seed | 42 | 42 | 42/123/456 | 42/123/456 | 42 |
| fold_idx | 0 | 0 | 0 | 0 | 0,1,2,3,4 |
| stable_score | - | `mean(top3(val_acc[-10:]))` | - | - | - |

### Phase D 方法名与参数

| 方法名 | op_name | magnitude | probability | 说明 |
|--------|---------|-----------|-------------|------|
| Baseline | "Baseline" | "N/A" | "N/A" | S0 基础增强 |
| RandAugment | "RandAugment" | "N/A" | "N/A" | N=2, M=9 (标准设置) |
| Cutout | "Cutout" | "N/A" | "N/A" | length=16 |
| Ours_p1 | "Ours_p1" | "m1+m2+..." | "1.0+1.0+..." | 消融: 所有 p=1.0 |
| Ours_optimal | "Ours_optimal" | "m1+m2+..." | "p1+p2+..." | 最终策略 |

## 示例数据

参见 `docs/csv_format_example.csv`

## 早停/轮次设置

| 阶段 | epochs | 策略 | 预期 epochs_run | 备注 |
|------|--------|------|-----------------|------|
| Phase A | 40 | min_epochs=20, patience=15, monitor=val_acc | 20-40 | 低保真筛选 |
| Phase B | 40 → 100 → 200 | ASHA 多轮淘汰 | 依赖淘汰结果 | 最终 rung 多 seed |
| Phase C | 200 | min_epochs=60, patience=60, monitor=val_acc | 120-200 | 贪心组合 |
| Phase D | 200 | min_epochs=60, patience=60, monitor=val_acc | 120-200 | 5 folds |
| Supplement | varies | varies | varies | CIFAR-10 & Ablation |

## 补充实验输出格式 (Supplementary)

### CIFAR-10 Generalization (`outputs/cifar10_50shot_results.csv`) ❌ 论文中未使用

| 字段 | 类型 | 说明 |
|------|------|------|
| Method | str | Baseline, RandAugment, Ours |
| Mean | float | 平均 val_acc (%) |
| Std | float | 标准差 |
| Seeds | str | 使用的随机种子列表 (JSON 格式) |
| Accs | str | 各 seed 的准确率列表 (JSON 格式) |

### Stability Verification (`outputs/stability_seeds_results.csv`)

| 字段 | 类型 | 说明 |
|------|------|------|
| Method | str | RandAugment, Ours |
| Mean | float | 平均 val_acc (%) |
| Std | float | 标准差 |
| Seeds | str | 使用的随机种子列表 (JSON 格式) |
| Accs | str | 各 seed 的准确率列表 (JSON 格式) |

### Ablation Study (`outputs/ablation/ablation_p0.5_raw.csv`)
复用统一字段定义 (`PhaseB` 格式)，但在 `phase` 字段填入 "PhaseB" (复用代码逻辑) 或 "Ablation"。

### Search Workflow Ablation (`outputs/search_ablation_results.csv`)

用于验证 ASHA 调优 (Phase B) 的必要性。

| 字段 | 说明 |
|------|------|
| `method` | PhaseA_Only (仅 Sobol 筛选，无 ASHA 调优) |
| `op_name` | 操作名 (Phase A 最佳: RandomPerspective) |
| `magnitude` | 强度值 |
| `probability` | 应用概率 |
| `fold_idx` | Fold 索引 (0-4) |
| `val_acc` | 验证集准确率 (%) |
| `best_epoch` | 最佳 epoch |
| `timestamp` | 时间戳 |

**重要说明**：
- 此文件**仅存储 Phase A only** 的实验结果（使用 Sobol 筛选的最佳配置，无 ASHA 调优）
- **Full SAS** 对照组数据复用 `phase_d_results.csv` 中的 `Best_SingleOp`（即完整 SAS 流程的输出）
- 论文 Table 5 中：Phase A only = 35.80% ± 1.65%，Full SAS = 40.74% ± 0.78%

### Ablation Summary (`outputs/ablation/ablation_p0.5_summary.csv`)
用于论文绘图的聚合数据：

| 字段 | 说明 |
|------|------|
| `op_name` | 操作名 (如 ColorJitter) |
| `magnitude` | 搜索的幅度值 |
| `probability` | 固定值 (0.5) |
| `mean_val_acc` | 最终 epoch 的准确率 |
| `std_val_acc` | 标准差 (若单 seed 则为 0) |
| `mean_top5_acc` | Top-5 准确率均值 |
| `std_top5_acc` | Top-5 准确率标准差 |
| `mean_train_acc` | 训练集准确率均值 |
| `std_train_acc` | 训练集准确率标准差 |
| `mean_runtime_sec` | 平均运行时间 |
| `n_seeds` | 使用的 seed 数量 |

### Destructiveness Metrics (`outputs/destructiveness_metrics.csv`)
语义保真度分析结果：

| 字段 | 说明 |
|------|------|
| `Strategy` | 增强策略名称 |
| `SSIM_Mean` | SSIM 均值 (越高越好) |
| `SSIM_Std` | SSIM 标准差 |
| `LPIPS_Mean` | LPIPS 均值 (越低越好) |
| `LPIPS_Std` | LPIPS 标准差 |

### Table 1 Extended (`outputs/table1_extended.csv`)
论文 Table 1 扩展统计数据：

| 字段 | 说明 |
|------|------|
| `Method` | 方法名 (Baseline, RandAugment, SAS) |
| `Mean` | 5-fold 均值准确率 |
| `Std` | 5-fold 标准差 |
| `Min Acc` | 最小准确率 |
| `Lower Bound` | Mean - Std |
| `95% CI Low` | 95% 置信区间下界 |
| `95% CI High` | 95% 置信区间上界 |
| `Fold Values` | 各 fold 准确率列表 |

### Shot Sweep Results (`outputs/shot_sweep_results.csv`) ❌ 论文中未使用
不同样本量下的实验原始结果：

| 字段 | 说明 |
|------|------|
| `shot` | 每类样本数 (20, 50, 100, 200) |
| `method` | 方法名 (Baseline, RandAugment, SAS) |
| `fold_idx` | Fold 索引 (0-4) |
| `seed` | 训练随机种子 (权重初始化、dropout 等) |
| `data_seed` | 数据采样种子 (选择哪些图片)，保持固定以隔离训练方差 |
| `val_acc` | 验证集准确率 (%) |
| `val_loss` | 验证集损失 |
| `top5_acc` | Top-5 准确率 (%) |
| `train_acc` | 训练集准确率 (%) |
| `train_loss` | 训练集损失 |
| `epochs_run` | 实际运行的 epoch 数 |
| `best_epoch` | 最佳 epoch |
| `runtime_sec` | 总运行时间 (秒) |
| `epoch_time_avg` | 平均每 epoch 时间 (秒) |
| `timestamp` | ISO 8601 时间戳 |
| `error` | 错误信息 |

### Shot Sweep Summary (`outputs/shot_sweep_summary.csv`) ❌ 论文中未使用
不同样本量下的汇总统计：

| 字段 | 说明 |
|------|------|
| `shot` | 每类样本数 |
| `method` | 方法名 |
| `mean_acc` | 平均准确率 (%) |
| `std_acc` | 标准差 |
| `min_acc` | 最小准确率 (%) |
| `lower_bound` | Mean - Std |
| `epoch_time_avg` | 平均每 epoch 时间 (秒) |
| `n_folds` | 完成的 fold 数量 |

## 论文写作用途

| 论文内容 | 使用字段 |
|----------|----------|
| 准确率表格 | val_acc, top5_acc |
| 收敛曲线 | train_acc, train_loss, val_acc, val_loss |
| 训练效率 | runtime_sec, epochs_run |
| 过拟合分析 | train_acc vs val_acc |
| 鲁棒性分析 | 同 op+magnitude 不同 seed 的 std |
| 早停分析 | best_epoch, early_stopped |
| 可复现性 | timestamp, seed, fold_idx |
