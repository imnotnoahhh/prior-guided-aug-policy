# CSV 输出格式规范

本文档定义了所有实验阶段（Baseline、Phase A/B/C/D）的统一 CSV 输出格式。

## 统一字段定义

```
phase, op_name, magnitude, probability, seed, fold_idx, val_acc, val_loss, top5_acc, train_acc, train_loss, epochs_run, best_epoch, early_stopped, runtime_sec, timestamp, error
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

## 各阶段字段使用

| 字段 | Baseline | Phase A | Phase B | Phase C | Phase D |
|------|----------|---------|---------|---------|---------|
| phase | "Baseline" | "PhaseA" | "PhaseB" | "PhaseC" | "PhaseD" |
| op_name | "Baseline" | 单个 op | 单个 op | "op1+op2" | "FinalPolicy" |
| magnitude | "0.0" | Sobol采样值 | Grid采样值 | "m1+m2" | 最终组合值 |
| probability | "1.0" | Sobol采样值 | Grid采样值 | "p1+p2" | 最终组合值 |
| seed | 42 | 42 | 42/123/456 | 42 | 42 |
| fold_idx | 0 | 0 | 0 | 0 | 0,1,2,3,4 |

## 示例数据

参见 `docs/csv_format_example.csv`

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
