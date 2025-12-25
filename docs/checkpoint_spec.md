# Checkpoint 保存策略

本文档定义了各实验阶段的模型保存策略。

## 保存策略总览 (v5.1 更新)

| 阶段 | 保存模型 | 理由 |
|------|----------|------|
| **Baseline** | ✅ 是 | 对比基准，仅 1 个模型 |
| **Phase A** | ❌ 否 | 筛选阶段，256 个配置太多 |
| **Phase B** | ❌ 否 | ASHA 淘汰赛，仅最终存活配置保留结果 |
| **Phase C** | ✅ **是** | v5.1 新增：禁用早停后保存所有尝试的策略 |
| **Phase D** | ✅ 是 | 最终验证，5-fold 模型用于论文 (仅 Ours_optimal) |

## 保存路径

```
outputs/
└── checkpoints/
    ├── baseline_best.pth                          # Baseline 最佳模型
    ├── phase_c_ColorJitter_seed42_best.pth        # Phase C 单操作策略
    ├── phase_c_ColorJitter+GaussianBlur_seed42_best.pth  # Phase C 组合策略
    ├── ...                                        # Phase C 其他策略
    ├── phase_d_fold0_best.pth                     # Phase D Fold-0 最佳模型
    ├── phase_d_fold1_best.pth                     # Phase D Fold-1 最佳模型
    ├── phase_d_fold2_best.pth                     # Phase D Fold-2 最佳模型
    ├── phase_d_fold3_best.pth                     # Phase D Fold-3 最佳模型
    └── phase_d_fold4_best.pth                     # Phase D Fold-4 最佳模型
```

## Checkpoint 内容

每个 `.pth` 文件包含：

```python
{
    "model_state_dict": model.state_dict(),      # 模型权重
    "optimizer_state_dict": optimizer.state_dict(),  # 优化器状态
    "scheduler_state_dict": scheduler.state_dict(),  # 调度器状态
    "epoch": best_epoch,                          # 最佳 epoch
    "val_acc": best_val_acc,                      # 验证准确率
    "top5_acc": best_top5_acc,                    # Top-5 准确率
    "val_loss": best_val_loss,                    # 验证损失
    "config": {                                   # 训练配置
        "seed": seed,
        "fold_idx": fold_idx,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": 0.05,
        "weight_decay": 1e-3,
        "momentum": 0.9,
    }
}
```

## 磁盘占用估计

| 模型 | 数量 | 单个大小 | 总大小 |
|------|------|----------|--------|
| Baseline | 1 | ~45MB | ~45MB |
| Phase C | ~24 (8 ops × 3 seeds) | ~45MB | ~1GB |
| Phase D | 5 | ~45MB | ~225MB |
| **总计** | ~30 | - | **~1.3GB** |

## 用途

| 模型 | 论文用途 |
|------|----------|
| Baseline | 基准对比、消融实验 |
| Phase C | 策略构建过程的完整证据链、可复现性 |
| Phase D × 5 | 报告 5-fold 平均性能、计算 std、开源复现、可视化 |

## 加载示例

```python
import torch
from src.models import create_model

# 加载 checkpoint
checkpoint = torch.load("outputs/checkpoints/baseline_best.pth")

# 重建模型
model = create_model(num_classes=100, pretrained=False)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# 查看性能
print(f"Best Epoch: {checkpoint['epoch']}")
print(f"Val Acc: {checkpoint['val_acc']:.2f}%")
print(f"Top-5 Acc: {checkpoint['top5_acc']:.2f}%")
```
