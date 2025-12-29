# Checkpoint 获取与规范

已发布的权重通过 GitHub Release 提供，并默认下载到 `outputs/checkpoints/`。本文档说明包含哪些文件、如何下载校验与加载。

## 发布内容（Release: `v1.0`）

| 文件 | 作用 | SHA-256 |
|------|------|---------|
| `baseline_best.pth` | Baseline 最佳模型 | `959ca894db899c9b76b6bb81bb4830f892e1b2e589013522c57afbf21cc93c51` |
| `phase_d_fold0_best.pth` | Fold-0 最佳模型 | `edd6b4f4ab4868cfaed203d69351fcbcd228603efb0bc645fa125bf14defbfbc` |
| `phase_d_fold1_best.pth` | Fold-1 最佳模型 | `eebf6785e2bab208f5ce7a4dcf76a6ec4d3d1f589187edcee6a476d6d972a0a3` |
| `phase_d_fold2_best.pth` | Fold-2 最佳模型 | `25bcd7734d303bd5f84558ef1fcdbfbeda6aa023d05bdb2983d64be28b2c55f1` |
| `phase_d_fold3_best.pth` | Fold-3 最佳模型 | `142b9627016db7013399ed607da1d50f5327ccab21d392274e94e7408704b693` |
| `phase_d_fold4_best.pth` | Fold-4 最佳模型 | `6fe22981fbee87e866da77a542dbeebd97018a2ef90d15757e0a9aeef35e70e7` |

## 下载与校验

- 推荐：在仓库根目录使用 GitHub CLI 下载到默认路径（自动创建目录）：
  ```bash
  gh release download v1.0 -D outputs/checkpoints -p "*.pth"
  ```
- 校验完整性：
  ```bash
  cd outputs/checkpoints
  shasum -a 256 *.pth
  ```
  对照上表的 SHA-256。

## 文件内容结构

每个 `.pth` 是 `torch.save` 的 dict，主要键：

```python
{
    "model_state_dict": ...,
    "optimizer_state_dict": ...,
    "scheduler_state_dict": ...,
    "epoch": best_epoch,
    "val_acc": best_val_acc,
    "top5_acc": best_top5_acc,
    "val_loss": best_val_loss,
    "config": {  # 训练配置
        "seed": seed,
        "fold_idx": fold_idx,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": 0.1,
        "weight_decay": 1e-2,
        "momentum": 0.9,
    },
}
```

## 加载示例

```python
import torch
from src.models import create_model

ckpt_path = "outputs/checkpoints/baseline_best.pth"
checkpoint = torch.load(ckpt_path, map_location="cpu")

model = create_model(num_classes=100, pretrained=False)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print(f"Best Epoch: {checkpoint['epoch']}")
print(f"Val Acc: {checkpoint['val_acc']:.2f}%")
print(f"Top-5 Acc: {checkpoint['top5_acc']:.2f}%")
```
