# 先验引导增强策略搜索 (Prior-Guided Augmentation Policy Search)

## 项目概述

**研究目标**: 在低数据体制（每类仅 100 张图像）下，通过系统化搜索管道寻找多种数据增强操作的最优组合。

**核心假设**: 多种数据增强操作的合理叠加（多 Op 组合）比单一操作更有效。

**目标期刊/会议**: WACV / BMVC / ICIP 级别

---

## 1. 基础信息

### 1.1 数据集

| 项目 | 配置 |
|------|------|
| **数据集** | CIFAR-100 |
| **子采样** | 每类 20%（约 100 张/类） |
| **总样本数** | 10,000 张/Fold |
| **划分方式** | 5-Fold 分层划分 |
| **单 Fold 划分** | Train 9,000 / Val 1,000 |
| **搜索阶段** | 仅使用 Fold-0 |
| **最终评估** | 5-Fold 交叉验证，汇报 Mean ± Std |

### 1.2 模型架构

| 项目 | 配置 |
|------|------|
| **模型** | ResNet-18 (torchvision 标准实现) |
| **修改** | 仅替换最后 FC 层为 100 类 |
| **预训练** | 无 (from scratch) |
| **No-NAS 约束** | 严格固定架构，仅改变数据增强 |

### 1.3 训练超参数

| 参数 | 值 | 说明 |
|------|------|------|
| **Optimizer** | SGD | |
| **Learning Rate** | 0.1 | |
| **Weight Decay** | **1e-2** | 增强正则化 |
| **Momentum** | 0.9 | |
| **Label Smoothing** | 0.1 | |
| **Scheduler** | CosineAnnealingLR | T_max = epochs - warmup |
| **Warmup** | 5 epochs | 线性 warmup |
| **Batch Size** | 128 | |
| **All Phase Epochs** | **200** | 统一训练预算，公平对比 |
| **AMP** | Enabled | 混合精度加速 |
| **num_workers** | 8 | |
| **prefetch_factor** | 4 | |
| **channels_last** | True | NHWC 内存格式优化 |

### 1.4 早停策略

| 阶段 | epochs | min_epochs | patience | 说明 |
|------|--------|------------|----------|------|
| **Phase A** | **40** | 20 | 15 | 低保真快速筛选 |
| **Phase B** | ASHA | N/A | N/A | 多轮淘汰 (40→100→200 ep) |
| **Phase C** | 200 | 60 | 60 | 策略构建 |
| **Phase D** | 200 | 60 | 60 | 最终评估 |

- **监控指标**: `val_acc` (mode="max")
- **min_delta**: 0.2（过滤噪声波动）

---

## 2. 增强操作空间

### 2.1 基础增强 (S0 Baseline)

始终开启，作为所有实验的起点：
- **RandomCrop**: padding=4, size=32
- **RandomHorizontalFlip**: p=0.5

### 2.2 搜索候选池 (8 个操作)

```python
OP_SEARCH_SPACE = {
    # 温和操作 - 可用较高概率和强度
    "ColorJitter":       {"m": [0.1, 0.9], "p": [0.3, 0.9]},
    "RandomGrayscale":   {"m": [0.5, 0.5], "p": [0.1, 0.7]},   # m 固定，只搜 p
    "GaussianNoise":     {"m": [0.02, 0.4], "p": [0.3, 0.9]},
    
    # 中等操作
    "RandomResizedCrop": {"m": [0.4, 0.95], "p": [0.3, 0.8]},
    "RandomRotation":    {"m": [0.0, 0.5], "p": [0.2, 0.7]},    # ~0-22.5 度
    "GaussianBlur":      {"m": [0.0, 0.5], "p": [0.2, 0.7]},
    
    # 破坏性操作 - 需低概率
    "RandomErasing":     {"m": [0.02, 0.35], "p": [0.1, 0.6]},
    "RandomPerspective": {"m": [0.0, 0.4], "p": [0.1, 0.6]},
}
```

**参数说明**:
- `m`: magnitude 搜索范围 [min, max]，控制增强强度
- `p`: probability 搜索范围 [min, max]，控制应用概率

### 2.3 互斥约束

```python
MUTUAL_EXCLUSION = {
    "RandomResizedCrop": ["S0_RandomCrop"],  # 禁用 S0 的 RandomCrop
    "RandomRotation": ["RandomPerspective"],  # 二选一
    "RandomPerspective": ["RandomRotation"],
}
```

### 2.4 破坏性权重

用于组合多操作时自动调整概率，防止总体增广强度过高：

```python
OP_DESTRUCTIVENESS = {
    "RandomErasing":      0.85,   # 直接遮挡像素 - 最高破坏
    "RandomPerspective":  0.80,   # 严重几何变形
    "RandomResizedCrop":  0.65,   # 可能丢失关键区域
    "RandomRotation":     0.40,   # 中等几何变化
    "GaussianBlur":       0.30,   # 轻微模糊
    "RandomGrayscale":    0.30,   # 去色但保留结构
    "GaussianNoise":      0.20,   # 轻微噪声
    "ColorJitter":        0.20,   # 最温和 - 仅色彩变化
}
```

**权重含义**: 0 = 无破坏，1 = 完全破坏。高破坏性操作在组合时会被更激进地降低概率。

---

## 3. 实验流程 (4 阶段)

### 3.1 Phase A: 广度筛选 (Low-Fidelity Screening)

**目标**: 在 (m, p) 联合空间快速剔除无效增强

| 项目 | 配置 |
|------|------|
| **输入** | S0 + Op_i(m, p)（单操作） |
| **采样** | 2D Sobol Sequence, 32 组/Op |
| **训练** | 40 epochs, seed=42, Fold-0 |
| **评分** | `mean(top3(val_acc[30:40]))` - 更稳定 |
| **实验量** | 8 ops × 32 点 = 256 组 |
| **预计时间** | ~20-30min (4 GPU 并行) |

**低保真筛选原理**: 
40ep 训练与 200ep 最终性能高度相关（通常 ρ > 0.8），但仅需 1/5 计算量。

**晋级规则**: 每个 Op 选 8 个配置晋级 Phase B
- Top-6: 按 stable_score 排名前 6
- +2 Diversity: 从边界区域（低 m 和高 m）各选 1 个

**输出**: `outputs/phase_a_results.csv`（含 stable_score 列）

### 3.2 Phase B: 深度微调 (ASHA Tuning)

**目标**: 高效搜索最佳 (m, p) 参数

| 项目 | 配置 |
|------|------|
| **输入** | Phase A 晋级的 Ops (8 configs/Op) |
| **采样** | Sobol Sequence, 30 组/Op |
| **算法** | ASHA 早停淘汰赛 |
| **Rungs** | [40, 100, 200] epochs |
| **Reduction Factor** | 1/2（每轮保留最好的一半） |
| **Final Rung Multi-Seed** | 3 seeds 平均 (42, 123, 456) |
| **实验量** | ~240 初始配置 → ~60 完成 200ep |
| **预计时间** | ~2-4h (4 GPU 并行) |

**Rungs 对齐**: 40ep 与 Phase A 一致，便于 warm-start。

**ASHA 优势**:
- 采样更多点（不怕错过最优）
- 早停节省算力（差配置不浪费 200ep）
- 理论最优性保证

**输出**: `outputs/phase_b_tuning_summary.csv`（按 mean_val_acc 排序的 (Op, m*, p*) 列表）

### 3.3 Phase C: 贪心组合 (Prior-Guided Ensemble)

**目标**: 构建最终多操作组合策略

| 项目 | 配置 |
|------|------|
| **算法** | Multi-Start Greedy Search |
| **起点** | Top-K (默认 3) from Phase A + B |
| **验证** | 3 Random Seeds × 200 epochs |
| **判定阈值** | mean_acc > Acc(P) + **0.1%** |
| **最大操作数** | 3 个额外操作 |
| **概率调整** | p_any_target = 0.5（50% 样本被增广） |
| **预计时间** | ~1h (单 GPU) |

**贪心算法流程**:
1. 初始化 P = S0, Acc(P) = Baseline_200ep_acc
2. 按 Phase B 排名，逐个尝试添加 Op(m*, p*)
3. 检查互斥约束，训练 3 seeds × 200ep
4. **接受条件**:
   - mean_acc > Acc(P) + 0.1%
5. 输出最终策略 P_final

**概率调整机制**:

当组合多个操作时，每个操作独立应用会导致总体增广强度过高。使用以下公式调整：

```
p'_i = clip(α × w_i × p_i, 0, 1)

其中 (v5.5):
- w_i = 1 - d_i × g(m_i)
- d_i: 操作的破坏性权重
- g(m_i): magnitude 影响因子 (线性映射, g(m) = m)
- α 由二分搜索求解，使得 P(至少一个增广) = p_any_target
```

**同一操作，高 magnitude 会被更激进地降低概率。**

**示例**:
```
原始策略: RandomErasing(p=0.34) + GaussianNoise(p=0.76)
P(至少一个) = 1 - (1-0.34)(1-0.76) = 84%  ← 过高

调整后 (p_any_target=0.5):
RandomErasing: p' = 0.04 (d=0.85, 大幅降低)
GaussianNoise: p' = 0.47 (d=0.20, 适度降低)
P(至少一个) = 1 - (1-0.04)(1-0.47) = 49% ≈ 50% ✓
```

**输出 JSON 格式**:
```json
{
  "version": "current",
  "p_any_target": 0.5,
  "ops": [
    {
      "name": "GaussianNoise",
      "magnitude": 0.34,
      "probability_original": 0.76,
      "probability_adjusted": 0.47
    },
    {
      "name": "RandomErasing",
      "magnitude": 0.24,
      "probability_original": 0.34,
      "probability_adjusted": 0.04
    }
  ]
}
```

**输出文件**:
- `outputs/phase_c_history.csv` - 每次尝试的记录
- `outputs/phase_c_final_policy.json` - 最终策略定义

### 3.4 Phase D: SOTA 对比 (Benchmark Comparison)

**目标**: 证明方法优于或接近现有 SOTA

| 项目 | 配置 |
|------|------|
| **验证** | 5-Fold 交叉验证 |
| **训练** | 200 epochs/fold, seed=42 |
| **实验量** | 6 methods × 5 folds = 30 组 |
| **预计时间** | ~0.5-1h (4 GPU 并行) |

**对比方法 (6 个)**:

| 方法 | 说明 | 参数 |
|------|------|------|
| **Baseline** | S0 基础增强 | RandomCrop + HFlip |
| **Baseline-NoAug** | 无增强消融 | 仅 ToTensor |
| **RandAugment** | 自动增强 SOTA | N=2, M=9 |
| **Cutout** | 遮挡增强 SOTA | n_holes=1, length=16 |
| **Best_SingleOp** | 单操作最优 | Phase B 最佳单操作 |
| **Ours_p1** | 消融对照 | 策略使用 probability_original |
| **Ours_optimal** | 最终方法 | 策略使用 probability_adjusted |

**输出文件**:
- `outputs/phase_d_results.csv` - 每个 (method, fold) 的结果
- `outputs/phase_d_summary.csv` - Mean ± Std 汇总（用于论文表格）
- `outputs/checkpoints/phase_d_fold{0-4}_best.pth` - Ours_optimal 的 5-fold 模型

---

## 4. 预期结果

### 4.1 性能排名

```
Ours_optimal ≥ RandAugment > Baseline > Baseline-NoAug
```

**解读**:
- `Ours_optimal ≥ RandAugment`: 我们的方法应该接近或超过通用 SOTA
- `RandAugment > Baseline`: SOTA 方法优于基础增强
- `Baseline > Baseline-NoAug`: 基础增强有明显价值

### 4.2 消融验证

| 对比 | 验证目标 |
|------|----------|
| `Ours_optimal > Ours_p1` | 概率优化 + magnitude 调整的价值 |
| `Ours_optimal > Best_SingleOp` | 多操作组合的价值 |
| `Ours_optimal > Baseline` | 搜索策略的整体价值 |
| `Baseline > Baseline-NoAug` | 基础增强的价值 |

### 4.3 论文核心图表

1. **Performance Table**: Ours vs. RandAugment vs. Baseline (Mean ± Std)
2. **2D Heatmaps**: 展示 (m, p) 联合效应
3. **Ablation Study**: p=1.0 vs p=optimal 对比表格
4. **Efficiency Plot**: 搜索耗时 vs. 准确率
5. **Case Study**: 展示被 P_final 修正的样本

---

## 5. 硬件配置

| 项目 | 配置 |
|------|------|
| **云服务商** | 阿里云 |
| **实例规格** | ecs.gn7i-4x.8xlarge |
| **GPU** | 4 × NVIDIA A10 (4 × 24 GB) |
| **vCPU** | 32 vCPU |
| **内存** | 128 GiB |
| **CUDA** | 12.8 |
| **Python** | 3.14.2 |
| **Conda 环境** | pga |

---

## 6. 总时间估算 (4 GPU 并行)

| 阶段 | 时间 |
|------|------|
| Baseline | ~15 min |
| Phase A | ~1-1.5h |
| Phase B | ~2-4h |
| Phase C | ~1h |
| Phase D | ~0.5-1h |
| **总计** | **~5-8h** |

---

## 7. 核心创新点

1. **2D 搜索空间**: 联合搜索 (magnitude, probability)，而非固定 p=1.0
2. **先验引导范围**: 每操作定制 (m, p) 搜索范围，体现人类先验知识
3. **ASHA 早停**: Phase B 使用多保真淘汰赛，比 Grid Search 快 ~10 倍
4. **Multi-Start Greedy**: Phase C 从多起点出发，避免局部最优
5. **概率调整机制**: 组合多操作时，按破坏性权重自动调节概率，控制总体增广强度
6. **No-NAS 约束**: 严格固定模型架构，仅改变数据输入分布

---

## 8. 文件结构

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
    ├── phase_c_*.pth               # Phase C 策略 checkpoints
    └── phase_d_fold{0-4}_best.pth  # 最终 5-fold 模型
```

---

## 9. 关键代码位置

| 模块 | 文件 | 说明 |
|------|------|------|
| **增强操作** | `src/augmentations.py` | OP_SEARCH_SPACE, OP_DESTRUCTIVENESS, 概率调整函数 |
| **数据集** | `src/dataset.py` | CIFAR100Subsampled, 5-Fold 划分 |
| **模型** | `src/models.py` | ResNet-18 创建 |
| **训练工具** | `src/utils.py` | 优化器、调度器、训练循环 |
| **Phase A** | `main_phase_a.py` | 广度筛选 |
| **Phase B** | `main_phase_b.py` | ASHA 深度微调 |
| **Phase C** | `main_phase_c.py` | 贪心组合 + 概率调整 |
| **Phase D** | `main_phase_d.py` | SOTA 对比 |
| **一键脚本** | `scripts/train_single_gpu.sh` | 完整训练流程 |

---

## 10. 设计审阅要点

请帮我审阅以下方面：

1. **超参数选择**: lr=0.1, wd=1e-2, batch_size=128 是否合适？
2. **搜索空间设计**: 各操作的 (m, p) 范围是否合理？
3. **阶段逻辑**: A→B→C→D 的流程是否有漏洞？
4. **概率调整公式**: p'_i = α × w_i × p_i 的数学是否正确？
5. **实验设计**: 能否验证"多 Op 组合 > 单 Op > 无增强"的核心假设？
6. **公平性**: 所有方法统一 200 epochs 是否足够公平？

