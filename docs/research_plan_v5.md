# Research Plan: Prior-Guided Augmentation Policy Search in Low-Data Regimes
# 实验计划：低数据体制下的先验引导增强策略搜索

**Version**: v5 (2D Search Space: Magnitude + Probability)
**Target**: Academic Conference (WACV / BMVC / ICIP Level)
**Dataset**: CIFAR-100 (Subsampled: 20% per class, Stratified)

---

## Changelog (v4 → v5)

| 变更类型 | 内容 | 代码影响 |
|:---|:---|:---|
| **[NEW]** | 引入 probability 作为第二搜索维度 | `augmentations.py`: 添加概率包装器 |
| **[CHANGED]** | Phase A: 1D Sobol → 2D Sobol (m, p) | `main_phase_a.py`: 2D 采样逻辑 |
| **[CHANGED]** | Phase B: Grid → ASHA 早停淘汰 (v5.3) | `main_phase_b.py`: ASHA + Sobol |
| **[NEW]** | 每操作定制 (m, p) 范围 | `OP_SEARCH_SPACE` 配置 |
| **[NEW]** | 消融实验: p=1.0 vs p=optimal | 使用 `archive_p100/` 数据对比 |

### 设计动机

v4 实验发现：
1. 所有操作使用 `p=1.0`（100% 应用增强）
2. 最佳单操作提升仅 +2.1%
3. `RandomGrayscale` 效果最好，因为它内置了概率控制
4. 结论：**概率化应用增强可能比 100% 应用更有效**

---

## 1. 核心研究假设与目标 (Hypothesis & Objective)

### 1.1 研究目标 (Research Goal)
在极少样本（每类 100 张）限制下，通过系统化的搜索管道（Pipeline），寻找**多种基础数据增强操作的最优组合**，验证叠加策略能够提高模型的准确率和鲁棒性。

* **核心假设**: 多种数据增强操作的合理叠加（多 Op 组合）比单一操作更有效。
* **核心痛点**: 现有的 AutoAugment/RandAugment 在低数据下往往因增强过强导致欠拟合，或因搜索代价过高而不可行。
* **解决方案**: 基于"人类先验（Human Priors）"初始化的三阶段搜索管道（Pipeline）：
    1. Phase A: 广度筛选有效操作
    2. Phase B: 深度调参找最佳 (m, p)
    3. Phase C: 贪心叠加构建组合策略
* **v5 新增**: 联合搜索 (magnitude, probability)，而非固定 p=1.0。
* **去NAS化 (No-NAS)**: 严格固定模型架构和超参，仅改变数据输入分布。

### 1.2 严格的数据验证协议 (Strict Validation Protocol)
*必须严格执行，防止数据泄露和方差欺骗。*

* **全集定义**: CIFAR-100 (50,000 images).
* **5-Fold 鲁棒性验证 (The "Golden Standard")**:
    * 将全集按类别分层划分为 5 个互斥的数据折 (Folds)，每折 10,000 张。
    * **目的**: 最终汇报的 Accuracy 必须是 **Mean ± Std (over 5 Folds)**。
* **搜索用代理集 (Proxy for Search)**:
    * 仅使用 `Fold-0` 进行搜索。
    * 在 `Fold-0` 内部划分：**Train (9,000) / Val (1,000)**。

---

## 2. 增强操作空间 (Augmentation Space)

### 2.1 基础增强 ($S_0$ - Baseline)
*始终开启，作为所有实验的起点。*
* **RandomCrop**: padding=4, size=32
* **RandomHorizontalFlip**: p=0.5

### 2.2 搜索候选池 (Candidate Pool) — **v5 更新**

*每个操作有独立的 (m, p) 搜索范围，体现先验知识。*

| 操作 (Op) | m 范围 | p 范围 | 类型 | 关键约束 |
| :--- | :--- | :--- | :--- | :--- |
| **RandomResizedCrop** | [0.3, 0.9] | [0.3, 0.8] | 中等 | **互斥**: 禁用 $S_0$ 的 RandomCrop |
| **RandomRotation** | [0.0, 0.4] | [0.2, 0.6] | 中等 | **互斥**: 与 Perspective 二选一 |
| **RandomPerspective** | [0.0, 0.3] | [0.1, 0.5] | 破坏性 | **互斥**: 与 Rotation 二选一 |
| **ColorJitter** | [0.1, 0.8] | [0.2, 0.8] | 温和 | b/c/s 同步变化 |
| **RandomGrayscale** | [0.5, 0.5] | [0.1, 0.6] | 温和 | m 固定，只搜 p |
| **GaussianBlur** | [0.0, 0.3] | [0.2, 0.6] | 中等 | sigma ∈ [0.1, 2.0] |
| **RandomErasing** | [0.05, 0.3] | [0.1, 0.4] | 破坏性 | 需低 p，放在 Tensor 后 |
| **GaussianNoise** | [0.05, 0.5] | [0.2, 0.8] | 温和 | 需 `torch.clamp` |

**代码配置** (`OP_SEARCH_SPACE`):
```python
OP_SEARCH_SPACE = {
    "RandomResizedCrop": {"m": [0.3, 0.9], "p": [0.3, 0.8]},
    "RandomRotation":    {"m": [0.0, 0.4], "p": [0.2, 0.6]},
    "RandomPerspective": {"m": [0.0, 0.3], "p": [0.1, 0.5]},
    "ColorJitter":       {"m": [0.1, 0.8], "p": [0.2, 0.8]},
    "RandomGrayscale":   {"m": [0.5, 0.5], "p": [0.1, 0.6]},  # m固定，只搜p
    "GaussianBlur":      {"m": [0.0, 0.3], "p": [0.2, 0.6]},
    "RandomErasing":     {"m": [0.05, 0.3], "p": [0.1, 0.4]},
    "GaussianNoise":     {"m": [0.05, 0.5], "p": [0.2, 0.8]},
}
```

---

## 3. 实验执行流程 (Execution Pipeline)

### 阶段 A：广度筛选 (Screening) — **v5 更新**
*目标：在 (m, p) 联合空间快速剔除无效增强。*

* **配置**:
    * **Input**: $S_0 + Op_i(m, p)$ (Single Op with magnitude and probability)
    * **Sampling**: **2D Sobol Sequence** (32 组 $(m, p)$/Op, seed=42)
    * **Prior-Guided Ranges**: 每操作使用定制的 $(m, p)$ 范围（见 2.2 节）
    * **Training**: **200 Epochs** (开启 ASHA 早停，Grace Period=40)
* **实验量**: 8 ops × 32 点 × 1 seed = **256 组**
* **预计时间**: ~1.2h (4 GPU 并行)
* **保留标准 (Retention Criteria)**:
    满足以下任一条件即可晋级：
    1.  **Top-1 Acc**: $\Delta \ge -0.5\%$ (允许轻微掉点)。
    2.  **Top-5 Acc**: $\Delta > 0\%$。
    3.  **Loss Analysis**: `min(train_loss) <= baseline_train_loss` (收敛不差于 Baseline)。

### 阶段 B：深度微调 (Tuning) — **v5.3 ASHA 更新**
*目标：在 (m, p) 空间高效搜索最佳参数。*

* **v5.3 改进**: Grid Search → **ASHA 早停淘汰赛**
    * 更多采样点（Sobol 30 点/op），更少浪费
    * 多保真早停：30ep → 80ep → 200ep
    * 每轮保留 top 1/3，淘汰弱配置

* **配置**:
    * **Input**: 阶段 A 晋级的 Ops。
    * **Sampling**: **Sobol Sequence** (30 个 $(m, p)$ 点/Op)
    * **ASHA Rungs**: [30, 80, 200] epochs
    * **Reduction Factor**: 1/3 (每轮保留最好的 1/3)
* **实验量**: ~8 ops × 30 samples = 240 初始配置
    * Rung 1 (30ep): 240 配置
    * Rung 2 (80ep): ~80 配置 (top 1/3)
    * Rung 3 (200ep): ~27 配置 (top 1/9)
* **预计时间**: ~2-4h (4 GPU 并行) — **比 Grid Search 快 ~10 倍**
* **输出**: 按 `val_acc` 排序的 $(Op, m^*, p^*)$ 列表。

> **ASHA 优势**: 采样更多点（不怕错过最优），早停节省算力（差配置不浪费 200ep），理论最优性保证。

### 阶段 C：先验贪心组合 (Prior-Guided Ensemble) — **v5.4 统一 200ep**
*目标：构建最终策略，验证多操作叠加的有效性。*
* **算法**:
    1.  初始化策略 $P = S_0$，$\text{Acc}(P) = \text{Baseline\_200ep\_acc}$（与 A/B 一致）。
    2.  按阶段 B 的 `mean_val_acc` 排名，尝试叠加 $Op_{new}(m^*, p^*)$。
    3.  **互斥检查**: 跳过与已选操作互斥的候选（如 RandomRotation ↔ RandomPerspective）。
    4.  **Training**: **3 Random Seeds × 200 Epochs**（与 Phase A/B 一致，公平对比）。
    5.  **判定**: 若 $\text{mean\_acc}(P + Op_{new}) > \text{Acc}(P) + 0.3\%$，则接受并更新 $P$。
    6.  **约束**: 组合中额外操作不超过 3 个。
* **实验量**: ~8 ops × 3 seeds × 200 ep（最坏情况）
* **预计时间**: ~1h (单 GPU)
* **输出**: 
    - `phase_c_history.csv` - 每次尝试的记录
    - `phase_c_final_policy.json` - 最终策略定义

> **v5.4 变更**: 统一使用 200 epochs，确保与 Phase A/B 使用相同的 baseline 基准（37%），公平验证"多 Op 组合 > 单 Op > 无增强"的核心假设。

### 阶段 D：SOTA 对比实验 (Benchmark Comparison) — **v5.4 统一 200ep**
*目标：证明你的方法比现成的 SOTA 方法更好。*

在 5 个 Data Folds 上，使用同样的训练设置 (200 Epochs) 运行以下对比组：
1.  **Baseline**: $S_0$ only.
2.  **RandAugment**: N=2, M=9 (标准设置)。
3.  **Cutout**: n_holes=1, length=16.
4.  **Ours (p=1.0)**: 阶段 C 策略但所有 $p$ 强制为 1.0（消融对照）。
5.  **Ours (p=optimal)**: 阶段 C 产出的最终策略 $P_{final}$。

* **实验量**: 5 methods × 5 folds × 200 ep × 1 seed
* **预计时间**: ~1.5h (4 GPU 并行)
* **输出**:
    - `phase_d_results.csv` - 每个 (method, fold) 的结果
    - `phase_d_summary.csv` - Mean ± Std 汇总（用于论文表格）
    - `checkpoints/phase_d_fold{0-4}_best.pth` - Ours_optimal 的 5-fold 模型

> **v5.4 变更**: 统一使用 200 epochs，与 Phase A/B/C 保持一致的训练预算，确保公平对比。

---

## 4. 交付物与分析 (Deliverables & Analysis)

### 4.1 数据文件结构
* `archive_p100/` - v4 实验数据 (p=1.0 固定，用于历史对比)
* `outputs/baseline_result.csv` - Baseline 200ep 结果（所有阶段共用）
* `outputs/phase_a_results.csv` - v5 Phase A 结果 (m, p 联合搜索)
* `outputs/phase_b_tuning_raw.csv` - v5 Phase B 原始结果 (每个 seed)
* `outputs/phase_b_tuning_summary.csv` - v5 Phase B 汇总结果 (mean ± std)
* `outputs/phase_c_history.csv` - Phase C 组合搜索历史
* `outputs/phase_c_final_policy.json` - Phase C 最终策略定义
* `outputs/phase_d_results.csv` - Phase D 原始结果 (每个 method × fold)
* `outputs/phase_d_summary.csv` - Phase D 汇总结果 (Mean ± Std，用于论文)
* `outputs/checkpoints/phase_d_fold{0-4}_best.pth` - 最终 5-fold 模型

### 4.2 论文核心图表 (Paper Figures) — **v5 更新**
1.  **The Performance Table**: Ours vs. RandAugment vs. Baseline (Mean ± Std)。
2.  **2D Heatmaps**: 展示 $(m, p)$ 联合效应
    - X轴=magnitude, Y轴=probability, 颜色=accuracy
3.  **Ablation Study**: p=1.0 vs p=optimal 对比表格
4.  **Efficiency Plot**: 搜索耗时 vs. 准确率。
5.  **Case Study**: 展示被 $P_{final}$ 修正的样本。

---

## 5. 附录：不可变动的超参数 (Fixed Hyperparameters)

| 参数 | 值 | 说明 |
| :--- | :--- | :--- |
| **Model** | ResNet-18 | 标准 Backbones |
| **Optimizer** | SGD | |
| **Learning Rate** | 0.1 | 标准学习率 |
| **Weight Decay** | 5e-3 | 增强正则化 |
| **Label Smoothing** | 0.1 | 标签平滑正则化 |
| **Momentum** | 0.9 | |
| **Scheduler** | CosineAnnealingLR | T_max = Total Epochs - Warmup |
| **Warmup** | 5 epochs | 线性 warmup 到 lr=0.1 |
| **Batch Size** | 128 | 标准 batch size |
| **num_workers** | 8 | DataLoader 并行加载 |
| **prefetch_factor** | 4 | DataLoader 预取 (v5.2: 从 2 增加到 4) |
| **channels_last** | True | NHWC 内存格式 (v5.2 新增) |
| **All Phase Epochs** | **200** | 统一训练预算，公平对比 |
| **AMP** | Enabled | 混合精度加速 |
| **Sobol Seed** | 42 | 可复现性 |

---

## 6. 附录：早停策略 (Early Stopping Strategy) — **v5.1 新增**

### 设计原则

1. **监控指标**: 使用 `val_acc` (mode="max")，而非 `val_loss`
   - 原因: 低数据 + 强增广场景下，val_loss 和 val_acc 趋势可能不一致
2. **min_epochs**: 确保 CosineAnnealingLR 有足够时间发挥作用
3. **min_delta=0.2**: 过滤 val_acc 的噪声波动 (±2-3%)

### 各阶段早停设置

| 阶段 | epochs | min_epochs | patience | min_delta | 说明 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Phase A** | 200 | 80 | 80 | 0.2 | 快速筛选，≥0.4×epochs |
| **Phase B (ASHA)** | 30/80/200 | N/A | N/A | N/A | ASHA 多轮淘汰，每轮保留 top 1/3 |
| **Phase C** | 200 | 80 | 80 | 0.2 | 策略构建，与 A/B 一致 |
| **Phase D** | 200 | 80 | 80 | 0.2 | 最终评估，与 A/B 一致 |

> **Note**: min_epochs 和 patience 按 step 等价量调整，保证 batch size (128) 下 cosine 调度有足够时间生效。

### 论文解释

> Phase A/B 为搜索阶段，使用早停加速筛选（min_epochs=100-120, patience=30-40）。
> Phase C/D 为最终评估阶段，禁用早停以确保公平对比（所有方法训练相同 epochs）。
> 所有阶段按 val_acc 选择最佳 checkpoint。

---

## 7. 附录：训练优化 (Training Optimizations) — **v5.2 新增**

### 优化概述

为加速训练并减少 GPU 空闲时间，应用以下优化：

| 优化项 | 设置 | 预期提速 | 说明 |
| :--- | :--- | :--- | :--- |
| **channels_last** | `memory_format=torch.channels_last` | 10-20% | 避免 NCHW→NHWC 转换 |
| **prefetch_factor** | 4 (从 2 增加) | 5-10% | 减少 DataLoader 等待 |
| **pin_memory** | True | 5-10% | 异步 H2D 传输 |
| **persistent_workers** | True | 3-5% | 避免 worker 重启开销 |

### 实现细节

**1. 模型使用 channels_last 内存格式**
```python
model = model.to(device)
if device.type == "cuda":
    model = model.to(memory_format=torch.channels_last)
```

**2. 输入数据使用 channels_last 传输**
```python
# 在 src/utils.py 的 train_one_epoch 和 evaluate 中
images = images.to(device, non_blocking=True, memory_format=torch.channels_last)
```

**3. DataLoader 优化配置**
```python
DataLoader(
    dataset,
    batch_size=128,
    num_workers=8,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=4,  # v5.2: 从 2 增加到 4
)
```

### 兼容性说明

- **channels_last**: PyTorch 1.5+ 支持，ResNet 全面兼容
- **硬件要求**: CUDA GPU（CPU 训练自动跳过这些优化）
- **精度影响**: 无影响（只改变内存布局，不改变数值）

### 预期效果

| 阶段 | 原时间估算 | 优化后估算 | 提速 |
| :--- | :--- | :--- | :--- |
| Phase A | ~1.5h | ~1h | ~33% |
| Phase B (ASHA) | ~35h (Grid) | **~1h** | **~97%** |
| Phase C | ~1h | ~45min | ~25% |
| Phase D | ~1.5h | ~1h | ~33% |

> **v5.3 亮点**: Phase B 使用 ASHA 早停淘汰，从 ~35 小时降到 ~2-4 小时！

