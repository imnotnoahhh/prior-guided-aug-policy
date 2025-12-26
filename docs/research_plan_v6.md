# Research Plan: Prior-Guided Augmentation Policy Search in Low-Data Regimes
# 实验计划：低数据体制下的先验引导增强策略搜索

**Version**: v6 (Dynamic Prior-Guided Policy)
**Target**: Academic Conference (WACV / BMVC / ICIP Level)
**Dataset**: CIFAR-100 (Subsampled: 20% per class, Stratified)

---

## Changelog (v5 → v6)

| 变更类型 | 内容 | 代码影响 |
|:---|:---|:---|
| **[NEW]** | **Dynamic Strategy**: 放弃静态组合，采用动态随机采样 | `augmentations.py`: 新增 `DynamicAugment` |
| **[CHANGED]** | Phase C: Greedy Search → **Top-K Elite Selection** | `main_phase_c.py`: 选 Top-K 优选池 |
| **[CHANGED]** | Policy Definition: List[Op] → Pool[Op] + N=2 | `phase_c_final_policy.json` 结构变更 |
| **[REMOVED]** | 移除复杂的 greedy search 循环 | 大幅简化 Phase C 流程 |

### 设计动机 (v6 Update)

**核心痛点**: 
1. **静态策略的局限**: v5 搜索出的静态组合（如 `ColorJitter + Rotation`）对所有图片一视同仁，缺乏多样性。
2. **RandAugment 的优势**: 在低数据下，RandAugment 的强项在于"每次迭代都不同"（Dynamic Diversity），这是一种极其强大的正则化。
3. **RandAugment 的劣势**: 它是"盲目"的，操作池里包含不适合当前数据的"有害操作"（如在极少样本下 Cutout 可能遮挡关键特征）。

**v6 解决方案 (Best of Both Worlds)**:
**"Dynamic Prior-Guided Policy"**
1. **Smart Pruning (智能筛选)**: 利用 Phase A/B 剔除有害操作，保留 **Top-K 精英操作**。
2. **Dynamic Sampling (动态采样)**: 在训练时，从精英池中每张图随机采样 N=2 个操作（模拟 RandAugment）。
**预期**: 比 RandAugment 更稳（无垃圾操作），比 v5 静态策略更强（更多样性）。

---

## 1. 核心研究假设与目标 (Hypothesis & Objective)

### 1.1 研究目标 (Research Goal)
在极少样本（每类 100 张）限制下，验证 **"精选子集上的动态策略 > 全集上的盲目动态策略 (RandAugment)"**。

* **核心假设**: 
    1. 在低数据体制下，操作池的质量（Pureness）比数量（Quantity）更重要。
    2. 动态随机性（Stochasticity）是防止过拟合的关键。
* **解决方案**: 三阶段搜索管道（Pipeline）：
    1. Phase A: 广度筛选有效操作 (Screening)
    2. Phase B: 深度调参找最佳 (m, p) (Tuning)
    3. **Phase C (v6)**: **构建精英池 (Elite Pool Construction) & 动态验证**
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
| **RandomErasing** | [0.05, 0.3] | [0.1, 0.5] | 破坏性 | 需低 p，放在 Tensor 后 (v5.4 扩展) |
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
    "RandomErasing":     {"m": [0.05, 0.3], "p": [0.1, 0.5]},  # v5.4: 扩展到 0.5
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
    * 每轮保留 top 1/2，淘汰弱配置

* **配置**:
    * **Input**: 阶段 A 晋级的 Ops。
    * **Sampling**: **Sobol Sequence** (30 个 $(m, p)$ 点/Op)
    * **ASHA Rungs**: [30, 80, 200] epochs
    * **Reduction Factor**: 1/2 (每轮保留最好的 1/2)
* **实验量**: ~8 ops × 30 samples = 240 初始配置
    * Rung 1 (30ep): 240 配置
    * Rung 2 (80ep): ~120 配置 (top 1/2)
    * Rung 3 (200ep): ~60 配置 (top 1/4)
* **预计时间**: ~2-4h (4 GPU 并行) — **比 Grid Search 快 ~10 倍**
* **输出**: 按 `val_acc` 排序的 $(Op, m^*, p^*)$ 列表。

### 阶段 C：精英池构建与动态验证 (Elite Pool Construction) — **v6 Dynamic**
*目标：构建精英操作池，验证"动态策略"的有效性。*

* **算法**:
    1.  **Selection**: 选取 Phase B 中表现最好的 **Top-K (e.g. 6)** 个操作，构成 "Elite Pool"。
    2.  **Validation**: 使用 **Top-K 精英池** + **N=2 随机采样** 进行训练 (Dynamic Augmentation)。
    3.  **Constraint**: 仅使用 Phase B 的最佳 $m$ 值，不使用复杂的组合搜索。
* **Training**: **3 Random Seeds × 200 Epochs**。
* **判定**: 若 Dynamic Policy 优于 Baseline 且逼近/超越 Best Single Op，则为成功。
* **实验量**: 1 Policy × 3 seeds × 200 ep
* **预计时间**: ~30 min (单 GPU)
* **输出**: 
    - `phase_c_final_policy.json` - 包含 Elite Pool 操作列表和参数 N。

> **v6 变更**: 移除了贪心搜索，改为更纯粹的 "Filter + Random Select" 模式。

### 阶段 D：SOTA 对比实验 (Benchmark Comparison) — **v6 Dynamic**
*目标：证明"Prior-Guided Dynamic Policy"比"Blind Dynamic Policy (RandAugment)"更好。*

在 5 个 Data Folds 上，使用同样的训练设置 (200 Epochs) 运行以下对比组：
1.  **Baseline**: $S_0$ only.
2.  **RandAugment**: N=2, M=9 (标准设置，盲目全集采样)。
3.  **Cutout**: n_holes=1, length=16.
4.  **Best Single Op**: Phase B 也就是最佳单操作 (Upper Bound check)。
5.  **Ours_dynamic**: **Elite Pool (K=6) + N=2** (本次提出的方法)。

* **实验量**: 5 methods × 5 folds × 200 ep × 1 seed
* **预计时间**: ~1.5h (4 GPU 并行)
* **输出**:
    - `phase_d_results.csv`
    - `phase_d_summary.csv`
    - `checkpoints/phase_d_*.pth`

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
| **Weight Decay** | 1e-2 | 增强正则化 (v5.4: 从 5e-3 增加到 1e-2) |
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
| **Phase A** | 200 | 60 | 60 | 0.2 | 快速筛选，≥0.3×epochs |
| **Phase B (ASHA)** | 30/80/200 | N/A | N/A | N/A | ASHA 多轮淘汰，每轮保留 top 1/2 |
| **Phase C** | 200 | 60 | 60 | 0.2 | 策略构建，与 A/B 一致 |
| **Phase D** | 200 | 60 | 60 | 0.2 | 最终评估，与 A/B 一致 |

> **Note**: min_epochs=60 和 patience=60 针对 batch size=128 优化，小 batch 下每个 epoch step 数更多，模型更早收敛。

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
