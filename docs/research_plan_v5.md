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
| **[CHANGED]** | Phase B: 1D Grid → 2D Grid (m, p) | `main_phase_b.py`: 2D 网格搜索 |
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
在极少样本（每类 100 张）限制下，寻找一种**计算高效**且**具备鲁棒性**的数据增强策略组合。
* **核心痛点**: 现有的 AutoAugment/RandAugment 在低数据下往往因增强过强导致欠拟合，或因搜索代价过高而不可行。
* **解决方案**: 基于"人类先验（Human Priors）"初始化的三阶段搜索管道（Pipeline）。
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

### 阶段 B：深度微调 (Tuning) — **v5 更新**
*目标：在 (m, p) 空间精细搜索最佳参数。*

* **配置**:
    * **Input**: 阶段 A 晋级的 Ops。
    * **Search**: **5×5 Grid Search** in $(m, p)$ space (步长 0.1，以 Top-K centers 为中心 ±0.2)
    * **Robustness Check**: 每个参数跑 **3 个 Random Seeds**。
* **实验量**: ~promoted_ops × ~50 点 × 3 seeds（去重后实际点数）
* **预计时间**: ~1.5h (4 GPU 并行)
* **输出**: 按 `Mean Validation Acc` 排序的 $(Op, m^*, p^*)$ 列表。

### 阶段 C：先验贪心组合 (Prior-Guided Ensemble)
*目标：构建最终策略。*
* **算法**:
    1.  初始化策略 $P = S_0$。
    2.  按阶段 B 的排名，尝试叠加 $Op_{new}(m^*, p^*)$。
    3.  **Training**: **800 Epochs**。
    4.  **判定**: 若 $\text{Acc}(P + Op_{new}) > \text{Acc}(P) + 0.1\%$，则更新 $P$。
    5.  **约束**: 组合中额外操作不超过 3 个。

### 阶段 D：SOTA 对比实验 (Benchmark Comparison)
*目标：证明你的方法比现成的 SOTA 方法更好。*

在 5 个 Data Folds 上，使用同样的训练设置 (800 Epochs) 运行以下对比组：
1.  **Baseline**: $S_0$ only.
2.  **RandAugment**: N=2, M=9 (标准设置)。
3.  **Cutout**: n_holes=1, length=16.
4.  **Ours (p=1.0)**: archive_p100 的结果（消融对照）。
5.  **Ours (p=optimal)**: 阶段 C 产出的最终策略 $P_{final}$。

---

## 4. 交付物与分析 (Deliverables & Analysis)

### 4.1 数据文件结构
* `archive_p100/` - v4 实验数据 (p=1.0 固定)
* `outputs/baseline_result.csv` - Baseline 结果
* `outputs/phase_a_results.csv` - v5 Phase A 结果 (m, p 联合搜索)
* `outputs/phase_b_tuning_raw.csv` - v5 Phase B 原始结果 (每个 seed)
* `outputs/phase_b_tuning_summary.csv` - v5 Phase B 汇总结果 (mean ± std)
* `results/phase_c_history.csv` - 组合搜索历史 (待实现)
* `results/final_test_5folds.csv` - 最终 5-fold 测试结果 (待实现)

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
| **Learning Rate** | 0.05 | 小数据平衡优化 |
| **Weight Decay** | 1e-3 | 增强正则化 |
| **Momentum** | 0.9 | |
| **Scheduler** | CosineAnnealingLR | T_max = Total Epochs |
| **Batch Size** | 64 | |
| **Phase A Epochs** | 200 | 快速筛选 |
| **Phase C/D Epochs** | **800** | 补偿数据量减少 |
| **AMP** | Enabled | 混合精度加速 |
| **Sobol Seed** | 42 | 可复现性 |

