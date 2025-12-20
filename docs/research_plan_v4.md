# Research Plan: Prior-Guided Augmentation Policy Search in Low-Data Regimes
# 实验计划：低数据体制下的先验引导增强策略搜索

**Version**: Final-v4 (Publication Ready)
**Target**: Academic Conference (WACV / BMVC / ICIP Level)
**Dataset**: CIFAR-100 (Subsampled: 20% per class, Stratified)

---

## 1. 核心研究假设与目标 (Hypothesis & Objective)

### 1.1 研究目标 (Research Goal)
在极少样本（每类 100 张）限制下，寻找一种**计算高效**且**具备鲁棒性**的数据增强策略组合。
* **核心痛点**: 现有的 AutoAugment/RandAugment 在低数据下往往因增强过强导致欠拟合，或因搜索代价过高而不可行。
* **解决方案**: 基于“人类先验（Human Priors）”初始化的三阶段搜索管道（Pipeline）。
* **去NAS化 (No-NAS)**: 严格固定模型架构和超参，仅改变数据输入分布。

### 1.2 严格的数据验证协议 (Strict Validation Protocol)
*必须严格执行，防止数据泄露和方差欺骗。*

* **全集定义**: CIFAR-100 (50,000 images).
* **5-Fold 鲁棒性验证 (The "Golden Standard")**:
    * 将全集按类别分层划分为 5 个互斥的数据折 (Folds)，每折 10,000 张。
    * **目的**: 最终汇报的 Accuracy 必须是 **Mean ± Std (over 5 Folds)**。这意味着你的策略必须在 5 种不同的数据子采样上都有效。
* **搜索用代理集 (Proxy for Search)**:
    * 仅使用 `Fold-0` 进行搜索。
    * 在 `Fold-0` 内部划分：**Train (9,000) / Val (1,000)**。
    * **注意**: 搜索阶段的所有决策基于此 Val Set，但最终汇报基于 Official Test Set。

---

## 2. 增强操作空间 (Augmentation Space)

### 2.1 基础增强 ($S_0$ - Baseline)
*始终开启，作为所有实验的起点。*
* **RandomCrop**: padding=4, size=32
* **RandomHorizontalFlip**: p=0.5

### 2.2 搜索候选池 (Candidate Pool)
*参数范围已针对 CIFAR-100 (32x32) 尺寸进行物理校准。*

| 操作 (Op) | 物理参数范围 ($m \to \text{Real Value}$) | 关键约束 (Implementation Note) |
| :--- | :--- | :--- |
| **RandomResizedCrop** | scale $\in [\mathbf{0.25}, 1.0]$, ratio $\in [0.75, 1.33]$ | **互斥**: 选中时自动禁用 $S_0$ 的 RandomCrop |
| **RandomRotation** | degrees $\in [-30, 30]$ | **互斥**: 与 Perspective 二选一 |
| **RandomPerspective** | distortion_scale $\in [0, 0.5]$ | **互斥**: 与 Rotation 二选一 |
| **ColorJitter** | brightness/contrast/sat $\in [0, 0.8]$ | |
| **RandomGrayscale** | $p \in [0, 0.5]$ (无幅度) | |
| **GaussianBlur** | sigma $\in [0.1, 2.0]$ | |
| **RandomErasing** | scale $\in [0.02, 0.4]$ | 必须放在 Tensor 转换之后 |
| **GaussianNoise** | $\sigma \in [0, 0.1]$ | **必须**: Apply `torch.clamp` 防止溢出 |

---

## 3. 实验执行流程 (Execution Pipeline)

### 阶段 A：广度筛选 (Screening)
*目标：在低成本下快速剔除无效增强，保留高潜力项。*
* **配置**:
    * **Input**: $S_0 + Op_i$ (Single Op)
    * **Sampling**: Sobol Sequence (32 组参数/Op)
    * **Training**: **200 Epochs** (开启 ASHA 早停，Grace Period=40)
* **保留标准 (Retention Criteria)**:
    满足以下任一条件即可晋级：
    1.  **Top-1 Acc**: $\Delta \ge -0.5\%$ (允许轻微掉点)。
    2.  **Top-5 Acc**: $\Delta > 0\%$ (Top-1 掉但 Top-5 涨，说明有正则化潜力)。
    3.  **Loss Analysis**: 训练 Loss 收敛速度快于 Baseline。

### 阶段 B：深度微调 (Tuning)
*目标：消除随机性，确定每个操作的最佳形态。*
* **配置**:
    * **Input**: 阶段 A 晋级的 Ops。
    * **Search**: Grid Search (Top-4 参数附近细搜)。
    * **Robustness Check**: 每个参数跑 **3 个 Random Seeds** (不同初始化)。
* **输出**: 按 `Mean Validation Acc` 排序的参数列表 $\{(Op_1, p_1, m_1), (Op_2, p_2, m_2), ...\}$。

### 阶段 C：先验贪心组合 (Prior-Guided Ensemble)
*目标：构建最终策略。*
* **算法**:
    1.  初始化策略 $P = S_0$。
    2.  按阶段 B 的排名，尝试叠加 $Op_{new}$。
    3.  **Training**: **800 Epochs** (关键！低数据必须增加 Iterations)。
    4.  **判定**: 若 $\text{Acc}(P + Op_{new}) > \text{Acc}(P) + 0.1\%$，则更新 $P$。
    5.  **约束**: 组合中额外操作不超过 3 个（防止过强破坏分布）。

### 阶段 D：SOTA 对比实验 (Benchmark Comparison) —— **论文发表关键**
*目标：证明你的方法比现成的 SOTA 方法更好。*
在 5 个 Data Folds 上，使用同样的训练设置 (800 Epochs) 运行以下对比组：
1.  **Baseline**: $S_0$ only.
2.  **RandAugment**: N=2, M=9 (标准设置)。
3.  **Cutout**: n_holes=1, length=16.
4.  **Ours**: 阶段 C 产出的最终策略 $P_{final}$。

---

## 4. 交付物与分析 (Deliverables & Analysis)

### 4.1 数据文件结构
运行代码后，必须自动生成以下 Log：
* `results/phase_a_screening.csv`: [Op, Param, Acc, Loss_Slope]
* `results/phase_c_history.csv`: [Step, Added_Op, Val_Acc, Improvement]
* `results/final_test_5folds.csv`: [Fold_ID, Method, Test_Acc]

### 4.2 论文核心图表 (Paper Figures)
1.  **The Performance Table**: 你的方法 vs. RandAugment vs. Baseline (Mean $\pm$ Std)。
2.  **Heatmaps (Phase A)**: 展示不同增强操作对强度的敏感性（例如：Rotation 在 15度最好，30度就崩了）。
3.  **Efficiency Plot**: X轴=搜索耗时 (GPU Hours)，Y轴=最终准确率。展示你的方法比 AutoAugment 快几个数量级。
4.  **Case Study**: 展示被 $P_{final}$ 修正的样本（Baseline 分错，Ours 分对的图）。

---

## 5. 附录：不可变动的超参数 (Fixed Hyperparameters)
*为了保证公平性（No-NAS），以下参数在所有阶段（A/B/C/D）严禁修改。*

| 参数 | 值 | 说明 |
| :--- | :--- | :--- |
| **Model** | ResNet-18 | 标准 Backbones |
| **Optimizer** | SGD | |
| **Learning Rate** | 0.05 | 小数据平衡优化 (原0.1) |
| **Weight Decay** | 1e-3 | 增强正则化 (原5e-4) |
| **Momentum** | 0.9 | |
| **Scheduler** | CosineAnnealingLR | T_max = Total Epochs |
| **Batch Size** | 64 | 增加更新次数 (原128) |
| **Phase A Epochs** | 200 | 快速筛选 |
| **Phase C/D Epochs** | **800** | **关键修正**: 补偿数据量减少 |
| **AMP** | Enabled | 混合精度加速 |