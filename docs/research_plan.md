# Research Plan: Prior-Guided Augmentation: A Reliable Strategy for Small-Sample Datasets
# 实验计划：先验引导增强——小样本数据集的可靠策略

**Target**: Academic Conference (WACV / BMVC / ICIP Level)
**Dataset**: CIFAR-100 (Subsampled: 20% per class, Stratified)

---

## 1. 核心研究假设与目标

### 1.1 研究目标

在极少样本（每类 100 张）限制下，探究**增强策略复杂度与模型性能**的关系。通过先验引导的搜索流程，验证简单策略在低数据体制下的优越性。

*   **核心假设 (Simplicity Hypothesis)**: 在小样本场景下，**精选的单一操作 (Optimal Single-Op)** 足以逼近 SOTA (RandAugment) 的性能，且具备更高的稳定性（更低方差）和可解释性。复杂的组合增强策略（Multi-Op）往往会导致收益递减甚至过拟合。
*   **研究定位**: 我们不追求绝对的 SOTA 准确率（Accuracy），而是追求 **"Accuracy-Stability-Complexity"** 三者的最佳平衡（Pareto Optimal）。
*   **四阶段搜索管道**:
    1.  Phase 0: 超参校准（确定 weight_decay, label_smoothing）
    2.  Phase A: 广度筛选有效操作
    3.  Phase B: 深度调参找最佳 (m, p)
    4.  Phase C: 验证性组合搜索（验证多操作是否真正有效，体现“奥卡姆剃刀”原则）
*   **最终评估**: Phase D: 5-Fold 交叉验证，重点对比 **Ours (Stability)** vs **RandAugment (Complexity)**。

---

## 2. 增强操作搜索空间

### 搜索范围

```python
OP_SEARCH_SPACE = {
    # Mild operations
    "ColorJitter":       {"m": [0.1, 0.9], "p": [0.3, 0.9]},
    "RandomGrayscale":   {"m": [0.5, 0.5], "p": [0.1, 0.7]},
    "GaussianNoise":     {"m": [0.02, 0.4], "p": [0.3, 0.9]},
    
    # Medium operations
    "RandomResizedCrop": {"m": [0.4, 0.95], "p": [0.3, 0.8]},
    "RandomRotation":    {"m": [0.0, 0.5], "p": [0.2, 0.7]},
    "GaussianBlur":      {"m": [0.0, 0.5], "p": [0.2, 0.7]},
    
    # Destructive operations
    "RandomErasing":     {"m": [0.02, 0.35], "p": [0.1, 0.6]},
    "RandomPerspective": {"m": [0.0, 0.4], "p": [0.1, 0.6]},
}
```

### 参数说明

- `m`: magnitude 搜索范围 [min, max]，控制增强强度
- `p`: probability 搜索范围 [min, max]，控制应用概率

### 互斥约束

```python
MUTUAL_EXCLUSION = {
    "RandomResizedCrop": ["S0_RandomCrop"],
    "RandomRotation": ["RandomPerspective"],
    "RandomPerspective": ["RandomRotation"],
}
```

---

## 3. 实验流程

### 3.0 Phase 0: 超参校准 (Hyperparameter Calibration)

**目标**: 确定最优正则化参数组合

| 项目 | 配置 |
|------|------|
| **候选** | weight_decay ∈ {1e-4, 1e-3, 1e-2, 5e-2}, label_smoothing ∈ {0.0, 0.1, 0.2} |
| **训练** | 100 epochs, 3 seeds, Fold-0 |
| **选择** | 按 mean_val_acc 选最佳组合 |
| **输出** | `outputs/phase0_summary.csv` |

### 3.1 Phase A: 广度筛选 (Low-Fidelity Screening)

**目标**: 在 (m, p) 联合空间快速剔除无效增强

| 项目 | 配置 |
|------|------|
| **输入** | S0 + Op_i(m, p)（单操作） |
| **采样** | 2D Sobol Sequence, 32 组/Op |
| **训练** | 40 epochs, seed=42, Fold-0 |
| **评分** | `mean(top3(val_acc[30:40]))` |
| **实验量** | 8 ops × 32 点 = 256 组 |

### 3.2 Phase B: 深度微调 (ASHA Tuning)

**目标**: 高效搜索最佳 (m, p) 参数

| 项目 | 配置 |
|------|------|
| **采样** | Sobol Sequence, 30 组/Op |
| **算法** | ASHA 早停淘汰赛 |
| **Rungs** | [40, 100, 200] epochs |
| **Reduction Factor** | 1/2（每轮保留最好的一半） |
| **Final Rung** | 3 seeds 平均 (42, 123, 456) |

### 3.3 Phase C: 贪心组合搜索

**目标**: 构建最终多操作组合策略

**算法伪代码**:

```
输入: Phase B 结果 (按 val_acc 排序)
输出: 最优策略 (1~max_ops 个操作)

参数:
  max_ops = 3           # 最多叠加 3 个操作
  min_improvement = 0.1 # 至少提升 0.1%
  p_any_target = 0.7    # 目标总增广概率

算法:
1. current_policy = [Phase B 最佳操作]
   current_acc = evaluate(current_policy, seeds=3)

2. FOR i = 2 to max_ops:
     candidates = Phase B 中未使用的操作 (排除互斥)
     
     FOR each candidate in candidates:
       test_policy = current_policy + candidate
       调整概率: adjust_probabilities(test_policy, p_any_target)
       test_acc = evaluate(test_policy, seeds=3)
       
     best_candidate = 找出最高 test_acc
     improvement = best_candidate_acc - current_acc
     p_value = t_test(best_candidate_results, current_results)
     
     IF improvement >= min_improvement AND p_value < 0.2:
       current_policy.append(best_candidate)
       current_acc = best_candidate_acc
     ELSE:
       BREAK

3. 返回 current_policy
```

### 3.5 Phase D: SOTA 对比 (Benchmark Comparison)

**目标**: 证明方法优于或接近现有 SOTA

| 项目 | 配置 |
|------|------|
| **验证** | 5-Fold 交叉验证 |
| **训练** | 200 epochs/fold |
| **方法** | Baseline, Baseline-NoAug, RandAugment, Cutout, Best_SingleOp, Ours_p1, Ours_optimal (7 个) |

### 3.6 补充实验 (Supplementary Experiments)

**目标**: 验证泛化性 (Generalization) 和设计必要性 (Necessity)

| 实验 | 描述 | 目的 |
|---|---|---|
| **CIFAR-10 Generalization** | 50-shot setting, 5-fold, 200 epochs | 证明方法不局限于 CIFAR-100，具备通用鲁棒性 |
| **Ablation (Fixed Probability)** | 固定 p=0.5，搜索 m | 证明搜索 Magnitude 的必要性 (Sensitivity Analysis) |

---

## 4. 超参数配置

| 参数 | 值 |
|------|------|
| Model | ResNet-18 |
| Optimizer | SGD (lr=0.1, momentum=0.9) |
| Weight Decay | 1e-2 (Phase 0 确定) |
| Label Smoothing | 0.1 (Phase 0 确定) |
| Scheduler | CosineAnnealingLR + Warmup 5 epochs |
| Epochs | 200 |
| Batch Size | 128 |

---

## 5. 使用方法

```bash
# 完整训练流程
bash scripts/train_single_gpu.sh

# 单独运行 Phase C
python main_phase_c.py --max_ops 3 --min_improvement 0.1 --p_any_target 0.7

# Dry run 测试
python main_phase_c.py --dry_run
```

---

## 6. 输出文件

```
outputs/
├── phase0_summary.csv           # Phase 0 超参校准结果
├── baseline_result.csv          # Baseline 结果
├── phase_a_results.csv          # Phase A 筛选结果
├── phase_b_tuning_summary.csv   # Phase B 汇总结果
├── phase_c_history.csv          # Phase C 组合历史
├── phase_c_final_policy.json    # Phase C 最终策略
├── phase_d_results.csv          # Phase D 原始结果
└── phase_d_summary.csv          # Phase D 汇总结果 (论文用)
├── ablation/                    # 消融实验结果
└── cifar10_50shot_results.csv   # CIFAR-10 泛化实验结果
```
