# Research Plan: Prior-Guided Augmentation Policy Search
# 实验计划：先验引导增强策略搜索

**Target**: Academic Conference (WACV / BMVC / ICIP Level)
**Dataset**: CIFAR-100 (Subsampled: 20% per class, Stratified)

---

## 1. 核心研究假设与目标

### 1.1 研究目标

在极少样本（每类 100 张）限制下，通过先验引导的搜索找到最优增强策略。

* **核心假设**: 精选的操作组合 ≥ 盲目的 RandAugment（或至少相当，但更高效）。
* **三阶段搜索管道**:
    1. Phase A: 广度筛选有效操作
    2. Phase B: 深度调参找最佳 (m, p)
    3. Phase C: 贪心组合搜索 + 验证

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

### 3.4 Phase D: SOTA 对比 (Benchmark Comparison)

**目标**: 证明方法优于或接近现有 SOTA

| 项目 | 配置 |
|------|------|
| **验证** | 5-Fold 交叉验证 |
| **训练** | 200 epochs/fold |
| **方法** | Baseline, RandAugment, Cutout, Best_SingleOp, Ours_p1, Ours_optimal |

---

## 4. 超参数配置

| 参数 | 值 |
|------|------|
| Model | ResNet-18 |
| Optimizer | SGD (lr=0.1, momentum=0.9) |
| Weight Decay | Phase 0 搜索确定 |
| Label Smoothing | Phase 0 搜索确定 |
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
```
