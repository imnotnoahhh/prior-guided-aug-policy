# Research Plan: Prior-Guided Augmentation Policy Search in Low-Data Regimes
# 实验计划：低数据体制下的先验引导增强策略搜索

**Version**: v7 (Greedy Combination with Validation)
**Target**: Academic Conference (WACV / BMVC / ICIP Level)
**Dataset**: CIFAR-100 (Subsampled: 20% per class, Stratified)

---

## Changelog (v6 → v7)

| 变更类型 | 内容 | 代码影响 |
|:---|:---|:---|
| **[CHANGED]** | OP_SEARCH_SPACE: 扩大 magnitude 范围 | `augmentations.py`: 约 1.5x 放大 |
| **[CHANGED]** | Phase C: Dynamic Pool → **Greedy Combination** | `main_phase_c.py`: 验证组合有效性 |
| **[NEW]** | 组合验证: min_improvement + t-test | 确保组合真的优于单操作 |
| **[CHANGED]** | Policy JSON: 保留 probability | 完整 (name, m, p) 信息 |

### 设计动机 (v7 Update)

**v6 问题诊断**:
1. **Magnitude 范围过于保守**: v5.5 为避免"必炸区"设置了太小的范围，导致增广强度不足。
2. **组合未验证**: v6 直接假设组合优于单操作，但实验显示 Ours (40.2%) < Best_SingleOp (40.4%)。
3. **严重过拟合**: Train 99% vs Val 40%，说明增广没有有效阻止过拟合。

**v7 解决方案**:
1. **扩大搜索范围**: Magnitude 上限从 0.2~0.3 扩展到 0.35~0.5。
2. **贪心验证组合**: 只有当组合 > 单操作 + 0.1% 时才接受。
3. **统计显著性检验**: 使用 t-test (p < 0.2) 验证改进的可靠性。

---

## 1. 核心研究假设与目标

### 1.1 研究目标

在极少样本（每类 100 张）限制下，通过先验引导的搜索找到最优增强策略。

* **核心假设**: 精选的操作组合 > 盲目的 RandAugment（或至少相当，但更高效）。
* **三阶段搜索管道**:
    1. Phase A: 广度筛选有效操作
    2. Phase B: 深度调参找最佳 (m, p)
    3. **Phase C (v7)**: **贪心组合搜索 + 验证**

---

## 2. 增强操作搜索空间 (v7 更新)

### v7 扩展后的搜索范围

```python
OP_SEARCH_SPACE = {
    # Mild operations
    "ColorJitter":       {"m": [0.1, 0.9], "p": [0.3, 0.9]},   # v7: expanded
    "RandomGrayscale":   {"m": [0.5, 0.5], "p": [0.1, 0.7]},
    "GaussianNoise":     {"m": [0.02, 0.4], "p": [0.3, 0.9]},  # v7: expanded
    
    # Medium operations
    "RandomResizedCrop": {"m": [0.4, 0.95], "p": [0.3, 0.8]},
    "RandomRotation":    {"m": [0.0, 0.5], "p": [0.2, 0.7]},   # v7: expanded
    "GaussianBlur":      {"m": [0.0, 0.5], "p": [0.2, 0.7]},   # v7: expanded
    
    # Destructive operations
    "RandomErasing":     {"m": [0.02, 0.35], "p": [0.1, 0.6]}, # v7: expanded
    "RandomPerspective": {"m": [0.0, 0.4], "p": [0.1, 0.6]},   # v7: expanded
}
```

---

## 3. Phase C: 贪心组合搜索 (v7 核心算法)

### 算法伪代码

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

### 关键参数说明

| 参数 | 值 | 说明 |
|:---|:---|:---|
| `max_ops` | 3 | 防止过度叠加 |
| `min_improvement` | 0.1% | 必须有实质性提升 |
| `p_any_target` | 0.7 | 70% 图会被增广 |
| t-test threshold | p < 0.2 | 统计显著性门槛 |

---

## 4. 预期效果

| 版本 | Ours Acc | RandAugment | 差距 |
|:---|:---|:---|:---|
| v6 (原) | 40.2% | 42.2% | -2.0% |
| **v7 (改)** | **41.5%** (预期) | 42.2% | -0.7% (预期) |

**改进来源**:
1. 更大的 magnitude → 更强的正则化
2. 验证组合 → 确保不会退化
3. 概率调整 → 控制总增广强度

---

## 5. 使用方法

```bash
# 完整运行 (v7 贪心组合)
python main_phase_c.py --max_ops 3 --min_improvement 0.1 --p_any_target 0.7

# Dry run 测试
python main_phase_c.py --dry_run
```

---

## 6. 附录：超参数 (与 v6 相同)

| 参数 | 值 |
|:---|:---|
| Model | ResNet-18 |
| Optimizer | SGD (lr=0.1, momentum=0.9) |
| Weight Decay | 0.005 (from Phase 0) |
| Label Smoothing | 0.0 (from Phase 0) |
| Scheduler | CosineAnnealingLR + Warmup 5 |
| Epochs | 200 |
| Batch Size | 128 |
