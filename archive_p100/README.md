# Archive: p=1.0 Experiments (消融实验对照组)

**归档日期**: 2025-12-23
**归档原因**: 原实验使用固定 probability=1.0，现改为搜索 (magnitude, probability) 联合空间

## 归档内容

### outputs/
- `baseline_result.csv` - Baseline (S0) 实验结果
- `phase_a_results.csv` - Phase A 广度筛选结果 (256 组实验)
- `phase_b_results.csv` - Phase B 深度微调结果 (448 组实验)

### logs/
- `baseline.log` - Baseline 训练日志
- `phase_a_gpu*.log` - Phase A 各 GPU 日志
- `phase_b_gpu*.log` - Phase B 各 GPU 日志

## 实验配置

| 参数 | 值 |
|:---|:---|
| probability | **1.0 (固定)** |
| magnitude | 0~1 (32 档 Sobol) |
| Phase A | 8 ops × 32 m × 1 seed = 256 |
| Phase B | 8 ops × ~18 m × 3 seeds = 448 |

## 关键发现

1. 最佳单操作提升: +2.1% (RandomGrayscale)
2. 发现 100% 增强可能过强
3. 启发了 v5 版本引入 probability 搜索

## 用途

此数据用于论文的 **消融实验 (Ablation Study)**:
- 对比 p=1.0 vs p=optimal
- 证明 probability 搜索的重要性

