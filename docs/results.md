# 结果与配置汇总（模板）

- 数据集：CIFAR-100 5-Fold，子采样 20%/类。
- 训练配置：epochs=200，batch_size=128，lr=0.1，wd=1e-2，momentum=0.9，warmup=5，label smoothing=0.1，seeds=42/123/456（Phase B/C），folds=0-4（Phase D）。
- 硬件/驱动：CUDA 12.8.1，Driver 570.195.03，cuDNN 9.8.0.87。

## Phase D 主表（Mean ± Std）
来源：`outputs/phase_d_summary.csv`
- Baseline:
- Baseline-NoAug:
- RandAugment:
- Cutout:
- Best_SingleOp:
- Ours_p1:
- Ours_optimal:

## 最终策略
来源：`outputs/phase_c_final_policy.json`  
示例：`[("Op", m, p_adjusted), ...]`

## 关键图表（生成后嵌入）
- Phase A (m,p) 热力图：`outputs/figures/phase_a_heatmap.png`
- 效率/收敛曲线：`outputs/figures/efficiency.png`
- Ablation 柱状：`outputs/figures/ablation.png`
- 稳定性箱线（Phase B multi-seed）：`outputs/figures/stability.png`
- 类别均衡：`outputs/figures/class_balance.png`
- 增强可视化：`outputs/figures/augment_examples.png`
