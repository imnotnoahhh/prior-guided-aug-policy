# 复现指南

- 硬件/驱动：CUDA 12.8、NVIDIA Driver 570.133.20、cuDNN 9.8.0.87（Single NVIDIA A10）。
- 环境：`conda env create -f environment.yml && conda activate rethinking_aug`。如已创建，`conda env update -f environment.yml` 保持同步。
- 数据：自动下载 CIFAR-100 至 `./data`，若离线，请预放置官方二进制文件。

### 数据分割详解

```
┌─────────────────────────────────────────────────────────────┐
│  CIFAR-100 完整训练集: 50,000 张 (100 类 × 500 张/类)       │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
              ┌───────────────────────────────────┐
              │  Stratified 5-Fold 分层划分       │
              │  (保证每个 Fold 类别分布一致)      │
              └───────────────────────────────────┘
                              │
        ┌─────────┬─────────┬─┴─────────┬─────────┐
        ▼         ▼         ▼           ▼         ▼
    ┌───────┐ ┌───────┐ ┌───────┐ ┌───────┐ ┌───────┐
    │Fold 0 │ │Fold 1 │ │Fold 2 │ │Fold 3 │ │Fold 4 │
    │10,000 │ │10,000 │ │10,000 │ │10,000 │ │10,000 │
    │100/类 │ │100/类 │ │100/类 │ │100/类 │ │100/类 │
    └───────┘ └───────┘ └───────┘ └───────┘ └───────┘
        │
        ▼ (以 Fold 0 为例)
    ┌─────────────────────────────────────────┐
    │  Fold 内 90/10 分割                      │
    │                                         │
    │  ┌─────────────────┐  ┌───────────────┐ │
    │  │   训练集 (90%)   │  │  验证集 (10%) │ │
    │  │   9,000 张       │  │   1,000 张    │ │
    │  │   90 张/类       │  │   10 张/类    │ │
    │  └─────────────────┘  └───────────────┘ │
    └─────────────────────────────────────────┘
                │
                ▼
    论文中称为 "~90-shot" (每类约 90 个训练样本)
```

**关键点**:
- 论文中 "~90-shot" 指的是**训练样本数**，不是 Fold 总样本数
- 5 个 Fold 的数据**互不重叠**，用于交叉验证
- 搜索阶段 (Phase A/B/C) 只用 Fold 0
- 最终评估 (Phase D) 用全部 5 个 Fold，汇报 Mean ± Std

## 逐阶段命令（单 GPU）
```bash
bash scripts/train_single_gpu.sh
```
产物默认写入 `outputs/`，日志在 `logs/`。关键文件：
- `outputs/phase0_summary.csv`
- `outputs/baseline_result.csv`
- `outputs/phase_a_results.csv`
- `outputs/phase_b_tuning_raw.csv`, `outputs/phase_b_tuning_summary.csv`
- `outputs/phase_c_history.csv`, `outputs/phase_c_final_policy.json`
- `outputs/phase_d_results.csv`, `outputs/phase_d_summary.csv`
- `outputs/checkpoints/*.pth`

## 验证
- 冒烟：`python main_phase_a.py --dry_run` 等各阶段 dry_run 模式（1-2 epoch）。
- 完整：`phase_d_results.csv` 应含 7 methods × 5 folds = 35 行结果。
- 汇总：`phase_d_summary.csv` 包含各方法的 Mean ± Std 统计。

## 补充实验复现 (Paper Revision)
为了回应审稿人关于“破坏性”和“公平性”的质疑，请运行以下脚本：

### 1. 破坏性分析 (Semantic Preservation) ✅ 论文中使用
计算 LPIPS/SSIM 指标，验证语义保真度（论文 Fig.8, Section IV）：
```bash
python scripts/calculate_destructiveness.py
```
Output: `outputs/destructiveness_metrics.csv`

### 2. 稳定性验证 (Zero Variance Verification) ❌ 论文中未使用
在 3 个随机种子下验证 "0 方差" 现象：
```bash
python scripts/run_stability_check.py
```
Output: `outputs/stability_seeds_results.csv`

### 3. 公平性对比 (Tuned RandAugment) ✅ 论文中使用
搜索最佳 RandAugment 参数 (N, M) 并进行全量验证（论文 Section IV, 35.30%）：
```bash
# 步骤 1: 搜索最佳参数
python scripts/run_tuned_randaugment.py
# Output: outputs/tuned_randaugment_results.csv

# 步骤 2: 使用最佳参数 (N=1, M=2) 跑全量验证
python scripts/run_final_tuned_ra.py
# Output: outputs/final_tuned_ra_result.txt



### 4. 搜索流程消融 (Search Workflow Ablation) ✅ 论文中使用
验证各搜索阶段的必要性（论文 Table III）：
```bash
python scripts/run_search_ablation.py
```
对比 Phase A only (Sobol 筛选最优配置) vs Full SAS (含 ASHA 调优)。

Output: `outputs/search_ablation_results.csv`

### 5. CIFAR-10 泛化实验 (Cross-dataset Generalization) ❌ 论文中未使用
在 CIFAR-10 (50-shot) 上验证泛化性（因结果异常已删除）：
```bash
python scripts/run_cifar10_50shot.py
```
Output: `outputs/cifar10_50shot_results.csv`

### 6. Magnitude 消融 (Ablation: Fixed p=0.5) ✅ 论文中使用
验证 Magnitude 搜索的必要性（论文 Fig.10）：
```bash
python scripts/run_ablation_fixed_p.py
```
Output: `outputs/ablation/ablation_p0.5_*.csv`

### 7. Table 1 统计分析 ✅ 论文中使用（部分）
生成扩展统计指标（论文 Table II 使用 CV, Width）：
```bash
python scripts/analyze_table1_stats.py
```
Output: `outputs/table1_extended.csv`

## 后台运行示例
如需断线续跑，可使用 nohup：
```bash
nohup bash scripts/train_single_gpu.sh > logs/full_run.log 2>&1 &
tail -f logs/full_run.log
```

## Shot Sweep 实验 (Trend Analysis) ❌ 论文中未使用
验证 SAS 在不同样本量下的表现趋势（备用数据，未纳入 EUSIPCO 论文）：

**重要配置**:
- 每个 Step 使用独立 output_dir，避免结果污染
- 使用 `--no_early_stop` 禁用 early stopping，保证每个 fold 训练预算一致
- Windows 上使用 `--num_workers 0`

```bash
# Step 1: 验证 SAS 收敛 (~15-20 min)
python scripts/run_shot_sweep.py --shots 50 --folds 0 --epochs 50 \
  --methods Baseline,SAS --output_dir outputs_step1 --batch_size 128 --num_workers 8 --no_early_stop

# Step 2: 加 RandAugment 和 20-shot 判断趋势 (~1-1.5h total)
python scripts/run_shot_sweep.py --shots 50,20 --folds 0 --epochs 50 \
  --methods Baseline,RandAugment,SAS --output_dir outputs_step2 --batch_size 128 --num_workers 8 --no_early_stop

# 画图检查趋势
python scripts/plot_shot_sweep.py --output_dir outputs_step2

# Step 3: 完整版 (只有 Step 2 有意义才跑)
python scripts/run_shot_sweep.py --shots 20,50,100,200 --folds 0,1,2,3,4 --epochs 200 \
  --methods Baseline,RandAugment,SAS --output_dir outputs/shot_sweep_final \
  --batch_size 128 --num_workers 0 --no_early_stop --sas_config ColorJitter,0.2575,0.4239 \
  --log_file outputs/shot_sweep_final/run.log

# 画最终图
python scripts/plot_shot_sweep.py --output_dir outputs/shot_sweep_final
```

注意:
- `--no_early_stop` 禁用 early stopping，所有 fold 统一跑满 epochs，保证训练预算一致
- `--sas_config` 显式指定 SAS 参数 (Phase B 调优结果)
- `--log_file` 将输出同时写入日志文件和终端
- `--data_seed` 控制数据采样 (默认 42，保持固定)
- `--seed` 控制训练随机性

Output:
- `outputs_stepX/shot_sweep_results.csv` (原始结果)
- `outputs_stepX/shot_sweep_summary.csv` (汇总统计)
- `outputs_stepX/figures/fig_shot_sweep_*.png` (可视化图表)

---

## 绘图 (Visualization)
训练和评估完成后，运行以下脚本生成论文插图：

### 1. 生成主要实验图表
```bash
python scripts/generate_paper_figures.py
```
该脚本会自动读取 `outputs/` 下的各类 CSV 结果，生成：
- `fig1_complexity_gap.png`: Accuracy-Stability Trade-off
- `fig4_search_space_colorjitter.png`: Phase A 搜索空间热力图
- `fig5_stability_boxplot.png`: Phase D 稳定性箱线图 (5 random splits)
- `fig6_cifar10_generalization.png`: 泛化实验对比
- `fig7_ablation_magnitude.png`: Magnitude 消融分析
- `fig8_destructiveness.png`: 语义保真度分析 (SSIM/LPIPS)
- `strategic_collapse.png`: Policy Selection Analysis (Figure 2)

**输出目录**: `outputs/figures/`

### 2. Policy Selection Visualization
Generate Policy Selection (Figure 2) analysis chart:
```bash
python scripts/plot_strategic_collapse.py
```
Output: `outputs/figures/strategic_collapse.png`

### 3. 增强效果可视化
```bash
python scripts/visualize_augmentations.py --policy outputs/phase_c_final_policy.json
```
生成最终策略在真实图片上的增强效果示例。

**输出目录**: `outputs/figures/augment_examples/`

### 4. Failure Cases 可视化
```bash
python scripts/visualize_failure_cases.py
```
对比 RandAugment vs SAS 的增强效果，展示语义破坏问题。

协议:
- 验证集随机抽 N=10，seed=42 固定
- 每张: 原图 → RandAugment (2次采样) → SAS (1次采样)
- 标注: 预测结果、置信度、SSIM 值
- 使用 baseline_best.pth 模型预测

Output:
- `outputs/figures/fig_failure_cases.png` (完整 10 行)
- `outputs/figures/fig_failure_cases_teaser.png` (Intro 用 3 行版)

---



### 图片数据验证

| 图片 | 验证内容 |
|------|----------|
| fig1_complexity_gap.png | 匹配 `phase_d_summary.csv` |
| fig5_stability_boxplot.png | 匹配 `phase_d_results.csv` |
| fig6_cifar10_generalization.png | 匹配 `cifar10_50shot_results.csv` |
| fig7_ablation_magnitude.png | 匹配 `ablation_p0.5_summary.csv` |
| fig8_destructiveness.png | 匹配 `destructiveness_metrics.csv` |

```bash
# 重新生成所有图片
python scripts/generate_paper_figures.py
```


## Rebuttal 准备

EUSIPCO 2026 常见质疑及回应：

| 可能质疑 | 准备的回应 |
|----------|------------|
| **"SAS 准确率比 RandAugment 低，为什么要用它？"** | 论文核心论点是 predictability，不是 accuracy。引用 utility function: $U = \mu - \lambda\sigma$，当 $\lambda > 1.28$ 时 SAS 更优。强调 one-shot training 场景。 |
| **"CV 差异 (2.77% vs 1.91%) 真的显著吗？"** | 承认 n=5 样本量小，Wilcoxon p=0.19 不显著。但强调方向一致性：Std、Width、CV 三个指标全部改善。 |
| **"为什么只用 ColorJitter？"** | 这是搜索结果，不是人为选择。Table III 展示 Phase A→B→C 的消融过程。搜索收敛到单操作体现了"奥卡姆剃刀"原则。 |
| **"SSIM/LPIPS 能解释准确率差异吗？"** | 改为"sensitivity to artifacts"表述，避免过强因果声称。这是 correlation 而非 causation。 |
| **"为什么不用预训练模型？"** | 研究场景是"从头训练"，如医学影像等领域预训练模型不适用。这是明确的 scope limitation。 |
| **"Tuned RA 只跑了 10 个配置够吗？"** | 承认预算有限，但结果已经说明问题：即使最佳配置也只有 35.30%，远低于默认 RA (42.24%)。 |
| **"~90-shot 的表述是否准确？"** | 每个 Fold 100 样本/类，90/10 分割后训练集 90/类。用 $\sim$ 表示近似值是合理的。 |

---
