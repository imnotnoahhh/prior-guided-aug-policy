# 复现指南

- 硬件/驱动：CUDA 12.8、NVIDIA Driver 570.133.20、cuDNN 9.8.0.87（Single NVIDIA A10）。
- 环境：`conda env create -f environment.yml && conda activate rethinking_aug`。如已创建，`conda env update -f environment.yml` 保持同步。
- 数据：自动下载 CIFAR-100 至 `./data`，若离线，请预放置官方二进制文件。

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

### 1. 破坏性分析 (Semantic Preservation)
计算 LPIPS/SSIM 指标，验证语义保真度：
```bash
python scripts/calculate_destructiveness.py
```
Output: `outputs/destructiveness_metrics.csv`

### 2. 稳定性验证 (Zero Variance Verification)
在 3 个随机种子下验证 "0 方差" 现象：
```bash
python scripts/run_stability_check.py
```
Output: `outputs/stability_seeds_results.csv`

### 3. 公平性对比 (Tuned RandAugment)
搜索最佳 RandAugment 参数 (N, M) 并进行全量验证：
```bash
# 步骤 1: 搜索最佳参数
python scripts/run_tuned_randaugment.py
# Output: outputs/tuned_randaugment_results.csv

# 步骤 2: 使用最佳参数 (N=1, M=2) 跑全量验证
python scripts/run_final_tuned_ra.py
# Output: outputs/final_tuned_ra_result.txt



## 后台运行示例
如需断线续跑，可使用 nohup：
```bash
nohup bash scripts/train_single_gpu.sh > logs/full_run.log 2>&1 &
tail -f logs/full_run.log
```

## Shot Sweep 实验 (Trend Analysis)
验证 SAS 在不同样本量下的表现趋势：

**重要**: 每个 Step 使用独立 output_dir，避免结果污染！

```bash
# Step 1: 验证 SAS 收敛 (~15-20 min)
python scripts/run_shot_sweep.py --shots 50 --folds 0 --epochs 50 \
  --methods Baseline,SAS --output_dir outputs_step1 --batch_size 128 --num_workers 8

# Step 2: 加 RandAugment 和 20-shot 判断趋势 (~1-1.5h total)
python scripts/run_shot_sweep.py --shots 50,20 --folds 0 --epochs 50 \
  --methods Baseline,RandAugment,SAS --output_dir outputs_step2 --batch_size 128 --num_workers 8

# 画图检查趋势
python scripts/plot_shot_sweep.py --output_dir outputs_step2

# Step 3: 完整版 (只有 Step 2 有意义才跑)
python scripts/run_shot_sweep.py --shots 20,50,200 --folds 0,1,2,3,4 --epochs 200 \
  --methods Baseline,RandAugment,SAS --output_dir outputs_final --batch_size 128 --num_workers 8

# 画最终图
python scripts/plot_shot_sweep.py --output_dir outputs_final
```

注意:
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
- `fig5_stability_boxplot.png`: Phase D 稳定性箱线图 (5-fold)
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
