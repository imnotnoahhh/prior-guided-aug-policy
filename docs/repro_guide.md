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
- 完整：`phase_d_summary.csv` 应含 7 methods 的 Mean ± Std。

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
# Output: Console logs
# 步骤 2: 使用最佳参数 (N=1, M=2) 跑全量验证
python scripts/run_final_tuned_ra.py
# Output: Console logs



## 后台运行示例
如需断线续跑，可使用 nohup：
```bash
nohup bash scripts/train_single_gpu.sh > logs/full_run.log 2>&1 &
tail -f logs/full_run.log
```

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
- `fig6_cifar10_generalization.png`: 泛化实验对比
- `fig7_ablation_magnitude.png`: Magnitude 消融分析
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
