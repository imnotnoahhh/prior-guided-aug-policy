# 复现指南

- 硬件/驱动：CUDA 12.8、NVIDIA Driver 570.133.20、cuDNN 9.8.0.87（4 × A10）。
- 环境：`conda env create -f environment.yml && conda activate pga`。如已创建，`conda env update -f environment.yml` 保持同步。
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

## 后台运行示例
如需断线续跑，可使用 nohup：
```bash
nohup bash scripts/train_single_gpu.sh > logs/full_run.log 2>&1 &
tail -f logs/full_run.log
```

## 绘图（训练结束后运行）
读取上述 CSV/JSON，生成：
- Phase A (m,p) 热力图：`python scripts/plot_phase_a_heatmap.py --csv outputs/phase_a_results.csv --out_dir outputs/figures`
- Phase B 稳定性：`python scripts/plot_phase_b_stability.py --csv outputs/phase_b_tuning_raw.csv --out_dir outputs/figures`
- 主表/ablation：`python scripts/plot_phase_d_ablation.py --csv outputs/phase_d_summary.csv --out_dir outputs/figures`
- 效率曲线：`python scripts/plot_efficiency.py --csv outputs/phase_b_tuning_raw.csv --out_dir outputs/figures`
- 收敛曲线（解析日志）：`python scripts/plot_convergence_from_logs.py --logs logs/baseline.log logs/phase_c.log --labels Baseline Ours --out_dir outputs/figures`
- 类别均衡：`python scripts/plot_class_balance.py --data_root ./data --fold_idx 0 --out_dir outputs/figures`
- 增强可视化：`python scripts/visualize_augmentations.py --policy outputs/phase_c_final_policy.json --out_dir outputs/figures/augment_examples`
- Phase C 历史：`python scripts/plot_phase_c_history.py --csv outputs/phase_c_history.csv --out_dir outputs/figures`
