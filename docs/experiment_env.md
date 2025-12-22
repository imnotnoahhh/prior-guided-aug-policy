# 实验环境记录 (Experiment Environment Log)

## 1. 服务器配置

### Phase A + Baseline

| 项目 | 配置 |
|------|------|
| **云服务商** | 阿里云 |
| **实例规格** | ecs.gn8is.2xlarge |
| **GPU** | NVIDIA L20 (48GB) |
| **vCPU** | 8 核 |
| **内存** | 64 GiB |

### 系统环境

| 项目 | 版本 |
|------|------|
| **操作系统** | Ubuntu 24.04 64-bit |
| **CUDA** | 12.8 |
| **NVIDIA Driver** | 570.113.20 |
| **Python** | 3.14.2 (conda: pga) |

---

## 2. 运行命令

### Baseline 测试

```bash
python run_baseline.py | tee logs/baseline_result.log
```

> **注意**: `run_baseline.py` 无命令行参数，使用硬编码设置：
> epochs=200, batch_size=64, lr=0.05, wd=1e-3, early_stop_patience=5, seed=42, fold_idx=0

### Phase A 筛选

```bash
# 冒烟测试 (先验证环境)
bash scripts/smoke_test_phase_a.sh

# 后台运行，保存日志
nohup python main_phase_a.py > logs/phase_a_full.log 2>&1 &

# 或前台运行（可实时查看）
python main_phase_a.py | tee logs/phase_a_full.log

# 完整参数版本 (如需自定义)
# python main_phase_a.py \
#     --epochs 200 \             # 默认值
#     --n_samples 32 \           # 默认值，每个 op 的 Sobol 采样数
#     --fold_idx 0 \             # 默认值
#     --output_dir outputs \     # 默认值
#     --seed 42 \                # 默认值
#     --num_workers 6 \          # 默认值
#     --early_stop_patience 5    # 默认值
```

---

## 3. 运行记录

| 阶段 | 日期 | 耗时 | 状态 |
|------|------|------|------|
| Phase A | 2025-12-21 | 3h 47min | ✅ 完成 |
| Baseline | 2025-12-21 | ~1min | ✅ 完成 |

---

## 4. 输出文件

| 文件 | 说明 |
|------|------|
| `outputs/phase_a_results.csv` | Phase A 筛选结果 (256 configs) |
| `outputs/baseline_result.csv` | S0 Baseline 结果 |
| `logs/phase_a_full.log` | Phase A 运行日志 |
| `logs/baseline_result.log` | Baseline 运行日志 |

---

## 5. Phase B 深度微调

### 运行命令

```bash
# 冒烟测试 (先验证环境)
bash scripts/smoke_test_phase_b.sh

# 完整运行 (后台执行，使用默认参数)
nohup python main_phase_b.py --deterministic > logs/phase_b_full.log 2>&1 &

# 或前台运行
python main_phase_b.py --deterministic | tee logs/phase_b_full.log

# 完整参数版本 (如需自定义)
# python main_phase_b.py \
#     --epochs 200 \                               # 默认值
#     --seeds 42,123,456 \                         # 默认值
#     --output_dir outputs \                       # 默认值
#     --phase_a_csv outputs/phase_a_results.csv \  # 默认值
#     --baseline_csv outputs/baseline_result.csv \ # 默认值
#     --fold_idx 0 \                               # 默认值
#     --num_workers 6 \                            # 默认值
#     --early_stop_patience 5 \                    # 默认值
#     --top_k 4 \                                  # 默认值，每个 op 取 Top-K 配置作为网格中心
#     --grid_step 0.05 \                           # 默认值，网格步长
#     --grid_n_steps 2 \                           # 默认值，每个方向的步数
#     --deterministic                              # 推荐开启，保证可复现
#
# 调试参数 (可选):
#     --ops ColorJitter,Brightness \               # 仅调优指定的 ops
#     --dry_run                                    # 仅运行少量配置用于测试
```

### 输入文件

| 文件 | 说明 |
|------|------|
| `outputs/phase_a_results.csv` | Phase A 筛选结果 (自动读取) |
| `outputs/baseline_result.csv` | Baseline 结果 (用于晋级判定) |

### 输出文件

| 文件 | 说明 |
|------|------|
| `outputs/phase_b_tuning_raw.csv` | 每个 (op, magnitude, seed) 的原始结果 |
| `outputs/phase_b_tuning_summary.csv` | 聚合结果，按 mean_val_acc 降序排列 |
| `logs/phase_b_full.log` | Phase B 运行日志 |

### 运行记录

| 阶段 | 日期 | 耗时 | 状态 |
|------|------|------|------|
| Phase B | - | - | 待运行 |
