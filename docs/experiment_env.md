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

### Phase A 筛选

```bash
# 后台运行，保存日志
nohup python main_phase_a.py > logs/phase_a_full.log 2>&1 &

# 或前台运行（可实时查看）
python main_phase_a.py | tee logs/phase_a_full.log
```

---

## 3. 运行记录

| 阶段 | 日期 | 耗时 | 状态 |
|------|------|------|------|
| Phase A | 2024-12-21 | 3h 47min | ✅ 完成 |
| Baseline | 2024-12-21 | ~1min | ✅ 完成 |

---

## 4. 输出文件

| 文件 | 说明 |
|------|------|
| `outputs/phase_a_results.csv` | Phase A 筛选结果 (256 configs) |
| `outputs/baseline_result.csv` | S0 Baseline 结果 |
| `logs/phase_a_full.log` | Phase A 运行日志 |
| `logs/baseline_result.log` | Baseline 运行日志 |
