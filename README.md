# Prior-Guided Augmentation Policy Search

官方 PyTorch 代码：在 CIFAR-100 低数据场景（每类约 100 张）下，利用先验约束的 (magnitude, probability) 联合搜索和多阶段管线找到最优增强策略。

## 环境

```bash
conda env create -f environment.yml
conda activate pga
```

## 快速验证（冒烟）

针对各阶段的轻量检查：

- Phase A：`bash scripts/smoke_test_phase_a.sh`
- Phase B：`bash scripts/smoke_test_phase_b.sh`
- Phase C：`bash scripts/smoke_test_phase_c.sh`
- Phase D：`bash scripts/smoke_test_phase_d.sh`

验证只跑 1～2 个 epoch，确保依赖、数据下载、日志写入路径都正常。

## 全流程（单 GPU）

```bash
bash scripts/train_single_gpu.sh
```

顺序运行 Baseline → Phase A → Phase B → Phase C → Phase D，所有结果保存在 `outputs/`，日志在 `logs/`。

## 关键脚本

- `run_baseline.py`：S0 基线，权重衰减 1e-2，与后续阶段一致。
- `main_phase_a.py`：40ep 低保真筛选，2D Sobol 采样 (m, p)。
- `main_phase_b.py`：ASHA 多轮淘汰，rungs = [40, 100, 200]。
- `main_phase_c.py`：多起点贪心组合，自动调整多操作概率。
- `main_phase_d.py`：5-Fold 对比实验，默认 200 epochs。

## 结构

```
├── src/                # 数据、模型、增强、训练工具
├── scripts/            # 冒烟与全流程脚本
├── main_phase_*.py     # 各阶段入口
├── docs/               # 设计与格式文档
└── outputs/            # 运行产物（自动创建）
```

## License

See [LICENSE](LICENSE) for details.
