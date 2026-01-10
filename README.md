# When More is Not Better: Rethinking Data Augmentation under Small-Sample Regimes
# 小样本场景下的增强策略再思考：多未必更好

[![Python 3.14](https://img.shields.io/badge/python-3.14-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![PyTorch 2.5](https://img.shields.io/badge/pytorch-2.5-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Official PyTorch Implementation** of the paper: *"When More is Not Better: Rethinking Data Augmentation under Small-Sample Regimes"* (WACV/BMVC Submission Target).
> 
> **Official Implementation**: This study reveals the counter-intuitive finding that increasing augmentation complexity often yields diminishing returns in small-sample regimes.

---

## 📖 Abstract / 摘要

**English**:  
Data augmentation is critical for deep learning in data-scarce regimes. While complex automated strategies like RandAugment achieve state-of-the-art results on large datasets, their efficacy in small-sample settings (e.g., CIFAR-100, 100-shot) remains under-explored. We find that blindly increasing augmentation complexity yields diminishing returns while significantly increasing training instability. 

Through a multi-phase search protocol combining Sobol sampling and ASHA scheduling, we discover that a single, well-tuned operation (ColorJitter) can achieve competitive accuracy (40.74%) compared to RandAugment (42.24%) but with **significantly lower variance (Std: 0.78 vs 1.17)**. Our findings suggest that in small-sample regimes, augmentation design should prioritize stability over complexity.

**中文**:  
在数据匮乏的场景下，数据增强至关重要。虽然像 RandAugment 这样的自动增强策略在大规模数据集上表现出色，但在小样本（如 CIFAR-100 每类 100 张）场景下，盲目增加增强操作的复杂度不仅收益递减，还会显著增加训练的不稳定性。

通过一套结合 Sobol 采样和 ASHA 调度的多阶段搜索流程，我们发现单一的、调优良好的操作（如 ColorJitter）能达到与 RandAugment 相当的准确率（40.74% vs 42.24%），同时**方差显著降低（Std: 0.78 vs 1.17）**。我们的发现表明，在小样本场景下，增强策略设计应优先考虑稳定性而非复杂度。

---

## 📊 Key Results / 核心结果

Experiments conducted on CIFAR-100 (100 samples/class), ResNet-18, 5-Fold Cross-Validation.

| Method | Mean Acc (%) | Stability (Std) | Complexity | Note |
| :--- | :---: | :---: | :---: | :--- |
| Baseline (S0) | 39.90 | 1.01 | Low | Basic Crop/Flip |
| **RandAugment** (N=2,M=9) | **42.24** | 1.17 | High | **Unstable** (High Variance) |
| **Tuned RandAugment** (N=1,M=2)| 35.30 | N/A | Low | Tuning fails (Underfitting) |
| **Single-Op (ColorJitter)** | 40.74 | **0.78** | **Low** | **Most Stable and Reliable** |

### Why Single-Op? / Why single-operation policy?
1.  **Zero Variance in Stability Check**: Verified to converge consistently (50.00% ± 0.00%) across 3 random seeds in 50-shot scenarios.
2.  **High Semantic Fidelity**: LPIPS score (0.091) is comparable to baseline, unlike RandAugment (0.124) which distorts images heavily.
3.  **Efficiency**: Search cost is only ~4 GPU hours, finding the optimal policy without expensive reinforcement learning.

---

## 🛠 Installation / 安装

```bash
# Clone the repository
git clone https://github.com/yourusername/prior-guided-aug.git
cd prior-guided-aug

# Create Conda environment
conda env create -f environment.yml
conda activate pga

# (Optional) Verify installation
python -c "import torch; print(torch.cuda.is_available())"
```

---

## 🚀 Quick Start / 快速开始

### 1. Reproduction (One-Click) / 一键复现
Run the full pipeline (Phase A -> B -> C -> D) on a single GPU:

```bash
bash scripts/train_single_gpu.sh
```

### 2. Supplementary Experiments / 补充实验 (Paper Revision)
Reproduce the specific proofs for stability and fairness:

```bash
# Verify Semantic Preservation (Destructiveness)
python scripts/calculate_destructiveness.py

# Verify Zero Variance (Stability)
python scripts/run_stability_check.py

# Verify Tuned RandAugment Failure (Fairness)
python scripts/run_tuning_randaugment.py  # Search
python scripts/run_final_tuned_ra.py      # Validation

# Verify Policy Selection (Figure 2)
python scripts/plot_strategic_collapse.py


```

### 3. Visualization / 绘图
Generate all figures used in the paper:

```bash
python scripts/generate_paper_figures.py
```
Output: `outputs/figures/`

---

## 📂 Project Structure / 项目结构

```
.
├── src/                # Core implementation
│   ├── augmentations.py  # Search space & Augmentation logic
│   ├── dataset.py        # CIFAR-100 Subsampled dataset
│   └── models.py         # ResNet-18
├── scripts/            # Experiment scripts
│   ├── train_single_gpu.sh      # Full pipeline runner
│   ├── calculate_destructiveness.py # LPIPS/SSIM analysis
│   └── generate_paper_figures.py    # Plotting tools
├── docs/               # Documentation
│   ├── paper_draft.tex   # LaTeX draft
│   └── repro_guide.md    # Detailed guide
├── outputs/            # Experiment results (Auto-generated)
└── logs/               # Training logs
```

---

## 📜 Citation / 引用

If you find this work useful, please stay tuned! The citation will be updated upon acceptance.

<!--
```bibtex
@article{qin2026prior,
  title={When More is Not Better: Rethinking Data Augmentation under Small-Sample Regimes},
  author={Qin, Fuyao},
  journal={arXiv preprint},
  year={2026}
}
```
-->

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
