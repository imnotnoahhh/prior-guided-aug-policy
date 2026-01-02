# Prior-Guided Augmentation: A Reliable Strategy for Small-Sample Datasets
# 先验引导增强：小样本数据集的可靠策略

[![Python 3.14](https://img.shields.io/badge/python-3.14-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![PyTorch 2.5](https://img.shields.io/badge/pytorch-2.5-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Official PyTorch Implementation** of the paper: *"Prior-Guided Augmentation: A Reliable Strategy for Small-Sample Datasets"* (WACV/BMVC Submission Target).
> 
> **官方实现**：本文提出了一种在极小样本（Low-Data Regime）下比 RandAugment 更稳定、更高效的数据增强搜索策略。

---

## 📖 Abstract / 摘要

**English**:  
Data augmentation is critical for deep learning in data-scarce regimes. While complex automated strategies like RandAugment achieve state-of-the-art results on large datasets, we reveal a **"Complexity Gap"** in small-sample settings (e.g., CIFAR-100, 100-shot): blindly increasing augmentation complexity yields diminishing returns while significantly increasing training instability. 

We propose a **Prior-Guided Augmentation** search framework that prioritizes **stability** and **semantic preservation**. Our method identifies a single, optimal operation (e.g., ColorJitter) that achieves competitive accuracy (40.74%) compared to RandAugment (42.24%) but with **significantly lower variance (Std: 0.78 vs 1.17)** and better interpretability. We further prove that "tuning" RandAugment fails in this regime, collapsing to weak augmentations (35.30%), whereas our method robustly finds the "Sweet Spot".

**中文**:  
在数据匮乏的场景下，数据增强至关重要。虽然像 RandAugment 这样的自动增强策略在大规模数据集上表现出色，但在小样本（如 CIFAR-100 每类 100 张）场景下，我们发现了一个**“复杂度陷阱 (Complexity Gap)”**：盲目增加增强操作的复杂度不仅收益递减，还会显著增加训练的不稳定性。

我们提出了一种**先验引导 (Prior-Guided)** 的增强搜索框架，该框架将“稳定性”和“语义保真度”作为核心指标。实验表明，我们搜索到的单一最优操作（如 ColorJitter）虽然简单，但能达到与 RandAugment 相当的准确率（40.74% vs 42.24%），同时**方差显著降低（Std: 0.78 vs 1.17）**。进一步的对比实验证明，在小样本下直接对 RandAugment 进行调参会失效（仅 35.30%），而我们的方法能稳健地找到最佳平衡点。

---

## 📊 Key Results / 核心结果

Experiments conducted on CIFAR-100 (100 samples/class), ResNet-18, 5-Fold Cross-Validation.

| Method | Mean Acc (%) | Stability (Std) | Complexity | Note |
| :--- | :---: | :---: | :---: | :--- |
| Baseline (S0) | 39.90 | 1.01 | Low | Basic Crop/Flip |
| **RandAugment** (N=2,M=9) | **42.24** | 1.17 | High | **Unstable** (High Variance) |
| **Tuned RandAugment** (N=1,M=2)| 35.30 | N/A | Low | Tuning fails (Underfitting) |
| **Ours (Optimal)** | 40.74 | **0.78** | **Low** | **Most Stable & Reliable** |

### Why Ours? / 为什么选择我们的方法？
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

# Verify Strategic Collapse (Figure 2)
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
  title={Prior-Guided Augmentation: A Reliable Strategy for Small-Sample Datasets},
  author={Qin, Fuyao},
  journal={arXiv preprint},
  year={2026}
}
```
-->

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
