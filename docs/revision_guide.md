# ICIP 2026 论文修改指南 (综合版)

> **论文标题**: When More is Not Better: Rethinking Data Augmentation under Small-Sample Regimes  
> **建议新标题**: Stability over Complexity: Rethinking Data Augmentation for Small-Sample Learning  
> **方法命名**: SAS (Stability-aware Augmentation Search)  
> **目标会议**: IEEE ICIP 2026 (2026年9月13-17日，芬兰坦佩雷)  
> **投稿截止**: 2026年2月4日 (Anywhere on Earth)  
> **录用通知**: 2026年4月22日

---

## 📋 修改优先级总览 (重新排序版)

| 优先级 | 阶段 | 内容 | 是否需要实验 | 建议时间 |
|--------|------|------|--------------|----------|
| ✅ P0 | 已完成 | 匿名化 + 红旗问题文字说明 | ❌ | 已完成 |
| 🔴 P1 | 写作任务 | Abstract/Intro/贡献点/SAS命名/伪代码 | ❌ | Day 1-2 |
| 🟠 P2 | 数据分析 | 表格升级 + 统计检验 (现有数据) | ❌ | Day 2-3 |
| 🟡 P3 | 核心实验 | Shot Sweep + 搜索消融 | ✅ 必做 | Day 4-7 |
| 🟢 P4 | 增强实验 | 换Backbone + Seed方差 + 语义指标 | ✅ 建议 | Day 8-10 |
| 🔵 P5 | 可选实验 | 可视化 + 效率 + ViT + 真实数据集 | ✅ 可选 | Day 11-12 |
| ⚪ P6 | 提交检查 | PDF合规 + 最终校对 | ❌ | Day 13 |

---

## ✅ P0: 已完成项 (确认状态)

### 1. 双盲匿名化 ✅

| 检查项 | 状态 |
|--------|------|
| 删除作者信息 | ✅ `main.tex` Line 24-25 已为匿名占位符 |
| 删除 GitHub 链接 | ✅ 已改为 "Code will be made publicly available upon acceptance." |
| 自引处理 | ✅ `references.bib` 中无自引 |
| PDF 元数据清理 | ✅ 已添加 `hyperref` 包 |

### 2. 复现性底线 ✅

训练配置已完整包含于论文中 (5-fold CV, 200 epochs, SGD, etc.)

### 3. 红旗问题 (文字部分) ✅

| 问题 | 状态 |
|------|------|
| RandAugment 35.30% 异常 | ✅ 已在论文中说明搜索细节和过拟合原因 |
| K=8 算子列表 + 参数映射 | ✅ 已在 Section 3.1 和 Appendix A 添加 |
| 目标函数 α=1.0 定义 | ✅ 已在 Phase C 添加公式 |
| 复杂度 C 定义 | ✅ 已在 Section 3.1 添加 |
| CIFAR-10 50% 零方差 | ✅ 论文已有 3-seed 验证说明 |
| 评估协议选择偏差 | ✅ 已在 Limitations 添加说明 |

---

## 🔴 P1: 写作任务 (不需要实验) - Day 1-2

> 💡 **这些任务可以立即开始，不依赖任何实验结果**

### 1.1 方法命名 SAS ✅

**任务**: 给方法起正式名字，全文统一使用

**名称**: **SAS** (Stability-aware Augmentation Search)

**已完成修改**:
- [x] Abstract: "We propose **SAS** (Stability-aware Augmentation Search)..."
- [x] Introduction: "we propose \textbf{SAS}..."
- [x] 全文替换 "Single-Op" / "single-operation policy" → "SAS"
- [x] 表格 Table 1 和 CIFAR-10 表格

### 1.2 Abstract 重写 ✅

**已更新** (`main.tex` Line 34-36):

```latex
\begin{abstract}
Complex data augmentation strategies introduce significant training variance 
in small-sample regimes, undermining model reliability---a critical concern 
in domains like medical imaging where ``lucky seeds'' cannot be relied upon. 
This paper challenges the prevailing ``more is better'' assumption by 
systematically studying CIFAR-100 with only 100 samples per class. 
We observe a clear trade-off: while RandAugment achieves marginally higher 
mean accuracy (+1.5\%), it incurs 50\% higher fold variance. 
We propose \textbf{SAS} (Stability-aware Augmentation Search), a three-phase 
protocol that explicitly penalizes variance. SAS identifies a single, 
well-tuned operation (ColorJitter) that achieves competitive performance 
(40.74\% vs.\ 42.24\%) while reducing variance by 33\%. 
Through systematic evaluation across multiple folds and seeds, we demonstrate 
that in data-scarce scenarios, \textbf{stability should take precedence over complexity}.
\end{abstract}
```

### 1.3 Introduction 三条贡献 ✅

**已添加** (在 Section 1 末尾，`main.tex` Line 64-70):

```latex
Our contributions are threefold:
\begin{itemize}
    \item \textbf{Empirical Insight:} We reveal a stability-accuracy trade-off 
    in small-sample augmentation, showing that complex policies introduce 
    high variance that offsets their marginal accuracy gains (Section 4).
    \item \textbf{Methodology:} We propose SAS (Stability-aware Augmentation Search), 
    a three-phase protocol that explicitly penalizes variance using a 
    lower-bound criterion (Mean $-$ Std) for robust policy selection (Section 3).
    \item \textbf{Validation:} Through 5-fold cross-validation, multi-seed evaluation, 
    and semantic preservation analysis (SSIM/LPIPS), we provide systematic evidence 
    that single-operation policies offer the best reliability in data-scarce regimes 
    (Section 4, Appendix).
\end{itemize}
```

> 📝 **注**: 原建议中的 "shot-sweep experiments" 和 "multi-backbone evaluation" 已简化为当前实验状态。如完成 Shot Sweep 实验后可更新。

### 1.4 完善伪代码/算法框图 ✅

**已完成**: 将 Algorithm 1 从 "Greedy Selection (Phase C only)" 升级为完整的三阶段 SAS 算法

**更新内容** (`main.tex` Line 114-139):
- 算法标题改为 "SAS: Stability-aware Augmentation Search"
- 包含完整的 Phase A (Screening) / Phase B (Tuning) / Phase C (Composition)
- 明确展示 $\alpha$ 方差惩罚准则
- 注释说明 "Typically converges to single operation"

**引用更新**:
- Section 3.2 中的 `Procedure \ref{alg:phase_c}` → `Algorithm \ref{alg:sas}, Lines 17-23`

### 1.5 相关工作补充

**补充内容**:
1. **与 AutoAugment/RandAugment 的本质区别**
2. **Data-Efficient Learning (2024-2025)** 最新文献
3. **Augmentation Stability** 相关研究

### 1.6 Limitations 与 Future Work 扩展 ✅

**已完成** (`main.tex` Line 202-213):

- 将原 `\section{Limitations}` 改为 `\section{Limitations and Future Work}`
- 整合评估协议说明到 Limitations 段落
- 新增 Future Work 三条方向:
  1. Vision Transformers + self-supervised learning
  2. 真实世界小样本领域 (医学影像、卫星图像)
  3. 嵌套交叉验证协议

### 1.7 全文一致性检查 ✅

- [x] **术语统一**: 已将 `low-shot`, `low-data` → `small-sample` (3处)
- [x] **方法名称统一**: 已完成 "Single-Op" → "SAS" (W1)
- [x] **Std 含义统一**: Line 178 已明确 "Fold Variance (sensitivity to data splits)"
- [x] **图表编号与引用**: 已检查，6处引用全部匹配对应 label
- [ ] **图例字体**: 需手动检查 PNG 文件 (编译后确认 ≥ 9pt)

### 1.8 Figure 1 标注强化 ✅ (代码已更新)

**已修改** `scripts/generate_paper_figures.py` 的 `plot_complexity_tradeoff()` 函数:

- [x] 添加箭头从 RandAugment 指向 SAS
- [x] 添加 **"33% variance reduction"** 标注框
- [x] 更新标签 "Single-Op (Ours)" → "SAS (Ours)"
- [x] 字体已设置 fontsize=10-12 (≥ 9pt)

**运行命令重新生成图片**:
```bash
python scripts/generate_paper_figures.py
```

生成的图片: `outputs/figures/fig1_complexity_gap.png`

### 1.9 标题修改 (可选)

**建议新标题**: 
- **Stability over Complexity: Rethinking Data Augmentation for Small-Sample Learning**

---

## 🟠 P2: 数据分析任务 (基于现有数据) - Day 2-3

> 💡 **这些任务不需要跑新实验，只需分析现有数据**

### 2.1 Table 1 升级

从现有 5-fold 结果中提取额外指标：

| Policy | Val Acc % | Std Dev | **Min Acc** | **Lower Bound** | **95% CI** |
|--------|-----------|---------|-------------|-----------------|------------|
| Baseline | 39.90 | 1.01 | **待计算** | 38.89 | **待计算** |
| RandAugment | 42.24 | 1.17 | **待计算** | 41.07 | **待计算** |
| SAS | 40.74 | 0.78 | **待计算** | 39.96 | **待计算** |

**计算公式**:
- Min Acc: 5 个 fold 中的最低分
- Lower Bound: Mean - Std
- 95% CI: Mean ± 1.96 × Std/√5

**数据来源**: `outputs/phase_d_results.csv` 或 `outputs/baseline_result.csv`

### 2.2 统计显著性检验

**任务**: 对现有 5-fold 数据做 t-test 和方差检验

```python
from scipy import stats

# 假设有 5 个 fold 的准确率
sas_accs = [...]  # 从 CSV 读取
ra_accs = [...]

# 均值差异检验
t_stat, p_value = stats.ttest_rel(sas_accs, ra_accs)

# 方差差异检验 (Levene's test)
levene_stat, levene_p = stats.levene(sas_accs, ra_accs)
```

**输出**: 在论文中报告 p-value

```latex
We performed paired t-tests comparing SAS against RandAugment 
across 5 folds. While RandAugment achieves higher mean accuracy 
($p = 0.XX$), SAS exhibits significantly lower variance 
(Levene's test, $p < 0.05$).
```

---

## 🟡 P3: 核心实验 (必做) - Day 4-7

> ⚠️ **这两个实验对论文增强效果最大，优先完成**

### 3.1 Shot Sweep 实验 ⭐⭐⭐ (最高优先级)

**增强效果**: 极高 - 将单点实验升级为趋势规律，直接支撑核心论点

**设置**:
- 数据集: CIFAR-100
- Shot数: `[10, 20, 50, 100, 200]` samples/class
- 方法: Baseline, RandAugment, SAS
- 评估: 5-fold CV

**输出物** (3条曲线):
1. **Accuracy vs Shot**: 各方法性能随样本数变化
2. **Fold Std vs Shot**: 方差随样本数变化 (核心！)
3. **Lower Bound vs Shot**: 最坏情况性能

**预期故事**: 
> 随着样本减少，RandAugment 方差剧增，而 SAS 保持稳定。展示"拐点"位置。

**预计时间**: 2-3 天 (5 shot × 3 方法 × 5 fold × 200 epochs)

### 3.2 搜索流程消融 ⭐⭐ (高优先级)

**增强效果**: 高 - 防守"运气选到 ColorJitter"的质疑，证明方法论必要性

**三个版本对比**:
1. **Phase A only**: 仅 Sobol 筛选
2. **Phase A + B**: 筛选 + ASHA 调优
3. **Full SAS**: 筛选 + 调优 + Phase C 稳定性约束

**输出物**:

| Method | Mean Acc % | Std Dev | Lower Bound | Selected Op |
|--------|------------|---------|-------------|-------------|
| Phase A only | - | - | - | - |
| Phase A + B | - | - | - | - |
| Full SAS | 40.74 | 0.78 | 39.96 | ColorJitter |

**预计时间**: 1 天 (复用现有代码，只需分阶段跑)

---

## 🟢 P4: 增强实验 (建议做) - Day 8-10

> 💡 **这些实验能进一步增强说服力，按优先级排序**

### 4.1 更换 Backbone ⭐⭐ (高优先级)

**增强效果**: 高 - 证明结论泛化性，不仅限于 ResNet-18

**设置**:
- 数据: CIFAR-100, 100-shot
- 模型: ResNet-34 或 WideResNet-28-10 (选1个)
- 方法: Baseline, RandAugment, SAS

**输出物**:

| Backbone | Method | Mean Acc % | Std Dev | Lower Bound |
|----------|--------|------------|---------|-------------|
| ResNet-18 | SAS | 40.74 | 0.78 | 39.96 |
| WRN-28-10 | SAS | - | - | - |

**预计时间**: 1 天

### 4.2 Seed 方差报告 ⭐ (中高优先级)

**增强效果**: 中高 - 补充 fold 方差之外的另一维度

**设置**:
- 在 CIFAR-100 100-shot 主实验上
- 同一 Fold 0，使用 5 个不同随机种子
- 报告 Seed 方差

**输出物**:

| Method | Fold Std | Seed Std |
|--------|----------|----------|
| Baseline | 1.01 | - |
| RandAugment | 1.17 | - |
| SAS | 0.78 | - |

**预计时间**: 半天

### 4.3 语义保持硬指标 (Label Consistency) ⭐ (中优先级)

**增强效果**: 中 - 补充 SSIM/LPIPS 之外更"硬"的指标

**设置**:
```python
# 使用 ImageNet 预训练的 ResNet-50
pretrained_model = torchvision.models.resnet50(pretrained=True)

for img in validation_set:
    pred_original = pretrained_model(img)
    pred_augmented = pretrained_model(augment(img))
    consistency = (pred_original.argmax() == pred_augmented.argmax())
```

**输出物**:

| Method | SSIM ↑ | LPIPS ↓ | Label Consistency ↑ |
|--------|--------|---------|---------------------|
| Baseline | 0.198 | 0.084 | - |
| RandAugment | 0.147 | 0.124 | - |
| SAS | 0.196 | 0.091 | - |

**预计时间**: 2-3 小时

---

## 🔵 P5: 可选实验 (时间充裕时) - Day 11-12

> 📝 **按增强效果排序，优先考虑前几个**

### 5.1 Failure Cases 可视化 (中优先级)

**增强效果**: 中 - 直观展示问题，工作量小

**协议**:
1. 从验证集随机抽取 N=10 张图片 (seed=42)
2. 展示: 原图 / RandAugment 处理后 / SAS 处理后
3. 标注: 预测结果、置信度、SSIM

**预计时间**: 2-3 小时

### 5.2 训练效率对比 (低优先级)

**增强效果**: 低 - 非核心论点

**指标**: Time/Epoch, Throughput (img/s)

**预计时间**: 1 小时

### 5.3 RandAugment 局部扫描曲线 (低优先级)

**增强效果**: 低 - 已有文字说明，曲线是锦上添花

**设计**:
- 实验1: 固定 N=2，扫描 M = [1, 2, ..., 14]
- 实验2: 固定 M=9，扫描 N = [1, 2, 3]

**预计时间**: 2-3 小时

### 5.4 嵌套式交叉验证 (低优先级)

**增强效果**: 中 - 但已有文字说明，工作量大

**设计**:
```
外层: 5-fold 仅用于最终报告
  └── 内层: 每个外层训练折内部再划分
```

**预计时间**: 1-2 天

### 5.5 ViT 实验 (低优先级)

**增强效果**: 中 - 可放 Future Work

**设置**: 小型 ViT (如 ViT-Tiny) 在 CIFAR-100 100-shot

**预计时间**: 1 天

### 5.6 真实数据集 (最低优先级)

**增强效果**: 高 - 但工作量极大

**建议**: 
- 时间不足时，在 Future Work 中明确提及
- "Validating on real-world domains (medical imaging, satellite imagery)"

---

## ⚪ P6: 提交检查 - Day 13

### 格式合规检查 (ICIP 2026 硬性要求)

| 要求 | 规格 | 检查 |
|------|------|------|
| 技术内容 | ≤ 5页 | [ ] |
| 第6页 | 仅参考文献 | [ ] |
| 纸张尺寸 | US Letter (8.5" × 11") | [ ] |
| 文本区域 | 178mm × 229mm (7" × 9") | [ ] |
| 左边距 | 19mm (0.75") | [ ] |
| 上边距 | 25mm (首页35mm) | [ ] |
| 双栏宽度 | 每栏86mm，间距6mm | [ ] |
| 字体大小 | ≥ 9pt (全文包括图注) | [ ] |
| 字体类型 | Times-Roman 或 Computer Modern | [ ] |
| 页码 | **不要添加页码** | [ ] |

### PDF eXpress 验证

提交前**必须**通过 IEEE PDF eXpress 验证：
- 网址: https://ieee-pdf-express.org/account/login
- **Conference ID: 61757X**
- 创建账户 → 上传 PDF → 修复问题 → 获得合规版本

### PDF 技术检查

1. [ ] 使用 ICIP 2026 官方模板重新编译
2. [ ] 通过 PDF eXpress 验证
3. [ ] 检查字体嵌入:
   ```bash
   pdffonts your_paper.pdf
   # 确保所有字体都显示 "yes" 在 emb 列
   ```
4. [ ] 检查无页码
5. [ ] 检查图表文字 ≥ 9pt
6. [ ] 检查图表分辨率足够清晰

### 双盲版检查

- [ ] 作者栏为空或显示 "Anonymous Authors"
- [ ] 无GitHub链接、无个人主页
- [ ] 无 "我们之前的工作" 表述
- [ ] 自引 ≤ 2篇
- [ ] PDF元数据已清理

### 文件名检查

- [ ] 文件名不含个人信息
- [ ] 建议命名: `ICIP2026_submission.pdf` 或 `paper_blind.pdf`

### 提交材料清单

| 材料 | 格式 | 状态 |
|------|------|------|
| 匿名版论文 | PDF (≤200MB) | [ ] |
| 发布版论文 | PDF (PDF eXpress验证) | [ ] |
| 补充材料 (可选) | 匿名形式 | [ ] |

### 最终通读

重点检查:
- [ ] 贡献点是否一眼能读懂
- [ ] 审稿人最可能质疑的点是否都有对应图表或段落回应
- [ ] 拼写和语法错误
- [ ] 图表引用正确性
- [ ] 方法名称 SAS 是否一致使用

### 图片数据验证

> ⚠️ **重要**: 确保所有图片中的数据与实验结果一致

**验证清单**:

| 图片 | 验证内容 | 状态 |
|------|----------|------|
| fig1_complexity_gap.png | Baseline/SAS/RandAugment 的 Acc 和 σ 值是否匹配 `phase_d_summary.csv` | [ ] |
| fig4_search_space_colorjitter.png | 红圈标记的 Optimal 点是否是 ColorJitter 最高 val_acc | [ ] |
| fig5_stability_boxplot.png | 5 个 fold 的黑点 Y 值是否匹配 `phase_d_results.csv` | [ ] |
| fig6_cifar10_generalization.png | Mean/Std 是否匹配 `cifar10_50shot_results.csv` | [ ] |
| fig7_ablation_magnitude.png | 曲线数据是否匹配 `ablation_p0.5_summary.csv` | [ ] |
| fig8_destructiveness.png | SSIM/LPIPS 值是否匹配 `destructiveness_metrics.csv` | [ ] |

**验证命令**:
```bash
# 重新生成所有图片
python scripts/generate_paper_figures.py

# 对比数据文件
cat outputs/phase_d_summary.csv
cat outputs/cifar10_50shot_results.csv
```

---

## 📝 补充说明

### 1. ORCID 要求 (ICIP 2026 新要求)

> **所有作者必须提供 ORCID**，否则无法提交。请提前收集所有作者的 ORCID。

- 注册网址: https://orcid.org/register
- 这是 ICIP 2026 的硬性要求，无 ORCID 将被拒绝提交

### 2. Rebuttal 准备

ICIP 2026 有 Rebuttal 环节。建议提前准备以下问题的回应：

| 可能质疑 | 准备的回应 |
|----------|------------|
| "为什么不用更大的数据集验证?" | 小样本场景的实际应用背景 (医学影像等) |
| "ColorJitter 是否只是巧合?" | 搜索消融实验结果 |
| "为什么不考虑预训练模型?" | 从头训练的场景需求 (域不匹配时) |
| "Single-Op 准确率低于 RandAugment?" | Lower Bound 指标、稳定性价值 |
| "评估协议是否有选择偏差?" | 解释当前协议 + 承认局限性 |
| "RandAugment 35.30% 怎么回事?" | 局部扫描曲线 + 验证集过拟合解释 |

### 3. Supplementary Material 建议

ICIP 2026 允许提交匿名的补充材料，建议包含：

1. **完整实验日志**: 所有配置和结果
2. **K=8算子完整列表**: 参数化细节
3. **RandAugment局部扫描曲线**: 证明35.30%不是bug
4. **更多可视化**: Failure cases 完整版
5. **CIFAR-10每折原始值**: 解释50%零方差
6. **代码片段**: 关键实现 (匿名化)

### 4. arXiv 预印本注意

> **重要**: 在审稿结果公布前，**不得**将论文上传至 arXiv。
> 只有在收到录用通知后，才可上传预印本。

### 5. No-Show Policy

> 被录用的论文**必须**由作者之一现场报告，否则将从 IEEE Xplore 撤稿。

### 6. 🆕 真实数据集建议 (来自意见1)

**意见1强烈建议**: 补充真实世界数据集实验 (如 ISIC 皮肤病变)

**评估**:
- 学术上理想，但时间可能不够
- 如果无法完成，在 Future Work 中明确提及

**备选方案**:
- 在 Limitations 中承认只在 CIFAR 上验证
- 在 Future Work 中列出 "Validating on real-world domains"

---

## 📅 时间线建议 (重新排序版)

| 日期 | 阶段 | 任务 | 是否实验 | 产出物 |
|------|------|------|----------|--------|
| **Day 1** | P1 写作 | SAS命名 + Abstract重写 + 贡献点 | ❌ | 更新的论文 |
| **Day 2** | P1 写作 | 伪代码 + 相关工作 + Future Work | ❌ | 更新的论文 |
| **Day 2-3** | P2 分析 | Table 1 升级 + 统计检验 | ❌ | 表格 + p-value |
| **Day 4-5** | P3 核心 | Shot Sweep 实验 | ✅ | 3条曲线 |
| **Day 6-7** | P3 核心 | 搜索消融实验 | ✅ | 消融表 |
| **Day 8** | P4 增强 | 换 Backbone (WRN-28-10) | ✅ | 对比表 |
| **Day 9** | P4 增强 | Seed 方差 + 语义指标 | ✅ | 补充表格 |
| **Day 10-11** | P5 可选 | 可视化 / 效率 / 其他 | ✅ | 按需 |
| **Day 12** | 整合 | 全文一致性检查 + Figure 强化 | ❌ | 最终论文 |
| **Day 13** | P6 提交 | PDF eXpress + 最终校对 | ❌ | 提交版 PDF |

---

## 🔗 重要链接

- **ICIP 2026 主页**: https://2026.ieeeicip.org/
- **Author Kit**: https://2026.ieeeicip.org/author-kit/
- **重要日期**: https://2026.ieeeicip.org/important-dates/
- **PDF eXpress**: https://ieee-pdf-express.org/account/login (Conference ID: **61757X**)
- **投稿系统**: https://icip2026.exordo.com

---

## 🧪 可选实验 (时间充裕时考虑)

> 📝 以下实验为**可选项**，视时间和效果决定是否执行。当前已通过文字说明应对审稿人质疑。

### 可选实验 A: 嵌套式交叉验证

**目的**: 彻底消除评估协议选择偏差的质疑

**设计**:
```
外层: 5-fold 仅用于最终报告
  └── 内层: 每个外层训练折内部再划分 (如 80/20)
        ├── 内层 80%: 训练
        ├── 内层 20%: Phase A/B/C 搜索与早停
        └── 外层测试折: 仅评估一次，不参与搜索
```

**预计时间**: 1-2 天

**决策标准**:
- [ ] 时间是否充裕
- [ ] 是否值得增加论文复杂度

**如果结果一致**: 在论文中加一句 "We further validated with nested CV and obtained consistent results."

---

### 可选实验 B: RandAugment 局部扫描曲线

**目的**: 用曲线图证明 35.30% 确实是搜索最优，而非偶然或 bug

**设计**:
```
实验1: 固定 N=2，扫描 M = [1, 2, 3, ..., 14]
实验2: 固定 M=9，扫描 N = [1, 2, 3]
```

**预计时间**: 2-3 小时 (每个配置跑 40 epochs)

**输出物**: 一张曲线图

**图注示例**:
> "RandAugment hyperparameter sensitivity on CIFAR-100 (100-shot). 
> Left: Accuracy vs. Magnitude M (fixed N=2). Right: Accuracy vs. N (fixed M=9). 
> Results confirm that naive hyperparameter search leads to suboptimal configurations."

**脚本**: 需要时可以写一个 `scripts/run_ra_local_scan.py`

---

## 📊 修改清单汇总 (重新排序版)

### ✅ 已完成

| 编号 | 任务 | 状态 |
|------|------|------|
| C1 | 双盲匿名化 | ✅ |
| C2 | RandAugment 35.30% 文字说明 | ✅ |
| C3 | K=8 算子完整列表 | ✅ |
| C4 | 目标函数显式定义 (α=1.0) | ✅ |
| C5 | 复杂度 C 公式 | ✅ |
| C6 | CIFAR-10 50% 零方差解释 | ✅ |
| C7 | 评估协议说明/局限性承认 | ✅ |

### 🔴 P1: 写作任务 (不需要实验)

| 编号 | 任务 | 优先级 | 状态 |
|------|------|--------|------|
| W1 | 方法命名 SAS + 全文替换 | ⭐⭐⭐ | [x] ✅ |
| W2 | Abstract 重写 | ⭐⭐⭐ | [x] ✅ |
| W3 | Introduction 三条贡献 | ⭐⭐⭐ | [x] ✅ |
| W4 | 完善伪代码/算法框图 | ⭐⭐ | [x] ✅ |
| W5 | 相关工作补充 (2024-2025) | ⭐⭐ | [ ] |
| W6 | Limitations + Future Work 扩展 | ⭐⭐ | [x] ✅ |
| W7 | 全文一致性检查 | ⭐ | [x] ✅ |
| W8 | Figure 1 标注强化 | ⭐ | [x] ✅ 代码 |
| W9 | 标题修改 (可选) | ⭐ | [ ] |

### 🟠 P2: 数据分析任务 (基于现有数据)

| 编号 | 任务 | 优先级 | 状态 |
|------|------|--------|------|
| A1 | Table 1 升级 (Min Acc, Lower Bound, 95% CI) | ⭐⭐⭐ | [ ] |
| A2 | 统计显著性检验 (t-test, Levene's test) | ⭐⭐ | [ ] |

### 🟡 P3: 核心实验 (必做)

| 编号 | 任务 | 增强效果 | 优先级 | 状态 |
|------|------|----------|--------|------|
| E1 | Shot Sweep 实验 | 极高 - 核心论点支撑 | ⭐⭐⭐ | [ ] |
| E2 | 搜索流程消融 | 高 - 证明方法论必要性 | ⭐⭐⭐ | [ ] |

### 🟢 P4: 增强实验 (建议做)

| 编号 | 任务 | 增强效果 | 优先级 | 状态 |
|------|------|----------|--------|------|
| E3 | 换 Backbone (WRN-28-10) | 高 - 证明泛化性 | ⭐⭐ | [ ] |
| E4 | Seed 方差报告 | 中高 - 补充可信度 | ⭐⭐ | [ ] |
| E5 | 语义保持硬指标 (Label Consistency) | 中 - 补充评估维度 | ⭐ | [ ] |

### 🔵 P5: 可选实验 (时间充裕时)

| 编号 | 任务 | 增强效果 | 优先级 | 状态 |
|------|------|----------|--------|------|
| E6 | Failure Cases 可视化 | 中 - 直观展示 | ⭐ | [ ] |
| E7 | 训练效率对比 | 低 - 非核心 | ⭐ | [ ] |
| E8 | RandAugment 局部扫描曲线 | 低 - 已有文字 | ⭐ | [ ] |
| E9 | 嵌套式交叉验证 | 中 - 工作量大 | ⭐ | [ ] |
| E10 | ViT 实验 | 中 - 可放 Future Work | ⭐ | [ ] |
| E11 | 真实数据集 (ISIC等) | 高 - 工作量极大 | ⭐ | [ ] |

### ⚪ P6: 提交检查

| 编号 | 任务 | 状态 |
|------|------|------|
| S1 | 格式合规 (5页+参考文献) | [ ] |
| S2 | PDF eXpress 验证 | [ ] |
| S3 | 字体嵌入检查 | [ ] |
| S4 | 页码移除确认 | [ ] |
| S5 | 最终通读 | [ ] |
| S6 | **图片数据验证** (6张图 vs CSV) | [ ] |

---

*最后更新: 2026-01-23 (按新优先级重排)*
