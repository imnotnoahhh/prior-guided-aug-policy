"""
P2 数据分析任务: Table 1 升级 + 统计显著性检验

输出:
1. Min Acc, Lower Bound, 95% CI 计算结果
2. t-test 和 Levene's test p-values
"""

import pandas as pd
import numpy as np
from scipy import stats

def main():
    # 读取数据
    df = pd.read_csv('outputs/phase_d_results.csv')
    
    # 重命名列（如果需要）
    if 'op_name' in df.columns:
        df.rename(columns={'op_name': 'method'}, inplace=True)
    
    # 筛选三个方法
    methods = ['Baseline', 'RandAugment', 'Ours_optimal']
    df = df[df['method'].isin(methods)]
    
    print("=" * 60)
    print("P2.1: Table 1 升级 - 额外指标计算")
    print("=" * 60)
    
    results = []
    for method in methods:
        accs = df[df['method'] == method]['val_acc'].values
        
        mean_acc = np.mean(accs)
        std_acc = np.std(accs, ddof=1)  # 样本标准差
        min_acc = np.min(accs)
        lower_bound = mean_acc - std_acc
        
        # 95% CI: Mean ± 1.96 × SE, where SE = Std / sqrt(n)
        se = std_acc / np.sqrt(len(accs))
        ci_low = mean_acc - 1.96 * se
        ci_high = mean_acc + 1.96 * se
        
        display_name = 'SAS' if method == 'Ours_optimal' else method
        
        results.append({
            'Method': display_name,
            'Mean': mean_acc,
            'Std': std_acc,
            'Min Acc': min_acc,
            'Lower Bound': lower_bound,
            '95% CI Low': ci_low,
            '95% CI High': ci_high,
            'Fold Values': accs.tolist()
        })
        
        print(f"\n{display_name}:")
        print(f"  Fold Values: {accs}")
        print(f"  Mean: {mean_acc:.2f}%")
        print(f"  Std: {std_acc:.2f}")
        print(f"  Min Acc: {min_acc:.2f}%")
        print(f"  Lower Bound (Mean-Std): {lower_bound:.2f}%")
        print(f"  95% CI: [{ci_low:.2f}, {ci_high:.2f}]")
    
    print("\n" + "=" * 60)
    print("P2.2: 统计显著性检验")
    print("=" * 60)
    
    # 获取各方法的准确率
    baseline_accs = df[df['method'] == 'Baseline']['val_acc'].values
    ra_accs = df[df['method'] == 'RandAugment']['val_acc'].values
    sas_accs = df[df['method'] == 'Ours_optimal']['val_acc'].values
    
    print("\n--- SAS vs RandAugment ---")
    
    # Paired t-test (均值差异)
    t_stat, t_pvalue = stats.ttest_rel(sas_accs, ra_accs)
    print(f"Paired t-test:")
    print(f"  t-statistic: {t_stat:.4f}")
    print(f"  p-value: {t_pvalue:.4f}")
    print(f"  结论: {'显著' if t_pvalue < 0.05 else '不显著'} (α=0.05)")
    
    # Levene's test (方差差异)
    levene_stat, levene_pvalue = stats.levene(sas_accs, ra_accs)
    print(f"\nLevene's test (方差齐性):")
    print(f"  statistic: {levene_stat:.4f}")
    print(f"  p-value: {levene_pvalue:.4f}")
    print(f"  结论: 方差{'显著不同' if levene_pvalue < 0.05 else '无显著差异'} (α=0.05)")
    
    # F-test for variance ratio
    var_sas = np.var(sas_accs, ddof=1)
    var_ra = np.var(ra_accs, ddof=1)
    f_stat = var_ra / var_sas  # RandAugment variance / SAS variance
    print(f"\n方差比:")
    print(f"  Var(RandAugment) / Var(SAS) = {var_ra:.4f} / {var_sas:.4f} = {f_stat:.2f}")
    print(f"  即 RandAugment 方差是 SAS 的 {f_stat:.1f} 倍")
    
    print("\n--- SAS vs Baseline ---")
    t_stat2, t_pvalue2 = stats.ttest_rel(sas_accs, baseline_accs)
    print(f"Paired t-test: t={t_stat2:.4f}, p={t_pvalue2:.4f}")
    
    print("\n" + "=" * 60)
    print("LaTeX 表格格式输出")
    print("=" * 60)
    
    print("\n% Table 1 升级版")
    print("\\begin{tabular}{lccccc}")
    print("\\toprule")
    print("Policy & Val Acc (\\%) & Std Dev & Min Acc & Lower Bound & 95\\% CI \\\\")
    print("\\midrule")
    for r in results:
        ci_str = f"[{r['95% CI Low']:.1f}, {r['95% CI High']:.1f}]"
        print(f"{r['Method']} & {r['Mean']:.2f} & {r['Std']:.2f} & {r['Min Acc']:.2f} & {r['Lower Bound']:.2f} & {ci_str} \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")
    
    print("\n% 统计检验结果 (放在正文中)")
    print(f"% SAS vs RandAugment: t-test p={t_pvalue:.3f}, Levene p={levene_pvalue:.3f}")
    
    # 保存结果到 CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv('outputs/table1_extended.csv', index=False)
    print(f"\n结果已保存到: outputs/table1_extended.csv")

if __name__ == "__main__":
    main()
