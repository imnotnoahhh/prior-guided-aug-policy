import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import matplotlib.patches as mpatches

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({'font.size': 12, 'figure.dpi': 300})

OUTPUT_DIR = 'outputs/figures'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def plot_stability_boxplot():
    print("Generating Stability Boxplot...")
    import numpy as np
    np.random.seed(42)  # Fix jitter positions for reproducibility
    
    df = pd.read_csv('outputs/phase_d_results.csv')
    
    # Rename op_name to method for clarity
    if 'op_name' in df.columns:
        df.rename(columns={'op_name': 'method'}, inplace=True)
    
    # Filter methods and rename for display
    methods_data = ['Baseline', 'RandAugment', 'Ours_optimal'] 
    methods_display = ['Baseline', 'RandAugment', 'SAS (Ours)']  # Updated display names
    df = df[df['method'].isin(methods_data)]
    
    # Rename for display
    df['method'] = df['method'].replace({'Ours_optimal': 'SAS (Ours)'})
    
    # Order - set as categorical and sort
    df['method'] = pd.Categorical(df['method'], categories=methods_display, ordered=True)
    df = df.sort_values('method')  # Ensure data is sorted by category order
    
    plt.figure(figsize=(8, 6))
    
    # Custom color palette
    colors = {'Baseline': '#95a5a6', 'RandAugment': '#e74c3c', 'SAS (Ours)': '#2ecc71'}
    
    # Use order parameter to ensure consistent ordering
    # showfliers=False to avoid overlap with stripplot points
    sns.boxplot(x='method', y='val_acc', hue='method', data=df, palette=colors, 
                width=0.5, legend=False, order=methods_display, showfliers=False)
    sns.stripplot(x='method', y='val_acc', data=df, color='black', alpha=0.7, 
                  jitter=0.05, order=methods_display, size=6)  # Slight jitter to avoid overlap
    
    plt.title('Validation Accuracy Distribution (5 Folds)', fontsize=14, fontweight='bold')
    plt.ylabel('Top-1 Accuracy (%)')
    plt.xlabel('Method')
    
    # Add Std Dev annotation
    stats = df.groupby('method', observed=True)['val_acc'].agg(['mean', 'std'])
    for i, method in enumerate(methods_display):
        mean = stats.loc[method, 'mean']
        std = stats.loc[method, 'std']
        plt.text(i, mean + 1.5, f"Mean: {mean:.1f}%\n$ \\sigma $: {std:.2f}",
                 ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.ylim(36, 45)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig5_stability_boxplot.png'))
    plt.close()

def plot_search_space_heatmap():
    print("Generating Phase A Search Space Heatmap...")
    df = pd.read_csv('outputs/phase_a_results.csv')
    
    # Filter for the winning op: ColorJitter
    df_op = df[df['op_name'] == 'ColorJitter'].copy()
    
    plt.figure(figsize=(8, 6))
    
    sc = plt.scatter(df_op['probability'], df_op['magnitude'], 
                     c=df_op['val_acc'], cmap='viridis', s=100, edgecolors='black', alpha=0.8)
    
    plt.colorbar(sc, label='Validation Accuracy (%)')
    
    plt.title('Search Space Manifold: ColorJitter', fontsize=14, fontweight='bold')
    plt.xlabel('Probability ($p$)')
    plt.ylabel('Magnitude ($m$)')
    
    # Find and highlight the actual optimal point (highest val_acc)
    best_idx = df_op['val_acc'].idxmax()
    best_p = df_op.loc[best_idx, 'probability']
    best_m = df_op.loc[best_idx, 'magnitude']
    best_acc = df_op.loc[best_idx, 'val_acc']
    
    plt.scatter([best_p], [best_m], s=250, facecolors='none', edgecolors='red', linewidth=3, 
                label=f'Optimal (p={best_p:.2f}, m={best_m:.2f}, Acc={best_acc:.1f}%)')
    plt.legend(loc='upper left', fontsize=9)
    
    plt.xlim(0, 1.0)
    plt.ylim(0, 1.0)
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig4_search_space_colorjitter.png'))
    plt.close()

def plot_complexity_tradeoff():
    print("Generating Complexity-Stability Trade-off...")
    df_sum = pd.read_csv('outputs/phase_d_summary.csv')
    
    method_map = {
        'Baseline': 'Baseline',
        'Ours_optimal': 'SAS (Ours)',  # Updated name
        'RandAugment': 'RandAugment'
    }
    
    plot_data = []
    for m_id, label in method_map.items():
        row = df_sum[df_sum['method'] == m_id]
        if not row.empty:
            plot_data.append({
                'Method': label,
                'Accuracy': row.iloc[0]['mean_val_acc'],
                'Std': row.iloc[0]['std_val_acc'],
                'Complexity': 1 if m_id == 'Baseline' else (4 if m_id == 'Ours_optimal' else 8)
            })
    
    df_plot = pd.DataFrame(plot_data)
    
    plt.figure(figsize=(8, 6))
    colors = ['#95a5a6', '#2ecc71', '#e74c3c']
    
    # Bubble chart
    sizes = [(acc - 20) * 40 for acc in df_plot['Accuracy']]
    
    plt.scatter(df_plot['Complexity'], df_plot['Std'], s=sizes, c=colors, alpha=0.7, edgecolors='black')
    
    for i, row in df_plot.iterrows():
        plt.text(row['Complexity'], row['Std'] + 0.08, 
                 f"{row['Method']}\nAcc: {row['Accuracy']:.1f}%\n$ \\sigma $: {row['Std']:.2f}", 
                 ha='center', fontweight='bold', fontsize=10)

    # === NEW: Add arrow and "33% variance reduction" annotation ===
    # Get SAS and RandAugment data
    sas_row = df_plot[df_plot['Method'] == 'SAS (Ours)'].iloc[0]
    ra_row = df_plot[df_plot['Method'] == 'RandAugment'].iloc[0]
    
    # Draw arrow from RandAugment to SAS
    plt.annotate('', 
                 xy=(sas_row['Complexity'], sas_row['Std']),  # arrow head
                 xytext=(ra_row['Complexity'], ra_row['Std']),  # arrow tail
                 arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=2, 
                                connectionstyle='arc3,rad=-0.2'))
    
    # Calculate variance reduction percentage
    variance_reduction = (ra_row['Std'] - sas_row['Std']) / ra_row['Std'] * 100
    
    # Add "33% variance reduction" text
    mid_x = (sas_row['Complexity'] + ra_row['Complexity']) / 2
    mid_y = (sas_row['Std'] + ra_row['Std']) / 2 - 0.12
    plt.text(mid_x + 1.5, mid_y, f'{variance_reduction:.0f}% variance\nreduction', 
             ha='center', va='center', fontsize=11, fontweight='bold', 
             color='#c0392b', bbox=dict(boxstyle='round,pad=0.3', facecolor='#fadbd8', edgecolor='#e74c3c'))
    # === END NEW ===

    plt.title('The Complexity Gap: Accuracy-Stability Balance', fontsize=14, fontweight='bold')
    plt.ylabel('Instability (Standard Deviation $ \\sigma $) $\\downarrow$', fontsize=12)
    plt.xlabel('Algorithmic Complexity (Operations) $\\rightarrow$', fontsize=12)
    
    plt.xticks([1, 4, 8], ['Baseline', 'SAS (Ours)', 'RandAugment'])  # Spread out labels
    plt.ylim(0.5, 1.5)
    plt.xlim(0, 10)
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig1_complexity_gap.png'))
    plt.close()

def plot_ablation_magnitude():
    print("Generating Ablation: Magnitude Sensitivity...")
    try:
        df = pd.read_csv('outputs/ablation/ablation_p0.5_summary.csv')
        df = df.sort_values('magnitude')
        
        plt.figure(figsize=(8, 6))
        plt.plot(df['magnitude'], df['mean_val_acc'], marker='o', linewidth=2, color='#3498db', label='Validation Acc')
        
        best_row = df.loc[df['mean_val_acc'].idxmax()]
        # Marking both optimal and a standard high point
        plt.scatter([best_row['magnitude']], [best_row['mean_val_acc']], color='red', s=120, zorder=5, label=f'Optimal ($m={best_row["magnitude"]:.2f}$)')
        
        # Mark the "default" point if likely m=0.5
        m_05 = df.iloc[(df['magnitude']-0.54).abs().argsort()[:1]]
        if not m_05.empty:
             plt.scatter(m_05['magnitude'], m_05['mean_val_acc'], color='gray', marker='x', s=100, zorder=5, label='Common Fixed Magnitude')

        plt.title('Sensitivity Analysis: Impact of Magnitude (Fixed $p=0.5$)', fontsize=14, fontweight='bold')
        plt.xlabel('Magnitude Parameter ($m$)')
        plt.ylabel('Top-1 Accuracy (%)')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend()
        
        # Annotation
        min_acc = df['mean_val_acc'].min()
        max_acc = df['mean_val_acc'].max()
        plt.annotate(f'Accuracy Gap: {max_acc - min_acc:.1f}%', 
                     xy=(best_row['magnitude'], max_acc), 
                     xytext=(best_row['magnitude']-0.15, max_acc - 2),
                     arrowprops=dict(facecolor='black', shrink=0.08, width=1, headwidth=6))

        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'fig7_ablation_magnitude.png'))
        plt.close()
    except Exception as e:
        print(f"Skipping Ablation plot: {e}")

def plot_cifar10_generalization():
    print("Generating CIFAR-10 Generalization Comparison...")
    try:
        df = pd.read_csv('outputs/cifar10_50shot_results.csv')
        
        plt.figure(figsize=(8, 6))
        colors = ['#95a5a6', '#e74c3c', '#2ecc71']
        
        bars = plt.bar(df['Method'], df['Mean'], yerr=df['Std'], color=colors, capsize=12, alpha=0.8, edgecolor='black')
        
        plt.title('Generalization Benchmarking: CIFAR-10 (50-shot)', fontsize=14, fontweight='bold')
        plt.ylabel('Mean Accuracy (%)')
        plt.xlabel('Method')
        plt.ylim(0, 75)
        
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 3,
                     f'{height:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)

        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'fig6_cifar10_generalization.png'))
        plt.close()
    except Exception as e:
        print(f"Skipping CIFAR-10 plot: {e}")

def plot_destructiveness():
    print("Generating Destructiveness (LPIPS/SSIM) Analysis...")
    try:
        df = pd.read_csv('outputs/destructiveness_metrics.csv')
        
        # Clean names
        df['Strategy'] = df['Strategy'].apply(lambda x: x.split(' (')[0])
        
        # Plot SSIM and LPIPS side by side
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        colors = ['#95a5a6', '#e74c3c', '#2ecc71'] # Baseline, RandAug, Ours
        
        # SSIM (Higher is better)
        sns.barplot(x='Strategy', y='SSIM_Mean', hue='Strategy', legend=False, data=df, ax=axes[0], palette=colors, alpha=0.9, edgecolor='black')
        axes[0].errorbar(x=range(len(df)), y=df['SSIM_Mean'], yerr=df['SSIM_Std'], fmt='none', c='black', capsize=5)
        axes[0].set_title('Structure Preservation (SSIM) $\\uparrow$', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('SSIM Score')
        axes[0].set_ylim(0, 0.3)
        
        # LPIPS (Lower is better)
        sns.barplot(x='Strategy', y='LPIPS_Mean', hue='Strategy', legend=False, data=df, ax=axes[1], palette=colors, alpha=0.9, edgecolor='black')
        axes[1].errorbar(x=range(len(df)), y=df['LPIPS_Mean'], yerr=df['LPIPS_Std'], fmt='none', c='black', capsize=5)
        axes[1].set_title('Perceptual Deviation (LPIPS) $\\downarrow$', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('LPIPS Distance')
        axes[1].set_ylim(0, 0.2)
        
        for ax in axes:
            ax.set_xlabel('')
            ax.grid(axis='y', linestyle='--', alpha=0.5)

        plt.suptitle('Destructiveness Analysis: Semantic Preservation', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'fig8_destructiveness.png'), bbox_inches='tight')
        plt.close()
            
    except Exception as e:
        print(f"Skipping Destructiveness plot: {e}")

if __name__ == "__main__":
    try:
        plot_stability_boxplot()
        plot_search_space_heatmap()
        plot_complexity_tradeoff()
        plot_ablation_magnitude()
        plot_cifar10_generalization()
        plot_destructiveness()
        print(f"\nAll figures saved successfully to: {OUTPUT_DIR}")
    except Exception as e:
        print(f"Error generating figures: {e}")
