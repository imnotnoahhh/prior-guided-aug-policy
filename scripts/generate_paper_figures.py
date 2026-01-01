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
    df = pd.read_csv('outputs/phase_d_results.csv')
    
    # Rename op_name to method for clarity
    if 'op_name' in df.columns:
        df.rename(columns={'op_name': 'method'}, inplace=True)
    
    # Filter methods
    methods = ['Baseline', 'RandAugment', 'Ours_optimal'] 
    df = df[df['method'].isin(methods)]
    
    # Order
    df['method'] = pd.Categorical(df['method'], categories=methods, ordered=True)
    
    plt.figure(figsize=(8, 6))
    
    # Custom color palette
    colors = {'Baseline': '#95a5a6', 'RandAugment': '#e74c3c', 'Ours_optimal': '#2ecc71'}
    
    sns.boxplot(x='method', y='val_acc', data=df, palette=colors, width=0.5)
    sns.stripplot(x='method', y='val_acc', data=df, color='black', alpha=0.6, jitter=0.1)
    
    plt.title('Stability Analysis: Training Variance (5 Folds)', fontsize=14, fontweight='bold')
    plt.ylabel('Validation Accuracy (%)')
    plt.xlabel('')
    
    # Add Std Dev annotation
    stats = df.groupby('method')['val_acc'].agg(['mean', 'std'])
    for i, method in enumerate(methods):
        mean = stats.loc[method, 'mean']
        std = stats.loc[method, 'std']
        plt.text(i, mean + 1.5, f"Mean: {mean:.1f}%\n$ \\sigma $: {std:.2f}",
                 ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.ylim(36, 45)  # Zoom in to see variance
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig1_stability_boxplot.png'))
    plt.close()

def plot_search_space_heatmap():
    print("Generating Phase A Search Space Heatmap...")
    df = pd.read_csv('outputs/phase_a_results.csv')
    
    # Filter for the winning op: ColorJitter
    df_op = df[df['op_name'] == 'ColorJitter'].copy()
    
    plt.figure(figsize=(8, 6))
    
    sc = plt.scatter(df_op['probability'], df_op['magnitude'], 
                     c=df_op['val_acc'], cmap='viridis', s=100, edgecolors='black', alpha=0.8)
    
    plt.colorbar(sc, label='Valid Accuracy (%)')
    
    plt.title('Prior-Guided Search Space (ColorJitter)', fontsize=14, fontweight='bold')
    plt.xlabel('Probability (p)')
    plt.ylabel('Magnitude (m)')
    
    # Highlight the chosen point roughly (Phase C selected ~ m=0.25, p=0.42)
    plt.scatter([0.42], [0.25], s=200, facecolors='none', edgecolors='red', linewidth=2, label='Optimal Found')
    plt.legend(loc='upper left')
    
    plt.xlim(0, 1.0)
    plt.ylim(0, 1.0)
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig2_search_space_colorjitter.png'))
    plt.close()

def plot_complexity_tradeoff():
    print("Generating Complexity-Stability Trade-off...")
    
    # Load data from summary
    df_sum = pd.read_csv('outputs/phase_d_summary.csv')
    
    # Map methods to our simplified labels
    method_map = {
        'Baseline': 'Baseline',
        'Ours_optimal': 'Ours (Optimal)',
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
                'Complexity': 1 if m_id == 'Baseline' else (2 if m_id == 'Ours_optimal' else 8)
            })
    
    df_plot = pd.DataFrame(plot_data)
    
    plt.figure(figsize=(8, 6))
    colors = ['#95a5a6', '#2ecc71', '#e74c3c']
    
    # Bubble chart
    sizes = [(acc - 20) * 30 for acc in df_plot['Accuracy']]
    
    plt.scatter(df_plot['Complexity'], df_plot['Std'], s=sizes, c=colors, alpha=0.7, edgecolors='black')
    
    # Add labels
    for i, row in df_plot.iterrows():
        plt.text(row['Complexity'], row['Std'] + 0.05, 
                 f"{row['Method']}\nAcc: {row['Accuracy']:.1f}%\nStd: {row['Std']:.2f}", 
                 ha='center', fontweight='bold', fontsize=9)

    plt.title('The Complexity Gap: Stability vs. Complexity', fontsize=14, fontweight='bold')
    plt.ylabel('Instability (Standard Deviation) $\\downarrow$', fontsize=12)
    plt.xlabel('Algorithmic Complexity (Search Space Size) $\\rightarrow$', fontsize=12)
    
    plt.xticks([1, 2, 8], ['None', 'Single-Op', 'Multi-Op (N=2,M=9)'])
    plt.ylim(0.5, 1.5)
    plt.xlim(0, 10)
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig3_complexity_gap.png'))
    plt.close()

def plot_ablation_magnitude():
    print("Generating Ablation: Magnitude Sensitivity...")
    try:
        df = pd.read_csv('outputs/ablation/ablation_p0.5_summary.csv')
        df = df.sort_values('magnitude')
        
        plt.figure(figsize=(8, 6))
        plt.plot(df['magnitude'], df['mean_val_acc'], marker='o', linewidth=2, color='#3498db', label='Validation Acc')
        
        # Highlight best
        best_row = df.loc[df['mean_val_acc'].idxmax()]
        plt.scatter([best_row['magnitude']], [best_row['mean_val_acc']], color='red', s=100, zorder=5, label='Optimal Magnitude')
        
        plt.title('Ablation: Impact of Magnitude (Fixed P=0.5)', fontsize=14, fontweight='bold')
        plt.xlabel('Magnitude (m)')
        plt.ylabel('Validation Accuracy (%)')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend()
        
        # Add annotation for the gap
        min_acc = df['mean_val_acc'].min()
        max_acc = df['mean_val_acc'].max()
        plt.annotate(f'Gap: {max_acc - min_acc:.1f}%', 
                     xy=(best_row['magnitude'], max_acc), 
                     xytext=(best_row['magnitude']+0.1, max_acc - 2),
                     arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5))

        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'fig4_ablation_magnitude.png'))
        plt.close()
    except Exception as e:
        print(f"Skipping Ablation plot: {e}")

def plot_cifar10_generalization():
    print("Generating CIFAR-10 Generalization Comparison...")
    try:
        df = pd.read_csv('outputs/cifar10_50shot_results.csv')
        
        plt.figure(figsize=(8, 6))
        colors = ['#95a5a6', '#e74c3c', '#2ecc71']
        
        bars = plt.bar(df['Method'], df['Mean'], yerr=df['Std'], color=colors, capsize=10, alpha=0.8, edgecolor='black')
        
        plt.title('Generalization: CIFAR-10 (50-shot)', fontsize=14, fontweight='bold')
        plt.ylabel('Mean Accuracy (%)')
        plt.ylim(0, 70)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 2,
                     f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'fig5_cifar10_generalization.png'))
        plt.close()
    except Exception as e:
        print(f"Skipping CIFAR-10 plot: {e}")

if __name__ == "__main__":
    try:
        plot_stability_boxplot()
        plot_search_space_heatmap()
        plot_complexity_tradeoff()
        plot_ablation_magnitude()
        plot_cifar10_generalization()
        print(f"\nAll figures saved successfully to: {OUTPUT_DIR}")
    except Exception as e:
        print(f"Error generating figures: {e}")
