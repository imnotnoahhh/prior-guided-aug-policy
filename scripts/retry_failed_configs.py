#!/usr/bin/env python3
"""
补跑失败的配置
用法: python scripts/retry_failed_configs.py
"""
import sys
import re
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils import get_device, set_seed_deterministic, load_phase0_best_config
from main_phase_b import train_to_epoch, write_raw_csv_row, aggregate_results
import pandas as pd

def get_failed_configs_from_csv(csv_path):
    """从 CSV 中读取失败的配置"""
    if not csv_path.exists():
        return []
    
    df = pd.read_csv(csv_path)
    failed = df[df["error"].notna() & (df["error"] != "")]
    
    if len(failed) == 0:
        return []
    
    # 提取 (op_name, magnitude, probability)
    configs = []
    for _, row in failed.iterrows():
        configs.append((
            row["op_name"],
            float(row["magnitude"]),
            float(row["probability"])
        ))
    
    return configs

def get_failed_configs_from_logs(log_dir="logs"):
    """从日志文件中提取失败的配置"""
    log_dir = Path(log_dir)
    configs = []
    
    # 匹配格式: ERROR in GaussianNoise (m=0.2546, p=0.333): ...
    pattern = r"ERROR in (\w+) \(m=([\d.]+), p=([\d.]+)\):"
    
    # 查找所有日志文件
    for log_file in log_dir.glob("*.log"):
        try:
            with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                matches = re.findall(pattern, content)
                for op_name, m, p in matches:
                    configs.append((op_name, float(m), float(p)))
        except Exception as e:
            print(f"Warning: 无法读取日志 {log_file}: {e}")
    
    # 去重
    return list(set(configs))

def main():
    # 手动指定的失败配置（从日志中确认的）
    MANUAL_FAILED_CONFIGS = [
        ("GaussianNoise", 0.0948, 0.4239),
        ("GaussianNoise", 0.0295, 0.6418),
        ("GaussianNoise", 0.2546, 0.333),
    ]
    
    # 加载 Phase 0 超参
    phase0_cfg = load_phase0_best_config()
    wd = phase0_cfg[0] if phase0_cfg else 1e-2
    ls = phase0_cfg[1] if phase0_cfg else 0.1
    
    device = get_device()
    set_seed_deterministic(42, deterministic=True)
    
    output_dir = Path("outputs")
    raw_csv_path = output_dir / "phase_b_tuning_raw.csv"
    
    # 优先使用手动指定的配置
    failed_configs = MANUAL_FAILED_CONFIGS
    
    # 如果手动配置为空，尝试从 CSV 中读取
    if len(failed_configs) == 0:
        failed_configs = get_failed_configs_from_csv(raw_csv_path)
    
    # 如果还是没有，尝试从日志中提取
    if len(failed_configs) == 0:
        print("CSV 中未找到失败的配置，尝试从日志中提取...")
        failed_configs = get_failed_configs_from_logs(output_dir.parent / "logs")
        
        if len(failed_configs) == 0:
            print("=" * 70)
            print("没有发现失败的配置")
            print("=" * 70)
            return
        else:
            print(f"从日志中找到 {len(failed_configs)} 个失败的配置")
    else:
        print(f"使用手动指定的 {len(failed_configs)} 个失败的配置")
    
    # 读取 CSV，准备更新失败的配置记录（保持原始顺序）
    if raw_csv_path.exists():
        df = pd.read_csv(raw_csv_path)
        # 保存原始顺序（添加原始索引列）
        df["_original_index"] = df.index
        print(f"读取 CSV: {len(df)} 条记录")
    else:
        print("⚠️  CSV 文件不存在，将创建新文件")
        df = pd.DataFrame()
    
    # 创建失败配置的标识（用于匹配）
    failed_keys = {(op, m, p) for op, m, p in failed_configs}
    
    print("=" * 70)
    print("补跑失败的配置")
    print("=" * 70)
    print(f"设备: {device}")
    print(f"超参: wd={wd}, ls={ls}")
    print(f"失败的配置数: {len(failed_configs)}")
    print("=" * 70)
    
    # ASHA rungs: [40, 100, 200]
    rungs = [40, 100, 200]
    
    # 先补跑所有失败的配置，收集结果
    retried_results = []
    
    for i, (op_name, m, p) in enumerate(failed_configs, 1):
        print(f"\n[{i}/{len(failed_configs)}] {op_name}: m={m:.4f}, p={p:.4f}")
        print("-" * 70)
        
        checkpoint = None
        final_result = None
        
        try:
            # 按 ASHA 流程补跑：40 → 100 → 200 epochs
            for rung_idx, target_epochs in enumerate(rungs, 1):
                print(f"  Rung {rung_idx}/{len(rungs)}: Training to {target_epochs} epochs...")
                
                result, checkpoint = train_to_epoch(
                    op_name=op_name,
                    magnitude=m,
                    probability=p,
                    target_epochs=target_epochs,
                    device=device,
                    checkpoint=checkpoint,  # 从上一个 rung 继续
                    fold_idx=0,
                    batch_size=128,
                    num_workers=4,  # 减少内存占用
                    seed=42,
                    deterministic=True,
                    weight_decay=wd,
                    label_smoothing=ls,
                )
                
                print(f"    → val_acc={result['val_acc']:.2f}%")
                final_result = result
                
                # 如果中间 rung 结果太差，可以提前停止（可选）
                # if rung_idx < len(rungs) and result['val_acc'] < 15.0:
                #     print(f"    → 结果太差，停止补跑")
                #     break
            
            if final_result:
                retried_results.append(final_result)
                print(f"✅ 补跑成功: val_acc={final_result['val_acc']:.2f}%")
            
        except Exception as e:
            print(f"❌ 补跑失败: {e}")
            import traceback
            traceback.print_exc()
    
    # 对比补跑结果和 CSV 中的配置，决定是否替换
    if retried_results and not df.empty:
        print("\n" + "=" * 70)
        print("对比补跑结果和 CSV 中的配置...")
        print("=" * 70)
        
        # 按 val_acc 排序找到最差的配置（但保持原始索引）
        df_sorted = df.sort_values("val_acc", ascending=False).reset_index(drop=True)
        
        # 对每个补跑结果，检查是否需要替换
        for retried in retried_results:
            retried_acc = retried["val_acc"]
            worst_acc = df_sorted.iloc[-1]["val_acc"]
            
            if retried_acc > worst_acc:
                # 补跑结果更好，找到最差配置的原始索引
                worst_original_idx = df_sorted.iloc[-1]["_original_index"]
                worst_row = df_sorted.iloc[-1]
                
                print(f"替换: {worst_row['op_name']} (m={worst_row['magnitude']}, p={worst_row['probability']}, acc={worst_acc:.2f}%)")
                print(f"   → {retried['op_name']} (m={retried['magnitude']}, p={retried['probability']}, acc={retried_acc:.2f}%)")
                
                # 在原始 DataFrame 中替换（保持原始位置）
                original_idx = df[df["_original_index"] == worst_original_idx].index[0]
                for key, value in retried.items():
                    df.at[original_idx, key] = value
                
                # 重新排序用于下次比较
                df_sorted = df.sort_values("val_acc", ascending=False).reset_index(drop=True)
            else:
                print(f"跳过: {retried['op_name']} (m={retried['magnitude']}, p={retried['probability']}, acc={retried_acc:.2f}%) < 最差配置 ({worst_acc:.2f}%)")
        
        print(f"\n最终 CSV 记录数: {len(df)} (保持 60 条)")
        
        # 恢复原始顺序（如果有原始索引）
        if "_original_index" in df.columns:
            df = df.sort_values("_original_index").reset_index(drop=True)
            df = df.drop(columns=["_original_index"])
            print("已恢复原始顺序")
    
    # 保存更新后的 CSV（保持原始顺序）
    if not df.empty:
        # 确保字段顺序正确
        fieldnames = [
            "phase", "op_name", "magnitude", "probability", "seed", "fold_idx",
            "val_acc", "val_loss", "top5_acc", "train_acc", "train_loss",
            "epochs_run", "best_epoch", "early_stopped", "runtime_sec",
            "timestamp", "error", "stable_score"
        ]
        # 只保留存在的字段
        existing_fields = [f for f in fieldnames if f in df.columns]
        df = df[existing_fields]
        df.to_csv(raw_csv_path, index=False)
        print(f"\n已保存更新后的 CSV: {len(df)} 条记录（保持原始顺序）")
    
    # 重新生成 summary（包含补跑后的结果）
    if not df.empty:
        print("\n" + "=" * 70)
        print("重新生成 summary...")
        print("=" * 70)
        
        # 删除旧的 summary（确保重新生成）
        summary_path = output_dir / "phase_b_tuning_summary.csv"
        if summary_path.exists():
            summary_path.unlink()
            print(f"已删除旧的 summary: {summary_path}")
        
        summary = aggregate_results(
            raw_csv_path=raw_csv_path,
            summary_csv_path=summary_path
        )
        if not summary.empty:
            print("\nTop 10 configurations by mean_val_acc:")
            print(summary.head(10).to_string(index=False))
            print(f"\nSummary 已更新: {output_dir / 'phase_b_tuning_summary.csv'}")
    
    print("\n" + "=" * 70)
    print("补跑完成！")
    print(f"Raw CSV 已更新: {raw_csv_path}")
    print(f"Summary 已重新生成: {output_dir / 'phase_b_tuning_summary.csv'}")
    print("=" * 70)

if __name__ == "__main__":
    main()

