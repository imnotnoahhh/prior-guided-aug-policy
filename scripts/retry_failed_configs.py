#!/usr/bin/env python3
"""
补跑失败的配置
用法: python scripts/retry_failed_configs.py
"""
import sys
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

def main():
    # 加载 Phase 0 超参
    phase0_cfg = load_phase0_best_config()
    wd = phase0_cfg[0] if phase0_cfg else 1e-2
    ls = phase0_cfg[1] if phase0_cfg else 0.1
    
    device = get_device()
    set_seed_deterministic(42, deterministic=True)
    
    output_dir = Path("outputs")
    raw_csv_path = output_dir / "phase_b_tuning_raw.csv"
    
    # 从 CSV 中读取失败的配置
    failed_configs = get_failed_configs_from_csv(raw_csv_path)
    
    if len(failed_configs) == 0:
        print("=" * 70)
        print("没有发现失败的配置")
        print("=" * 70)
        return
    
    # 读取 CSV，准备更新失败的配置记录（保持原始顺序）
    if raw_csv_path.exists():
        df = pd.read_csv(raw_csv_path)
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
            
            # 更新 CSV 中对应的记录（保持原始顺序）
            if not df.empty:
                # 找到对应的失败记录（考虑浮点数精度问题）
                # CSV 中保存的是 str(round(m, 4))，所以需要转换为 float 比较
                df_m = df["magnitude"].astype(str).astype(float)
                df_p = df["probability"].astype(str).astype(float)
                mask = (
                    (df["op_name"] == op_name) &
                    (abs(df_m - m) < 1e-5) &  # 考虑浮点数精度
                    (abs(df_p - p) < 1e-5) &
                    (df["error"].notna() & (df["error"] != ""))
                )
                if mask.any():
                    # 更新记录
                    idx = df[mask].index[0]
                    for key, value in final_result.items():
                        df.at[idx, key] = value
                    print(f"✅ 成功: 已更新记录 (val_acc={final_result['val_acc']:.2f}%)")
                else:
                    # 如果找不到（可能已经被删除），追加到最后
                    df = pd.concat([df, pd.DataFrame([final_result])], ignore_index=True)
                    print(f"✅ 成功: 已追加记录 (val_acc={final_result['val_acc']:.2f}%)")
            else:
                # CSV 为空，创建新记录
                df = pd.DataFrame([final_result])
                print(f"✅ 成功: 已创建记录 (val_acc={final_result['val_acc']:.2f}%)")
            
        except Exception as e:
            print(f"❌ 失败: {e}")
            import traceback
            traceback.print_exc()
    
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
        print(f"\n已保存更新后的 CSV: {len(df)} 条记录")
    
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

