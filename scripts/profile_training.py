#!/usr/bin/env python
"""
PyTorch Profiler 脚本 - 分析训练瓶颈
用法:
    python scripts/profile_training.py
    python scripts/profile_training.py --batch_size 512 --num_workers 20
    python scripts/profile_training.py --batch_size 512 --num_workers 20 --disable_amp
    python scripts/profile_training.py --batch_size 512 --record_shapes --profile_memory

v4 更新:
- 默认参数改为 batch_size=512, num_workers=20
- 始终启用 channels_last 内存格式（模型和输入）
- 移除 --channels_last 参数（现在始终启用）

v3 更新:
- 移除所有 torch.cuda.synchronize() 调用
- warmup 使用额外的 pre-profiler batches 替代显式同步

v2 更新:
- 添加 --disable_amp 选项避免 GradScaler 的隐式同步
- 添加 DataLoader 优化参数
"""
import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.profiler import profile, record_function, ProfilerActivity, schedule

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import CIFAR100Subsampled
from src.augmentations import get_baseline_transform, get_val_transform, build_transform_with_op
from src.models import create_model
from src.utils import set_seed_deterministic, get_device


def run_profiler(
    batch_size: int = 512,
    num_workers: int = 20,
    num_batches: int = 20,
    warmup_batches: int = 5,
    record_shapes: bool = False,
    profile_memory: bool = False,
    disable_amp: bool = False,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    prefetch_factor: int = 4,
    drop_last: bool = True,
):
    """运行 profiler 分析训练瓶颈
    
    Args:
        batch_size: 批次大小 (default: 512)
        num_workers: DataLoader worker 数量 (default: 20)
        num_batches: 要 profile 的批次数 (active steps)
        warmup_batches: profiler 外部的预热批次数
        record_shapes: 是否记录 tensor shapes (增加开销)
        profile_memory: 是否 profile 内存使用 (增加开销)
        disable_amp: 禁用 AMP 以避免 GradScaler 的隐式同步
        pin_memory: DataLoader pin_memory
        persistent_workers: DataLoader persistent_workers
        prefetch_factor: DataLoader prefetch_factor
        drop_last: DataLoader drop_last
    """
    
    set_seed_deterministic(42)
    device = get_device()
    use_cuda = device.type == "cuda"
    
    print("=" * 70)
    print("PyTorch Profiler - 训练瓶颈分析 (v4)")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}")
    print(f"Num workers: {num_workers}")
    print(f"Warmup batches (pre-profiler): {warmup_batches}")
    print(f"Profile batches: {num_batches}")
    print(f"Record shapes: {record_shapes}")
    print(f"Profile memory: {profile_memory}")
    print("-" * 70)
    print(f"AMP: {'DISABLED' if disable_amp else 'enabled'}")
    print(f"Channels last (NHWC): ALWAYS ENABLED")
    print(f"DataLoader: pin_memory={pin_memory}, persistent_workers={persistent_workers}, "
          f"prefetch_factor={prefetch_factor}, drop_last={drop_last}")
    print("=" * 70)
    
    # 使用一个典型的增强配置
    train_transform = build_transform_with_op(
        op_name="ColorJitter",
        magnitude=0.5,
        probability=0.5,
        include_baseline=True,
        include_normalize=False,
    )
    val_transform = get_val_transform(include_normalize=False)
    
    # 创建数据集
    train_dataset = CIFAR100Subsampled(
        root="./data",
        train=True,
        fold_idx=0,
        transform=train_transform,
        download=True,
    )
    
    # 创建 DataLoader (优化配置)
    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": True,
        "num_workers": num_workers,
        "pin_memory": pin_memory if use_cuda else False,
        "drop_last": drop_last,
    }
    # persistent_workers 和 prefetch_factor 需要 num_workers > 0
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = persistent_workers
        loader_kwargs["prefetch_factor"] = prefetch_factor
    
    train_loader = DataLoader(train_dataset, **loader_kwargs)
    
    print(f"\nDataLoader 配置: {loader_kwargs}")
    
    # 创建模型，始终使用 channels_last 内存格式
    model = create_model(num_classes=100, pretrained=False)
    model = model.to(device).to(memory_format=torch.channels_last)
    print("模型已转换为 channels_last (NHWC) 格式")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=1e-3)
    
    # AMP scaler (可选) - GradScaler.step() 内部会检查 inf/nan 可能导致隐式同步
    scaler = None
    use_amp = use_cuda and not disable_amp
    if use_amp:
        scaler = torch.amp.GradScaler()
        print("AMP 已启用 (注意: GradScaler 可能引入隐式同步)")
    else:
        print(f"AMP 已{'禁用' if disable_amp else '不可用 (非 CUDA 设备)'}")
    
    print("\n开始 Profiling...")
    print("-" * 70)
    
    # =========================================================================
    # Pre-profiler warmup (outside profiler to avoid measuring warmup)
    # NOTE: 不使用 torch.cuda.synchronize()！
    # 使用足够多的 warmup batches 让 CUDA kernel 完成编译和缓存
    # =========================================================================
    model.train()
    data_iter = iter(train_loader)
    
    for i in range(warmup_batches):
        try:
            images, labels = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            images, labels = next(data_iter)
        
        # H2D copy 时直接生成 channels_last，避免 GPU 上的 nchwToNhwcKernel 转换
        images = images.to(device, non_blocking=True, memory_format=torch.channels_last)
        labels = labels.to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)
        
        if use_amp:
            with torch.amp.autocast(device_type="cuda"):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    
    # =========================================================================
    # CRITICAL: 不要在这里调用 torch.cuda.synchronize()！
    # 那会被 profiler 捕获并显示为 cudaDeviceSynchronize
    # 使用 profiler 的 wait/warmup 阶段来让 GPU 稳定
    # =========================================================================
    
    print(f"Pre-profiler warmup 完成 ({warmup_batches} batches, 无显式同步)")
    
    # Profile with schedule for accurate CUDA timing
    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)
    
    # Schedule: wait=3, warmup=2, active=num_batches
    # wait 阶段: profiler 不收集数据，但 GPU 继续运行 (替代 synchronize)
    # warmup 阶段: profiler 开始收集但结果会被丢弃
    # active 阶段: 真正收集的数据
    prof_wait = 3
    prof_warmup = 2
    prof_active = num_batches
    total_steps = prof_wait + prof_warmup + prof_active
    
    prof_sched = schedule(
        wait=prof_wait,
        warmup=prof_warmup,
        active=prof_active,
        repeat=1,
    )
    
    # =========================================================================
    # PROFILING LOOP - 严格禁止以下操作:
    # - .item() / float(tensor) / int(tensor)
    # - tensor.cpu() / tensor.numpy()
    # - torch.cuda.synchronize() / cudaDeviceSynchronize
    # - print(tensor) / tqdm.set_postfix(tensor)
    # - 任何会触发 GPU-CPU 同步的操作
    # =========================================================================
    
    with profile(
        activities=activities,
        schedule=prof_sched,
        record_shapes=record_shapes,
        profile_memory=profile_memory,
        with_stack=False,  # Disable stack tracing to reduce overhead
    ) as prof:
        for i in range(total_steps):
            try:
                images, labels = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                images, labels = next(data_iter)
            
            with record_function("data_to_device"):
                # H2D copy 时直接生成 channels_last，避免 GPU 上的 nchwToNhwcKernel 转换
                images = images.to(device, non_blocking=True, memory_format=torch.channels_last)
                labels = labels.to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            
            if use_amp:
                with record_function("forward"):
                    with torch.amp.autocast(device_type="cuda"):
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                
                with record_function("backward"):
                    scaler.scale(loss).backward()
                
                with record_function("optimizer_step"):
                    scaler.step(optimizer)
                    scaler.update()
            else:
                with record_function("forward"):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                
                with record_function("backward"):
                    loss.backward()
                
                with record_function("optimizer_step"):
                    optimizer.step()
            
            # CRITICAL: No .item(), no print, no sync here!
            # Must call prof.step() at the end of each iteration
            prof.step()
    
    print(f"Profiling 完成 (schedule: wait={prof_wait}, warmup={prof_warmup}, active={prof_active})")
    
    # 输出结果
    print("\n" + "=" * 70)
    print("按 CPU 时间排序 (Top 20)")
    print("=" * 70)
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))
    
    if device.type == "cuda":
        print("\n" + "=" * 70)
        print("按 CUDA 时间排序 (Top 20)")
        print("=" * 70)
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
    
    # 保存详细 trace
    trace_path = PROJECT_ROOT / "outputs" / "profiler_trace.json"
    trace_path.parent.mkdir(exist_ok=True)
    prof.export_chrome_trace(str(trace_path))
    print(f"\n详细 trace 已保存到: {trace_path}")
    print("可以在 Chrome 浏览器打开 chrome://tracing 加载查看")
    
    # 简单分析
    print("\n" + "=" * 70)
    print("瓶颈分析建议")
    print("=" * 70)
    
    # 计算各部分时间占比
    key_averages = prof.key_averages()
    total_cpu_time = sum(item.cpu_time_total for item in key_averages)
    
    if total_cpu_time > 0:
        data_time = sum(item.cpu_time_total for item in key_averages 
                        if "DataLoader" in item.key or "data_to_device" in item.key)
        forward_time = sum(item.cpu_time_total for item in key_averages 
                           if "forward" in item.key.lower() or "conv" in item.key.lower())
        backward_time = sum(item.cpu_time_total for item in key_averages 
                            if "backward" in item.key.lower())
        
        print(f"数据加载占比: ~{data_time/total_cpu_time*100:.1f}%")
        print(f"前向传播占比: ~{forward_time/total_cpu_time*100:.1f}%")
        print(f"反向传播占比: ~{backward_time/total_cpu_time*100:.1f}%")
        
        # 检查同步操作
        sync_ops = [item for item in key_averages 
                    if "item" in item.key.lower() or "synchronize" in item.key.lower()]
        if sync_ops:
            print("\n⚠️  检测到潜在同步操作:")
            for op in sync_ops:
                print(f"    - {op.key}: count={op.count}, cpu_time={op.cpu_time_total/1000:.2f}ms")
            print("    建议使用 --disable_amp 禁用 AMP 重新 profile")
        else:
            print("\n✅ 未检测到显式同步操作 (cudaDeviceSynchronize=0)")
        
        # 检查 NCHW→NHWC 转换
        layout_ops = [item for item in key_averages 
                      if "nchw" in item.key.lower() or "nhwc" in item.key.lower()]
        if layout_ops:
            print("\n⚠️  检测到内存布局转换:")
            for op in layout_ops:
                cuda_time = getattr(op, 'self_cuda_time_total', 0) or 0
                print(f"    - {op.key}: count={op.count}, cuda_time={cuda_time/1000:.2f}ms")
        else:
            print("✅ 未检测到 NCHW/NHWC 布局转换")
        
        print("\n建议:")
        if data_time / total_cpu_time > 0.3:
            print("  ⚠️  数据加载占比较高，建议:")
            print("      - 增加 num_workers")
            print("      - 检查数据增强是否过于复杂")
            print("      - 考虑预加载数据到内存")
            print("      - 确保 pin_memory=True, persistent_workers=True")
            print("      - 尝试增加 --prefetch_factor 8")
        
        if device.type == "cuda":
            try:
                cuda_util = sum(item.self_cuda_time_total for item in key_averages 
                               if hasattr(item, 'self_cuda_time_total') and item.self_cuda_time_total > 0)
                if cuda_util / total_cpu_time < 0.5:
                    print("  ⚠️  GPU 利用率可能较低，建议:")
                    print("      - 增大 batch_size")
                    print("      - 检查是否有 CPU-GPU 同步瓶颈")
                    print("      - 尝试 --disable_amp 看同步是否减少")
            except:
                print("  (无法获取 CUDA 时间统计)")
    else:
        print("  (无法计算时间占比，total_cpu_time=0)")
    
    print("\n" + "=" * 70)
    print("Profiling 完成!")
    print("=" * 70)


def parse_args():
    parser = argparse.ArgumentParser(
        description="PyTorch Profiler - 训练瓶颈分析",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 基础 profiling (使用默认参数: batch_size=512, num_workers=20)
  python scripts/profile_training.py
  
  # 禁用 AMP 避免 GradScaler 同步 (推荐用于诊断同步问题)
  python scripts/profile_training.py --batch_size 512 --num_workers 20 --disable_amp
  
  # 优化 DataLoader 配置
  python scripts/profile_training.py --batch_size 512 --num_workers 20 --prefetch_factor 8
  
  # 完整诊断模式
  python scripts/profile_training.py --batch_size 512 --num_workers 20 --disable_amp --record_shapes --profile_memory
"""
    )
    
    # 基础参数
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size (default: 512)")
    parser.add_argument("--num_workers", type=int, default=20, help="DataLoader workers (default: 20)")
    parser.add_argument("--num_batches", type=int, default=20, 
                        help="Number of batches to profile (active steps, default: 20)")
    parser.add_argument("--warmup_batches", type=int, default=5, 
                        help="Pre-profiler warmup batches (default: 5)")
    
    # Profiler 选项
    parser.add_argument("--record_shapes", action="store_true", 
                        help="Record tensor shapes (adds overhead)")
    parser.add_argument("--profile_memory", action="store_true", 
                        help="Profile memory usage (adds overhead)")
    
    # 同步控制
    parser.add_argument("--disable_amp", action="store_true",
                        help="Disable AMP to avoid GradScaler implicit sync (recommended for diagnosing sync issues)")
    
    # DataLoader 优化参数
    parser.add_argument("--pin_memory", action="store_true", default=True,
                        help="Enable pin_memory for faster H2D transfer (default: True)")
    parser.add_argument("--no_pin_memory", action="store_false", dest="pin_memory",
                        help="Disable pin_memory")
    parser.add_argument("--persistent_workers", action="store_true", default=True,
                        help="Enable persistent_workers (default: True)")
    parser.add_argument("--no_persistent_workers", action="store_false", dest="persistent_workers",
                        help="Disable persistent_workers")
    parser.add_argument("--prefetch_factor", type=int, default=4,
                        help="DataLoader prefetch_factor (default: 4, try 8 for large batches)")
    parser.add_argument("--drop_last", action="store_true", default=True,
                        help="Drop last incomplete batch (default: True)")
    parser.add_argument("--no_drop_last", action="store_false", dest="drop_last",
                        help="Keep last incomplete batch")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_profiler(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_batches=args.num_batches,
        warmup_batches=args.warmup_batches,
        record_shapes=args.record_shapes,
        profile_memory=args.profile_memory,
        disable_amp=args.disable_amp,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers,
        prefetch_factor=args.prefetch_factor,
        drop_last=args.drop_last,
    )
