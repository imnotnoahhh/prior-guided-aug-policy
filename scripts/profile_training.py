#!/usr/bin/env python
"""
PyTorch Profiler 脚本 - 分析训练瓶颈
用法:
    python scripts/profile_training.py
    python scripts/profile_training.py --batch_size 256
    python scripts/profile_training.py --batch_size 256 --record_shapes --profile_memory
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
    batch_size: int = 64,
    num_workers: int = 6,
    num_batches: int = 20,
    warmup_batches: int = 5,
    record_shapes: bool = False,
    profile_memory: bool = False,
):
    """运行 profiler 分析训练瓶颈"""
    
    set_seed_deterministic(42)
    device = get_device()
    
    print("=" * 70)
    print("PyTorch Profiler - 训练瓶颈分析")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}")
    print(f"Num workers: {num_workers}")
    print(f"Warmup batches (pre-profiler): {warmup_batches}")
    print(f"Profile batches: {num_batches}")
    print(f"Record shapes: {record_shapes}")
    print(f"Profile memory: {profile_memory}")
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
    
    # 创建 DataLoader
    use_cuda = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_cuda,
        drop_last=False,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=4 if num_workers > 0 else None,
    )
    
    # 创建模型
    model = create_model(num_classes=100, pretrained=False)
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=1e-3)
    
    # AMP scaler
    scaler = None
    use_amp = device.type == "cuda"
    if use_amp:
        scaler = torch.amp.GradScaler()
    
    print("\n开始 Profiling...")
    print("-" * 70)
    
    # Pre-profiler warmup (outside profiler to avoid measuring warmup)
    model.train()
    data_iter = iter(train_loader)
    for i in range(warmup_batches):
        try:
            images, labels = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            images, labels = next(data_iter)
        
        # non_blocking=True for async H2D transfer
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        optimizer.zero_grad()
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
    
    # Sync before profiling to ensure warmup is complete
    if use_cuda:
        torch.cuda.synchronize()
    
    print(f"Pre-profiler warmup 完成 ({warmup_batches} batches)")
    
    # Profile with schedule for accurate CUDA timing
    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)
    
    # Schedule: wait=2, warmup=2, active=num_batches
    # Total steps = wait + warmup + active
    prof_wait = 2
    prof_warmup = 2
    prof_active = num_batches
    total_steps = prof_wait + prof_warmup + prof_active
    
    prof_sched = schedule(
        wait=prof_wait,
        warmup=prof_warmup,
        active=prof_active,
        repeat=1,
    )
    
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
                # non_blocking=True for async H2D transfer
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
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
        
        print("\n建议:")
        if data_time / total_cpu_time > 0.3:
            print("  ⚠️  数据加载占比较高，建议:")
            print("      - 增加 num_workers")
            print("      - 检查数据增强是否过于复杂")
            print("      - 考虑预加载数据到内存")
        
        if device.type == "cuda":
            try:
                cuda_util = sum(item.self_cuda_time_total for item in key_averages 
                               if hasattr(item, 'self_cuda_time_total') and item.self_cuda_time_total > 0)
                if cuda_util / total_cpu_time < 0.5:
                    print("  ⚠️  GPU 利用率可能较低，建议:")
                    print("      - 增大 batch_size")
                    print("      - 检查是否有 CPU-GPU 同步瓶颈")
            except:
                print("  (无法获取 CUDA 时间统计)")
    else:
        print("  (无法计算时间占比，total_cpu_time=0)")
    
    print("\n" + "=" * 70)
    print("Profiling 完成!")
    print("=" * 70)


def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch Profiler - 训练瓶颈分析")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=6, help="DataLoader workers")
    parser.add_argument("--num_batches", type=int, default=20, help="Number of batches to profile (active steps)")
    parser.add_argument("--warmup_batches", type=int, default=5, help="Pre-profiler warmup batches")
    parser.add_argument("--record_shapes", action="store_true", help="Record tensor shapes (adds overhead)")
    parser.add_argument("--profile_memory", action="store_true", help="Profile memory usage (adds overhead)")
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
    )
