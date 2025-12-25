#!/usr/bin/env python
"""
PyTorch Profiler - 训练瓶颈分析 (v7)

用法示例:
  # 真实训练建议：BF16 + channels_last (默认)
  python scripts/profile_training.py --batch_size 512 --num_workers 20 --prefetch_factor 8

  # FP32 baseline：关 AMP
  python scripts/profile_training.py --batch_size 512 --num_workers 20 --prefetch_factor 8 --disable_amp

  # FP32 + 关 TF32（你刚跑的那条，用来消除某些内部格式转换）
  python scripts/profile_training.py --batch_size 512 --num_workers 20 --prefetch_factor 8 --disable_amp --no_tf32

  # deterministic（注意：会更慢；v7 自动处理 CUBLAS_WORKSPACE_CONFIG）
  python scripts/profile_training.py --batch_size 512 --num_workers 20 --prefetch_factor 8 --disable_amp --deterministic --no_tf32

  # 强制 channels_last（即使 disable_amp 也强制）
  python scripts/profile_training.py --batch_size 512 --num_workers 20 --prefetch_factor 8 --disable_amp --force_channels_last
"""
import argparse
import os
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
from src.augmentations import build_transform_with_op, get_val_transform
from src.models import create_model
from src.utils import get_device


def _configure_reproducibility(deterministic: bool):
    """
    deterministic=True 时：
    - 关闭 cudnn benchmark
    - 启用 deterministic algorithms
    - 设置 CUBLAS_WORKSPACE_CONFIG，避免你刚才的 CuBLAS 报错
    """
    if not deterministic:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        return

    # 必须在触发 cublas 之前设置（一般在第一次 matmul/conv 之前设置就行）
    if "CUBLAS_WORKSPACE_CONFIG" not in os.environ:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def _configure_tf32(enable_tf32: bool):
    # TF32 影响 matmul / cudnn（A100 上通常对吞吐很关键）
    torch.backends.cuda.matmul.allow_tf32 = enable_tf32
    torch.backends.cudnn.allow_tf32 = enable_tf32


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
    no_tf32: bool = False,
    deterministic: bool = False,
    force_channels_last: bool = False,
):
    device = get_device()
    use_cuda = device.type == "cuda"

    # deterministic / tf32 配置要尽早做
    _configure_reproducibility(deterministic=deterministic)
    _configure_tf32(enable_tf32=(use_cuda and not no_tf32))

    # AMP 配置：默认 BF16 autocast（不使用 GradScaler，避免隐式同步）
    use_amp = use_cuda and not disable_amp
    autocast_dtype = torch.bfloat16  # A100 原生支持 BF16

    # channels_last 策略：
    # - 默认：启用（与你 v4/v5 目标一致）
    # - 如果 disable_amp 且没 force_channels_last：允许你做 NCHW baseline（更容易对比）
    channels_last_on = True
    if disable_amp and not force_channels_last:
        channels_last_on = False

    print("=" * 70)
    print("PyTorch Profiler - 训练瓶颈分析 (v7)")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}")
    print(f"Num workers: {num_workers}")
    print(f"Warmup batches (pre-profiler): {warmup_batches}")
    print(f"Profile batches (active): {num_batches}")
    print(f"Record shapes: {record_shapes}")
    print(f"Profile memory: {profile_memory}")
    print("-" * 70)
    print(f"AMP: {'DISABLED' if disable_amp else 'enabled (BF16, no GradScaler)'}")
    print(f"Channels last: {'ON (forced)' if force_channels_last else ('ON' if channels_last_on else 'OFF')}")
    print(f"TF32: {'disabled' if (use_cuda and no_tf32) else ('enabled' if use_cuda else 'n/a')}")
    print(f"Deterministic: {'YES' if deterministic else 'NO'}")
    print(f"DataLoader: pin_memory={pin_memory}, persistent_workers={persistent_workers}, "
          f"prefetch_factor={prefetch_factor}, drop_last={drop_last}")
    print("=" * 70)

    # transforms
    train_transform = build_transform_with_op(
        op_name="ColorJitter",
        magnitude=0.5,
        probability=0.5,
        include_baseline=True,
        include_normalize=False,
    )
    _ = get_val_transform(include_normalize=False)

    train_dataset = CIFAR100Subsampled(
        root="./data",
        train=True,
        fold_idx=0,
        transform=train_transform,
        download=True,
    )

    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": True,
        "num_workers": num_workers,
        "pin_memory": pin_memory if use_cuda else False,
        "drop_last": drop_last,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = persistent_workers
        loader_kwargs["prefetch_factor"] = prefetch_factor

    train_loader = DataLoader(train_dataset, **loader_kwargs)
    print(f"\nDataLoader 配置: {loader_kwargs}")

    model = create_model(num_classes=100, pretrained=False).to(device)
    if channels_last_on or force_channels_last:
        model = model.to(memory_format=torch.channels_last)
        print("模型: channels_last (NHWC)")
    else:
        print("模型: NCHW (默认)")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=1e-3)

    if use_amp:
        print(f"AMP 已启用: autocast dtype={autocast_dtype}, 无 GradScaler")
    else:
        print("AMP 已禁用")

    print("\n开始 Profiling...")
    print("-" * 70)

    model.train()
    data_iter = iter(train_loader)

    # Pre-profiler warmup（不显式 synchronize）
    for _ in range(warmup_batches):
        try:
            images, labels = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            images, labels = next(data_iter)

        if channels_last_on or force_channels_last:
            images = images.to(device, non_blocking=True, memory_format=torch.channels_last)
        else:
            images = images.to(device, non_blocking=True)

        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with torch.amp.autocast(device_type="cuda", dtype=autocast_dtype):
                outputs = model(images)
                loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    print(f"Pre-profiler warmup 完成 ({warmup_batches} batches, 无显式 synchronize)")

    activities = [ProfilerActivity.CPU]
    if use_cuda:
        activities.append(ProfilerActivity.CUDA)

    prof_wait = 3
    prof_warmup = 2
    prof_active = num_batches
    total_steps = prof_wait + prof_warmup + prof_active

    prof_sched = schedule(wait=prof_wait, warmup=prof_warmup, active=prof_active, repeat=1)

    with profile(
        activities=activities,
        schedule=prof_sched,
        record_shapes=record_shapes,
        profile_memory=profile_memory,
        with_stack=False,
    ) as prof:
        for _ in range(total_steps):
            try:
                images, labels = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                images, labels = next(data_iter)

            with record_function("data_to_device"):
                if channels_last_on or force_channels_last:
                    images = images.to(device, non_blocking=True, memory_format=torch.channels_last)
                else:
                    images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            if use_amp:
                with record_function("forward"):
                    with torch.amp.autocast(device_type="cuda", dtype=autocast_dtype):
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                with record_function("backward"):
                    loss.backward()
                with record_function("optimizer_step"):
                    optimizer.step()
            else:
                with record_function("forward"):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                with record_function("backward"):
                    loss.backward()
                with record_function("optimizer_step"):
                    optimizer.step()

            prof.step()

    print(f"Profiling 完成 (schedule: wait={prof_wait}, warmup={prof_warmup}, active={prof_active})")

    print("\n" + "=" * 70)
    print("按 CPU 时间排序 (Top 20)")
    print("=" * 70)
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))

    if use_cuda:
        print("\n" + "=" * 70)
        print("按 CUDA 时间排序 (Top 20)")
        print("=" * 70)
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

    trace_path = PROJECT_ROOT / "outputs" / "profiler_trace.json"
    trace_path.parent.mkdir(exist_ok=True)
    prof.export_chrome_trace(str(trace_path))
    print(f"\n详细 trace 已保存到: {trace_path}")
    print("Chrome 打开 chrome://tracing 加载查看")

    # v7 快速检查：只盯“真实转换”和“真实同步”关键词
    print("\n" + "=" * 70)
    print("快速检查 (v7)")
    print("=" * 70)

    key_averages = prof.key_averages()

    sync_hits = [it for it in key_averages if "cudaDeviceSynchronize" in it.key or it.key == "aten::item"]
    if sync_hits:
        print("⚠️  同步相关 op（少量可能是 profiler 自己统计造成）:")
        for it in sync_hits:
            print(f"    - {it.key}: count={it.count}, cpu_total={it.cpu_time_total/1000:.3f}ms")
    else:
        print("✅ 未发现显式同步相关 op")

    # “真实转换”重点看这些：convertTensor / nchwToNhwc / nhwcToNchw
    convert_hits = [it for it in key_averages if
                    ("convertTensor" in it.key) or
                    ("nchwToNhwc" in it.key) or
                    ("nhwcToNchw" in it.key)]
    if convert_hits:
        print("⚠️  检测到真实布局/格式转换 op:")
        for it in convert_hits[:10]:
            cuda_self = getattr(it, "self_cuda_time_total", 0) or 0
            print(f"    - {it.key}: count={it.count}, self_cuda={cuda_self/1000:.3f}ms")
    else:
        print("✅ 未发现 convertTensor / nchw<->nhwc 等真实转换 op")

    print("\n" + "=" * 70)
    print("Done")
    print("=" * 70)


def parse_args():
    p = argparse.ArgumentParser(
        description="PyTorch Profiler - 训练瓶颈分析 (v7)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--num_workers", type=int, default=20)
    p.add_argument("--num_batches", type=int, default=20)
    p.add_argument("--warmup_batches", type=int, default=5)

    p.add_argument("--record_shapes", action="store_true")
    p.add_argument("--profile_memory", action="store_true")

    p.add_argument("--disable_amp", action="store_true")

    p.add_argument("--pin_memory", action="store_true", default=True)
    p.add_argument("--no_pin_memory", action="store_false", dest="pin_memory")

    p.add_argument("--persistent_workers", action="store_true", default=True)
    p.add_argument("--no_persistent_workers", action="store_false", dest="persistent_workers")

    p.add_argument("--prefetch_factor", type=int, default=4)

    p.add_argument("--drop_last", action="store_true", default=True)
    p.add_argument("--no_drop_last", action="store_false", dest="drop_last")

    # v7 新增
    p.add_argument("--no_tf32", action="store_true", help="Disable TF32 (CUDA matmul + cuDNN)")
    p.add_argument("--deterministic", action="store_true", help="Enable deterministic algorithms (slower)")
    p.add_argument("--force_channels_last", action="store_true", help="Force channels_last even when --disable_amp")

    return p.parse_args()


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
        no_tf32=args.no_tf32,
        deterministic=args.deterministic,
        force_channels_last=args.force_channels_last,
    )