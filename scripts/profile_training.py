#!/usr/bin/env python
"""
PyTorch Profiler 脚本 - 分析训练瓶颈

用法:
    python scripts/profile_training.py
    python scripts/profile_training.py --batch_size 512 --num_workers 20
    python scripts/profile_training.py --batch_size 512 --num_workers 20 --disable_amp
    python scripts/profile_training.py --batch_size 512 --record_shapes --profile_memory

v5 目标:
- 默认 batch_size=512, num_workers=20
- 模型始终 channels_last
- 输入 H2D copy 时直接 channels_last（尽量避免 GPU 上做 NCHW->NHWC）
- AMP: BF16 autocast, 不用 GradScaler（减少隐式同步）
- 可选启用 TF32（A100 推荐）
- profiler schedule: wait/warmup/active
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
from src.augmentations import get_val_transform, build_transform_with_op
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
    tf32: bool = True,
):
    set_seed_deterministic(42)
    device = get_device()
    use_cuda = (device.type == "cuda")

    # TF32（A100 上通常推荐开，能更快；如果你追求严格数值一致可关）
    if use_cuda:
        torch.backends.cuda.matmul.allow_tf32 = tf32
        torch.backends.cudnn.allow_tf32 = tf32

    print("=" * 70)
    print("PyTorch Profiler - 训练瓶颈分析 (v5)")
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
    print("Channels last (NHWC): ALWAYS ENABLED (model + inputs)")
    print(f"TF32: {'enabled' if (use_cuda and tf32) else 'disabled'}")
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

    # dataset
    train_dataset = CIFAR100Subsampled(
        root="./data",
        train=True,
        fold_idx=0,
        transform=train_transform,
        download=True,
    )

    # dataloader
    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": True,
        "num_workers": num_workers,
        "pin_memory": (pin_memory if use_cuda else False),
        "drop_last": drop_last,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = persistent_workers
        loader_kwargs["prefetch_factor"] = prefetch_factor

    train_loader = DataLoader(train_dataset, **loader_kwargs)
    print(f"\nDataLoader 配置: {loader_kwargs}")

    # model
    model = create_model(num_classes=100, pretrained=False)
    model = model.to(device).to(memory_format=torch.channels_last)
    model.train()
    print("模型已转换为 channels_last (NHWC)")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=1e-3)

    # AMP (BF16 on A100)
    use_amp = use_cuda and (not disable_amp)
    autocast_dtype = torch.bfloat16  # A100 原生 BF16
    if use_amp:
        print(f"AMP 已启用: autocast dtype={autocast_dtype}, 无 GradScaler")
    else:
        print("AMP 已禁用" if disable_amp else "AMP 不可用 (非 CUDA)")

    # -------------------------
    # pre-profiler warmup
    # -------------------------
    print("\n开始 Profiling...")
    print("-" * 70)
    data_iter = iter(train_loader)

    for _ in range(warmup_batches):
        try:
            images, labels = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            images, labels = next(data_iter)

        # 关键：H2D 时直接 channels_last，减少 GPU 上 layout 转换
        images = images.to(device, non_blocking=True, memory_format=torch.channels_last)
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

    # -------------------------
    # profiler schedule
    # -------------------------
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
                images = images.to(device, non_blocking=True, memory_format=torch.channels_last)
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

    # results
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

    # simple checks
    print("\n" + "=" * 70)
    print("快速检查")
    print("=" * 70)
    key_averages = prof.key_averages()

    sync_ops = [it for it in key_averages if ("item" in it.key.lower() or "synchronize" in it.key.lower())]
    if sync_ops:
        print("⚠️  检测到潜在同步相关 op（不一定全是你代码触发，但值得关注）:")
        for op in sync_ops[:10]:
            print(f"    - {op.key}: count={op.count}, cpu_total={op.cpu_time_total/1000:.3f}ms")
    else:
        print("✅ 未发现明显同步相关 op（item/synchronize）")

    layout_ops = [it for it in key_averages if ("nchw" in it.key.lower() or "nhwc" in it.key.lower())]
    if layout_ops:
        print("⚠️  仍然看到布局转换相关 op（建议对比 trace 看是谁触发）:")
        for op in layout_ops[:10]:
            cuda_self = getattr(op, "self_cuda_time_total", 0) or 0
            print(f"    - {op.key}: count={op.count}, self_cuda={cuda_self/1000:.3f}ms")
    else:
        print("✅ 未发现明显 NCHW/NHWC 布局转换 op")

    print("\n" + "=" * 70)
    print("Done")
    print("=" * 70)


def parse_args():
    p = argparse.ArgumentParser(description="PyTorch Profiler - 训练瓶颈分析")
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

    p.add_argument("--tf32", action="store_true", default=True)
    p.add_argument("--no_tf32", action="store_false", dest="tf32")

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
        tf32=args.tf32,
    )