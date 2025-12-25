#!/usr/bin/env python
"""
PyTorch Profiler 脚本 - 训练瓶颈分析 (v6)

变化:
- 默认 profiling 非 deterministic：cudnn.benchmark=True（更接近真实吞吐）
- 修正“布局转换”检测：只抓 convertTensor / nchwToNhwc / foldedNhwc 等
- 默认 channels_last，但在 --disable_amp(FP32) 时默认关闭 channels_last
  （可用 --force_channels_last 强制开启）
"""
import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.profiler import profile, record_function, ProfilerActivity, schedule

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import CIFAR100Subsampled
from src.augmentations import get_val_transform, build_transform_with_op
from src.models import create_model
from src.utils import get_device


def _set_perf_flags(use_cuda: bool, tf32: bool, deterministic: bool):
    if not use_cuda:
        return
    # TF32
    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.allow_tf32 = tf32

    # Determinism（profiling 通常不需要 deterministic）
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = (not deterministic)
    try:
        torch.use_deterministic_algorithms(deterministic)
    except Exception:
        pass


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
    deterministic: bool = False,
    channels_last: bool = True,
    force_channels_last: bool = False,
):
    device = get_device()
    use_cuda = (device.type == "cuda")

    _set_perf_flags(use_cuda=use_cuda, tf32=tf32, deterministic=deterministic)

    # 关键策略：
    # - BF16 AMP：channels_last 通常更好
    # - FP32（disable_amp）：你现在看到 convertTensor 很重，默认关 channels_last 避免 cuDNN 频繁转换
    use_channels_last = channels_last
    if disable_amp and use_cuda and (not force_channels_last):
        use_channels_last = False

    print("=" * 70)
    print("PyTorch Profiler - 训练瓶颈分析 (v6)")
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
    print(f"Channels last: {'ON' if use_channels_last else 'OFF'}"
          + (" (forced)" if (force_channels_last and disable_amp) else ""))
    print(f"TF32: {'enabled' if (use_cuda and tf32) else 'disabled'}")
    print(f"Deterministic: {'YES' if deterministic else 'NO'}")
    print(f"DataLoader: pin_memory={pin_memory}, persistent_workers={persistent_workers}, "
          f"prefetch_factor={prefetch_factor}, drop_last={drop_last}")
    print("=" * 70)

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
        "pin_memory": (pin_memory if use_cuda else False),
        "drop_last": drop_last,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = persistent_workers
        loader_kwargs["prefetch_factor"] = prefetch_factor

    train_loader = DataLoader(train_dataset, **loader_kwargs)
    print(f"\nDataLoader 配置: {loader_kwargs}")

    model = create_model(num_classes=100, pretrained=False).to(device)
    if use_channels_last:
        model = model.to(memory_format=torch.channels_last)
        print("模型: channels_last (NHWC)")
    else:
        print("模型: NCHW (默认)")

    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=1e-3)

    use_amp = use_cuda and (not disable_amp)
    autocast_dtype = torch.bfloat16
    if use_amp:
        print(f"AMP 已启用: autocast dtype={autocast_dtype}, 无 GradScaler")
    else:
        print("AMP 已禁用" if disable_amp else "AMP 不可用 (非 CUDA)")

    print("\n开始 Profiling...")
    print("-" * 70)

    # warmup（不做 synchronize）
    data_iter = iter(train_loader)
    for _ in range(warmup_batches):
        try:
            images, labels = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            images, labels = next(data_iter)

        if use_channels_last:
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

    prof_wait, prof_warmup, prof_active = 3, 2, num_batches
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
                if use_channels_last:
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

    print("\n" + "=" * 70)
    print("快速检查 (v6 更准)")
    print("=" * 70)
    key_averages = prof.key_averages()

    # 同步检查：item/synchronize
    sync_ops = [
        it for it in key_averages
        if ("aten::item" in it.key.lower())
        or ("cudaDeviceSynchronize" in it.key)
        or ("cudaStreamSynchronize" in it.key)
        or ("synchronize" in it.key.lower())
    ]
    if sync_ops:
        print("⚠️  同步相关 op（少量可能是 profiler 自己做统计）:")
        for op in sync_ops[:10]:
            print(f"    - {op.key}: count={op.count}, cpu_total={op.cpu_time_total/1000:.3f}ms")
    else:
        print("✅ 未发现明显同步相关 op")

    # 布局转换：只抓“真正转换”关键字，别把 NHWC 正常 kernel 当转换
    layout_keys = ("convertTensor", "nchwToNhwc", "nhwcToNchw", "FoldedNhwc", "foldedNhwc")
    layout_ops = [it for it in key_averages if any(k in it.key for k in layout_keys)]
    if layout_ops:
        print("⚠️  检测到真实布局/格式转换 op:")
        for op in layout_ops[:10]:
            cuda_self = getattr(op, "self_cuda_time_total", 0) or 0
            print(f"    - {op.key}: count={op.count}, self_cuda={cuda_self/1000:.3f}ms")
    else:
        print("✅ 未发现 convertTensor / nchw<->nhwc 等真实转换 op")

    print("\n" + "=" * 70)
    print("Done")
    print("=" * 70)


def parse_args():
    p = argparse.ArgumentParser(description="PyTorch Profiler - 训练瓶颈分析 (v6)")
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

    p.add_argument("--deterministic", action="store_true", default=False)

    p.add_argument("--channels_last", action="store_true", default=True)
    p.add_argument("--no_channels_last", action="store_false", dest="channels_last")
    p.add_argument("--force_channels_last", action="store_true", default=False,
                   help="Force channels_last even when --disable_amp (FP32)")

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
        deterministic=args.deterministic,
        channels_last=args.channels_last,
        force_channels_last=args.force_channels_last,
    )