import json
import math
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch import nn
from torch.utils.flop_counter import FlopCounterMode

from .models import IMAGE_SHAPE, MODELS, NUM_CLASSES, make_model
from .telemetry import Monitor, gpu_metadata
from .traffic import LogicalTrafficMode


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str) + "\n", encoding="utf-8")


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def step(model, optimizer, criterion, x, target, device, data_mode):
    if data_mode == "host":
        x = x.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
    optimizer.zero_grad(set_to_none=True)
    loss = criterion(model(x), target)
    loss.backward()
    optimizer.step()
    return loss


def make_workload(name, batch, seed, device, data_mode):
    torch.manual_seed(seed)
    model = make_model(name).to(device)
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    host = torch.device("cpu")
    source = host if data_mode == "host" else device
    x = torch.randn((batch, *IMAGE_SHAPE), device=source)
    y = torch.randint(0, NUM_CLASSES, (batch,), device=source)
    if data_mode == "host" and device.type == "cuda":
        x, y = x.pin_memory(), y.pin_memory()
    return model, optimizer, criterion, x, y


def workload_complexity(model, optimizer, criterion, x, y, device, data_mode):
    with FlopCounterMode(display=False) as mode:
        step(model, optimizer, criterion, x, y, device, data_mode)
    sync(device)
    flops = int(mode.get_total_flops())
    with LogicalTrafficMode(device_type=device.type) as counter:
        step(model, optimizer, criterion, x, y, device, data_mode)
    sync(device)
    return {
        "flops_per_training_step_theoretical": flops,
        "logical_tensor_bytes_per_step_estimate": counter.logical_bytes,
        "logical_operator_count": counter.ops,
        "estimated_arithmetic_intensity_flop_per_logical_byte": (
            flops / counter.logical_bytes if counter.logical_bytes else None
        ),
        "flop_note": "ATen formula-based FLOPs (supported operators only), including backward; not hardware executed FLOPs",
        "traffic_note": "Logical ATen tensor read/write volume proxy; NOT measured DRAM bytes, cache traffic or PCIe transfers",
    }


def trace_step(path, model, optimizer, criterion, x, y, device, data_mode):
    from torch.profiler import ProfilerActivity, profile
    acts = [ProfilerActivity.CPU]
    if device.type == "cuda":
        acts.append(ProfilerActivity.CUDA)
    try:
        with profile(activities=acts, record_shapes=True, profile_memory=True) as prof:
            step(model, optimizer, criterion, x, y, device, data_mode)
            sync(device)
        prof.export_chrome_trace(str(path))
        return True
    except Exception as exc:
        print(f"Trace unavailable: {exc}")
        return False


def run(env, output, models, repeats=3, warmup=15, steps=120, batch=32,
        data_mode="device", interval=0.2, seed=1337, device_name="cuda", trace=False):
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable. Install the official CUDA-enabled PyTorch wheel and verify nvidia-smi.")
    if data_mode == "host" and device_name != "cuda":
        raise ValueError("Host transfer scenario requires a CUDA GPU")
    device = torch.device(device_name)
    # Use strict FP32 GEMM semantics. GPU kernels may still differ across GPU/cuDNN versions.
    torch.set_float32_matmul_precision("highest")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    result_root = Path(output) / env
    save_json(result_root / "hardware.json", gpu_metadata())
    try:
        deps = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, timeout=30)
        if deps.returncode == 0:
            (result_root / "dependencies.txt").write_text(deps.stdout, encoding="utf-8")
    except (OSError, subprocess.TimeoutExpired):
        pass
    for name in models:
        if name not in MODELS:
            raise ValueError(f"Unknown model {name}")
        for repeat in range(repeats):
            model, opt, criterion, x, y = make_workload(name, batch, seed, device, data_mode)
            params = sum(p.numel() for p in model.parameters())
            # Running the model for counting changes its weights, but not its shapes.
            complexity = workload_complexity(model, opt, criterion, x, y, device, data_mode)
            for _ in range(warmup):
                step(model, opt, criterion, x, y, device, data_mode)
            sync(device)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
            events = []
            tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            with Monitor(interval=interval) as monitor:
                t0 = time.perf_counter()
                for i in range(steps):
                    if device.type == "cuda":
                        start = torch.cuda.Event(enable_timing=True)
                        end = torch.cuda.Event(enable_timing=True)
                        start.record()
                    step(model, opt, criterion, x, y, device, data_mode)
                    if device.type == "cuda":
                        end.record()
                        events.append((start, end))
                sync(device)
                wall_s = time.perf_counter() - t0
            event_ms = [a.elapsed_time(b) for a, b in events]
            mean_step_s = wall_s / steps
            telemetry_path = result_root / "telemetry" / f"{name}_{data_mode}_{tag}_r{repeat}.csv"
            monitor.save(telemetry_path)
            telemetry = monitor.summarize()
            result = {
                "environment": env, "model": name, "repeat": repeat,
                "timestamp_utc": tag, "precision": "FP32 (TF32 disabled)",
                "data_mode": data_mode, "batch_size": batch, "image_shape": list(IMAGE_SHAPE),
                "seed": seed, "warmup_steps": warmup, "measured_steps": steps,
                "parameter_count": params, "wall_time_s": wall_s,
                "wall_step_ms": mean_step_s * 1000,
                "gpu_step_ms_median": statistics.median(event_ms) if event_ms else None,
                "gpu_step_ms_p90": sorted(event_ms)[max(0, math.ceil(0.9 * len(event_ms)) - 1)] if event_ms else None,
                "images_per_s": batch / mean_step_s,
                "achieved_theoretical_gflops_per_s": complexity["flops_per_training_step_theoretical"] / mean_step_s / 1e9,
                "gpu_peak_allocated_mb": torch.cuda.max_memory_allocated() / 1e6 if device.type == "cuda" else None,
                "telemetry": telemetry,
                "telemetry_csv": str(telemetry_path),
                "hardware": gpu_metadata(),
                "complexity": complexity,
            }
            save_json(result_root / "runs" / f"{name}_{data_mode}_{tag}_r{repeat}.json", result)
            print(f"{env:20s} {name:4s} repeat={repeat} {result['wall_step_ms']:.3f} ms/step  "
                  f"{result['images_per_s']:.1f} img/s  {result['achieved_theoretical_gflops_per_s']:.1f} model-GFLOP/s")
            if trace and repeat == 0:
                trace_step(result_root / f"trace_{name}_{data_mode}.json", model, opt, criterion, x, y, device, data_mode)
            del model, opt, criterion, x, y
            if device.type == "cuda":
                torch.cuda.empty_cache()


def _gpu_event_times(operation, iterations):
    events = []
    for _ in range(iterations):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        operation()
        b.record()
        events.append((a, b))
    torch.cuda.synchronize()
    # Lowest latency median over steady-state repeats for one consistent algorithm.
    return statistics.median(a.elapsed_time(b) for a, b in events)


def calibrate(env, output, matrix_size=2048, copy_mib=128, iterations=25):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for roofline calibrations")
    device = torch.device("cuda")
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    a = torch.randn((matrix_size, matrix_size), device=device, dtype=torch.float32)
    b = torch.randn_like(a)
    c = torch.empty_like(a)

    def matmul():
        torch.mm(a, b, out=c)

    for _ in range(8):
        matmul()
    torch.cuda.synchronize()
    gemm_ms = _gpu_event_times(matmul, iterations)
    gemm_flops = 2 * matrix_size ** 3
    del a, b, c
    elems = int(copy_mib * 1024**2 / 4)
    src = torch.randn(elems, device=device, dtype=torch.float32)
    dst = torch.empty_like(src)

    def copy():
        dst.copy_(src)

    for _ in range(8):
        copy()
    torch.cuda.synchronize()
    copy_ms = _gpu_event_times(copy, iterations)
    mem_bytes = elems * 4 * 2
    result = {
        "environment": env, "hardware": gpu_metadata(),
        "precision": "FP32 (TF32 disabled)",
        "empirical_compute_ceiling_gflops_s": gemm_flops / (gemm_ms / 1e3) / 1e9,
        "empirical_bandwidth_ceiling_gbytes_s": mem_bytes / (copy_ms / 1e3) / 1e9,
        "matmul_size": matrix_size, "copy_buffer_mib_each": copy_mib,
        "matmul_median_ms": gemm_ms, "copy_median_ms": copy_ms,
        "iterations": iterations,
        "note": "Empirical, single-GPU FP32 GEMM and GPU-device copy ceilings; not vendor peaks or guaranteed DRAM bandwidth. Copy buffers should exceed GPU last-level cache.",
    }
    save_json(Path(output) / env / "calibration.json", result)
    print(json.dumps(result, indent=2, default=str))
