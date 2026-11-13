"""Optional hardware-counter validation and aggregation, with no fabricated metrics."""

import csv
from pathlib import Path

import torch
from torch import nn

from .benchmark import make_workload, step, sync

METRICS = ("dram__bytes_read.sum", "dram__bytes_write.sum")


def capture_step(model, batch=32, warmup=5, seed=1337):
    if not torch.cuda.is_available():
        raise RuntimeError("Nsight capture requires CUDA")
    device = torch.device("cuda")
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    net, optimizer, criterion, x, y = make_workload(model, batch, seed, device, "device")
    for _ in range(warmup):
        step(net, optimizer, criterion, x, y, device, "device")
    sync(device)
    from torch.cuda import profiler
    profiler.start()
    try:
        step(net, optimizer, criterion, x, y, device, "device")
        sync(device)
    finally:
        profiler.stop()
    print(f"Captured one {model} training step at batch={batch}, FP32")


def extract_csv(source, output, env, model, data_mode="device"):
    rows = []
    with Path(source).open(newline="", encoding="utf-8-sig") as f:
        for row in csv.reader(f):
            rows.append(row)
    header_index = next((i for i, r in enumerate(rows) if all(x in r for x in METRICS)), None)
    if header_index is None:
        raise ValueError("CSV does not contain both DRAM read/write metrics. Export with '--page raw --csv --print-units base'.")
    header = rows[header_index]
    ridx, widx = header.index(METRICS[0]), header.index(METRICS[1])
    data = []
    for row in rows[header_index + 1:]:
        if len(row) <= max(ridx, widx):
            continue
        try:
            reads = float(row[ridx].replace(",", ""))
            writes = float(row[widx].replace(",", ""))
            data.append((reads, writes))
        except (ValueError, TypeError):
            continue
    if not data:
        raise ValueError("No hardware counter rows parsed. Check permissions, units and Nsight report content.")
    bytes_total = sum(a + b for a, b in data)
    path = Path(output)
    prior = []
    if path.exists():
        with path.open(newline="", encoding="utf-8") as f:
            prior = list(csv.DictReader(f))
    key = (env, model, data_mode)
    prior = [r for r in prior if (r.get("environment"), r.get("model"), r.get("data_mode", "device")) != key]
    prior.append({"environment": env, "model": model, "data_mode": data_mode,
                  "dram_bytes_per_step": int(bytes_total)})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["environment", "model", "data_mode", "dram_bytes_per_step"])
        writer.writeheader()
        writer.writerows(prior)
    print(f"Parsed {len(data)} kernel launches; total DRAM bytes for ONE captured step: {bytes_total:,.0f}")
    print(f"Updated {path}")
