"""Optional hardware-counter validation and aggregation, with no fabricated metrics."""

import csv
from decimal import Decimal, InvalidOperation
from pathlib import Path
from tempfile import NamedTemporaryFile

METRICS = ("dram__bytes_read.sum", "dram__bytes_write.sum")


def capture_step(model, batch=32, warmup=100, seed=1337):
    import torch

    from .benchmark import make_workload, step, sync

    if batch < 1 or warmup < 1:
        raise ValueError("Capture requires positive batch size and warmup")
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


DRAM_FIELDS = (
    "environment", "model", "data_mode", "batch_size", "warmup_steps", "seed",
    "captured_steps", "kernel_count", "gpu_id", "profiler_version", "raw_csv",
    "dram_read_bytes_per_step", "dram_write_bytes_per_step", "dram_bytes_per_step",
)


def read_raw_counters(source):
    """Read base-byte counters from every detail row in a single Nsight export."""
    with Path(source).open(newline="", encoding="utf-8-sig") as stream:
        rows = csv.reader(stream)
        required = ("ID", "Kernel Name", *METRICS)
        header = next((row for row in rows if all(name in row for name in required)), None)
        if header is None:
            raise ValueError("Missing Nsight raw per-launch header")
        id_col, kernel_col, read_col, write_col = map(header.index, required)
        units = next(rows, [])
        if len(units) != len(header) or any(
            units[col].strip().lower() not in {"byte", "bytes"} for col in (read_col, write_col)
        ):
            raise ValueError("DRAM counters must use base bytes")

        launches = {}
        for row in rows:
            if not any(row):
                continue
            if row[0].lstrip().startswith("=="):
                if "ERROR" in row[0]:
                    raise ValueError(row[0])
                continue
            if row[0].strip().lower() in {"summary", "total", "average"}:
                continue
            try:
                if len(row) != len(header) or not row[kernel_col].strip():
                    raise ValueError
                launch_id = int(row[id_col])
                if launch_id < 0 or launch_id in launches:
                    raise ValueError
                counts = [Decimal(row[col].replace(",", "")) for col in (read_col, write_col)]
                if any(not n.is_finite() or n < 0 or n != n.to_integral_value() for n in counts):
                    raise ValueError
            except (InvalidOperation, ValueError) as exc:
                raise ValueError(f"Invalid or incomplete kernel row at line {rows.line_num}") from exc
            launches[launch_id] = tuple(map(int, counts))

    if not launches or set(launches) != set(range(len(launches))):
        raise ValueError("Missing or noncontiguous kernel-launch IDs")
    read_bytes = sum(pair[0] for pair in launches.values())
    write_bytes = sum(pair[1] for pair in launches.values())
    if min(read_bytes, write_bytes) <= 0:
        raise ValueError("Both DRAM directions must have positive step totals")
    return len(launches), read_bytes, write_bytes


def extract_csv(source, output, env, model, data_mode="device", *, batch=32,
                warmup=100, seed=1337, gpu_id="", profiler_version=""):
    """Record counters for a caller-verified, complete single-step capture."""
    kernels, reads, writes = read_raw_counters(source)
    path = Path(output)
    old_rows = []
    if path.exists():
        with path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            if tuple(reader.fieldnames or ()) != DRAM_FIELDS:
                raise ValueError("Use a fresh output for the per-direction counter schema")
            old_rows = list(reader)
    key = (env, model, data_mode)
    old_rows = [r for r in old_rows if (r["environment"], r["model"], r["data_mode"]) != key]
    old_rows.append(dict(zip(DRAM_FIELDS, (
        env, model, data_mode, batch, warmup, seed, 1, kernels, gpu_id, profiler_version,
        Path(source).name, reads, writes, reads + writes,
    ))))
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(mode="w", newline="", encoding="utf-8", dir=path.parent,
                            delete=False) as stream:
        temp = Path(stream.name)
        writer = csv.DictWriter(stream, fieldnames=DRAM_FIELDS)
        writer.writeheader()
        writer.writerows(old_rows)
    try:
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)
    print(f"Parsed {kernels} kernel launches: {reads:,} read bytes, {writes:,} write bytes")
