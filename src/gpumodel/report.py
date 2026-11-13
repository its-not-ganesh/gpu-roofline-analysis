import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def measured_bytes(path):
    if not path:
        return {}
    with open(path, newline="", encoding="utf-8") as f:
        data = list(csv.DictReader(f))
    ret = {}
    for row in data:
        key = (row["environment"], row["model"], row.get("data_mode", "device"))
        ret[key] = float(row["dram_bytes_per_step"])
    return ret


def _telemetry_chart(source, destination, title):
    if not source.exists():
        return
    with source.open(newline="", encoding="utf-8") as f:
        data = list(csv.DictReader(f))
    if not data:
        return
    time0 = float(data[0]["seconds"])
    t = [float(r["seconds"]) - time0 for r in data]
    fig, ax = plt.subplots(figsize=(10, 4.4))
    shown = False
    for field, label in (("gpu_util_pct", "GPU engine busy (%)"),
                         ("gpu_mem_controller_util_pct", "Memory-controller busy (%)")):
        ys = [float(r[field]) if r.get(field) not in (None, "") else math.nan for r in data]
        if any(math.isfinite(y) for y in ys):
            ax.plot(t, ys, label=label)
            shown = True
    ax.set_xlabel("Elapsed wall-clock time (s)")
    ax.set_ylabel("Utilization (%)")
    ax.set_ylim(0, 105)
    ax.grid(alpha=0.3)
    second = ax.twinx()
    powers = [float(r["power_w"]) if r.get("power_w") not in (None, "") else math.nan for r in data]
    if any(math.isfinite(y) for y in powers):
        second.plot(t, powers, linestyle="--", color="tab:gray", label="GPU power (W)")
    second.set_ylabel("Board power (W)")
    ax.set_title(title + " — independently sampled NVML activity")
    if shown:
        ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(destination, dpi=160)
    plt.close(fig)


def generate(input_dir="results", output_dir="figures", baseline="desktop-4070", dram_csv=None):
    root = Path(input_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    runs = [load_json(p) for p in sorted(root.glob("*/runs/*.json"))]
    if not runs:
        raise ValueError(f"No run files found under {root}/*/runs/ . Copy all result folders here first.")
    calibration = {p.parent.name: load_json(p) for p in root.glob("*/calibration.json")}
    actual_bytes = measured_bytes(dram_csv)
    by_key = defaultdict(list)
    for result in runs:
        by_key[(result["environment"], result["model"], result["data_mode"])].append(result)
    rows = []
    for (env, name, data_mode), items in sorted(by_key.items()):
        def avg(field):
            return statistics.mean(float(r[field]) for r in items)
        ms = [float(r["wall_step_ms"]) for r in items]
        c = items[0]["complexity"]
        flop_count = float(c["flops_per_training_step_theoretical"])
        traffic_proxy = float(c["logical_tensor_bytes_per_step_estimate"])
        dram = actual_bytes.get((env, name, data_mode))
        bytes_per_step = dram if dram is not None else traffic_proxy
        labels = "measured_dram" if dram is not None else "logical_tensor_proxy"
        # Dividing the mean flop count by mean wall time is comparable across repeats.
        rows.append({
            "environment": env, "model": name, "data_mode": data_mode,
            "repeat_count": len(items), "batch_size": items[0]["batch_size"],
            "parameters": items[0]["parameter_count"],
            "flops_per_step_estimate": int(flop_count),
            "bytes_per_step": bytes_per_step, "arithmetic_intensity_source": labels,
            "arithmetic_intensity_flop_per_byte": flop_count / bytes_per_step if bytes_per_step else math.nan,
            "step_ms_mean": statistics.mean(ms),
            "step_ms_stddev": statistics.stdev(ms) if len(ms) > 1 else 0,
            "images_per_second": avg("images_per_s"),
            "achieved_model_gflops_per_s": flop_count / (statistics.mean(ms) * 1e-3) / 1e9,
            "gpu_util_pct_mean": statistics.mean(float(r["telemetry"]["gpu_util_pct_mean"])
                      for r in items if r["telemetry"]["gpu_util_pct_mean"] is not None)
                      if any(r["telemetry"]["gpu_util_pct_mean"] is not None for r in items) else "",
            "power_w_mean": statistics.mean(float(r["telemetry"]["power_w_mean"])
                      for r in items if r["telemetry"]["power_w_mean"] is not None)
                      if any(r["telemetry"]["power_w_mean"] is not None for r in items) else "",
            "gpu_peak_allocated_mb": max(float(r["gpu_peak_allocated_mb"] or 0) for r in items),
        })
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    # Multiple environments and workloads share one baseline, without inventing missing comparisons.
    for data_mode in sorted(set(r["data_mode"] for r in rows)):
        subset = [r for r in rows if r["data_mode"] == data_mode]
        models = sorted({r["model"] for r in subset})
        envs = sorted({r["environment"] for r in subset})
        x = np.arange(len(models))
        width = 0.8 / len(envs)
        fig, ax = plt.subplots(figsize=(10, 5))
        for i, env in enumerate(envs):
            lookup = {r["model"]: r for r in subset if r["environment"] == env}
            vals = [lookup[m]["step_ms_mean"] if m in lookup else math.nan for m in models]
            errs = [lookup[m]["step_ms_stddev"] if m in lookup else 0 for m in models]
            ax.bar(x - 0.4 + width * (i + 0.5), vals, width, yerr=errs, capsize=2, label=env)
        ax.set_xticks(x, models)
        ax.set_ylabel("Synchronized end-to-end training step (ms), lower is better")
        ax.set_title(f"Identical PyTorch workloads by environment — {data_mode} input")
        ax.grid(axis="y", alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output / f"latency_{data_mode}.png", dpi=160)
        plt.close(fig)
        base_lookup = {r["model"]: r for r in subset if r["environment"] == baseline}
        if base_lookup:
            fig, ax = plt.subplots(figsize=(10, 5))
            for i, env in enumerate(envs):
                lookup = {r["model"]: r for r in subset if r["environment"] == env}
                speedup = [(base_lookup[m]["step_ms_mean"] / lookup[m]["step_ms_mean"]
                            if m in base_lookup and m in lookup else math.nan) for m in models]
                ax.bar(x - 0.4 + width * (i + 0.5), speedup, width, label=env)
            ax.axhline(1, color="black", linewidth=1)
            ax.set_xticks(x, models)
            ax.set_ylabel(f"Speedup relative to {baseline} (×)")
            ax.set_title("End-to-end training throughput comparison")
            ax.legend()
            fig.tight_layout()
            fig.savefig(output / f"speedup_{data_mode}.png", dpi=160)
            plt.close(fig)

    for env, cal in sorted(calibration.items()):
        subset = [r for r in rows if r["environment"] == env]
        if not subset:
            continue
        compute = float(cal["empirical_compute_ceiling_gflops_s"])
        bw = float(cal["empirical_bandwidth_ceiling_gbytes_s"])
        intensities = [float(r["arithmetic_intensity_flop_per_byte"]) for r in subset]
        if not all(x > 0 for x in intensities):
            continue
        ridge = compute / bw
        lower = max(1e-3, min(min(intensities) / 4, ridge / 10))
        upper = max(max(intensities) * 4, ridge * 10)
        x = np.logspace(math.log10(lower), math.log10(upper), 240)
        y = np.minimum(bw * x, compute)
        fig, ax = plt.subplots(figsize=(9, 5.5))
        ax.loglog(x, y, linewidth=2, label="Empirical ceiling: min(BW × AI, GEMM rate)")
        for r in subset:
            color = None
            ax.scatter(r["arithmetic_intensity_flop_per_byte"], r["achieved_model_gflops_per_s"], s=85,
                       label=f"{r['model']} / {r['data_mode']} ({r['arithmetic_intensity_source']})", color=color)
        ax.axvline(ridge, linestyle="--", color="gray", alpha=0.7, label=f"Ridge ≈ {ridge:.1f} FLOP/B")
        ax.set_xlabel("Arithmetic intensity (FLOP/byte)")
        ax.set_ylabel("Estimated model FLOPs / measured training time (GFLOP/s)")
        ax.set_title(f"Empirical roofline — {env} (FP32, device-copy calibration)")
        ax.legend(fontsize=8)
        ax.grid(True, which="both", alpha=0.25)
        fig.tight_layout()
        fig.savefig(output / f"roofline_{env}.png", dpi=160)
        plt.close(fig)

    for (env, name, mode), items in by_key.items():
        source = Path(items[0]["telemetry_csv"])
        if not source.exists():
            # Run folders may have been copied to a different machine/path.
            source = root / env / "telemetry" / source.name
        _telemetry_chart(source, output / f"telemetry_{env}_{name}_{mode}.png",
                         f"{env} — {name} / {mode}")
    count = len(runs)
    missing_cal = sorted(set(r["environment"] for r in rows) - set(calibration))
    print(f"Analyzed {count} runs / {len(rows)} model-environment-mode cells; output: {output}")
    if missing_cal:
        print("WARNING: No calibration for " + ", ".join(missing_cal) + "; rooflines omitted for these environments.")
    if not actual_bytes:
        print("NOTE: Arithmetic intensity uses LOGICAL TENSOR BYTE PROXIES, not measured GPU DRAM bytes.")
        print("Supply --dram-csv with measured per-step DRAM totals for a hardware-traffic-based roofline.")
