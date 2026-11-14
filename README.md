# GPU Training Performance and Roofline Analysis

A reproducible, single-GPU PyTorch benchmark for characterizing training performance across NVIDIA GPU systems. The same MLP, CNN, and vision-transformer workloads run under a fixed experimental configuration. The framework records synchronized timing, CPU/GPU telemetry, workload-complexity estimates, calibrated compute/bandwidth limits, and comparative roofline visualizations.

## Experimental design

| Variable | Baseline setting |
|---|---|
| Training models | `mlp`, `cnn`, `vit` (see `src/gpumodel/models.py`) |
| Input | Synthetic image tensors, `3 Ã— 64 Ã— 64`; 10 synthetic classes |
| Batch size | 32 |
| Numeric precision | FP32; TF32 disabled |
| Optimizer and objective | SGD, cross-entropy |
| Timing | 100 warm-up steps, 10,000 measured steps, 3 independent repeats |
| Hardware | One NVIDIA CUDA GPU per run |
| Input placement | GPU-resident by default (`--data-mode device`) |
| Output | Per-trial JSON and CSV telemetry; aggregated CSV and PNG figures |

The experiment isolates architecture and execution environment rather than model convergence. Each environment uses identical source code, tensor shapes, and measurement settings. Workload complexity is computed at training-step scope, including the supported forward and backward operations. Design and metric definitions appear in [`docs/EXPERIMENT_DESIGN.md`](docs/EXPERIMENT_DESIGN.md) and [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md).

## Software requirements

- Python 3.10+ with a compatible, **CUDA-enabled** PyTorch build.
- NVIDIA driver supporting the installed CUDA PyTorch runtime and the target GPU.
- `git` for version tracking and reproduction.
- Dependencies installed from [`pyproject.toml`](pyproject.toml).
- Optional: NVIDIA Nsight Compute for per-kernel DRAM counters; administrator-enabled GPU performance-counter access may be required.

CPU execution is supported for smoke tests (`--device cpu`) but does not produce GPU performance comparisons or device calibration. No CUDA benchmark silently falls back to CPU.

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
# Install CUDA-enabled torch using https://pytorch.org/get-started/locally/
python -m pip install -e .
python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
.\scripts\run_local_windows.ps1 -Environment MACHINE_LABEL
```

### Linux or a GPU VM

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# Install CUDA-enabled torch using https://pytorch.org/get-started/locally/
python -m pip install -e .
python -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'
bash scripts/run_linux.sh MACHINE_LABEL
```

Replace `MACHINE_LABEL` with a unique filesystem-safe identifier. Run commands at the repository root. On managed clusters, use a provided Python environment/module if `venv` is unavailable and follow the site's scheduler policy. Virtual environments, GPU drivers, scheduler accounts, and allocation credentials are intentionally not committed to source control.

### Slurm cluster

The batch script runs on the GPU compute node selected by the site scheduler. It expects a Python environment at `.venv/bin/python` relative to the repository, or `PYTHON_BIN` set to an absolute Python interpreter with the project and CUDA-enabled torch installed. The environment label is mandatory to avoid misattributing GPU measurements.

```bash
sbatch --account=ACCOUNT --partition=GPU_PARTITION \
  --export=ALL,ENV_LABEL=CLUSTER_GPU \
  scripts/hpc_run.sbatch
```

Partition, account, resource and wall-time settings depend on the HPC scheduler and authorized allocation. `SLURM_SUBMIT_DIR` determines the repository location. If the partition does not automatically assign a GPU, include the site's required GPU resource request. No privileged host configuration is part of the benchmark.

## Command-line interface

```bash
python -m gpumodel describe
python -m gpumodel calibrate --env MACHINE_LABEL --output results_final
python -m gpumodel run --env MACHINE_LABEL --output results_final --models mlp cnn vit \
  --repeats 3 --warmup 100 --steps 10000 --batch-size 32 --interval 0.2 --data-mode device
python -m gpumodel report --input results_final --output figures_final --baseline MACHINE_LABEL
```

`describe` emits hardware and software metadata. `calibrate` times FP32 matrix multiplication and device-memory copying. `run` executes three training models, saves a record for each repeat, and samples GPU/CPU activity asynchronously. `report` aggregates any environment directories present under `results_final/`, computes comparisons, and creates per-GPU rooflines where calibration is available.

The platform scripts execute `describe`, `calibrate`, and the default `run` configuration. All commands can also be called individually. For a small CPU test of the application path:

```bash
python -m gpumodel run --env cpu-check --device cpu --models mlp --batch-size 2 --steps 2 --warmup 1 --repeats 1
```

### Aggregating multi-machine results

The run directory is portable:

```text
results_final/
  MACHINE_A/
    hardware.json
    calibration.json
    dependencies.txt
    runs/*.json
    telemetry/*.csv
  MACHINE_B/
    ...
```

Combine each machine's `results_final/MACHINE_LABEL` folder under a common `results_final/` root, preserving its contents. Run the `report` command in that copy of the repository. Generated outputs include `figures_final/summary.csv`, latency and comparison charts, telemetry traces, and `figures_final/roofline_<environment>_logical_tensor.png` for environments with calibration data. Keep generated artifacts outside version control. Plotting rejects mixed batch, seed, warmup, step count, precision, or source revisions within a data mode. Keep short smoke tests in a separate results root.

For a second Windows host, use the same source revision and workload settings, with a fresh Python 3.11 environment and a [CUDA PyTorch build](https://pytorch.org/get-started/locally/) supporting the installed GPU. Record the adapter and vendor power profile before the run. Use `laptop-rtx50` as the environment label and check the actual device reported by `describe`.

### Optional GPU DRAM profiling (Nsight Compute)

Nsight Compute is a separate, short, **one-training-step** counter capture and is never used to time the sustained benchmark. It requires installed `ncu` and GPU performance-counter permission. The CLI brackets its single measured step using `cudaProfilerStart` and `cudaProfilerStop`; this is triggered by `--profile-from-start off`.

Windows PowerShell:

```powershell
.\scripts\profile_ncu_windows.ps1 -Environment MACHINE_LABEL -Output results_final
```

Linux (run only in a GPU allocation with `ncu` available):

```bash
bash scripts/profile_ncu_linux.sh MACHINE_LABEL results_final
```

The Windows script uses kernel replay, leaves cache flushing disabled, and uses 100 warmup steps at batch 32 and seed 1337. Capture into a fresh destination. It records Nsight version, GPU UUID, kernel count and separate totals for read and write bytes. A blocked or incomplete capture leaves the final counter table unavailable. Each successful script produces `results_final/MACHINE_LABEL/nsight/{mlp,cnn,vit}_step.ncu-rep`, raw CSV, and `measured_dram.csv`. Verify that each raw capture contains the kernels for **exactly one complete training step**, not warm-up kernels or partial steps, before using the numbers as per-step traffic. Nsight counter availability depends on GPU, tool version, permissions, and workload. On managed clusters, unavailable counters should be recorded as missing rather than estimated.

To build rooflines using validated measurements, use the machine-specific counter table alongside its raw exports and run:

```bash
python -m gpumodel report --input results_final --output figures_measured \
  --baseline MACHINE_LABEL --dram-csv results_final/MACHINE_LABEL/nsight/measured_dram.csv
```

When the DRAM CSV is not supplied, the framework uses a logical-tensor traffic proxy and labels it accordingly. Plot performance uses estimated operator FLOPs divided by measured end-to-end training step time; it is not a hardware-executed FLOP metric.

### Reproducibility data archive

```bash
python scripts/bundle_results.py --input results_final --env MACHINE_LABEL --figures figures_final --output experiment_data_bundle.zip
```

Point `--figures` at plots for the selected machine. The archive includes JSON runs, per-run telemetry, calibration, dependency versions, summary CSV, figures, and optional Nsight raw CSV and report files. It does not include local virtual environments or source-control credentials. Generated measurements are kept outside version control by default.

## Measurement provenance and limitations

| Quantity | Source | Interpretation |
|---|---|---|
| Training step latency | Synchronized host timer | Measured end-to-end step interval |
| GPU operation latency | CUDA events | Measured device timing (distinct from wall time) |
| Device activity, power, temperature, VRAM, clocks | NVML sampled by project code | Measured samples; activity percentages are **not** bytes/s |
| CPU utilization and resident memory | `psutil` | Measured process statistics |
| Training FLOPs per step | PyTorch `FlopCounterMode` | Formula-based count for supported ATen ops, not executed hardware FLOPs |
| Logical tensor traffic | `LogicalTrafficMode` | Estimated read/write bytes at operator boundaries; not DRAM bytes |
| FP32 compute and copy ceilings | Project GPU microbenchmarks | Empirical calibration, not vendor peak specifications |
| GPU DRAM traffic (optional) | Nsight Compute | Hardware counters when correctly collected |

The roofline calculation is `min(compute_ceiling, arithmetic_intensity Ã— bandwidth_ceiling)`. Proxy-based points must not be interpreted as measured hardware DRAM arithmetic intensity. The copy microbenchmark provides a practical bandwidth reference, not an exact hardware limit. CUDA libraries, GPU clocks, thermal state, background workload, and counter availability may affect comparisons. Synthetic GPU-resident input deliberately excludes file I/O, decoding, and dataset-loader effects.

## Verification and source layout

```text
src/gpumodel/          Workloads, telemetry, complexity, calibration, plotting, CLI
scripts/               Windows, Linux and Slurm launch scripts
docs/EXPERIMENT_DESIGN.md   Research question, hypotheses, fixed comparison grid
docs/METHODOLOGY.md    Metrics and mathematical assumptions
docs/VALIDATION.md     Verified behavior and untested hardware paths
tests/                 Model and traffic estimator smoke tests
```

Test command:

```bash
python -m pip install pytest
python -m pytest -q
```

The project records observed results only; no hardware performance figures are included without an executed benchmark. See [`docs/VALIDATION.md`](docs/VALIDATION.md) for the tested execution scope.
