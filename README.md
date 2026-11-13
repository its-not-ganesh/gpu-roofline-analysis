# GPU Training Performance and Roofline Analysis

A reproducible, single-GPU PyTorch benchmark for characterizing training performance across NVIDIA GPU systems. The same MLP, CNN, and vision-transformer workloads run under a fixed experimental configuration. The framework records synchronized timing, CPU/GPU telemetry, workload-complexity estimates, calibrated compute/bandwidth limits, and comparative roofline visualizations.

## Experimental design

| Variable | Baseline setting |
|---|---|
| Training models | `mlp`, `cnn`, `vit` (see `src/gpumodel/models.py`) |
| Input | Synthetic image tensors, `3 × 64 × 64`; 10 synthetic classes |
| Batch size | 32 |
| Numeric precision | FP32; TF32 disabled |
| Optimizer and objective | SGD, cross-entropy |
| Timing | 15 warm-up steps, 120 measured steps, 3 independent repeats |
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
py -3.11 -m venv .venv
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
python -m gpumodel calibrate --env MACHINE_LABEL
python -m gpumodel run --env MACHINE_LABEL --models mlp cnn vit \
  --repeats 3 --warmup 15 --steps 120 --batch-size 32
python -m gpumodel report --input results --output figures --baseline MACHINE_LABEL
```

`describe` emits hardware and software metadata. `calibrate` times FP32 matrix multiplication and device-memory copying. `run` executes three training models, saves a record for each repeat, and samples GPU/CPU activity asynchronously. `report` aggregates any environment directories present under `results/`, computes comparisons, and creates per-GPU rooflines where calibration is available.

The platform scripts execute `describe`, `calibrate`, and the default `run` configuration. All commands can also be called individually. For a small CPU test of the application path:

```bash
python -m gpumodel run --env cpu-check --device cpu --models mlp --batch-size 2 --steps 2 --warmup 1 --repeats 1
```

### Aggregating multi-machine results

The run directory is portable:

```text
results/
  MACHINE_A/
    hardware.json
    calibration.json
    dependencies.txt
    runs/*.json
    telemetry/*.csv
  MACHINE_B/
    ...
```

Combine each machine's `results/MACHINE_LABEL` folder under a common `results/` root, preserving its contents. Run the `report` command in that copy of the repository. Generated outputs include `figures/summary.csv`, latency/speedup charts, telemetry traces, and `figures/roofline_<environment>.png` for environments with calibration data. Raw generated data and figures are excluded from Git by default.

### Optional measured DRAM-byte capture

The default arithmetic intensity is a **logical-tensor traffic proxy**. Actual GPU DRAM-byte counters, if available, may be collected separately with Nsight Compute and provided to the project plotting pipeline. Counter capture is not part of the synchronized timing benchmark:

```bash
ncu --profile-from-start off \
  --metrics dram__bytes_read.sum,dram__bytes_write.sum \
  -o step_profile python -m gpumodel capture-step --model cnn --batch-size 32
ncu --import step_profile.ncu-rep --page raw --csv --print-units base > step_profile.csv
python -m gpumodel extract-dram --input step_profile.csv \
  --env MACHINE_LABEL --model cnn --output measured_dram.csv
python -m gpumodel report --input results --output figures \
  --baseline MACHINE_LABEL --dram-csv measured_dram.csv
```

Counters and command-line options can differ by Nsight Compute version and GPU; verify the captured metrics and full step scope before interpreting DRAM arithmetic intensity. The `extract-dram` command expects CSV columns named `dram__bytes_read.sum` and `dram__bytes_write.sum`. Hardware counter access may be unavailable on managed systems.

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

The roofline calculation is `min(compute_ceiling, arithmetic_intensity × bandwidth_ceiling)`. Proxy-based points must not be interpreted as measured hardware DRAM arithmetic intensity. The copy microbenchmark provides a practical bandwidth reference, not an exact hardware limit. CUDA libraries, GPU clocks, thermal state, background workload, and counter availability may affect comparisons. Synthetic GPU-resident input deliberately excludes file I/O, decoding, and dataset-loader effects.

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
