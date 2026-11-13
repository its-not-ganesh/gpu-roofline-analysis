# Validation scope

The following behavior was checked in a Linux CPU environment during development:

- Package import and command-line help.
- Forward and backward passes for `mlp`, `cnn`, and `vit`.
- Formula-based FLOP counting and logical ATen traffic instrumentation.
- CPU-only benchmark execution using small synthetic inputs.
- Report aggregation and figure generation from CPU smoke-run data.
- Nsight CSV parsing against synthetic format fixtures (parser verification only; **not** benchmark results).

The following require execution on the target GPU machines and were not exercised in the development environment:

- CUDA event and NVML sampling paths.
- Device-memory copy and FP32 GEMM calibration.
- NVIDIA Nsight Compute hardware counter permissions and coverage.
- Slurm job scheduling, accelerator allocation and GPU VM provisioning.

No real GPU throughput values were measured during development. Hardware results, if produced, are stored separately under `results/` on the execution machines.
