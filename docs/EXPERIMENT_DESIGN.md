# Experiment design

## Research question

How do neural-network topology and GPU execution environment influence end-to-end training throughput, accelerator utilization, memory behavior and performance relative to an empirically calibrated roofline?

## Hypotheses

- **H1 — Architecture:** An MLP, a CNN and a small vision transformer exhibit different computational intensities and GPU efficiencies due to their operator composition and memory requirements.
- **H2 — Hardware:** Training speedups between GPU environments vary with architecture because compute throughput, memory bandwidth, submission overhead and power/clock constraints differ.
- **H3 — Host-to-device input (secondary comparison):** Repeated host-to-device input transfer changes total step latency relative to a GPU-resident input baseline.

Hypotheses are evaluated against collected benchmark data and are not assumed to hold a priori.

## Experimental grid

The primary matrix consists of three workloads (`mlp`, `cnn`, `vit`) on two or three GPU environments. A baseline observation uses one accelerator, fixed model definitions, synthetic `3 × 64 × 64` inputs, batch size 32, SGD, cross-entropy, FP32, TF32 disabled, 15 warm-up iterations, 120 timed training steps and three independent repetitions. The same code and configuration are executed on every machine. Differences in GPU type and software environment are captured in machine metadata.

The environment label is an identifier, not an assertion about cloud versus on-premises location. Execution substrate (local workstation, cloud VM, or HPC-managed VM/node) is described using independently verified deployment facts.

Optional host-input experiments are analyzed as a distinct `data_mode` and are not combined into baseline/device-resident comparisons. Synthetic GPU-resident inputs isolate the compute path rather than end-to-end dataset ingestion.

## Experimental outputs

Each measured repeat stores elapsed time, throughput, theoretical FLOP work, estimated logical tensor traffic, CUDA timing statistics, memory allocation and a raw device/CPU telemetry CSV. Independent calibration records GPU copy bandwidth and FP32 matrix-product throughput. Per-environment roofline plots and cross-environment comparisons are generated from the collected records.

A fixed random seed governs model initialization and synthetic inputs. Library kernel selection, scheduling, background utilization, clock policies and versions may differ across machines; the captured hardware and dependency metadata preserve relevant context.
