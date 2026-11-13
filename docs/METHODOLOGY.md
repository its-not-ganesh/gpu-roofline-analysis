# Measurement and modeling methodology

## Workload definitions

Three PyTorch classification networks take tensors with shape `(B, 3, 64, 64)` and produce logits for 10 classes. MLP consists of a flattened image and three linear layers; CNN uses convolution, pooling and a linear classifier; TinyViT uses patch embedding, two attention blocks, token pooling and a classifier. SGD is applied after each cross-entropy backward pass.

No model accuracy or convergence is used as a performance endpoint. Batch size, iteration count, warm-up, numerical precision, optimizer and seed are fixed within a comparison grid.

## Complexity estimates

Parameter count is the sum of trainable parameter elements. Forward and backward FLOPs are estimated with PyTorch's operator formula counter, which may omit unsupported operations. Basic leading-order expressions include:

- Linear: `2 * B * K * N` forward floating-point operations for a `B×K` operand and `K×N` weights, excluding bias/nonlinear operations.
- Convolution: `2 * B * Cout * Hout * Wout * Cin * kh * kw` forward floating-point operations, excluding groups and non-multiply-add operations.
- Attention: projection work scales with `O(B*T*D^2)` and attention score/value products with `O(B*T^2*D)` for batch `B`, sequence length `T`, and embedding size `D`.

Actual backpropagation involves additional operators and backend-dependent algorithms; the PyTorch counter is used instead of applying an arbitrary fixed multiplier.

`LogicalTrafficMode` counts input/output tensor sizes for selected materializing ATen operations. It approximates operator-level logical tensor movement and does **not** measure unique off-chip DRAM transactions, cache misses or bus traffic.

## Timings and telemetry

Training step latency is measured as wall-clock time divided by the number of steps, after warm-up and a CUDA synchronization. CUDA events also time individual enqueued training steps. NVML metrics are sampled by a separate thread at a configured period; process CPU/memory values are sampled with `psutil`. Device-level NVML activity and power measurements are not attributed exclusively to the benchmark when another workload shares the accelerator. Coarse utilization sampling may miss short bursts.

The reported achieved model throughput is:

`GFLOP/s = estimated_training_step_FLOPs / mean_step_time_s / 1e9`

`images/s = batch_size / mean_step_time_s`

## Roofline construction

For calibrated FP32 GEMM rate `C` (GFLOP/s), calibrated device-copy bandwidth `M` (GB/s), and arithmetic intensity `I` (FLOP/byte):

`P_roof(I) = min(C, M * I)` (GFLOP/s)

`I_ridge = C / M` (FLOP/byte)

If no validated Nsight DRAM data is available, plotted intensity is computed as `estimated_step_FLOPs / logical_tensor_bytes`. This proxy denominator is *not* hardware-measured DRAM traffic. If validated Nsight counters are available for the full training step, intensity can instead use `estimated_step_FLOPs / (DRAM_read_bytes + DRAM_write_bytes)`. In either case, the numerator remains a formula-based model estimate, not a hardware instruction count.

The calibrated ceilings are observed microbenchmark rates rather than guaranteed hardware maxima. Practical application points can deviate from these ceilings because the calibration kernel, operator mix, execution path and proxy traffic definition are not identical.

## Cross-environment comparisons

All latency comparisons use the same workload model, input placement, and benchmark configuration. Summary statistics report mean and sample standard deviation across repeats. The baseline speedup is `baseline_mean_step_time / comparison_mean_step_time`, with absent model/environment results left missing rather than imputed. Model and environment are analyzed separately as factors; driver, CUDA, PyTorch, scheduling and thermal differences are observable but not necessarily controlled.
