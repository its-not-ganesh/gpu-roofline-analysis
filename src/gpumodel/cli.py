import argparse
import json
from pathlib import Path

from .benchmark import calibrate, run
from .telemetry import gpu_metadata


def main():
    ap = argparse.ArgumentParser(description="GPU performance characterization experiment")
    sub = ap.add_subparsers(dest="command", required=True)
    sub.add_parser("describe", help="Print machine hardware and software metadata")
    r = sub.add_parser("run", help="Run model training trials with custom telemetry")
    r.add_argument("--env", required=True, help="Unique machine label, e.g. desktop-4070")
    r.add_argument("--output", default="results")
    r.add_argument("--models", nargs="+", choices=["mlp", "cnn", "vit"], default=["mlp", "cnn", "vit"])
    r.add_argument("--repeats", type=int, default=3)
    r.add_argument("--warmup", type=int, default=15)
    r.add_argument("--steps", type=int, default=120)
    r.add_argument("--batch-size", type=int, default=32)
    r.add_argument("--data-mode", choices=["device", "host"], default="device")
    r.add_argument("--interval", type=float, default=0.2)
    r.add_argument("--seed", type=int, default=1337)
    r.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    r.add_argument("--trace", action="store_true", help="Also collect one PyTorch operator trace per model")
    c = sub.add_parser("calibrate", help="Run FP32 GEMM + device-copy calibration")
    c.add_argument("--env", required=True)
    c.add_argument("--output", default="results")
    c.add_argument("--matrix-size", type=int, default=2048)
    c.add_argument("--copy-mib", type=int, default=128)
    c.add_argument("--iterations", type=int, default=25)
    z = sub.add_parser("capture-step", help="One CUDA-profiler API-delimited step for optional Nsight counters")
    z.add_argument("--model", required=True, choices=["mlp", "cnn", "vit"])
    z.add_argument("--batch-size", type=int, default=32)
    z.add_argument("--warmup", type=int, default=5)
    z.add_argument("--seed", type=int, default=1337)
    e = sub.add_parser("extract-dram", help="Sum Nsight Compute kernel DRAM bytes for exactly one captured step")
    e.add_argument("--input", required=True, help="raw CSV exported from Nsight Compute")
    e.add_argument("--output", default="measured_dram.csv")
    e.add_argument("--env", required=True)
    e.add_argument("--model", required=True, choices=["mlp", "cnn", "vit"])
    e.add_argument("--data-mode", default="device", choices=["device", "host"])
    p = sub.add_parser("report", help="Plot comparison, telemetry and rooflines from multiple environment folders")
    p.add_argument("--input", default="results")
    p.add_argument("--output", default="figures")
    p.add_argument("--baseline", default="desktop-4070")
    p.add_argument("--dram-csv", help="Optional measured DRAM bytes per step: environment,model,dram_bytes_per_step")
    args = ap.parse_args()
    if args.command == "describe":
        print(json.dumps(gpu_metadata(), indent=2, default=str))
    elif args.command == "run":
        if args.steps < 1 or args.repeats < 1 or args.warmup < 0 or args.batch_size < 1 or args.interval <= 0:
            ap.error("Steps, repeats, batch and interval must be positive; warmup may be zero")
        run(args.env, args.output, args.models, args.repeats, args.warmup,
            args.steps, args.batch_size, args.data_mode, args.interval, args.seed, args.device, args.trace)
    elif args.command == "calibrate":
        calibrate(args.env, args.output, args.matrix_size, args.copy_mib, args.iterations)
    elif args.command == "capture-step":
        from .ncu import capture_step
        capture_step(args.model, args.batch_size, args.warmup, args.seed)
    elif args.command == "extract-dram":
        from .ncu import extract_csv
        extract_csv(args.input, args.output, args.env, args.model, args.data_mode)
    elif args.command == "report":
        from .report import generate
        generate(args.input, args.output, args.baseline, args.dram_csv)


if __name__ == "__main__":
    main()
