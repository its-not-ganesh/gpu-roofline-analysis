#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo "Usage: bash scripts/run_linux.sh <environment-label> [output-directory]" >&2
    exit 2
fi
cd "$(dirname "$0")/.."
PYTHON_BIN="${PYTHON_BIN:-python}"
OUT="${2:-results_final}"
"$PYTHON_BIN" -c 'import torch; assert torch.cuda.is_available(), "CUDA is required"'
"$PYTHON_BIN" -m gpumodel describe
"$PYTHON_BIN" -m gpumodel calibrate --env "$1" --output "$OUT"
"$PYTHON_BIN" -m gpumodel run --env "$1" --output "$OUT" --models mlp cnn vit --repeats 3 --steps 10000 --warmup 100 --batch-size 32 --interval 0.2 --data-mode device
