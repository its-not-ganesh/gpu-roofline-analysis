#!/usr/bin/env bash
set -euo pipefail
if [[ "$#" -ne 1 ]]; then
    echo "Usage: bash scripts/run_linux.sh <environment-label>" >&2
    exit 2
fi
cd "$(dirname "$0")/.."
PYTHON_BIN="${PYTHON_BIN:-python}"
"$PYTHON_BIN" -c 'import torch; assert torch.cuda.is_available(), "CUDA is required"'
"$PYTHON_BIN" -m gpumodel describe
"$PYTHON_BIN" -m gpumodel calibrate --env "$1"
"$PYTHON_BIN" -m gpumodel run --env "$1" --models mlp cnn vit --repeats 3 --steps 120 --warmup 15 --batch-size 32
