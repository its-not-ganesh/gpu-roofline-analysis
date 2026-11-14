#!/usr/bin/env bash
#SBATCH --job-name=gpu-roofline
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
set -euo pipefail
: "${ENV_LABEL:?ENV_LABEL must identify the allocated GPU environment}"
cd "${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is required}"
PYTHON_BIN="${PYTHON_BIN:-$PWD/.venv/bin/python}"
OUT="${RESULTS_DIR:-results_final}"
if [[ ! -x "$PYTHON_BIN" ]]; then echo "Missing Python: $PYTHON_BIN" >&2; exit 2; fi
"$PYTHON_BIN" -c 'import torch; assert torch.cuda.is_available(), "CUDA is required"'
"$PYTHON_BIN" -m gpumodel describe
"$PYTHON_BIN" -m gpumodel calibrate --env "$ENV_LABEL" --output "$OUT"
"$PYTHON_BIN" -m gpumodel run --env "$ENV_LABEL" --output "$OUT" --models mlp cnn vit --repeats 3 --steps 10000 --warmup 100 --batch-size 32 --interval 0.2 --data-mode device
