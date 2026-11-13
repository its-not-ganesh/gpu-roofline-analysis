#!/usr/bin/env bash
#SBATCH --job-name=gpu-roofline
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
set -euo pipefail

: "${ENV_LABEL:?ENV_LABEL must identify the actual allocated GPU environment}"
cd "${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is required}"

if [[ -z "${PYTHON_BIN:-}" ]]; then
    PYTHON_BIN="$PWD/.venv/bin/python"
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Python executable is unavailable: $PYTHON_BIN" >&2
    exit 2
fi

"$PYTHON_BIN" -c 'import torch; assert torch.cuda.is_available(), "CUDA is required"'
"$PYTHON_BIN" -m gpumodel describe
"$PYTHON_BIN" -m gpumodel calibrate --env "$ENV_LABEL"
"$PYTHON_BIN" -m gpumodel run --env "$ENV_LABEL" --models mlp cnn vit --repeats 3 --steps 120 --warmup 15 --batch-size 32
