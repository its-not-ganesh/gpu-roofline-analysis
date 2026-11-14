#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: bash scripts/profile_ncu_linux.sh ENV_LABEL [OUTPUT_DIRECTORY]" >&2
  exit 2
fi
cd "$(dirname "$0")/.."
ENV_LABEL="$1"
OUT="${2:-results_final}"
PYTHON_BIN="${PYTHON_BIN:-python}"
NCU="${NCU:-ncu}"
DEST="$OUT/$ENV_LABEL/nsight"
mkdir -p "$DEST"
if [[ -n "$(ls -A "$DEST")" ]]; then
  echo "Choose a fresh Nsight destination: $DEST" >&2
  exit 1
fi
version=$("$NCU" --version | awk '/^Version /')
gpu=$("$PYTHON_BIN" -c 'import torch; print(torch.cuda.get_device_properties(0).uuid)')
for model in mlp cnn vit; do
  base="$DEST/${model}_step"
  "$NCU" --profile-from-start off --replay-mode kernel --cache-control none \
    --metrics dram__bytes_read.sum,dram__bytes_write.sum -o "$base" \
    "$PYTHON_BIN" -m gpumodel capture-step --model "$model" --batch-size 32 --warmup 100 --seed 1337 \
    2>&1 | tee "${base}_capture.txt"
  report="${base}.ncu-rep"
  if [[ -f "${base}.ncu-repz" ]]; then
    report="${base}.ncu-repz"
  fi
  if grep -Eq 'ERR_NVGPUCTRPERM|==ERROR==|No kernels were profiled' "${base}_capture.txt" ||
     ! grep -Fq "Captured one $model training step" "${base}_capture.txt" ||
     [[ ! -f "$report" ]]; then
    echo "No complete Nsight capture for $model" >&2
    exit 1
  fi
  "$NCU" --import "$report" --page raw --csv --print-units base --log-file "${base}_raw.csv"
  "$PYTHON_BIN" -m gpumodel extract-dram --input "${base}_raw.csv" --env "$ENV_LABEL" \
    --output "$DEST/measured_dram.pending.csv" --gpu-id "$gpu" --profiler-version "$version"
done
mv "$DEST/measured_dram.pending.csv" "$DEST/measured_dram.csv"
