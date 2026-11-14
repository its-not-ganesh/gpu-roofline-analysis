param(
  [Parameter(Mandatory=$true)][string]$Environment,
  [string]$Output = "results_final",
  [string]$Ncu = "ncu"
)
$ErrorActionPreference = "Stop"
Set-Location (Resolve-Path (Join-Path $PSScriptRoot ".."))
$python = (Resolve-Path ".venv\Scripts\python.exe").Path
$versionOutput = & $Ncu --version
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
$version = ($versionOutput | Select-String '^Version ' | Select-Object -First 1).Line
$gpu = & $python -c "import torch; print(torch.cuda.get_device_properties(0).uuid)"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
$folder = Join-Path (Join-Path $Output $Environment) "nsight"
New-Item -ItemType Directory -Force -Path $folder | Out-Null
$folder = (Resolve-Path $folder).Path
$totals = Join-Path $folder "measured_dram.csv"
if (Get-ChildItem -LiteralPath $folder -File) {
    throw "Nsight destination already contains files. Choose a fresh output directory."
}
$metrics = "dram__bytes_read.sum,dram__bytes_write.sum"
$query = & $Ncu --query-metrics --query-metrics-mode suffix --metrics dram__bytes_read,dram__bytes_write 2>&1
$code = $LASTEXITCODE
$query | Set-Content -Encoding UTF8 (Join-Path $folder "counter-query.txt")
if ($code -ne 0) { exit $code }
if ($query -match 'ERR_NVGPUCTRPERM|==ERROR==') {
    $query | Write-Host
    throw "GPU performance counters are unavailable."
}
$pending = Join-Path $folder "measured_dram.pending.csv"
foreach ($model in @("mlp", "cnn", "vit")) {
    $base = Join-Path $folder "${model}_step"
    $capture = & $Ncu --profile-from-start off --replay-mode kernel --cache-control none --metrics $metrics -o $base $python -m gpumodel capture-step --model $model --batch-size 32 --warmup 100 --seed 1337 2>&1
    $code = $LASTEXITCODE
    $capture | Set-Content -Encoding UTF8 "${base}_capture.txt"
    $capture | Write-Host
    if ($code -ne 0) { exit $code }
    $reports = @(Get-Item -Path @("${base}.ncu-rep", "${base}.ncu-repz") -ErrorAction SilentlyContinue)
    if ($capture -match 'ERR_NVGPUCTRPERM|==ERROR==|No kernels were profiled' -or
        -not ($capture -match "Captured one $model training step") -or
        $reports.Count -ne 1) {
        throw "No complete Nsight capture for $model. See ${base}_capture.txt."
    }
    & $Ncu --import $reports[0].FullName --page raw --csv --print-units base --log-file "${base}_raw.csv"
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    & $python -m gpumodel extract-dram --input "${base}_raw.csv" --env $Environment --model $model --output $pending --gpu-id "$gpu" --profiler-version "$version"
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
Move-Item -LiteralPath $pending -Destination $totals
Write-Host "Review $folder for one complete step per model before plotting measured DRAM."
