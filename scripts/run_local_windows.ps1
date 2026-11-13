param(
    [Parameter(Mandatory=$true)][string]$Environment
)
$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $root
$python = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    throw "Python environment not found: $python"
}
& $python -c "import torch; assert torch.cuda.is_available(), 'CUDA is required'"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $python -m gpumodel describe
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $python -m gpumodel calibrate --env $Environment
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $python -m gpumodel run --env $Environment --models mlp cnn vit --repeats 3 --steps 120 --warmup 15 --batch-size 32
exit $LASTEXITCODE
