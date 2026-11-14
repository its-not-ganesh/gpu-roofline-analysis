param(
    [Parameter(Mandatory=$true)][string]$Environment,
    [string]$Output = "results_final"
)
$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $root
$python = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) { throw "Python environment not found: $python" }
& $python -c "import torch; assert torch.cuda.is_available(), 'CUDA is required'"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
$destination = Join-Path $Output $Environment
if (Test-Path -LiteralPath $destination) { throw "Choose a fresh output directory for $Environment" }
& $python -m gpumodel describe
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $python -m gpumodel calibrate --env $Environment --output $Output
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $python -m gpumodel run --env $Environment --output $Output --models mlp cnn vit --repeats 3 --steps 10000 --warmup 100 --batch-size 32 --interval 0.2 --data-mode device
exit $LASTEXITCODE
