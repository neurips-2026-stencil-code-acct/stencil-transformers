param(
    [string]$Devices = "cuda:0,cuda:1",
    [switch]$Resume
)

$ErrorActionPreference = "Stop"
$env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"
$repo = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$projectPython = Join-Path (Split-Path -Parent $repo) ".venv\Scripts\python.exe"
$localPython = Join-Path $repo ".venv\Scripts\python.exe"

if (Test-Path -LiteralPath $projectPython) {
    $python = $projectPython
} elseif (Test-Path -LiteralPath $localPython) {
    $python = $localPython
} else {
    $python = (Get-Command python -ErrorAction Stop).Source
}

Push-Location $repo
try {
    & $python analysis\positive_control\selftest.py
    $arguments = @(
        "analysis\positive_control\run.py",
        "--devices", $Devices,
        "--out", "analysis/positive_control/results",
        "--expected-models", "120"
    )
    if ($Resume) {
        $arguments += "--resume"
    }
    & $python @arguments
    & $python analysis\positive_control\validate.py `
        --results analysis/positive_control/results `
        --expected-models 120 `
        --require-both-gpus
    & $python analysis\positive_control\plot.py
}
finally {
    Pop-Location
}
