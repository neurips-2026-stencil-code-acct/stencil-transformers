[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$ArtifactRoot = Split-Path -Parent $PSScriptRoot
$ParentVenvPython = Join-Path (Split-Path -Parent $ArtifactRoot) '.venv\Scripts\python.exe'
$LocalVenvPython = Join-Path $ArtifactRoot '.venv\Scripts\python.exe'

if (Test-Path -LiteralPath $ParentVenvPython) {
    $Python = $ParentVenvPython
} elseif (Test-Path -LiteralPath $LocalVenvPython) {
    $Python = $LocalVenvPython
} else {
    $Python = (Get-Command python -ErrorAction Stop).Source
}

$Checks = @(
    'analysis\what_do_heads_learn\selftest.py',
    'analysis\what_do_heads_learn\robustness_selftest.py',
    'analysis\parameter_sensitive\selftest.py',
    'analysis\is_it_mechanistic\selftest.py',
    'analysis\is_it_mechanistic\head_preserving_selftest.py',
    'analysis\fourier_domain_operator\selftest.py',
    'analysis\fourier_domain_operator\support_selftest.py',
    'analysis\robustness_checks\test_robustness_checks.py',
    'analysis\positive_control\selftest.py'
)

Push-Location $ArtifactRoot
try {
    foreach ($Check in $Checks) {
        Write-Host "[check] $Check"
        & $Python -B $Check
        if ($LASTEXITCODE -ne 0) {
            throw "Check failed with exit code ${LASTEXITCODE}: $Check"
        }
    }
} finally {
    Pop-Location
}

Write-Host 'All packaged analysis checks passed.'
