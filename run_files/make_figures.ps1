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

$env:MPLBACKEND = 'Agg'

Push-Location $ArtifactRoot
try {
    & $Python -B 'analysis\positive_control\plot.py'
    if ($LASTEXITCODE -ne 0) { throw 'Figure 1 generation failed.' }

    & $Python -B 'analysis\workshop_figures\figure4_operator_robust.py'
    if ($LASTEXITCODE -ne 0) { throw 'Figure 2 generation failed.' }

    New-Item -ItemType Directory -Path 'figures' -Force | Out-Null
    Copy-Item -LiteralPath 'analysis\positive_control\results\figure_detectability.pdf' -Destination 'figures\Figure1_detectability.pdf' -Force
    Copy-Item -LiteralPath 'analysis\positive_control\results\figure_detectability.png' -Destination 'figures\Figure1_detectability.png' -Force
    Copy-Item -LiteralPath 'analysis\workshop_figures\output_robustness\figure4_operator_robust.pdf' -Destination 'figures\Figure2_fourier_response.pdf' -Force
    Copy-Item -LiteralPath 'analysis\workshop_figures\output_robustness\figure4_operator_robust.png' -Destination 'figures\Figure2_fourier_response.png' -Force
} finally {
    Pop-Location
}

Write-Host 'Rebuilt Figure 1 and Figure 2 in figures\.'
