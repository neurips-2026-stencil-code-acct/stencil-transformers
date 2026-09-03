[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$ArtifactRoot = Split-Path -Parent $PSScriptRoot
$SourceRoot = Split-Path -Parent $ArtifactRoot

if ((Resolve-Path -LiteralPath $ArtifactRoot).Path -eq (Resolve-Path -LiteralPath $SourceRoot).Path) {
    throw 'Artifact and source roots must be different.'
}

function Copy-MappedFile {
    param(
        [Parameter(Mandatory = $true)][string]$SourceRelativePath,
        [Parameter(Mandatory = $true)][string]$DestinationRelativePath
    )

    $Source = Join-Path $SourceRoot $SourceRelativePath
    $Destination = Join-Path $ArtifactRoot $DestinationRelativePath
    if (-not (Test-Path -LiteralPath $Source -PathType Leaf)) {
        throw "Missing source file: $Source"
    }
    New-Item -ItemType Directory -Path (Split-Path -Parent $Destination) -Force | Out-Null
    Copy-Item -LiteralPath $Source -Destination $Destination -Force
}

function Copy-SamePathFile {
    param([Parameter(Mandatory = $true)][string]$RelativePath)
    Copy-MappedFile -SourceRelativePath $RelativePath -DestinationRelativePath $RelativePath
}

function Copy-MappedDirectory {
    param(
        [Parameter(Mandatory = $true)][string]$SourceRelativePath,
        [Parameter(Mandatory = $true)][string]$DestinationRelativePath,
        [hashtable]$FileRenames = @{}
    )

    $Source = Join-Path $SourceRoot $SourceRelativePath
    if (-not (Test-Path -LiteralPath $Source -PathType Container)) {
        throw "Missing source directory: $Source"
    }

    Get-ChildItem -LiteralPath $Source -Recurse -File | ForEach-Object {
        $Child = [System.IO.Path]::GetRelativePath($Source, $_.FullName)
        $DestinationChild = if ($FileRenames.ContainsKey($Child)) {
            $FileRenames[$Child]
        } else {
            $Child
        }
        $SourceChild = Join-Path $SourceRelativePath $Child
        $DestinationChild = Join-Path $DestinationRelativePath $DestinationChild
        Copy-MappedFile -SourceRelativePath $SourceChild -DestinationRelativePath $DestinationChild
    }
}

function Update-PackagedMetadataPaths {
    $MetadataFiles = @(
        'analysis\robustness_checks\results\paired_bootstrap\run_metadata.json',
        'analysis\robustness_checks\results\paired_bootstrap\head_preserving_bootstrap_metadata.json',
        'analysis\robustness_checks\results\j_minus_identity\run_metadata.json'
    )
    $Replacements = [ordered]@{
        'analysis\\Q4\\results\\q4_head_preserving' = 'analysis\\is_it_mechanistic\\results\\head_preserving'
        'analysis\\Q4\\results\\q4' = 'analysis\\is_it_mechanistic\\results'
        'analysis\\Q5\\results\\q5' = 'analysis\\fourier_domain_operator\\results'
    }

    foreach ($RelativePath in $MetadataFiles) {
        $Path = Join-Path $ArtifactRoot $RelativePath
        if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
            continue
        }
        $Text = [System.IO.File]::ReadAllText($Path)
        foreach ($Entry in $Replacements.GetEnumerator()) {
            $Text = $Text.Replace($Entry.Key, $Entry.Value)
        }
        [System.IO.File]::WriteAllText($Path, $Text)
    }
}

# Parent-level implementation files have no analysis-module imports to rewrite.
$CoreMappings = @(
    @('transformer.py', 'transformer.py'),
    @('generate_heat_equation.py', 'generate_data\generate_heat_equation.py'),
    @('generate_data\generate_lax_friedrichs.py', 'generate_data\generate_lax_friedrichs.py'),
    @('analysis\metrics.py', 'analysis\metrics.py'),
    @('check_converge.py', 'check_convergence.py'),
    @('compute_random_baselines.py', 'compute_random_baselines.py'),
    @('run_heat.py', 'run_files\run_heat.py'),
    @('run_friedrichs.py', 'run_files\run_friedrichs.py')
)
foreach ($Mapping in $CoreMappings) {
    Copy-MappedFile -SourceRelativePath $Mapping[0] -DestinationRelativePath $Mapping[1]
}

# Analysis source is intentionally maintained in this package. The parent still
# uses the legacy numbered module names, so copying it verbatim would undo this
# package's descriptive filenames and break imports. Refresh only portable data
# products and saved evidence from those legacy source directories.
Copy-MappedFile `
    -SourceRelativePath 'analysis\Q1\predictors.npz' `
    -DestinationRelativePath 'analysis\what_do_heads_learn\predictors.npz'

foreach ($ProfileKind in @('trained', 'baseline')) {
    $ProfileSource = Join-Path $SourceRoot "analysis\Q1\profiles\$ProfileKind"
    Get-ChildItem -LiteralPath $ProfileSource -File | Where-Object {
        $_.Name -like 'heat_*' -or $_.Name -like 'lf_*'
    } | ForEach-Object {
        Copy-MappedFile `
            -SourceRelativePath ([System.IO.Path]::GetRelativePath($SourceRoot, $_.FullName)) `
            -DestinationRelativePath (Join-Path "analysis\what_do_heads_learn\profiles\$ProfileKind" $_.Name)
    }
}

Copy-MappedDirectory `
    -SourceRelativePath 'analysis\Q1\results\q1' `
    -DestinationRelativePath 'analysis\what_do_heads_learn\results' `
    -FileRenames @{
        'q1_labels.pdf' = 'head_labels.pdf'
        'q1_labels.png' = 'head_labels.png'
    }
Copy-MappedDirectory `
    -SourceRelativePath 'analysis\Q1\results\trained_vs_random' `
    -DestinationRelativePath 'analysis\what_do_heads_learn\results\trained_vs_random'
Copy-MappedDirectory `
    -SourceRelativePath 'analysis\Q1\results\q1_workshop_robustness' `
    -DestinationRelativePath 'analysis\what_do_heads_learn\results\robustness'

Copy-MappedFile `
    -SourceRelativePath 'analysis\Q2\xcorr.npz' `
    -DestinationRelativePath 'analysis\parameter_sensitive\xcorr.npz'
Copy-MappedDirectory `
    -SourceRelativePath 'analysis\Q2\results\q2' `
    -DestinationRelativePath 'analysis\parameter_sensitive\results' `
    -FileRenames @{
        'q2_parameter_tracking.pdf' = 'parameter_tracking.pdf'
        'q2_parameter_tracking.png' = 'parameter_tracking.png'
    }

foreach ($Name in @('jacobian.csv', 'jacobian_profiles.npz', 'substitution.csv')) {
    Copy-MappedFile `
        -SourceRelativePath (Join-Path 'analysis\Q4\results\q4' $Name) `
        -DestinationRelativePath (Join-Path 'analysis\is_it_mechanistic\results' $Name)
}
Copy-MappedDirectory `
    -SourceRelativePath 'analysis\Q4\results\q4_head_preserving' `
    -DestinationRelativePath 'analysis\is_it_mechanistic\results\head_preserving'

Copy-MappedDirectory `
    -SourceRelativePath 'analysis\Q5\results\q5' `
    -DestinationRelativePath 'analysis\fourier_domain_operator\results' `
    -FileRenames @{
        'q5_fourier_symbols.png' = 'fourier_symbols.png'
        'q5_support_errors.png' = 'support_errors.png'
    }

Copy-SamePathFile -RelativePath 'analysis\robustness_checks\references.bib'
Copy-MappedDirectory `
    -SourceRelativePath 'analysis\robustness_checks\results\j_minus_identity' `
    -DestinationRelativePath 'analysis\robustness_checks\results\j_minus_identity'
Copy-MappedDirectory `
    -SourceRelativePath 'analysis\robustness_checks\results\paired_bootstrap' `
    -DestinationRelativePath 'analysis\robustness_checks\results\paired_bootstrap' `
    -FileRenames @{
        'q4_all_layer_bootstrap_summary.csv' = 'all_layer_bootstrap_summary.csv'
        'q4_all_layer_seed_contrasts.csv' = 'all_layer_seed_contrasts.csv'
        'q4_head_preserving_bootstrap_metadata.json' = 'head_preserving_bootstrap_metadata.json'
        'q4_head_preserving_bootstrap_summary.csv' = 'head_preserving_bootstrap_summary.csv'
        'q4_head_preserving_seed_contrasts.csv' = 'head_preserving_seed_contrasts.csv'
        'q5_fourier_bootstrap_summary.csv' = 'fourier_bootstrap_summary.csv'
        'q5_fourier_seed_deltas.csv' = 'fourier_seed_deltas.csv'
    }
Update-PackagedMetadataPaths

foreach ($Name in @(
    'figure4_operator_robust.pdf',
    'figure4_operator_robust.png'
)) {
    Copy-MappedFile `
        -SourceRelativePath (Join-Path 'analysis\workshop_figures\output_robustness' $Name) `
        -DestinationRelativePath (Join-Path 'analysis\workshop_figures\output_robustness' $Name)
}

Copy-MappedDirectory `
    -SourceRelativePath 'extension\positive_control\results' `
    -DestinationRelativePath 'analysis\positive_control\results' `
    -FileRenames @{
        'q1_assignment.csv' = 'attention_assignment.csv'
    }

$ParentVenvPython = Join-Path $SourceRoot '.venv\Scripts\python.exe'
if (Test-Path -LiteralPath $ParentVenvPython) {
    $Python = $ParentVenvPython
} else {
    $Python = (Get-Command python -ErrorAction Stop).Source
}

Push-Location $ArtifactRoot
try {
    New-Item -ItemType Directory -Path 'figures' -Force | Out-Null
    Copy-Item -LiteralPath 'analysis\positive_control\results\figure_detectability.pdf' -Destination 'figures\Figure1_detectability.pdf' -Force
    Copy-Item -LiteralPath 'analysis\positive_control\results\figure_detectability.png' -Destination 'figures\Figure1_detectability.png' -Force
    Copy-Item -LiteralPath 'analysis\workshop_figures\output_robustness\figure4_operator_robust.pdf' -Destination 'figures\Figure2_fourier_response.pdf' -Force
    Copy-Item -LiteralPath 'analysis\workshop_figures\output_robustness\figure4_operator_robust.png' -Destination 'figures\Figure2_fourier_response.png' -Force

    & $Python -B 'build_manifest.py'
    if ($LASTEXITCODE -ne 0) { throw 'Manifest creation failed.' }
} finally {
    Pop-Location
}

Write-Host 'Portable inputs and saved evidence refreshed into the descriptive package layout.'
