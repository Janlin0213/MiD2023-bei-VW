<#
.SYNOPSIS
Run the maintained thesis workflow from this checkout.
.EXAMPLE
.\run-pipeline.ps1 -Mode models -Overwrite
.EXAMPLE
.\run-pipeline.ps1 -Mode full -Overwrite
.EXAMPLE
.\run-pipeline.ps1 -Mode full -Overwrite -Resume
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet('models', 'full')]
    [string] $Mode,
    [switch] $Overwrite,
    [switch] $Resume,
    [switch] $Restart,
    [switch] $Plan
)

$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $false
$repo = $PSScriptRoot
$checkpoint = Join-Path $repo 'outputs/.pipeline-interrupted.json'
$python = Join-Path $repo '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Project Python environment is missing: $python"
}
Set-Location -LiteralPath $repo

$held = @()
$completed = @()
$currentStage = $null
$stageOrder = @(
    'Selected raw extraction', 'Household backbone', 'Licensed driver resources',
    'Habitual car use', 'Reference day mobility', 'Phase 1 home chains',
    'Phase 1 diagnostics', 'Phase 2 vehicle states', 'Phase 3 strict baseline',
    'Decision-timing sensitivity input', 'Allocation candidate',
    'Observed driver-vehicle patterns', 'Allocation sample support',
    'Strict tour features', 'Sensitivity tour features', 'Strict MNL input',
    'Sensitivity MNL input', 'Allocation MNL input', 'Single-driver design matrix',
    'Association screening', 'Employment-age diagnostic', 'Stage-0 V2',
    'Stage-0 V1', 'Core-V1', 'Core-V2', 'Purpose comparison QA'
)
if ($Mode -eq 'models') {
    $stageOrder = @('Stage-0 V2', 'Stage-0 V1', 'Core-V1', 'Core-V2')
}
$resumeIndex = 0
$guarded = @(
    'data_processed/reconstruction/allocation_candidate',
    'data_processed/eda/bipartite',
    'data_processed/eda/allocation_sample_support',
    'outputs/eda/bipartite',
    'outputs/eda/allocation_sample_support',
    'outputs/mnl/v2_only',
    'outputs/mnl/v1_only',
    'outputs/mnl/stage1/core_v1',
    'outputs/mnl/stage1/core_v2'
)
$guardedFiles = @(
    'data_processed/model_input/allocation_candidate_model_input.csv',
    'data_processed/model_input/allocation_candidate_model_input_QA.csv',
    'data_processed/model_input/single_driver_mnl_design_matrix.csv',
    'data_processed/model_input/single_driver_mnl_design_matrix_QA.csv',
    'data_processed/model_input/single_driver_mnl_encoding_spec.md'
)
$stageTargets = @{
    'Allocation candidate' = @('data_processed/reconstruction/allocation_candidate')
    'Observed driver-vehicle patterns' = @('data_processed/eda/bipartite', 'outputs/eda/bipartite')
    'Allocation sample support' = @('data_processed/eda/allocation_sample_support', 'outputs/eda/allocation_sample_support')
    'Allocation MNL input' = @($guardedFiles[0], $guardedFiles[1])
    'Single-driver design matrix' = @($guardedFiles[2], $guardedFiles[3], $guardedFiles[4])
    'Stage-0 V2' = @('outputs/mnl/v2_only')
    'Stage-0 V1' = @('outputs/mnl/v1_only')
    'Core-V1' = @('outputs/mnl/stage1/core_v1')
    'Core-V2' = @('outputs/mnl/stage1/core_v2')
}

function Should-Run([string] $stage) {
    $index = [array]::IndexOf($stageOrder, $stage)
    if ($index -lt 0) { throw "Stage is not in the selected workflow: $stage" }
    return $index -ge $resumeIndex
}

function Get-Fingerprints([string] $kind) {
    $files = if ($kind -eq 'sources') {
        @(Get-ChildItem -LiteralPath (Join-Path $repo 'scripts'), (Join-Path $repo 'src'),
            (Join-Path $repo 'config') -Recurse -File -ErrorAction SilentlyContinue |
            Where-Object { $_.Extension -in @('.py', '.toml', '.json', '.yaml', '.yml', '.ps1') }) +
        @(Get-Item -LiteralPath (Join-Path $repo 'run-pipeline.ps1'),
            (Join-Path $repo 'pyproject.toml'), (Join-Path $repo 'uv.lock'))
    } elseif ($kind -eq 'inputs') {
        $relative = if ($Mode -eq 'full') {
            @('data_raw/MiD2023_Haushalte.csv', 'data_raw/MiD2023_Personen.csv',
                'data_raw/MiD2023_Wege.csv', 'data_raw/MiD2023_Autos.csv',
                'data/reference/codebooks/MiD2023_Codepläne_B1_Standard_v1.1.xlsx')
        } else {
            @('data_processed/model_input/single_driver_mnl_design_matrix.csv',
                'data_processed/model_input/allocation_candidate_model_input.csv',
                'data_processed/reconstruction/phase4/home_based_tour_features.csv')
        }
        @($relative | ForEach-Object {
            $path = Join-Path $repo $_
            if (Test-Path -LiteralPath $path -PathType Leaf) { Get-Item -LiteralPath $path }
        })
    } else {
        @(Get-ChildItem -LiteralPath (Join-Path $repo 'data_processed'),
            (Join-Path $repo 'outputs') -Recurse -File -ErrorAction SilentlyContinue |
            Where-Object { $_.FullName -notmatch '[\\/ ]\.pipeline_hold_' -and
                $_.Name -ne '.pipeline-interrupted.json' -and
                $_.FullName -notmatch '[\\/]\.core_v2_replacement_staging[\\/]' })
    }
    $result = @{}
    foreach ($file in $files) {
        $relative = $file.FullName.Substring($repo.Length + 1).Replace('\', '/')
        $result[$relative] = (Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash
    }
    return $result
}

function Assert-Fingerprints($expected, [string] $kind) {
    $actual = Get-Fingerprints $kind
    if ($actual.Count -ne @($expected.PSObject.Properties).Count) {
        throw "Resume rejected: $kind file inventory changed."
    }
    foreach ($property in $expected.PSObject.Properties) {
        if (-not $actual.ContainsKey($property.Name) -or $actual[$property.Name] -ne $property.Value) {
            throw "Resume rejected: $kind changed: $($property.Name)"
        }
    }
}

function Assert-CompletedOutputs([string[]] $stages) {
    foreach ($stage in $stages) {
        if (-not $stageTargets.ContainsKey($stage)) { continue }
        foreach ($relative in $stageTargets[$stage]) {
            $path = Join-Path $repo $relative
            if (-not (Test-Path -LiteralPath $path)) {
                throw "Resume rejected: completed $stage output missing: $relative"
            }
            $item = Get-Item -LiteralPath $path
            if ($item.PSIsContainer) {
                if (-not @(Get-ChildItem -LiteralPath $path -Recurse -File).Count) {
                    throw "Resume rejected: completed $stage output empty: $relative"
                }
            } elseif ($item.Length -eq 0) {
                throw "Resume rejected: completed $stage output empty: $relative"
            }
        }
    }
    foreach ($definition in @(
            @('Core-V1', 'outputs/mnl/stage1/core_v1/estimation/review/core_v1_stage5_manifest.json', 'decision_status'),
            @('Core-V2', 'outputs/mnl/stage1/core_v2/estimation/review/core_v2_output_manifest.json', 'status'))) {
        if ($definition[0] -notin $stages) { continue }
        $path = Join-Path $repo $definition[1]
        if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
            throw "Resume rejected: completed $($definition[0]) manifest missing."
        }
        $manifest = Get-Content -LiteralPath $path -Raw | ConvertFrom-Json
        if ($manifest.($definition[2]) -ne 'ESTIMATED_PENDING_MANUAL_REVIEW') {
            throw "Resume rejected: completed $($definition[0]) manifest has incomplete status."
        }
    }
}

function Safe-Path([string] $relative) {
    if ($relative -notin $guarded -and $relative -notin $guardedFiles -and
            $relative -ne 'outputs/mnl/stage1/.core_v2_replacement_staging') {
        throw "Unlisted output target: $relative"
    }
    $path = [IO.Path]::GetFullPath((Join-Path $repo $relative))
    if (-not $path.StartsWith($repo + [IO.Path]::DirectorySeparatorChar,
            [StringComparison]::OrdinalIgnoreCase)) {
        throw "Output path escapes the repository: $relative"
    }
    if (Test-Path -LiteralPath $path) {
        $item = Get-Item -LiteralPath $path -Force
        $items = @($item)
        if ($item.PSIsContainer) {
            $items += @(Get-ChildItem -LiteralPath $path -Recurse -Force)
        }
        if (@($items | Where-Object { ($_.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0 }).Count) {
            throw "Output directory contains a link or junction: $relative"
        }
    }
    return $path
}

function Remove-Generated([string] $path) {
    $absolute = [IO.Path]::GetFullPath($path)
    if (-not $absolute.StartsWith($repo + [IO.Path]::DirectorySeparatorChar,
            [StringComparison]::OrdinalIgnoreCase)) {
        throw "Removal path escapes the repository: $path"
    }
    if (-not (Test-Path -LiteralPath $absolute)) { return }
    $item = Get-Item -LiteralPath $absolute -Force
    $items = @($item)
    if ($item.PSIsContainer) {
        $items += @(Get-ChildItem -LiteralPath $absolute -Recurse -Force)
    }
    if (@($items | Where-Object { ($_.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0 }).Count) {
        throw "Removal target contains a link or junction: $path"
    }
    if ($item.PSIsContainer) {
        Remove-Item -LiteralPath $absolute -Recurse -Force
    } else {
        Remove-Item -LiteralPath $absolute -Force
    }
}

function Hold-Output([string] $relative, [string] $stage) {
    if (-not (Should-Run $stage)) { return }
    $script:currentStage = $stage
    $path = Safe-Path $relative
    if (-not (Test-Path -LiteralPath $path)) { return }
    if (-not $Overwrite) { throw "Existing output requires -Overwrite: $relative" }
    $parent = Split-Path -Parent $path
    $name = Split-Path -Leaf $path
    $hold = Join-Path $parent ('.pipeline_hold_' + $name)
    if (Test-Path -LiteralPath $hold) { throw "Unresolved previous run: $hold" }
    Move-Item -LiteralPath $path -Destination $hold
    $script:held += [pscustomobject]@{ Current = $path; Previous = $hold }
}

function Complete-Stage([string] $stage) {
    foreach ($item in $script:held) {
        if (Test-Path -LiteralPath $item.Previous) { Remove-Generated $item.Previous }
    }
    $script:held = @()
    $script:completed += $stage
    $script:currentStage = $null
}

function Invoke-Stage([string] $label, [string] $relative,
        [string[]] $arguments = @(), [string] $StageId = $label, [switch] $KeepHold) {
    if (-not (Should-Run $StageId)) { return }
    $script:currentStage = $StageId
    $script = Join-Path $repo $relative
    if (-not (Test-Path -LiteralPath $script -PathType Leaf)) {
        throw "Missing maintained entry point: $script"
    }
    Write-Host "`n=== $label ===" -ForegroundColor Cyan
    Write-Host "$relative $($arguments -join ' ')"
    & $python $script @arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$label failed with exit code $LASTEXITCODE"
    }
    if (-not $KeepHold) { Complete-Stage $StageId }
}

# An interrupted run requires an explicit resume or restart. A resume is accepted
# only when its completed prefix and every recorded source/input/output hash match.
if ($Resume -and $Restart) { throw 'Choose -Resume or -Restart, not both.' }
if ($Restart -and -not $Overwrite) { throw '-Restart requires -Overwrite.' }
if (Test-Path -LiteralPath $checkpoint) {
    if (-not $Resume -and -not $Restart) {
        throw "An interrupted run exists. Use -Resume or explicitly use -Restart: $checkpoint"
    }
    if ($Resume) {
        $saved = Get-Content -LiteralPath $checkpoint -Raw | ConvertFrom-Json
        if ($saved.mode -ne $Mode) { throw "Resume checkpoint is for $($saved.mode), not $Mode." }
        $resumeIndex = [array]::IndexOf($stageOrder, [string]$saved.failed_stage)
        if ($resumeIndex -lt 0 -or @($saved.completed).Count -ne $resumeIndex) {
            throw 'Resume checkpoint has an invalid stage prefix.'
        }
        for ($i = 0; $i -lt $resumeIndex; $i++) {
            if ($saved.completed[$i] -ne $stageOrder[$i]) {
                throw 'Resume checkpoint completed stages are out of order.'
            }
        }
        Assert-Fingerprints $saved.sources 'sources'
        Assert-Fingerprints $saved.inputs 'inputs'
        Assert-Fingerprints $saved.generated 'generated'
        Assert-CompletedOutputs @($saved.completed)
        $completed = @($saved.completed)
    }
} elseif ($Resume) {
    throw "No interrupted run checkpoint exists: $checkpoint"
}

# Enumerate every known collision in the selected suffix before expensive work.
$preflight = @()
foreach ($stage in $stageOrder[$resumeIndex..($stageOrder.Count - 1)]) {
    if ($stageTargets.ContainsKey($stage)) { $preflight += $stageTargets[$stage] }
}
if ($Mode -eq 'full') {
    $ordinary = @('data_processed/selected_raw', 'data_processed/reconstruction/phase1',
        'data_processed/reconstruction/phase2', 'data_processed/reconstruction/phase3',
        'data_processed/reconstruction/phase3_sensitivity',
        'data_processed/reconstruction/phase4',
        'data_processed/reconstruction/phase4_sensitivity',
        'data_processed/model_input', 'data_processed/eda/statistical_test',
        'outputs/eda/statistical_tests/theme1_single_vs_multicar')
    if ($resumeIndex -eq 0) { $preflight += $ordinary }
    foreach ($input in @('data_raw/MiD2023_Haushalte.csv', 'data_raw/MiD2023_Personen.csv',
            'data_raw/MiD2023_Wege.csv', 'data_raw/MiD2023_Autos.csv',
            'data/reference/codebooks/MiD2023_Codepläne_B1_Standard_v1.1.xlsx')) {
        if (-not (Test-Path -LiteralPath (Join-Path $repo $input) -PathType Leaf)) {
            throw "Required authorized input is missing: $input"
        }
    }
}
$conflicts = @($preflight | Sort-Object -Unique | Where-Object {
    Test-Path -LiteralPath (Join-Path $repo $_)
})
Write-Host "Selected stages: $($stageOrder[$resumeIndex..($stageOrder.Count - 1)] -join ' -> ')"
if ($conflicts.Count) {
    Write-Host "Existing generated outputs in selected workflow ($($conflicts.Count)):"
    $conflicts | ForEach-Object { Write-Host "  $_" }
}
if (-not $Overwrite -and $conflicts.Count -and -not $Plan) {
    throw "Existing generated outputs require -Overwrite: $($conflicts -join ', ')"
}
$staging = Safe-Path 'outputs/mnl/stage1/.core_v2_replacement_staging'
if (Test-Path -LiteralPath $staging) { throw "Unresolved Core-V2 staging directory: $staging" }
foreach ($relative in @($guarded) + @($guardedFiles)) {
    $path = Safe-Path $relative
    $hold = Join-Path (Split-Path -Parent $path) ('.pipeline_hold_' + (Split-Path -Leaf $path))
    if (Test-Path -LiteralPath $hold) { throw "Unresolved previous run: $hold" }
}
if ($Plan) { return }
if ($Restart -and (Test-Path -LiteralPath $checkpoint)) {
    Remove-Generated $checkpoint
}

$priorStaging = [Environment]::GetEnvironmentVariable('CORE_V2_STAGING', 'Process')
try {
    if ($Mode -eq 'full') {
        Invoke-Stage 'Selected raw extraction' 'scripts/00_data_preparation/01_extract_selected_raw.py'
        Invoke-Stage 'Household backbone' 'scripts/01_descriptive/01_build_household_backbone.py'
        Invoke-Stage 'Licensed driver resources' 'scripts/01_descriptive/02_licensed_driver_resources.py'
        Invoke-Stage 'Habitual car use' 'scripts/01_descriptive/03_habitual_car_use_orientation.py'
        Invoke-Stage 'Reference day mobility' 'scripts/01_descriptive/04_reference_day_mobility_intensity.py'
        Invoke-Stage 'Phase 1 home chains' 'scripts/02_choice_reconstruction/01_phase1_home_chain.py'
        Invoke-Stage 'Phase 1 diagnostics' 'scripts/02_choice_reconstruction/01b_phase1_diagnostics.py'
        Invoke-Stage 'Phase 2 vehicle states' 'scripts/02_choice_reconstruction/02_phase2_vehicle_states.py'
        Invoke-Stage 'Phase 3 strict baseline' 'scripts/02_choice_reconstruction/03_phase3_strict_baseline.py'
        Invoke-Stage 'Decision-timing sensitivity input' 'scripts/02_choice_reconstruction/04_decision_timing_sensitivity.py'
        Hold-Output 'data_processed/reconstruction/allocation_candidate' 'Allocation candidate'
        Invoke-Stage 'Allocation candidate' 'scripts/02_choice_reconstruction/05_allocation_analysis_candidate.py'
        Hold-Output 'data_processed/eda/bipartite' 'Observed driver-vehicle patterns'
        Hold-Output 'outputs/eda/bipartite' 'Observed driver-vehicle patterns'
        Invoke-Stage 'Observed driver-vehicle patterns' 'scripts/01_descriptive/05_driver_vehicle_bipartite_patterns.py'
        Hold-Output 'data_processed/eda/allocation_sample_support' 'Allocation sample support'
        Hold-Output 'outputs/eda/allocation_sample_support' 'Allocation sample support'
        Invoke-Stage 'Allocation sample support' 'scripts/01_descriptive/06_allocation_sample_support.py'
        Invoke-Stage 'Strict tour features' 'scripts/03_model_input/01_build_home_based_tour_features.py'
        Invoke-Stage 'Sensitivity tour features' 'scripts/03_model_input/01b_build_sensitivity_tour_features.py'
        Invoke-Stage 'Strict MNL input' 'scripts/03_model_input/02_build_mnl_vehicle_choice_model_input.py'
        Invoke-Stage 'Sensitivity MNL input' 'scripts/03_model_input/02b_build_sensitivity_model_input.py'
        Hold-Output 'data_processed/model_input/allocation_candidate_model_input.csv' 'Allocation MNL input'
        Hold-Output 'data_processed/model_input/allocation_candidate_model_input_QA.csv' 'Allocation MNL input'
        Invoke-Stage 'Allocation MNL input' 'scripts/03_model_input/04_build_allocation_candidate_model_input.py'
        Hold-Output 'data_processed/model_input/single_driver_mnl_design_matrix.csv' 'Single-driver design matrix'
        Hold-Output 'data_processed/model_input/single_driver_mnl_design_matrix_QA.csv' 'Single-driver design matrix'
        Hold-Output 'data_processed/model_input/single_driver_mnl_encoding_spec.md' 'Single-driver design matrix'
        Invoke-Stage 'Single-driver design matrix' 'scripts/03_model_input/05_build_single_driver_mnl_design_matrix.py'
        Invoke-Stage 'Association screening' 'scripts/04_screening/01_hierarchical_mixed_association_screening.py'
        Invoke-Stage 'Employment-age diagnostic' 'scripts/04_screening/02_employment_share_by_age_diagnostic.py'
    }

    Hold-Output 'outputs/mnl/v2_only' 'Stage-0 V2'
    Invoke-Stage 'Stage-0 V2' 'scripts/05_mnl/01_estimate_v2_only_mnl.py'
    Hold-Output 'outputs/mnl/v1_only' 'Stage-0 V1'
    Invoke-Stage 'Stage-0 V1' 'scripts/05_mnl/02_estimate_v1_only_mnl.py'

    Hold-Output 'outputs/mnl/stage1/core_v1' 'Core-V1'
    Invoke-Stage 'Core-V1 preparation' 'scripts/05_mnl/stage1/01_prepare_core_v1.py' -StageId 'Core-V1' -KeepHold
    Invoke-Stage 'Core-V1 prepared baseline' 'scripts/05_mnl/stage1/00_build_core_v1_baselines.py' @('prepared') -StageId 'Core-V1' -KeepHold
    Invoke-Stage 'Core-V1 diagnostics' 'scripts/05_mnl/stage1/02_diagnose_core_v1.py' -StageId 'Core-V1' -KeepHold
    Invoke-Stage 'Core-V1 diagnosed baseline' 'scripts/05_mnl/stage1/00_build_core_v1_baselines.py' @('diagnosed') -StageId 'Core-V1' -KeepHold
    Invoke-Stage 'Core-V1 sealed diagnostics' 'scripts/05_mnl/stage1/02_diagnose_core_v1.py' -StageId 'Core-V1' -KeepHold
    Invoke-Stage 'Core-V1 estimation baseline' 'scripts/05_mnl/stage1/00_build_core_v1_baselines.py' @('estimation') -StageId 'Core-V1' -KeepHold
    Invoke-Stage 'Core-V1 estimation' 'scripts/05_mnl/stage1/03_estimate_core_v1.py' -StageId 'Core-V1'

    if (Should-Run 'Core-V2') {
        $currentStage = 'Core-V2'
        [Environment]::SetEnvironmentVariable('CORE_V2_STAGING', '1', 'Process')
        Invoke-Stage 'Core-V2 preparation' 'scripts/05_mnl/stage1/01_prepare_core_v2.py' -StageId 'Core-V2' -KeepHold
        Invoke-Stage 'Core-V2 diagnostics' 'scripts/05_mnl/stage1/02_diagnose_core_v2.py' -StageId 'Core-V2' -KeepHold
        Invoke-Stage 'Core-V2 estimation' 'scripts/05_mnl/stage1/03_estimate_core_v2.py' -StageId 'Core-V2' -KeepHold
        [Environment]::SetEnvironmentVariable('CORE_V2_STAGING', $priorStaging, 'Process')
        Hold-Output 'outputs/mnl/stage1/core_v2' 'Core-V2'
        Move-Item -LiteralPath $staging -Destination (Join-Path $repo 'outputs/mnl/stage1/core_v2')
        Complete-Stage 'Core-V2'
    }

    if ($Mode -eq 'full') {
        Invoke-Stage 'Purpose comparison QA' 'scripts/03_model_input/06_build_purpose_comparison_qa.py'
    }

    if (Test-Path -LiteralPath $checkpoint) { Remove-Generated $checkpoint }
    Write-Host "`nPipeline completed: $Mode" -ForegroundColor Green
} catch {
    [Environment]::SetEnvironmentVariable('CORE_V2_STAGING', $priorStaging, 'Process')
    if (Test-Path -LiteralPath $staging) {
        Remove-Generated $staging
    }
    [array]::Reverse($held)
    foreach ($item in $held) {
        if (Test-Path -LiteralPath $item.Previous) {
            if (Test-Path -LiteralPath $item.Current) {
                Remove-Generated $item.Current
            }
            Move-Item -LiteralPath $item.Previous -Destination $item.Current
        }
    }
    if ($currentStage) {
        try {
            $record = [ordered]@{
                mode = $Mode
                failed_stage = $currentStage
                completed = @($completed)
                sources = Get-Fingerprints 'sources'
                inputs = Get-Fingerprints 'inputs'
                generated = Get-Fingerprints 'generated'
            }
            $record | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $checkpoint -Encoding utf8
            Write-Warning "Recovery checkpoint saved: $checkpoint"
        } catch {
            Write-Warning "Could not save recovery checkpoint: $_"
        }
    }
    Write-Error "Pipeline stopped: $_"
    exit 1
} finally {
    [Environment]::SetEnvironmentVariable('CORE_V2_STAGING', $priorStaging, 'Process')
}
