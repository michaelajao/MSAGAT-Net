# ============================================================================
# MSAGAT-Net Comprehensive Experiment Runner
# ============================================================================
# This script runs ALL experiments for the paper:
# 1. Main experiments (Tables 1 & 2) - All datasets with their respective horizons
# 2. Ablation studies (Tables 3 & 4) - All datasets with horizons [3, 7, 14]
# 3. (Optional) Aggregate results
# 4. (Optional) Generate publication figures
#
# Total experiments:
# - Main: 7 datasets * avg 3.4 horizons * 5 seeds ≈ 120 experiments
# - Ablation: 7 datasets * 4 ablations * 3 horizons * 1 seed = 84 experiments
# - Total: ~204 experiments @ ~6 min each = ~20 hours (with GPU optimizations)
# ============================================================================

$ErrorActionPreference = "Continue"

Write-Host "=" * 80 -ForegroundColor Cyan
Write-Host "MSAGAT-Net Comprehensive Experiment Runner" -ForegroundColor Cyan
Write-Host "=" * 80 -ForegroundColor Cyan
Write-Host ""

# Activate conda environment
Write-Host "[SETUP] Activating dl_env..." -ForegroundColor Yellow
conda activate dl_env

# Create timestamp for logs
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logdir = "logs"
if (-not (Test-Path $logdir)) {
    New-Item -ItemType Directory -Path $logdir | Out-Null
}

# Define all datasets
$all_datasets = @(
    "japan",
    "region785",
    "state360",
    "australia-covid",
    "spain-covid",
    "nhs_timeseries",
    "ltla_timeseries"
)

# Output directory for checkpoints
$save_dir = "save_final"

# What to run
# NOTE: For the final training sweep, keep these False (experiments only).
$run_aggregate_results = $false
$run_generate_figures = $false

Write-Host "[CONFIG] Datasets:" -ForegroundColor Yellow
foreach ($ds in $all_datasets) {
    Write-Host "  - $ds" -ForegroundColor Gray
}
Write-Host ""

# ============================================================================
# PART 1: MAIN EXPERIMENTS (Tables 1 & 2)
# ============================================================================
Write-Host "=" * 80 -ForegroundColor Green
Write-Host "PART 1: MAIN EXPERIMENTS (Tables 1 & 2)" -ForegroundColor Green
Write-Host "=" * 80 -ForegroundColor Green
Write-Host ""

$main_log = "$logdir\main_experiments_$timestamp.log"
Write-Host "[RUNNING] Main experiments for all datasets with 5 seeds..." -ForegroundColor Yellow
Write-Host "[LOG] $main_log" -ForegroundColor Gray
Write-Host ""

$datasets_str = $all_datasets -join " "
$main_cmd = "python -m src.scripts.experiments --datasets $datasets_str --experiment main --seeds 5 30 45 123 1000 --cpu --save_dir $save_dir"

Write-Host "[CMD] $main_cmd" -ForegroundColor Cyan
Write-Host ""

Invoke-Expression $main_cmd 2>&1 | Tee-Object -FilePath $main_log

if ($LASTEXITCODE -eq 0) {
    Write-Host "[SUCCESS] Main experiments completed!" -ForegroundColor Green
} else {
    Write-Host "[ERROR] Main experiments failed with exit code $LASTEXITCODE" -ForegroundColor Red
}
Write-Host ""

# ============================================================================
# PART 2: ABLATION STUDY (Tables 3 & 4)
# ============================================================================
Write-Host "=" * 80 -ForegroundColor Green
Write-Host "PART 2: ABLATION STUDY (Tables 3 & 4)" -ForegroundColor Green
Write-Host "=" * 80 -ForegroundColor Green
Write-Host ""

$ablation_log = "$logdir\ablation_experiments_$timestamp.log"
Write-Host "[RUNNING] Ablation studies for all datasets with horizons [3, 7, 14]..." -ForegroundColor Yellow
Write-Host "[LOG] $ablation_log" -ForegroundColor Gray
Write-Host ""

$ablation_cmd = "python -m src.scripts.experiments --datasets $datasets_str --experiment ablation --seeds 5 --cpu --save_dir $save_dir"

Write-Host "[CMD] $ablation_cmd" -ForegroundColor Cyan
Write-Host ""

Invoke-Expression $ablation_cmd 2>&1 | Tee-Object -FilePath $ablation_log

if ($LASTEXITCODE -eq 0) {
    Write-Host "[SUCCESS] Ablation experiments completed!" -ForegroundColor Green
} else {
    Write-Host "[ERROR] Ablation experiments failed with exit code $LASTEXITCODE" -ForegroundColor Red
}
Write-Host ""

# ============================================================================
# PART 3: AGGREGATE RESULTS
# ============================================================================
if ($run_aggregate_results) {
Write-Host "=" * 80 -ForegroundColor Green
Write-Host "PART 3: AGGREGATING RESULTS" -ForegroundColor Green
Write-Host "=" * 80 -ForegroundColor Green
Write-Host ""

Write-Host "[RUNNING] Aggregating all experimental results..." -ForegroundColor Yellow
Write-Host ""

$aggregate_cmd = "python -m src.scripts.aggregate_results"

Write-Host "[CMD] $aggregate_cmd" -ForegroundColor Cyan
Write-Host ""

Invoke-Expression $aggregate_cmd

if ($LASTEXITCODE -eq 0) {
    Write-Host "[SUCCESS] Results aggregated!" -ForegroundColor Green
} else {
    Write-Host "[ERROR] Aggregation failed with exit code $LASTEXITCODE" -ForegroundColor Red
}
Write-Host ""
} else {
    Write-Host "[SKIP] Aggregating results (disabled)" -ForegroundColor Yellow
    Write-Host ""
}

# ============================================================================
# PART 4: GENERATE PUBLICATION FIGURES
# ============================================================================
if ($run_generate_figures) {
Write-Host "=" * 80 -ForegroundColor Green
Write-Host "PART 4: GENERATING PUBLICATION FIGURES" -ForegroundColor Green
Write-Host "=" * 80 -ForegroundColor Green
Write-Host ""

Write-Host "[RUNNING] Generating publication-ready figures..." -ForegroundColor Yellow
Write-Host ""

$figures_cmd = "python -m src.scripts.generate_figures"

Write-Host "[CMD] $figures_cmd" -ForegroundColor Cyan
Write-Host ""

Invoke-Expression $figures_cmd

if ($LASTEXITCODE -eq 0) {
    Write-Host "[SUCCESS] Figures generated!" -ForegroundColor Green
} else {
    Write-Host "[ERROR] Figure generation failed with exit code $LASTEXITCODE" -ForegroundColor Red
}
Write-Host ""
} else {
    Write-Host "[SKIP] Generating figures (disabled)" -ForegroundColor Yellow
    Write-Host ""
}

# ============================================================================
# FINAL SUMMARY
# ============================================================================
Write-Host "=" * 80 -ForegroundColor Cyan
Write-Host "COMPLETE EXPERIMENT SUMMARY" -ForegroundColor Cyan
Write-Host "=" * 80 -ForegroundColor Cyan
Write-Host ""

Write-Host "Datasets processed:" -ForegroundColor Yellow
foreach ($ds in $all_datasets) {
    Write-Host "  - $ds" -ForegroundColor Gray
}
Write-Host ""

Write-Host "Configuration:" -ForegroundColor Yellow
Write-Host "  Main experiments: All datasets with dataset-specific horizons, 5 seeds" -ForegroundColor Gray
Write-Host "  Ablation studies: All datasets with horizons [3, 7, 14], 1 seed" -ForegroundColor Gray
Write-Host "  Aggregate results: $run_aggregate_results" -ForegroundColor Gray
Write-Host "  Generate figures: $run_generate_figures" -ForegroundColor Gray
Write-Host ""

Write-Host "Outputs saved:" -ForegroundColor Yellow
Write-Host "  Logs: $main_log" -ForegroundColor Gray
Write-Host "        $ablation_log" -ForegroundColor Gray
Write-Host "  Model checkpoints: $save_dir/" -ForegroundColor Gray
Write-Host "  Results CSVs: report/results/{dataset}/all_results.csv" -ForegroundColor Gray
if ($run_generate_figures) {
    Write-Host "  Publication figures: report/figures/paper/" -ForegroundColor Gray
}
Write-Host ""

Write-Host "=" * 80 -ForegroundColor Cyan
Write-Host "ALL PIPELINE STEPS COMPLETE!" -ForegroundColor Cyan
Write-Host "=" * 80 -ForegroundColor Cyan
