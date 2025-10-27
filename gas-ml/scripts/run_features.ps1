# PowerShell script untuk generate features
# Usage: .\run_features.ps1 [-Input "data/blocks.csv"] [-Output "data/features.parquet"]

param(
    [string]$Input = "data/blocks.csv",
    [string]$Output = "data/features.parquet"
)

Write-Host "================================" -ForegroundColor Cyan
Write-Host "  Gas Fee ML - Feature Engineering" -ForegroundColor Cyan
Write-Host "================================" -ForegroundColor Cyan
Write-Host ""

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Green
& .\venv\Scripts\Activate.ps1

# Run feature engineering
Write-Host "Generating features from $Input..." -ForegroundColor Green
python -m src.features --in $Input --out $Output

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "================================" -ForegroundColor Cyan
    Write-Host "  Feature engineering completed!" -ForegroundColor Green
    Write-Host "================================" -ForegroundColor Cyan
} else {
    Write-Host ""
    Write-Host "Error: Feature engineering failed!" -ForegroundColor Red
    exit 1
}
