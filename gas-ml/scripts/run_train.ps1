# PowerShell script untuk train model
# Usage: .\run_train.ps1 [-Config "cfg/exp.yaml"] [-Input "data/features.parquet"]

param(
    [string]$Config = "cfg/exp.yaml",
    [string]$Input = "data/features.parquet",
    [string]$OutputDir = "models"
)

Write-Host "================================" -ForegroundColor Cyan
Write-Host "  Gas Fee ML - Model Training" -ForegroundColor Cyan
Write-Host "================================" -ForegroundColor Cyan
Write-Host ""

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Green
& .\venv\Scripts\Activate.ps1

# Run training
Write-Host "Training hybrid LSTM+XGBoost model..." -ForegroundColor Green
python -m src.train --cfg $Config --in $Input --out-dir $OutputDir

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "================================" -ForegroundColor Cyan
    Write-Host "  Model training completed!" -ForegroundColor Green
    Write-Host "  Model saved to: $OutputDir" -ForegroundColor Green
    Write-Host "================================" -ForegroundColor Cyan
} else {
    Write-Host ""
    Write-Host "Error: Model training failed!" -ForegroundColor Red
    exit 1
}
