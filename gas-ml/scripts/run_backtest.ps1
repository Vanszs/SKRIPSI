# PowerShell script untuk run backtest
# Usage: .\run_backtest.ps1 [-Model "models"] [-Data "data/features.parquet"]

param(
    [string]$Model = "models",
    [string]$Data = "data/features.parquet",
    [string]$Baseline = "eip1559",
    [string]$OutputDir = "reports"
)

Write-Host "================================" -ForegroundColor Cyan
Write-Host "  Gas Fee ML - Backtesting" -ForegroundColor Cyan
Write-Host "================================" -ForegroundColor Cyan
Write-Host ""

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Green
& .\venv\Scripts\Activate.ps1

# Run backtest
Write-Host "Running backtest simulation..." -ForegroundColor Green
python -m src.backtest --model $Model --data $Data --baseline $Baseline --output-dir $OutputDir

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "================================" -ForegroundColor Cyan
    Write-Host "  Backtest completed!" -ForegroundColor Green
    Write-Host "  Results saved to: $OutputDir" -ForegroundColor Green
    Write-Host "================================" -ForegroundColor Cyan
} else {
    Write-Host ""
    Write-Host "Error: Backtest failed!" -ForegroundColor Red
    exit 1
}
