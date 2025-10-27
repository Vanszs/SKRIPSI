# PowerShell script untuk run inference
# Usage: .\run_infer.ps1 [-Model "models"] [-Continuous]

param(
    [string]$Model = "models",
    [string]$Network = "sepolia",
    [switch]$Continuous
)

Write-Host "================================" -ForegroundColor Cyan
Write-Host "  Gas Fee ML - Inference" -ForegroundColor Cyan
Write-Host "================================" -ForegroundColor Cyan
Write-Host ""

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Green
& .\venv\Scripts\Activate.ps1

# Run inference
if ($Continuous) {
    Write-Host "Starting continuous prediction mode..." -ForegroundColor Green
    Write-Host "Press Ctrl+C to stop" -ForegroundColor Yellow
    Write-Host ""
    python -m src.infer --model $Model --network $Network --continuous
} else {
    Write-Host "Making single prediction..." -ForegroundColor Green
    python -m src.infer --model $Model --network $Network
}

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "================================" -ForegroundColor Cyan
    Write-Host "  Inference completed!" -ForegroundColor Green
    Write-Host "================================" -ForegroundColor Cyan
}
