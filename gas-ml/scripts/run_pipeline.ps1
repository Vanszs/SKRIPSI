# PowerShell script untuk run complete pipeline
# Usage: .\run_pipeline.ps1

Write-Host ""
Write-Host "======================================" -ForegroundColor Cyan
Write-Host "  Gas Fee ML - Complete Pipeline" -ForegroundColor Cyan
Write-Host "======================================" -ForegroundColor Cyan
Write-Host ""

$ErrorActionPreference = "Stop"

try {
    # Step 1: Fetch data
    Write-Host "[1/5] Fetching block data..." -ForegroundColor Yellow
    & .\scripts\run_fetch.ps1 -NBlocks 8000
    
    Write-Host ""
    
    # Step 2: Generate features
    Write-Host "[2/5] Generating features..." -ForegroundColor Yellow
    & .\scripts\run_features.ps1
    
    Write-Host ""
    
    # Step 3: Train model
    Write-Host "[3/5] Training model..." -ForegroundColor Yellow
    & .\scripts\run_train.ps1
    
    Write-Host ""
    
    # Step 4: Run backtest
    Write-Host "[4/5] Running backtest..." -ForegroundColor Yellow
    & .\scripts\run_backtest.ps1
    
    Write-Host ""
    
    # Step 5: Test inference
    Write-Host "[5/5] Testing inference..." -ForegroundColor Yellow
    & .\scripts\run_infer.ps1
    
    Write-Host ""
    Write-Host "======================================" -ForegroundColor Cyan
    Write-Host "  Pipeline completed successfully!" -ForegroundColor Green
    Write-Host "======================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Next steps:" -ForegroundColor Yellow
    Write-Host "  - Check models/ for trained model" -ForegroundColor White
    Write-Host "  - Check reports/ for backtest results" -ForegroundColor White
    Write-Host "  - Run continuous inference: .\scripts\run_infer.ps1 -Continuous" -ForegroundColor White
    Write-Host ""
    
} catch {
    Write-Host ""
    Write-Host "======================================" -ForegroundColor Red
    Write-Host "  Pipeline failed!" -ForegroundColor Red
    Write-Host "======================================" -ForegroundColor Red
    Write-Host "Error: $_" -ForegroundColor Red
    exit 1
}
