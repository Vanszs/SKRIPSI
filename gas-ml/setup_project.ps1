# PowerShell script untuk initial setup
# Usage: .\setup_project.ps1

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Gas Fee ML - Project Setup" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

$ErrorActionPreference = "Stop"

try {
    # Check Python version
    Write-Host "[1/6] Checking Python version..." -ForegroundColor Yellow
    $pythonVersion = python --version 2>&1
    Write-Host "  Found: $pythonVersion" -ForegroundColor Green
    
    if (-not ($pythonVersion -match "Python 3\.(1[0-9]|[2-9][0-9])")) {
        Write-Host "  Warning: Python 3.10+ recommended" -ForegroundColor Yellow
    }
    
    # Create virtual environment
    Write-Host ""
    Write-Host "[2/6] Creating virtual environment..." -ForegroundColor Yellow
    
    if (Test-Path "venv") {
        Write-Host "  Virtual environment already exists" -ForegroundColor Green
    } else {
        python -m venv venv
        Write-Host "  Virtual environment created" -ForegroundColor Green
    }
    
    # Activate virtual environment
    Write-Host ""
    Write-Host "[3/6] Activating virtual environment..." -ForegroundColor Yellow
    & .\venv\Scripts\Activate.ps1
    Write-Host "  Virtual environment activated" -ForegroundColor Green
    
    # Install dependencies
    Write-Host ""
    Write-Host "[4/6] Installing dependencies..." -ForegroundColor Yellow
    Write-Host "  This may take several minutes..." -ForegroundColor Gray
    pip install -r requirements.txt --quiet
    Write-Host "  Dependencies installed" -ForegroundColor Green
    
    # Setup environment file
    Write-Host ""
    Write-Host "[5/6] Setting up environment file..." -ForegroundColor Yellow
    
    if (Test-Path ".env") {
        Write-Host "  .env file already exists" -ForegroundColor Green
    } else {
        Copy-Item ".env.example" ".env"
        Write-Host "  .env file created from template" -ForegroundColor Green
        Write-Host "  Please edit .env with your RPC URL" -ForegroundColor Yellow
    }
    
    # Test connection
    Write-Host ""
    Write-Host "[6/6] Testing connection..." -ForegroundColor Yellow
    
    $testOutput = python -m src.rpc 2>&1
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  Connection test successful!" -ForegroundColor Green
    } else {
        Write-Host "  Connection test failed - please check .env configuration" -ForegroundColor Yellow
    }
    
    # Summary
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "  Setup Complete!" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Next steps:" -ForegroundColor Yellow
    Write-Host "  1. Edit .env file dengan RPC URL yang valid" -ForegroundColor White
    Write-Host "     - Free RPC: https://rpc..org" -ForegroundColor Gray
    Write-Host "     - Or get Infura key: https://infura.io" -ForegroundColor Gray
    Write-Host ""
    Write-Host "  2. Test connection:" -ForegroundColor White
    Write-Host "     python -m src.rpc" -ForegroundColor Gray
    Write-Host ""
    Write-Host "  3. Run complete pipeline:" -ForegroundColor White
    Write-Host "     .\scripts\run_pipeline.ps1" -ForegroundColor Gray
    Write-Host ""
    Write-Host "  4. Or run step by step:" -ForegroundColor White
    Write-Host "     .\scripts\run_fetch.ps1" -ForegroundColor Gray
    Write-Host "     .\scripts\run_features.ps1" -ForegroundColor Gray
    Write-Host "     .\scripts\run_train.ps1" -ForegroundColor Gray
    Write-Host ""
    Write-Host "Documentation:" -ForegroundColor Yellow
    Write-Host "  - README.md  : Quick start guide" -ForegroundColor Gray
    Write-Host "  - SETUP.md   : Detailed setup & troubleshooting" -ForegroundColor Gray
    Write-Host ""
    
} catch {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Red
    Write-Host "  Setup Failed!" -ForegroundColor Red
    Write-Host "========================================" -ForegroundColor Red
    Write-Host "Error: $_" -ForegroundColor Red
    Write-Host ""
    Write-Host "Please check:" -ForegroundColor Yellow
    Write-Host "  - Python 3.10+ is installed" -ForegroundColor White
    Write-Host "  - Internet connection is available" -ForegroundColor White
    Write-Host "  - Execution policy allows scripts:" -ForegroundColor White
    Write-Host "    Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser" -ForegroundColor Gray
    Write-Host ""
    exit 1
}
