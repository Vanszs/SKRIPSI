# PowerShell script untuk run tests
# Usage: .\run_tests.ps1

Write-Host "================================" -ForegroundColor Cyan
Write-Host "  Gas Fee ML - Running Tests" -ForegroundColor Cyan
Write-Host "================================" -ForegroundColor Cyan
Write-Host ""

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Green
& .\venv\Scripts\Activate.ps1

# Run pytest
Write-Host "Running unit tests..." -ForegroundColor Green
pytest tests/ -v --cov=src --cov-report=html --cov-report=term

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "================================" -ForegroundColor Cyan
    Write-Host "  All tests passed!" -ForegroundColor Green
    Write-Host "  Coverage report: htmlcov/index.html" -ForegroundColor Green
    Write-Host "================================" -ForegroundColor Cyan
} else {
    Write-Host ""
    Write-Host "Error: Some tests failed!" -ForegroundColor Red
    exit 1
}
