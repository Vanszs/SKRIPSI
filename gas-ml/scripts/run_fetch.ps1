# PowerShell script untuk fetch block data dari Sepolia
# Usage: .\run_fetch.ps1 [-NBlocks 8000] [-Output "blocks.csv"]

param(
    [int]$NBlocks = 8000,
    [string]$Output = "blocks.csv",
    [string]$Network = "sepolia"
)

Write-Host "================================" -ForegroundColor Cyan
Write-Host "  Gas Fee ML - Data Fetcher" -ForegroundColor Cyan
Write-Host "================================" -ForegroundColor Cyan
Write-Host ""

# Check if virtual environment exists
if (-Not (Test-Path "venv")) {
    Write-Host "Error: Virtual environment not found!" -ForegroundColor Red
    Write-Host "Please run: python -m venv venv" -ForegroundColor Yellow
    exit 1
}

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Green
& .\venv\Scripts\Activate.ps1

# Run fetch script
Write-Host "Fetching $NBlocks blocks from $Network..." -ForegroundColor Green
python -m src.fetch --network $Network --n-blocks $NBlocks --output $Output

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "================================" -ForegroundColor Cyan
    Write-Host "  Data fetch completed!" -ForegroundColor Green
    Write-Host "================================" -ForegroundColor Cyan
} else {
    Write-Host ""
    Write-Host "Error: Data fetch failed!" -ForegroundColor Red
    exit 1
}
