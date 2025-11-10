# Experiment 5: XGBoost Standalone Training & Comparison

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host " EXP5: XGBoost Standalone vs Hybrid" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

# Get script directory
$SCRIPT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path
$PROJECT_ROOT = Split-Path -Parent (Split-Path -Parent $SCRIPT_DIR)

Write-Host "📂 Project root: $PROJECT_ROOT" -ForegroundColor Gray
Write-Host "📂 Experiment dir: $SCRIPT_DIR`n" -ForegroundColor Gray

# Change to project root for data access
Set-Location $PROJECT_ROOT

# Verify data exists
$DATA_FILE = "data\features.parquet"
if (-not (Test-Path $DATA_FILE)) {
    Write-Host "❌ Error: $DATA_FILE not found!" -ForegroundColor Red
    Write-Host "   Please run data preparation first.`n" -ForegroundColor Yellow
    exit 1
}

Write-Host "✓ Data file found: $DATA_FILE`n" -ForegroundColor Green

# Train XGBoost standalone model
Write-Host "========================================" -ForegroundColor Yellow
Write-Host " STEP 1: Training XGBoost Standalone" -ForegroundColor Yellow
Write-Host "========================================`n" -ForegroundColor Yellow

$CONFIG_FILE = "$SCRIPT_DIR\config.yaml"
$OUTPUT_DIR = "$SCRIPT_DIR\model"

Write-Host "Configuration: $CONFIG_FILE" -ForegroundColor Gray
Write-Host "Output dir: $OUTPUT_DIR`n" -ForegroundColor Gray

Write-Host "⏳ Training in progress (this may take 5-15 minutes)...`n" -ForegroundColor Cyan

python "$SCRIPT_DIR\train_xgboost_only.py" `
    --config $CONFIG_FILE `
    --data $DATA_FILE `
    --output $OUTPUT_DIR

if ($LASTEXITCODE -ne 0) {
    Write-Host "`n❌ Training failed!`n" -ForegroundColor Red
    exit 1
}

Write-Host "`n✓ Training complete!`n" -ForegroundColor Green

# Compare models
Write-Host "========================================" -ForegroundColor Yellow
Write-Host " STEP 2: Comparing Models" -ForegroundColor Yellow
Write-Host "========================================`n" -ForegroundColor Yellow

Set-Location $SCRIPT_DIR
python compare_models.py

if ($LASTEXITCODE -ne 0) {
    Write-Host "`n⚠️  Comparison completed with warnings`n" -ForegroundColor Yellow
} else {
    Write-Host "`n✓ Comparison complete!`n" -ForegroundColor Green
}

# Summary
Write-Host "========================================" -ForegroundColor Cyan
Write-Host " EXPERIMENT COMPLETE" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

Write-Host "📁 Results location: $SCRIPT_DIR" -ForegroundColor White
Write-Host "   - model/             (Trained XGBoost model)" -ForegroundColor Gray
Write-Host "   - metrics.json       (Evaluation metrics)" -ForegroundColor Gray
Write-Host "   - comparison_results.csv (Comparison table)`n" -ForegroundColor Gray

Write-Host "📊 Next steps:" -ForegroundColor Yellow
Write-Host "   1. Review comparison_results.csv" -ForegroundColor White
Write-Host "   2. Check metrics.json for detailed results" -ForegroundColor White
Write-Host "   3. Document findings in thesis`n" -ForegroundColor White

Set-Location $PROJECT_ROOT
