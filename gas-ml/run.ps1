<#
.SYNOPSIS
    Unified Gas ML Pipeline Runner
    
.DESCRIPTION
    Single PowerShell script untuk menjalankan semua operasi gas-ml pipeline.
    Menggantikan multiple run_*.ps1 scripts dengan single unified interface.
    
.EXAMPLE
    .\run.ps1 predict
    .\run.ps1 train -Blocks 8000
    .\run.ps1 backtest
    .\run.ps1 fetch -NBlocks 5000
    
.NOTES
    Version: 2.0 (Unified)
    Author: Gas ML Project
#>

param(
    [Parameter(Mandatory=$true, Position=0)]
    [ValidateSet('predict', 'train', 'fetch', 'features', 'backtest', 'info', 'test', 'select-features')]
    [string]$Command,
    
    [Parameter(Mandatory=$false)]
    [string]$RpcUrl = "https://eth.llamarpc.com",
    
    [Parameter(Mandatory=$false)]
    [int]$NBlocks = 5000,
    
    [Parameter(Mandatory=$false)]
    [switch]$Continuous,
    
    [Parameter(Mandatory=$false)]
    [string]$Data = "data/blocks_5k.csv",
    
    [Parameter(Mandatory=$false)]
    [string]$Config = "cfg/exp.yaml",
    
    [Parameter(Mandatory=$false)]
    [string]$Features = "data/features.parquet",
    
    [Parameter(Mandatory=$false)]
    [string]$SelectedFeatures,
    
    [Parameter(Mandatory=$false)]
    [switch]$Verbose
)

# Colors
$SuccessColor = "Green"
$ErrorColor = "Red"
$InfoColor = "Cyan"
$WarningColor = "Yellow"

function Write-Step {
    param([string]$Message)
    Write-Host "`n🔹 $Message" -ForegroundColor $InfoColor
}

function Write-Success {
    param([string]$Message)
    Write-Host "✅ $Message" -ForegroundColor $SuccessColor
}

function Write-ErrorMsg {
    param([string]$Message)
    Write-Host "❌ $Message" -ForegroundColor $ErrorColor
}

# Check if virtual environment exists
if (-not (Test-Path "venv\Scripts\python.exe")) {
    Write-ErrorMsg "Virtual environment not found!"
    Write-Host "Please run: python -m venv venv" -ForegroundColor $WarningColor
    exit 1
}

$pythonExe = "venv\Scripts\python.exe"

# Check if .env exists
if (-not (Test-Path ".env")) {
    Write-Warning ".env file not found. Using default RPC URL."
}

Write-Host @"
╔════════════════════════════════════════════════════════════╗
║        Gas Fee ML - Unified Pipeline Runner v2.0          ║
╚════════════════════════════════════════════════════════════╝
"@ -ForegroundColor $InfoColor

switch ($Command) {
    'predict' {
        Write-Step "Running real-time gas fee prediction..."
        
        $args = @("cli.py", "predict", "--rpc", $RpcUrl)
        
        if ($Continuous) {
            $args += "--continuous"
            Write-Host "📊 Continuous monitoring mode enabled" -ForegroundColor $InfoColor
        }
        
        & $pythonExe $args
        
        if ($LASTEXITCODE -eq 0) {
            Write-Success "Prediction completed successfully!"
        } else {
            Write-ErrorMsg "Prediction failed with exit code $LASTEXITCODE"
            exit 1
        }
    }
    
    'info' {
        Write-Step "Displaying model information..."
        
        & $pythonExe cli.py info
        
        if ($LASTEXITCODE -eq 0) {
            Write-Success "Model info displayed"
        }
    }
    
    'fetch' {
        Write-Step "Fetching $NBlocks blocks from ..."
        
        $args = @("-m", "src.fetch", "--network", "", "--n-blocks", $NBlocks)
        
        if ($Verbose) {
            $args += "--verbose"
        }
        
        & $pythonExe $args
        
        if ($LASTEXITCODE -eq 0) {
            Write-Success "Data fetched successfully! Output: data/blocks.csv"
        } else {
            Write-ErrorMsg "Fetch failed"
            exit 1
        }
    }
    
    'features' {
        Write-Step "Generating features from block data..."
        
        $inputFile = if (Test-Path "data/blocks.csv") { "data/blocks.csv" } else { $Data }
        
        $args = @("-m", "src.features", "--in", $inputFile, "--out", "data/features.parquet")
        
        & $pythonExe $args
        
        if ($LASTEXITCODE -eq 0) {
            Write-Success "Features generated! Output: data/features.parquet"
        } else {
            Write-ErrorMsg "Feature generation failed"
            exit 1
        }
    }
    
    'select-features' {
        Write-Step "Running feature selection analysis..."
        
        $method = Read-Host "Select method [analyze/importance] (default: analyze)"
        if ([string]::IsNullOrEmpty($method)) { $method = "analyze" }
        
        $args = @("-m", "src.feature_selector", $method, "--in", $Features, "--out", "data/selected_features.txt")
        
        if ($method -eq "importance") {
            if (-not (Test-Path "models/xgb.bin")) {
                Write-ErrorMsg "XGBoost model not found! Train model first."
                exit 1
            }
            $args += @("--model", "models/xgb.bin")
        }
        
        & $pythonExe $args
        
        if ($LASTEXITCODE -eq 0) {
            Write-Success "Feature selection completed! Output: data/selected_features.txt"
        }
    }
    
    'train' {
        Write-Step "Training hybrid LSTM-XGBoost model..."
        
        if (-not (Test-Path $Config)) {
            Write-ErrorMsg "Config file not found: $Config"
            exit 1
        }
        
        if (-not (Test-Path $Features)) {
            Write-ErrorMsg "Features file not found: $Features"
            Write-Host "Run: .\run.ps1 fetch && .\run.ps1 features" -ForegroundColor $WarningColor
            exit 1
        }
        
        $args = @("-m", "src.train", "--cfg", $Config, "--in", $Features)
        
        if ($SelectedFeatures -and (Test-Path $SelectedFeatures)) {
            Write-Host "📊 Using selected features from $SelectedFeatures" -ForegroundColor $InfoColor
            $args += @("--selected-features", $SelectedFeatures)
        }
        
        if ($Verbose) {
            $args += "--verbose"
        }
        
        Write-Host "`n⏱️  Training may take 10-60 minutes depending on CPU/GPU..." -ForegroundColor $WarningColor
        
        & $pythonExe $args
        
        if ($LASTEXITCODE -eq 0) {
            Write-Success "Training completed! Models saved to models/"
        } else {
            Write-ErrorMsg "Training failed"
            exit 1
        }
    }
    
    'backtest' {
        Write-Step "Running backtest analysis..."
        
        if (-not (Test-Path $Data)) {
            Write-ErrorMsg "Data file not found: $Data"
            exit 1
        }
        
        & $pythonExe cli.py backtest --data $Data
        
        if ($LASTEXITCODE -eq 0) {
            Write-Success "Backtest completed! Check reports/"
        } else {
            Write-ErrorMsg "Backtest failed"
            exit 1
        }
    }
    
    'test' {
        Write-Step "Running unit tests..."
        
        & $pythonExe -m pytest tests/ -v
        
        if ($LASTEXITCODE -eq 0) {
            Write-Success "All tests passed!"
        } else {
            Write-ErrorMsg "Some tests failed"
            exit 1
        }
    }
}

Write-Host "`n" -NoNewline
