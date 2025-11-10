# Gas Fee ML Project Structure

## 📁 Directory Organization

```
gas-ml/
├── src/                    # Core source code
│   ├── train.py           # Training pipeline
│   ├── stack.py           # Hybrid LSTM-XGBoost model
│   ├── infer.py           # Inference engine
│   ├── policy.py          # Gas fee policy logic
│   ├── features.py        # Feature engineering
│   ├── evaluate.py        # Model evaluation
│   └── fetch.py           # Data fetching from RPC
│
├── models/                 # Production model files (CURRENT)
│   ├── lstm.pt            # LSTM weights
│   ├── xgb.bin            # XGBoost model
│   ├── hybrid_metadata.pkl # Model configuration
│   ├── scaler.pkl         # Feature scaler
│   ├── target_scaler.pkl  # Target scaler (for denormalization)
│   ├── metrics.json       # Evaluation metrics
│   └── training_info.json # Training metadata
│
├── data/                   # Dataset files
│   ├── features.parquet   # Engineered features (4999 blocks)
│   ├── blocks_5k.csv      # Raw block data
│   ├── selected_features.txt # Feature selection list
│   └── test/              # Test data for experiments
│
├── cfg/                    # Configuration files
│   └── exp.yaml           # Experiment hyperparameters
│
├── outputs/                # All generated outputs
│   ├── predictions/       # Inference results
│   ├── logs/              # Training logs
│   └── archived_models/   # Previous model versions
│       ├── v1_before_normalization/ # First training (37.89% under-est)
│       └── backup/        # Other backups
│
├── docs/                   # Documentation
│   ├── audit/             # Audit reports & fixes
│   │   ├── AUDIT_REPORT.md
│   │   ├── FIXES_APPLIED.md
│   │   └── RETRAINING_CHECKLIST.md
│   ├── CLI_USAGE.md       # Command-line interface guide
│   ├── GPU_TRAINING.md    # GPU setup instructions
│   └── IMPROVEMENT_STRATEGY.md # Optimization strategies
│
├── notebooks/              # Jupyter notebooks for analysis
│   ├── 01_eda_gas_fee.ipynb
│   ├── 02_model_evaluation.ipynb
│   └── 03_xgboost_model_analysis.ipynb
│
├── experiments_comparison/ # Historical experiment tracking
│   ├── exp1_lstm_no_temporal/
│   ├── exp2_xgb_stacking/
│   ├── exp3_tcn/
│   └── exp4_tft/
│
├── tests/                  # Unit tests
│   └── test_metrics.py
│
├── cli.py                  # Command-line interface
├── requirements.txt        # Python dependencies
├── setup_project.ps1      # Setup script
└── run.ps1                # Quick training script
```

## 🎯 Current Model Status

**Version:** V2 (Production-Ready with Target Normalization)
**Date:** November 10, 2025
**Location:** `models/`

### Performance Metrics
- **MAE:** 1.95 Gwei
- **MAPE:** 2.03% (excellent, baseline 15-25%)
- **R²:** 0.9723
- **Under-estimation Rate:** 10.40% (acceptable for production)
- **Hit Rate @ ε=5%:** 88.65%

### Key Improvements Applied
1. ✅ Fixed LSTM placeholder prediction head
2. ✅ Fixed data leakage (shuffle=False)
3. ✅ Target normalization with StandardScaler
4. ✅ Asymmetric loss (2.5x penalty for under-estimation)
5. ✅ Gradient clipping + weight decay
6. ✅ Buffer tuning (25% safety buffer)

## 🚀 Quick Start

### Training
```powershell
python src\train.py --cfg cfg\exp.yaml --in data\features.parquet
```

### Inference
```powershell
python cli.py predict
```

### Model Info
```powershell
python cli.py info
```

## 📊 Model Architecture

**Hybrid Stacking Approach:**
1. **LSTM Feature Extractor**
   - Input: 20-block sequences × 14 features
   - Hidden: 192 units, 2 layers, dropout 0.25
   - Output: Temporal features + initial prediction

2. **XGBoost Final Predictor**
   - Input: LSTM features + original features
   - Trees: 700, max_depth: 6
   - Custom asymmetric objective (2.5x under-penalty)

## 🔧 Configuration

Key parameters in `cfg/exp.yaml`:
- `buffer_multiplier: 1.25` - 25% safety buffer
- `priority_fee_percentile: 0.6` - 60th percentile
- `max_fee_multiplier: 1.8` - Max fee cap
- `sequence_length: 20` - LSTM lookback window

## 📈 Data Flow

```
blocks_5k.csv → features.parquet → normalize → LSTM → XGBoost → predictions
                                      ↓
                                  target_scaler
                                      ↓
                                  denormalize
```

## 🎓 For Thesis Documentation

### Critical Files for Analysis
- `models/metrics.json` - Quantitative results
- `models/training_info.json` - Training details
- `docs/audit/AUDIT_REPORT.md` - Bug discovery & resolution
- `outputs/archived_models/` - Model evolution comparison

### Key Findings
- **V1 vs V2:** Slight accuracy decrease (1.49%→2.03% MAPE) but 72% reduction in under-estimation (37.89%→10.40%)
- **Trade-off:** Production safety prioritized over perfect accuracy
- **GPU Acceleration:** RTX 3050 enables 10-60 min training vs hours on CPU

## ⚠️ Important Notes

1. **Never shuffle time series data** - Temporal order must be preserved
2. **Always normalize targets** - Critical for LSTM training stability
3. **Asymmetric loss required** - Under-estimation causes transaction failures
4. **Target scaler persistence** - Must save/load target_scaler.pkl for denormalization

## 📝 Maintenance

### Before Retraining
1. Backup current models: `Copy-Item models outputs\archived_models\backup_YYYYMMDD -Recurse`
2. Update `cfg/exp.yaml` if needed
3. Verify GPU availability: `python -c "import torch; print(torch.cuda.is_available())"`

### After Retraining
1. Compare metrics.json with previous version
2. Test inference: `python cli.py predict`
3. Commit changes to git
