# 📋 RE-TRAINING CHECKLIST

## Prerequisites ✅

- [x] Virtual environment activated
- [x] All dependencies installed (`requirements.txt`)
- [x] Data files exist (`data/blocks_5k.csv`, `data/features.parquet`)
- [x] Code fixes applied (`src/stack.py`)
- [x] Configuration ready (`cfg/exp.yaml`)

## Re-training Steps

### Step 1: Verify Data Exists
```powershell
# Check data files
if (Test-Path data/features.parquet) { 
    Write-Host "✓ features.parquet exists" 
} else { 
    Write-Host "✗ Need to generate features first"
    Write-Host "  Run: .\run.ps1 features"
}
```

### Step 2: Backup Old Models (if any)
```powershell
# Optional: backup old models
if (Test-Path models/xgboost_only.bin) {
    Move-Item models/xgboost_only.bin models/xgboost_only.bin.backup
}
```

### Step 3: Train Model
```powershell
# Start training
.\run.ps1 train

# Or manual:
# python -m src.train --cfg cfg/exp.yaml --in data/features.parquet
```

**Expected Duration**: 10-60 minutes (depending on CPU/GPU)

**Expected Output**:
```
Training LSTM feature extractor...
Epoch 10/120 - Train Loss: 0.xxxx, Val Loss: 0.xxxx
...
✓ LSTM training complete
Extracting LSTM features...
Training XGBoost meta-learner...
✓ XGBoost training complete

EVALUATION METRICS
====================
MAE:  X.XX Gwei
MAPE: X.XX%
R²:   0.XXXX
Hit@5%: XX.XX%
====================

✓ Models saved to models/
```

### Step 4: Verify Model Files Generated
```powershell
# Check all required files exist
$required = @('lstm.pt', 'xgb.bin', 'hybrid_metadata.pkl', 'scaler.pkl', 'metrics.json')
foreach ($file in $required) {
    if (Test-Path "models/$file") {
        Write-Host "✓ models/$file" -ForegroundColor Green
    } else {
        Write-Host "✗ models/$file MISSING" -ForegroundColor Red
    }
}
```

**Expected Output**:
```
✓ models/lstm.pt
✓ models/xgb.bin
✓ models/hybrid_metadata.pkl
✓ models/scaler.pkl
✓ models/metrics.json
```

### Step 5: Run System Validation
```powershell
python validate_system.py
```

**Expected Output**:
```
Test Results:
  ✓ Passed:  36
  ✗ Failed:  0
  ⚠ Warnings: 1
  Pass Rate: 100.0%

✓ System is PRODUCTION-READY!
```

### Step 6: Test Model Loading
```powershell
python cli.py info
```

**Expected Output**:
```
============================================================
🧠 MODEL INFORMATION
============================================================

📊 Performance Metrics:
   MAE:              X.XXXX Gwei
   MAPE:             X.XX%
   RMSE:             X.XXXX Gwei
   R²:               0.XXXX
   Hit@5%:           XX.XX%
   Under-estimation: XX.XX%

🏗️  Architecture:
   Type:             Hybrid LSTM → XGBoost
   LSTM hidden:      192
   LSTM layers:      2
   ...
```

### Step 7: Test Prediction (Optional - needs RPC)
```powershell
# Test with public RPC
python cli.py predict --rpc https://eth.llamarpc.com
```

### Step 8: Run Unit Tests
```powershell
pytest tests/ -v
```

**Expected**: All tests should pass

## Troubleshooting

### Issue: "No module named 'torch'"
**Fix**: Install dependencies
```powershell
pip install -r requirements.txt
```

### Issue: "CUDA out of memory"
**Fix 1**: Use CPU
```powershell
# Training will automatically use CPU if CUDA unavailable
```

**Fix 2**: Reduce batch size in `cfg/exp.yaml`
```yaml
training:
  batch_size: 16  # Reduce from 48
```

### Issue: Training very slow
**Fix**: Normal on CPU, 20-60 minutes expected

### Issue: "FileNotFoundError: data/features.parquet"
**Fix**: Generate features first
```powershell
.\run.ps1 features
```

## Post-Training Validation

### ✅ All Files Generated
- [ ] `models/lstm.pt` exists
- [ ] `models/xgb.bin` exists
- [ ] `models/hybrid_metadata.pkl` exists
- [ ] `models/scaler.pkl` exists
- [ ] `models/metrics.json` exists
- [ ] `models/training_info.json` exists

### ✅ Metrics Are Reasonable
- [ ] MAE < 10 Gwei (preferably < 5)
- [ ] MAPE < 15% (preferably < 5%)
- [ ] R² > 0.85 (preferably > 0.90)
- [ ] Hit@5% > 70%
- [ ] Under-estimation < 25%

### ✅ System Validation Passes
- [ ] `python validate_system.py` shows 100% pass rate
- [ ] Model can be loaded: `python cli.py info`
- [ ] No import errors

### ✅ Ready for Production
- [ ] All checklists above completed
- [ ] Documentation reviewed
- [ ] Code committed to git

## Expected Training Time

| Hardware | Expected Duration |
|----------|-------------------|
| CPU Only | 30-60 minutes |
| GPU (CUDA) | 10-20 minutes |
| High-end GPU | 5-10 minutes |

## Next Steps After Success

1. **Commit Changes to Git**
   ```powershell
   git add .
   git commit -m "Fix: LSTM placeholder, data leakage, add validation"
   ```

2. **Update Documentation**
   - Add training results to thesis
   - Document fixes applied
   - Include validation results

3. **Run Experiments**
   - Test different configurations
   - Compare with baselines
   - Analyze results

4. **Deploy (if ready)**
   - Test on testnet
   - Monitor performance
   - Iterate improvements

---

**Good Luck! 🚀**
