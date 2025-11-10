# Experiment 5: XGBoost Standalone

## Objective
Compare standalone XGBoost performance against Hybrid LSTM-XGBoost model to evaluate the contribution of LSTM temporal feature extraction.

## Configuration

### XGBoost Parameters (Same as Hybrid)
- **n_estimators:** 700
- **max_depth:** 6
- **learning_rate:** 0.045
- **subsample:** 0.85
- **colsample_bytree:** 0.85
- **min_child_weight:** 2
- **gamma:** 0.05
- **reg_alpha:** 0.03
- **reg_lambda:** 1.2
- **early_stopping_rounds:** 70
- **objective:** Custom asymmetric MSE (2.5x under-estimation penalty)

### Data
- **Source:** `data/features.parquet` (4999 blocks)
- **Features:** Same 14 features as hybrid model
- **Split:** 70/15/15 (train/val/test)
- **Normalization:** StandardScaler (features + targets)

## Key Differences from Hybrid

| Aspect | Hybrid Model | XGBoost Only |
|--------|-------------|--------------|
| Features | Original + LSTM temporal | Original only (14 features) |
| Sequence | 20 blocks lookback | Single block (no sequence) |
| Training | LSTM → XGBoost (2 stages) | XGBoost only (1 stage) |
| Complexity | High (LSTM + XGBoost) | Low (XGBoost only) |

## Expected Outcomes

**Hypothesis:** Hybrid model should outperform standalone XGBoost due to temporal feature extraction, especially in:
- Capturing sequential patterns
- Predicting sudden gas fee spikes
- Overall prediction accuracy

**Metrics to Compare:**
- MAE, MAPE, R²
- Under-estimation rate
- Training time
- Model complexity (# parameters)

## Results

Will be filled after training completes.
