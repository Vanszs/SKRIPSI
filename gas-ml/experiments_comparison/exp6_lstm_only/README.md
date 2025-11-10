# Experiment 6: LSTM Standalone

## Objective
Evaluate standalone LSTM performance against Hybrid and XGBoost models to isolate the contribution of temporal feature learning.

## Configuration

### LSTM Parameters (Same as Hybrid)
- **hidden_size:** 192
- **num_layers:** 2
- **dropout:** 0.25
- **bidirectional:** False
- **sequence_length:** 20 blocks
- **batch_size:** 48
- **learning_rate:** 0.0008
- **epochs:** 120
- **early_stopping_patience:** 22

### Training Setup
- **Loss Function:** Asymmetric MSE (2.5x penalty for under-estimation)
- **Optimizer:** Adam with weight_decay=1.5e-4
- **Gradient Clipping:** max_norm=1.0
- **Data Split:** 70/15/15 (train/val/test) - SAME as hybrid
- **Normalization:** StandardScaler for features AND targets

## Key Points for Fair Comparison

| Aspect | Hybrid | XGBoost Only | LSTM Only |
|--------|--------|--------------|-----------|
| Data Size | 3480/730/732 | 3480/730/732 | 3480/730/732 ✓ |
| Loss Function | Asymmetric MSE | Asymmetric MSE | Asymmetric MSE ✓ |
| Sequence | 20 blocks | Aligned (drop 19) | 20 blocks ✓ |
| Normalization | Features+Targets | Features+Targets | Features+Targets ✓ |
| Output | XGBoost on LSTM features | XGBoost on original | LSTM direct prediction |

## Expected Insights

**Hypothesis:**
- LSTM-only will capture temporal patterns but may lack XGBoost's non-linear feature interaction capability
- Performance should be between XGBoost-only and Hybrid
- Will show pure contribution of sequence modeling

## Results

Will be filled after training completes.
