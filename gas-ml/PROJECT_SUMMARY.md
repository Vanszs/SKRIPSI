# Project Summary Document

## 📋 Project Overview

**Title**: Model Prediksi Gas Fee Ethereum Berbasis Hybrid LSTM–XGBoost untuk Optimasi Biaya Transaksi Pasca-EIP-1559

**Type**: Research Project / Undergraduate Thesis

**Technology Stack**:
- Python 3.10+
- PyTorch (LSTM)
- XGBoost (Gradient Boosting)
- Web3.py (Ethereum)
- Pandas/NumPy (Data Processing)

**Network**: Ethereum Sepolia Testnet

## 🎯 Research Objectives

1. Mengembangkan model prediksi gas fee menggunakan hybrid LSTM-XGBoost
2. Meningkatkan akurasi prediksi dibanding baseline EIP-1559
3. Mengoptimalkan biaya transaksi dengan rekomendasi gas fee yang cerdas
4. Mengevaluasi performance model dengan metrics komprehensif

## 🏗️ System Architecture

```
Data Collection → Feature Engineering → Model Training → Inference → Recommendation
     (RPC)            (Temporal)         (LSTM+XGBoost)    (Real-time)   (Policy)
```

### Components:

1. **RPC Client** (`src/rpc.py`)
   - Connect ke Sepolia testnet
   - Fetch block data dengan retry mechanism
   - Handle multiple RPC endpoints

2. **Data Fetcher** (`src/fetch.py`)
   - Download historical blocks
   - Extract features (baseFee, gasUsed, utilization, etc.)
   - Save to CSV format

3. **Feature Engineering** (`src/features.py`)
   - Temporal features (hour, day_of_week)
   - Statistical features (EMA, rolling stats)
   - Delta features (changes between blocks)
   - Utilization metrics

4. **Hybrid Model** (`src/stack.py`)
   - LSTM: Extract temporal patterns
   - XGBoost: Meta-learner for prediction
   - Stacking ensemble approach

5. **Training Pipeline** (`src/train.py`)
   - Data splitting (train/val/test)
   - Feature normalization
   - Model training with early stopping
   - Model persistence

6. **Inference Engine** (`src/infer.py`)
   - Real-time prediction
   - Continuous monitoring mode
   - Network status display

7. **Policy Recommender** (`src/policy.py`)
   - Calculate buffered baseFee
   - Estimate priority fee
   - Generate maxFeePerGas recommendation
   - Confidence scoring

8. **Evaluation & Backtesting** (`src/evaluate.py`, `src/backtest.py`)
   - MAE, MAPE, RMSE, R²
   - Hit@ε (accuracy within tolerance)
   - Under-estimation rate
   - Cost-saving analysis vs baseline

## 📊 Key Features

### Model Features (Input):
- baseFeePerGas (current)
- gasUsed, gasLimit, utilization
- Transaction count
- Delta features (changes)
- EMA features (short/long term)
- Rolling statistics
- Temporal features (hour, day)

### Target (Output):
- baseFee_next (next block's base fee)

### Derived Recommendations:
- priorityFee (tip for miners)
- maxFeePerGas (with safety buffer)
- Confidence level
- Transaction status (Optimal/Moderate/Low)

## 📈 Expected Performance

| Metric | Target | Baseline |
|--------|--------|----------|
| MAE | 2-5 Gwei | 5-10 Gwei |
| MAPE | 5-15% | 15-25% |
| Hit@5% | 70-85% | 50-60% |
| Under-est Rate | <15% | 20-30% |
| Cost Saving | 10-30% | 0% |

## 🚀 Usage Workflow

### For Development/Testing:
```powershell
1. Setup environment: .\setup_project.ps1
2. Configure .env with RPC URL
3. Run pipeline: .\scripts\run_pipeline.ps1
4. Check results in reports/
```

### For Research/Analysis:
```powershell
1. Fetch data: python -m src.fetch --n-blocks 8000
2. Generate features: python -m src.features --in data/blocks.csv --out data/features.parquet
3. Train model: python -m src.train --cfg cfg/exp.yaml --in data/features.parquet
4. Run backtest: python -m src.backtest --model models/ --data data/features.parquet
5. Analyze results in reports/backtest_*.json
```

### For Real-time Inference:
```powershell
# Single prediction
python -m src.infer --model models/

# Continuous monitoring
python -m src.infer --model models/ --continuous
```

## 📁 File Structure

```
gas-ml/
├── src/              # Source code
├── cfg/              # Configuration
├── data/             # Datasets
├── models/           # Trained models
├── reports/          # Evaluation results
├── scripts/          # Automation scripts
├── tests/            # Unit tests
├── .env              # Environment config
├── requirements.txt  # Dependencies
├── README.md         # Main documentation
├── SETUP.md          # Setup guide
└── CHANGELOG.md      # Version history
```

## 🎓 Academic Contributions

1. **Novel Approach**: Hybrid LSTM-XGBoost untuk gas fee prediction
2. **Real-world Application**: Testnet implementation dengan RPC integration
3. **Comprehensive Evaluation**: Multiple metrics termasuk cost-saving analysis
4. **Reproducible Research**: Complete pipeline dengan automation scripts
5. **Open Framework**: Extensible untuk future research

## 📚 Key References

1. EIP-1559: Fee market change for ETH 1.0 chain
2. XGBoost: A Scalable Tree Boosting System (Chen & Guestrin, 2016)
3. Long Short-Term Memory Networks (Hochreiter & Schmidhuber, 1997)
4. Ethereum Gas Price Prediction Studies

## 🔮 Future Work

- Multi-network support (Mainnet, L2s)
- Transformer-based models
- MEV-aware predictions
- Web dashboard
- REST API
- Online learning capability

## 📝 Notes for Thesis

### Methodology Section:
- Data collection dari Sepolia (timestamp, blocks, features)
- Feature engineering techniques
- Model architecture (LSTM specs, XGBoost params)
- Training procedure (splits, epochs, early stopping)
- Evaluation metrics definition

### Results Section:
- Model performance metrics
- Comparison dengan baseline
- Cost-saving analysis
- Prediction examples
- Backtesting results

### Discussion Section:
- Model strengths dan limitations
- Testnet vs Mainnet considerations
- Practical implications
- Future improvements

### Implementation Section:
- Technology choices justification
- System architecture
- Deployment considerations
- Scalability analysis

---

**Last Updated**: 2025-01-27
**Status**: Implementation Complete ✅
**Next Phase**: Experimentation & Analysis
