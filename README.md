# ⛽ Gas Fee Prediction Model - Hybrid LSTM + XGBoost

> **Model Prediksi Gas Fee Ethereum Berbasis Hybrid LSTM–XGBoost untuk Optimasi Biaya Transaksi Pasca-EIP-1559**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-orange.svg)](https://xgboost.readthedocs.io/)

---

## 🚀 Quick Start

### Installation
```powershell
# Create virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Configure RPC
cp .env.example .env
# Edit .env with your RPC URL
```

### Real-Time Prediction
```bash
# Single prediction
python cli.py predict --rpc https://eth.llamarpc.com

# Continuous monitoring
python cli.py predict --rpc https://eth.llamarpc.com --continuous
```

### Training Custom Model
```powershell
# Complete pipeline
.\run.ps1 fetch -NBlocks 5000
.\run.ps1 features
.\run.ps1 train

# With feature selection
.\run.ps1 select-features
.\run.ps1 train -SelectedFeatures "data\selected_features.txt"
```

---

## 📁 Project Structure

```
gas-ml/
├── cli.py                    # ⭐ Main CLI entry point
├── run.ps1                   # ⭐ Unified PowerShell runner
├── cfg/exp.yaml              # Configuration
├── src/
│   ├── rpc.py               # RPC client
│   ├── fetch.py             # Data fetcher
│   ├── features.py          # Feature engineering
│   ├── feature_selector.py  # Unified feature selection
│   ├── train.py             # Training pipeline
│   ├── stack.py             # Hybrid LSTM-XGBoost
│   ├── infer.py             # Inference engine
│   ├── policy.py            # Fee recommendation
│   └── evaluate.py          # Metrics evaluation
├── data/                     # Datasets
├── models/                   # Trained models
├── reports/                  # Evaluation results
├── notebooks/                # Analysis notebooks
├── docs/                     # Technical documentation
└── tests/                    # Unit tests
```

---

## ⚙️ Commands

### PowerShell Runner (Recommended)
```powershell
.\run.ps1 fetch              # Fetch blockchain data
.\run.ps1 features           # Generate features
.\run.ps1 select-features    # Feature selection
.\run.ps1 train              # Train model
.\run.ps1 predict            # Real-time prediction
.\run.ps1 backtest           # Run backtest
.\run.ps1 test               # Run unit tests
.\run.ps1 info               # Model information
```

### Production CLI
```bash
python cli.py predict --rpc <RPC_URL>
python cli.py predict --rpc <RPC_URL> --continuous
python cli.py info
python cli.py backtest --data data/blocks_5k.csv
```

### Python Modules
```bash
# Data pipeline
python -m src.fetch --network sepolia --n-blocks 8000
python -m src.features --in data/blocks.csv --out data/features.parquet

# Feature selection
python -m src.feature_selector analyze --in data/features.parquet
python -m src.feature_selector importance --model models/xgb.bin --in data/features.parquet
python -m src.feature_selector filter --in data/features.parquet --features data/selected_features.txt --out data/filtered.parquet

# Training
python -m src.train --cfg cfg/exp.yaml --in data/features.parquet
python -m src.train --cfg cfg/exp.yaml --in data/features.parquet --selected-features data/selected_features.txt

# Inference
python -m src.infer --model models/
python -m src.infer --model models/ --continuous --interval 12
```

---

## 🎯 Model Overview

### Architecture
- **LSTM**: Temporal pattern extraction (2 layers, 128 hidden units)
- **XGBoost**: Meta-learner for final prediction (500 trees)
- **Stacking**: Hybrid ensemble approach

### Input Features
- Base metrics: baseFeePerGas, gasUsed, gasLimit, txCount
- Delta features: Δ baseFee, Δ utilization
- EMA features: Short/long-term moving averages
- Statistical: Rolling mean, std, CV
- Temporal: hour_of_day, day_of_week, cyclic encodings

### Output
- `baseFee_next`: Predicted base fee for next block
- `priorityFee`: Recommended priority fee (tip)
- `maxFee`: Maximum fee per gas (with buffer)
- `confidence`: Prediction confidence score

### Performance Targets
| Metric | Target | Baseline |
|--------|--------|----------|
| MAE | 2-5 Gwei | 5-10 Gwei |
| MAPE | 5-15% | 15-25% |
| Hit@5% | 70-85% | 50-60% |
| Cost Saving | 10-30% | 0% |

---

## 📊 Evaluation Metrics

- **MAE**: Mean Absolute Error (Gwei)
- **MAPE**: Mean Absolute Percentage Error (%)
- **RMSE**: Root Mean Squared Error
- **R²**: Coefficient of Determination
- **Under-estimation Rate**: % predictions < actual (failed transactions)
- **Hit@ε**: % predictions within tolerance ε (5%)
- **Cost-saving**: Cost reduction vs baseline EIP-1559

---

## 🧪 Testing

```powershell
# Run all tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=src --cov-report=html --cov-report=term

# Or use unified runner
.\run.ps1 test
```

---

## 🛠️ Troubleshooting

### RPC Connection Issues
```bash
# Try alternative RPC
python cli.py predict --rpc https://ethereum-sepolia.publicnode.com
```

### GPU Support
```powershell
# For CUDA 11.8
pip install torch --index-url https://download.pytorch.org/whl/cu118

# For CPU only
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### Memory Issues
Reduce batch size in `cfg/exp.yaml`:
```yaml
training:
  batch_size: 16  # default: 32
```

---

## 📚 Documentation

- **Technical Docs**: `docs/` directory
  - `CLI_USAGE.md` - Detailed CLI guide
  - `GPU_TRAINING.md` - GPU setup
  - `IMPROVEMENT_STRATEGY.md` - Development roadmap
  - `RATE_LIMITS.md` - API rate limits

---

## 🎓 Academic Use

### For Thesis/Research

This project implements a hybrid deep learning approach for Ethereum gas fee prediction:

**Methodology**:
- Data collection from Sepolia testnet
- Feature engineering (temporal, statistical, domain-specific)
- Hybrid LSTM-XGBoost architecture
- Comprehensive evaluation metrics
- Cost-saving analysis vs baseline

**Citation**:
```bibtex
@software{gas_fee_ml_2025,
  title = {Hybrid LSTM-XGBoost Gas Fee Prediction Model},
  author = {Your Name},
  year = {2025},
  note = {Research implementation for Ethereum gas fee optimization post-EIP-1559}
}
```

---

## 🔮 Future Work

- Multi-network support (Mainnet, Arbitrum, Polygon)
- Transformer-based models
- MEV-aware predictions
- Web dashboard
- REST API
- Online learning capability

---

## 📝 License

MIT License - Academic Research Project

---

## ⚠️ Disclaimer

**For academic research and proof-of-concept only. Use on testnet (Sepolia). Not for production.**

---

**Version**: 2.0 (Clean & Compact)  
**Status**: ✅ Production Ready  
**Last Updated**: November 2025
