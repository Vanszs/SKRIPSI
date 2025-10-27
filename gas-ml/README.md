# ⛽ Gas Fee Prediction Model - Hybrid LSTM + XGBoost

> **Model Prediksi Gas Fee Ethereum Berbasis Hybrid LSTM–XGBoost untuk Optimasi Biaya Transaksi Pasca-EIP-1559**

Proyek riset untuk memprediksi gas fee Ethereum menggunakan pendekatan hybrid machine learning yang menggabungkan LSTM (Long Short-Term Memory) untuk ekstraksi pola temporal dan XGBoost untuk prediksi akhir.

## 📋 Deskripsi

Sistem ini dirancang untuk:
- ⛓️ Mengambil data blok dari Ethereum testnet Sepolia
- 🔧 Membangun fitur temporal dan statistik dari data blok
- 🤖 Melatih model hybrid LSTM→XGBoost
- 📊 Memprediksi `baseFee` blok berikutnya
- 💰 Memberikan rekomendasi `priorityFee`, `maxFee`, dan buffer optimal
- 📈 Mengevaluasi efisiensi biaya vs baseline EIP-1559

## 🏗️ Struktur Proyek

```
gas-ml/
├── cfg/
│   └── exp.yaml              # Konfigurasi eksperimen
├── data/                     # Dataset (CSV/Parquet)
├── models/                   # Model tersimpan
├── reports/                  # Hasil evaluasi
├── src/
│   ├── rpc.py               # RPC client untuk Sepolia
│   ├── fetch.py             # Data fetcher
│   ├── features.py          # Feature engineering
│   ├── stack.py             # LSTM→XGBoost hybrid
│   ├── train.py             # Training pipeline
│   ├── infer.py             # Real-time inference
│   ├── policy.py            # Fee recommendation policy
│   ├── backtest.py          # Backtesting & simulation
│   └── evaluate.py          # Evaluation metrics
├── scripts/
│   ├── run_fetch.sh         # Fetch data
│   ├── run_train.sh         # Train model
│   └── run_infer.sh         # Run inference
└── tests/                   # Unit tests

```

## 🚀 Quick Start

### 1️⃣ Setup Environment

```powershell
# Navigate to project directory
cd gas-ml

# Create virtual environment
python -m venv venv

# Activate (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# If execution policy error:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Install dependencies
pip install -r requirements.txt
```

### 2️⃣ Configure Environment

```powershell
# Copy environment template
cp .env.example .env

# Edit .env dengan Infura/RPC URL (get free API key dari infura.io)
notepad .env
```

**Minimal configuration (public RPC):**
```
SEPOLIA_RPC_URL=https://rpc.sepolia.org
```

### 2.5️⃣ Complete Pipeline (Automated)

```powershell
# Run complete pipeline otomatis (recommended untuk first-time)
.\scripts\run_pipeline.ps1
```

Atau lanjutkan dengan manual steps di bawah...

### 3️⃣ Fetch Data

```powershell
# Option A: Use PowerShell script
.\scripts\run_fetch.ps1 -NBlocks 8000

# Option B: Direct Python command
python -m src.fetch --network sepolia --n-blocks 8000

# Output: data/blocks.csv (~10-15 minutes)
```

### 4️⃣ Build Features

```powershell
# Option A: Use PowerShell script
.\scripts\run_features.ps1

# Option B: Direct Python command
python -m src.features --in data/blocks.csv --out data/features.parquet

# Output: data/features.parquet
```

### 5️⃣ Train Model

```powershell
# Option A: Use PowerShell script
.\scripts\run_train.ps1

# Option B: Direct Python command
python -m src.train --cfg cfg/exp.yaml --in data/features.parquet

# Output: models/ directory (CPU: ~30-60 min, GPU: ~10-20 min)
#   - lstm.pt
#   - xgb.bin
#   - hybrid_metadata.pkl
#   - scaler.pkl
#   - training_info.json
```

### 6️⃣ Run Inference

```powershell
# Single prediction
.\scripts\run_infer.ps1

# Continuous mode (setiap 12 detik)
.\scripts\run_infer.ps1 -Continuous

# Direct Python command
python -m src.infer --model models/

# Example Output:
# ⛽ GAS FEE RECOMMENDATION
# ========================================
# 📊 Prediction:
#   Base Fee (predicted): 22.7431 Gwei
#   Base Fee (buffered):  25.0174 Gwei
#   Buffer: 10.0%
#
# 💡 Recommendation:
#   Priority Fee: 1.3245 Gwei
#   Max Fee Per Gas: 26.3419 Gwei
#
# ✓ Status: Optimal
#   High confidence - Ready to broadcast
#   Confidence: 87.3%
```

### 7️⃣ Evaluate & Backtest

```powershell
# Run backtest
.\scripts\run_backtest.ps1

# Direct Python command
python -m src.backtest --model models/ --data data/features.parquet

# Output: reports/ directory
#   - backtest_metrics.json
#   - backtest_predictions.csv
#   - backtest_full.json
```

## 📊 Fitur Model

### Input Features
- `baseFeePerGas`: Base fee blok saat ini (Wei)
- `gasUsed`: Total gas digunakan dalam blok
- `gasLimit`: Limit gas blok
- `txCount`: Jumlah transaksi dalam blok
- `Δ baseFee`: Perubahan base fee
- `utilization`: gasUsed / gasLimit
- `EMA_baseFee`: Exponential Moving Average base fee
- `EMA_utilization`: EMA utilization
- `hour_of_day`, `day_of_week`: Temporal features

### Output
- `baseFee_next`: Prediksi base fee untuk blok berikutnya
- `priorityFee`: Rekomendasi priority fee (tip)
- `maxFee`: Maximum fee per gas (dengan buffer)

## 🎯 Metrik Evaluasi

| Metrik | Deskripsi |
|--------|-----------|
| **MAE** | Mean Absolute Error (Gwei) |
| **MAPE** | Mean Absolute Percentage Error (%) |
| **RMSE** | Root Mean Squared Error |
| **Under-estimation Rate** | % prediksi < actual (transaksi gagal) |
| **Hit@ε** | % prediksi dalam toleransi ε (5%) |
| **Cost-saving** | Penghematan biaya vs baseline (%) |

## 🔬 Metodologi

1. **Data Collection**: Fetch blok dari Sepolia via RPC/Etherscan API
2. **Feature Engineering**: Ekstraksi fitur temporal, statistik, dan domain-specific
3. **Model Architecture**:
   - LSTM (2 layers, 128 hidden units) → ekstraksi pola temporal
   - XGBoost (500 trees) → meta-learner untuk prediksi akhir
4. **Training**: Supervised learning dengan sliding window (24 blok lookback)
5. **Inference**: Real-time prediction + policy-based fee recommendation
6. **Evaluation**: Backtesting dengan historical data + cost analysis

## 🧪 Testing

```powershell
# Run unit tests with script
.\scripts\run_tests.ps1

# Or direct pytest
pytest tests/ -v

# With coverage report
pytest tests/ --cov=src --cov-report=html --cov-report=term

# Open HTML coverage report
start htmlcov/index.html
```

## 📖 Dokumentasi

- [Setup Guide](SETUP.md) - Detailed setup dan troubleshooting
- [Changelog](CHANGELOG.md) - Version history dan roadmap
- Inline Documentation - Setiap modul punya docstring lengkap
- Script Examples - Check `scripts/` directory untuk automation

### Command Reference

```powershell
# Data Pipeline
python -m src.fetch --network sepolia --n-blocks 8000 --output data/blocks.csv
python -m src.features --in data/blocks.csv --out data/features.parquet

# Training & Evaluation
python -m src.train --cfg cfg/exp.yaml --in data/features.parquet --out-dir models
python -m src.backtest --model models/ --data data/features.parquet --output-dir reports

# Inference
python -m src.infer --model models/ --network sepolia
python -m src.infer --model models/ --continuous --interval 12

# Testing
pytest tests/ -v --cov=src
```

## 🎯 Performance Expectations

Berdasarkan eksperimen awal:
- **MAE**: ~2-5 Gwei (pada testnet Sepolia)
- **MAPE**: ~5-15%
- **Hit@5%**: ~70-85%
- **Under-estimation Rate**: <15%
- **Cost Saving vs Baseline**: 10-30%

*Note: Performance tergantung pada kondisi network dan kualitas data*

## 🤝 Kontributor

- **Peneliti**: [Nama Anda]
- **Pembimbing**: [Nama Dosen Pembimbing]
- **Institusi**: [Nama Universitas]
- **Tahun**: 2025

## 📄 Lisensi

MIT License - Proyek Riset Akademik

## 🙏 Acknowledgments

- EIP-1559 specification authors
- Ethereum Foundation untuk Sepolia testnet
- XGBoost dan PyTorch communities
- Public RPC providers

## 📚 Referensi

- [EIP-1559: Fee market change for ETH 1.0 chain](https://eips.ethereum.org/EIPS/eip-1559)
- [XGBoost: A Scalable Tree Boosting System](https://arxiv.org/abs/1603.02754)
- [LSTM Networks for Time Series](https://www.bioinf.jku.at/publications/older/2604.pdf)

---

⚠️ **Disclaimer**: Proyek ini untuk keperluan riset akademik. Gunakan di testnet Sepolia. Tidak untuk produksi.
