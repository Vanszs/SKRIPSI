# Setup Instructions untuk Gas Fee Prediction Model

## Prerequisites

- Python 3.10 atau lebih baru
- Windows dengan PowerShell 5.1+
- Git (optional)
- Internet connection untuk fetch data dari Sepolia

## 🚀 Quick Setup

### 1. Setup Virtual Environment

```powershell
# Navigate ke project directory
cd gas-ml

# Create virtual environment
python -m venv venv

# Activate virtual environment (PowerShell)
.\venv\Scripts\Activate.ps1

# Jika ada error execution policy:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### 2. Install Dependencies

```powershell
# Install semua dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch, xgboost, pandas; print('All dependencies installed!')"
```

### 3. Configure Environment

```powershell
# Copy environment template
cp .env.example .env

# Edit .env file dengan text editor
notepad .env
```

**Minimal configuration:**
```
SEPOLIA_RPC_URL=https://rpc.sepolia.org
```

**Recommended (dengan Infura):**
```
SEPOLIA_RPC_URL=https://sepolia.infura.io/v3/YOUR_INFURA_KEY
ETHERSCAN_API_KEY=YOUR_ETHERSCAN_API_KEY
```

Dapatkan free Infura API key: https://infura.io/

### 4. Test Connection

```powershell
# Test RPC connection
python -m src.rpc

# Expected output:
# ✓ Connection test successful!
```

## 📊 Complete Pipeline

### Option A: Automated Pipeline (Recommended)

```powershell
# Run complete pipeline otomatis
.\scripts\run_pipeline.ps1
```

Pipeline ini akan:
1. Fetch 8000 blocks dari Sepolia
2. Generate features
3. Train hybrid model
4. Run backtest
5. Test inference

### Option B: Manual Step-by-Step

#### Step 1: Fetch Data

```powershell
# Fetch 8000 blocks terakhir
python -m src.fetch --network sepolia --n-blocks 8000

# Output: data/blocks.csv
```

#### Step 2: Generate Features

```powershell
# Generate features dari raw blocks
python -m src.features --in data/blocks.csv --out data/features.parquet

# Output: data/features.parquet
```

#### Step 3: Train Model

```powershell
# Train hybrid LSTM+XGBoost model
python -m src.train --cfg cfg/exp.yaml --in data/features.parquet

# Output: models/ directory dengan trained models
```

Training time:
- CPU: ~30-60 minutes
- GPU: ~10-20 minutes

#### Step 4: Evaluate (Backtest)

```powershell
# Run backtest simulation
python -m src.backtest --model models/ --data data/features.parquet

# Output: reports/ directory dengan metrics
```

#### Step 5: Inference

```powershell
# Single prediction
python -m src.infer --model models/

# Continuous prediction (setiap 12 detik)
python -m src.infer --model models/ --continuous
```

## 🧪 Run Tests

```powershell
# Run unit tests
pytest tests/ -v

# Run dengan coverage
pytest tests/ --cov=src --cov-report=html

# Open coverage report
start htmlcov/index.html
```

## 🐛 Troubleshooting

### Import Error: web3 module not found

```powershell
pip install --upgrade web3
```

### CUDA/PyTorch issues

```powershell
# For CPU-only (jika tidak ada GPU)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# For CUDA (jika punya NVIDIA GPU)
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### Connection timeout ke Sepolia

Coba alternative RPC:
```
SEPOLIA_RPC_URL=https://ethereum-sepolia.publicnode.com
```

### Memory error saat training

Reduce batch size di `cfg/exp.yaml`:
```yaml
training:
  batch_size: 16  # default: 32
```

## 📁 Project Structure

```
gas-ml/
├── cfg/
│   └── exp.yaml              # Experiment configuration
├── data/
│   ├── blocks.csv            # Raw block data
│   └── features.parquet      # Engineered features
├── models/
│   ├── lstm.pt               # LSTM weights
│   ├── xgb.bin               # XGBoost model
│   ├── hybrid_metadata.pkl   # Model metadata
│   └── training_info.json    # Training info
├── reports/
│   ├── backtest_metrics.json
│   └── backtest_predictions.csv
├── src/
│   ├── rpc.py               # RPC client
│   ├── fetch.py             # Data fetcher
│   ├── features.py          # Feature engineering
│   ├── stack.py             # Hybrid model
│   ├── train.py             # Training pipeline
│   ├── infer.py             # Inference engine
│   ├── policy.py            # Fee policy
│   ├── backtest.py          # Backtesting
│   └── evaluate.py          # Metrics
├── scripts/
│   ├── run_fetch.ps1
│   ├── run_train.ps1
│   ├── run_infer.ps1
│   ├── run_backtest.ps1
│   └── run_pipeline.ps1
└── tests/
    └── test_metrics.py
```

## 🎓 Academic Usage

### For Research/Thesis

1. **Data Collection**: Document RPC endpoint dan timestamp
2. **Model Training**: Save training logs dan hyperparameters
3. **Evaluation**: Generate comprehensive metrics report
4. **Visualization**: Create plots dari backtest results

### Citation

```
@software{gas_fee_ml,
  title = {Hybrid LSTM-XGBoost Gas Fee Prediction Model},
  author = {Your Name},
  year = {2025},
  note = {Research implementation for Ethereum gas fee optimization}
}
```

## 📚 Additional Resources

- [EIP-1559 Specification](https://eips.ethereum.org/EIPS/eip-1559)
- [Sepolia Testnet Faucet](https://sepoliafaucet.com/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [PyTorch LSTM Guide](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html)

## 💬 Support

Jika ada masalah:
1. Check troubleshooting section
2. Review log files di `logs/` (jika ada)
3. Check GitHub issues (jika project di GitHub)
4. Contact pembimbing untuk academic guidance

## 🔒 Security Notes

⚠️ **Important:**
- Jangan commit `.env` file ke Git
- Jangan share API keys
- Hanya gunakan testnet (Sepolia)
- Tidak untuk production use

---

Last updated: {{ datetime.now().strftime('%Y-%m-%d') }}
