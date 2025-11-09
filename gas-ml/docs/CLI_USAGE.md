# 🚀 Gas Fee ML - CLI Documentation

## Command Line Interface

Production-ready CLI untuk prediksi gas fee Ethereum real-time.

---

## 📦 Installation

```bash
# Activate virtual environment
.\.venv\Scripts\Activate.ps1

# Ensure dependencies installed
pip install click
```

---

## 🎯 Commands

### 1. **Real-Time Prediction**

```bash
# Single prediction
python cli.py predict --rpc https://eth.llamarpc.com

# Continuous monitoring (every 12 seconds)
python cli.py predict --rpc https://eth.llamarpc.com --continuous

# Save predictions to JSON
python cli.py predict --rpc https://eth.llamarpc.com --save

# Custom interval (30 seconds)
python cli.py predict --rpc https://eth.llamarpc.com --continuous --interval 30
```

**Options:**
- `--rpc`: Ethereum RPC URL (required)
- `--model-dir`: Model directory (default: `models/`)
- `--network`: Network name (default: `mainnet`)
- `--continuous`: Enable continuous prediction mode
- `--interval`: Interval in seconds (default: 12)
- `--save`: Save predictions to JSON file

**Output:**
```
⛽ GAS FEE RECOMMENDATION
============================================================

🌐 Network: MAINNET
📍 Current Block: #18450123
💰 Current Base Fee: 15.3421 Gwei
📊 Current Utilization: 67.3%

🔮 Prediction:
   Base Fee (predicted): 15.8234 Gwei
   Base Fee (buffered):  17.4057 Gwei
   Confidence Interval:  ±0.0042 Gwei (95% CI)

💡 Recommendation:
   Priority Fee:    1.5000 Gwei
   Max Fee Per Gas: 18.9057 Gwei
   Buffer Applied:  10.0%

✅ Status: Optimal
   High confidence - Ready to broadcast
   Confidence: 92.3%
```

---

### 2. **Model Information**

```bash
python cli.py info
```

**Output:**
```
🧠 MODEL INFORMATION
============================================================

📊 Performance Metrics:
   MAE:              0.0021 Gwei
   MAPE:             2.19%
   RMSE:             0.0027 Gwei
   R²:               0.9701
   Hit@5%:           85.36%
   Under-estimation: 16.01%

🏗️  Architecture:
   Type:             Hybrid LSTM → XGBoost
   LSTM hidden:      192
   LSTM layers:      2
   LSTM dropout:     0.25
   XGBoost trees:    700
   XGBoost depth:    6
   Sequence length:  24 blocks

📦 Training Data:
   Features:         19
   Timestamp:        2025-10-27T...
```

---

### 3. **Historical Backtesting**

```bash
python cli.py backtest --data data/blocks_5k.csv

# Custom output file
python cli.py backtest --data data/blocks_5k.csv --output reports/my_backtest.csv
```

**Options:**
- `--data`: Historical blocks CSV file (required)
- `--model-dir`: Model directory (default: `models/`)
- `--output`: Output CSV file (default: `reports/backtest.csv`)

**Output:**
```
📊 BACKTEST RESULTS
============================================================

Accuracy Metrics:
   MAE:              0.0021 Gwei
   MAPE:             2.19%
   RMSE:             0.0027 Gwei
   Hit@5%:           85.36%

Economic Impact:
   Cost Saving:      15.32%
   Model Success:    83.99%
   Baseline Success: 100.00%

💾 Results saved to: reports/backtest.csv
```

---

## 🔧 Advanced Usage

### Using with Custom RPC Providers

```bash
# Alchemy
python cli.py predict --rpc https://eth-mainnet.g.alchemy.com/v2/YOUR_API_KEY

# Infura
python cli.py predict --rpc https://mainnet.infura.io/v3/YOUR_PROJECT_ID

# Local node
python cli.py predict --rpc http://localhost:8545

# Sepolia testnet
python cli.py predict --rpc https://rpc.sepolia.org --network sepolia
```

### Continuous Monitoring with Logging

```bash
# Redirect output to log file
python cli.py predict --rpc https://eth.llamarpc.com --continuous > logs/predictions.log 2>&1
```

### Automation with Cron/Task Scheduler

**Linux (cron):**
```bash
# Run every minute
* * * * * cd /path/to/gas-ml && python cli.py predict --rpc https://eth.llamarpc.com --save
```

**Windows (Task Scheduler):**
```powershell
# Create task
$action = New-ScheduledTaskAction -Execute "python" -Argument "cli.py predict --rpc https://eth.llamarpc.com --save" -WorkingDirectory "D:\SKRIPSI\gas-ml"
$trigger = New-ScheduledTaskTrigger -Once -At (Get-Date) -RepetitionInterval (New-TimeSpan -Minutes 1)
Register-ScheduledTask -TaskName "GasFeePredictor" -Action $action -Trigger $trigger
```

---

## 📊 Prediction Output Files

Saved predictions (when using `--save`) are stored in `predictions/`:

```json
{
  "network": "mainnet",
  "current_block": 18450123,
  "current_base_fee_gwei": 15.3421,
  "current_utilization": 0.673,
  "predicted_base_fee_wei": 15823400000,
  "predicted_base_fee_gwei": 15.8234,
  "buffered_base_fee_gwei": 17.4057,
  "priority_fee_gwei": 1.5,
  "max_fee_per_gas_gwei": 18.9057,
  "buffer_multiplier": 1.1,
  "confidence": 0.923,
  "confidence_interval_gwei": 0.0042,
  "status": "Optimal",
  "status_description": "High confidence - Ready to broadcast",
  "timestamp": "2025-10-27T15:30:45.123456"
}
```

---

## 🔍 Troubleshooting

### Connection Errors

```bash
# Test RPC connection first
python -m src.rpc

# Check network connectivity
curl https://eth.llamarpc.com
```

### Model Not Found

```bash
# Ensure model exists
ls models/

# If missing, train model first
python src/train.py --cfg cfg/exp.yaml --in data/features_5k_selected.parquet
```

### Rate Limiting

```bash
# Increase interval to avoid rate limits
python cli.py predict --rpc YOUR_RPC --continuous --interval 60

# Use premium RPC provider
python cli.py predict --rpc https://eth-mainnet.g.alchemy.com/v2/YOUR_KEY
```

---

## 📚 Integration Examples

### Python Script

```python
from cli import RealtimePredictor

# Initialize
predictor = RealtimePredictor(
    rpc_url='https://eth.llamarpc.com',
    model_dir='models',
    network='mainnet'
)

# Get recommendation
recommendation = predictor.get_recommendation()

print(f"Recommended maxFee: {recommendation['max_fee_per_gas_gwei']:.4f} Gwei")
```

### Web API (Flask)

```python
from flask import Flask, jsonify
from cli import RealtimePredictor

app = Flask(__name__)
predictor = RealtimePredictor(rpc_url='https://eth.llamarpc.com')

@app.route('/predict')
def predict():
    recommendation = predictor.get_recommendation()
    return jsonify(recommendation)

if __name__ == '__main__':
    app.run(port=5000)
```

### Discord Bot

```python
import discord
from cli import RealtimePredictor

client = discord.Client()
predictor = RealtimePredictor(rpc_url='https://eth.llamarpc.com')

@client.event
async def on_message(message):
    if message.content == '!gas':
        rec = predictor.get_recommendation()
        await message.channel.send(
            f"💰 Recommended Gas Fee: {rec['max_fee_per_gas_gwei']:.2f} Gwei"
        )

client.run('YOUR_BOT_TOKEN')
```

---

## 🎯 Best Practices

1. **Rate Limiting**: Use `--interval` >= 12 seconds for free RPC endpoints
2. **Monitoring**: Always use `--save` for continuous mode to track predictions
3. **Validation**: Run `info` command to verify model performance before deployment
4. **Backtesting**: Test on historical data before using in production
5. **RPC Reliability**: Use multiple fallback RPC endpoints
6. **Error Handling**: Implement retry logic in production integrations

---

## 📝 See Also

- [Main README](../README.md)
- [Model Training](../docs/training.md)
- [API Documentation](../docs/api.md)
- [Deployment Guide](../docs/deployment.md)
