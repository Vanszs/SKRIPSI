"""Quick XGBoost training test script."""
import sys
sys.path.insert(0, 'src')
import yaml
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load config
with open('cfg/exp.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Load data
df = pd.read_parquet('data/features.parquet')
print(f"=== Data Info ===")
print(f"Shape: {df.shape}")
print(f"\n=== Target (baseFee_next) Stats ===")
print(df['baseFee_next'].describe())

# Check distribution split
n = len(df)
train_end = int(n * 0.8 * 0.85)
val_end = int(n * 0.8)

print(f"\n=== Distribution by Split ===")
print(f"Train (0:{train_end}): mean={df['baseFee_next'][:train_end].mean()/1e9:.4f} Gwei")
print(f"Val ({train_end}:{val_end}): mean={df['baseFee_next'][train_end:val_end].mean()/1e9:.4f} Gwei")
print(f"Test ({val_end}:{n}): mean={df['baseFee_next'][val_end:].mean()/1e9:.4f} Gwei")

# Prepare features
exclude_cols = ['number', 'timestamp', 'datetime', 'baseFee_next']
feature_columns = [col for col in df.columns if col not in exclude_cols]
X = df[feature_columns].values
y = df['baseFee_next'].values

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, shuffle=False)

# Normalize
scaler = StandardScaler()
X_train_norm = scaler.fit_transform(X_train)
X_val_norm = scaler.transform(X_val)
X_test_norm = scaler.transform(X_test)

print(f"\n=== Training XGBoost ===")
from models import XGBoostGasFeeModel

model = XGBoostGasFeeModel(name='xgboost_test', config=config['model']['xgboost'])
model.fit(X_train_norm, y_train, X_val_norm, y_val)

preds = model.predict(X_test_norm)

print(f"\n=== Predictions vs Actual ===")
print(f"Predictions - min: {preds.min()/1e9:.6f} Gwei, max: {preds.max()/1e9:.6f} Gwei, mean: {preds.mean()/1e9:.6f} Gwei")
print(f"Actual      - min: {y_test.min()/1e9:.6f} Gwei, max: {y_test.max()/1e9:.6f} Gwei, mean: {y_test.mean()/1e9:.6f} Gwei")

print(f"\n=== Sample Predictions ===")
for i in range(min(10, len(preds))):
    print(f"  {i}: Pred={preds[i]/1e9:.6f} Gwei, Actual={y_test[i]/1e9:.6f} Gwei")

# Metrics
mae = mean_absolute_error(y_test, preds)
rmse = np.sqrt(mean_squared_error(y_test, preds))
r2 = r2_score(y_test, preds)
mape = np.mean(np.abs((y_test - preds) / (y_test + 1e-10))) * 100

print(f"\n=== Final Metrics ===")
print(f"MAE: {mae/1e9:.6f} Gwei")
print(f"RMSE: {rmse/1e9:.6f} Gwei") 
print(f"R2: {r2:.4f}")
print(f"MAPE: {mape:.2f}%")
