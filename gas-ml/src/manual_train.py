import argparse
import logging
import pandas as pd
import numpy as np
import yaml
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def asymmetric_mse_obj(y_pred, y_true):
    # Native XGBoost custom objective signature: preds, dtrain
    # But wait, sklearn API is (y_true, y_pred). Native is (preds, dtrain).
    # Let's use sklearn API for simplicity if possible, or handle native correctly.
    # Actually, for native xgb.train: obj(preds, dtrain) -> grad, hess
    labels = y_true.get_label()
    residual = y_pred - labels
    # Penalty of 2.5x for under-estimation (residual < 0 means pred < label)
    # wait: residual = pred - label. If pred < label, residual is negative. Correct.
    grad = np.where(residual < 0, 2.5 * residual, residual)
    hess = np.where(residual < 0, 2.5 * np.ones_like(residual), np.ones_like(residual))
    return grad, hess

def main():
    parser = argparse.ArgumentParser(description='Manual XGBoost Training')
    parser.add_argument('--cfg', type=str, required=True)
    parser.add_argument('--in_file', type=str, required=True)
    args = parser.parse_args()

    # 1. Load Config
    with open(args.cfg, 'r') as f:
        config = yaml.safe_load(f)
    
    # 2. Load Data
    logger.info(f"Loading data from {args.in_file}...")
    df = pd.read_parquet(args.in_file)
    
    # Features
    exclude = ['number', 'timestamp', 'datetime', 'baseFee_next']
    features = [c for c in df.columns if c not in exclude]
    target = 'baseFee_next'
    
    X = df[features].values
    y = df[target].values
    
    # 3. Split
    # Config: val=0.15, test=0.20 (Matching Inference Notebook to reproduce 98% R2)
    n = len(df)
    test_len = int(n * 0.20)
    val_len = int(n * 0.15)
    train_len = n - val_len - test_len
    
    X_train = X[:train_len]
    y_train = y[:train_len]
    
    X_val = X[train_len:train_len+val_len]
    y_val = y[train_len:train_len+val_len]
    
    X_test = X[train_len+val_len:]
    y_test = y[train_len+val_len:]
    
    logger.info(f"Split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
    
    # 4. Normalize Features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    # 5. Prepare XGBoost DMatrices
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=features)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=features)
    dtest = xgb.DMatrix(X_test, label=y_test, feature_names=features)
    
    # 6. Train
    # Extract params
    params = config['model']['xgboost'].copy()
    
    # Clean params for native xgb.train
    # 'n_estimators' -> num_boost_round
    num_rounds = params.pop('n_estimators', 1000)
    early_stopping_rounds = params.pop('early_stopping_rounds', 50)
    use_custom = params.pop('use_custom_objective', False)
    
    # Remove sklearn-specific params if any
    params.pop('n_jobs', None) 
    
    # Add native params
    # OPTIMIZATION: Initialize base prediction to mean of target
    # This is critical for regression on large values (Wei)
    mean_y = float(np.mean(y_train))
    logger.info(f"Setting base_score to mean(y_train): {mean_y}")
    
    params.update({
        'eval_metric': 'rmse',
        'base_score': mean_y,
    })
    
    obj_func = None
    if use_custom:
        logger.info("Using Custom Asymmetric Objective (2.5x penalty)")
        # Note: obj function for xgb.train is different from sklearn API
        # It is passed as 'obj' argument, not inside params['objective'] usually
        # But 'objective' param must be removed if obj is passed
        params.pop('objective', None)
        obj_func = asymmetric_mse_obj
    else:
        logger.info("Using Standard RMSE Objective")
        params['objective'] = 'reg:squarederror'

    logger.info("Training...")
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=num_rounds,
        evals=[(dtrain, 'train'), (dval, 'val')],
        obj=obj_func,
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=100
    )
    
    # 7. Evaluate
    logger.info("Evaluating...")
    y_pred = model.predict(dtest)
    
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    
    logger.info("\n" + "="*40)
    logger.info("MANUAL TRAINING RESULTS")
    logger.info("="*40)
    logger.info(f"R2 Score: {r2:.4f}")
    logger.info(f"MAPE:     {mape:.4f}%")
    logger.info(f"MAE:      {mae:.4f}")
    logger.info("="*40 + "\n")
    
    with open('manual_metrics.txt', 'w') as f:
        f.write(f"R2: {r2}\nMAPE: {mape}\nMAE: {mae}\n")

if __name__ == "__main__":
    main()
