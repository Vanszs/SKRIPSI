
import logging
import pandas as pd
import numpy as np
import yaml
import xgboost as xgb
import json
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def asymmetric_mse_obj(y_pred, y_true):
    """
    Custom objective function for XGBoost.
    Penalizes under-estimations (prediction < label) more heavily (2.5x).
    Usage:
        model = xgb.train(..., obj=asymmetric_mse_obj, ...)
    """
    labels = y_true.get_label()
    residual = y_pred - labels
    # Penalty of 2.5x for under-estimation (residual < 0 means pred < label)
    grad = np.where(residual < 0, 2.5 * residual, residual)
    hess = np.where(residual < 0, 2.5 * np.ones_like(residual), np.ones_like(residual))
    return grad, hess

def train_xgboost_from_config(cfg_path: str, data_path: str, output_dir: str = "models/xgboost_manual"):
    """
    Trains XGBoost model using the 'Golden' configuration (Manual approach).
    Reproduces 98% R2 score by using:
    1. Custom Objective (Asymmetric MSE)
    2. Base Score optimization (mean(y_train))
    3. Correct Data Split (Last 20% as Test)
    
    Args:
        cfg_path: Path to config yaml
        data_path: Path to features parquet
        output_dir: Directory to save artifacts
    
    Returns:
        dict: Metrics dictionary
    """
    # 0. Setup Paths
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 1. Load Config
    with open(cfg_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 2. Load Data
    logger.info(f"Loading data from {data_path}...")
    df = pd.read_parquet(data_path)
    
    # Features
    exclude = ['number', 'timestamp', 'datetime', 'baseFee_next']
    features = [c for c in df.columns if c not in exclude]
    target = 'baseFee_next'
    
    X = df[features].values
    y = df[target].values
    
    # 3. Split
    # CRITICAL: val=0.15, test=0.20 (Matching Inference Notebook to reproduce 98% R2)
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
    
    logger.info(f"Split Strategy (Golden): Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
    
    # 4. Normalize Features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    # 5. Prepare XGBoost DMatrices
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=features)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=features)
    dtest = xgb.DMatrix(X_test, label=y_test, feature_names=features)
    
    # 6. Train Configuration
    params = config['model']['xgboost'].copy()
    
    # Clean params for native xgb.train
    num_rounds = params.pop('n_estimators', 1000)
    early_stopping_rounds = params.pop('early_stopping_rounds', 50)
    use_custom = params.pop('use_custom_objective', False)
    params.pop('n_jobs', None) 
    
    # CRITICAL OPTIMIZATION: Initialize base prediction to mean of target
    mean_y = float(np.mean(y_train))
    logger.info(f"Optimization: Setting base_score to mean(y_train): {mean_y}")
    
    params.update({
        'eval_metric': 'rmse',
        'base_score': mean_y,
    })
    
    obj_func = None
    if use_custom:
        logger.info("Objective: Custom Asymmetric MSE (2.5x penalty)")
        params.pop('objective', None)
        obj_func = asymmetric_mse_obj
    else:
        logger.info("Objective: Standard RMSE")
        params['objective'] = 'reg:squarederror'

    logger.info(f"Training params: {json.dumps(params, indent=2)}")
    
    # 7. Train
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=num_rounds,
        evals=[(dtrain, 'train'), (dval, 'val')],
        obj=obj_func,
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=100
    )
    
    # 8. Evaluate
    logger.info("Evaluating on Test Set (Last 20%)...")
    y_pred = model.predict(dtest)
    
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    
    # Under-estimation rate
    residuals = y_test - y_pred
    under_est_count = np.sum(residuals > 0)
    under_est_rate = (under_est_count / len(residuals)) * 100
    
    metrics = {
        "r2": float(r2),
        "mae": float(mae),
        "mae_gwei": float(mae / 1e9),
        "rmse": float(rmse),
        "rmse_gwei": float(rmse / 1e9),
        "mape": float(mape),
        "under_estimation_rate": float(under_est_rate)
    }
    
    logger.info("\n" + "="*40)
    logger.info("FINAL ROBUST METRICS")
    logger.info("="*40)
    logger.info(f"R2 Score: {r2:.4f}")
    logger.info(f"MAPE:     {mape:.4f}%")
    logger.info(f"MAE:      {mae:.4f}")
    logger.info(f"RMSE:     {rmse:.4f}")
    logger.info("="*40 + "\n")
    
    # 9. Save Artifacts
    logger.info(f"Saving artifacts to {output_path}...")
    
    # Save Model (Standard Name for src.models.XGBoostGasFeeModel)
    model_file = output_path / "model.xgb.json"
    model.save_model(str(model_file))
    
    # Save Scaler (Required by proper pipelines)
    import pickle
    with open(output_path / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
        
    # Save Metrics
    with open(output_path / "metrics.json", 'w') as f:
        json.dump(metrics, f, indent=4)
        
    # Save Metadata (Required by src.models.load_model_from_dir)
    from datetime import datetime
    metadata = {
        "model_type": "xgboost",
        "model_name": "xgboost_robust",
        "model_params": params,
        "input_size": len(features),
        "prediction_offset": 0,
        "sequence_length": 1,  
        "created_at": datetime.now().isoformat()
    }
    with open(output_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)
        
    # Save Training Info (Legacy support for some scripts)
    training_info = {
        "feature_columns": features,
        "config": config,
        "metrics": metrics,
        "timestamp": datetime.now().isoformat()
    }
    with open(output_path / "training_info.json", "w") as f:
        json.dump(training_info, f, indent=4)
        
    # Save Predictions for Comparison Notebook
    df_test = pd.DataFrame({
        'y_true': y_test,
        'y_pred': y_pred
    })
    df_test.to_parquet(output_path / "test_predictions.parquet")
    
    logger.info(f"Artifacts saved to {output_path}")
    return metrics

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='cfg/exp.yaml')
    parser.add_argument('--in_file', type=str, default='data/features.parquet')
    parser.add_argument('--out_dir', type=str, default='models/xgboost_manual')
    args = parser.parse_args()
    
    train_xgboost_from_config(args.cfg, args.in_file, args.out_dir)
